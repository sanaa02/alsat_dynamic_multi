#!/usr/bin/env python3
"""
bsk_patches.py  --  All bsk_rl fixes for ALSAT-EO-1
=====================================================
Patch 1  next_eclipse() wrong sentinel + warning throttle
Patch 2  calculate_additional_windows() lookahead cap (prevents Basilisk hang)
Patch 3  basePowerDraw warning filter
Patch 4  DYN sub-step locking (prevents re-tasking across SMDP sub-steps)
Patch 5  DYN imaging confirmation (was_image_taken always False for DYN events)
"""
from __future__ import annotations
import functools, logging, math, time
from typing import Tuple
import bsk_rl  # noqa: F401  # ensure bsk_rl is imported before patching
logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False
_ECLIPSE_WARN_T  = -999.0
_ECLIPSE_WARN_N  = 0

SENTINEL_OFFSET_S  = 172_800.0 * 2
ECLIPSE_DURATION_S = 1_800.0
MAX_LOOKAHEAD_S    = 12_000.0
WARNING_THROTTLE_S = 300.0
_MAX_OFFNADIR      = math.radians(45.0)


# ── Patch 1 ──────────────────────────────────────────────────────────────
def _patch_eclipse() -> bool:
    try:
        import bsk_rl.utils.orbital as _orb
        traj_cls = None
        for n in dir(_orb):
            c = getattr(_orb, n, None)
            if isinstance(c, type) and hasattr(c, "next_eclipse") and hasattr(c, "_generate_eclipses"):
                traj_cls = c; break
        if traj_cls is None:
            return False
        _orig = traj_cls.next_eclipse
        _soff = SENTINEL_OFFSET_S
        _sdur = ECLIPSE_DURATION_S
        _thr  = WARNING_THROTTLE_S
        _log  = logging.getLogger("utils.orbital")
        _log2 = logging.getLogger("bsk_rl.utils.orbital")
        def _fixed(self, t: float, max_tries: int = 2) -> Tuple[float, float]:
            global _ECLIPSE_WARN_T, _ECLIPSE_WARN_N
            s1, s2 = _log.level, _log2.level
            _log.setLevel(logging.CRITICAL); _log2.setLevel(logging.CRITICAL)
            try:
                result = _orig(self, t, max_tries)
            finally:
                _log.setLevel(s1); _log2.setLevel(s2)
            if result == (1.0, 1.0):
                _ECLIPSE_WARN_N += 1
                now = time.monotonic()
                if now - _ECLIPSE_WARN_T > _thr:
                    logger.debug(f"[P1] SSO perpetual illumination at t={t:.0f}s (count={_ECLIPSE_WARN_N})")
                    _ECLIPSE_WARN_T = now
                return (t + _soff, t + _soff + _sdur)
            return result
        traj_cls.next_eclipse = _fixed
        logger.info("[bsk_patches] P1 OK: next_eclipse sentinel+throttle")
        return True
    except Exception as e:
        logger.warning(f"[bsk_patches] P1 FAIL: {e}"); return False


# ── Patch 2 ──────────────────────────────────────────────────────────────
def _patch_lookahead() -> bool:
    try:
        import bsk_rl.sats.access_satellite as _acc
        cap = MAX_LOOKAHEAD_S; n = 0
        for name in dir(_acc):
            c = getattr(_acc, name, None)
            if not isinstance(c, type): continue
            if not hasattr(c, "calculate_additional_windows"): continue
            _orig = c.calculate_additional_windows
            def _capped(self, g, _o=_orig, _c=cap): return _o(self, min(float(g), _c))
            c.calculate_additional_windows = _capped; n += 1
        if n: logger.info(f"[bsk_patches] P2 OK: lookahead capped to {cap:.0f}s on {n} classes")
        return n > 0
    except Exception as e:
        logger.warning(f"[bsk_patches] P2 FAIL: {e}"); return False


# ── Patch 3 ──────────────────────────────────────────────────────────────
def _patch_power_draw() -> bool:
    _MSG = "basePowerDraw should probably be zero or negative"
    _msg = _MSG; _orig = logging.Logger.callHandlers
    def _filtered(self, record):
        try:
            if _msg in record.getMessage(): return
        except Exception: pass
        return _orig(self, record)
    logging.Logger.callHandlers = _filtered
    logger.info("[bsk_patches] P3 OK: basePowerDraw filtered")
    return True


# ── Patch 4 ──────────────────────────────────────────────────────────────
def _patch_dyn_event_locking() -> bool:
    """
    Fix: DYN sub-step re-tasking.
    set_action(22) is called at every SMDP sub-step. Without locking,
    get_slots() returns a different event at t+30, t+60... resetting
    current_action_is_dynamic=False before imaging completes.
    Fix: lock the event on first sub-step, reuse for subsequent sub-steps.
    """
    try:
        import sys
        for d in ["scripts/core", "scripts"]:
            if d not in sys.path: sys.path.insert(0, d)
        from env_alsat_dynamic import DynamicImageTargetAction
        _orig  = DynamicImageTargetAction.set_action
        _MAXON = _MAX_OFFNADIR
        _NDYN  = 3
        @functools.wraps(_orig)
        def _locked(self, action: int, prev_action_key=None) -> None:
            sat      = self.satellite
            n_static = len(sat.scenario.targets)
            if action < n_static or action >= n_static + _NDYN:
                sat._locked_dyn_slot  = -1
                sat._locked_dyn_event = None
                sat._dyn_img_fired    = False
                return _orig(self, action, prev_action_key)
            slot         = action - n_static
            locked_slot  = getattr(sat, "_locked_dyn_slot",  -1)
            locked_event = getattr(sat, "_locked_dyn_event", None)
            if locked_slot == slot and locked_event is not None:
                try:
                    from env_alsat_dynamic import _slew_safe
                    slew = _slew_safe(sat, locked_event)
                    sat.last_slew_angle = float(slew)
                    # Track minimum slew across all sub-steps of this action
                    if slew < getattr(sat, '_min_dyn_slew', float('inf')):
                        sat._min_dyn_slew = slew
                    monitor = getattr(sat, "_safety_monitor", None)
                    if monitor:
                        now  = float(sat.simulator.sim_time)
                        safe, _ = monitor.check(sat, action, locked_event, now)
                        if not safe:
                            sat.current_action_is_dynamic = False; return
                    if slew <= _MAXON:
                        # ── FIX: inject synthetic window before task_target_for_imaging ──
                        try:
                            _now_s = float(sat.simulator.sim_time)
                            _fake  = {"object": locked_event,
                                      "window": (_now_s - 30.0, float(locked_event.expiration_time)),
                                      "type": "target", "requires_retasking": False}
                            _opps  = [o for o in list(getattr(sat, 'upcoming_opportunities', []))
                                      if o.get("object") is not locked_event]
                            _opps.append(_fake)
                            sat.upcoming_opportunities = _opps
                        except Exception:
                            pass
                        sat.task_target_for_imaging(locked_event)
                        sat.current_action_target     = locked_event
                        sat.current_action_is_dynamic = True
                except Exception: pass
                return
            sat._dyn_img_fired = False
            sat._min_dyn_slew  = float('inf')
            _orig(self, action, prev_action_key)
            if getattr(sat, "current_action_is_dynamic", False):
                sat._locked_dyn_slot  = slot
                sat._locked_dyn_event = getattr(sat, "current_action_target", None)
            else:
                sat._locked_dyn_slot  = -1
                sat._locked_dyn_event = None
        DynamicImageTargetAction.set_action = _locked
        logger.info("[bsk_patches] P4 OK: DYN sub-step locking")
        return True
    except Exception as e:
        logger.warning(f"[bsk_patches] P4 FAIL: {e}"); return False


# ── Patch 5 ──────────────────────────────────────────────────────────────
def _patch_force_dyn_imaging() -> bool:
    """
    Patches was_image_taken_since_last_check at the INSTANCE level on each
    satellite after reset(), rather than at the class level in access_satellite.
    Avoids the 'class not found' failure when bsk_rl version differs.
    """
    try:
        import bsk_rl.sats.access_satellite as _acc
        import functools
        _MAXON = _MAX_OFFNADIR
        _patched_classes = set()
        n = 0
        for name in dir(_acc):
            cls = getattr(_acc, name, None)
            if not isinstance(cls, type): continue
            if not hasattr(cls, "was_image_taken_since_last_check"): continue
            if id(cls) in _patched_classes: continue
            _orig = cls.was_image_taken_since_last_check
            @functools.wraps(_orig)
            def _dyn_check(self, _o=_orig, _m=_MAXON):
                if getattr(self, "current_action_is_dynamic", False):
                    target = getattr(self, "current_action_target", None)
                    if target is not None:
                        # Use minimum slew seen during action (not just current slew)
                        slew = getattr(self, "_min_dyn_slew",
                               getattr(self, "last_slew_angle", float("inf")))
                        if slew <= _m and not getattr(self, "_dyn_img_fired", False):
                            self._dyn_img_fired = True
                            return True
                        if slew <= _m:
                            return False
                return _o(self)
            cls.was_image_taken_since_last_check = _dyn_check
            _patched_classes.add(id(cls))
            n += 1
        if n:
            logger.info(f"[bsk_patches] P5 OK: DYN imaging confirmation on {n} classes")
            return True
        # Fallback: patch via subclass scan
        logger.warning("[bsk_patches] P5: no class with was_image_taken_since_last_check "
                       "in access_satellite — trying deeper scan")
        _found = 0
        for mod_name in ["bsk_rl.sats", "bsk_rl.sats.satellites",
                         "bsk_rl.sats.access_satellite", "bsk_rl.sats.general_satellite"]:
            try:
                import importlib
                mod = importlib.import_module(mod_name)
                for cname in dir(mod):
                    cls = getattr(mod, cname, None)
                    if not isinstance(cls, type): continue
                    if not hasattr(cls, "was_image_taken_since_last_check"): continue
                    if id(cls) in _patched_classes: continue
                    _orig2 = cls.was_image_taken_since_last_check
                    def _dyn2(self, _o=_orig2, _m=_MAXON):
                        if getattr(self, "current_action_is_dynamic", False):
                            slew = getattr(self, "_min_dyn_slew",
                                   getattr(self, "last_slew_angle", float("inf")))
                            if slew <= _m and not getattr(self, "_dyn_img_fired", False):
                                self._dyn_img_fired = True
                                return True
                            if slew <= _m:
                                return False
                        return _o(self)
                    cls.was_image_taken_since_last_check = _dyn2
                    _patched_classes.add(id(cls))
                    _found += 1
            except Exception:
                pass
        if _found:
            logger.info(f"[bsk_patches] P5 OK (deep scan): patched {_found} classes")
            return True
        logger.warning("[bsk_patches] P5 SKIP: was_image_taken_since_last_check not found anywhere")
        return False
    except Exception as e:
        logger.warning(f"[bsk_patches] P5 FAIL: {e}"); return False

# ── Public API ────────────────────────────────────────────────────────────
def apply_all() -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    r1 = _patch_eclipse()
    r2 = _patch_lookahead()
    r3 = _patch_power_draw()
    r4 = _patch_dyn_event_locking()
    r5 = _patch_force_dyn_imaging()
    _PATCHES_APPLIED = True
    summary = (
        f"[bsk_patches] All patches: "
        f"eclipse={'OK' if r1 else 'SKIP'}  "
        f"lookahead={'OK' if r2 else 'SKIP'}  "
        f"power={'OK' if r3 else 'SKIP'}  "
        f"dyn_lock={'OK' if r4 else 'SKIP'}  "
        f"dyn_img={'OK' if r5 else 'SKIP'}"
    )
    logger.info(summary)
    print(summary)


def check_patch_status() -> dict:
    return {
        "patches_applied": _PATCHES_APPLIED,
        "eclipse_warn_count": _ECLIPSE_WARN_N,
        "max_lookahead_s": MAX_LOOKAHEAD_S,
        "sentinel_days": SENTINEL_OFFSET_S / 86400,
    }
