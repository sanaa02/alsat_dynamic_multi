#!/usr/bin/env python3
"""
env_alsat_dynamic_patch.py  --  Targeted surgical patches for env_alsat_dynamic.py
====================================================================================
Apply by importing at the top of train_ppo_smdp_full.py BEFORE any env is created:

    import env_alsat_dynamic_patch  # noqa — applies all patches on import

Or apply selectively by calling the individual patch functions.

Patches
-------
PATCH-1  Urgency direction in DynamicObsWrapper.step() injection block.
         Was:   urgency = 1.0 + 0.5 * frac_ELAPSED      (late = more reward, WRONG)
         Fixed: urgency = 1.0 + 0.5 * frac_REMAINING    (fresh = more reward, CORRECT)
         Note:  compare_log_states() already uses frac_remaining correctly (line 1725).
                This patch aligns the injection block to match.

PATCH-2  Set info["dynamic_imaging_occurred"] = True on successful DYN imaging.
         This key is read by DynamicRewardShaper._detect_dyn_success() for the
         urgency bonus, but was never set.  Without it, urgency bonus never fired.

PATCH-3  DYN cloud threshold unification.
         _dyn_imaging_check() used _DYN_CLOUD_THRESH = 0.9 (vs CLOUD_THRESH = 0.6
         everywhere else). A cloudy event (cloud=0.85) could yield full DYN_MULTIPLIER
         reward via the geometric check while the same cloud rejected a static target.
         Fix: use CLOUD_THRESH = 0.6 in _dyn_imaging_check() too.

PATCH-4  Separate missed-event penalty attribution.
         The missed-event penalty is accumulated on ANY step when an event expires,
         regardless of what action the agent took that step.  If the agent chose a
         DYN slot, it sees: penalty - 0.3 (cloudy) = -1.04, which is far worse
         than drift (-0.5).  Fix: attribute missed-event penalties to a dedicated
         info key so reward_shaping can distinguish them from imaging failures.

PATCH-5  info["last_dyn_urgency"] set on successful DYN imaging.
         Used by DynamicRewardShaper._compute_urgency() to compute the correct
         urgency without needing to re-derive it from satellite state.
"""
from __future__ import annotations

import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PATCH-1 + PATCH-2 + PATCH-4 + PATCH-5:
# Monkey-patch DynamicObsWrapper.step() to apply all fixes without
# touching the main env file (preserves git blame / diff cleanliness).
# ─────────────────────────────────────────────────────────────────────────────

def _patched_step(self, action: int):
    """
    Patched DynamicObsWrapper.step() with:
      PATCH-1: urgency direction fixed (frac_remaining, not frac_elapsed)
      PATCH-2: info["dynamic_imaging_occurred"] set on success
      PATCH-4: info["missed_event_penalty"] logged separately
      PATCH-5: info["last_dyn_urgency"] set for reward shaper
    """
    from env_alsat_dynamic import (
        N_STATIC_TARGETS, N_DYN_SLOTS, BASE_STEP_S, MAX_ACTION_DUR_S,
        MAX_SUB_STEPS, _action_duration, _slew_safe, _dyn_imaging_check,
        _compute_tta, TIME_NORM_S,
    )
    from env_alsat_debug import (
        calculate_slew_energy_wh, CLOUD_THRESH, SLEW_ENERGY_ALPHA,
    )
    from dynamic_event import DynamicEvent, DYN_MULTIPLIER, MAX_OFFNADIR_RAD

    _N_STATIC = N_STATIC_TARGETS
    _is_dyn_action = _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS

    # Pre-step drain + flag reset (unchanged from original)
    if _is_dyn_action:
        try:
            _sat_pre = self.env.unwrapped.satellites[0]
            _sat_pre.was_image_taken_since_last_check()
            _sat_pre._dyn_img_fired    = False
            _sat_pre._dyn_reward_given = False
        except Exception:
            pass

    if int(action) < N_STATIC_TARGETS:
        self._n_static_actions_ep = getattr(self, "_n_static_actions_ep", 0) + 1

    # Compute SMDP duration
    try:
        sat = self.env.unwrapped.satellites[0]
        sat._dyn_reward_given = False
        if _is_dyn_action:
            try:
                sat.was_image_taken_since_last_check()
            except Exception:
                pass
        tau = _action_duration(sat, int(action))
    except Exception:
        tau = BASE_STEP_S

    tau   = float(np.clip(tau, BASE_STEP_S, MAX_ACTION_DUR_S))
    n_sub = max(1, min(MAX_SUB_STEPS, int(math.ceil(tau / BASE_STEP_S))))

    # Keep event_manager pointer live
    try:
        for _sx in self.env.unwrapped.satellites:
            _sx._event_manager = self._mgr
    except Exception:
        pass

    DRIFT_ACT = N_STATIC_TARGETS + N_DYN_SLOTS
    total_r   = 0.0
    last_obs  = None
    term = trunc = False
    info: dict = {}

    for _i in range(n_sub):
        obs_i, r_i, term, trunc, info = self.env.step(action)
        total_r  += (self._gamma_sub ** _i) * r_i
        last_obs  = obs_i
        if term or trunc:
            break

    smdp_discount = self._gamma_sub ** (tau / BASE_STEP_S)

    # Geometric imaging check
    try:
        _sat        = self.env.unwrapped.satellites[0]
        _locked_evt = getattr(_sat, "_locked_dyn_event", None)
        _has_locked = (
            _locked_evt is not None
            and not getattr(_locked_evt, "imaged", False)
            and _locked_evt.expiration_time > float(_sat.simulator.sim_time)
        )
        if _has_locked:
            _dyn_r = _dyn_imaging_check(_sat, info)
            if _dyn_r > 0.0:
                total_r += _dyn_r
                _sat._dyn_reward_given = True
                _sat._locked_dyn_event = None
                _sat._locked_dyn_slot  = None
    except Exception:
        pass

    # ── DYN reward injection (PATCH-1 + PATCH-2 + PATCH-5 applied here) ──
    info["dynamic_imaging_occurred"] = False   # PATCH-2: initialize
    info["last_dyn_urgency"]         = 0.0     # PATCH-5: initialize

    if _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS:
        try:
            _sat    = self.env.unwrapped.satellites[0]
            _slot   = int(action) - _N_STATIC
            _target = getattr(_sat, "_locked_dyn_event", None)
            _l_slot = getattr(_sat, "_locked_dyn_slot",  -1)
            _slew   = getattr(_sat, "last_slew_angle",   float("inf"))
            _fired  = getattr(_sat, "_dyn_reward_given", False)
            _already_done = _target.imaged if _target else False

            if (
                _target is not None
                and isinstance(_target, DynamicEvent)
                and _l_slot == _slot
                and _slew <= MAX_OFFNADIR_RAD
                and not _fired
                and not _already_done
                and _target.expiration_time > float(_sat.simulator.sim_time)
            ):
                _sat._dyn_reward_given = True

                _cloud = float(_target.cloud_cover)
                _prio  = float(_target.priority)
                _now   = float(_sat.simulator.sim_time)

                # ── PATCH-1: urgency uses frac_REMAINING (early = more) ──
                try:
                    _total_dur  = max(1.0, float(_target.expiration_time)
                                          - float(_target.appearance_time))
                    _remaining  = max(0.0, float(_target.expiration_time) - _now)
                    _frac_remaining = min(1.0, _remaining / _total_dur)
                    _urgency = 1.0 + 0.5 * _frac_remaining   # PATCH-1 FIX
                    # Was: 1.0 + 0.5 * frac_ELAPSED  (rewarded late imaging)
                    logger.debug(
                        f"[PATCH-1] Urgency (FIXED): remaining={_remaining:.0f}s  "
                        f"frac_rem={_frac_remaining:.2f}  urgency={_urgency:.2f}"
                    )
                except Exception:
                    _urgency = 1.25  # neutral fallback

                # ── PATCH-5: expose urgency to reward shaper ──────────────
                info["last_dyn_urgency"] = float(_urgency)

                if _cloud < CLOUD_THRESH:
                    _slew_mult   = getattr(_sat, "_slew_energy_multiplier", 1.0)
                    _slew_energy = calculate_slew_energy_wh(_slew, _slew_mult)
                    _dyn_r = (DYN_MULTIPLIER * _prio * (1.0 - _cloud) * _urgency
                              - SLEW_ENERGY_ALPHA * _slew_energy)
                    _sat._metrics["n_cloud_free"] += 1
                    # ── PATCH-2: flag successful imaging ─────────────────
                    info["dynamic_imaging_occurred"] = True
                else:
                    _dyn_r = -0.3 * _prio
                    _sat._metrics["n_cloudy"] += 1

                # Update metrics
                _sat._metrics["n_imaged"]           += 1
                _sat._metrics["total_reward"]        += _dyn_r
                _sat._metrics["total_slew_angle_deg"] += math.degrees(_slew)

                _evt_mgr = getattr(_sat, "_event_manager", None)
                if _evt_mgr is not None:
                    _evt_mgr.mark_imaged(_target, _now, _dyn_r)
                    _sat._metrics["n_dyn_imaged"] = _evt_mgr._metrics["n_imaged"]
                else:
                    _sat._metrics["n_dyn_imaged"] = \
                        _sat._metrics.get("n_dyn_imaged", 0) + 1

                info.setdefault("episode_metrics", {})["n_dyn_imaged"] = \
                    _sat._metrics.get("n_dyn_imaged", 0)

                # Log event details
                try:
                    _sat._last_dyn_event_log = {
                        "type":     _target.event_type,
                        "lat":      float(math.degrees(_target.lat_rad)),
                        "lon":      float(math.degrees(_target.lon_rad)),
                        "priority": _prio,
                        "cloud":    _cloud,
                        "reward":   _dyn_r,
                        "urgency":  _urgency,
                        "slot":     _slot,
                    }
                except Exception:
                    pass

                total_r += _dyn_r

        except Exception as _exc:
            logger.debug(f"[PATCH] DYN injection error: {_exc}")

    # ── Event lifecycle + PATCH-4: missed-penalty attribution ────────────
    total_missed_penalty = 0.0
    try:
        sat = self.env.unwrapped.satellites[0]
        now = float(sat.simulator.sim_time)
        dt  = max(0.0, now - self._prev_time)

        new_events = self._gen.step(now, dt)
        self._mgr.add_events(new_events)

        for _exp_evt in list(self._mgr._events):
            if not _exp_evt.imaged and _exp_evt.expiration_time <= now:
                _cloud_e = float(_exp_evt.cloud_cover)
                _prio_e  = float(_exp_evt.priority)
                _miss_p  = -0.5 * _prio_e * (1.0 - _cloud_e)
                total_r  += _miss_p
                total_missed_penalty += _miss_p
                sat._metrics["total_reward"] += _miss_p
                sat._metrics.setdefault("n_missed_events", 0)
                sat._metrics["n_missed_events"] += 1

        # PATCH-4: expose missed penalty separately so shaper can isolate it
        info["missed_event_penalty"] = float(total_missed_penalty)

        self._mgr.purge_expired(now)
        self._prev_time = now
        sat._metrics["n_dyn_detected"] = self._mgr._metrics["n_detected"]
    except Exception as exc:
        logger.debug(f"[PATCH] Event lifecycle error: {exc}")

    info["smdp_tau_s"]      = tau
    info["smdp_n_sub"]      = n_sub
    info["dynamic_metrics"] = self._mgr.get_metrics()

    tau_norm = tau / MAX_ACTION_DUR_S

    # Static forgetting penalty (unchanged)
    if term or trunc:
        n_static = getattr(self, "_n_static_actions_ep", 0)
        if n_static == 0:
            total_r -= 1.0
        self._n_static_actions_ep = 0

    return self._build_obs(last_obs, tau_norm), total_r, term, trunc, info


# ─────────────────────────────────────────────────────────────────────────────
# PATCH-3: Fix cloud threshold in _dyn_imaging_check
# ─────────────────────────────────────────────────────────────────────────────

def _patched_dyn_imaging_check(sat, info: dict) -> float:
    """
    PATCH-3: use CLOUD_THRESH=0.6 (same as static) instead of 0.9.
    Original used _DYN_CLOUD_THRESH = 0.9, allowing cloudy DYN imaging.
    """
    import numpy as _np
    import math as _math
    from env_alsat_debug import CLOUD_THRESH as _CLOUD_THRESH
    from dynamic_event import DYN_MULTIPLIER as _DYN_MULT

    _MAX_OFFNADIR_DEG = 45.0   # must match MAX_OFFNADIR_RAD in dynamic_event.py

    if getattr(sat, "_locked_dyn_event", None) is None:
        return 0.0
    locked_ev = getattr(sat, "_locked_dyn_event", None)
    if locked_ev is None or getattr(sat, "_dyn_reward_given", False):
        return 0.0

    try:
        try:
            r_sat = _np.asarray(sat.dynamics.r_SC_N, dtype=float).flatten()
        except AttributeError:
            try:
                r_sat = _np.asarray(sat.dynamics.r_BN_N, dtype=float).flatten()
            except AttributeError:
                return 0.0
        r_evt = _np.asarray(locked_ev.r_LP_P, dtype=float).flatten()
    except AttributeError:
        return 0.0

    norm_sat = float(_np.linalg.norm(r_sat))
    if norm_sat < 1e3:
        return 0.0

    nadir_unit = -r_sat / norm_sat
    to_evt     = r_evt - r_sat
    d          = float(_np.linalg.norm(to_evt))
    if d < 1.0:
        return 0.0
    to_evt_unit = to_evt / d

    cos_a       = float(_np.clip(_np.dot(nadir_unit, to_evt_unit), -1.0, 1.0))
    offnadir_deg = _math.degrees(_math.acos(cos_a))

    cloud = float(getattr(locked_ev, "cloud_cover", 1.0))

    # PATCH-3 FIX: use CLOUD_THRESH=0.6 instead of 0.9
    if offnadir_deg <= _MAX_OFFNADIR_DEG and cloud < _CLOUD_THRESH:
        sat._dyn_reward_given = True
        ep = info.setdefault("episode_metrics", {})
        ep["n_dyn_imaged"] = ep.get("n_dyn_imaged", 0) + 1
        if hasattr(locked_ev, "mark_accessed"):
            locked_ev.mark_accessed()
        pri = float(getattr(locked_ev, "priority", 1.0))
        logger.debug(
            f"[PATCH-3] Geometric check: offnadir={offnadir_deg:.1f}°  "
            f"cloud={cloud:.2f} < {_CLOUD_THRESH}  → reward granted"
        )
        return _DYN_MULT * pri * (1.0 - cloud)

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Apply all patches on import
# ─────────────────────────────────────────────────────────────────────────────

def apply_all_patches():
    """Apply all patches to the live module objects."""
    import env_alsat_dynamic as _dyn_mod

    # PATCH-1 + PATCH-2 + PATCH-4 + PATCH-5: replace step()
    original_step = _dyn_mod.DynamicObsWrapper.step
    _dyn_mod.DynamicObsWrapper.step = _patched_step
    logger.info("[PATCH] DynamicObsWrapper.step patched (urgency+flags+attribution)")

    # PATCH-3: replace _dyn_imaging_check module-level function
    _dyn_mod._dyn_imaging_check = _patched_dyn_imaging_check
    logger.info("[PATCH] _dyn_imaging_check patched (cloud threshold 0.9 → 0.6)")

    return {
        "original_step": original_step,
    }


# Auto-apply when imported
_patch_handles = apply_all_patches()
logger.info("[PATCH] env_alsat_dynamic_patch.py applied successfully (5 patches)")


def restore_originals():
    """Undo all patches — useful for ablation testing."""
    import env_alsat_dynamic as _dyn_mod
    _dyn_mod.DynamicObsWrapper.step = _patch_handles["original_step"]
    logger.info("[PATCH] Patches restored to originals")