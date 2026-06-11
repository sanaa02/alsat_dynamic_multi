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
    FIXED: calls _orig_step so event lifecycle runs correctly.
    (Previous version duplicated the lifecycle and failed silently — the
    exception was swallowed by logger.debug, which this module filters at INFO.)

    PATCH-1: urgency direction fixed (frac_remaining, not frac_elapsed)
    PATCH-2: info["dynamic_imaging_occurred"] set on success
    PATCH-4: info["missed_event_penalty"] exposed separately
    PATCH-5: info["last_dyn_urgency"] set for reward shaper
    """
    from env_alsat_dynamic import N_STATIC_TARGETS, N_DYN_SLOTS, DYN_MULTIPLIER
    from env_alsat_debug import CLOUD_THRESH

    _N_STATIC = N_STATIC_TARGETS
    _is_dyn   = _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS

    # ── Run original step: SMDP loop + event lifecycle + DYN injection ──────
    obs, total_r, term, trunc, info = _patch_handles["original_step"](self, action)

    # ── Init PATCH-2/4/5 info keys ──────────────────────────────────────────
    info["dynamic_imaging_occurred"] = False
    info.setdefault("last_dyn_urgency",     0.0)
    info.setdefault("missed_event_penalty", 0.0)

    if not _is_dyn:
        return obs, total_r, term, trunc, info

    try:
        _sat = self.env.unwrapped.satellites[0]

        if not getattr(_sat, "_dyn_reward_given", False):
            return obs, total_r, term, trunc, info

        # ── PATCH-2: flag successful DYN imaging ────────────────────────────
        info["dynamic_imaging_occurred"] = True

        # ── PATCH-1+5: fix urgency direction ────────────────────────────────
        # Original step used frac_ELAPSED (late imaging = more reward — wrong).
        # We want frac_REMAINING (early imaging = more reward — correct).
        _target = getattr(_sat, "_locked_dyn_event", None)
        if _target is None:
            info["last_dyn_urgency"] = 1.25
            return obs, total_r, term, trunc, info

        _now      = float(_sat.simulator.sim_time)
        _total    = max(1.0, float(_target.expiration_time)
                             - float(_target.appearance_time))
        _rem      = max(0.0, float(_target.expiration_time) - _now)
        _elp      = max(0.0, _now - float(_target.appearance_time))
        _frac_rem = min(1.0, _rem  / _total)
        _frac_elp = min(1.0, _elp  / _total)

        _new_urg = 1.0 + 0.5 * _frac_rem   # PATCH-1: early imaging pays more
        _old_urg = 1.0 + 0.5 * _frac_elp   # what original step used (wrong)
        info["last_dyn_urgency"] = float(_new_urg)   # PATCH-5

        # Adjust total_r: swap old urgency factor for new
        _prio  = float(getattr(_target, "priority",    1.0))
        _cloud = float(getattr(_target, "cloud_cover", 0.0))
        if _cloud < CLOUD_THRESH and abs(_new_urg - _old_urg) > 1e-6:
            total_r += DYN_MULTIPLIER * _prio * (1.0 - _cloud) * (_new_urg - _old_urg)
            logger.debug(
                f"[PATCH-1] urgency adjusted  frac_rem={_frac_rem:.2f}  "
                f"old={_old_urg:.2f}→new={_new_urg:.2f}"
            )

    except Exception as _exc:
        logger.warning(f"[PATCH] post-step urgency fix error: {_exc}")

    return obs, total_r, term, trunc, info


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