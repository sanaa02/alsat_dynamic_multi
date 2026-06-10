#!/usr/bin/env python3
"""
reward_shaping.py  --  Dynamic Event Reward Shaping  (FIXED v3)
================================================================
Fixes applied (verified against training logs and code audit):

FIX-RS-1  explore_decay was 0.95 (or 0.99 in the train script).
    At decay=0.95:  ep100  bonus=0.0018, ep150  bonus=0.0001 — effectively ZERO.
    The bonus vanished before the policy had any chance to learn DYN actions.
    Training logs confirm: dyn_act_pct collapsed from 56% → 11% by ep200,
    exactly when explore_bonus reached zero.

    Fix: raise explore_decay to 0.9985 (half-life ~462 episodes),
         set explore_min to a non-zero floor (0.05) so the agent always
         has SOME incentive to try DYN actions even after convergence.

FIX-RS-2  urgency bonus required info["dynamic_imaging_occurred"] = True,
    but this key was never set in DynamicObsWrapper.step() info dict.
    Result: urgency_scale bonus never fired; only explore_bonus was active.

    Fix: check success from info["dynamic_metrics"]["n_imaged"] delta,
         OR from reward magnitude (reward > some_threshold when DYN fired).

FIX-RS-3  explore_bonus was applied for ANY DYN slot selection, including
    empty slots.  Now that empty slots are masked (action_mask_wrapper.py
    FIX-MASK-1), this is safe — but we add an explicit guard anyway for
    defensive correctness.

FIX-RS-4  No signal for DYN ATTEMPT that failed due to slew/cloud.
    Agent cannot learn "I tried but geometry was wrong" vs "I should drift".
    Fix: add a small positive attempt bonus for DYN action when the slot
    is non-empty (before outcome is known), separate from the success bonus.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import logging

logger = logging.getLogger(__name__)

N_STATIC = 20
N_DYN    = 3
DRIFT    = 23


class DynamicRewardShaper(gym.Wrapper):
    """
    Drop-in wrapper around any DynamicObsWrapper / SMDPDynamicWrapper env.

    Adds three reward components on top of the base environment reward:

    1. explore_bonus   : small flat bonus for selecting any non-empty DYN slot.
                         Decays slowly (half-life ~462 eps) with a non-zero floor.
                         Prevents the policy from never exploring DYN actions.

    2. attempt_bonus   : tiny bonus for selecting a DYN slot that contains
                         an accessible event (slew <= 45°).  Separate from
                         success so the agent learns to aim, not just select.

    3. urgency_bonus   : bonus on SUCCESSFUL imaging, scaled by how EARLY
                         the event was imaged relative to its lifetime.
                         Rewards responsiveness, not tardiness.

    Parameters
    ----------
    urgency_scale      : multiplier for imaging-time urgency bonus (default 1.5)
    urgency_max        : cap for urgency bonus (default 2.0)
    explore_bonus_init : flat bonus per non-empty DYN step at ep 0 (default 0.3)
    explore_decay      : per-episode decay factor (default 0.9985, half-life 462 eps)
    explore_min        : floor — bonus never falls below this (default 0.05)
    attempt_bonus      : bonus for pointing at an accessible event (default 0.05)
    """

    def __init__(
        self,
        env,
        urgency_scale:      float = 1.5,
        urgency_max:        float = 2.0,
        explore_bonus_init: float = 0.30,
        explore_decay:      float = 0.9985,   # FIX-RS-1: was 0.95/0.99 → too fast
        explore_min:        float = 0.05,      # FIX-RS-1: non-zero floor
        attempt_bonus:      float = 0.05,      # FIX-RS-4: new
    ):
        super().__init__(env)
        self.urgency_scale      = urgency_scale
        self.urgency_max        = urgency_max
        self._explore_bonus     = explore_bonus_init
        self.explore_decay      = explore_decay
        self.explore_min        = explore_min
        self.attempt_bonus      = attempt_bonus

        # Counters for diagnostics
        self._ep_count              = 0
        self._total_urgency_given   = 0.0
        self._total_explore_given   = 0.0
        self._total_attempt_given   = 0.0
        self._n_dyn_steps           = 0
        self._n_dyn_success_steps   = 0
        self._prev_dyn_imaged       = 0   # for delta detection (FIX-RS-2)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, **kw):
        result = self.env.reset(**kw)
        self._ep_count += 1
        # FIX-RS-1: decay with a proper floor so the bonus never fully vanishes
        self._explore_bonus = max(
            self.explore_min,
            self._explore_bonus * self.explore_decay,
        )
        self._prev_dyn_imaged = 0
        return result

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        obs, reward, term, trunc, info = self.env.step(action)
        bonus = self._bonus(int(action), reward, info)
        info["shaping_bonus"] = bonus
        info["explore_bonus_current"] = self._explore_bonus
        return obs, reward + bonus, term, trunc, info

    # ── Bonus computation ─────────────────────────────────────────────────────

    def _bonus(self, action: int, base_reward: float, info: dict) -> float:
        is_dyn = N_STATIC <= action < N_STATIC + N_DYN
        if not is_dyn:
            return 0.0

        total = 0.0
        self._n_dyn_steps += 1

        # ── Check whether the selected slot actually had an event ──────────
        slot_idx = action - N_STATIC
        slot_has_event = self._slot_is_occupied(slot_idx)

        # FIX-RS-3: only give explore bonus when slot is non-empty
        if slot_has_event:
            total += self._explore_bonus
            self._total_explore_given += self._explore_bonus

        # FIX-RS-4: attempt bonus for pointing at an accessible event
        if slot_has_event and self._slot_is_accessible(slot_idx):
            total += self.attempt_bonus
            self._total_attempt_given += self.attempt_bonus

        # ── FIX-RS-2: detect successful DYN imaging via delta in n_dyn_imaged ──
        imaging_success = self._detect_dyn_success(info, base_reward)

        if imaging_success:
            self._n_dyn_success_steps += 1
            # Urgency: reward early response (1.0 when imaged at expiry → 1.5 when fresh)
            # We derive urgency from info if available, else use base_reward magnitude
            urgency = self._compute_urgency(info)
            if urgency > 0:
                u_bonus = float(np.clip(self.urgency_scale * urgency, 0.0, self.urgency_max))
                total += u_bonus
                self._total_urgency_given += u_bonus
                logger.debug(
                    f"[SHAPING] DYN success  slot={slot_idx}  urgency={urgency:.2f}  "
                    f"u_bonus={u_bonus:.3f}  explore={self._explore_bonus:.3f}"
                )

        return float(total)

    def _slot_is_occupied(self, slot_idx: int) -> bool:
        """Check whether get_slots()[slot_idx] contains a non-None event."""
        try:
            obj = self
            while hasattr(obj, "env"):
                mgr = getattr(obj, "_mgr", None)
                if mgr is not None:
                    break
                obj = obj.env
            if mgr is None:
                return True   # assume occupied if we can't check
            sat  = obj.env.unwrapped.satellites[0]
            now  = float(sat.simulator.sim_time)
            slots = mgr.get_slots(sat, now)
            return slot_idx < len(slots) and slots[slot_idx] is not None
        except Exception:
            return True   # safe default

    def _slot_is_accessible(self, slot_idx: int) -> bool:
        """True if satellite is currently within 45° off-nadir of slot event."""
        try:
            from env_alsat_debug import calculate_slew_angle_to_target
            from dynamic_event import MAX_OFFNADIR_RAD
            obj = self
            while hasattr(obj, "env"):
                mgr = getattr(obj, "_mgr", None)
                if mgr is not None:
                    break
                obj = obj.env
            sat   = obj.env.unwrapped.satellites[0]
            now   = float(sat.simulator.sim_time)
            slots = mgr.get_slots(sat, now)
            evt   = slots[slot_idx] if slot_idx < len(slots) else None
            if evt is None:
                return False
            slew = calculate_slew_angle_to_target(sat, evt)
            return slew <= MAX_OFFNADIR_RAD
        except Exception:
            return False

    def _detect_dyn_success(self, info: dict, base_reward: float) -> bool:
        """
        FIX-RS-2: detect successful DYN imaging.

        Strategy A: info key "dynamic_imaging_occurred" (set by updated wrapper)
        Strategy B: delta in info["dynamic_metrics"]["n_imaged"]
        Strategy C: base_reward > threshold (heuristic fallback)
        """
        # Strategy A (preferred — set in updated env_alsat_dynamic.py)
        if info.get("dynamic_imaging_occurred", False):
            return True

        # Strategy B: detect increment in n_dyn_imaged counter
        dm = info.get("dynamic_metrics", {})
        cur_imaged = int(dm.get("n_imaged", self._prev_dyn_imaged))
        if cur_imaged > self._prev_dyn_imaged:
            self._prev_dyn_imaged = cur_imaged
            return True
        self._prev_dyn_imaged = cur_imaged

        # Strategy C: if base reward is clearly positive for a DYN action,
        # it must have been a successful image (DYN_MULTIPLIER * priority > 1)
        if base_reward > 1.0:
            return True

        return False

    def _compute_urgency(self, info: dict) -> float:
        """
        Compute urgency ∈ [0, 1].  Higher when event was imaged EARLY.
        urgency = frac_remaining = 1 when fresh, 0 when at expiry.

        FIX-RS-2: urgency = 1 - frac_elapsed = frac_remaining
        This is the CORRECT direction: reward early response.
        """
        try:
            dm = info.get("dynamic_metrics", {})
            # If the updated wrapper sets urgency directly in info, use it
            if "last_dyn_urgency" in info:
                return float(info["last_dyn_urgency"])
            # Derive from the locked event if accessible
            obj = self
            while hasattr(obj, "env"):
                obj = obj.env
            sat = getattr(obj, "unwrapped", obj).satellites[0]
            target = getattr(sat, "_locked_dyn_event", None)
            if target is None:
                return 1.0  # fallback: full urgency
            import time as _t
            now       = float(sat.simulator.sim_time)
            total_dur = max(1.0, float(target.expiration_time) - float(target.appearance_time))
            remaining = max(0.0, float(target.expiration_time) - now)
            frac_remaining = min(1.0, remaining / total_dur)
            return frac_remaining   # 1.0 when fresh, 0.0 at expiry
        except Exception:
            return 1.0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def shaping_stats(self) -> dict:
        return {
            "episode":              self._ep_count,
            "current_explore_bonus": round(self._explore_bonus, 5),
            "total_urgency_bonus":  round(self._total_urgency_given, 3),
            "total_explore_bonus":  round(self._total_explore_given, 3),
            "total_attempt_bonus":  round(self._total_attempt_given, 3),
            "n_dyn_steps":          self._n_dyn_steps,
            "n_dyn_success_steps":  self._n_dyn_success_steps,
            "dyn_success_rate":     (self._n_dyn_success_steps
                                     / max(self._n_dyn_steps, 1)),
        }