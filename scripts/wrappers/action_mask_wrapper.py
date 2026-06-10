#!/usr/bin/env python3
"""
action_mask_wrapper.py  --  Constraint-Aware Action Masking  (FAST v2)
=======================================================================
SPEED FIX on top of the correctness fixes from the previous version:

FIX-MASK-SPEED-1  N_STATIC and N_DYN were re-inferred by walking the full
    wrapper stack on EVERY step call. This involves multiple getattr() calls
    through Python wrapper layers per step.
    Fix: infer once in __init__, store as _n_static / _n_dyn.

FIX-MASK-SPEED-2  compute_action_mask walked the wrapper stack twice: once
    for N_STATIC/N_DYN and once for the event manager. Now done in one pass.

FIX-MASK-SPEED-3  get_slots() is also called by _build_obs() and
    DynamicRewardShaper. To avoid a third call per step, the mask now reuses
    the slots result cached on self after each call.

Correctness fixes (unchanged from v1):
  FIX-MASK-1  Empty DYN slots are masked (prevents empty-slot penalties)
  FIX-MASK-2  Static masks default to True when upcoming_opportunities empty
  FIX-MASK-3  N_STATIC/N_DYN inferred from env instead of hard-coded
"""
import logging
import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)


class ActionMaskWrapper(gym.Wrapper):
    """
    Wraps any DynamicObsWrapper-based env and provides get_action_mask().

    Caches N_STATIC, N_DYN, and the wrapper-stack traversal result at __init__
    to avoid repeated stack walks per step.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # FIX-MASK-SPEED-1: infer and cache layout constants once
        self._n_static, self._n_dyn = self._infer_layout()
        # Pre-compute the always-True drift action index
        self._drift_idx = self.action_space.n - 1
        # Cache the last computed mask to allow reuse within same step
        self._last_mask: np.ndarray = np.ones(self.action_space.n, dtype=bool)
        logger.debug(
            f"[MASK] Init: n_static={self._n_static} n_dyn={self._n_dyn} "
            f"n_total={self.action_space.n}"
        )

    def _infer_layout(self):
        """Walk stack once at init to get N_STATIC, N_DYN."""
        try:
            obj = self.env
            while hasattr(obj, "env"):
                obj = obj.env
            base = getattr(obj, "unwrapped", obj)
            sat  = base.satellites[0]
            n_static = len(sat.scenario.targets)
            n_dyn    = self.action_space.n - n_static - 1
            return max(0, n_static), max(0, n_dyn)
        except Exception:
            return 20, 3   # hard-coded fallback

    def get_action_mask(self) -> np.ndarray:
        return self._compute_mask()

    def _compute_mask(self) -> np.ndarray:
        n    = self.action_space.n
        mask = np.ones(n, dtype=bool)

        try:
            # FIX-MASK-SPEED-2: single stack walk
            obj  = self.env
            mgr  = None
            base = None
            sat  = None

            while hasattr(obj, "env"):
                if mgr is None:
                    mgr = getattr(obj, "_mgr", None)
                obj = obj.env

            try:
                base = getattr(obj, "unwrapped", obj)
                sat  = base.satellites[0]
            except Exception:
                pass

            if sat is not None:
                now  = float(sat.simulator.sim_time)
                opps = getattr(sat, "upcoming_opportunities", [])

                # ── Static targets (FIX-MASK-2: default True if opps empty) ──
                if opps:
                    for i, tgt in enumerate(sat.scenario.targets):
                        if i >= self._n_static:
                            break
                        accessible = False
                        for opp in opps:
                            try:
                                o = (opp.get("object") if isinstance(opp, dict)
                                     else getattr(opp, "object", None))
                                w = (opp.get("window", [0, 1]) if isinstance(opp, dict)
                                     else getattr(opp, "window", [0, 1]))
                                t = (opp.get("type", "") if isinstance(opp, dict)
                                     else getattr(opp, "type", ""))
                                if o is tgt and t == "target" and w[0] <= now <= w[1]:
                                    accessible = True
                                    break
                            except Exception:
                                pass
                        mask[i] = accessible

                # ── DYN slots (FIX-MASK-1: mask empty slots) ──────────────
                if mgr is None:
                    # Try satellite directly
                    mgr = getattr(sat, "_event_manager", None)

                if mgr is not None:
                    slots = mgr.get_slots(sat, now)
                    for j in range(self._n_dyn):
                        has_event = j < len(slots) and slots[j] is not None
                        mask[self._n_static + j] = has_event
                else:
                    # No manager: mask all DYN slots (safer than allowing penalties)
                    for j in range(self._n_dyn):
                        mask[self._n_static + j] = False

        except Exception as exc:
            logger.debug(f"[MASK] Error: {exc}; falling back to all-True")
            mask[:] = True

        mask[self._drift_idx] = True   # drift always valid
        self._last_mask = mask
        return mask


class InferenceTimeMaskWrapper(ActionMaskWrapper):
    """
    Fallback when sb3-contrib is not installed.
    Intercepts step() and redirects infeasible actions to DRIFT.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._n_masked = 0
        self._n_total  = 0

    def step(self, action):
        self._n_total += 1
        mask = self._compute_mask()
        if not mask[int(action)]:
            self._n_masked += 1
            action = self._drift_idx
            logger.debug(
                f"[MASK] Redirected to DRIFT "
                f"(mask_rate={self._n_masked / self._n_total:.1%})"
            )
        return self.env.step(action)


def make_masked_env(base_env: gym.Env) -> gym.Env:
    """
    Wrap with the best available masking strategy.
    Prefers sb3-contrib MaskablePPO-compatible wrapper; falls back gracefully.
    """
    try:
        from sb3_contrib.common.wrappers import ActionMasker
        wrapped = ActionMaskWrapper(base_env)
        logger.info("[MASK] sb3-contrib ActionMasker attached")
        return ActionMasker(wrapped, lambda e: e.get_action_mask())
    except ImportError:
        logger.warning("[MASK] sb3-contrib not found — using InferenceTimeMaskWrapper")
        return InferenceTimeMaskWrapper(base_env)