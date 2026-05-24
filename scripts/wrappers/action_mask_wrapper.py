#!/usr/bin/env python3
"""
action_mask_wrapper.py  --  Constraint-Aware Action Masking
Masks physically impossible actions (Wang et al. AEOS-Former NeurIPS 2025).
Install sb3-contrib for training-time masking: pip install sb3-contrib
"""
import logging, numpy as np, gymnasium as gym
logger = logging.getLogger(__name__)
N_STATIC, N_DYN = 20, 3


def compute_action_mask(env: gym.Env) -> np.ndarray:
    """True = action is feasible right now."""
    n    = env.action_space.n
    mask = np.ones(n, dtype=bool)
    try:
        obj = env
        while hasattr(obj, "env"): obj = obj.env
        base = getattr(obj, "unwrapped", obj)
        sat  = base.satellites[0]
        now  = float(sat.simulator.sim_time)
        opps = getattr(sat, "upcoming_opportunities", [])

        for i, tgt in enumerate(sat.scenario.targets):
            if i >= N_STATIC: break
            accessible = False
            for opp in opps:
                try:
                    o = opp.get("object") if isinstance(opp, dict) else getattr(opp, "object", None)
                    w = opp.get("window", [0,1]) if isinstance(opp, dict) else getattr(opp, "window", [0,1])
                    t = opp.get("type","") if isinstance(opp, dict) else getattr(opp, "type","")
                    if o is tgt and t == "target" and w[0] <= now <= w[1]:
                        accessible = True; break
                except Exception: pass
            mask[i] = accessible

        mgr = None
        obj = env
        while hasattr(obj, "env"):
            mgr = getattr(obj, "_mgr", None)
            if mgr: break
            obj = obj.env
        if mgr:
            slots = mgr.get_slots(sat, now)
            for j in range(N_DYN):
                mask[N_STATIC + j] = (j < len(slots) and slots[j] is not None)
    except Exception as exc:
        logger.debug(f"mask error: {exc}"); mask[:] = True

    mask[n - 1] = True  # drift always valid
    return mask


class ActionMaskWrapper(gym.Wrapper):
    def get_action_mask(self): return compute_action_mask(self)


class InferenceTimeMaskWrapper(gym.Wrapper):
    """Fallback for when sb3-contrib not installed."""
    def step(self, action):
        mask = compute_action_mask(self)
        if not mask[int(action)]:
            valid = np.where(mask)[0]
            action = int(valid[0]) if len(valid) > 0 else self.action_space.n - 1
        return self.env.step(action)
    def get_action_mask(self): return compute_action_mask(self)


def make_masked_env(base_env: gym.Env) -> gym.Env:
    try:
        from sb3_contrib.common.wrappers import ActionMasker
        return ActionMasker(ActionMaskWrapper(base_env), lambda e: e.get_action_mask())
    except ImportError:
        logger.warning("sb3-contrib not found; inference-only masking. pip install sb3-contrib")
        return InferenceTimeMaskWrapper(base_env)