#!/usr/bin/env python3
"""
smdp_dynamic.py  —  ALSAT-EO-1  Phase 3  SMDP Wrapper for Dynamic Env
=======================================================================
Converts the DynamicObsWrapper (fixed-step MDP) into a proper SMDP by:

  [SMDP-1] Variable action duration
           τ(a) = slew_time(a) + IMAGING_DUR_S  (or BASE_STEP_S for drift)
           Uses bang-bang slew model: t_slew = 2 * sqrt(θ / α_max)

  [SMDP-2] Discounted reward accumulation
           R = Σ γ^(i * BASE_STEP_S / STEP_REF) * r_i   over n_sub sub-steps
           γ_eff per sub-step = γ ^ (BASE_STEP_S / STEP_REF)

  [SMDP-3] Sojourn-time feature
           Observation is extended from (55,) to (56,) by appending
           τ_norm = τ / MAX_ACTION_DUR_S  ∈ [0, 1].
           This lets the policy learn to trade fast short actions against
           high-reward long-slew actions.

Survey reference: §3.3.1, Table 3, Figure 2 — semi-Markov formulation.
Proposal §3.2 — variable-horizon temporal abstraction.

Obs space  : Box(-inf, inf, shape=(56,))  (DynamicObsWrapper (55,) + τ_norm)
Action space: Discrete(24)  (unchanged from DynamicObsWrapper)

Usage
-----
    from smdp_dynamic import make_smdp_dynamic_env
    env = make_smdp_dynamic_env(targets_path, cloud_json_path, event_rate=2.0)
    obs, info = env.reset(seed=42)  # obs.shape == (56,)
"""
from __future__ import annotations
# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------




import math
import logging
from typing import Optional

import gymnasium as gym
import numpy as np

from env_alsat_dynamic import (
    make_dynamic_env,
    DynamicObsWrapper,
    SingleSatelliteEnv,
    N_TOTAL_ACTIONS,
    OBS_TOTAL_DIM,
)
from env_alsat_debug import (
    calculate_slew_angle_to_target,
    calculate_slew_time,
    IMAGING_DUR_S,
    SCHED_STEP_S,
    SIM_DURATION_S,
    BSK_SIM_RATE_S,
)

logger = logging.getLogger(__name__)

# ── SMDP constants ────────────────────────────────────────────────────────────
BASE_STEP_S      = 30.0      # sub-step duration (matches smdp_wrapper.py)
STEP_REF_S       = SCHED_STEP_S   # 1200 s — reference horizon for discounting
MAX_ACTION_DUR_S = 200.0     # cap for sojourn normalisation (> max slew + 20 s)
OBS_SMDP_DIM     = OBS_TOTAL_DIM + 1   # 56


def _compute_action_duration(satellite, action: int, n_static: int) -> float:
    """
    Compute sojourn time τ(a) for this action.
      - Drift / empty dynamic slot → BASE_STEP_S
      - Static target i            → slew_time(i) + IMAGING_DUR_S
      - Dynamic event slot j       → slew_time(event_j) + IMAGING_DUR_S
    """
    drift_action = n_static + 3   # last action index
    if action >= drift_action:
        return BASE_STEP_S

    if action < n_static:
        # Static target
        target = satellite.scenario.targets[action]
        slew   = calculate_slew_angle_to_target(satellite, target)
        return calculate_slew_time(slew) + IMAGING_DUR_S

    # Dynamic event slot
    slot      = action - n_static
    event_mgr = getattr(satellite, "_event_manager", None)
    if event_mgr is None:
        return BASE_STEP_S

    try:
        now   = float(satellite.simulator.sim_time)
        slots = event_mgr.get_slots(satellite, now)
        event = slots[slot] if slot < len(slots) else None
    except Exception:
        event = None

    if event is None:
        return BASE_STEP_S

    slew = calculate_slew_angle_to_target(satellite, event)
    return calculate_slew_time(slew) + IMAGING_DUR_S


# ============================================================================
#  SMDPDynamicWrapper
# ============================================================================

class SMDPDynamicWrapper(gym.Wrapper):
    """
    Wraps DynamicObsWrapper to implement SMDP semantics.

    Key behaviour
    -------------
    step(action):
      1. Compute sojourn time τ = slew_time + imaging_time (or BASE_STEP_S).
      2. Run ceil(τ / BASE_STEP_S) sub-steps of the inner DynamicObsWrapper.
      3. Accumulate reward with per-sub-step discount γ_sub = γ^(BASE_STEP_S/STEP_REF).
      4. Return (augmented_obs, discounted_reward, done, truncated, info).

    Observation augmentation
    ------------------------
    Appends sojourn_time_norm = τ / MAX_ACTION_DUR_S to each observation,
    giving the policy explicit information about action cost.

    Parameters
    ----------
    env            : DynamicObsWrapper
    gamma          : discount factor per STEP_REF_S (default 0.99)
    max_sub_steps  : safety cap (default 20)
    """

    def __init__(self,
                 env:           DynamicObsWrapper,
                 gamma:         float = 0.99,
                 max_sub_steps: int   = 20):
        super().__init__(env)
        self.gamma         = gamma
        self.max_sub_steps = max_sub_steps
        # Discount per sub-step (BASE_STEP_S per sub-step normalised to STEP_REF_S)
        self._gamma_sub    = gamma ** (BASE_STEP_S / STEP_REF_S)

        self.last_sojourn_s  = 0.0
        self.last_n_sub      = 0

        # Extended observation space
        self.observation_space = gym.spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (OBS_SMDP_DIM,),
            dtype = np.float32,
        )
        # Action space unchanged
        self.action_space = gym.spaces.Discrete(N_TOTAL_ACTIONS)

    # ── reset / step ──────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_sojourn_s = 0.0
        self.last_n_sub     = 0
        return self._aug(obs, 0.0), info

    def step(self, action: int):
        # Determine sojourn time
        try:
            sat      = self.env.env.unwrapped.satellites[0]
            n_static = len(sat.scenario.targets)
            tau      = _compute_action_duration(sat, int(action), n_static)
        except Exception:
            tau = BASE_STEP_S

        tau = float(np.clip(tau, BASE_STEP_S, MAX_ACTION_DUR_S))
        n_sub = max(1, min(self.max_sub_steps, int(math.ceil(tau / BASE_STEP_S))))

        self.last_sojourn_s = tau
        self.last_n_sub     = n_sub

        total_reward = 0.0
        discount     = 1.0
        last_obs     = None
        terminated   = truncated = False
        info         = {}

        for _ in range(n_sub):
            obs_i, r_i, term_i, trunc_i, info_i = self.env.step(action)
            total_reward += discount * r_i
            discount     *= self._gamma_sub
            last_obs      = obs_i
            info          = info_i
            if term_i or trunc_i:
                terminated = term_i
                truncated  = trunc_i
                break

        tau_norm = tau / MAX_ACTION_DUR_S
        info["smdp_sojourn_s"] = tau
        info["smdp_n_sub"]     = n_sub
        return self._aug(last_obs, tau_norm), total_reward, terminated, truncated, info

    # ── Augment observation with sojourn feature ──────────────────────────────

    @staticmethod
    def _aug(base_obs: np.ndarray, tau_norm: float) -> np.ndarray:
        return np.append(base_obs.astype(np.float32),
                         np.float32(np.clip(tau_norm, 0.0, 1.0)))


# ============================================================================
#  Factory
# ============================================================================

def make_smdp_dynamic_env(
    targets_path:    str,
    cloud_json_path: str,
    event_rate:      float = 2.0,
    duration_s:      float = SIM_DURATION_S,
    sim_rate:        float = BSK_SIM_RATE_S,
    sat_name:        str   = "ALSAT-1",
    sat_args:        dict  = None,
    seed:            int   = 42,
    gamma:           float = 0.99,
    max_sub_steps:   int   = 20,
    render_mode             = None,
) -> SMDPDynamicWrapper:
    """
    Build the full SMDP dynamic targeting environment.

    Returns a SMDPDynamicWrapper with:
      obs shape  = (56,)  — 43 base + 12 dynamic event + 1 sojourn
      action sp  = Discrete(24)

    Use this as the training environment for Phase 3 SMDP-PPO.
    """
    dyn_env = make_dynamic_env(
        targets_path    = targets_path,
        cloud_json_path = cloud_json_path,
        event_rate      = event_rate,
        duration_s      = duration_s,
        sim_rate        = sim_rate,
        sat_name        = sat_name,
        sat_args        = sat_args,
        seed            = seed,
        render_mode     = render_mode,
    )
    return SMDPDynamicWrapper(dyn_env, gamma=gamma, max_sub_steps=max_sub_steps)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys, logging
    os.environ.setdefault("BSK_OUTPUT_LEVEL", "2")
    os.environ.setdefault("BSK_LOG_LEVEL", "WARNING")
    logging.basicConfig(level=logging.INFO)

    ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TARGETS    = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
    CLOUD_JSON = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")

    print("=" * 70)
    print("smdp_dynamic.py  —  sanity check")
    print("=" * 70)

    env = make_smdp_dynamic_env(TARGETS, CLOUD_JSON, event_rate=2.0)
    obs, info = env.reset(seed=42)
    print(f"  obs shape : {obs.shape}   (expected 56)")
    print(f"  sojourn   : obs[-1] = {obs[-1]:.4f}  (0 at reset)")
    assert obs.shape == (OBS_SMDP_DIM,), f"Bad obs shape {obs.shape}"

    print("\n  Running 5 steps (cycling through static + dynamic actions)...")
    for step in range(5):
        action = step % N_TOTAL_ACTIONS
        obs, r, term, trunc, info = env.step(action)
        soj  = info.get("smdp_sojourn_s", 0.0)
        nsub = info.get("smdp_n_sub", 0)
        print(f"    step {step+1}  act={action:2d}  r={r:+.4f}  "
              f"sojourn={soj:.1f}s  n_sub={nsub}  tau_feat={obs[-1]:.3f}")
        if term or trunc:
            break

    print("\nTest passed.")
    env.close()
