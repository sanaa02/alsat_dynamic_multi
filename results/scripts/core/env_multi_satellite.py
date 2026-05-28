#!/usr/bin/env python3
"""
env_multi_satellite.py  --  Multi-Satellite Coordination Environment
====================================================================
MultiSatelliteCoordVecEnv  implements parameter-sharing MAPPO (CTDE).

Architecture
------------
  N SingleSatelliteEnv instances share:
    - Same targets + events (via same config files)
    - A ClaimRegistry (in-process dict) that tracks imaged targets per episode

  Each satellite's observation is AUGMENTED with:
    [own_obs (56) | claimed_bitmap (20) | other_sat_summary (4*(N-1))]

  other_sat_summary per satellite j:
    [norm_angle_to_j, j_last_action/24, j_cf_forecast, j_priority_target]

  This gives each agent coordination context — it knows which targets are
  already claimed and the approximate positions of other satellites.

SB3 Integration
---------------
  MultiSatelliteCoordVecEnv extends VecEnv directly, so SB3's PPO trains
  on all N satellites simultaneously with full parameter sharing.

  To train:
    vec = MultiSatelliteCoordVecEnv(make_env_fn, n_satellites=2, seed=42)
    model = PPO("MlpPolicy", vec, ...)
    model.learn(total_timesteps=...)

Citation: Yu et al. 2022 "The Surprising Effectiveness of PPO in MARL" (MAPPO)
"""
from __future__ import annotations

import os, sys, logging
from typing import Callable, Optional, List, Any
import numpy as np
import gymnasium as gym

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs, VecEnvStepReturn, VecEnvIndices,
)

logger = logging.getLogger(__name__)

N_STATIC   = 20
N_DYN      = 3
N_ACTIONS  = 24   # 20 static + 3 dyn + 1 drift
OBS_DIM    = 56   # base DynamicObsWrapper obs
N_CLAIM    = 20   # claimed targets bitmap (1 per static target)
N_OTHER    = 4    # features per other satellite


class ClaimRegistry:
    """
    Shared in-process registry tracking which targets/events have been
    imaged in the current episode across all satellites.

    Thread safety: not needed — DummyVecEnv-style single-threaded execution.
    """

    def __init__(self):
        self._claimed_static: set = set()   # indices of claimed static targets
        self._claimed_events: set = set()   # event IDs of claimed dyn events

    def reset(self):
        self._claimed_static.clear()
        self._claimed_events.clear()

    def try_claim_static(self, target_idx: int) -> bool:
        """Returns True if claim was successful (not already taken)."""
        if target_idx in self._claimed_static:
            return False
        self._claimed_static.add(target_idx)
        return True

    def try_claim_event(self, event_id: Any) -> bool:
        if event_id in self._claimed_events:
            return False
        self._claimed_events.add(event_id)
        return True

    @property
    def bitmap(self) -> np.ndarray:
        """Returns a (20,) binary array: 1 = target already claimed."""
        bm = np.zeros(N_STATIC, dtype=np.float32)
        for idx in self._claimed_static:
            if 0 <= idx < N_STATIC:
                bm[idx] = 1.0
        return bm


class MultiSatelliteCoordVecEnv(VecEnv):
    """
    Multi-satellite VecEnv with parameter-sharing and coordination.

    Parameters
    ----------
    make_env_fn     : callable(sat_idx, seed) -> gym.Env (unwrapped DynamicObs env)
    n_satellites    : number of satellites (default 2)
    seed            : base random seed (each sat gets seed + sat_idx)
    """

    def __init__(
        self,
        make_env_fn: Callable[[int, int], gym.Env],
        n_satellites: int = 2,
        seed: int = 42,
    ):
        self.n_sats     = n_satellites
        self.seed       = seed
        self._claim_reg = ClaimRegistry()

        # Build N envs
        self.envs: List[gym.Env] = [
            make_env_fn(sat_idx=i, seed=seed + i)
            for i in range(n_satellites)
        ]

        # Augmented obs/action spaces
        aug_obs_dim = OBS_DIM + N_CLAIM + N_OTHER * (n_satellites - 1)
        obs_space = gym.spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (aug_obs_dim,),
            dtype = np.float32,
        )
        act_space = gym.spaces.Discrete(N_ACTIONS)

        super().__init__(
            num_envs       = n_satellites,
            observation_space = obs_space,
            action_space      = act_space,
        )

        # Per-satellite last-known state for augmentation
        self._last_obs     = np.zeros((n_satellites, OBS_DIM), dtype=np.float32)
        self._last_actions = np.zeros(n_satellites, dtype=np.int32)
        self._buf_obs      = np.zeros((n_satellites, aug_obs_dim), dtype=np.float32)
        self._buf_rew      = np.zeros(n_satellites, dtype=np.float32)
        self._buf_done     = np.zeros(n_satellites, dtype=bool)
        self._buf_info     = [{} for _ in range(n_satellites)]

        logger.info(
            f"MultiSatelliteCoordVecEnv: {n_satellites} satellites, "
            f"obs_dim={aug_obs_dim} (base {OBS_DIM} + claim {N_CLAIM} + "
            f"other {N_OTHER*(n_satellites-1)})"
        )

    # ── VecEnv API ────────────────────────────────────────────────────────

    def reset(self) -> VecEnvObs:
        self._claim_reg.reset()
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self.seed + i)
            self._last_obs[i] = obs
        self._last_actions[:] = N_ACTIONS - 1  # drift
        return self._augment_all_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = actions.copy()

    def step_wait(self) -> VecEnvStepReturn:
        actions = self._pending_actions

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            action = int(action)
            obs, rew, term, trunc, info = env.step(action)

            # ── Apply claim penalty ──────────────────────────────────────
            if 0 <= action < N_STATIC:
                if not self._claim_reg.try_claim_static(action):
                    rew = 0.0   # already imaged by another satellite
                    info["double_imaging_penalty"] = True
            elif N_STATIC <= action < N_STATIC + N_DYN:
                event_id = info.get("event_id", action)  # use action idx as fallback
                if not self._claim_reg.try_claim_event(event_id):
                    rew = 0.0
                    info["double_imaging_penalty"] = True

            self._last_obs[i]     = obs
            self._last_actions[i] = action
            self._buf_rew[i]      = rew
            self._buf_done[i]     = term or trunc
            self._buf_info[i]     = info

        obs_aug = self._augment_all_obs()

        # Reset claim registry when all envs are done
        if np.all(self._buf_done):
            self._claim_reg.reset()

        # Handle episode resets for individual envs
        for i, (done, env) in enumerate(zip(self._buf_done, self.envs)):
            if done:
                obs_reset, _ = env.reset(seed=self.seed + i)
                self._last_obs[i] = obs_reset
                obs_aug[i] = self._augment_obs(i)

        return obs_aug, self._buf_rew.copy(), self._buf_done.copy(), self._buf_info.copy()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        indices = self._get_indices(indices)
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        indices = self._get_indices(indices)
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        indices = self._get_indices(indices)
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
                for i in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        indices = self._get_indices(indices)
        return [isinstance(self.envs[i], wrapper_class) for i in indices]

    def seed(self, seed=None):
        return [env.reset(seed=seed) for env in self.envs]

    # ── Private ────────────────────────────────────────────────────────────

    def _augment_all_obs(self) -> np.ndarray:
        return np.stack([self._augment_obs(i) for i in range(self.n_sats)])

    def _augment_obs(self, sat_idx: int) -> np.ndarray:
        own_obs = self._last_obs[sat_idx]
        claim_bm = self._claim_reg.bitmap   # (20,)

        # Other satellite summaries (4 features each)
        other_parts = []
        for j in range(self.n_sats):
            if j == sat_idx:
                continue
            other_obs    = self._last_obs[j]
            other_action = self._last_actions[j]
            # Extract a compact summary: (norm_angle, act/n_act, cf, priority)
            # We use obs positions that roughly correspond to:
            #   [0:3]  satellite state, [3:6] orbital, [6:8] target CF/priority (first target)
            # This is heuristic — adjust if obs layout changes
            summary = np.array([
                float(other_obs[0]) if len(other_obs) > 0 else 0.0,   # sat state feat 0
                float(other_action) / N_ACTIONS,                        # normalized action
                float(other_obs[6]) if len(other_obs) > 6 else 0.0,   # first target CF
                float(other_obs[7]) if len(other_obs) > 7 else 0.0,   # first target priority
            ], dtype=np.float32)
            other_parts.append(summary)

        other_arr = np.concatenate(other_parts) if other_parts else np.zeros(0, dtype=np.float32)
        return np.concatenate([own_obs, claim_bm, other_arr]).astype(np.float32)

    def _get_indices(self, indices: Optional[VecEnvIndices]) -> List[int]:
        if indices is None:
            return list(range(self.n_sats))
        if isinstance(indices, int):
            return [indices]
        return list(indices)
