#!/usr/bin/env python3
"""
reward_shaping.py  --  Dynamic Event Reward Shaping
====================================================
DynamicRewardShaper   gym.Wrapper that adds urgency + exploration bonuses
                      to step rewards, directly addressing dyn_suc=0%.

Design (SOTA-aligned)
---------------------
  1. Urgency bonus: extra reward ∝ (1 - elapsed/lifetime) when a dynamic
     event is imaged. Events near expiry pay more. (Pinedo 2016, deadline scheduling)
  2. Exploration bonus: small flat bonus for ANY dynamic action, decaying
     per episode via explore_decay. Forces early exploration of slots 20-22.
  3. No bonus leakage: bonuses are logged separately so ablation can isolate them.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import logging

logger = logging.getLogger(__name__)

N_STATIC  = 20
N_DYN     = 3
DRIFT     = 23


class DynamicRewardShaper(gym.Wrapper):
    """
    Drop-in wrapper around any DynamicObsWrapper env.

    Parameters
    ----------
    urgency_scale      : multiplier for imaging-time urgency bonus (default 3.0)
    urgency_max        : cap for urgency bonus (default 4.0)
    explore_bonus_init : flat bonus per dynamic step at episode 0 (default 0.3)
    explore_decay      : per-episode decay of explore_bonus (default 0.99)
    explore_min        : floor for explore_bonus (default 0.0)
    """

    def __init__(
        self,
        env,
        urgency_scale:      float = 3.0,
        urgency_max:        float = 4.0,
        explore_bonus_init: float = 0.3,
        explore_decay:      float = 0.99,
        explore_min:        float = 0.0,
    ):
        super().__init__(env)
        self.urgency_scale      = urgency_scale
        self.urgency_max        = urgency_max
        self._explore_bonus     = explore_bonus_init
        self.explore_decay      = explore_decay
        self.explore_min        = explore_min

        # Stats for debugging
        self._total_urgency_given  = 0.0
        self._total_explore_given  = 0.0
        self._n_dyn_steps          = 0

    def reset(self, **kw):
        result = self.env.reset(**kw)
        # Decay the exploration bonus each episode
        self._explore_bonus = max(
            self.explore_min,
            self._explore_bonus * self.explore_decay,
        )
        return result

    def step(self, action: int):
        obs, reward, term, trunc, info = self.env.step(action)
        bonus = self._bonus(int(action), info)
        info["shaping_bonus"] = bonus
        return obs, reward + bonus, term, trunc, info

    def _bonus(self, action: int, info: dict) -> float:
        is_dyn = N_STATIC <= action < N_STATIC + N_DYN
        if not is_dyn:
            return 0.0

        self._n_dyn_steps += 1
        total = 0.0

        # ── 1. Exploration bonus (always, for any dynamic action) ─────────
        total += self._explore_bonus
        self._total_explore_given += self._explore_bonus

        # ── 2. Urgency bonus (only on successful imaging) ─────────────────
        urgency = info.get("event_urgency", None)
        if urgency is None:
            # Derive from dynamic_metrics if available
            dm = info.get("dynamic_metrics", {})
            urgency = dm.get("urgency", None)
        
        imaging_happened = info.get("dynamic_imaging_occurred", False)
        if urgency is not None and imaging_happened:
            u_bonus = float(np.clip(self.urgency_scale * urgency, 0.0, self.urgency_max))
            total += u_bonus
            self._total_urgency_given += u_bonus

        return float(total)

    @property
    def shaping_stats(self) -> dict:
        return {
            "total_urgency_bonus":  self._total_urgency_given,
            "total_explore_bonus":  self._total_explore_given,
            "n_dyn_steps":          self._n_dyn_steps,
            "current_explore_bonus": self._explore_bonus,
        }
