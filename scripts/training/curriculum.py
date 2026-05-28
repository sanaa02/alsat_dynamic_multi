#!/usr/bin/env python3
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -----------------------------------------------------------------
"""
curriculum.py  --  ALSAT-EO-1  Phase 3  Curriculum Learning
============================================================
4-phase schedule: static_clear -> static_clouds -> dynamic_sparse -> dynamic_dense
Uses env_dynamic_factory.make_env() instead of deprecated smdp wrappers.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)
# Indices of cloud_cover_forecast features in the observation vector (after battery addition: 57 dims)
# Update these if the observation layout changes.
CLOUD_FORECAST_OBS_INDICES = [44, 48, 52]   # slots 0/1/2 cloud forecast features

@dataclass
class CurriculumPhase:
    name:                 str
    event_rate:           float
    clear_sky:            bool
    graduation_threshold: Optional[float]
    min_episodes:         int = 50


PHASES: List[CurriculumPhase] = [
    # Thresholds lowered — old values caused permanent stuck-at-phase-0
    # static_clear mean ~6-7 in practice; 10.0 was unreachable
    CurriculumPhase("static_clear",    0.0, True,  7.0, 50),   # was 10.0
    CurriculumPhase("static_clouds",   0.0, False,  5.0, 75),   # was  7.0
    CurriculumPhase("dynamic_sparse",  0.5, False,  6.5, 100),  # was  9.0
    CurriculumPhase("dynamic_dense",   2.0, False, None, 250),
    CurriculumPhase("balanced_mix", 2.0, False, None, 500)
]


class ClearSkyWrapper(gym.ObservationWrapper):
    """Phase 1: zero out cloud forecast features so the agent learns scheduling first."""
    def observation(self, obs):
        obs = obs.copy()
        for idx in CLOUD_FORECAST_OBS_INDICES:
            if idx < len(obs):
                obs[idx] = 0.0
        return obs


class CurriculumScheduler:
    def __init__(self, phases=None, n_grad_window=30, verbose=True):
        self.phases         = phases or PHASES
        self.n_grad_window  = n_grad_window
        self.verbose        = verbose
        self._phase_idx     = 0
        self._ep_rewards:   List[float] = []
        self._ep_in_phase   = 0
        self._history:      list = []

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self._phase_idx]

    @property
    def is_final_phase(self) -> bool:
        return self._phase_idx == len(self.phases) - 1

    def maybe_advance(self, ep_reward: float) -> bool:
        self._ep_rewards.append(ep_reward)
        self._ep_in_phase += 1
        ph = self.current_phase
        if self.is_final_phase or self._ep_in_phase < ph.min_episodes:
            return False
        mean_r = float(np.mean(self._ep_rewards[-self.n_grad_window:]))
        if ph.graduation_threshold is not None and mean_r >= ph.graduation_threshold:
            self._history.append({"phase": ph.name, "ep": self._ep_in_phase, "r": mean_r})
            self._phase_idx   += 1
            self._ep_in_phase  = 0
            if self.verbose:
                print(f"\n[CURRICULUM] '{ph.name}' -> '{self.current_phase.name}'  "
                      f"(mean_r={mean_r:+.3f})\n")
            return True
        return False

    def make_env(self, targets_path, cloud_json_path, seed=42,
                 duration_s=172800.0, use_smdp=False, cfg=None,
                 with_safety=True) -> gym.Env:
        from env_dynamic_factory import make_env, Config
        ph = self.current_phase
        chosen_cfg = cfg or Config.DYN_MODIS
        import path_setup
        root = path_setup.root_path()
        env = make_env(
            cfg=chosen_cfg,
            targets_path=targets_path, cloud_json_path=cloud_json_path,
            event_rate=ph.event_rate, duration_s=duration_s,
            seed=seed, with_safety=with_safety,
            with_clear_sky=ph.clear_sky,
            cnn_path=os.path.join(root,"models/cloud_cnn_real.pt"),
        )
        return env

    def get_history(self): return list(self._history)

    def summary(self) -> str:
        lines = [f"Curriculum: phase={self.current_phase.name}  idx={self._phase_idx}"]
        for h in self._history:
            lines.append(f"  Graduated '{h['phase']}' ep={h['ep']} r={h['r']:+.3f}")
        return "\n".join(lines)
