#!/usr/bin/env python3
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'
))
import path_setup  # noqa
# -----------------------------------------------------------------
"""
env_dynamic_factory.py  --  ALSAT-EO-1  Phase 3  Unified Environment Factory
=============================================================================
Simplified after the env_alsat_dynamic.py rewrite:
  - SMDP is now BUILT INTO DynamicObsWrapper  (obs always 56)
  - smdp_dynamic.py is deprecated
  - Config enum now has 4 variants

Config.BASE_MODIS      -- Phase 2 env  obs=43  actions=21  (no dynamic events)
Config.DYN_MODIS       -- Phase 3 SMDP obs=56  actions=24  Gaussian noise cloud
Config.DYN_VISION      -- Phase 3 SMDP obs=56  actions=24  CNN on synthetic patch
Config.DYN_REAL_VISION -- Phase 3 SMDP obs=56  actions=24  CNN on REAL MODIS patch
                          (recommended — breaks circular dependency)

The DYN_REAL_VISION config requires:
  - patches_dir  : directory of real .npy patches  (data/modis_patches/)
  - cnn_path     : path to trained weights          (models/cloud_cnn_real.pt)
"""
import os, logging
from enum import Enum
import gymnasium as gym
import numpy as np

from env_alsat_dynamic import (
    make_dynamic_env, OBS_TOTAL_DIM, N_TOTAL_ACTIONS,
)
from env_alsat_debug import SIM_DURATION_S, BSK_SIM_RATE_S

logger = logging.getLogger(__name__)

import torch

class Config(str, Enum):
    BASE_MODIS      = "base_modis"       # Phase 2 env: obs=43, actions=21
    DYN_MODIS       = "dyn_modis"        # Phase 3 SMDP: obs=56, Gaussian noise
    DYN_VISION      = "dyn_vision"       # Phase 3 SMDP: obs=56, CNN on synthetic patch
    DYN_REAL_VISION = "dyn_real_vision"  # Phase 3 SMDP: obs=56, CNN on real MODIS patch


def obs_dim(cfg: Config) -> int:
    if cfg == Config.BASE_MODIS:
        return 43
    else:
        from env_alsat_dynamic import OBS_TOTAL_DIM
        return OBS_TOTAL_DIM   # now returns 57


def n_actions(cfg: Config) -> int:
    return 21 if cfg == Config.BASE_MODIS else N_TOTAL_ACTIONS  # 24


def _make_cloud_model(
    cloud_json_path: str,
    cfg:             Config,
    cnn_path:        str,
    patches_dir:     str,
    seed:            int,
):
    """
    Instantiate the appropriate cloud model for the given Config.

    DYN_REAL_VISION  → RealVisionCloudModel  (real patches, no label leakage)
    DYN_VISION       → VisionCloudModel      (synthetic patches, circular — kept for ablation)
    DYN_MODIS        → ModisCloudModel       (Gaussian noise)
    """

    # ── Option A: Real MODIS patches (non-circular) ──────────────────────
    if cfg == Config.DYN_REAL_VISION:
        try:
            from real_vision_cloud_model import RealVisionCloudModel
            m = RealVisionCloudModel(
                cloud_json_path = cloud_json_path,
                patches_dir     = patches_dir,
                cnn_path        = cnn_path,
                seed            = seed,
            )
            logger.info(
                f"RealVisionCloudModel loaded (mode={m.mode}, "
                f"patches={m._provider.n_patches})"
            )
            return m
        except Exception as exc:
            logger.warning(
                f"RealVisionCloudModel unavailable ({exc}). "
                "Falling back to Gaussian noise."
            )

    # ── Option B: Synthetic patches + CNN (original, circular — ablation only) ─
    elif cfg == Config.DYN_VISION:
        try:
            from cloud_cnn import VisionCloudModel
            m = VisionCloudModel(cloud_json_path, cnn_path=cnn_path)
            logger.info(f"VisionCloudModel loaded (mode={m._predictor.mode})")
            return m
        except Exception as exc:
            logger.warning(
                f"VisionCloudModel unavailable ({exc}). "
                "Falling back to Gaussian noise."
            )

    # ── Option C: Gaussian noise (Phase 2 baseline) ──────────────────────
    from env_alsat_debug import ModisCloudModel
    logger.info("Using ModisCloudModel (Gaussian noise fallback).")
    return ModisCloudModel(cloud_json_path, seed=seed)


def make_env(
    cfg:             Config = Config.DYN_REAL_VISION,
    targets_path:    str    = None,
    cloud_json_path: str    = None,
    patches_dir:     str    = None,
    event_rate:      float  = 2.0,
    duration_s:      float  = SIM_DURATION_S,
    seed:            int    = 42,
    with_safety:     bool   = True,
    with_clear_sky:  bool   = False,
    sat_args:        dict   = None,
    cnn_path:        str    = None,
    gamma:           float  = 0.99,
    render_mode             = None,
    with_action_mask=False,
    with_domain_rand=False, 
    domain_rand_seed=None
) -> gym.Env:
    """
    Build and return a configured ALSAT-EO-1 Gymnasium environment.

    Parameters
    ----------
    cfg : Config
        Which environment variant to create.
    patches_dir : str | None
        Path to the directory of real MODIS .npy patches.
        Only required when cfg == Config.DYN_REAL_VISION.
        Defaults to <project_root>/data/modis_patches.
    cnn_path : str | None
        Path to the trained CNN weights.
        Defaults to <project_root>/models/cloud_cnn_real.pt for DYN_REAL_VISION,
        or <project_root>/models/cloud_cnn.pt for DYN_VISION.
    """
    import path_setup
    root = path_setup.root_path()

    # Resolve paths
    if targets_path    is None:
        targets_path    = os.path.join(root, "config/targets/algeria_20_targets.json")
    if cloud_json_path is None:
        cloud_json_path = os.path.join(root, "config/cloud_reality/algeria_real_clouds.json")
    if patches_dir     is None:
        patches_dir     = os.path.join(root, "data/modis_patches")
    if cnn_path        is None:
        # Use the real-data-trained model for DYN_REAL_VISION; synthetic model otherwise
        if cfg == Config.DYN_REAL_VISION:
            cnn_path = os.path.join(root, "models/cloud_cnn_real.pt")
            # Fallback to old name if new name absent
            if not os.path.exists(cnn_path):
                _alt = os.path.join(root, "models/cloud_cnn_real.pt")
                if os.path.exists(_alt):
                    logger.warning(
                        f"cloud_cnn_real.pt not found; using {_alt} instead."
                    )
                    cnn_path = _alt
        else:
            cnn_path = os.path.join(root, "models/cloud_cnn_real.pt")

    # ── BASE_MODIS: phase-2 static environment ─────────────────────────
    if cfg == Config.BASE_MODIS:
        from env_alsat_debug import make_env as _base
        from env_alsat_dynamic import SingleSatelliteEnv
        env = SingleSatelliteEnv(
            _base(targets_path, cloud_json_path,
                  duration_s=duration_s, seed=seed, sat_args=sat_args)
        )

    # ── Phase-3 dynamic environments ───────────────────────────────────
    else:
        cloud_model = _make_cloud_model(
            cloud_json_path = cloud_json_path,
            cfg             = cfg,
            cnn_path        = cnn_path,
            patches_dir     = patches_dir,
            seed            = seed,
        )

        safety_mon = None
        if with_safety:
            try:
                from safety_monitor import SafetyMonitor
                safety_mon = SafetyMonitor()
                logger.info("SafetyMonitor attached to satellite.")
            except ImportError:
                logger.warning(
                    "safety_monitor.py not found — running without safety monitor."
                )

        env = make_dynamic_env(
            targets_path=targets_path, cloud_json_path=cloud_json_path,
            event_rate=event_rate, duration_s=duration_s, seed=seed,
            cloud_model=cloud_model, safety_monitor=safety_mon,
            gamma=gamma, sat_args=sat_args, render_mode=render_mode,
        )

    if with_clear_sky:
        from curriculum import ClearSkyWrapper
        env = ClearSkyWrapper(env)

    logger.debug(
        f"make_env: cfg={cfg.value}  obs={env.observation_space.shape}  "
        f"acts={env.action_space.n}  safety={with_safety}"
    )

    if with_domain_rand:
        try:
            from wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
            env = DomainRandomizationWrapper(env, seed=domain_rand_seed)
        except ImportError: pass
    if with_action_mask:
        try:
            from wrappers.action_mask_wrapper import make_masked_env
            env = make_masked_env(env)
        except ImportError: pass
    return env


def make_vec_env(cfg=Config.DYN_REAL_VISION, n_envs=1, use_subproc=False, **kwargs):
    """Vectorised environment factory. use_subproc=True enables SubprocVecEnv for ~4x speedup."""
    from stable_baselines3.common.monitor import Monitor
    seed = kwargs.get("seed", 42)
    env_kwargs = {k: v for k, v in kwargs.items() if k != "seed"}

    def _make(s):
        def _fn():
            return Monitor(make_env(cfg, seed=s, **env_kwargs))
        return _fn

    fns = [_make(seed + i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        return SubprocVecEnv(fns)
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        return DummyVecEnv(fns)


if __name__ == "__main__":
    for c in Config:
        print(f"  {c.value:<18}  obs={obs_dim(c)}  acts={n_actions(c)}")
