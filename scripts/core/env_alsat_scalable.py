#!/usr/bin/env python3
from __future__ import annotations
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -----------------------------------------------------------------
"""
env_alsat_scalable.py  --  ALSAT-EO-1  Scalable Target Configuration
=====================================================================
Removes the hardcoded 20-target / 3-dynamic-slot limitation.

The base bsk_rl observation always shows the N_AHEAD_OBSERVE (=6) highest-
priority upcoming opportunities regardless of how many targets exist,
so the obs dim stays at 56 for any target set size.  The action space
IS variable and equals n_targets + n_dyn_slots + 1.

This module provides:
  1. Target list utilities   -- load, validate, combine, generate
  2. DynamicSlotConfig       -- dataclass for slot/capacity settings
  3. make_scalable_env()     -- factory accepting arbitrary target list
  4. generate_random_targets()-- synthetic target generator for benchmarking

Supported configurations
------------------------
  20 static + 3 dyn  (default, matching Algeria deployment)
  50 static + 5 dyn  (medium constellation, N. Africa + Middle East)
  100 static + 8 dyn (full constellation scale)
  Variable per JSON  (load any target file)

Note on attention + scalability
--------------------------------
When using >6 targets, the MLP policy never sees beyond the 6 best
ahead-slots.  Switch to SchedulerAttentionExtractor (attention_policy.py)
which reads all slot tokens simultaneously and can theoretically handle
any number once the obs is restructured.  See make_scalable_env() docs.
"""


import json
import math
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)


# =============================================================================
#  DynamicSlotConfig -- centralises all "hardcoded" constants
# =============================================================================

@dataclass
class DynamicSlotConfig:
    """
    Runtime configuration for slot counts and observation layout.

    Parameters
    ----------
    n_dyn_slots   : number of dynamic event slots (default 3, max ~8)
    n_ahead_obs   : bsk_rl OpportunityProperties n_ahead_observe (fixed 6)
    n_target_feats: features per static slot (fixed 5 in current obs_spec)
    n_dyn_feats   : features per dynamic slot (fixed 4)

    Derived attributes (obs layout)
    --------------------------------
    obs_base_dim  = 43  (satellite state + 6×5 opportunity props)
    obs_dyn_dim   = n_dyn_slots * n_dyn_feats
    obs_total_dim = obs_base_dim + obs_dyn_dim + 1 (sojourn)
    n_actions     = n_targets + n_dyn_slots + 1   (variable with target set)
    """
    n_dyn_slots:    int = 3
    n_ahead_obs:    int = 6
    n_target_feats: int = 5
    n_dyn_feats:    int = 4

    @property
    def obs_base_dim(self) -> int:
        return 13 + self.n_ahead_obs * self.n_target_feats  # 13 + 30 = 43

    @property
    def obs_dyn_dim(self) -> int:
        return self.n_dyn_slots * self.n_dyn_feats

    @property
    def obs_total_dim(self) -> int:
        return self.obs_base_dim + self.obs_dyn_dim + 1  # +1 for sojourn

    def n_actions(self, n_targets: int) -> int:
        return n_targets + self.n_dyn_slots + 1

    def validate(self) -> None:
        assert 1 <= self.n_dyn_slots <= 10, "n_dyn_slots must be 1-10"
        assert self.n_ahead_obs == 6, "n_ahead_obs must match bsk_rl obs_spec (6)"


# ── Preset configurations ─────────────────────────────────────────────────────
SLOT_CONFIG_SMALL  = DynamicSlotConfig(n_dyn_slots=3)   # 20 targets + 3 dyn
SLOT_CONFIG_MEDIUM = DynamicSlotConfig(n_dyn_slots=5)   # 50 targets + 5 dyn
SLOT_CONFIG_LARGE  = DynamicSlotConfig(n_dyn_slots=8)   # 100 targets + 8 dyn


# =============================================================================
#  Target list utilities
# =============================================================================

def load_targets(path: str) -> List[Dict[str, Any]]:
    """Load target config from JSON (list or dict of dicts)."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        targets = list(raw.values())
    else:
        targets = list(raw)
    logger.info(f"Loaded {len(targets)} targets from {path}")
    return targets


def validate_targets(targets: List[dict]) -> List[dict]:
    """Ensure each target has required fields; add defaults."""
    cleaned = []
    for i, t in enumerate(targets):
        lat = float(t.get("lat_deg", t.get("lat", 0.0)))
        lon = float(t.get("lon_deg", t.get("lon", 0.0)))
        if not (-90 <= lat <= 90):
            logger.warning(f"Target {i} lat={lat} out of range; skipping.")
            continue
        cleaned.append({
            "name":     str(t.get("name", t.get("id", f"T{i:03d}"))),
            "lat_deg":  lat,
            "lon_deg":  lon,
            "alt_m":    float(t.get("alt_m", 0.0)),
            "priority": float(t.get("priority", 1.0)),
        })
    return cleaned


def combine_target_files(*paths: str) -> List[dict]:
    """Merge multiple target JSON files into one list."""
    combined = []
    seen = set()
    for p in paths:
        for t in load_targets(p):
            key = (round(float(t.get("lat_deg",t.get("lat",0))),4),
                   round(float(t.get("lon_deg",t.get("lon",0))),4))
            if key not in seen:
                combined.append(t)
                seen.add(key)
    return combined


def generate_random_targets(
    n:           int   = 50,
    lat_range:   tuple = (30.0, 37.0),
    lon_range:   tuple = (-8.0, 12.0),
    seed:        int   = 0,
    priority_range: tuple = (0.5, 1.0),
) -> List[dict]:
    """
    Generate synthetic target set for scalability benchmarking.

    Use this to test with 50, 100, or 200 targets before deploying
    a real larger target list.
    """
    rng = np.random.default_rng(seed)
    targets = []
    for i in range(n):
        targets.append({
            "name":     f"synthetic_T{i:04d}",
            "lat_deg":  float(rng.uniform(*lat_range)),
            "lon_deg":  float(rng.uniform(*lon_range)),
            "alt_m":    0.0,
            "priority": float(rng.uniform(*priority_range)),
        })
    logger.info(f"Generated {n} synthetic targets in "
                f"lat={lat_range}, lon={lon_range}")
    return targets


def save_targets(targets: List[dict], path: str) -> None:
    """Save target list to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(targets, f, indent=2)
    logger.info(f"Saved {len(targets)} targets -> {path}")


# =============================================================================
#  Scalable environment factory
# =============================================================================

def make_scalable_env(
    targets_path:     str,
    cloud_json_path:  str,
    slot_config:      DynamicSlotConfig = SLOT_CONFIG_SMALL,
    event_rate:       float = 2.0,
    duration_s:       float = 172800.0,
    seed:             int   = 42,
    with_safety:      bool  = True,
    cloud_model              = None,
    gamma:            float = 0.99,
    render_mode               = None,
) -> gym.Env:
    """
    Build a scalable SMDP dynamic targeting environment.

    This wraps make_dynamic_env() with a configurable n_dyn_slots
    and validates the target list before constructing the env.

    Parameters
    ----------
    targets_path  : path to targets JSON  (any size, not just 20)
    slot_config   : DynamicSlotConfig controlling slot counts
    event_rate    : dynamic events per hour
    with_safety   : attach SafetyMonitor

    Returns
    -------
    DynamicObsWrapper
      obs_dim    = slot_config.obs_total_dim  (variable with n_dyn_slots)
      n_actions  = n_targets + n_dyn_slots + 1

    Notes
    -----
    For n_targets > 6 (n_ahead_observe), the MLP policy still works —
    it only ever sees the 6 best candidates per step — but you gain
    nothing from more targets unless you switch to attention:

        from attention_policy import make_attention_ppo
        model = make_attention_ppo(make_vec_env(..., slot_config=SLOT_CONFIG_LARGE))
    """
    from env_alsat_dynamic import (
        make_dynamic_env, DynamicAlsatSatellite, DynamicObsWrapper,
        SingleSatelliteEnv, DynamicScienceReward, EventGenerator, EventManager,
        N_DYN_SLOTS as _DEFAULT_DYN_SLOTS,
        OBS_TOTAL_DIM as _DEFAULT_OBS_DIM,
    )
    from env_alsat_debug import (
        load_targets_config, ModisCloudModel, AlsatScenario, SCHED_STEP_S, BSK_SIM_RATE_S,
    )
    from bsk_rl.gym import GeneralSatelliteTasking

    slot_config.validate()

    # Validate targets
    raw_targets = load_targets_config(targets_path)
    targets     = validate_targets(raw_targets)
    if not targets:
        raise ValueError(f"No valid targets found in {targets_path}")
    logger.info(f"  Scalable env: {len(targets)} targets  "
                f"n_dyn={slot_config.n_dyn_slots}  "
                f"obs={slot_config.obs_total_dim}  "
                f"acts={slot_config.n_actions(len(targets))}")

    if cloud_model is None:
        cloud_model = ModisCloudModel(cloud_json_path, seed=seed)

    safety_mon = None
    if with_safety:
        try:
            from safety_monitor import SafetyMonitor
            safety_mon = SafetyMonitor()
        except ImportError:
            logger.warning("SafetyMonitor not found; running without.")

    scenario     = AlsatScenario(targets, cloud_model)
    gen_duration = duration_s

    event_gen = EventGenerator(rate_per_hour=event_rate, seed=seed)
    event_mgr = EventManager(n_slots=slot_config.n_dyn_slots)

    satellite = DynamicAlsatSatellite(
        name="ALSAT-1", scenario=scenario,
        event_manager=event_mgr, safety_monitor=safety_mon,
        generation_duration=gen_duration, initial_generation_duration=gen_duration,
    )

    base_env = GeneralSatelliteTasking(
        satellites=[satellite], scenario=scenario,
        rewarder=DynamicScienceReward(reward_scale=1.0),
        time_limit=duration_s, sim_rate=BSK_SIM_RATE_S,
        max_step_duration=SCHED_STEP_S, render_mode=render_mode,
    )

    flat_env = SingleSatelliteEnv(base_env)

    # Override N_DYN_SLOTS in the wrapper by subclassing
    class ScalableDynamicObsWrapper(DynamicObsWrapper):
        """DynamicObsWrapper with configurable n_dyn_slots."""
        _N_DYN_SLOTS = slot_config.n_dyn_slots
        _N_DYN_FEATS = slot_config.n_dyn_feats

        @property
        def _obs_total_dim(self):
            return slot_config.obs_total_dim

        def __init__(self, env, gen, mgr, gamma):
            super().__init__(env, gen, mgr, gamma)
            # Recompute obs/action spaces for new slot counts
            import gymnasium as gym
            import numpy as np
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(slot_config.obs_total_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(
                slot_config.n_actions(len(targets)))

        def _build_obs(self, base_obs, tau_norm):
            from env_alsat_dynamic import _compute_tta, _slew_safe, TIME_NORM_S
            import numpy as np, math
            try:
                sat   = self.env.unwrapped.satellites[0]
                now   = float(sat.simulator.sim_time)
                slots = self._mgr.get_slots(sat, now)
            except Exception:
                slots = [None] * slot_config.n_dyn_slots
                sat = None; now = 0.0

            feats = []
            for evt in slots:
                if evt is None:
                    feats.extend([0.0, 0.0, 1.0, 0.0])
                else:
                    try:
                        slew = _slew_safe(sat, evt)
                        tta  = _compute_tta(sat, evt, now)
                        feats.extend([
                            float(np.clip(evt.priority,             0.0, 1.0)),
                            float(np.clip(evt.cloud_cover_forecast, 0.0, 1.0)),
                            float(np.clip(tta / TIME_NORM_S,        0.0, 1.0)),
                            float(np.clip(slew / (math.pi / 2),     0.0, 1.0)),
                        ])
                    except Exception:
                        feats.extend([0.0, 0.0, 1.0, 0.0])

            dyn_arr     = np.array(feats, dtype=np.float32)
            sojourn_arr = np.array([np.clip(tau_norm, 0.0, 1.0)], dtype=np.float32)
            return np.concatenate([base_obs.astype(np.float32), dyn_arr, sojourn_arr])

    return ScalableDynamicObsWrapper(flat_env, event_gen, event_mgr, gamma)


# =============================================================================
#  Benchmark helper
# =============================================================================

def benchmark_scalability(targets_path: str, cloud_json_path: str,
                           configs: List[tuple] = None,
                           n_steps: int = 10) -> dict:
    """
    Test env creation + stepping for various target set sizes.

    Parameters
    ----------
    configs : list of (n_targets, n_dyn_slots) tuples
    n_steps : steps per config

    Returns
    -------
    dict mapping config label -> {obs_dim, n_actions, step_time_ms}
    """
    import time
    import numpy as np

    if configs is None:
        configs = [(20, 3), (50, 5), (100, 8)]

    results = {}
    base_targets = load_targets(targets_path)

    for n_tgt, n_dyn in configs:
        label = f"{n_tgt}tgt_{n_dyn}dyn"
        print(f"  Testing {label}...", end="", flush=True)

        # Pad or truncate target list
        if len(base_targets) < n_tgt:
            extras = generate_random_targets(n_tgt - len(base_targets), seed=n_tgt)
            targets = base_targets + extras
        else:
            targets = base_targets[:n_tgt]

        # Write temp targets file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(targets, f)
            tmp_path = f.name

        cfg = DynamicSlotConfig(n_dyn_slots=n_dyn)
        try:
            env  = make_scalable_env(tmp_path, cloud_json_path, slot_config=cfg, seed=0)
            obs, _ = env.reset(seed=0)
            t0 = time.perf_counter()
            for _ in range(n_steps):
                env.step(env.action_space.sample())
            elapsed_ms = (time.perf_counter() - t0) / n_steps * 1000

            results[label] = {
                "n_targets":  n_tgt,
                "n_dyn":      n_dyn,
                "obs_dim":    obs.shape[0],
                "n_actions":  env.action_space.n,
                "step_ms":    round(elapsed_ms, 1),
            }
            print(f"  obs={obs.shape[0]}  acts={env.action_space.n}  "
                  f"{elapsed_ms:.0f}ms/step")
            env.close()
        except Exception as e:
            results[label] = {"error": str(e)}
            print(f"  ERROR: {e}")
        finally:
            os.unlink(tmp_path)

    return results


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import path_setup
    ROOT = path_setup.root_path()

    ap = argparse.ArgumentParser(description="Scalable env utilities")
    ap.add_argument("--targets",   default=os.path.join(ROOT,"config/targets/algeria_20_targets.json"))
    ap.add_argument("--cloud",     default=os.path.join(ROOT,"config/cloud_reality/algeria_real_clouds.json"))
    ap.add_argument("--generate",  type=int, default=0,
                    help="Generate N synthetic targets and save to config/targets/synthetic_N.json")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    if args.generate > 0:
        synth = generate_random_targets(args.generate, seed=42)
        out   = os.path.join(ROOT, f"config/targets/synthetic_{args.generate}.json")
        save_targets(synth, out)
        print(f"Generated {args.generate} synthetic targets -> {out}")

    if args.benchmark:
        print("Scalability benchmark (10 steps each):")
        results = benchmark_scalability(args.targets, args.cloud)
        print("\nResults:")
        for k, v in results.items():
            print(f"  {k}: {v}")

    else:
        # Quick test
        print("Scalable env quick test (20 targets, 3 dyn slots):")
        env = make_scalable_env(args.targets, args.cloud,
                                slot_config=SLOT_CONFIG_SMALL, seed=42)
        obs, _ = env.reset(seed=42)
        print(f"  obs.shape={obs.shape}  n_actions={env.action_space.n}")
        obs, r, *_ = env.step(5)
        print(f"  step OK  r={r:+.4f}")
        env.close()
        print("Test passed.")

        print("\nSlot config summary:")
        for name, cfg in [("SMALL", SLOT_CONFIG_SMALL),
                          ("MEDIUM",SLOT_CONFIG_MEDIUM),
                          ("LARGE", SLOT_CONFIG_LARGE)]:
            print(f"  {name:<8} n_dyn={cfg.n_dyn_slots}  "
                  f"obs={cfg.obs_total_dim}  "
                  f"acts(20tgt)={cfg.n_actions(20)}  "
                  f"acts(100tgt)={cfg.n_actions(100)}")
