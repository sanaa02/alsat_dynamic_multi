#!/usr/bin/env python3
"""
baselines_dynamic.py  —  ALSAT-EO-1  Phase 3  Dynamic Targeting Baselines
==========================================================================
Two new baseline policies that operate in the dynamic targeting environment:

  greedy_dynamic_scout
      At each step picks the highest-value target (static OR dynamic),
      using CNN forecast for clouds.  Value function:
        static  : priority * (1 - forecast_cloud)
        dynamic : priority * (1 - forecast_cloud) + DYNAMIC_BONUS
      This is the PRIMARY baseline for dynamic-targeting evaluation.

  greedy_ignore_dynamic
      Same as greedy_scout from baselines.py — uses CNN cloud forecast
      but ignores all dynamic events.  Represents a system with NO
      dynamic targeting capability.  The difference in total reward
      between this and greedy_dynamic_scout quantifies the value of
      dynamic targeting.

Usage
-----
    from baselines_dynamic import run_all_dynamic_baselines, print_dynamic_table
    results = run_all_dynamic_baselines(targets_path, cloud_json_path,
                                        event_rate=2.0, n_episodes=5)
    print_dynamic_table(results)
"""

# ---- ALSAT path-setup --------------------------------------------
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------


import os, sys, math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scripts.core.env_alsat_dynamic import (
    make_dynamic_env,
    DynamicObsWrapper,
    N_STATIC_TARGETS,
    N_DYN_SLOTS,
    N_TOTAL_ACTIONS,
)
from scripts.core.env_alsat_debug import CLOUD_THRESH, SCHED_STEP_S, SIM_DURATION_S
from scripts.core.dynamic_event import DYNAMIC_BONUS, MAX_OFFNADIR_RAD

import gymnasium as gym


# ============================================================================
#  Result dataclass
# ============================================================================

@dataclass
class DynamicEpisodeResult:
    policy_name:      str
    event_rate:       float = 0.0
    total_reward:     float = 0.0
    n_images:         int   = 0
    n_cloud_free:     int   = 0
    n_cloudy:         int   = 0
    n_dyn_detected:   int   = 0
    n_dyn_imaged:     int   = 0
    n_steps:          int   = 0
    total_slew_deg:   float = 0.0
    cloud_free_rate:  float = 0.0
    dyn_success_rate: float = 0.0
    avg_delay_s:      float = 0.0
    dyn_reward_share: float = 0.0  # fraction of total reward from dynamic events

    def finalise(self, dyn_metrics: dict) -> "DynamicEpisodeResult":
        self.cloud_free_rate  = (self.n_cloud_free / self.n_images
                                 if self.n_images > 0 else 0.0)
        self.dyn_success_rate = dyn_metrics.get("success_rate", 0.0)
        self.avg_delay_s      = dyn_metrics.get("avg_delay_s",  0.0)
        self.n_dyn_detected   = dyn_metrics.get("n_detected",   0)
        self.n_dyn_imaged     = dyn_metrics.get("n_imaged",     0)
        return self


def _aggregate_dynamic(results: List[DynamicEpisodeResult]) -> dict:
    return {
        "mean_reward":     float(np.mean([r.total_reward    for r in results])),
        "std_reward":      float(np.std( [r.total_reward    for r in results])),
        "mean_cf_rate":    float(np.mean([r.cloud_free_rate for r in results])),
        "mean_dyn_success":float(np.mean([r.dyn_success_rate for r in results])),
        "mean_delay_s":    float(np.mean([r.avg_delay_s     for r in results])),
        "mean_dyn_detected":float(np.mean([r.n_dyn_detected for r in results])),
        "mean_dyn_imaged": float(np.mean([r.n_dyn_imaged    for r in results])),
        "episodes": [
            {"total_reward": r.total_reward, "n_images": r.n_images,
             "cloud_free_rate": r.cloud_free_rate,
             "dyn_success_rate": r.dyn_success_rate,
             "n_dyn_imaged": r.n_dyn_imaged}
            for r in results
        ],
    }


# ============================================================================
#  Helpers
# ============================================================================

def _get_sat(env: DynamicObsWrapper):
    return env.env.unwrapped.satellites[0]


def _is_accessible_static(satellite, target_idx: int, now: float) -> bool:
    try:
        target = satellite.scenario.targets[target_idx]
        for opp in satellite.upcoming_opportunities:
            if opp["object"] is target and opp["type"] == "target":
                t0, t1 = opp["window"]
                if t0 <= now <= t1:
                    return True
    except Exception:
        pass
    return False


def _is_accessible_dynamic(satellite, event, now: float) -> bool:
    """Dynamic event accessible if current slew angle <= 45°."""
    try:
        from scripts.core.env_alsat_debug import calculate_slew_angle_to_target
        slew = calculate_slew_angle_to_target(satellite, event)
        return slew <= MAX_OFFNADIR_RAD
    except Exception:
        return False


# ============================================================================
#  Policy 1: Greedy dynamic scout  (MAIN dynamic baseline)
# ============================================================================

def run_greedy_dynamic_episode(
    targets_path:    str,
    cloud_json_path: str,
    event_rate:      float = 2.0,
    duration_s:      float = SIM_DURATION_S,
    seed:            int   = 0,
) -> DynamicEpisodeResult:
    """
    Greedy policy that considers BOTH static targets and dynamic events.
    Uses CNN forecast for clouds.  Dynamic events have higher value due to DYNAMIC_BONUS.

    Value:
      static  : priority * (1 - cloud_forecast)
      dynamic : priority * (1 - cloud_forecast) + DYNAMIC_BONUS
    """
    env    = make_dynamic_env(targets_path, cloud_json_path,
                               event_rate=event_rate, duration_s=duration_s, seed=seed)
    obs, _ = env.reset(seed=seed)
    result = DynamicEpisodeResult(policy_name="greedy_dynamic_scout", event_rate=event_rate)
    sat    = _get_sat(env)
    done   = False
    last_dyn_metrics: dict = {}

    while not done:
        now      = float(sat.simulator.sim_time)
        n_static = len(sat.scenario.targets)
        best_act = N_TOTAL_ACTIONS - 1   # default: drift
        best_val = -1.0

        # ── Evaluate static targets ────────────────────────────────────────
        for tid, target in enumerate(sat.scenario.targets):
            if not _is_accessible_static(sat, tid, now):
                continue
            fc = float(target.cloud_cover_forecast)
            if fc < CLOUD_THRESH:
                val = float(target.priority) * (1.0 - fc)
                if val > best_val:
                    best_val = val
                    best_act = tid

        # ── Evaluate dynamic events ─────────────────────────────────────────
        slots = env.event_manager.get_slots(sat, now)
        for slot_idx, event in enumerate(slots):
            if event is None:
                continue
            action_idx = n_static + slot_idx
            if not _is_accessible_dynamic(sat, event, now):
                continue
            fc  = float(event.cloud_cover_forecast)
            val = float(event.priority) * (1.0 - fc) + DYNAMIC_BONUS
            if val > best_val:
                best_val = val
                best_act = action_idx

        obs, r, term, trunc, info = env.step(best_act)
        result.total_reward += r
        result.n_steps      += 1
        last_dyn_metrics     = info.get("dynamic_metrics", {})

        metrics = sat.get_metrics()
        result.n_images    = metrics["n_imaged"]
        result.n_cloud_free = metrics["n_cloud_free"]
        result.n_cloudy    = metrics["n_cloudy"]

        done = term or trunc

    env.close()
    return result.finalise(last_dyn_metrics)


# ============================================================================
#  Policy 2: Greedy ignore dynamic  (no dynamic capability)
# ============================================================================

def run_ignore_dynamic_episode(
    targets_path:    str,
    cloud_json_path: str,
    event_rate:      float = 2.0,
    duration_s:      float = SIM_DURATION_S,
    seed:            int   = 0,
) -> DynamicEpisodeResult:
    """
    Uses only CNN forecast for static targets; never images dynamic events.
    Represents a legacy system with no dynamic targeting.
    """
    env    = make_dynamic_env(targets_path, cloud_json_path,
                               event_rate=event_rate, duration_s=duration_s, seed=seed)
    obs, _ = env.reset(seed=seed)
    result = DynamicEpisodeResult(policy_name="greedy_ignore_dynamic", event_rate=event_rate)
    sat    = _get_sat(env)
    done   = False
    last_dyn_metrics: dict = {}

    while not done:
        now      = float(sat.simulator.sim_time)
        n_static = len(sat.scenario.targets)
        best_act = N_TOTAL_ACTIONS - 1   # drift
        best_val = -1.0

        # Only consider static targets
        for tid, target in enumerate(sat.scenario.targets):
            if not _is_accessible_static(sat, tid, now):
                continue
            fc = float(target.cloud_cover_forecast)
            if fc < CLOUD_THRESH:
                val = float(target.priority) * (1.0 - fc)
                if val > best_val:
                    best_val = val
                    best_act = tid

        obs, r, term, trunc, info = env.step(best_act)
        result.total_reward += r
        result.n_steps      += 1
        last_dyn_metrics     = info.get("dynamic_metrics", {})

        metrics = sat.get_metrics()
        result.n_images    = metrics["n_imaged"]
        result.n_cloud_free = metrics["n_cloud_free"]
        result.n_cloudy    = metrics["n_cloudy"]
        done = term or trunc

    env.close()
    return result.finalise(last_dyn_metrics)


# ============================================================================
#  Runner and reporting
# ============================================================================

def run_all_dynamic_baselines(
    targets_path:    str,
    cloud_json_path: str,
    event_rate:      float = 2.0,
    duration_s:      float = SIM_DURATION_S,
    n_episodes:      int   = 5,
    seed:            int   = 0,
    verbose:         bool  = True,
) -> Dict[str, dict]:
    """
    Run both dynamic baselines for n_episodes each.
    Returns dict suitable for print_dynamic_table().
    """
    policies = [
        ("greedy_dynamic_scout",  run_greedy_dynamic_episode),
        ("greedy_ignore_dynamic", run_ignore_dynamic_episode),
    ]
    results: Dict[str, dict] = {}

    for name, runner in policies:
        if verbose:
            print(f"  Running baseline: {name} ({n_episodes} episodes, rate={event_rate}/hr)...")
        eps = []
        for ep in range(n_episodes):
            r = runner(targets_path, cloud_json_path,
                       event_rate=event_rate, duration_s=duration_s, seed=seed + ep)
            eps.append(r)
            if verbose:
                print(f"    ep {ep+1:2d}  reward={r.total_reward:+.3f}  "
                      f"imgs={r.n_images:2d}  cf={r.cloud_free_rate:.0%}  "
                      f"dyn_det={r.n_dyn_detected}  dyn_img={r.n_dyn_imaged}  "
                      f"dyn_suc={r.dyn_success_rate:.0%}")
        results[name] = _aggregate_dynamic(eps)

    return results


def print_dynamic_table(baseline_results: Dict[str, dict],
                        rl_stats: Optional[dict] = None,
                        event_rate: float = 2.0) -> None:
    """Print Table 2: dynamic targeting comparison."""
    print()
    print("=" * 80)
    print(f"  DYNAMIC TARGETING COMPARISON  (event_rate={event_rate}/hr)")
    print("=" * 80)
    header = (f"  {'Policy':<26}  {'Reward':>8}  {'±':>5}  "
              f"{'CF%':>5}  {'DynDet':>7}  {'DynImg':>7}  "
              f"{'Suc%':>5}  {'Delay(s)':>9}")
    print(header)
    print("  " + "-" * 78)

    all_rows = {}
    all_rows.update(baseline_results)
    if rl_stats:
        all_rows["RL-PPO (dynamic)"] = rl_stats

    for name, stats in all_rows.items():
        row = (f"  {name:<26}  "
               f"{stats['mean_reward']:>+8.3f}  "
               f"{stats['std_reward']:>5.3f}  "
               f"{stats['mean_cf_rate']:>5.0%}  "
               f"{stats.get('mean_dyn_detected',0):>7.1f}  "
               f"{stats.get('mean_dyn_imaged',0):>7.1f}  "
               f"{stats.get('mean_dyn_success',0):>5.0%}  "
               f"{stats.get('mean_delay_s',0):>9.0f}")
        print(row)
    print("=" * 80)

    # Dynamic targeting value
    if ("greedy_dynamic_scout" in baseline_results and
            "greedy_ignore_dynamic" in baseline_results):
        gap = (baseline_results["greedy_dynamic_scout"]["mean_reward"] -
               baseline_results["greedy_ignore_dynamic"]["mean_reward"])
        print(f"  Dynamic targeting value (greedy_dynamic - ignore_dynamic): {gap:+.3f}")
    print()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TARGETS    = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
    CLOUD_JSON = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")

    print("baselines_dynamic.py — quick test (1 episode each)\n")
    results = run_all_dynamic_baselines(
        TARGETS, CLOUD_JSON, event_rate=2.0, n_episodes=1, seed=42, verbose=True
    )
    print_dynamic_table(results, event_rate=2.0)
