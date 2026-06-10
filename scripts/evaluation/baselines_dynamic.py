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


# ============================================================
# ADD: EDF + Greedy composite baseline
# ============================================================

def edf_greedy_baseline(
    env,
    n_episodes: int = 10,
    seed: int = 100,
) -> dict:
    """Earliest-Deadline-First (EDF) for dynamic events + greedy-priority static.

    Strategy:
    - At each decision step, inspect observation.
    - If any DYN event slot is active (non-zero priority), select the one
      with the smallest remaining time (EDF policy).
    - Otherwise, select the highest-priority static target with the
      smallest TTA (cloud-aware greedy).
    - Drift if nothing feasible is visible.

    This is a stronger baseline than greedy_dynamic_scout because:
    1. It uses EDF scheduling theory for deadline-constrained tasks.
    2. It cloud-weights static targets.
    3. It accounts for TTA instead of random ordering.
    """
    obs_dim = env.observation_space.shape[0]

    # Observation layout constants (must match env_alsat_dynamic.py)
    N_STATE     = 13
    N_TF        = 5   # features per static target slot
    N_SLOTS_ST  = 6
    N_DF        = 4   # features per DYN slot
    N_SLOTS_DYN = 3
    # DYN slot offsets: [priority, cloud, TTA, remaining_time_frac]
    IDX_DYN_START = N_STATE + N_TF * N_SLOTS_ST  # 13 + 30 = 43
    IDX_STATIC_START = N_STATE                   # 13

    results = []
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1, 10000)))
        done, total_r, step = False, 0.0, 0

        while not done:
            # --- Parse DYN slots ---
            dyn_scores = []
            for s in range(N_SLOTS_DYN):
                base = IDX_DYN_START + s * N_DF
                if base + N_DF <= len(obs):
                    prio, cloud, tta, rem_frac = obs[base:base+N_DF]
                    if prio > 0.01:  # slot occupied
                        # EDF score: higher priority / lower remaining time → higher priority
                        edf_score = prio * (1.0 - float(cloud)) / max(0.01, float(rem_frac))
                        dyn_scores.append((edf_score, 20 + s))  # action = 20,21,22

            if dyn_scores:
                # Take highest EDF-score DYN action
                action = max(dyn_scores, key=lambda x: x[0])[1]
            else:
                # No DYN events: greedy static (priority × cloud × 1/TTA)
                static_scores = []
                for s in range(N_SLOTS_ST):
                    base = IDX_STATIC_START + s * N_TF
                    if base + N_TF <= len(obs):
                        prio, cloud, tta, slew, _ = obs[base:base+N_TF]
                        if tta > 0.0 and prio > 0.01:
                            score = prio * max(0.0, 1.0 - float(cloud)) / max(0.001, float(tta))
                            static_scores.append((score, s))
                if static_scores:
                    action = max(static_scores, key=lambda x: x[0])[1]
                else:
                    action = 23  # drift

            obs, r, term, trunc, info = env.step(action)
            total_r += r
            done = term or trunc
            step += 1

        metrics = info.get("episode_metrics", {})
        results.append({
            "episode_reward":  total_r,
            "n_dyn_imaged":    metrics.get("n_dyn_imaged", 0),
            "n_static_imaged": metrics.get("n_cloud_free", 0),
            "n_steps":         step,
        })

    # Summary
    rew = np.array([r["episode_reward"] for r in results])
    dyn = np.array([r["n_dyn_imaged"]   for r in results])
    print(f"\n[EDF+Greedy Baseline] n={n_episodes}")
    print(f"  Reward:       {rew.mean():.3f} ± {rew.std():.3f}")
    print(f"  DYN imaged:   {dyn.mean():.1f} ± {dyn.std():.1f}")
    return {"reward": rew.tolist(), "n_dyn": dyn.tolist()}


# ============================================================
# ADD: Greedy oracle for static scheduling (upper bound)
# ============================================================

def greedy_static_oracle(
    targets_with_windows: list,
    cloud_coverage: dict,
    episode_duration_s: float = 172800.0,
) -> dict:
    """Greedy oracle for static target scheduling with perfect cloud knowledge.

    Solves a simplified scheduling problem: given all access windows for
    static targets and ground-truth cloud cover, greedily select the
    highest-value non-overlapping set.

    This provides an approximate UPPER BOUND on static scheduling
    performance (true optimal requires ILP).

    Returns
    -------
    dict with keys: selected_targets, total_reward, n_imaged
    """
    # Sort windows by (priority × clear-sky fraction) descending
    candidates = []
    for t in targets_with_windows:
        for window in t.get("windows", []):
            cloud  = float(cloud_coverage.get(t["id"], 0.5))
            value  = float(t["priority"]) * max(0.0, 1.0 - cloud)
            start  = float(window["start_s"])
            end    = float(window["end_s"])
            if value > 0.0:
                candidates.append({
                    "id": t["id"], "priority": t["priority"],
                    "cloud": cloud, "value": value,
                    "start": start, "end": end,
                })

    candidates.sort(key=lambda c: -c["value"])

    # Greedy interval scheduling
    selected   = []
    last_end   = 0.0
    total_r    = 0.0
    for c in candidates:
        if c["start"] >= last_end:  # no overlap
            selected.append(c)
            total_r += c["value"]
            last_end = c["end"]

    return {
        "selected_targets": selected,
        "total_reward":     total_r,
        "n_imaged":         len(selected),
    }


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
