#!/usr/bin/env python3
"""
eval_dynamic.py  —  ALSAT-EO-1  Phase 3  Dynamic Targeting Evaluation
======================================================================
Implements the three evaluation scenarios from specification Section 4:

  Scenario 1 — No events    (event_rate = 0.0 /hr)
      Baseline: should match Phase 2 performance exactly.

  Scenario 2 — Sparse events (event_rate = 0.5 /hr)
      ~1-2 events per 4-hour orbit pass; modest priority.
      Agent must decide whether to divert from static targets.

  Scenario 3 — Dense events  (event_rate = 2.0 /hr)
      Frequent events (~96 per 48h); agent must constantly
      interleave dynamic events with static scheduled targets.

Metrics (spec §2.6):
  n_dyn_detected     — number of dynamic events that appeared
  n_dyn_imaged       — number successfully imaged
  dyn_success_rate   — n_imaged / n_detected
  avg_delay_s        — mean time from appearance to imaging
  total_reward       — including dynamic bonuses
  cloud_free_rate    — for static images
  additional_science — reward delta vs no-events baseline

Usage
-----
    # Evaluate baselines only (fast, no trained model needed)
    python scripts/eval_dynamic.py --no-rl

    # Evaluate baselines + RL agent
    python scripts/eval_dynamic.py --model models/ppo_dynamic_final.zip

    # Quick single episode per scenario
    python scripts/eval_dynamic.py --episodes 1 --no-rl
"""
from __future__ import annotations
# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------

import argparse, json, os, sys, time
import logging
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

RESULTS_DIR = os.path.join(_ROOT, "results")
PLOTS_DIR   = os.path.join(_ROOT, "data/outputs/plots")

# Suppress bsk_rl noise
import logging as _lg
_SKIP = frozenset(["Creating logger for new env",
                   "Old environments in process",
                   "basePowerDraw should probably be zero or negative"])
_orig = _lg.Logger.callHandlers
def _quiet(self, r):
    try:
        if any(s in r.getMessage() for s in _SKIP): return
    except Exception: pass
    _orig(self, r)
_lg.Logger.callHandlers = _quiet


from scripts.core.env_alsat_dynamic import (
    make_dynamic_env, N_TOTAL_ACTIONS, OBS_TOTAL_DIM,
)
from scripts.core.env_alsat_debug import SCHED_STEP_S, SIM_DURATION_S, CLOUD_THRESH
from scripts.evaluation.baselines_dynamic import (
    run_greedy_dynamic_episode,
    run_ignore_dynamic_episode,
    print_dynamic_table,
    _aggregate_dynamic,
    DynamicEpisodeResult,
)
from scripts.core.dynamic_event import DYNAMIC_BONUS


# ── Scenario definitions ─────────────────────────────────────────────────────

SCENARIOS = [
    dict(name="no_events",    event_rate=0.0,  label="No events (baseline)"),
    dict(name="sparse_events",event_rate=0.5,  label="Sparse events (0.5/hr)"),
    dict(name="dense_events", event_rate=2.0,  label="Dense events (2.0/hr)"),
]


# ============================================================================
#  RL agent evaluation helper
# ============================================================================

def _eval_rl_episode(model, targets_path, cloud_json_path,
                     event_rate, duration_s, seed) -> DynamicEpisodeResult:
    """Run one episode with the trained RL model; return DynamicEpisodeResult."""
    env    = make_dynamic_env(targets_path, cloud_json_path,
                               event_rate=event_rate, duration_s=duration_s, seed=seed)
    obs, _ = env.reset(seed=seed)
    result = DynamicEpisodeResult(policy_name="RL-PPO-dynamic", event_rate=event_rate)
    last_dyn: dict = {}

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(action))
        result.total_reward += r
        result.n_steps      += 1
        last_dyn = info.get("dynamic_metrics", {})
        done = term or trunc

    sat = env.env.unwrapped.satellites[0]
    m   = sat.get_metrics()
    result.n_images     = m["n_imaged"]
    result.n_cloud_free = m["n_cloud_free"]
    result.n_cloudy     = m["n_cloudy"]
    env.close()
    return result.finalise(last_dyn)


# ============================================================================
#  Main evaluation loop
# ============================================================================

def evaluate_all_scenarios(
    targets_path:    str,
    cloud_json_path: str,
    n_episodes:      int   = 5,
    seed:            int   = 100,
    duration_s:      float = SIM_DURATION_S,
    model_path:      Optional[str] = None,
    verbose:         bool  = True,
) -> dict:
    """
    Run all 3 scenarios × all policies × n_episodes.

    Returns
    -------
    dict  keyed by scenario name, each value is
          {"greedy_dynamic": {...}, "greedy_ignore": {...}, "rl": {...}}
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    # Optionally load RL model
    model = None
    if model_path and os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            if verbose:
                print(f"  [RL] Loaded model from {model_path}")
        except Exception as exc:
            print(f"  [WARN] Could not load RL model: {exc}")

    scenario_results: dict = {}

    for sc in SCENARIOS:
        name       = sc["name"]
        rate       = sc["event_rate"]
        label      = sc["label"]
        if verbose:
            print()
            print("─" * 70)
            print(f"  SCENARIO: {label}")
            print("─" * 70)

        eps_dynamic: List[DynamicEpisodeResult] = []
        eps_ignore:  List[DynamicEpisodeResult] = []
        eps_rl:      List[DynamicEpisodeResult] = []

        for ep in range(n_episodes):
            s = seed + ep

            # Greedy dynamic
            r_d = run_greedy_dynamic_episode(
                targets_path, cloud_json_path,
                event_rate=rate, duration_s=duration_s, seed=s)
            eps_dynamic.append(r_d)

            # Greedy ignore
            r_i = run_ignore_dynamic_episode(
                targets_path, cloud_json_path,
                event_rate=rate, duration_s=duration_s, seed=s)
            eps_ignore.append(r_i)

            # RL (if model loaded)
            if model is not None:
                r_rl = _eval_rl_episode(
                    model, targets_path, cloud_json_path,
                    event_rate=rate, duration_s=duration_s, seed=s)
                eps_rl.append(r_rl)

            if verbose:
                det   = eps_dynamic[-1].n_dyn_detected
                suc   = eps_dynamic[-1].dyn_success_rate
                print(f"    ep {ep+1:2d}  "
                      f"greedy_dyn={r_d.total_reward:+.3f}  "
                      f"ignore={r_i.total_reward:+.3f}  "
                      f"det={det}  suc={suc:.0%}")

        agg_dyn  = _aggregate_dynamic(eps_dynamic)
        agg_ign  = _aggregate_dynamic(eps_ignore)
        sc_stats = {
            "greedy_dynamic_scout":  agg_dyn,
            "greedy_ignore_dynamic": agg_ign,
        }
        if eps_rl:
            sc_stats["RL-PPO-dynamic"] = _aggregate_dynamic(eps_rl)

        scenario_results[name] = {
            "label":   label,
            "rate":    rate,
            "results": sc_stats,
        }

        if verbose:
            print()
            print_dynamic_table(sc_stats, event_rate=rate)

    # Compute additional science value: greedy_dynamic vs ignore at dense scenario
    if ("dense_events" in scenario_results and
            "greedy_dynamic_scout"  in scenario_results["dense_events"]["results"] and
            "greedy_ignore_dynamic" in scenario_results["dense_events"]["results"]):
        r_dyn = scenario_results["dense_events"]["results"]["greedy_dynamic_scout"]["mean_reward"]
        r_ign = scenario_results["dense_events"]["results"]["greedy_ignore_dynamic"]["mean_reward"]
        if verbose:
            print(f"  Additional science value (dense, dynamic-scout vs ignore): "
                  f"{r_dyn - r_ign:+.3f}")

    return scenario_results


# ============================================================================
#  Plotting
# ============================================================================

def plot_scenario_comparison(scenario_results: dict, save_dir: str = PLOTS_DIR):
    """Generate 4-panel plot: reward, cloud-free rate, dynamic success, delay."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ALSAT-EO-1 Phase 3 — Dynamic Targeting Evaluation", fontsize=13)

    scenario_labels = [sc["label"] for sc in SCENARIOS
                       if sc["name"] in scenario_results]
    x = np.arange(len(scenario_labels))
    width = 0.25

    # Collect per-policy per-scenario values
    policies = ["greedy_dynamic_scout", "greedy_ignore_dynamic", "RL-PPO-dynamic"]
    colors   = ["steelblue", "tomato", "darkorange"]
    p_labels = ["Greedy Dynamic", "Ignore Dynamic", "RL-PPO"]

    def _val(sc_name, pol, key, default=0.0):
        try:
            return scenario_results[sc_name]["results"][pol][key]
        except (KeyError, TypeError):
            return default

    # 1 — Total reward
    ax = axes[0, 0]
    for i, (pol, col, lbl) in enumerate(zip(policies, colors, p_labels)):
        vals = [_val(sc["name"], pol, "mean_reward") for sc in SCENARIOS
                if sc["name"] in scenario_results]
        stds = [_val(sc["name"], pol, "std_reward")  for sc in SCENARIOS
                if sc["name"] in scenario_results]
        if any(v != 0.0 for v in vals):
            ax.bar(x + i*width, vals, width, yerr=stds, label=lbl,
                   color=col, alpha=0.8, capsize=4)
    ax.set_xticks(x + width); ax.set_xticklabels(scenario_labels, fontsize=8)
    ax.set_ylabel("Mean total reward"); ax.set_title("Total Reward per Scenario")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # 2 — Cloud-free rate
    ax = axes[0, 1]
    for i, (pol, col, lbl) in enumerate(zip(policies, colors, p_labels)):
        vals = [_val(sc["name"], pol, "mean_cf_rate") for sc in SCENARIOS
                if sc["name"] in scenario_results]
        if any(v != 0.0 for v in vals):
            ax.bar(x + i*width, vals, width, label=lbl, color=col, alpha=0.8)
    ax.set_xticks(x + width); ax.set_xticklabels(scenario_labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Cloud-free rate")
    ax.set_title("Static Cloud-Free Rate"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # 3 — Dynamic success rate
    ax = axes[1, 0]
    for i, (pol, col, lbl) in enumerate(zip(policies, colors, p_labels)):
        vals = [_val(sc["name"], pol, "mean_dyn_success") for sc in SCENARIOS
                if sc["name"] in scenario_results]
        if any(v != 0.0 for v in vals):
            ax.bar(x + i*width, vals, width, label=lbl, color=col, alpha=0.8)
    ax.set_xticks(x + width); ax.set_xticklabels(scenario_labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Dynamic event success rate")
    ax.set_title("Dynamic Imaging Success Rate"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # 4 — Average delay
    ax = axes[1, 1]
    for i, (pol, col, lbl) in enumerate(zip(policies, colors, p_labels)):
        vals = [_val(sc["name"], pol, "mean_delay_s") / 60.0 for sc in SCENARIOS
                if sc["name"] in scenario_results]   # convert to minutes
        if any(v != 0.0 for v in vals):
            ax.bar(x + i*width, vals, width, label=lbl, color=col, alpha=0.8)
    ax.set_xticks(x + width); ax.set_xticklabels(scenario_labels, fontsize=8)
    ax.set_ylabel("Average delay (min)"); ax.set_title("Avg. Event-to-Imaging Delay")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "dynamic_eval_comparison.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")
    return path


# ============================================================================
#  CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="ALSAT-EO-1 Dynamic Targeting Eval")
    ap.add_argument("--model", default=os.path.join(_ROOT, "models/ppo_smdp_full.zip"),
                help="Path to trained PPO model (.zip). Skipped if not found.")
    ap.add_argument("--targets", default=os.path.join(_ROOT, "scripts/config/targets/algeria_20_targets.json"))
    ap.add_argument("--cloud", default=os.path.join(_ROOT, "scripts/config/cloud_reality/algeria_real_clouds.json"))
    ap.add_argument("--no-rl",     action="store_true",
                    help="Skip RL model evaluation even if model file exists.")
    ap.add_argument("--episodes",  type=int, default=5)
    ap.add_argument("--seed",      type=int, default=100)
    ap.add_argument("--duration",  type=float, default=SIM_DURATION_S)
    ap.add_argument("--no-plot",   action="store_true")
    args = ap.parse_args()

    model_path = None if args.no_rl else args.model

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    print("=" * 70)
    print("  ALSAT-EO-1  Phase 3  Dynamic Targeting Evaluation")
    print("=" * 70)
    print(f"  Scenarios  : {len(SCENARIOS)}"
          f"  Episodes   : {args.episodes}  "
          f"  Duration   : {args.duration/3600:.1f}h")
    print(f"  RL model   : {model_path or '(skipped)'}\n")

    t0 = time.time()
    results = evaluate_all_scenarios(
        targets_path    = args.targets,
        cloud_json_path = args.cloud,
        n_episodes      = args.episodes,
        seed            = args.seed,
        duration_s      = args.duration,
        model_path      = model_path,
        verbose         = True,
    )

    # Save JSON
    log_path = os.path.join(RESULTS_DIR, "dynamic_eval_results.json")
    # Convert dataclass results to plain dicts
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results → {log_path}")
    print(f"  Elapsed  : {(time.time()-t0)/60:.1f} min")

    if not args.no_plot:
        plot_scenario_comparison(results, PLOTS_DIR)


if __name__ == "__main__":
    main()
