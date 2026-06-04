#!/usr/bin/env python3
"""
eval_smdp_explain.py  —  ALSAT-EO-1  Phase 3  Explainability Evaluation
========================================================================
Post-training explainability analysis for the SMDP dynamic targeting agent.
Satisfies Proposal §5 "mechanisms to interpret and visualise decisions".

Outputs
-------
  <output_dir>/
    feature_importance.png      — top-20 SHAP/finite-diff bar chart
    decision_timeline.png       — 48h action timeline with SHAP heatmap
    decisions.json              — full step-by-step decision log (NL explanations)
    shap_summary.txt            — text report: top-5 features per scenario

Usage
-----
    # Called from train_ppo_smdp_full.py --explain, or standalone:
    python scripts/eval_smdp_explain.py \\
        --model models/ppo_smdp_full.zip \\
        --event-rate 2.0 --seed 300

    # Quick test (no SHAP, finite-diff only, 1 episode):
    python scripts/eval_smdp_explain.py --model models/ppo_smdp_full.zip \\
        --episodes 1 --no-shap
"""
from __future__ import annotations
# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------




import argparse, json, os, sys, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

# ── Suppress bsk_rl noise ────────────────────────────────────────────────────
import logging as _lg
_SKIP = frozenset(["Creating logger","Old environments","basePowerDraw"])
_orig = _lg.Logger.callHandlers
def _q(self, r):
    try:
        if any(s in r.getMessage() for s in _SKIP): return
    except Exception: pass
    _orig(self, r)
_lg.Logger.callHandlers = _q

logger = logging.getLogger(__name__)

from env_dynamic_factory import Config, make_env, obs_dim, n_actions
from explainability import (
    DecisionLogger, PolicyExplainer,
    TimelineRenderer, build_feature_names, FEATURE_NAMES,
)
from env_alsat_debug import SIM_DURATION_S


# ============================================================================
#  Background state collector
# ============================================================================

def collect_background_states(
    model,
    cfg:        Config,
    targets_path:    str,
    cloud_json_path: str,
    event_rate: float = 2.0,
    n_steps:    int   = 200,
    seed:       int   = 500,
) -> np.ndarray:
    """
    Run model for n_steps, collecting observations as background for SHAP.
    Returns (n_steps, obs_dim) array.
    """
    env  = make_env(cfg, targets_path, cloud_json_path,
                    event_rate=event_rate, seed=seed)
    obs, _ = env.reset(seed=seed)
    bg = [obs.copy()]
    for _ in range(n_steps - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(action))
        bg.append(obs.copy())
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.array(bg, dtype=np.float32)


# ============================================================================
#  Single-episode rollout with full logging
# ============================================================================

def rollout_with_logging(
    model,
    cfg:         Config,
    targets_path:    str,
    cloud_json_path: str,
    event_rate:  float = 2.0,
    seed:        int   = 300,
) -> DecisionLogger:
    """Run one full episode, recording every decision."""
    env       = make_env(cfg, targets_path, cloud_json_path,
                         event_rate=event_rate, seed=seed)
    obs, _    = env.reset(seed=seed)
    dec_log   = DecisionLogger()
    done      = False

    while not done:
        # Get policy distribution for top-k alternatives
        try:
            import torch
            with torch.no_grad():
                t    = torch.FloatTensor(obs.reshape(1,-1))
                dist = model.policy.get_distribution(t)
                probs = dist.distribution.probs.squeeze(0).numpy()
        except Exception:
            probs = None

        action, _ = model.predict(obs, deterministic=True)
        action    = int(action)

        # Get satellite for logging
        try:
            sat      = env.unwrapped.satellites[0]
            sim_time = float(sat.simulator.sim_time)
        except Exception:
            sim_time = 0.0
            sat      = None

        obs_new, r, term, trunc, info = env.step(action)
        done = term or trunc

        if sat is not None:
            dec_log.record(sat, obs, action, r, sim_time, policy_probs=probs)
        obs = obs_new

    env.close()
    return dec_log


# ============================================================================
#  Feature importance bar chart
# ============================================================================

def plot_feature_importance(shap_values: np.ndarray,
                             feature_names: list,
                             save_path: str,
                             top_k: int = 20) -> None:
    """Bar chart of mean |SHAP| attribution for top_k features."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_k), mean_abs[idx][::-1],
                   color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance — SMDP Policy Value Function")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Feature importance plot → {save_path}")


# ============================================================================
#  Text summary
# ============================================================================

def write_shap_summary(shap_values: np.ndarray,
                        feature_names: list,
                        dec_log: DecisionLogger,
                        save_path: str) -> str:
    mean_abs  = np.abs(shap_values).mean(axis=0)
    top5_idx  = np.argsort(mean_abs)[::-1][:5]
    summary   = dec_log.summary()

    lines = [
        "ALSAT-EO-1 Phase 3 — Explainability Summary",
        "=" * 50,
        "",
        f"Episode decisions: {summary['n_decisions']}",
        f"  Static targets : {summary['n_static']}",
        f"  Dynamic events : {summary['n_dynamic']}",
        f"  Drift          : {summary['n_drift']}",
        f"  Total reward   : {summary['total_reward']:+.3f}",
        "",
        "Top-5 most influential features (by mean |SHAP|):",
    ]
    for rank, i in enumerate(top5_idx):
        lines.append(f"  {rank+1}. {feature_names[i]:<30}  "
                     f"mean|SHAP|={mean_abs[i]:.4f}")

    # Sample explanations
    lines += ["", "Sample decision explanations:"]
    imaging = [r for r in dec_log.records if r.action < 23][:5]
    for rec in imaging:
        lines.append(f"  t={rec.sim_time/3600:.1f}h  {rec.explanation}")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write(text)
    logger.info(f"SHAP summary → {save_path}")
    print(text)
    return text


# ============================================================================
#  Main function (called by train_ppo_smdp_full.py and CLI)
# ============================================================================

def run_explainability_report(
    model,
    targets_path:    str,
    cloud_json_path: str,
    cfg:         Config = Config.DYN_VISION,
    event_rate:  float  = 2.0,
    seed:        int    = 300,
    output_dir:  str    = "results/explainability_report",
    n_bg_states: int    = 150,
    use_shap:    bool   = True,
    episodes:    int    = 1,
) -> None:
    """
    Full explainability pipeline for a trained model.

    Steps:
      1. Collect background states (for SHAP)
      2. Rollout episode(s) with DecisionLogger
      3. Compute SHAP/finite-diff attributions
      4. Plot feature importance + timeline
      5. Save decision JSON + text summary
    """
    os.makedirs(output_dir, exist_ok=True)
    fnames = build_feature_names()[:obs_dim(cfg)]

    print(f"\n  Collecting {n_bg_states} background states...")
    bg = collect_background_states(
        model, cfg, targets_path, cloud_json_path,
        event_rate=event_rate, n_steps=n_bg_states, seed=seed+1000,
    )

    # Build explainer
    explainer = PolicyExplainer(model, bg, feature_names=fnames)

    all_shap   = []
    all_records = []
    for ep in range(episodes):
        print(f"  Rolling out episode {ep+1}/{episodes}...")
        dec_log = rollout_with_logging(
            model, cfg, targets_path, cloud_json_path,
            event_rate=event_rate, seed=seed+ep,
        )
        # Compute attributions for a sample of steps
        sample_records = dec_log.records[::max(1, len(dec_log.records)//50)]
        print(f"  Computing attributions for {len(sample_records)} steps...")
        for rec in sample_records:
            if use_shap:
                attr = explainer.explain(rec.obs)
                rec.shap_values = attr
                all_shap.append(attr)
        all_records.extend(dec_log.records)

    if all_shap:
        shap_matrix = np.array(all_shap)
        # Feature importance plot
        plot_feature_importance(
            shap_matrix, fnames,
            os.path.join(output_dir, "feature_importance.png"),
        )
        # SHAP summary text
        write_shap_summary(
            shap_matrix, fnames, dec_log,
            os.path.join(output_dir, "shap_summary.txt"),
        )
    else:
        print("  [WARN] No SHAP values computed.")

    # Timeline (uses last episode's records with SHAP)
    if 'dec_log' not in dir() or dec_log is None:
        print("  [WARN] No episodes completed, skipping timeline.")
        return
    print("  Rendering decision timeline...")
    TimelineRenderer().render(
        dec_log.records,
        os.path.join(output_dir, "decision_timeline.png"),
        feature_names=fnames,
    )

    # Save full decision log
    dec_log.save(os.path.join(output_dir, "decisions.json"))

    print(f"\n  Explainability report → {output_dir}/")
    print(f"    feature_importance.png")
    print(f"    decision_timeline.png")
    print(f"    decisions.json")
    print(f"    shap_summary.txt")


# ============================================================================
#  CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="ALSAT-EO-1 explainability eval")
    ap.add_argument("--model",      required=True)
    ap.add_argument("--targets",    default=os.path.join(_ROOT,
                    "config/targets/algeria_20_targets.json"))
    ap.add_argument("--cloud",      default=os.path.join(_ROOT,
                    "config/cloud_reality/algeria_real_clouds.json"))
    ap.add_argument("--event-rate", type=float, default=2.0)
    ap.add_argument("--seed",       type=int,   default=300)
    ap.add_argument("--episodes",   type=int,   default=1)
    ap.add_argument("--output-dir", default=os.path.join(_ROOT,
                    "results/explainability_report"))
    ap.add_argument("--no-vision",  action="store_true")
    ap.add_argument("--no-shap",    action="store_true")
    ap.add_argument("--bg-states",  type=int, default=150)
    args = ap.parse_args()

    cnn_path  = os.path.join(_ROOT, "models/cloud_cnn_real.pt")
    use_vis   = not args.no_vision and os.path.exists(cnn_path)
    cfg       = Config.DYN_MODIS if not use_vis else Config.DYN_VISION
    print(f"  Config: {cfg.value}  obs_dim={obs_dim(cfg)}")

    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    print(f"  Model loaded from {args.model}")

    run_explainability_report(
        model=model,
        targets_path=args.targets,
        cloud_json_path=args.cloud,
        cfg=cfg,
        event_rate=args.event_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        n_bg_states=args.bg_states,
        use_shap=not args.no_shap,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
