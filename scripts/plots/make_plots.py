#!/usr/bin/env python3
"""
make_plots.py  —  Training + Ablation visualizations for ALSAT-EO-1
====================================================================
Usage:
  # After single training run:
  python scripts/plots/make_plots.py --log results/training_log.json

  # After ablation study:
  python scripts/plots/make_plots.py --ablation results/ablation

  # Both:
  python scripts/plots/make_plots.py \
      --log results/training_log.json \
      --ablation results/ablation \
      --out results/plots
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

def _smooth(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")

def _load_log(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
def plot_training(log_path: str, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    data = _load_log(log_path)
    eps        = [d["episode"]          for d in data]
    rewards    = [d["reward"]           for d in data]
    dyn_suc    = [d["dyn_success_rate"] for d in data]
    n_dyn_img  = [d["n_dyn_imaged"]     for d in data]
    n_dyn_det  = [d["n_dyn_detected"]   for d in data]
    cf_rates   = [d["cf_rate"]          for d in data]
    ent        = [d.get("ent_coef", 0)  for d in data]
    slew_deg   = [d.get("total_slew_deg", 0) for d in data]

    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("ALSAT-EO-1  Phase 3  Training Metrics", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def _ax(row, col, title, ylabel, xs, ys, color="steelblue", smooth=True):
        ax = fig.add_subplot(gs[row, col])
        ax.plot(xs, ys, alpha=0.25, color=color, linewidth=0.8)
        if smooth and len(ys) >= 20:
            s = _smooth(np.array(ys))
            ax.plot(xs[len(xs)-len(s):], s, color=color, linewidth=1.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return ax

    _ax(0, 0, "Episode Reward",         "Reward",     eps, rewards,   "steelblue")
    _ax(0, 1, "DYN Success Rate",        "Success %",  eps, [d*100 for d in dyn_suc], "darkorange")
    _ax(0, 2, "DYN Imaged / Episode",    "Count",      eps, n_dyn_img, "green")

    _ax(1, 0, "Cloud-Free Rate",         "CF %",       eps, [d*100 for d in cf_rates], "purple")
    _ax(1, 1, "DYN Detected / Episode",  "Count",      eps, n_dyn_det, "crimson")
    _ax(1, 2, "Total Slew / Episode (°)","Degrees",    eps, slew_deg,  "brown", smooth=False)

    # Reward distribution
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_hist.hist(rewards, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax_hist.axvline(np.mean(rewards), color="red", linestyle="--", label=f"μ={np.mean(rewards):.2f}")
    ax_hist.set_title("Reward Distribution"); ax_hist.set_xlabel("Reward")
    ax_hist.legend(fontsize=8); ax_hist.grid(True, alpha=0.3)

    # Entropy annealing
    ax_ent = fig.add_subplot(gs[2, 1])
    ax_ent.plot(eps, ent, color="teal", linewidth=1.5)
    ax_ent.set_title("Entropy Coefficient"); ax_ent.set_xlabel("Episode")
    ax_ent.set_ylabel("ent_coef"); ax_ent.grid(True, alpha=0.3)

    # DYN success scatter
    ax_sc = fig.add_subplot(gs[2, 2])
    sc = ax_sc.scatter(rewards, [d*100 for d in dyn_suc],
                       c=eps, cmap="viridis", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax_sc, label="Episode")
    ax_sc.set_title("Reward vs DYN Success")
    ax_sc.set_xlabel("Reward"); ax_sc.set_ylabel("DYN Success %")
    ax_sc.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Training curves → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation(ablation_dir: str, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    variants_order = [
        "full_system", "no_bc", "no_curriculum",
        "no_smdp", "gaussian_cloud", "circular_cnn"
    ]
    variant_labels = {
        "full_system":   "Full System\n(A)",
        "no_bc":         "No BC\n(B)",
        "no_curriculum": "No Curriculum\n(C)",
        "no_smdp":       "No SMDP\n(D)",
        "gaussian_cloud":"Gaussian Cloud\n(E)",
        "circular_cnn":  "Circular CNN\n(F)",
    }
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0", "#795548"]

    results: dict[str, list[dict]] = {}
    for v in variants_order:
        vdir = os.path.join(ablation_dir, v)
        if not os.path.isdir(vdir):
            continue
        seeds = []
        for sd in os.listdir(vdir):
            p = os.path.join(vdir, sd, "eval_metrics.json")
            if os.path.exists(p):
                with open(p) as f:
                    seeds.append(json.load(f))
        if seeds:
            results[v] = seeds

    if not results:
        print("No ablation results found.")
        return

    os.makedirs(out_dir, exist_ok=True)
    present = [v for v in variants_order if v in results]
    n = len(present)
    x = np.arange(n)
    w = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("ALSAT-EO-1  Ablation Study", fontsize=14, fontweight="bold")

    metrics = [
        ("mean_reward",  "std_reward",  "Mean Episode Reward"),
        ("cf_rate",      "cf_rate_std", "Cloud-Free Rate"),
        ("dyn_suc",      "dyn_suc_std", "DYN Success Rate"),
    ]

    for ax, (m_key, s_key, title) in zip(axes, metrics):
        means = [np.mean([s[m_key] for s in results[v]]) for v in present]
        stds  = [np.std( [s[m_key] for s in results[v]]) for v in present]
        bars  = ax.bar(x, means, yerr=stds, capsize=5,
                       color=[colors[variants_order.index(v)] for v in present],
                       alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([variant_labels.get(v, v) for v in present], fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds)*0.05,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "ablation_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Ablation comparison → {out_path}")

    # ── Per-variant learning curves (if training_log.json exists) ─────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("ALSAT-EO-1  Per-Variant Learning Curves", fontsize=13, fontweight="bold")
    for ax, v in zip(axes2.flat, present):
        vdir = os.path.join(ablation_dir, v)
        seed_curves = []
        for sd in os.listdir(vdir):
            lp = os.path.join(vdir, sd, "training_log.json")
            if os.path.exists(lp):
                log = _load_log(lp)
                seed_curves.append([e["reward"] for e in log])
        if not seed_curves:
            ax.text(0.5, 0.5, "No training log", transform=ax.transAxes, ha="center")
        else:
            max_len = max(len(c) for c in seed_curves)
            for curve in seed_curves:
                ax.plot(range(len(curve)), curve, alpha=0.2, linewidth=0.8,
                        color=colors[variants_order.index(v)])
            # Mean across seeds
            padded = [c + [c[-1]] * (max_len - len(c)) for c in seed_curves]
            mean_curve = np.mean(padded, axis=0)
            smooth_mean = _smooth(mean_curve)
            ax.plot(range(len(smooth_mean)), smooth_mean, linewidth=2.2,
                    color=colors[variants_order.index(v)], label="mean")
            ax.set_title(variant_labels.get(v, v).replace("\n", " "), fontsize=10)
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "ablation_learning_curves.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Ablation learning curves → {out_path2}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",      type=str, default=None, help="Path to training_log.json")
    ap.add_argument("--ablation", type=str, default=None, help="Path to ablation results dir")
    ap.add_argument("--out",      type=str, default="results/plots")
    args = ap.parse_args()

    if args.log:
        plot_training(args.log, args.out)
    if args.ablation:
        plot_ablation(args.ablation, args.out)
    if not args.log and not args.ablation:
        print("Usage: --log <path> and/or --ablation <dir>")
