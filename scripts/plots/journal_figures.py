#!/usr/bin/env python3
"""
journal_figures.py  —  ALSAT-EO-1  Publication-Quality Figure Generator
========================================================================
Outputs in  data/outputs/plots/journal/:

  training_curves.png     — reward + DYN-success % + entropy over training
  policy_comparison.png   — 4-panel grouped bar chart (3 scenarios × 3 policies)
  dyn_focused.png         — clean DYN success/count comparison (main paper figure)
  feature_importance.png  — top-15 finite-diff feature attributions
  decision_timeline.png   — 24-h episode action + reward timeline
  results_table.tex       — LaTeX table ready to paste into paper

Usage (from project root  ~/Documents/alsat_dynamic_multi/):

  # Fast: reuse cached eval results
  python scripts/plots/journal_figures.py \
      --model   models/ppo_smdp_v5_seed42.zip \
      --log     data/outputs/results/training_live.json \
      --results data/outputs/results/dynamic_eval_results.json \
      --n-eval  10

  # Skip feature importance + timeline (saves ~5 min):
  python scripts/plots/journal_figures.py --no-explain [...]
"""
from __future__ import annotations

import argparse, json, os, sys, logging
import numpy as np
import warnings
# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # alsat_dynamic_multi/
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)
try:
    import path_setup  # noqa
except ImportError:
    pass

# ── suppress bsk_rl noise ────────────────────────────────────────────────────
import logging as _lg
_SKIP = frozenset(["Creating logger for new env",
                   "Old environments in process",
                   "basePowerDraw should probably be zero or negative"])
_orig_ch = _lg.Logger.callHandlers
def _quiet(self, r):
    try:
        if any(s in r.getMessage() for s in _SKIP): return
    except Exception: pass
    _orig_ch(self, r)
_lg.Logger.callHandlers = _quiet
_lg.getLogger("bsk_rl").setLevel(_lg.ERROR)
_lg.getLogger("bsk_rl.sats").setLevel(_lg.ERROR)
_lg.getLogger("bsk_rl.sats.access_satellite").setLevel(_lg.ERROR)
warnings.filterwarnings("ignore", module="bsk_rl")
warnings.filterwarnings("ignore", module="Basilisk")


logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── journal style (colorblind-safe, Wong 2011) ────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

C_RL     = "#0072B2"   # blue
C_GREEDY = "#E69F00"   # orange
C_IGNORE = "#CC79A7"   # pink/magenta
C_STATIC = "#009E73"   # green
C_DYN    = "#D55E00"   # vermillion
C_DRIFT  = "#999999"   # grey

POLICY_COLORS = [C_GREEDY, C_IGNORE, C_RL]
POLICY_LABELS = ["Greedy-Scout", "Greedy-Ignore", "RL-PPO (ours)"]
POLICY_KEYS   = ["greedy_dynamic_scout", "greedy_ignore_dynamic", "RL-PPO-dynamic"]
SCENARIO_KEYS = ["no_events", "sparse_events", "dense_events"]
SCENARIO_LABELS = ["No events\n(0/hr)", "Sparse events\n(0.5/hr)", "Dense events\n(2.0/hr)"]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, w: int = 100) -> np.ndarray:
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _load_training_log(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognised training log format: {path}")


def _val(results: dict, sc: str, pol: str, key: str, default: float = 0.0) -> float:
    try:
        return float(results[sc]["results"][pol][key])
    except (KeyError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 1 — Training Learning Curves
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_curves(log_path: str, out_path: str) -> None:
    episodes = _load_training_log(log_path)
    if not episodes:
        print("  [SKIP] Training log empty.")
        return

    def _g(d, *keys):
        for k in keys:
            if k in d:
                return float(d[k])
        return 0.0

    ep  = np.array([_g(d, "ep", "episode")                  for d in episodes])
    rew = np.array([_g(d, "reward", "total_reward")          for d in episodes])
    ds  = np.array([_g(d, "dyn_suc", "dyn_success_rate")*100 for d in episodes])
    ent = np.array([_g(d, "ent_coef", "entropy_coef")        for d in episodes])

    W = 100

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"hspace": 0.5})
    fig.suptitle("ALSAT-EO-1 SMDP-PPO — Training Progress",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax, y, color, ylabel, title in [
        (axes[0], rew, C_RL,  "Episode Reward",         "(a) Episode Reward"),
        (axes[1], ds,  C_DYN, "DYN Success Rate (%)",   "(b) Dynamic Event Imaging Success Rate"),
        (axes[2], ent, "teal","Entropy coefficient",    "(c) Entropy Annealing"),
    ]:
        ax.fill_between(ep, y, 0, alpha=0.10, color=color)
        ax.plot(ep, y, alpha=0.20, linewidth=0.6, color=color)
        if len(y) >= W:
            ep_s = ep[W - 1:]
            final = np.mean(y[-100:])
            label = f"100-ep mean  (final = {final:.1f})" + ("%" if "%" in ylabel else "")
            ax.plot(ep_s, _smooth(y, W), color=color, linewidth=2.0, label=label)
            ax.legend(loc="lower right" if ax is not axes[2] else "upper right")
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ax is axes[2]:
            ax.set_xlabel("Training episode")

    # stats box
    stats = (f"Final 100-ep stats:\n"
             f"  Reward  {np.mean(rew[-100:]):+.2f} ± {np.std(rew[-100:]):.2f}\n"
             f"  DYN suc {np.mean(ds[-100:]):.1f}% ± {np.std(ds[-100:]):.1f}%\n"
             f"  Episodes trained: {int(ep[-1]):,}")
    fig.text(0.72, 0.11, stats, fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Fig 1 — Training curves       → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 2 — Multi-Scenario Policy Comparison (4 panels)
# ─────────────────────────────────────────────────────────────────────────────

def fig_policy_comparison(results: dict, out_path: str) -> None:
    sc_keys    = [k for k in SCENARIO_KEYS if k in results]
    sc_labels  = [SCENARIO_LABELS[SCENARIO_KEYS.index(k)] for k in sc_keys]
    x = np.arange(len(sc_keys))
    W = 0.22
    offsets = [-W, 0, W]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "ALSAT-EO-1 Phase 3 — Policy Comparison Across Event Density Scenarios",
        fontsize=12, fontweight="bold", y=1.01)

    def _bars(ax, key, ylabel, title, scale=1.0, ylim=None):
        for i, (pol, col, lbl) in enumerate(
                zip(POLICY_KEYS, POLICY_COLORS, POLICY_LABELS)):
            vals = [_val(results, sc, pol, key) * scale   for sc in sc_keys]
            stds_key = "std_reward" if key == "mean_reward" else key
            stds = [0.0] * len(sc_keys)
            if key == "mean_reward":
                stds = [_val(results, sc, pol, "std_reward") * scale for sc in sc_keys]
            ax.bar(x + offsets[i], vals, W, yerr=stds, label=lbl, color=col,
                   alpha=0.85, capsize=4, error_kw={"linewidth": 1.2})
        ax.set_xticks(x); ax.set_xticklabels(sc_labels, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc="upper right")
        if ylim:
            ax.set_ylim(*ylim)

    _bars(axes[0, 0], "mean_reward",      "Mean Episode Reward",
          "(a) Total Reward per Scenario")
    axes[0, 0].axhline(0, color="k", linewidth=0.5, linestyle="--")

    _bars(axes[0, 1], "mean_cf_rate",     "Cloud-Free Rate",
          "(b) Static Imaging Quality (CF Rate)", ylim=(0, 1.1))

    _bars(axes[1, 0], "mean_dyn_success", "DYN Event Success Rate",
          "(c) Dynamic Event Imaging Success Rate", ylim=(0, None))

    _bars(axes[1, 1], "mean_delay_s",     "Mean Response Delay (s)",
          "(d) Average Event-to-Imaging Delay")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Fig 2 — Policy comparison     → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 3 — Focused DYN Metrics (cleaner, for main paper body)
# ─────────────────────────────────────────────────────────────────────────────

def fig_dyn_focused(results: dict, out_path: str) -> None:
    """2-panel: DYN success rate + mean DYN images/episode. Sparse+Dense only."""
    sc_keys   = [k for k in ["sparse_events", "dense_events"] if k in results]
    sc_labels = ["Sparse\n(0.5 ev/hr)", "Dense\n(2.0 ev/hr)"]
    x = np.arange(len(sc_keys))
    W = 0.22; offsets = [-W, 0, W]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Dynamic Event Imaging Performance — ALSAT-EO-1 SMDP-PPO",
                 fontsize=12, fontweight="bold")

    for i, (pol, col, lbl) in enumerate(
            zip(POLICY_KEYS, POLICY_COLORS, POLICY_LABELS)):
        vals1 = [_val(results, sc, pol, "mean_dyn_success") * 100 for sc in sc_keys]
        vals2 = [_val(results, sc, pol, "mean_dyn_imaged")        for sc in sc_keys]
        ax1.bar(x + offsets[i], vals1, W, label=lbl, color=col, alpha=0.85)
        ax2.bar(x + offsets[i], vals2, W, label=lbl, color=col, alpha=0.85)

    for ax, ylabel, title in [
        (ax1, "DYN Event Success Rate (%)",        "(a) Dynamic Imaging Success Rate"),
        (ax2, "Mean DYN Events Imaged / Episode",  "(b) Dynamic Events Imaged"),
    ]:
        ax.set_xticks(x); ax.set_xticklabels(sc_labels)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Fig 3 — DYN focused chart     → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 4 — Feature Importance (finite-diff attributions)
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_importance(model, cfg, targets_path, cloud_path,
                            out_path, n_bg=150, event_rate=2.0,
                            seed=500, top_k=15) -> None:
    try:
        from explainability import PolicyExplainer, build_feature_names
        from eval_smdp_explain import collect_background_states
        from env_dynamic_factory import obs_dim
    except ImportError as e:
        print(f"  [SKIP] Explainability imports failed ({e})")
        return

    print(f"  Collecting {n_bg} background states...")
    bg = collect_background_states(
        model, cfg, targets_path, cloud_path,
        event_rate=event_rate, n_steps=n_bg, seed=seed)

    fnames   = build_feature_names()[:obs_dim(cfg)]
    explainer = PolicyExplainer(model, bg, feature_names=fnames)

    sample_idx = np.linspace(0, len(bg) - 1, min(30, len(bg)), dtype=int)
    attrs = []
    print(f"  Computing {len(sample_idx)} attributions...")
    for idx in sample_idx:
        try:
            attrs.append(explainer.explain(bg[idx]))
        except Exception as e:
            logger.debug(f"Attribution {idx}: {e}")

    if not attrs:
        print("  [SKIP] No attributions computed.")
        return

    attr_matrix = np.array(attrs)
    mean_abs    = np.abs(attr_matrix).mean(0)
    top_idx     = np.argsort(mean_abs)[::-1][:top_k]

    def _col(name):
        if name.startswith("dyn_"):    return C_DYN
        if name.startswith("target_"): return C_STATIC
        if name == "sojourn_norm":     return "#56B4E9"
        return "#BBBBBB"

    feat_names  = [fnames[i] if i < len(fnames) else f"feat_{i}" for i in top_idx]
    feat_vals   = mean_abs[top_idx]
    feat_colors = [_col(n) for n in feat_names]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(top_k), feat_vals[::-1], color=feat_colors[::-1],
            alpha=0.85, edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(feat_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |Δ log-prob| / ε  (finite-difference attribution)")
    ax.set_title("Feature Importance — ALSAT-EO-1 SMDP Policy\n"
                 "(larger = policy more sensitive to this input feature)")
    legend_handles = [
        mpatches.Patch(color=C_DYN,    label="Dynamic event features"),
        mpatches.Patch(color=C_STATIC, label="Static target features"),
        mpatches.Patch(color="#56B4E9",label="SMDP sojourn time"),
        mpatches.Patch(color="#BBBBBB",label="State (pos/vel/time)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Fig 4 — Feature importance    → {out_path}")
    print("       Top-5 features:")
    for rank in range(min(5, top_k)):
        i = top_idx[rank]
        n = fnames[i] if i < len(fnames) else f"feat_{i}"
        print(f"         {rank+1}. {n:<28}  |attr|={mean_abs[i]:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIG 5 — Decision Timeline (single episode)
# ─────────────────────────────────────────────────────────────────────────────

def fig_decision_timeline(model, cfg, targets_path, cloud_path,
                           out_path, event_rate=2.0, seed=300) -> None:
    try:
        from eval_smdp_explain import rollout_with_logging
    except ImportError as e:
        print(f"  [SKIP] rollout_with_logging import failed ({e})")
        return

    print("  Rolling out one episode for decision timeline...")
    dec_log = rollout_with_logging(
        model, cfg, targets_path, cloud_path,
        event_rate=event_rate, seed=seed)

    if not dec_log.records:
        print("  [SKIP] No decision records.")
        return

    records = dec_log.records
    times   = np.array([r.sim_time / 3600 for r in records])
    rewards = np.array([r.reward for r in records])
    cum_r   = np.cumsum(rewards)
    n_static = sum(1 for r in records if not r.is_dynamic and r.action < 20)
    n_dyn    = sum(1 for r in records if r.is_dynamic)
    n_drift  = len(records) - n_static - n_dyn

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [2.5, 1, 1.5],
                                          "hspace": 0.45})
    fig.suptitle(
        f"ALSAT-EO-1 SMDP-PPO — 24-h Episode Decision Timeline  "
        f"(event_rate={event_rate}/hr, seed={seed})",
        fontsize=11, fontweight="bold")

    # Panel A: cumulative reward
    ax = axes[0]
    ax.plot(times, cum_r, color=C_RL, linewidth=2,
            label=f"Cumulative reward = {float(cum_r[-1]):+.2f}")
    ax.fill_between(times, cum_r, 0, alpha=0.10, color=C_RL)
    ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
    for r in records:
        if r.is_dynamic and r.reward > 0.1:
            ax.axvline(r.sim_time / 3600, color=C_DYN, alpha=0.45,
                       linewidth=1.0, linestyle=":")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("(a) Cumulative Reward  (dotted = DYN imaging success)")
    ax.legend(loc="upper left")

    # Panel B: action type
    ax = axes[1]
    for r in records:
        if r.action < 20:       col = C_STATIC
        elif r.action < 23:     col = C_DYN
        else:                   col = C_DRIFT
        ax.bar(r.sim_time / 3600, 1, width=0.4, color=col, alpha=0.75, linewidth=0)
    patches = [
        mpatches.Patch(color=C_STATIC, label=f"Static ({n_static})"),
        mpatches.Patch(color=C_DYN,    label=f"Dynamic ({n_dyn})"),
        mpatches.Patch(color=C_DRIFT,  label=f"Drift ({n_drift})"),
    ]
    ax.legend(handles=patches, loc="upper right", ncol=3)
    ax.set_yticks([]); ax.set_ylabel("Action type")
    ax.set_title("(b) Action Type at Each Decision Step")

    # Panel C: per-step reward heatmap
    ax = axes[2]
    vmax = max(0.1, float(np.abs(rewards).max()))
    im = ax.imshow(rewards.reshape(1, -1), aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax,
                   extent=[times.min(), times.max(), 0, 1])
    plt.colorbar(im, ax=ax, label="Step reward", fraction=0.02)
    ax.set_yticks([]); ax.set_ylabel("Reward")
    ax.set_xlabel("Simulation time (h)")
    ax.set_title("(c) Per-Step Reward Heatmap  (green=positive, red=penalty)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Fig 5 — Decision timeline     → {out_path}")
    print(f"       decisions: {len(records)}  static={n_static}  "
          f"dyn={n_dyn}  drift={n_drift}  total_r={float(cum_r[-1]):+.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def write_latex_table(results: dict, out_path: str) -> None:
    pol_display = {
        "greedy_dynamic_scout":  "Greedy-Scout",
        "greedy_ignore_dynamic": "Greedy-Ignore",
        "RL-PPO-dynamic":        r"\textbf{RL-PPO (ours)}",
    }
    sc_display = {
        "no_events":     "No events",
        "sparse_events": "Sparse (0.5/hr)",
        "dense_events":  "Dense (2.0/hr)",
    }

    def _b(val, best, fmt):
        s = fmt.format(val)
        return r"\textbf{" + s + "}" if abs(val - best) < 1e-4 else s

    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{ALSAT-EO-1 Phase 3: Policy comparison (mean $\pm$ std over "
        r"10 episodes). \textbf{Bold} = best per column.}",
        r"\label{tab:policy_comparison}",
        r"\small",
        r"\begin{tabular}{l l r@{\,$\pm$\,}l r r r}",
        r"\toprule",
        r"Scenario & Policy & \multicolumn{2}{c}{Reward} "
        r"& CF\% & DYN suc\% & DYN imaged \\",
        r"\midrule",
    ]

    for sc in SCENARIO_KEYS:
        if sc not in results: continue
        bests = {
            "r":  max(_val(results, sc, p, "mean_reward")       for p in POLICY_KEYS),
            "cf": max(_val(results, sc, p, "mean_cf_rate")      for p in POLICY_KEYS),
            "ds": max(_val(results, sc, p, "mean_dyn_success")  for p in POLICY_KEYS),
            "di": max(_val(results, sc, p, "mean_dyn_imaged",0) for p in POLICY_KEYS),
        }
        first = True
        for pol in POLICY_KEYS:
            r   = _val(results, sc, pol, "mean_reward")
            rs  = _val(results, sc, pol, "std_reward")
            cf  = _val(results, sc, pol, "mean_cf_rate") * 100
            ds  = _val(results, sc, pol, "mean_dyn_success") * 100
            di  = _val(results, sc, pol, "mean_dyn_imaged", 0)
            row = (f"{sc_display.get(sc,'') if first else ''} & "
                   f"{pol_display.get(pol, pol)} & "
                   f"{_b(r,bests['r'],'{:+.2f}')} & {rs:.2f} & "
                   f"{_b(cf,bests['cf'],'{:.0f}')}\\% & "
                   f"{_b(ds,bests['ds'],'{:.1f}')}\\% & "
                   f"{_b(di,bests['di'],'{:.1f}')} \\\\")
            lines.append(row)
            first = False
        lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ LaTeX table               → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation (skipped if cached results exist)
# ─────────────────────────────────────────────────────────────────────────────

def _run_or_load_eval(model_path, targets, cloud, n_eval, seed, cache_path):
    if os.path.isfile(cache_path):
        print(f"  Using cached results: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"  Running evaluation ({n_eval} eps/scenario) — this takes ~20 min ...")
    from eval_dynamic import evaluate_all_scenarios
    results = evaluate_all_scenarios(
        targets_path=targets, cloud_json_path=cloud,
        n_episodes=n_eval, seed=seed, model_path=model_path, verbose=True,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Results cached → {cache_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ALSAT-EO-1 journal figure generator")
    ap.add_argument("--model",
        default=os.path.join(_ROOT, "models/ppo_smdp_v5_seed42.zip"))
    ap.add_argument("--log",
        default=os.path.join(_ROOT, "data/outputs/results/training_live.json"))
    ap.add_argument("--results",
        default=os.path.join(_ROOT, "data/outputs/results/dynamic_eval_results.json"))
    ap.add_argument("--targets",
        default=os.path.join(_ROOT, "scripts/config/targets/algeria_20_targets.json"))
    ap.add_argument("--cloud",
        default=os.path.join(_ROOT, "scripts/config/cloud_reality/algeria_real_clouds.json"))
    ap.add_argument("--out",
        default=os.path.join(_ROOT, "data/outputs/plots/journal"))
    ap.add_argument("--n-eval",   type=int, default=10)
    ap.add_argument("--seed",     type=int, default=200)
    ap.add_argument("--no-explain", action="store_true",
        help="Skip feature importance + timeline (~5 min per figure)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"\n{'='*62}")
    print(f"  ALSAT-EO-1  Journal Figure Generator")
    print(f"{'='*62}")
    print(f"  Output: {args.out}")

    # Fig 1: training curves
    if os.path.isfile(args.log):
        fig_training_curves(args.log, os.path.join(args.out, "training_curves.png"))
    else:
        print(f"  [SKIP] Training log not found: {args.log}")

    # Load model
    model = None
    if not os.path.isfile(args.model):
        print(f"  [ERROR] Model not found: {args.model}")
    else:
        print(f"\n  Loading model ...")
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        print(f"  [RL] Model loaded OK")

    # Evaluation
    results = None
    if model is not None:
        results = _run_or_load_eval(
            args.model, args.targets, args.cloud,
            args.n_eval, args.seed, args.results)

    # Figs 2 + 3 + LaTeX table
    if results is not None:
        fig_policy_comparison(results, os.path.join(args.out, "policy_comparison.png"))
        fig_dyn_focused(results, os.path.join(args.out, "dyn_focused.png"))
        write_latex_table(results, os.path.join(args.out, "results_table.tex"))

    # Figs 4 + 5
    if model is not None and not args.no_explain:
        try:
            from env_dynamic_factory import Config
            cfg = Config.DYN_VISION
            fig_feature_importance(
                model, cfg, args.targets, args.cloud,
                out_path=os.path.join(args.out, "feature_importance.png"),
                event_rate=2.0, seed=args.seed)
            fig_decision_timeline(
                model, cfg, args.targets, args.cloud,
                out_path=os.path.join(args.out, "decision_timeline.png"),
                event_rate=2.0, seed=args.seed)
        except Exception as exc:
            print(f"  [WARN] Explainability figures failed: {exc}")
            import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*62}")
    print(f"  All figures → {args.out}/")
    for f in sorted(os.listdir(args.out)):
        if f.endswith((".png", ".tex")):
            kb = os.path.getsize(os.path.join(args.out, f)) // 1024
            print(f"    {f:<38}  {kb:>4} KB")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()