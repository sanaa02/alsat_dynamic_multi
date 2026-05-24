#!/usr/bin/env python3
"""
train_ppo_dynamic.py  —  ALSAT-EO-1  Phase 3  Dynamic Targeting PPO Training
=============================================================================
Trains a PPO agent on the dynamic targeting environment and evaluates it
against the two greedy baselines across three event-rate scenarios.

Key differences from train_ppo.py (Phase 2)
--------------------------------------------
  - Env factory : make_dynamic_env()   obs=(55,)  actions=Discrete(24)
  - Event rate  : configurable via --event-rate (default 2.0 events/hr)
  - Metrics     : tracks n_dyn_detected, n_dyn_imaged, dyn_success_rate
                  in addition to all Phase 2 metrics
  - Callback    : DynamicTrainingLogger (extends AlsatTrainingLogger)
  - Evaluation  : runs 3 scenarios after training
  - Model path  : models/ppo_dynamic_final.zip (separate from Phase 2 model)

PPO hyperparameters are identical to Phase 2 (MLP 256x256, lr=3e-4,
n_steps=144, batch=72, gamma=0.99) — only obs/action dims differ.

Usage
-----
    python scripts/train_ppo_dynamic.py                    # 2/hr, 500 eps
    python scripts/train_ppo_dynamic.py --event-rate 0.5   # sparse
    python scripts/train_ppo_dynamic.py --event-rate 0.0   # baseline (= Phase 2)
    python scripts/train_ppo_dynamic.py --episodes 1000
    python scripts/train_ppo_dynamic.py --no-eval          # skip post-training eval
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

DEFAULT_TARGETS    = os.path.join(_ROOT, "config/targets/algeria_20_targets.json")
DEFAULT_CLOUD_JSON = os.path.join(_ROOT, "config/cloud_reality/algeria_real_clouds.json")
DEFAULT_MODEL_OUT  = os.path.join(_ROOT, "models/ppo_dynamic_final.zip")
RESULTS_DIR        = os.path.join(_ROOT, "results")
PLOTS_DIR          = os.path.join(_ROOT, "data/outputs/plots")

# ── Suppress bsk_rl log noise ────────────────────────────────────────────────
_BSK_SKIP = frozenset([
    "Creating logger for new env",
    "Old environments in process",
    "basePowerDraw should probably be zero or negative",
])
_orig_ch = logging.Logger.callHandlers
def _quiet_ch(self, record):
    try:
        if any(s in record.getMessage() for s in _BSK_SKIP): return
    except Exception: pass
    _orig_ch(self, record)
logging.Logger.callHandlers = _quiet_ch

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from scripts.core.env_alsat_dynamic import (
        make_dynamic_env, N_TOTAL_ACTIONS, OBS_TOTAL_DIM,
    )
    from scripts.core.env_alsat_debug import SCHED_STEP_S, CLOUD_THRESH, SIM_DURATION_S
    from scripts.core.dynamic_event import DYNAMIC_BONUS
except ImportError as exc:
    print(f"[ERROR] Could not import env_alsat_dynamic: {exc}"); sys.exit(1)

try:
    from scripts.evaluation.baselines_dynamic import (
        run_greedy_dynamic_episode,
        run_ignore_dynamic_episode,
        print_dynamic_table,
        _aggregate_dynamic,
    )
    HAS_BASELINES = True
except ImportError:
    HAS_BASELINES = False
    print("[WARN] baselines_dynamic.py not found — baseline comparison skipped.")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    import bsk_rl
except ImportError as exc:
    print(f"[ERROR] {exc}"); sys.exit(1)

import gymnasium as gym


# ============================================================================
#  Training callback
# ============================================================================

class DynamicTrainingLogger(BaseCallback):
    """
    Logs per-episode metrics for the dynamic targeting environment:
      - total reward (static + dynamic)
      - cloud-free rate (static images)
      - dynamic event detection / imaging counts
      - dynamic success rate and average delay

    Periodically runs greedy_dynamic_scout for live comparison.
    """

    def __init__(self, targets_path, cloud_json_path, duration_s,
                 event_rate, eval_every_steps, seed=42, verbose=1):
        super().__init__(verbose)
        self.targets_path     = targets_path
        self.cloud_json_path  = cloud_json_path
        self.duration_s       = duration_s
        self.event_rate       = event_rate
        self.eval_every_steps = eval_every_steps
        self.seed             = seed

        # Episode history
        self.ep_rewards:     list = []
        self.ep_cf_rates:    list = []
        self.ep_dyn_success: list = []
        self.eval_results:   list = []

        self._ep_reward    = 0.0
        self._ep_count     = 0
        self._last_eval    = 0
        self._steps_per_ep = int(duration_s / 1200) if duration_s else 144
        self._log_every    = 1   # overridden after construction via logger_cb._log_every
        self._action_counts = [0] * N_TOTAL_ACTIONS

        self.save_every = 50                # save every 50 episodes
        self.log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'results', f'training_autosave_{self.seed}.json'
        )
        self._save_freq = 100

    def _auto_save(self):
        """Save current episode history to disk (overwrites)."""
        data = {
            "episodes_done": self._ep_count,
            "ep_rewards": self.ep_rewards,
            "ep_cf_rates": self.ep_cf_rates,
            "ep_dyn_success": self.ep_dyn_success,
            "eval_results": self.eval_results,
        }
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=2, default=float)

    def _on_step(self) -> bool:
        reward = float(self.locals.get("rewards", [0.0])[0])
        done   = bool(self.locals.get("dones",   [False])[0])
        action_taken = int(self.locals.get("actions", [23])[0])
        if action_taken < len(self._action_counts):
            self._action_counts[action_taken] += 1
        self._ep_reward += reward

        # Step heartbeat every 30 steps so you see progress within an episode
        if self.verbose >= 1 and self.num_timesteps % 30 == 0 and not done:
            step_in_ep = self.num_timesteps % max(self._steps_per_ep, 1)
            pct = 100.0 * step_in_ep / max(self._steps_per_ep, 1)
            print(f"    · step {self.num_timesteps:6d} | ep {self._ep_count+1} "
                  f"[{pct:3.0f}%] | step_r={reward:+.3f} | ep_acc={self._ep_reward:+.3f}")

        if done:
            self.ep_rewards.append(self._ep_reward)
            if self._ep_count % self.save_every == 0:
                self._auto_save()

            self._ep_count += 1

            # Read episode metrics saved in SingleSatelliteEnv.step()
            infos = self.locals.get("infos", [{}])
            ep_m  = infos[0].get("episode_metrics", {}) if infos else {}

            n_img     = ep_m.get("n_imaged",        0)
            n_cf      = ep_m.get("n_cloud_free",     0)
            n_dyn_det = ep_m.get("n_dyn_detected",   0)
            n_dyn_img = ep_m.get("n_dyn_imaged",     0)
            slew_e    = ep_m.get("total_slew_energy_wh", 0.0)

            cf_rate  = n_cf      / n_img     if n_img     > 0 else 0.0
            dyn_suc  = n_dyn_img / n_dyn_det if n_dyn_det > 0 else 0.0
            self.ep_cf_rates.append(cf_rate)
            self.ep_dyn_success.append(dyn_suc)

            if self.verbose >= 1 and self._ep_count % self._log_every == 0:
                r10   = np.mean(self.ep_rewards[-10:])
                cf10  = np.mean(self.ep_cf_rates[-10:])   if self.ep_cf_rates    else 0.0
                dyn10 = np.mean(self.ep_dyn_success[-10:]) if self.ep_dyn_success else 0.0
                dyn_frac = sum(self._action_counts[20:23]) / max(sum(self._action_counts), 1)
                avg_tag = f" | avg10: r={r10:+.2f} cf={cf10:.0%} dyn={dyn10:.0%}" if self._ep_count >= 10 else ""
                print(
                    f"  ▶ Ep {self._ep_count:4d}  "
                    f"r={self._ep_reward:+8.3f}{avg_tag}  "
                    f"imgs={n_img:2d}  cf={cf_rate:.0%}  "
                    f"dyn_det={n_dyn_det}  dyn_img={n_dyn_img}  suc={dyn_suc:.0%}  "
                    f"slew={slew_e:.1f}Wh  act_dyn={dyn_frac:.0%}"
                )
            self._ep_reward = 0.0
            model_save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'models',
                f'ppo_checkpoint_{self._ep_count}.zip'
            )
                # In DynamicTrainingLogger._on_step, inside the `if done:` block, at the end:
            if self._ep_count % self._save_freq == 0:
                ckpt = os.path.join(os.path.dirname(DEFAULT_MODEL_OUT),
                                    f"ppo_checkpoint_ep{self._ep_count:04d}.zip")
                self.model.save(ckpt)
                print(f"  💾 Checkpoint saved → {ckpt}")
            self.model.save(model_save_path)

        if HAS_BASELINES and (self.num_timesteps - self._last_eval) >= self.eval_every_steps:
            self._last_eval = self.num_timesteps
            self._run_eval()

        return True

    def _run_eval(self) -> None:
        """Live comparison: RL vs greedy_dynamic_scout at current training step."""
        if not HAS_BASELINES:
            return
        try:
            g_res     = run_greedy_dynamic_episode(
                self.targets_path, self.cloud_json_path,
                event_rate=self.event_rate,
                duration_s=self.duration_s,
                seed=self._ep_count)
            rl_reward = (float(np.mean(self.ep_rewards[-5:]))
                         if len(self.ep_rewards) >= 5
                         else (self.ep_rewards[-1] if self.ep_rewards else 0.0))
            rl_cf     = (float(np.mean(self.ep_cf_rates[-5:]))
                         if len(self.ep_cf_rates) >= 5 else 0.0)
            rl_dyn    = (float(np.mean(self.ep_dyn_success[-5:]))
                         if len(self.ep_dyn_success) >= 5 else 0.0)
            denom     = max(abs(g_res.total_reward), abs(rl_reward), 1.0)
            gap       = (rl_reward - g_res.total_reward) / denom * 100.0
            self.eval_results.append({
                "step":            self.num_timesteps,
                "rl_reward":       float(rl_reward),
                "greedy_reward":   float(g_res.total_reward),
                "rl_cf_rate":      float(rl_cf),
                "greedy_cf_rate":  float(g_res.cloud_free_rate),
                "rl_dyn_success":  float(rl_dyn),
                "greedy_dyn_success": float(g_res.dyn_success_rate),
            })
            print(f"\n  [EVAL] step={self.num_timesteps:7d}"
                  f"  RL={rl_reward:+.3f}(cf={rl_cf:.0%},dyn={rl_dyn:.0%})"
                  f"  Scout={g_res.total_reward:+.3f}(cf={g_res.cloud_free_rate:.0%},"
                  f"dyn={g_res.dyn_success_rate:.0%})"
                  f"  Gap={gap:+.1f}%\n")
        except Exception as exc:
            print(f"  [EVAL WARN] {exc}")


# ============================================================================
#  Env factory
# ============================================================================

def _make_flat_dynamic_env(targets_path, cloud_json_path,
                           duration_s, event_rate, seed,
                           render_mode=None) -> gym.Env:
    """Build env using RealVisionCloudModel (non-circular CNN pipeline)."""
    from env_dynamic_factory import Config, make_env as _factory_make
    env = _factory_make(
        cfg             = Config.DYN_REAL_VISION,
        targets_path    = targets_path,
        cloud_json_path = cloud_json_path,
        event_rate      = event_rate,
        duration_s      = duration_s,
        seed            = seed,
        render_mode     = render_mode,
    )
    return Monitor(env)


# ============================================================================
#  Training
# ============================================================================

def train(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(DEFAULT_MODEL_OUT), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    steps_per_ep     = int(args.duration / SCHED_STEP_S)
    total_steps      = args.episodes * steps_per_ep
    eval_every_steps = args.eval_every * steps_per_ep

    # Quick cloud model probe so banner shows real mode
    _cmode, _npatches = "unknown", "?"
    try:
        from env_dynamic_factory import Config, make_env as _pm
        _pe = _pm(cfg=Config.DYN_REAL_VISION, targets_path=args.targets,
                  cloud_json_path=args.cloud_json, event_rate=0.0,
                  duration_s=args.duration, seed=args.seed)
        _cmode    = getattr(getattr(_pe, '_cloud_model', None), 'mode', 'ok')
        _npatches = getattr(getattr(getattr(_pe, '_cloud_model', None),
                                    '_provider', None), 'n_patches', '?')
        _pe.close()
    except Exception as _ce:
        _cmode = f"err({_ce})"

    print("=" * 70)
    print("  ALSAT-EO-1  Phase 3  Dynamic Targeting PPO Training")
    print("=" * 70)
    print(f"  Cloud model: {_cmode}  ({_npatches} real MODIS patches)")
    print(f"  Event rate : {args.event_rate:.1f} events/hr")
    print(f"  Obs dim    : {OBS_TOTAL_DIM}  (43 base + 12 dynamic)")
    print(f"  Actions    : {N_TOTAL_ACTIONS}  (20 static + 3 dynamic + 1 drift)")
    print(f"  Episodes   : {args.episodes}   Duration: {args.duration:.0f}s")
    print(f"  Steps/ep   : {steps_per_ep}   Total: {total_steps:,}")
    print(f"  Eval every : {args.eval_every} eps  Log every: {args.log_every} eps\n")

    def _make():
        return _make_flat_dynamic_env(
            args.targets, args.cloud_json,
            args.duration, args.event_rate, args.seed)

    train_env = DummyVecEnv([_make])

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate = 3e-4,
        n_steps       = min(2048, steps_per_ep),
        batch_size    = 72,      # factor of n_steps=144; eliminates SB3 warning
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.02,   # slightly higher entropy to explore dynamic actions
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose       = 0,
        seed          = args.seed,
        device        = device,
    )
    print(f"Model device: {model.device}")

    logger_cb = DynamicTrainingLogger(
        targets_path    = args.targets,
        cloud_json_path = args.cloud_json,
        duration_s      = args.duration,
        event_rate      = args.event_rate,
        eval_every_steps = eval_every_steps,
        seed            = args.seed,
        verbose         = 1,
    )
    logger_cb._log_every = args.log_every
    logger_cb._save_freq = args.save_freq

    print("[PPO] Starting training...")
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_steps, callback=logger_cb,
                    progress_bar=True, reset_num_timesteps=True)
    except KeyboardInterrupt:
        print("\n  [INFO] Training interrupted by user.")



    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed/60:.1f} min")
    model.save(DEFAULT_MODEL_OUT)
    print(f"  Model saved → {DEFAULT_MODEL_OUT}")

    # ── Post-training RL evaluation (10 episodes) ─────────────────────────
    print("\n[EVAL] Evaluating trained RL agent (10 episodes)...")
    rl_rewards, rl_cf, rl_dyn_suc, rl_dyn_img, rl_dyn_det = [], [], [], [], []
    for s in range(10):
        eval_env = _make_flat_dynamic_env(
            args.targets, args.cloud_json,
            args.duration, args.event_rate, 1000 + s)
        obs_, _ = eval_env.reset(seed=1000 + s)
        ep_r, done = 0.0, False
        while not done:
            action, _ = model.predict(obs_, deterministic=True)
            obs_, r, term, trunc, info = eval_env.step(int(action))
            ep_r += r; done = term or trunc
        rl_rewards.append(ep_r)
        # Retrieve metrics from wrapped env
        try:
            sat = eval_env.env.unwrapped.satellites[0]
            m   = sat.get_metrics()
            cf  = m["n_cloud_free"] / m["n_imaged"] if m["n_imaged"] > 0 else 0.0
            dyn_m = eval_env.env.event_manager.get_metrics()
            rl_cf.append(cf)
            rl_dyn_suc.append(dyn_m["success_rate"])
            rl_dyn_img.append(dyn_m["n_imaged"])
            rl_dyn_det.append(dyn_m["n_detected"])
        except Exception:
            rl_cf.append(0.0); rl_dyn_suc.append(0.0)
            rl_dyn_img.append(0);  rl_dyn_det.append(0)
        eval_env.close()

    rl_stats = {
        "mean_reward":       float(np.mean(rl_rewards)),
        "std_reward":        float(np.std(rl_rewards)),
        "mean_cf_rate":      float(np.mean(rl_cf)),
        "mean_dyn_success":  float(np.mean(rl_dyn_suc)),
        "mean_dyn_detected": float(np.mean(rl_dyn_det)),
        "mean_dyn_imaged":   float(np.mean(rl_dyn_img)),
        "mean_delay_s":      0.0,   # not tracked here; see eval_dynamic.py
    }
    print(f"  RL  reward={rl_stats['mean_reward']:+.3f} ± {rl_stats['std_reward']:.3f}  "
          f"cf={rl_stats['mean_cf_rate']:.0%}  "
          f"dyn_suc={rl_stats['mean_dyn_success']:.0%}")

    # ── Baseline comparison ───────────────────────────────────────────────
    baseline_stats: dict = {}
    if HAS_BASELINES:
        print("\n[EVAL] Running greedy baselines (10 episodes each)...")
        eps_dyn = [run_greedy_dynamic_episode(
                       args.targets, args.cloud_json,
                       event_rate=args.event_rate,
                       duration_s=args.duration, seed=1000 + s)
                   for s in range(10)]
        eps_ign = [run_ignore_dynamic_episode(
                       args.targets, args.cloud_json,
                       event_rate=args.event_rate,
                       duration_s=args.duration, seed=1000 + s)
                   for s in range(10)]
        baseline_stats = {
            "greedy_dynamic_scout":  _aggregate_dynamic(eps_dyn),
            "greedy_ignore_dynamic": _aggregate_dynamic(eps_ign),
        }
        print_dynamic_table(baseline_stats, rl_stats=rl_stats,
                            event_rate=args.event_rate)

    # ── Save training log ─────────────────────────────────────────────────
    log = {
        "bsk_rl_version":    getattr(bsk_rl, "__version__", "unknown"),
        "event_rate":        args.event_rate,
        "obs_dim":           OBS_TOTAL_DIM,
        "n_actions":         N_TOTAL_ACTIONS,
        "total_steps":       total_steps,
        "n_episodes":        args.episodes,
        "duration_s":        args.duration,
        "elapsed_min":       round(elapsed / 60, 2),
        "seed":              args.seed,
        "episode_rewards":   logger_cb.ep_rewards,
        "episode_cf_rates":  logger_cb.ep_cf_rates,
        "episode_dyn_success": logger_cb.ep_dyn_success,
        "eval_results":      logger_cb.eval_results,
        "final_eval": {
            "rl":        rl_stats,
            "baselines": baseline_stats,
        },
        "ppo_params": {
            "lr": 3e-4, "n_steps": min(2048, steps_per_ep),
            "batch_size": 72, "n_epochs": 10, "gamma": 0.99,
        },
    }
    log_path = os.path.join(RESULTS_DIR, "phase3_dynamic_training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=float)
    print(f"  Log  → {log_path}")

    _plot_training(logger_cb, rl_stats, baseline_stats, args.event_rate)
    train_env.close()

    # ── Optional post-training 3-scenario eval ────────────────────────────
    if not args.no_eval:
        print("\n[EVAL] Running 3-scenario evaluation...")
        from evaluation.eval_dynamic import evaluate_all_scenarios, plot_scenario_comparison
        scenario_results = evaluate_all_scenarios(
            targets_path    = args.targets,
            cloud_json_path = args.cloud_json,
            n_episodes      = 3,
            seed            = 200,
            duration_s      = args.duration,
            model_path      = DEFAULT_MODEL_OUT,
            verbose         = True,
        )
        plot_scenario_comparison(scenario_results, PLOTS_DIR)

        sc_log_path = os.path.join(RESULTS_DIR, "phase3_scenario_eval.json")
        with open(sc_log_path, "w") as f:
            json.dump(scenario_results, f, indent=2, default=float)
        print(f"  Scenario results → {sc_log_path}")


# ============================================================================
#  Plots
# ============================================================================

def _plot_training(logger_cb: DynamicTrainingLogger,
                   rl_stats: dict, baseline_stats: dict,
                   event_rate: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"ALSAT-EO-1 Phase 3 — Dynamic Targeting PPO  "
        f"(event_rate={event_rate:.1f}/hr)", fontsize=12)

    ep_r   = logger_cb.ep_rewards
    ep_cf  = logger_cb.ep_cf_rates
    ep_dyn = logger_cb.ep_dyn_success
    evals  = logger_cb.eval_results

    def _smooth(arr, w=None):
        if not arr: return [], []
        w = w or max(1, len(arr) // 20)
        sm = np.convolve(arr, np.ones(w) / w, mode="valid")
        return np.arange(w - 1, len(arr)), sm

    # 1 — Learning curve
    ax = axes[0, 0]
    if ep_r:
        ax.plot(ep_r, color="steelblue", alpha=0.3, lw=0.7)
        xs, sm = _smooth(ep_r)
        ax.plot(xs, sm, color="darkblue", lw=2, label="MA reward")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
    ax.set_title("Learning Curve"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2 — Cloud-free rate
    ax = axes[0, 1]
    if ep_cf:
        ax.plot(ep_cf, color="mediumseagreen", alpha=0.3, lw=0.7)
        xs, sm = _smooth(ep_cf)
        ax.plot(xs, sm, color="darkgreen", lw=2, label="MA CF%")
    ax.axhline(0.65, color="orange", ls="--", lw=1.5, label="Target 65%")
    ax.set_ylim(0, 1); ax.set_xlabel("Episode"); ax.set_ylabel("CF rate")
    ax.set_title("Static Cloud-Free Rate"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3 — Dynamic success rate
    ax = axes[0, 2]
    if ep_dyn:
        ax.plot(ep_dyn, color="tomato", alpha=0.3, lw=0.7)
        xs, sm = _smooth(ep_dyn)
        ax.plot(xs, sm, color="darkred", lw=2, label="MA dyn success%")
    ax.axhline(0.5, color="purple", ls="--", lw=1.5, label="Target 50%")
    ax.set_ylim(0, 1); ax.set_xlabel("Episode"); ax.set_ylabel("Dynamic success rate")
    ax.set_title("Dynamic Event Imaging Rate"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4 — RL vs Greedy-dynamic eval curve
    ax = axes[1, 0]
    if evals:
        steps  = [e["step"] for e in evals]
        rl_v   = [e["rl_reward"] for e in evals]
        g_v    = [e["greedy_reward"] for e in evals]
        ax.plot(steps, rl_v, "o-", color="darkorange", lw=2, ms=5, label="RL-PPO")
        ax.plot(steps, g_v,  "s-", color="steelblue",  lw=2, ms=5, label="Greedy-dynamic")
        ax.fill_between(steps, g_v, rl_v,
                        where=[r >= g for r, g in zip(rl_v, g_v)],
                        alpha=0.2, color="green", label="RL > Scout")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Reward")
    ax.set_title("RL vs Greedy (live eval)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5 — Dynamic success: RL vs Greedy (live eval)
    ax = axes[1, 1]
    if evals:
        rl_ds  = [e.get("rl_dyn_success",    0) for e in evals]
        g_ds   = [e.get("greedy_dyn_success", 0) for e in evals]
        ax.plot(steps, rl_ds, "o-", color="darkorange", lw=2, ms=5, label="RL-PPO")
        ax.plot(steps, g_ds,  "s-", color="steelblue",  lw=2, ms=5, label="Greedy-dynamic")
    ax.set_ylim(0, 1); ax.set_xlabel("Timestep")
    ax.set_ylabel("Dyn success rate"); ax.set_title("Dynamic Success Rate (live eval)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 6 — Final comparison bar chart
    ax = axes[1, 2]
    pol_names  = list(baseline_stats.keys()) + ["RL-PPO"]
    pol_means  = [baseline_stats[p]["mean_reward"] for p in baseline_stats]
    pol_stds   = [baseline_stats[p]["std_reward"]  for p in baseline_stats]
    if rl_stats:
        pol_means.append(rl_stats["mean_reward"])
        pol_stds.append(rl_stats["std_reward"])
    colors = ["steelblue", "tomato", "darkorange"]
    x = np.arange(len(pol_names))
    bars = ax.bar(x, pol_means[:len(x)],
                  yerr=pol_stds[:len(x)],
                  color=colors[:len(x)], alpha=0.8, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in pol_names], fontsize=7)
    ax.set_ylabel("Mean total reward"); ax.set_title("Final Comparison")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "phase3_dynamic_training.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot → {path}")


# ============================================================================
#  CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ALSAT-EO-1 Phase 3 Dynamic Targeting PPO Training")
    ap.add_argument("--targets",     default=DEFAULT_TARGETS)
    ap.add_argument("--cloud-json",  default=DEFAULT_CLOUD_JSON)
    ap.add_argument("--episodes",    type=int,   default=500,
                    help="Training episodes (default 500)")
    ap.add_argument("--event-rate",  type=float, default=2.0,
                    help="Dynamic events per hour (default 2.0)")
    ap.add_argument("--duration",    type=float, default=SIM_DURATION_S,
                    help="Episode duration in seconds (default 172800 = 48h)")
    ap.add_argument("--eval-every",  type=int,   default=50,
                    help="Run live greedy eval every N episodes")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--no-eval",     action="store_true",
                    help="Skip 3-scenario evaluation after training")
    ap.add_argument("--log-every",   type=int,   default=1,
                    help="Print episode summary every N episodes (default 1 = every ep)")
    ap.add_argument("--save-freq",   type=int,   default=100,
                help="Save checkpoint every N episodes (default 100)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
