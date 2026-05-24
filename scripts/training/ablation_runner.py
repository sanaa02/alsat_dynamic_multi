#!/usr/bin/env python3
"""
ablation_runner.py  --  6-Variant Ablation Study for ALSAT-EO-1
===============================================================
Runs all ablation variants with multiple seeds and produces:
  - results/ablation/{variant}/{seed}/training_log.json
  - results/ablation/{variant}/{seed}/eval_metrics.json
  - results/ablation/ablation_table.csv
  - results/ablation/ablation_plot.png

Variants
--------
  A  full_system      BC + Curriculum + SMDP + real-vision CNN + mask + domrand
  B  no_bc            remove BC pretraining
  C  no_curriculum    remove curriculum warm-up
  D  no_smdp          DYN_MODIS (Gaussian cloud, no τ feature)
  E  gaussian_cloud   DYN_MODIS (Gaussian noise cloud, keep SMDP)
  F  circular_cnn     DYN_VISION (synthetic patches = circular dependency)

Usage
-----
    python -m scripts.training.ablation_runner --seeds 42 123 456 789 999 --episodes 2000
    python -m scripts.training.ablation_runner --variant full_system --seeds 42 123
"""
from __future__ import annotations

import os, sys, argparse, json, time, logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np
import logging, warnings


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_setup
ROOT = path_setup.root_path()
for _d in ["scripts/core", "scripts/training", "scripts/wrappers", "scripts"]:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Silence bsk_rl noise ─────────────────────────────────────────────────
# IMPORTANT: only use setLevel — NEVER add handlers to bsk_rl loggers
# (bsk_rl._configure_logging assumes all handlers have a filters[0] attribute)
logging.getLogger("bsk_rl").setLevel(logging.ERROR)
logging.getLogger("bsk_rl.sats.access_satellite").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*initial_generation_duration.*")
warnings.filterwarnings("ignore", message=".*nvrtc.*")
warnings.filterwarnings("ignore", message=".*CuDNN.*")
warnings.filterwarnings("ignore", message=".*GPU.*MLP.*")
# ─────────────────────────────────────────────────────────────────────────
TARGETS  = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
CLOUD    = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")
CNN_FP32 = os.path.join(ROOT, "models/cloud_cnn_real.pt")
CNN_INT8 = os.path.join(ROOT, "models/cloud_cnn_real_int8.ts")
MODELS   = os.path.join(ROOT, "models")
ABLATION = os.path.join(ROOT, "results/ablation")


@dataclass
class AblationConfig:
    name:           str
    description:    str
    use_bc:         bool  = True
    use_curriculum: bool  = True
    use_smdp:       bool  = True   # True = DYN_REAL_VISION, False = DYN_MODIS
    use_cnn:        str   = "real" # "real" | "gaussian" | "circular"
    use_mask:       bool  = True
    use_domrand:    bool  = True
    ent_coef_start: float = 0.15   # was 0.05 — fixed to match callbacks.py default
    ent_coef_end:   float = 0.01   # was 0.005 — fixed
    use_attention:  bool  = False  # True = SchedulerAttentionExtractor vs MLP
    dynamic_bonus:  float = 3.0   # passed via env_kwargs


VARIANTS: dict[str, AblationConfig] = {
    "full_system": AblationConfig(
        name="full_system",
        description="Complete pipeline — BC + Curriculum + SMDP + real-vision CNN",
    ),
    "no_bc": AblationConfig(
        name="no_bc",
        description="Remove BC pretraining — isolate BC contribution",
        use_bc=False,
    ),
    "no_curriculum": AblationConfig(
        name="no_curriculum",
        description="Remove curriculum warm-up — direct PPO from random init",
        use_curriculum=False,
    ),
    "no_smdp": AblationConfig(
        name="no_smdp",
        description="Fixed-step MDP with Gaussian cloud — proves SMDP adds value",
        use_smdp=False, use_cnn="gaussian",
    ),
    "gaussian_cloud": AblationConfig(
        name="gaussian_cloud",
        description="DYN_MODIS config — Gaussian noise cloud model baseline",
        use_cnn="gaussian",
    ),
    "circular_cnn": AblationConfig(
        name="circular_cnn",
        description="Circular CNN — synthetic patches encode ground truth (prior work baseline)",
        use_cnn="circular",
    ),
    "attention_policy": AblationConfig(
    name="attention_policy",
    description="Attention-based policy (SchedulerAttentionExtractor, ~200K params) "
                "vs MLP baseline — isolates policy architecture contribution",
    use_attention=True,  # everything else same as full_system
    ),
}


def run_variant(cfg: AblationConfig, seed: int, episodes: int, n_envs: int = 2) -> dict:
    """Train one (variant, seed) pair and return metrics dict."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CallbackList
    from env_dynamic_factory import Config, make_env, obs_dim, n_actions
    from env_alsat_debug import SCHED_STEP_S, SIM_DURATION_S
    from callbacks import (EntropyAnnealingCallback, DynamicEventCallback,
                           AutoCheckpointCallback)
    from reward_shaping import DynamicRewardShaper

    # ── Config → Config enum ───────────────────────────────────────────────
    if cfg.use_cnn == "gaussian" or not cfg.use_smdp:
        ppo_cfg = Config.DYN_MODIS
    elif cfg.use_cnn == "circular":
        ppo_cfg = Config.DYN_VISION
    else:
        ppo_cfg = Config.DYN_REAL_VISION

    cnn_path = CNN_FP32 if cfg.use_cnn in ("real", "circular") else None

    out_dir = os.path.join(ABLATION, cfg.name, f"seed_{seed}")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(out_dir, "ablation_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    steps_per_ep = int(SIM_DURATION_S / SCHED_STEP_S)
    total_steps  = episodes * steps_per_ep

    def _make_env(seed_offset: int = 0):
        def _factory():
            env = make_env(
                ppo_cfg, TARGETS, CLOUD,
                event_rate=2.0,
                duration_s=SIM_DURATION_S,
                seed=seed + seed_offset,
                with_safety=True,
                cnn_path=cnn_path,
                with_action_mask=cfg.use_mask,
                with_domain_rand=cfg.use_domrand,
            )
            env = DynamicRewardShaper(env, urgency_scale=3.0, explore_bonus_init=0.3)
            return Monitor(env)
        return _factory

    if n_envs > 1:
        vec = SubprocVecEnv([_make_env(i) for i in range(n_envs)])
    else:
        vec = DummyVecEnv([_make_env(0)])

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _n_steps = max(128, 2048 // n_envs)

    if cfg.use_attention:
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'models'))
            from attention_policy import make_attention_ppo
            model = make_attention_ppo(
                vec, ent_coef=cfg.ent_coef_start,
                n_steps=_n_steps, batch_size=64,
                seed=seed, device=_device,
            )
            print(f" [{cfg.name}|s{seed}] Using attention policy")
        except ImportError:
            print(f" [WARN] attention_policy not found, using MLP")
            cfg.use_attention = False  # fall back

    if not cfg.use_attention:
        model = PPO(
            "MlpPolicy", vec,
            learning_rate=3e-4, n_steps=_n_steps,
            batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            ent_coef=cfg.ent_coef_start,
            vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
            verbose=0, seed=seed, device=_device,
        )

    # ── Stages ────────────────────────────────────────────────────────────
    if cfg.use_bc:
        logger.info(f"  [{cfg.name}|s{seed}] Stage 1: BC pretraining")
        demo_path = os.path.join(ROOT, "data/demos.npz")
        try:
            from bc_pretrain import behavioral_cloning
            data = np.load(demo_path)
            behavioral_cloning(model, data["obs"], data["actions"],
                               n_epochs=50, verbose=False)
            logger.info(f"  [{cfg.name}|s{seed}] BC done")
        except Exception as e:
            logger.warning(f"  BC failed: {e}")

    if cfg.use_curriculum:
        logger.info(f"  [{cfg.name}|s{seed}] Stage 2: Curriculum")
        try:
            from curriculum import CurriculumScheduler
            sched = CurriculumScheduler(verbose=False)
            for ep in range(200):
                env_c = sched.make_env(TARGETS, CLOUD, seed=seed + ep,
                                       use_smdp=False, cfg=ppo_cfg,
                                       with_safety=True)
                obs, _ = env_c.reset(seed=seed + ep)
                done, ep_r = False, 0.0
                while not done:
                    act, _ = model.predict(obs, deterministic=False)
                    obs, r, t, tr, _ = env_c.step(int(act))
                    ep_r += r; done = t or tr
                env_c.close()
                sched.maybe_advance(ep_r)
                model.learn(total_timesteps=steps_per_ep, reset_num_timesteps=False)
        except Exception as e:
            logger.warning(f"  Curriculum failed: {e}")

    # ── PPO Main ──────────────────────────────────────────────────────────
    logger.info(f"  [{cfg.name}|s{seed}] Stage 3: SMDP-PPO ({episodes} eps)")
    ent_cb  = EntropyAnnealingCallback(
        start_val=cfg.ent_coef_start, end_val=cfg.ent_coef_end,
        total_timesteps=total_steps, verbose=0
    )
    dyn_cb  = DynamicEventCallback(log_every=50, verbose=1)
    ckpt_cb = AutoCheckpointCallback(
        save_freq=max(50_000, total_steps // 10),
        save_dir=ckpt_dir, exp_id=f"{cfg.name}_s{seed}",
        extra_meta=asdict(cfg), verbose=0,
    )
    cb_list = CallbackList([ent_cb, dyn_cb, ckpt_cb])

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb_list,
                reset_num_timesteps=True, progress_bar=False)
    elapsed = time.time() - t0

    model_path = os.path.join(MODELS, f"ppo_{cfg.name}_s{seed}.zip")
    model.save(model_path)
    vec.close()

    # ── Evaluate ──────────────────────────────────────────────────────────
    logger.info(f"  [{cfg.name}|s{seed}] Evaluating 30 episodes...")
    eval_metrics = _evaluate(model, ppo_cfg, cnn_path, cfg, seed, n_episodes=30)
    eval_metrics["elapsed_min"] = round(elapsed / 60, 2)
    eval_metrics["model_path"]  = model_path
    eval_metrics["dyn_success_history"] = dyn_cb.dyn_success_history[-100:]

    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2, default=float)

    logger.info(
        f"  [{cfg.name}|s{seed}] Done  "
        f"cf_rate={eval_metrics.get('cf_rate', 0):.1%}  "
        f"dyn_suc={eval_metrics.get('dyn_suc', 0):.1%}  "
        f"reward={eval_metrics.get('mean_reward', 0):+.2f}"
    )
    return eval_metrics


def _evaluate(model, cfg_enum, cnn_path, ablation_cfg, seed, n_episodes=30):
    from env_dynamic_factory import make_env

    env = make_env(
        cfg_enum, TARGETS, CLOUD,
        event_rate=2.0, seed=seed + 9999,
        with_safety=True, cnn_path=cnn_path,
        with_action_mask=ablation_cfg.use_mask,
        with_domain_rand=False,   # no randomisation during eval
    )
    rewards, cf_rates, dyn_suc_rates = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 9999 + ep)
        done, ep_r = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, t, tr, info = env.step(int(act))
            ep_r += r; done = t or tr
        m  = info.get("episode_metrics", {})
        ni = m.get("n_imaged", 0)
        nc = m.get("n_cloud_free", 0)
        nd = m.get("n_dyn_detected", 0)
        ndi = m.get("n_dyn_imaged", 0)
        rewards.append(ep_r)
        cf_rates.append(nc / ni if ni > 0 else 0.0)
        dyn_suc_rates.append(ndi / nd if nd > 0 else 0.0)

    env.close()
    return {
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "cf_rate":         float(np.mean(cf_rates)),
        "cf_rate_std":     float(np.std(cf_rates)),
        "dyn_suc":         float(np.mean(dyn_suc_rates)),
        "dyn_suc_std":     float(np.std(dyn_suc_rates)),
        "n_eval_episodes": n_episodes,
    }


def aggregate_results() -> None:
    """Read all eval_metrics.json files and write ablation_table.csv + plot."""
    import csv
    rows = []
    for variant in VARIANTS:
        var_dir = os.path.join(ABLATION, variant)
        if not os.path.isdir(var_dir):
            continue
        seed_metrics = []
        for seed_dir in os.listdir(var_dir):
            p = os.path.join(var_dir, seed_dir, "eval_metrics.json")
            if os.path.exists(p):
                with open(p) as f:
                    seed_metrics.append(json.load(f))
        if not seed_metrics:
            continue
        mean_r   = np.mean([m["mean_reward"] for m in seed_metrics])
        std_r    = np.std( [m["mean_reward"] for m in seed_metrics])
        mean_cf  = np.mean([m["cf_rate"]     for m in seed_metrics])
        std_cf   = np.std( [m["cf_rate"]     for m in seed_metrics])
        mean_ds  = np.mean([m["dyn_suc"]     for m in seed_metrics])
        std_ds   = np.std( [m["dyn_suc"]     for m in seed_metrics])
        rows.append({
            "variant": variant,
            "n_seeds": len(seed_metrics),
            "reward_mean": round(mean_r, 3),  "reward_std": round(std_r, 3),
            "cf_rate_mean": round(mean_cf, 4), "cf_rate_std": round(std_cf, 4),
            "dyn_suc_mean": round(mean_ds, 4), "dyn_suc_std": round(std_ds, 4),
        })

    csv_path = os.path.join(ABLATION, "ablation_table.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"\n✓ Ablation table → {csv_path}")
        # Pretty print
        print(f"\n{'Variant':<18} {'Reward':>10} {'CF rate':>10} {'DynSuc':>10}")
        print("-" * 54)
        for r in rows:
            print(f"  {r['variant']:<16} "
                  f"{r['reward_mean']:>7.3f}±{r['reward_std']:.2f}  "
                  f"{r['cf_rate_mean']:>6.3f}±{r['cf_rate_std']:.3f}  "
                  f"{r['dyn_suc_mean']:>6.3f}±{r['dyn_suc_std']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",    type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--n-envs",   type=int, default=2)
    ap.add_argument("--variant",  type=str, default=None,
                    choices=list(VARIANTS) + [None],
                    help="Run single variant (default: all)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Just aggregate existing results without training")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate_results(); exit(0)

    to_run = [args.variant] if args.variant else list(VARIANTS)

    for variant_name in to_run:
        cfg = VARIANTS[variant_name]
        for seed in args.seeds:
            out = os.path.join(ABLATION, cfg.name, f"seed_{seed}", "eval_metrics.json")
            if os.path.exists(out):
                print(f"  Skip {variant_name} seed={seed} (already done)")
                continue
            print(f"\n{'='*60}")
            print(f"  {variant_name}  seed={seed}")
            print(f"{'='*60}")
            try:
                run_variant(cfg, seed=seed, episodes=args.episodes, n_envs=args.n_envs)
            except Exception as e:
                logger.error(f"Variant {variant_name} seed {seed} FAILED: {e}", exc_info=True)

    aggregate_results()
