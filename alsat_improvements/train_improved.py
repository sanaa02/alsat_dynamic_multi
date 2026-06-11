#!/usr/bin/env python3
"""
train_improved.py  --  ALSAT-EO-1  Improved PPO Training Script
================================================================
Drop-in replacement for train_ppo_dynamic.py / train_ppo_smdp_full.py.

Key improvements vs. current training:

  FIX-1  Entropy collapse fixed: start_ent=0.15, anneal to 0.01 over 80%
         of training via EntropyAnnealCallback (was stuck at 0.0500).

  FIX-2  Larger rollout buffer: n_steps=1024 (was 144).
         With n_envs=4 → 4096 samples per PPO update (was 144).

  FIX-3  Parallel environments: n_envs=4 via DummyVecEnv or SubprocVecEnv.
         4× more diverse experience per episode wall-clock second.

  FIX-4  VecNormalize: normalises obs (position in 7M m, cloud in [0,1])
         and reward. This alone typically improves stability by 20–30%.

  FIX-5  Linear LR decay: 3e-4 → 1e-5 over full training.

  FIX-6  Multi-seed evaluation: evaluates on 3 seeds × 10 episodes every
         50K steps, matching Kangaslahti et al. (2024) methodology.

  FIX-7  Best model saving: saves best_model.zip on eval improvement.

Usage:
    python train_improved.py \\
        --targets config/targets/algeria_20_targets.json \\
        --cloud   config/cloud_reality/algeria_real_clouds.json \\
        --episodes 2000 \\
        --event-rate 2.0 \\
        --n-envs 4 \\
        --seed 42

To resume from a checkpoint:
    python train_improved.py ... --resume models/checkpoints_improved/ppo_ep00500.zip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging

import numpy as np

# ── Path setup (same as existing scripts) ────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in [
    os.path.join(_HERE, "..", "scripts", "core"),
    os.path.join(_HERE, "..", "scripts", "training"),
    os.path.join(_HERE, "..", "scripts", "wrappers"),
    os.path.join(_HERE, "..", "scripts"),
]:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import path_setup
    ROOT = path_setup.root_path()
except ImportError:
    ROOT = os.path.join(_HERE, "..")

# Silence BSK noise early
import logging as _logging
_BSK_NOISE = frozenset([
    "Creating logger for new env",
    "Old environments in process",
    "basePowerDraw should probably be zero or negative",
    "Could not find eclipse transitions",
    "initial_generation_duration is shorter than the maximum window length",
])
_orig_ch = _logging.Logger.callHandlers
def _quiet(self, record):
    try:
        if any(s in record.getMessage() for s in _BSK_NOISE): return
    except Exception: pass
    return _orig_ch(self, record)
_logging.Logger.callHandlers = _quiet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Stable-Baselines3 + torch ─────────────────────────────────────────────────
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    _HAS_SUBPROC = True
except ImportError:
    _HAS_SUBPROC = False

# ── BSK patches (must run before any bsk_rl import) ──────────────────────────
try:
    import bsk_patches
    bsk_patches.apply_all()
    logger.info("[train_improved] bsk_patches applied")
except ImportError:
    logger.warning("[train_improved] bsk_patches not found — skipping")

# ── Local imports ─────────────────────────────────────────────────────────────
from env_alsat_debug import SCHED_STEP_S, SIM_DURATION_S

try:
    from env_dynamic_factory import make_env as _factory_make_env, Config, obs_dim, n_actions
    _HAS_FACTORY = True
    logger.info("[train_improved] env_dynamic_factory available")
except ImportError:
    _HAS_FACTORY = False
    from env_alsat_dynamic import make_dynamic_env as _legacy_make_env

try:
    from callbacks_improved import build_callback_list
    _HAS_CALLBACKS = True
except ImportError:
    _HAS_CALLBACKS = False
    logger.warning("[train_improved] callbacks_improved.py not found — using basic logger")

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_TARGETS    = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
DEFAULT_CLOUD_JSON = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")
DEFAULT_MODELS_DIR = os.path.join(ROOT, "models")
DEFAULT_RESULTS    = os.path.join(ROOT, "results")


# =============================================================================
#  Linear LR schedule
# =============================================================================

def linear_schedule(initial_value: float, final_value: float = 1e-5):
    """Returns a callable schedule: f(progress_remaining) → lr."""
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


# =============================================================================
#  Environment factory
# =============================================================================

def make_single_env(
    targets_path: str,
    cloud_path:   str,
    event_rate:   float,
    duration_s:   float,
    seed:         int,
    with_safety:  bool  = True,
    cnn_path:     Optional[str] = None,
) -> "gym.Env":
    """Returns a single (non-vectorised) Monitor-wrapped env."""
    if _HAS_FACTORY:
        env = _factory_make_env(
            cfg           = Config.DYN_REAL_VISION if cnn_path and os.path.exists(cnn_path)
                            else Config.DYN_BASE,
            targets_path  = targets_path,
            cloud_json_path = cloud_path,
            event_rate    = event_rate,
            duration_s    = duration_s,
            seed          = seed,
            with_safety   = with_safety,
            cnn_path      = cnn_path or "",
        )
    else:
        env = _legacy_make_env(
            targets_path  = targets_path,
            cloud_json_path = cloud_path,
            event_rate    = event_rate,
            duration_s    = duration_s,
            seed          = seed,
        )
    return Monitor(env)


from typing import Optional


# =============================================================================
#  Build PPO model
# =============================================================================

def build_ppo(
    vec_env,
    args,
    total_timesteps: int,
    steps_per_ep:    int,
) -> PPO:
    """
    Build the improved PPO model.

    Key differences from current train_ppo_dynamic.py:
    - n_steps = 1024 (was 144) → larger rollout buffer
    - batch_size = 512 (was 72) → proper minibatch size
    - gamma = 0.995 (was 0.99) → better late-episode coverage
    - learning_rate = linear decay (was constant 3e-4)
    - ent_coef = 0.15 initial (was 0.03–0.05, immediately collapsed)
    """
    n_steps    = max(1024, steps_per_ep)   # At least 1024, or 1 full episode
    n_envs     = vec_env.num_envs
    n_samples  = n_steps * n_envs          # Total samples per update
    batch_size = min(512, n_samples // 4)  # ~25% of rollout per minibatch

    logger.info(
        f"[build_ppo] n_steps={n_steps}  n_envs={n_envs}  "
        f"n_samples={n_samples}  batch_size={batch_size}"
    )

    model = PPO(
        policy         = "MlpPolicy",
        env            = vec_env,
        learning_rate  = linear_schedule(3e-4, 1e-5),   # FIX-5: LR decay
        n_steps        = n_steps,                        # FIX-2: larger buffer
        batch_size     = batch_size,
        n_epochs       = 10,
        gamma          = 0.995,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.15,   # FIX-1: high initial entropy (annealed by callback)
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        policy_kwargs  = dict(
            net_arch       = dict(pi=[256, 256], vf=[256, 256]),
            activation_fn  = torch.nn.Tanh,
        ),
        verbose        = 0,
        seed           = args.seed,
        device         = "cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log= os.path.join(DEFAULT_RESULTS, "tb_logs"),
    )

    logger.info(f"[build_ppo] Device: {model.device}")
    return model


# =============================================================================
#  Main training function
# =============================================================================

def train(args: argparse.Namespace) -> None:
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RESULTS,    exist_ok=True)

    steps_per_ep     = int(args.duration / SCHED_STEP_S)
    total_timesteps  = args.episodes * steps_per_ep

    logger.info("=" * 70)
    logger.info("  ALSAT-EO-1  Improved PPO Training")
    logger.info("=" * 70)
    logger.info(f"  Episodes   : {args.episodes}   Steps/ep: {steps_per_ep}")
    logger.info(f"  Total steps: {total_timesteps:,}")
    logger.info(f"  n_envs     : {args.n_envs}")
    logger.info(f"  Event rate : {args.event_rate:.1f} events/hr")
    logger.info(f"  Seed       : {args.seed}")
    logger.info(f"  Duration   : {args.duration:.0f}s ({args.duration/3600:.1f}h)")
    logger.info("")

    # ── Build vectorised environment ─────────────────────────────────────────
    def _make_env_fn(rank: int):
        def _fn():
            return make_single_env(
                targets_path = args.targets,
                cloud_path   = args.cloud,
                event_rate   = args.event_rate,
                duration_s   = args.duration,
                seed         = args.seed + rank * 1000,
                with_safety  = not args.no_safety,
                cnn_path     = getattr(args, "cnn_model", None),
            )
        return _fn

    # FIX-3: Parallel environments
    if args.n_envs > 1 and _HAS_SUBPROC:
        logger.info(f"  Using SubprocVecEnv with {args.n_envs} processes")
        vec = SubprocVecEnv([_make_env_fn(i) for i in range(args.n_envs)])
    else:
        if args.n_envs > 1:
            logger.info(f"  SubprocVecEnv not available; using DummyVecEnv({args.n_envs})")
        vec = DummyVecEnv([_make_env_fn(i) for i in range(args.n_envs)])

    # FIX-4: VecNormalize — crucial for mixed-scale observations
    # IMPORTANT: Save the VecNormalize stats alongside the model so evaluation
    # can reconstruct the same normalization.
    vecnorm_path = os.path.join(DEFAULT_MODELS_DIR, "vec_normalize.pkl")
    vec = VecNormalize(
        vec,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        clip_reward = 10.0,
        gamma       = 0.995,
    )
    logger.info("  VecNormalize: obs=True  reward=True  clip=10.0")

    # ── Build PPO model ───────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        logger.info(f"  Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = build_ppo(vec, args, total_timesteps, steps_per_ep)

    # ── Build callbacks ───────────────────────────────────────────────────────
    if _HAS_CALLBACKS:
        callbacks = build_callback_list(
            model_dir        = DEFAULT_MODELS_DIR,
            targets_path     = args.targets,
            cloud_path       = args.cloud,
            total_episodes   = args.episodes,
            total_timesteps  = total_timesteps,
            event_rate       = args.event_rate,
            seed             = args.seed,
            duration_s       = args.duration,
            start_ent        = 0.15,
            end_ent          = 0.01,
            decay_fraction   = 0.80,
            eval_every_ts    = max(50_000, steps_per_ep * 50),
            eval_seeds       = (42, 123, 456),
            n_eval_per_seed  = 10,
        )
    else:
        # Fallback: basic callback
        from stable_baselines3.common.callbacks import CheckpointCallback
        callbacks = CheckpointCallback(
            save_freq  = steps_per_ep * 100,
            save_path  = os.path.join(DEFAULT_MODELS_DIR, "checkpoints_improved"),
            name_prefix= "ppo_improved",
        )

    # ── Training ─────────────────────────────────────────────────────────────
    logger.info(f"\n[PPO] Starting training for {total_timesteps:,} steps...")
    t0 = time.time()
    try:
        model.learn(
            total_timesteps     = total_timesteps,
            callback            = callbacks,
            progress_bar        = True,
            reset_num_timesteps = not bool(args.resume),
        )
    except KeyboardInterrupt:
        logger.info("\n  Training interrupted by user — saving current model...")

    elapsed = time.time() - t0
    logger.info(f"\n  Training complete in {elapsed/60:.1f} min")

    # ── Save final model + normalizer ─────────────────────────────────────────
    final_path = os.path.join(DEFAULT_MODELS_DIR, "ppo_improved_final.zip")
    model.save(final_path)
    vec.save(vecnorm_path)
    logger.info(f"  Model   → {final_path}")
    logger.info(f"  VecNorm → {vecnorm_path}")

    # ── Quick post-training evaluation ───────────────────────────────────────
    _quick_eval(model, args, n_episodes=20)

    vec.close()


def _quick_eval(model, args, n_episodes: int = 20):
    """Run a quick post-training evaluation on a fresh (non-normalised) env."""
    logger.info(f"\n[EVAL] Post-training evaluation ({n_episodes} episodes)...")

    rewards   = []
    dyn_rates = []
    cf_rates  = []

    for ep in range(n_episodes):
        try:
            env  = make_single_env(
                targets_path = args.targets,
                cloud_path   = args.cloud,
                event_rate   = args.event_rate,
                duration_s   = args.duration,
                seed         = 8000 + ep,
                with_safety  = not args.no_safety,
            )
            obs, _ = env.reset(seed=8000 + ep)
            done, ep_r = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(int(action))
                ep_r += r
                done  = term or trunc
            rewards.append(ep_r)

            try:
                sat   = env.env.unwrapped.satellites[0]
                m     = sat.get_metrics()
                cf    = m["n_cloud_free"] / m["n_imaged"] if m["n_imaged"] > 0 else 0.0
                cf_rates.append(cf)
                dyn_m = env.event_manager.get_metrics()
                dyn_rates.append(dyn_m.get("success_rate", 0.0))
            except Exception:
                cf_rates.append(0.0); dyn_rates.append(0.0)
            env.close()
        except Exception as exc:
            logger.warning(f"  Eval ep {ep} failed: {exc}")

    if rewards:
        logger.info(
            f"  Mean reward  : {np.mean(rewards):+.3f} ± {np.std(rewards):.3f}\n"
            f"  Cloud-free   : {np.mean(cf_rates):.0%} ± {np.std(cf_rates):.0%}\n"
            f"  Dynamic suc  : {np.mean(dyn_rates):.0%} ± {np.std(dyn_rates):.0%}"
        )

    result_path = os.path.join(DEFAULT_RESULTS, "improved_eval.json")
    with open(result_path, "w") as f:
        json.dump({
            "n_episodes":   n_episodes,
            "mean_reward":  float(np.mean(rewards))    if rewards   else 0.0,
            "std_reward":   float(np.std(rewards))     if rewards   else 0.0,
            "mean_cf_rate": float(np.mean(cf_rates))   if cf_rates  else 0.0,
            "mean_dyn_suc": float(np.mean(dyn_rates))  if dyn_rates else 0.0,
        }, f, indent=2)
    logger.info(f"  Results → {result_path}")


# =============================================================================
#  CLI
# =============================================================================

def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ALSAT-EO-1 Improved PPO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--targets",     default=DEFAULT_TARGETS,
                    help="Path to targets JSON")
    ap.add_argument("--cloud",       default=DEFAULT_CLOUD_JSON,
                    help="Path to cloud JSON")
    ap.add_argument("--episodes",    type=int,   default=2000,
                    help="Training episodes (recommend 2000+)")
    ap.add_argument("--event-rate",  type=float, default=2.0,
                    help="Dynamic events per hour")
    ap.add_argument("--duration",    type=float, default=SIM_DURATION_S,
                    help="Episode duration (seconds)")
    ap.add_argument("--n-envs",      type=int,   default=4,
                    help="Number of parallel environments (recommend 4)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--resume",      type=str,   default=None,
                    help="Path to .zip checkpoint to resume from")
    ap.add_argument("--no-safety",   action="store_true",
                    help="Disable SafetyMonitor")
    ap.add_argument("--cnn-model",   type=str,   default=None,
                    help="Path to trained CNN weights (.pt)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(args)
