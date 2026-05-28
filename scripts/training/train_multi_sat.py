#!/usr/bin/env python3
"""
train_multi_sat.py  --  Multi-Satellite MAPPO Training (Parameter Sharing)
=========================================================================
Trains N satellites sharing a single PPO policy via MultiSatelliteCoordVecEnv.

Key design decisions (aligned with Yu et al. 2022 MAPPO)
---------------------------------------------------------
  1. Parameter sharing: all satellites use the same MLP weights
  2. Observation augmentation: claim bitmap + other-satellite summary
  3. Claim registry: prevents reward double-counting across satellites
  4. Larger network [256,256] to handle augmented obs (80-dim for 2 sats)
  5. Same BC + Curriculum pipeline as single-satellite (applied per-satellite)

Usage
-----
    cd /home/sanaa/alsat_dynamic_improved
    CUDA_VISIBLE_DEVICES=1 python -m scripts.training.train_multi_sat \
        --n-satellites 2 --episodes 2000 --seed 42 --bc --curriculum

Citation: Yu et al. 2022 "The Surprising Effectiveness of IPPO in MARL"
"""
from __future__ import annotations

import os, sys, argparse, json, time, logging
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_setup
ROOT = path_setup.root_path()
for _d in ["scripts/core", "scripts/training", "scripts/wrappers", "scripts"]:
    _p = os.path.join(ROOT, _d); 
    if _p not in sys.path: sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

TARGETS = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
CLOUD   = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")
CNN_PT  = os.path.join(ROOT, "models/cloud_cnn_real.pt")
MODELS  = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")


def make_sat_env(sat_idx: int, seed: int, with_mask: bool, with_domrand: bool):
    """Factory for a single satellite's environment."""
    from env_dynamic_factory import Config, make_env
    from env_alsat_debug import SIM_DURATION_S
    from reward_shaping import DynamicRewardShaper
    from stable_baselines3.common.monitor import Monitor

    env = make_env(
        Config.DYN_REAL_VISION, TARGETS, CLOUD,
        event_rate=2.0,
        duration_s=SIM_DURATION_S,
        seed=seed,
        with_safety=True,
        cnn_path=CNN_PT,
        with_action_mask=with_mask,
        with_domain_rand=with_domrand,
    )
    env = DynamicRewardShaper(env, urgency_scale=3.0, explore_bonus_init=0.3)
    return Monitor(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-satellites", type=int, default=2)
    ap.add_argument("--episodes",     type=int, default=2000)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--event-rate",   type=float, default=2.0)
    ap.add_argument("--bc",           action="store_true")
    ap.add_argument("--curriculum",   action="store_true")
    ap.add_argument("--action-mask",  action="store_true")
    ap.add_argument("--domain-rand",  action="store_true")
    ap.add_argument("--eval",         action="store_true")
    ap.add_argument("--ent-coef",     type=float, default=0.05)
    ap.add_argument("--ent-coef", type=float, default=0.15)  # fix from 0.05
    ap.add_argument("--init-model", type=str, default=None,
                    help="Pre-trained single-sat model for warm-start (recommended!)")
    ap.add_argument("--cnn-model", type=str, default=None,
                    help="Path to cloud CNN model")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from env_alsat_debug import SCHED_STEP_S, SIM_DURATION_S
    from env_dynamic_factory import Config, obs_dim
    from env_multi_satellite import MultiSatelliteCoordVecEnv
    from callbacks import (EntropyAnnealingCallback, DynamicEventCallback,
                           AutoCheckpointCallback)

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    N    = args.n_satellites
    # Augmented obs dim: 56 base + 20 claim bitmap + 4*(N-1) other summary
    AUG_OBS = 56 + 20 + 4 * (N - 1)

    print(f"\n{'='*60}")
    print(f"  Multi-Satellite MAPPO  N={N}  obs_aug={AUG_OBS}  device={_device}")
    print(f"{'='*60}")

    def _make_fn(sat_idx: int, seed: int):
        return make_sat_env(sat_idx, seed, args.action_mask, args.domain_rand)

    vec = MultiSatelliteCoordVecEnv(
        make_env_fn=_make_fn, n_satellites=N, seed=args.seed
    )

    steps_per_ep = int(SIM_DURATION_S / SCHED_STEP_S)
    total_steps  = args.episodes * steps_per_ep

    # ── Warm-start from single-satellite pretrained policy ──────────
    # Copies MLP weights (actor/critic). Ignores obs dim mismatch via strict=False.
    # Multi-sat obs is 56 + 20 (claim bitmap) + 4*(N-1) — extra dims init to random.
    if args.init_model and os.path.exists(args.init_model):
        try:
            from stable_baselines3 import PPO as _PPO
            _src = _PPO.load(args.init_model)
            # Copy overlapping parameters (actor/critic MLP layers)
            state_dict = _src.policy.state_dict()
            current_keys = set(model.policy.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items() if k in current_keys
                        and model.policy.state_dict()[k].shape == v.shape}
            model.policy.load_state_dict(filtered, strict=False)
            n_copied = len(filtered)
            n_total = len(state_dict)
            print(f" [WARM-START] Copied {n_copied}/{n_total} param tensors "
                  f"from {args.init_model}")
        except Exception as _e:
            print(f" [WARN] Warm-start failed: {_e} — training from scratch")

    print(f"  Policy params: "
          f"{sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"  Total training steps: {total_steps:,}")

    # ── BC Pretraining (optional) ─────────────────────────────────────────
    if args.bc:
        demo_path = os.path.join(ROOT, "data/demos.npz")
        if os.path.exists(demo_path):
            logger.info("Stage 1: BC pretraining (multi-sat uses single-sat demos)")
            try:
                from bc_pretrain import behavioral_cloning
                data = np.load(demo_path)
                # Pad obs to augmented dim (fill extra with zeros)
                obs_bc  = data["obs"]
                pad_len = AUG_OBS - obs_bc.shape[1]
                if pad_len > 0:
                    obs_bc = np.concatenate(
                        [obs_bc, np.zeros((len(obs_bc), pad_len), dtype=np.float32)],
                        axis=1
                    )
                behavioral_cloning(model, obs_bc, data["actions"], n_epochs=50)
                logger.info("BC done")
            except Exception as e:
                logger.warning(f"BC failed: {e}")

    # ── Curriculum (optional) ─────────────────────────────────────────────
    if args.curriculum:
        logger.info("Stage 2: Curriculum warm-up (single-sat, shares weights)")
        try:
            from curriculum import CurriculumScheduler
            from env_dynamic_factory import make_env
            from reward_shaping import DynamicRewardShaper
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.monitor import Monitor

            sched = CurriculumScheduler(verbose=False)

            def _c_make():
                e = sched.make_env(TARGETS, CLOUD, seed=args.seed,
                                   use_smdp=False, cfg=Config.DYN_REAL_VISION,
                                   with_safety=True)
                e = DynamicRewardShaper(e)
                obs_raw, _ = e.reset()
                # NOTE: single-sat obs will be 56-dim; pad to AUG_OBS for model compatibility
                return Monitor(e)

            # Train curriculum with single-sat env (same policy weights)
            cvec = DummyVecEnv([_c_make])
            model.set_env(cvec)
            for ep in range(200):
                model.learn(total_timesteps=steps_per_ep, reset_num_timesteps=False)
            cvec.close()
            model.set_env(vec)
            logger.info("Curriculum done — back to multi-sat env")
        except Exception as e:
            logger.warning(f"Curriculum failed: {e}")

    # ── Main PPO Training ─────────────────────────────────────────────────
    print(f"\n  Stage 3: SMDP-PPO  {args.episodes} episodes  N={N} satellites")

    ckpt_dir = os.path.join(MODELS, "checkpoints_multisat")
    cb_list  = CallbackList([
        EntropyAnnealingCallback(
            start_val=args.ent_coef, end_val=0.005,
            total_timesteps=total_steps, verbose=1
        ),
        DynamicEventCallback(log_every=50, verbose=1),
        AutoCheckpointCallback(
            save_freq=max(50_000, total_steps // 10),
            save_dir=ckpt_dir, exp_id=f"multisat_N{N}_s{args.seed}",
            extra_meta={"n_satellites": N, "seed": args.seed}, verbose=1,
        ),
    ])

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=cb_list,
                progress_bar=True, reset_num_timesteps=True)
    elapsed = time.time() - t0

    out_path = os.path.join(MODELS, f"ppo_multisat_N{N}.zip")
    model.save(out_path)
    print(f"\n  Done {elapsed/60:.1f} min  model → {out_path}")
    vec.close()

    # ── Evaluation ────────────────────────────────────────────────────────
    if args.eval:
        logger.info("Evaluating multi-satellite policy...")
        eval_vec = MultiSatelliteCoordVecEnv(
            make_env_fn=lambda sat_idx, seed: make_sat_env(
                sat_idx, seed, args.action_mask, False),
            n_satellites=N, seed=args.seed + 10000,
        )
        rewards, coverage = [], []
        for ep in range(30):
            obs = eval_vec.reset()
            done = np.zeros(N, dtype=bool)
            ep_r = 0.0
            while not np.all(done):
                acts, _ = model.predict(obs, deterministic=True)
                obs, r, done, infos = eval_vec.step(acts)
                ep_r += float(np.mean(r))
            rewards.append(ep_r)

        eval_vec.close()
        print(f"\n  Multi-sat eval (30 eps): "
              f"mean_r={np.mean(rewards):+.3f} ± {np.std(rewards):.3f}")

        result = {"n_satellites": N, "mean_reward": float(np.mean(rewards)),
                  "std_reward": float(np.std(rewards)), "elapsed_min": elapsed / 60}
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, f"multisat_N{N}_eval.json"), "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
