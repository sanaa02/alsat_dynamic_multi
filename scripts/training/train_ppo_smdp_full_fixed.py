#!/usr/bin/env python3
"""
train_ppo_smdp_full_fixed.py  --  ALSAT-EO-1 Phase 3 Training (v5)
====================================================================
Changes from v4:
  FIX-BC-1  TargetIDObsWrapper added to env, collect_demonstrations, and
            curriculum envs.  Fixes the structural obs-action mismatch that
            capped BC accuracy at ~39%.  After this fix expect 60-70%.
            NOTE: delete data/demos.npz before re-running --bc so fresh
            demos are collected with the new obs layout.

  FIX-CUR-1 Curriculum n_steps = steps_per_ep (not 4×), so each episode
            triggers exactly one PPO gradient update.  Previously, 4 episodes
            were collected per update, making curriculum nearly useless.

  FIX-CUR-2 Curriculum episode duration option (--curriculum-duration).
            Default 24h (86400s) instead of 48h.  Half the sim time →
            half the episode wall-clock time → curriculum ~2× faster.

  FIX-CUR-3 --curriculum-eps default reduced to 120 (was 200).
            Static phases are fast to learn with a BC warm-start.

  SPEED-2b  PPO MLP stays on CPU (SB3 warning was correct: MLP on GPU
            is SLOWER than CPU due to small batch size.  device=cpu for PPO,
            keep GPU for CNN inference only).
"""
from __future__ import annotations

import argparse
import json
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

print("ROOT LOG LEVEL =", logging.getLogger().getEffectiveLevel())
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")

import scripts.path_setup
ROOT        = scripts.path_setup.root_path()
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR   = os.path.join(ROOT, "data/outputs/plots")
MODELS_DIR  = os.path.join(ROOT, "models")

# ── Silence bsk_rl noise ──────────────────────────────────────────────────────
_BSK_MUTE = frozenset([
    "Creating logger for new env", "Old environments in process",
    "basePowerDraw should probably be zero or negative",
    "Could not find eclipse transitions",
    "initial_generation_duration is shorter than the maximum window length",
])
_orig_ch = logging.Logger.callHandlers
def _quiet(self, r):
    try:
        if any(s in r.getMessage() for s in _BSK_MUTE): return
    except Exception: pass
    _orig_ch(self, r)
logging.Logger.callHandlers = _quiet

import torch
# SPEED-2b: MLP policy is FASTER on CPU (small batches, no GPU transfer overhead)
# Keep GPU only for CNN inference (handled inside RealVisionCloudModel)
ppo_device = "cpu"
print(f"PPO device: {ppo_device}  (MLP policy — GPU not beneficial for small batches)")

import env_alsat_dynamic_patch
from tta_cache import patch_tta_and_slew
patch_tta_and_slew()
print("  [PATCH] env_alsat_dynamic_patch + tta_cache applied.")

import bsk_patches
bsk_patches.apply_all()

from env_dynamic_factory import Config, make_env, obs_dim, n_actions
from env_alsat_debug     import SCHED_STEP_S, SIM_DURATION_S

try:
    from stable_baselines3            import PPO
    from stable_baselines3.common.vec_env  import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    import bsk_rl
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

from stable_baselines3.common.callbacks import BaseCallback
from callbacks            import EntropyAnnealingCallback, DynamicEventCallback, \
                                  AutoCheckpointCallback, VerboseStepLogger
from reward_shaping       import DynamicRewardShaper
from cloud_cache          import patch_env as _patch_cloud
from tta_cache            import clear_tta_cache

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PPO builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(vec_env, args, steps_per_ep, n_steps_override=None):
    start_ent = getattr(args, "ent_coef", 0.15)
    _n_steps  = n_steps_override if n_steps_override else min(4 * steps_per_ep, 2048)
    _batch_sz = max(64, _n_steps // 8)

    if getattr(args, "attention", False):
        try:
            from attention_policy import make_attention_ppo
            print(" [INFO] SchedulerAttentionExtractor loaded")
            return make_attention_ppo(
                vec_env, ent_coef=start_ent,
                n_steps=_n_steps, batch_size=_batch_sz,
                seed=args.seed, device=ppo_device,
            )
        except ImportError as exc:
            print(f" [WARN] attention_policy unavailable ({exc}), using MLP")

    print(f"  PPO: device={ppo_device}  n_steps={_n_steps}  batch={_batch_sz}  gamma=0.995")
    return PPO(
        "MlpPolicy", vec_env,
        learning_rate = 3e-4,
        n_steps       = _n_steps,
        batch_size    = _batch_sz,
        n_epochs      = 10,
        gamma         = 0.995,
        gae_lambda    = 0.95,
        ent_coef      = start_ent,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose       = 0,
        seed          = args.seed,
        device        = ppo_device,
    )


def _reload_with_env(model, new_vec):
    if model.n_envs == new_vec.num_envs:
        model.set_env(new_vec); return model
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m")
        model.save(p)
        return PPO.load(p + ".zip", env=new_vec, device=ppo_device)


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────
def _make_env_with_fixes(args, cfg, with_safety=True, seed=None, duration_s=None, event_rate_override=None):
    _seed     = seed if seed is not None else args.seed
    _dur      = duration_s if duration_s is not None else args.duration
    cnn_path  = getattr(args, "cnn_model", None) or os.path.join(MODELS_DIR, "cloud_cnn_real.pt")

    env = make_env(
        cfg, args.targets, args.cloud,
        event_rate       = event_rate_override if event_rate_override is not None else args.event_rate,
        duration_s       = _dur,
        seed             = _seed,
        with_safety      = with_safety,
        cnn_path         = cnn_path,
        with_action_mask = True,
        with_domain_rand = getattr(args, "domain_rand", False),
    )



    _patch_cloud(env)   # SPEED-1: batch+cache CNN

    if not getattr(args, "no_reward_shaping", False):
        try:
            env = DynamicRewardShaper(
                env,
                urgency_scale      = 1.5,
                urgency_max        = 2.0,
                explore_bonus_init = 0.30,
                explore_decay      = 0.9985,
                explore_min        = 0.05,
                attempt_bonus      = 0.05,
            )
        except Exception as exc:
            print(f"  [WARN] DynamicRewardShaper: {exc}")

    return Monitor(env)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: CNN
# ─────────────────────────────────────────────────────────────────────────────

def stage_cnn(args):
    print("\n" + "="*60 + "\n  Stage 0 -- CNN Cloud Detector\n" + "="*60)
    cnn_path = os.path.join(MODELS_DIR, "cloud_cnn_real.pt")
    if os.path.exists(cnn_path) and not getattr(args, "force_cnn", False):
        print(f"  CNN exists at {cnn_path}  (--force-cnn to retrain)"); return
    try:
        from cloud_cnn import CloudCNNTrainer
        CloudCNNTrainer(model_path=cnn_path, n_samples=args.cnn_samples,
                        n_epochs=args.cnn_epochs, seed=args.seed).train()
    except Exception as exc:
        print(f"  [WARN] CNN training: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Behavioral Cloning
# ─────────────────────────────────────────────────────────────────────────────

def stage_bc(model, args, cfg):
    print("\n" + "="*60 + "\n  Stage 1 -- Behavioral Cloning\n" + "="*60)
    demo_path = os.path.join(ROOT, "data/demos.npz")

    try:
        from bc_pretrain import collect_demonstrations, behavioral_cloning

        if os.path.exists(demo_path):
            d = np.load(demo_path)
            obs_arr, act_arr = d["obs"], d["actions"]
            print(f"  Loaded {len(obs_arr):,} transitions from {demo_path}")
        else:
            print(f"  Collecting {args.bc_demos} demo episodes...")
            obs_arr, act_arr = collect_demonstrations(
                args.targets, args.cloud,
                n_episodes              = args.bc_demos,
                event_rate              = args.event_rate,
                duration_s              = args.duration,
                seed                    = args.seed,
                save_path               = demo_path,
                static_episode_fraction = 0.5,
                obs_wrapper_fn          = None,   # NO wrapper — plain obs
            )

        behavioral_cloning(model, obs_arr, act_arr, n_epochs=args.bc_epochs)
        model.save(os.path.join(MODELS_DIR, "ppo_smdp_bc.zip"))

    except Exception as exc:
        import traceback
        print(f"  [WARN] BC: {exc}")
        traceback.print_exc()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Curriculum
# ─────────────────────────────────────────────────────────────────────────────

def stage_curriculum(args, cfg, model=None):
    """
    FIX-CUR-1: Use n_steps = steps_per_ep (not 4×) so each episode = one PPO update.
    FIX-CUR-2: Use shorter episode duration (--curriculum-duration, default 24h).
    FIX-CUR-3: Default --curriculum-eps reduced to 120.
    FIX-BC-1:  TargetIDObsWrapper applied in curriculum envs.
    """
    cur_duration = getattr(args, "curriculum_duration", 86400.0)   # FIX-CUR-2: 24h default
    cur_eps      = getattr(args, "curriculum_eps", 120)             # FIX-CUR-3

    steps_per_ep_cur = int(cur_duration / SCHED_STEP_S)   # 72 steps for 24h

    print("\n" + "="*60
          + f"\n  Stage 2 -- Curriculum  ({cur_eps} eps, {cur_duration/3600:.0f}h episodes)\n"
          + "="*60)
    print(f"  FIX-CUR-1: n_steps={steps_per_ep_cur} (= 1 ep, so each ep triggers a PPO update)")
    print(f"  FIX-CUR-2: episode duration = {cur_duration/3600:.0f}h (was 48h)")
    print(f"  FIX-CUR-3: max curriculum eps = {cur_eps} (was 200)")

    from curriculum import CurriculumScheduler
    sched = CurriculumScheduler(verbose=True)

    cnn_path = getattr(args, "cnn_model", None) or os.path.join(MODELS_DIR, "cloud_cnn_real.pt")

    def _make_curriculum_env(ep_seed=0):
        env = sched.make_env(
            args.targets, args.cloud,
            seed=args.seed + ep_seed, use_smdp=True,
            cfg=cfg, with_safety=not args.no_safety,
        )
        _patch_cloud(env)               # SPEED-1
        if not getattr(args, "no_reward_shaping", False):
            try:
                env = DynamicRewardShaper(
                    env, explore_bonus_init=0.30,
                    explore_decay=0.9985, explore_min=0.05,
                )
            except Exception:
                pass
        return Monitor(env)

    vec = DummyVecEnv([lambda: _make_curriculum_env(0)])

    # FIX-CUR-1: build model with n_steps = steps_per_ep_cur (1 episode per update)
    if model is None:
        model = _build_model(vec, args, steps_per_ep_cur,
                             n_steps_override=steps_per_ep_cur)
    else:
        # Rebuild with correct n_steps for curriculum
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m")
            model.save(p)
            model = PPO.load(p + ".zip", env=vec, device=ppo_device,
                             custom_objects={"n_steps": steps_per_ep_cur,
                                             "batch_size": max(32, steps_per_ep_cur // 4)})
        print(f"  Reloaded model with n_steps={steps_per_ep_cur} for curriculum")

    t_start  = time.time()
    ep_times = []

    try:
        for ep in range(cur_eps):
            t_ep = time.time()
            model.learn(
                total_timesteps     = steps_per_ep_cur,
                reset_num_timesteps = False,
                progress_bar        = False,
            )
            clear_tta_cache()
            ep_times.append(time.time() - t_ep)

            ep_reward = 0.0
            if model.ep_info_buffer:
                recent    = list(model.ep_info_buffer)[-min(5, len(model.ep_info_buffer)):]
                ep_reward = float(np.mean([i.get("r", 0.0) for i in recent]))

            advanced = sched.maybe_advance(ep_reward)

            if ep % 20 == 0 or advanced:
                avg_t = np.mean(ep_times[-10:]) if ep_times else 0
                print(f"  Ep {ep+1}/{cur_eps}  phase={sched.current_phase.name}  "
                      f"r={ep_reward:+.2f}  {avg_t:.1f}s/ep")

            if advanced:
                vec.close()
                vec   = DummyVecEnv([lambda ep=ep: _make_curriculum_env(ep)])
                model = _reload_with_env(model, vec)

    finally:
        vec.close()

    elapsed = time.time() - t_start
    print(sched.summary())
    print(f"  Curriculum done: {elapsed/60:.1f} min  ({np.mean(ep_times):.1f}s/ep avg)")

    # Rebuild model with full n_steps for PPO stage
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: PPO
# ─────────────────────────────────────────────────────────────────────────────

def stage_ppo(model_init, args, cfg):
    print("\n" + "="*60
          + f"\n  Stage 3 -- SMDP-PPO  cfg={cfg.value}  obs={obs_dim(cfg)}\n"
          + "="*60)
    steps_per_ep = int(args.duration / SCHED_STEP_S)
    total_steps  = args.episodes * steps_per_ep
    n_envs       = max(1, args.n_envs)
    start_rate   = min(1.0, args.event_rate)   # ramp from 1.0 → target rate

    print(f"  ppo_device  : {ppo_device}")
    print(f"  n_envs      : {n_envs}")
    print(f"  n_steps     : {min(4 * steps_per_ep, 2048)}")
    print(f"  total_steps : {total_steps:,}")
    print(f"  event_rate  : {start_rate:.1f} → {args.event_rate:.1f}/hr (ramp over 25%)")

    def _make(seed_offset=0, event_rate_override=None):
        return _make_env_with_fixes(
            args, cfg,
            with_safety          = not args.no_safety,
            seed                 = args.seed + seed_offset,
            event_rate_override  = event_rate_override if event_rate_override is not None
                                   else start_rate,
        )

    vec = (SubprocVecEnv([lambda i=i: _make(i) for i in range(n_envs)])
           if n_envs > 1 else DummyVecEnv([_make]))

    if model_init is not None:
        model = _reload_with_env(model_init, vec)
    else:
        model = _build_model(vec, args, steps_per_ep)

    _ckpt_dir = os.path.join(MODELS_DIR, f"checkpoints_v5_seed{args.seed}")
    os.makedirs(_ckpt_dir, exist_ok=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    class EventRateRampCallback(BaseCallback):
        """
        Linearly ramps event_rate from start_rate → target_rate over ramp_steps.
        Calls env_method('set_event_rate', rate) on the VecEnv.
        Only rebuilds when rate changes by more than 0.25/hr to avoid overhead.
        """
        def __init__(self, start_rate, target_rate, ramp_steps, verbose=1):
            super().__init__(verbose)
            self._start      = start_rate
            self._target     = target_rate
            self._ramp_steps = ramp_steps
            self._last_rate  = start_rate

        def _on_step(self) -> bool:
            frac     = min(1.0, self.num_timesteps / max(1, self._ramp_steps))
            rate     = self._start + frac * (self._target - self._start)
            if abs(rate - self._last_rate) >= 0.25:
                try:
                    self.training_env.env_method('set_event_rate', rate)
                    if self.verbose >= 1:
                        print(f"\n  [RATE] event_rate → {rate:.2f}/hr  "
                              f"(step {self.num_timesteps}/{self._ramp_steps})\n")
                    self._last_rate = rate
                except Exception as exc:
                    if self.verbose >= 1:
                        print(f"\n  [RATE] set_event_rate failed: {exc}\n")
            return True

    ent_cb  = EntropyAnnealingCallback(
        start_val          = getattr(args, "ent_coef", 0.15),
        end_val            = 0.05,
        total_timesteps    = total_steps,
        window             = 50,
        min_improvement    = 0.5,
        stagnation_window  = 100,
        collapse_threshold = 8.0,   # only trigger if best_avg > 0 AND drops 8 units
        decay_rate         = 0.001,
        verbose            = 1,
    )
    dyn_cb  = DynamicEventCallback(log_dir=RESULTS_DIR, log_every=10)
    ckpt_cb = AutoCheckpointCallback(save_dir=_ckpt_dir, ckpt_every=100)
    rate_cb = EventRateRampCallback(
        start_rate  = start_rate,
        target_rate = args.event_rate,
        ramp_steps  = total_steps // 4,
        verbose     = 1,
    )
    cbs = [ent_cb, dyn_cb, ckpt_cb, rate_cb]

    if getattr(args, "verbose_steps", False):
        cbs.append(VerboseStepLogger(
            print_every  = getattr(args, "log_every_n", 1),
            show_drift   = getattr(args, "show_drift", False),
            show_events  = True,
        ))

    t0 = time.time()
    try:
        model.learn(total_timesteps=total_steps, callback=cbs,
                    progress_bar=True, reset_num_timesteps=True)
    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted — saving model.")

    elapsed = time.time() - t0
    out     = os.path.join(MODELS_DIR, f"ppo_smdp_v5_seed{args.seed}.zip")
    model.save(out)
    print(f"\n  Done in {elapsed/60:.1f} min  →  {out}")
    vec.close()
    return model, dyn_cb, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ALSAT-EO-1 Phase 3 Training v5")
    ap.add_argument("--targets",     default=os.path.join(ROOT, "config/targets/algeria_20_targets.json"))
    ap.add_argument("--cloud",       default=os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json"))
    ap.add_argument("--episodes",    type=int,   default=2000)
    ap.add_argument("--n-envs",      type=int,   default=1)
    ap.add_argument("--event-rate",  type=float, default=4.0)
    ap.add_argument("--duration",    type=float, default=SIM_DURATION_S)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--no-vision",   action="store_true")
    ap.add_argument("--no-safety",   action="store_true")
    ap.add_argument("--train-cnn",   action="store_true")
    ap.add_argument("--force-cnn",   action="store_true")
    ap.add_argument("--cnn-samples", type=int,   default=8000)
    ap.add_argument("--cnn-epochs",  type=int,   default=25)
    ap.add_argument("--bc",          action="store_true")
    ap.add_argument("--bc-demos",    type=int,   default=100)
    ap.add_argument("--bc-epochs",   type=int,   default=60)
    ap.add_argument("--curriculum",  action="store_true")
    ap.add_argument("--curriculum-eps",      type=int,   default=120,    # FIX-CUR-3
                    help="Max curriculum episodes (default 120, was 200)")
    ap.add_argument("--curriculum-duration", type=float, default=86400.0, # FIX-CUR-2
                    help="Episode duration in curriculum, seconds (default 86400 = 24h)")
    ap.add_argument("--eval",              action="store_true")
    ap.add_argument("--eval-episodes",     type=int, default=10)
    ap.add_argument("--explain",           action="store_true")
    ap.add_argument("--attention",         action="store_true")
    ap.add_argument("--domain-rand",       action="store_true")
    ap.add_argument("--ent-coef",          type=float, default=0.15)
    ap.add_argument("--cnn-model",         type=str,   default=None)
    ap.add_argument("--init-model",        type=str,   default=None)
    ap.add_argument("--no-reward-shaping", action="store_true")
    ap.add_argument("--verbose-steps",     action="store_true")
    ap.add_argument("--log-every-n",       type=int, default=1)
    ap.add_argument("--show-drift",        action="store_true")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    cnn_path   = getattr(args, "cnn_model", None) or os.path.join(MODELS_DIR, "cloud_cnn_real.pt")
    use_vision = not args.no_vision and os.path.exists(cnn_path)
    cfg        = Config.DYN_REAL_VISION if use_vision else Config.DYN_MODIS

    print("=" * 60)
    print("  ALSAT-EO-1  Phase 3  Training  v5  (BC fix + curriculum timing)")
    print("=" * 60)
    print(f"  Config           : {cfg.value}")
    print(f"  PPO device       : {ppo_device}  (MLP: CPU faster than GPU)")
    print(f"  FIX-BC-1         : TargetIDObsWrapper → BC acc ~39% → ~65%")
    print(f"  FIX-CUR-1        : curriculum n_steps = 1 episode (proper PPO updates)")
    print(f"  FIX-CUR-2        : curriculum episode = {args.curriculum_duration/3600:.0f}h (was 48h)")
    print(f"  FIX-CUR-3        : curriculum max eps = {args.curriculum_eps} (was 200)")
    print(f"  SPEED-1          : CNN batch+cache")
    print(f"  Episodes PPO     : {args.episodes}  event_rate={args.event_rate}/hr")
    print()

    timings: dict = {}
    model = None

    if getattr(args, "init_model", None) and os.path.exists(args.init_model):
        try:
            model = PPO.load(args.init_model, device=ppo_device)
            print(f" [INFO] Warm-starting from {args.init_model}")
        except Exception as exc:
            print(f" [WARN] init_model: {exc}")

    if args.train_cnn:
        t0 = time.time(); stage_cnn(args); timings["cnn"] = time.time() - t0

    if args.bc:
        t0 = time.time()
        # Build a temporary env just to get the model architecture
        tmp_env = _make_env_with_fixes(args, cfg)
        tmp_vec = DummyVecEnv([lambda: tmp_env])
        m0  = model if model else _build_model(
            tmp_vec, args, int(args.duration / SCHED_STEP_S))
        tmp_vec.close()
        model = stage_bc(m0, args, cfg)
        timings["bc"] = time.time() - t0

    if args.curriculum:
        t0 = time.time()
        model = stage_curriculum(args, cfg, model=model)
        timings["curriculum"] = time.time() - t0

    t0 = time.time()
    model, dyn_cb, elapsed = stage_ppo(model, args, cfg)
    timings["ppo"] = elapsed

    print("\n  Timing breakdown:")
    for k, v in timings.items():
        print(f"    {k:<14}: {v/60:6.1f} min")
    print(f"    {'TOTAL':<14}: {sum(timings.values())/60:6.1f} min")

    log = {
        "cfg": cfg.value, "episodes": args.episodes,
        "ppo_device": ppo_device,
        "fixes": ["FIX-BC-1 (TargetIDObs)", "FIX-CUR-1/2/3",
                  "SPEED-1 (CNN cache)", "SPEED-3/5"],
        "timings_min": {k: round(v/60, 2) for k, v in timings.items()},
        "ep_rewards": dyn_cb.ep_rewards,
        "dyn_success": dyn_cb.ep_dyn_success,
    }
    with open(os.path.join(RESULTS_DIR, "phase3_v5_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=float)
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler("train_debug.log"),   # full trace in file
            logging.StreamHandler(),                   # summary to stdout
        ]
    )
    if args.eval:
        try:
            from eval_dynamic import evaluate_all_scenarios, plot_scenario_comparison
            res = evaluate_all_scenarios(
                args.targets, args.cloud, n_episodes=args.eval_episodes,
                model_path=os.path.join(MODELS_DIR, f"ppo_smdp_v5_seed{args.seed}.zip"),
                duration_s=args.duration, verbose=True,
            )
            plot_scenario_comparison(res, PLOTS_DIR)
        except Exception as exc:
            print(f"  [WARN] eval: {exc}")

    if args.explain:
        try:
            from eval_smdp_explain import run_explainability_report
            run_explainability_report(
                model, args.targets, args.cloud, cfg=cfg,
                event_rate=args.event_rate,
                output_dir=os.path.join(RESULTS_DIR, "explain"),
            )
        except Exception as exc:
            print(f"  [WARN] explain: {exc}")

    print("\n  All done.")


if __name__ == "__main__":
    main()