#!/usr/bin/env python3
from __future__ import annotations
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# from attention_policy import make_attention_ppo

# -----------------------------------------------------------------
"""
train_ppo_smdp_full.py  --  ALSAT-EO-1  Phase 3  Master Training Pipeline
=========================================================================
5-stage pipeline:
  Stage 0  CNN cloud-detector training         (--train-cnn)
  Stage 1  Behavioral Cloning pretraining      (--bc)
  Stage 2  Curriculum warm-up                  (--curriculum)
  Stage 3  SMDP-PPO main training              (always)
  Stage 4  3-scenario evaluation               (--eval)
  Stage 5  SHAP explainability report          (--explain)

Key changes vs. prior version:
  - SafetyMonitor is ON by default (pass --no-safety to disable)
  - All envs built via env_dynamic_factory.make_env()
  - obs_dim = 56  (SMDP built into DynamicObsWrapper)
  - smdp_dynamic.py is no longer needed / imported
"""

import argparse, json, os, sys, time, logging
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


import path_setup
ROOT        = path_setup.root_path()
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR   = os.path.join(ROOT, "data/outputs/plots")
MODELS_DIR  = os.path.join(ROOT, "models")

# Silence bsk_rl noise
_BSK = frozenset(["Creating logger for new env","Old environments in process",
                   "basePowerDraw should probably be zero or negative",
                    "Could not find eclipse transitions", 
                     "initial_generation_duration is shorter than the maximum window length"])
_orig_ch = logging.Logger.callHandlers
def _q(self, r):
    try:
        if any(s in r.getMessage() for s in _BSK): return
    except Exception: pass
    _orig_ch(self, r)
logging.Logger.callHandlers = _q

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using device: {device}")


from env_dynamic_factory import Config, make_env, make_vec_env, obs_dim, n_actions
from env_alsat_debug import SCHED_STEP_S, SIM_DURATION_S

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    import bsk_rl
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

import gymnasium as gym


class FullTrainingLogger(BaseCallback):
    """
    Per-episode training logger with:
    - Action distribution (shows if agent is taking DYN actions 20-22)
    - Cloud-free rate & dynamic success rate
    - Reward breakdown (static vs dynamic contributions)
    - Auto-save to JSON log file on each episode
    - Checkpoint every CKPT_EVERY episodes
    """

    CKPT_EVERY  = 100    # save model checkpoint every N episodes
    LOG_EVERY   = 10     # print to terminal every N episodes
    VERBOSE_MOD = 50     # print action distribution every N episodes

    def __init__(self, verbose=1, log_path=None, ckpt_dir=None):
        super().__init__(verbose)
        self.ep_rewards     = []
        self.ep_cf_rates    = []
        self.ep_dyn_success = []
        self.ep_dyn_acts_pct= []   # % of steps that were DYN actions 20-22
        self.ep_metrics_log = []   # full per-episode metrics for JSON
        self._ep_reward     = 0.0
        self._ep_count      = 0
        self._step_actions  = []   # actions taken this episode
        self._log_path      = log_path or os.path.join(RESULTS_DIR, "training_live.json")
        self._ckpt_dir = ckpt_dir or os.path.join(MODELS_DIR, "checkpoints")
        os.makedirs(self._ckpt_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals.get("rewards", [0.0])[0])

        # Track actions this episode
        for a in np.atleast_1d(self.locals.get("actions", [])):
            self._step_actions.append(int(a))

        done = bool(self.locals.get("dones", [False])[0])
        if not done:
            return True

        # ── Episode ended ──────────────────────────────────────────────────
        self._ep_count += 1
        infos = self.locals.get("infos", [{}])
        m     = (infos[0].get("episode_metrics", {}) if infos else {})

        ni    = m.get("n_imaged",       0)
        nc    = m.get("n_cloud_free",   0)
        n_cl  = m.get("n_cloudy",       0)
        nd    = m.get("n_dyn_detected", m.get("n_detected", 0))
        nim   = m.get("n_dyn_imaged",   0)
        slew  = m.get("total_slew_angle_deg", 0.0)
        cf    = nc  / ni if ni  > 0 else 0.0
        ds    = nim / nd if nd  > 0 else 0.0

        # Action distribution
        total       = max(len(self._step_actions), 1)
        dyn_n       = sum(1 for a in self._step_actions if 20 <= a <= 22)
        drift_n     = self._step_actions.count(23)
        static_n    = sum(1 for a in self._step_actions if  0 <= a <= 19)
        dyn_pct     = 100 * dyn_n   / total
        drift_pct   = 100 * drift_n / total
        static_pct  = 100 * static_n/ total

        self.ep_rewards.append(self._ep_reward)
        self.ep_cf_rates.append(cf)
        self.ep_dyn_success.append(ds)
        self.ep_dyn_acts_pct.append(dyn_pct)

        # ── JSON log entry ─────────────────────────────────────────────────
        entry = {
            "ep":          self._ep_count,
            "reward":      round(self._ep_reward, 4),
            "cf_rate":     round(cf,  4),
            "dyn_suc":     round(ds,  4),
            "n_imaged":    ni,
            "n_cloud_free":nc,
            "n_cloudy":    n_cl,
            "n_dyn_det":   nd,
            "n_dyn_img":   nim,
            "dyn_act_pct": round(dyn_pct,  2),
            "drift_pct":   round(drift_pct,2),
            "slew_deg":    round(slew, 1),
            "ent_coef":    round(float(self.model.ent_coef), 6)
                           if self.model else 0.0,
        }
        self.ep_metrics_log.append(entry)

        # Write live JSON (append-friendly) every 10 episodes
        if self._ep_count % self.LOG_EVERY == 0:
            with open(self._log_path, "w") as _f:
                import json as _json
                _json.dump({
                    "episodes": self.ep_metrics_log,
                    "summary": {
                        "n_episodes": self._ep_count,
                        "mean_reward": round(float(np.mean(self.ep_rewards[-100:])), 3),
                        "mean_cf":     round(float(np.mean(self.ep_cf_rates[-100:])), 4),
                        "mean_dyn_suc":round(float(np.mean(self.ep_dyn_success[-100:])), 4),
                        "mean_dyn_pct":round(float(np.mean(self.ep_dyn_acts_pct[-100:])), 2),
                    }
                }, _f, indent=2, default=float)

        # ── Checkpoint every CKPT_EVERY episodes ──────────────────────────
        if self._ep_count % self.CKPT_EVERY == 0 and self.model is not None:
            ckpt = os.path.join(self._ckpt_dir,
                                f"ppo_smdp_ep{self._ep_count:05d}.zip")
            self.model.save(ckpt)
            print(f"  [CKPT] Saved → {ckpt}")

        # ── Terminal print ─────────────────────────────────────────────────
        if self.verbose >= 1 and self._ep_count % self.LOG_EVERY == 0:
            r10   = np.mean(self.ep_rewards[-10:])
            cf10  = np.mean(self.ep_cf_rates[-10:])   if self.ep_cf_rates   else 0.0
            d10   = np.mean(self.ep_dyn_success[-10:]) if self.ep_dyn_success else 0.0
            dp10  = np.mean(self.ep_dyn_acts_pct[-10:]) if self.ep_dyn_acts_pct else 0.0
            ent   = float(self.model.ent_coef) if self.model else 0.0

            # Reward breakdown: static vs potential dynamic contribution
            dyn_contrib = "🎯 DYN!" if nim > 0 else ""
            r0_flag     = " ← DRIFT/NO-ACCESS" if self._ep_reward == 0 else ""
            dyn_alert   = " ⚠️ ENT-COLLAPSE" if dp10 < 0.5 else ""

            print(f"  Ep {self._ep_count:4d} "
                  f" r={self._ep_reward:+8.3f}{r0_flag}"
                  f"  avg10={r10:+7.3f}"
                  f"  cf={cf:.0%}(avg={cf10:.0%})"
                  f"  dyn_suc={ds:.0%}(avg={d10:.0%})"
                  f"  dyn_act={dyn_pct:.1f}%(avg={dp10:.1f}%)"
                  f"  ent={ent:.4f}"
                  f"  {dyn_contrib}")

            # Detailed action breakdown every VERBOSE_MOD episodes
            if self._ep_count % self.VERBOSE_MOD == 0:
                print(f"         Actions → static={static_pct:.0f}%  "
                      f"dyn={dyn_pct:.0f}%  drift={drift_pct:.0f}%  "
                      f"| n_dyn_detected={nd}  n_dyn_imaged={nim}"
                      f"  | slew={slew:.0f}°{dyn_alert}")

        self._ep_reward    = 0.0
        self._step_actions = []
        return True


def _build_model(vec_env, args, steps_per_ep):
    start_ent = getattr(args, 'ent_coef', 0.15)
    if getattr(args, 'attention', False):
        try:
            from attention_policy import make_attention_ppo
            print(" [INFO] SchedulerAttentionExtractor policy loaded")
            return make_attention_ppo(
                vec_env, ent_coef=start_ent,
                seed=args.seed, device="cuda",
            )
        except ImportError as e:
            print(f" [WARN] attention_policy not importable ({e}), falling back to MLP")
    # Default MLP policy
    _n_steps   = 4 * steps_per_ep          # 4 × 144 = 576 steps
    _batch_sz  = max(72, _n_steps // 8)    # ≈ 72 minimum, larger if possible
    # gamma: 0.995 gives effective horizon ~200 steps > 144-step episode;
    # covers the full 48 h episode without over-discounting late events.
    return PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=_n_steps,
        batch_size=_batch_sz,
        n_epochs=10,
        gamma=0.995,        # was 0.99; better late-episode coverage
        gae_lambda=0.95,
        ent_coef=start_ent,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
        seed=args.seed,
        device="cpu",
    )


def stage_cnn(args):
    print("\n" + "="*60 + "\n  Stage 0 -- CNN Cloud Detector\n" + "="*60)
    cnn_path = os.path.join(MODELS_DIR, "cloud_cnn_real.pt")
    if os.path.exists(cnn_path) and not args.force_cnn:
        print(f"  CNN exists at {cnn_path}  (--force-cnn to retrain)"); return
    try:
        from cloud_cnn import CloudCNNTrainer
        CloudCNNTrainer(model_path=cnn_path, n_samples=args.cnn_samples,
                        n_epochs=args.cnn_epochs, seed=args.seed).train()
    except Exception as e:
        print(f"  [WARN] CNN training failed: {e}")

def _reload_with_env(model, new_vec):
    if model.n_envs == new_vec.num_envs:
        model.set_env(new_vec)   # correct SB3 API for same n_envs
        return model
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "tmp_model")
        model.save(tmp)
        return PPO.load(tmp + ".zip", env=new_vec, device=device)

def stage_bc(model, args, cfg):
    print("\n" + "="*60 + "\n  Stage 1 -- Behavioral Cloning\n" + "="*60)
    demo_path = os.path.join(ROOT, "data/demos.npz")
    try:
        from bc_pretrain import collect_demonstrations, behavioral_cloning
        if os.path.exists(demo_path):
            data=np.load(demo_path); obs_arr=data["obs"]; act_arr=data["actions"]
            print(f"  Loaded {len(obs_arr):,} transitions from {demo_path}")
        else:
            print(f"  Collecting {args.bc_demos} demo episodes...")
            obs_arr, act_arr = collect_demonstrations(
                args.targets, args.cloud, n_episodes=args.bc_demos,
                event_rate=args.event_rate, duration_s=args.duration,
                seed=args.seed, use_smdp=False, save_path=demo_path)
        print(f"  BC pretraining ({args.bc_epochs} epochs)...")
        behavioral_cloning(model, obs_arr, act_arr, n_epochs=args.bc_epochs)
        model.save(os.path.join(MODELS_DIR,"ppo_smdp_bc.zip"))
    except Exception as e:
        import traceback
        print(f"  [WARN] BC stage error: {e}")
        traceback.print_exc() 
    return model

def stage_curriculum(args, cfg, model=None):
    print("\n" + "="*60 + "\n  Stage 2 -- Curriculum Warm-up\n" + "="*60)
    from curriculum import CurriculumScheduler
    sched = CurriculumScheduler(verbose=True)
    steps_per_ep = int(args.duration / SCHED_STEP_S)

    def _make():
        env = sched.make_env(args.targets, args.cloud, seed=args.seed,
                             use_smdp=False, cfg=cfg,
                             with_safety=not args.no_safety)
        return Monitor(env)

    vec = DummyVecEnv([_make])

    if model is None:
        model = _build_model(vec, args, steps_per_ep)
    else:
        # Replace the model's old (dead) environment with the new vec
        model = _reload_with_env(model, vec)
    try:
        current_phase = None
        for ep in range(min(args.curriculum_eps, 500)):
            if sched.current_phase.name != current_phase:
                vec.close()
                vec = DummyVecEnv([lambda ep=ep: Monitor(
                    sched.make_env(args.targets, args.cloud, seed=args.seed+ep,
                                   use_smdp=True, cfg=cfg,
                                   with_safety=not args.no_safety)
                )])
                model = _reload_with_env(model, vec)
                current_phase = sched.current_phase.name
            # Learn for 2× episode length to guarantee at least one full episode
            # terminates and is logged by Monitor (SMDP steps vary in length, so
            # the episode may need more than steps_per_ep wrapper steps to finish).
            model.learn(total_timesteps=2 * steps_per_ep, reset_num_timesteps=False)

            # Read episode reward from Monitor's buffer.
            # ep_info_buffer stores dicts with key 'r' (scalar float).
            # If empty (episode still in progress), default to 0.0 — never advances.
            if model.ep_info_buffer:
                # Average over last min(5, buffer_len) completed episodes for stability.
                recent_n = min(5, len(model.ep_info_buffer))
                ep_reward = float(np.mean(
                    [info["r"] for info in list(model.ep_info_buffer)[-recent_n:]]
                ))
            else:
                ep_reward = 0.0
                logger.warning(
                    f"[curriculum ep {ep}] ep_info_buffer empty — "
                    "episode may not have terminated. Consider increasing total_timesteps."
                )
            advanced = sched.maybe_advance(ep_reward)
            if ep % 25 == 0 or advanced:
                print(f"  Ep {ep+1}/{min(args.curriculum_eps,500)}  "
                      f"phase={sched.current_phase.name}  r={ep_reward:+.2f}")
    finally:
       vec.close()
    print(sched.summary())
    return model


def stage_ppo(model_init, args, cfg):
    print("\n" + "="*60 +
          f"\n  Stage 3 -- SMDP-PPO  cfg={cfg.value}  obs={obs_dim(cfg)}\n" +
          "="*60)
    steps_per_ep = int(args.duration / SCHED_STEP_S)
    total_steps  = args.episodes * steps_per_ep

    with_safety = not args.no_safety

    def _make():
        env = make_env(
            cfg, args.targets, args.cloud,
            event_rate=args.event_rate,
            duration_s=args.duration,
            seed=args.seed,
            with_safety=with_safety,
            cnn_path=args.cnn_model,
            with_action_mask=args.action_mask,
            with_domain_rand=args.domain_rand,
        )
        # Apply dynamic reward shaping (was written but never applied before)
        if not getattr(args, 'no_reward_shaping', False):
            try:
                from reward_shaping import DynamicRewardShaper
                env = DynamicRewardShaper(
                    env,
                    urgency_scale=3.0,
                    urgency_max=4.0,
                    explore_bonus_init=0.3,
                    explore_decay=0.99,
                    explore_min=0.0,
                )
            except ImportError:
                pass  # reward_shaping not available, skip silently
        return Monitor(env)

    vec = DummyVecEnv([_make] * max(1, args.n_envs))

    if model_init:
        model = _reload_with_env(model_init, vec)
    else:
        model = _build_model(vec, args, steps_per_ep)

    _live_log = os.path.join(RESULTS_DIR, "training_live.json")
    _ckpt_dir = os.path.join(MODELS_DIR, f"checkpoints_seed{args.seed}")
    os.makedirs(_ckpt_dir, exist_ok=True)

    cb = FullTrainingLogger(verbose=1, log_path=_live_log, ckpt_dir=_ckpt_dir)

    from callbacks import EntropyAnnealingCallback
    ent_cb = EntropyAnnealingCallback(
        start_val=getattr(args, 'ent_coef', 0.15),
        end_val=0.05,           # raised from 0.03 — keeps dynamic exploration alive
        total_timesteps=total_steps,
    )

    print(f"  Live log     -> {_live_log}")
    print(f"  Checkpoints  -> {_ckpt_dir}/ppo_smdp_epXXXXX.zip")

    # ── IMPORTANT: callbacks_list MUST be initialized here,
    #    before any conditional .append() calls below ─────────────────────────
    callbacks_list = [cb]

    if getattr(args, 'verbose_actions', False):
        try:
            from callbacks import VerboseStepLogger
            callbacks_list.append(VerboseStepLogger(print_every=1))
            print(" [INFO] Verbose step logging enabled (--verbose-actions)")
        except ImportError:
            print(" [WARN] VerboseStepLogger not found in callbacks.py")

    if getattr(args, 'verbose_steps', False):
        try:
            from callbacks import VerboseStepLogger
            vb = VerboseStepLogger(
                print_every=getattr(args, 'log_every_n', 1),
                show_drift=getattr(args, 'show_drift', False),
                show_events=True,
            )
            callbacks_list.append(vb)
            n = getattr(args, 'log_every_n', 1)
            print(f" [INFO] Verbose step logging enabled (every {n} steps)")
        except ImportError:
            print(" [WARN] VerboseStepLogger not found in callbacks.py")

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=[ent_cb] + callbacks_list,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\n  [INFO] Interrupted — saving current model.")

    elapsed = time.time() - t0
    out = os.path.join(MODELS_DIR, f"ppo_smdp_seed{args.seed}.zip")
    model.save(out)
    print(f"  Done {elapsed/60:.1f} min  model -> {out}")
    vec.close()
    return model, cb, elapsed


def main():
    ap = argparse.ArgumentParser(description="ALSAT-EO-1 Phase 3 Full Pipeline")
    ap.add_argument("--targets",     default=os.path.join(ROOT,"config/targets/algeria_20_targets.json"))
    ap.add_argument("--cloud",       default=os.path.join(ROOT,"config/cloud_reality/algeria_real_clouds.json"))
    ap.add_argument("--episodes",    type=int,   default=500)
    ap.add_argument("--n-envs",  type=int, default=1,
                        help="Number of parallel envs (SubprocVecEnv). Set to 4-8 for speedup.")
    ap.add_argument("--event-rate",  type=float, default=2.0)
    ap.add_argument("--duration",    type=float, default=SIM_DURATION_S)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--no-vision",   action="store_true")
    ap.add_argument("--no-safety",   action="store_true",
                    help="Disable SafetyMonitor (not recommended)")
    ap.add_argument("--train-cnn",   action="store_true")
    ap.add_argument("--force-cnn",   action="store_true")
    ap.add_argument("--cnn-samples", type=int, default=8000)
    ap.add_argument("--cnn-epochs",  type=int, default=25)
    ap.add_argument("--bc",          action="store_true")
    ap.add_argument("--bc-demos",    type=int, default=100)
    ap.add_argument("--bc-epochs",   type=int, default=50)
    ap.add_argument("--curriculum",  action="store_true")
    ap.add_argument("--curriculum-eps", type=int, default=200)
    ap.add_argument("--eval",        action="store_true")
    ap.add_argument("--eval-episodes",  type=int, default=5)
    ap.add_argument("--explain",     action="store_true")

    ap.add_argument("--verbose-actions", action="store_true",
                help="Print per-step action details (which target/event was imaged)")

    ap.add_argument("--action-mask", action="store_true",
                help="Use action masking to block infeasible actions")
    ap.add_argument("--domain-rand", action="store_true",    
                help="Enable domain randomisation (CNN noise, slew cost, etc.)")
    ap.add_argument("--ent-coef",   type=float, default=0.15,
                    help="PPO entropy coef start (anneals to 0.005). "
                         "Higher = more dynamic-event exploration.")
    ap.add_argument("--cnn-model", type=str, default=None, help="Path to a custom cloud CNN model")
    ap.add_argument("--attention", action="store_true",
                help="Use SchedulerAttentionExtractor cross-attention policy "
                     "(scripts/models/attention_policy.py) instead of MLP")
    ap.add_argument("--init-model", type=str, default=None,
                help="Path to a pre-trained model (.zip) to warm-start from "
                     "(e.g., ppo_smdp_full.zip from a previous run)")
    
    ap.add_argument("--no-reward-shaping", action="store_true",
    help="Disable DynamicRewardShaper (for ablation studies)")
    ap.add_argument("--verbose-steps", action="store_true",
        help="Per-step detail: cloud truth, slew, orbital pos, active events")
    ap.add_argument("--log-every-n", type=int, default=1,
        help="Print every N steps when --verbose-steps is on (default 1)")
    ap.add_argument("--show-drift", action="store_true",
        help="Also log DRIFT steps in verbose mode")
    
    args = ap.parse_args()

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    cnn_path   = os.path.join(MODELS_DIR,"cloud_cnn_real.pt")
    use_vision = not args.no_vision and os.path.exists(cnn_path)
    cfg        = Config.DYN_REAL_VISION if use_vision else Config.DYN_MODIS

    print("="*60)
    print("  ALSAT-EO-1  Phase 3  Master Training Pipeline")
    print("="*60)
    print(f"  Config  : {cfg.value}  obs={obs_dim(cfg)}  acts={n_actions(cfg)}")
    print(f"  Safety  : {'OFF (--no-safety)' if args.no_safety else 'ON (default)'}")
    print(f"  Episodes: {args.episodes}  rate={args.event_rate}/hr")

    model = None

    # ── Warm‑start: apply BEFORE BC/curriculum so they build on top of it ──
    if getattr(args, 'init_model', None) and os.path.exists(args.init_model):
        try:
            from stable_baselines3 import PPO as _PPO
            model = _PPO.load(args.init_model, device=device)
            print(f" [INFO] Warm-starting from {args.init_model}")
        except Exception as _e:
            print(f" [WARN] Could not load init model: {_e}")

    if args.bc:
        tmp = make_vec_env(cfg, seed=args.seed, event_rate=args.event_rate,
                           duration_s=args.duration,
                           with_safety=not args.no_safety,
                           cnn_path=cnn_path, n_envs=args.n_envs, use_subproc=False)
        # Use warm‑started model if available, otherwise build a fresh one
        m0 = model if model else _build_model(tmp, args, int(args.duration / SCHED_STEP_S))
        model = stage_bc(m0, args, cfg)
        tmp.close()

    if args.curriculum:
        # stage_curriculum currently creates its own model; we need to pass the warm‑started one.
        # For now, we keep the original call (it will ignore the warm‑start). 
        # To fully fix, you should modify stage_curriculum to accept an optional model.
        # But as a minimal fix, we call it with the warm‑started model if it exists.
        # Since stage_curriculum creates its own model, this won't use warm‑start.
        # The proper fix would require rewriting stage_curriculum to accept a model.
        # For the sake of the current bug fix, we move the warm‑start before but
        # curriculum will still train from scratch. This is not ideal but better than overwriting.
        # The full fix (modifying stage_curriculum) is outside this immediate bug list.
        # We keep the curriculum call as is.
        model = stage_curriculum(args, cfg, model=model)

    model, cb, elapsed = stage_ppo(model, args, cfg)

    # Save log
    log = {"cfg":cfg.value,"obs_dim":obs_dim(cfg),"event_rate":args.event_rate,
           "episodes":args.episodes,"elapsed_min":round(elapsed/60,2),
           "episode_rewards":cb.ep_rewards,"ep_cf_rates":cb.ep_cf_rates,
           "ep_dyn_success":cb.ep_dyn_success}
    with open(os.path.join(RESULTS_DIR,"phase3_full_log.json"),"w") as f:
        json.dump(log,f,indent=2,default=float)

    if args.eval:
        try:
            from eval_dynamic import evaluate_all_scenarios, plot_scenario_comparison
            res = evaluate_all_scenarios(args.targets, args.cloud,
                n_episodes=args.eval_episodes, model_path=os.path.join(MODELS_DIR, f"ppo_smdp_seed{args.seed}.zip"),
                duration_s=args.duration, verbose=True)
            plot_scenario_comparison(res, PLOTS_DIR)
        except Exception as e: print(f"  [WARN] eval: {e}")

    if args.explain:
        try:
            from eval_smdp_explain import run_explainability_report
            run_explainability_report(model, args.targets, args.cloud, cfg=cfg,
                event_rate=args.event_rate, output_dir=os.path.join(RESULTS_DIR,"explain"))
        except Exception as e: print(f"  [WARN] explain: {e}")

    print("\n  All done.")


if __name__ == "__main__":
    main()
