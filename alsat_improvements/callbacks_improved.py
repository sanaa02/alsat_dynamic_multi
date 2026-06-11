#!/usr/bin/env python3
"""
callbacks_improved.py  --  Improved SB3 Callbacks for ALSAT-EO-1 PPO Training
===============================================================================
Fixes the three critical training problems identified in the log analysis:

  1. EntropyAnnealCallback    -- fixes entropy collapse (ent stuck at 0.05)
  2. BestModelCallback        -- saves the best model by eval reward (not just last)
  3. MultiSeedEvalCallback    -- evaluates on multiple seeds for robust tracking
  4. DynSuccessMonitorCallback -- alerts when dynamic success collapses

Usage in train_improved.py:
    from callbacks_improved import (
        EntropyAnnealCallback, BestModelCallback,
        MultiSeedEvalCallback, build_callback_list
    )
    callbacks = build_callback_list(
        model_dir     = "models/",
        targets_path  = TARGETS,
        cloud_path    = CLOUD_JSON,
        total_episodes= N_EPISODES,
        event_rate    = 2.0,
        seed          = 42,
    )
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks)
"""
from __future__ import annotations

import os
import math
import json
import logging
import numpy as np
from typing import List, Optional

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logger = logging.getLogger(__name__)


# =============================================================================
#  1. Entropy Annealing Callback
#     Linearly decays ent_coef from start_ent to end_ent over decay_fraction
#     of total training. Prevents the entropy-collapse seen in training logs.
# =============================================================================

class EntropyAnnealCallback(BaseCallback):
    """
    Linearly anneals PPO's entropy coefficient over training.

    The training log shows ent=0.0500 for >2000 consecutive episodes,
    meaning entropy collapsed to its minimum value early in training.
    This callback ensures controlled decay from a high exploration value.

    Args:
        start_ent       : Initial entropy coefficient (default 0.15)
        end_ent         : Final entropy coefficient (default 0.01)
        decay_fraction  : Fraction of total timesteps over which to decay (default 0.80)
        total_timesteps : Total training timesteps (needed for schedule)
        log_every       : Print ent_coef every N calls to _on_step
        verbose         : Verbosity (0 = silent, 1 = log changes)
    """

    def __init__(
        self,
        start_ent:       float = 0.15,
        end_ent:         float = 0.01,
        decay_fraction:  float = 0.80,
        total_timesteps: int   = 720_000,
        log_every:       int   = 5000,
        verbose:         int   = 0,
    ):
        super().__init__(verbose)
        self.start_ent      = start_ent
        self.end_ent        = end_ent
        self.decay_fraction = decay_fraction
        self.total_ts       = total_timesteps
        self.log_every      = log_every
        self._decay_steps   = int(total_timesteps * decay_fraction)
        self._call_count    = 0

    def _on_training_start(self) -> None:
        self.model.ent_coef = self.start_ent
        if self.verbose >= 1:
            logger.info(
                f"[EntropyAnnealing] start={self.start_ent:.4f} "
                f"end={self.end_ent:.4f}  over {self._decay_steps:,} steps"
            )

    def _on_step(self) -> bool:
        self._call_count += 1
        ts = self.model.num_timesteps
        if ts >= self._decay_steps:
            new_ent = self.end_ent
        else:
            frac    = ts / self._decay_steps
            new_ent = self.start_ent + frac * (self.end_ent - self.start_ent)

        self.model.ent_coef = float(new_ent)

        if self.verbose >= 1 and self._call_count % self.log_every == 0:
            logger.info(
                f"[EntropyAnnealing] ts={ts:,}  ent_coef={new_ent:.5f}"
            )
        return True


# =============================================================================
#  2. Best Model Callback
#     Saves the model whenever mean eval reward improves.
#     The standard CheckpointCallback saves every N steps regardless of quality.
# =============================================================================

class BestModelCallback(BaseCallback):
    """
    Saves the model when a new best mean evaluation reward is achieved.

    Args:
        eval_env_fn     : Callable that returns a fresh evaluation env (not VecEnv)
        n_eval_episodes : Number of episodes per evaluation
        eval_every_n_ts : Evaluate every N timesteps
        save_path       : Directory where best_model.zip is saved
        deterministic   : Use deterministic actions for evaluation
        verbose         : 0 = silent, 1 = log on improvement
    """

    def __init__(
        self,
        eval_env_fn,
        n_eval_episodes: int   = 10,
        eval_every_n_ts: int   = 50_000,
        save_path:       str   = "models/",
        deterministic:   bool  = True,
        verbose:         int   = 1,
    ):
        super().__init__(verbose)
        self.eval_env_fn      = eval_env_fn
        self.n_eval_episodes  = n_eval_episodes
        self.eval_every_n_ts  = eval_every_n_ts
        self.save_path        = save_path
        self.deterministic    = deterministic
        self._best_mean_reward = -np.inf
        self._last_eval_ts    = 0
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        ts = self.model.num_timesteps
        if ts - self._last_eval_ts < self.eval_every_n_ts:
            return True
        self._last_eval_ts = ts

        rewards = []
        for ep in range(self.n_eval_episodes):
            env = self.eval_env_fn()
            obs, _ = env.reset(seed=9000 + ep)
            done   = False
            ep_r   = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, r, term, trunc, _ = env.step(int(action))
                ep_r += r
                done  = term or trunc
            env.close()
            rewards.append(ep_r)

        mean_r = float(np.mean(rewards))
        std_r  = float(np.std(rewards))

        if mean_r > self._best_mean_reward:
            self._best_mean_reward = mean_r
            path = os.path.join(self.save_path, "best_model.zip")
            self.model.save(path)
            if self.verbose >= 1:
                logger.info(
                    f"[BestModel] NEW BEST  ts={ts:,}  "
                    f"mean_r={mean_r:.3f} ± {std_r:.3f}  → {path}"
                )
        return True


# =============================================================================
#  3. Multi-Seed Evaluation Callback
#     Evaluates on multiple seeds and logs mean ± std for tracking convergence.
#     This is what the literature (Kangaslahti 2024, Breitfeld 2025) uses.
# =============================================================================

class MultiSeedEvalCallback(BaseCallback):
    """
    Periodic evaluation over multiple seeds.

    Records:
        - mean total reward ± std
        - cloud-free rate ± std
        - dynamic event success rate ± std

    Args:
        make_eval_env_fn : fn(seed) → gym.Env  — factory for eval environments
        eval_seeds       : list of integer seeds to evaluate on
        n_episodes_per_seed : episodes per seed per evaluation
        eval_every_n_ts  : evaluate every N timesteps
        log_path         : path to append JSON log entries
        verbose          : 0 = silent, 1 = print summary
    """

    def __init__(
        self,
        make_eval_env_fn,
        eval_seeds:          List[int] = (42, 123, 456),
        n_episodes_per_seed: int       = 10,
        eval_every_n_ts:     int       = 50_000,
        log_path:            str       = "results/multiseed_eval.json",
        verbose:             int       = 1,
    ):
        super().__init__(verbose)
        self.make_eval_env_fn    = make_eval_env_fn
        self.eval_seeds          = list(eval_seeds)
        self.n_eps_per_seed      = n_episodes_per_seed
        self.eval_every_n_ts     = eval_every_n_ts
        self.log_path            = log_path
        self._last_eval_ts       = 0
        self._eval_log: List[dict] = []
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def _on_step(self) -> bool:
        ts = self.model.num_timesteps
        if ts - self._last_eval_ts < self.eval_every_n_ts:
            return True
        self._last_eval_ts = ts
        self._run_evaluation(ts)
        return True

    def _run_evaluation(self, ts: int) -> None:
        all_rewards   = []
        all_cf_rates  = []
        all_dyn_rates = []
        all_delays    = []

        for seed in self.eval_seeds:
            for ep in range(self.n_eps_per_seed):
                ep_seed = seed * 1000 + ep
                try:
                    env  = self.make_eval_env_fn(ep_seed)
                    obs, _ = env.reset(seed=ep_seed)
                    done, ep_r = False, 0.0
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, r, term, trunc, info = env.step(int(action))
                        ep_r += r
                        done  = term or trunc
                    all_rewards.append(ep_r)

                    # Extract per-episode metrics from the wrapped env
                    try:
                        sat = env.env.unwrapped.satellites[0]
                        m   = sat.get_metrics()
                        cf  = m["n_cloud_free"] / m["n_imaged"] if m["n_imaged"] > 0 else 0.0
                        all_cf_rates.append(cf)

                        dyn_m = env.event_manager.get_metrics()
                        all_dyn_rates.append(dyn_m.get("success_rate", 0.0))
                        all_delays.append(dyn_m.get("avg_delay_s", 0.0))
                    except Exception:
                        all_cf_rates.append(0.0)
                        all_dyn_rates.append(0.0)
                        all_delays.append(0.0)
                    env.close()
                except Exception as exc:
                    logger.warning(f"[MultiSeedEval] episode error: {exc}")

        entry = {
            "timestep":       ts,
            "n_episodes":     len(all_rewards),
            "mean_reward":    float(np.mean(all_rewards))    if all_rewards   else 0.0,
            "std_reward":     float(np.std(all_rewards))     if all_rewards   else 0.0,
            "mean_cf_rate":   float(np.mean(all_cf_rates))   if all_cf_rates  else 0.0,
            "mean_dyn_suc":   float(np.mean(all_dyn_rates))  if all_dyn_rates else 0.0,
            "std_dyn_suc":    float(np.std(all_dyn_rates))   if all_dyn_rates else 0.0,
            "mean_delay_s":   float(np.mean(all_delays))     if all_delays    else 0.0,
        }
        self._eval_log.append(entry)

        with open(self.log_path, "w") as f:
            json.dump(self._eval_log, f, indent=2, default=float)

        if self.verbose >= 1:
            logger.info(
                f"[MultiSeedEval] ts={ts:,}  "
                f"reward={entry['mean_reward']:+.2f}±{entry['std_reward']:.2f}  "
                f"CF={entry['mean_cf_rate']:.0%}  "
                f"dyn_suc={entry['mean_dyn_suc']:.0%}±{entry['std_dyn_suc']:.0%}"
            )


# =============================================================================
#  4. Dynamic Success Monitor Callback
#     Alerts when dynamic success rate has been low for too long.
#     This signals the entropy-collapse / geometry problem early.
# =============================================================================

class DynSuccessMonitorCallback(BaseCallback):
    """
    Monitors dynamic success rate and triggers an alert if it stays below
    threshold for too long. Used for early detection of training problems.

    Args:
        window          : rolling window of episodes to average
        alert_threshold : alert if mean dyn_suc < this value
        alert_every_n_ts: check every N timesteps
        verbose         : 0 = silent, 1 = print alerts
    """

    def __init__(
        self,
        window:           int   = 50,
        alert_threshold:  float = 0.20,
        alert_every_n_ts: int   = 20_000,
        verbose:          int   = 1,
    ):
        super().__init__(verbose)
        self.window          = window
        self.alert_threshold = alert_threshold
        self.alert_every_n_ts = alert_every_n_ts
        self._last_alert_ts  = 0
        self._dyn_success_buffer: List[float] = []

    def _on_rollout_end(self) -> None:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            m = info.get("episode_metrics", {})
            nd  = m.get("n_dyn_detected", 0)
            nim = m.get("n_dyn_imaged",   0)
            if nd > 0:
                self._dyn_success_buffer.append(nim / nd)

        if len(self._dyn_success_buffer) > self.window * 2:
            self._dyn_success_buffer = self._dyn_success_buffer[-self.window:]

    def _on_step(self) -> bool:
        ts = self.model.num_timesteps
        if ts - self._last_alert_ts < self.alert_every_n_ts:
            return True
        self._last_alert_ts = ts

        if not self._dyn_success_buffer:
            return True

        recent = self._dyn_success_buffer[-self.window:]
        mean_suc = float(np.mean(recent))

        if mean_suc < self.alert_threshold and self.verbose >= 1:
            ent = getattr(self.model, "ent_coef", "?")
            logger.warning(
                f"[DynSuccessMonitor] ⚠️  ts={ts:,}  "
                f"dyn_suc={mean_suc:.0%} < threshold={self.alert_threshold:.0%}  "
                f"ent={ent:.4f} — consider restarting with higher entropy or "
                f"checking Keplerian TTA patch availability"
            )
        return True


# =============================================================================
#  5. Improved Training Logger
#     Enhanced version of FullTrainingLogger that logs to JSON every episode
#     and tracks more metrics for analysis.
# =============================================================================

class ImprovedTrainingLogger(BaseCallback):
    """
    Per-episode logger compatible with train_improved.py.

    Logs to JSON (append-friendly) every LOG_EVERY episodes.
    Saves checkpoint every CKPT_EVERY episodes.
    """

    CKPT_EVERY  = 100
    LOG_EVERY   = 10
    VERBOSE_MOD = 50

    def __init__(
        self,
        verbose:  int = 1,
        log_path: str = "results/training_improved.json",
        ckpt_dir: str = "models/checkpoints_improved/",
    ):
        super().__init__(verbose)
        self.ep_rewards      = []
        self.ep_dyn_success  = []
        self.ep_cf_rates     = []
        self.ep_metrics_log  = []
        self._ep_reward      = 0.0
        self._ep_count       = 0
        self._step_actions   = []
        self._log_path       = log_path
        self._ckpt_dir       = ckpt_dir
        os.makedirs(ckpt_dir,                          exist_ok=True)
        os.makedirs(os.path.dirname(log_path) or ".",  exist_ok=True)

    def _on_step(self) -> bool:
        self._ep_reward += float(np.atleast_1d(self.locals.get("rewards", [0.0]))[0])
        for a in np.atleast_1d(self.locals.get("actions", [])):
            self._step_actions.append(int(a))

        done = bool(np.atleast_1d(self.locals.get("dones", [False]))[0])
        if not done:
            return True

        self._ep_count += 1
        infos = self.locals.get("infos", [{}])
        m     = (infos[0].get("episode_metrics", {}) if infos else {})

        ni    = m.get("n_imaged",       0)
        nc    = m.get("n_cloud_free",   0)
        nd    = m.get("n_dyn_detected", m.get("n_detected", 0))
        nim   = m.get("n_dyn_imaged",   0)

        cf = nc  / ni if ni  > 0 else 0.0
        ds = nim / nd if nd  > 0 else 0.0

        total    = max(len(self._step_actions), 1)
        dyn_n    = sum(1 for a in self._step_actions if 20 <= a <= 22)
        drift_n  = self._step_actions.count(23)
        static_n = sum(1 for a in self._step_actions if  0 <= a <= 19)

        self.ep_rewards.append(self._ep_reward)
        self.ep_dyn_success.append(ds)
        self.ep_cf_rates.append(cf)

        ent_coef = float(getattr(self.model, "ent_coef", 0.0))

        entry = {
            "ep":           self._ep_count,
            "reward":       round(self._ep_reward, 4),
            "cf_rate":      round(cf, 4),
            "dyn_suc":      round(ds, 4),
            "n_imaged":     ni,
            "n_cloud_free": nc,
            "n_dyn_det":    nd,
            "n_dyn_img":    nim,
            "dyn_act_pct":  round(100 * dyn_n   / total, 2),
            "drift_pct":    round(100 * drift_n  / total, 2),
            "static_pct":   round(100 * static_n / total, 2),
            "ent_coef":     round(ent_coef, 6),
            "timesteps":    self.model.num_timesteps,
        }
        self.ep_metrics_log.append(entry)

        if self._ep_count % self.LOG_EVERY == 0:
            with open(self._log_path, "w") as f_:
                json.dump({
                    "episodes": self.ep_metrics_log,
                    "summary": {
                        "n_episodes":   self._ep_count,
                        "mean_reward":  round(float(np.mean(self.ep_rewards[-100:])), 3),
                        "mean_cf":      round(float(np.mean(self.ep_cf_rates[-100:])), 4),
                        "mean_dyn_suc": round(float(np.mean(self.ep_dyn_success[-100:])), 4),
                        "ent_coef":     round(ent_coef, 6),
                    }
                }, f_, indent=2, default=float)

        if self._ep_count % self.CKPT_EVERY == 0 and self.model is not None:
            ckpt = os.path.join(self._ckpt_dir, f"ppo_ep{self._ep_count:05d}.zip")
            self.model.save(ckpt)
            if self.verbose >= 1:
                print(f"  [CKPT] Saved → {ckpt}")

        if self.verbose >= 1 and self._ep_count % self.LOG_EVERY == 0:
            avg10 = float(np.mean(self.ep_rewards[-10:]))
            ds10  = float(np.mean(self.ep_dyn_success[-10:]))
            print(
                f"  Ep {self._ep_count:5d} "
                f"r={self._ep_reward:+8.3f}  "
                f"avg100={float(np.mean(self.ep_rewards[-100:])):+7.2f}  "
                f"dyn_suc={ds:.0%}(avg={ds10:.0%})  "
                f"ent={ent_coef:.4f}  "
                f"actions: static={100*static_n//total}%  "
                f"dyn={100*dyn_n//total}%  drift={100*drift_n//total}%"
            )

        self._ep_reward    = 0.0
        self._step_actions = []
        return True


# =============================================================================
#  Builder function — creates the full callback list for training
# =============================================================================

def build_callback_list(
    model_dir:       str,
    targets_path:    str,
    cloud_path:      str,
    total_episodes:  int,
    total_timesteps: int,
    event_rate:      float = 2.0,
    seed:            int   = 42,
    duration_s:      float = 172_800.0,
    start_ent:       float = 0.15,
    end_ent:         float = 0.01,
    decay_fraction:  float = 0.80,
    eval_every_ts:   int   = 50_000,
    eval_seeds:      tuple = (42, 123, 456),
    n_eval_per_seed: int   = 10,
) -> CallbackList:
    """
    Build the standard callback list for improved ALSAT training.

    Args:
        model_dir        : Root directory for saving models and results
        targets_path     : Path to algeria_20_targets.json
        cloud_path       : Path to algeria_real_clouds.json
        total_episodes   : Total training episodes (for step counting)
        total_timesteps  : Total SB3 timesteps (episodes × steps_per_ep)
        event_rate       : Dynamic event rate (events/hour)
        seed             : Training seed
        duration_s       : Episode duration (seconds)
        start_ent        : Initial entropy coefficient
        end_ent          : Final entropy coefficient
        decay_fraction   : Fraction of training over which entropy decays
        eval_every_ts    : Evaluate every N timesteps
        eval_seeds       : Seeds for multi-seed evaluation
        n_eval_per_seed  : Episodes per seed per evaluation
    """
    # Lazy import so this file doesn't need bsk_rl at import time
    def _make_eval_env(eval_seed: int):
        import sys
        for d in ["scripts/core", "scripts"]:
            if d not in sys.path:
                sys.path.insert(0, d)
        from env_alsat_dynamic import make_dynamic_env
        return make_dynamic_env(
            targets_path, cloud_path,
            event_rate=event_rate,
            duration_s=duration_s,
            seed=eval_seed,
        )

    os.makedirs(model_dir, exist_ok=True)
    results_dir = os.path.join(model_dir, "..", "results")
    ckpt_dir    = os.path.join(model_dir, "checkpoints_improved")
    log_path    = os.path.join(results_dir, "training_improved.json")
    mseval_path = os.path.join(results_dir, "multiseed_eval.json")

    callbacks = [
        EntropyAnnealCallback(
            start_ent       = start_ent,
            end_ent         = end_ent,
            decay_fraction  = decay_fraction,
            total_timesteps = total_timesteps,
            verbose         = 1,
        ),
        ImprovedTrainingLogger(
            verbose  = 1,
            log_path = log_path,
            ckpt_dir = ckpt_dir,
        ),
        BestModelCallback(
            eval_env_fn      = lambda: _make_eval_env(9999),
            n_eval_episodes  = 5,
            eval_every_n_ts  = eval_every_ts,
            save_path        = model_dir,
            verbose          = 1,
        ),
        MultiSeedEvalCallback(
            make_eval_env_fn    = _make_eval_env,
            eval_seeds          = list(eval_seeds),
            n_episodes_per_seed = n_eval_per_seed,
            eval_every_n_ts     = eval_every_ts,
            log_path            = mseval_path,
            verbose             = 1,
        ),
        DynSuccessMonitorCallback(
            window           = 50,
            alert_threshold  = 0.20,
            alert_every_n_ts = 30_000,
            verbose          = 1,
        ),
    ]
    return CallbackList(callbacks)
