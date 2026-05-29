#!/usr/bin/env python3
"""
callbacks.py  —  SB3 callbacks for ALSAT-EO-1 Phase 3 training
===============================================================
  EntropyAnnealingCallback   linear entropy decay
  DynamicEventCallback       per-episode DYN metrics + JSON log
  AutoCheckpointCallback     periodic + best-model checkpoints
"""
from __future__ import annotations
import json, math, os, time
from collections import deque
from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# ─────────────────────────────────────────────────────────────────────────────
class EntropyAnnealingCallback(BaseCallback):
    """Linearly decay ent_coef from start_val → end_val over total_timesteps."""

    def __init__(self, start_val: float = 0.15, end_val: float = 0.01,
                 total_timesteps: int = 288000, verbose: int = 0):
        super().__init__(verbose)
        self.start_val       = start_val
        self.end_val         = end_val
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total_timesteps)
        frac = min(1.0, self.num_timesteps / max(self.total_timesteps, 1))
        new_ent = self.start_val + frac * (self.end_val - self.start_val)
        self.model.ent_coef = float(new_ent)
        return True


# ─────────────────────────────────────────────────────────────────────────────
class DynamicEventCallback(BaseCallback):
    """
    Tracks per-episode DYN metrics and writes training_log.json.

    JSON format (one entry per episode):
      {
        "episode": 42,
        "timestep": 123456,
        "wall_time_s": 300.1,
        "reward": 15.3,
        "n_imaged": 8,
        "n_dyn_detected": 12,
        "n_dyn_imaged": 4,
        "dyn_success_rate": 0.333,
        "cf_rate": 0.75,
        "n_cloudy": 2,
        "total_slew_angle_deg": 320.5,
        "total_slew_energy_wh": 1.2,
        "ent_coef": 0.04
      }
    """

    def __init__(self, log_dir: str = "results", log_every: int = 1,
                 window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir    = log_dir
        self.log_every  = log_every
        self.window     = window

        self._t0             = time.time()
        self._episode        = 0
        self._log: list[dict] = []
        self.dyn_success_history: list[float] = []
        self._reward_window  = deque(maxlen=window)
        self._dyn_suc_window = deque(maxlen=window)
        self._json_path: Optional[str] = None

        self._event_log = []
        self._log_path = os.path.join("results", "training_live.json")

        self.ep_rewards: list = []
        self.ep_dyn_success: list = []
        self.ep_cf_rates: list = []

    def _on_training_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._json_path = os.path.join(self.log_dir, "training_log.json")
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.locals.get("dones") is None:
            return True

        for i, done in enumerate(self.locals["dones"]):
            info = (self.locals.get("infos") or [{}])[i]
            ep_m = info.get("episode_metrics", info.get("episode", {}))
            r    = float(info.get("episode", {}).get("r",
                         info.get("total_reward", 0.0)))

            ni   = int(ep_m.get("n_imaged", 0))
            nd   = int(ep_m.get("n_dyn_detected", 0))
            ndi  = int(ep_m.get("n_dyn_imaged", 0))
            nc   = int(ep_m.get("n_cloud_free", ni))
            ncl  = int(ep_m.get("n_cloudy", 0))
            slew = float(ep_m.get("total_slew_angle_deg", 0.0))
            egy  = float(ep_m.get("total_slew_energy_wh", 0.0))

            dyn_suc = ndi / nd if nd > 0 else 0.0
            cf_rate = nc / ni  if ni > 0 else 0.0

            if not done:
                continue   # skip until episode ends – don't append partial rewards

            self._episode += 1

            self._reward_window.append(r)
            self._dyn_suc_window.append(dyn_suc)
            self.dyn_success_history.append(dyn_suc)

            # ── Append episode-level metrics exactly once per episode ──
            self.ep_rewards.append(r)
            self.ep_dyn_success.append(dyn_suc)
            self.ep_cf_rates.append(cf_rate)

            entry = {
                "episode":           self._episode,
                "timestep":          int(self.num_timesteps),
                "wall_time_s":       round(time.time() - self._t0, 1),
                "reward":            round(r, 4),
                "n_imaged":          ni,
                "n_dyn_detected":    nd,
                "n_dyn_imaged":      ndi,
                "dyn_success_rate":  round(dyn_suc, 4),
                "cf_rate":           round(cf_rate, 4),
                "n_cloudy":          ncl,
                "total_slew_deg":    round(slew, 2),
                "total_slew_energy": round(egy, 4),
                "ent_coef":          round(float(getattr(self.model, "ent_coef", 0)), 5),
                "ep_rewards":        self.ep_rewards[-self.window:],
                "ep_dyn_success":    self.ep_dyn_success[-self.window:],
                "ep_cf_rates":       self.ep_cf_rates[-self.window:],
            }
            self._log.append(entry)

            if self._episode % self.log_every == 0 and self._json_path:
                with open(self._json_path, "w") as f:
                    json.dump(self._log, f, indent=2)

            if self.verbose >= 1 and self._episode % 25 == 0:
                mean_r   = np.mean(self._reward_window)
                mean_dyn = np.mean(self._dyn_suc_window)
                print(
                    f"  Ep {self._episode:5d}  "
                    f"r={mean_r:+7.3f}  "
                    f"dyn_suc={mean_dyn:.1%}  "
                    f"n_dyn_img={ndi}  "
                    f"ent={entry['ent_coef']:.4f}"
                )

            # ── Rich per-episode summary (fires at episode end) ──────────────
            ep = self.n_calls // (self.locals.get("n_steps", 2048))
            r_sum = self.locals.get("rewards", [0])
            ep_r = float(np.sum(r_sum)) if hasattr(r_sum, '__len__') else float(r_sum)

            sat = None
            try:
                if hasattr(self, 'training_env') and self.training_env is not None:
                    sat = self.training_env.unwrapped.satellites[0]
            except Exception:
                pass

            if sat is not None:
                m = sat._metrics
                n_det = m.get("n_dyn_detected", 0)
                n_img = m.get("n_dyn_imaged", 0)
                n_cf = m.get("n_cloud_free", 0)
                batt = "?"
                try:
                    batt = f"{sat.dynamics.battery_charge_fraction:.0%}"
                except:
                    pass

                last_evt = getattr(sat, "_last_dyn_event_log", None)
                evt_str = ""
                if last_evt:
                    evt_str = (f" | DYN: [{last_evt.get('type','?')}] "
                               f"{last_evt.get('lat',0):.1f}°N {last_evt.get('lon',0):.1f}°E "
                               f"prio={last_evt.get('priority',0):.2f} "
                               f"cloud={last_evt.get('cloud',0):.2f} "
                               f"r={last_evt.get('reward',0):+.2f}")
                last_tgt = getattr(sat, "_last_static_log", None)
                tgt_str = ""
                if last_tgt:
                    tgt_str = f" | STATIC: {last_tgt.get('name','?')} cloud={last_tgt.get('cloud',0):.2f}"

                suc_rate = f"{n_img/max(n_det,1):.0%}"
                print(
                    f" Ep {ep:4d} r={ep_r:+8.3f} "
                    f"dyn={n_img}/{n_det}({suc_rate}) "
                    f"cf={n_cf} batt={batt}"
                    f"{evt_str}{tgt_str}"
                )

            # ── Write live JSON for monitor (only at episode end) ──
            log_data = {
                "episode_rewards": self.ep_rewards,
                "ep_dyn_success": self.ep_dyn_success,
                "ep_cf_rates": self.ep_cf_rates,
                "event_log": self._event_log,
                "variant": getattr(self, "_variant", "full_system"),
                "seed": getattr(self, "_seed", 42),
            }
            with open(self._log_path, "w") as f:
                json.dump(log_data, f, indent=2)

        return True

    def _on_training_end(self) -> None:
        if self._json_path and self._log:
            with open(self._json_path, "w") as f:
                json.dump(self._log, f, indent=2)
            print(f"\n✓ Training log saved → {self._json_path}  ({len(self._log)} episodes)")


# ─────────────────────────────────────────────────────────────────────────────
class AutoCheckpointCallback(BaseCallback):
    """
    Saves model every save_freq timesteps + tracks best mean reward.
    Directory layout:
      {save_dir}/
        ckpt_{exp_id}_step{N}.zip
        best_{exp_id}.zip          ← best by 100-ep mean reward
        checkpoint_meta.json       ← all checkpoint metadata
    """

    def __init__(self, save_freq: int = 100_000, save_dir: str = "checkpoints",
                 exp_id: str = "alsat", extra_meta: dict = None,
                 reward_window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq     = save_freq
        self.save_dir      = save_dir
        self.exp_id        = exp_id
        self.extra_meta    = extra_meta or {}
        self.reward_window = reward_window

        self._best_mean   = -math.inf
        self._ep_rewards: deque = deque(maxlen=reward_window)
        self._meta: list[dict]  = []
        self._last_save   = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Collect episode rewards
        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", [])
        ):
            if done:
                r = float(info.get("episode", {}).get("r", 0.0))
                self._ep_rewards.append(r)

                # try:
                #     sat = self.training_env.unwrapped.satellites[0]
                #     m = sat._metrics
                #     n_det = m.get("n_dyn_detected", 0)
                #     n_img = m.get("n_dyn_imaged", 0)
                #     n_cf = m.get("n_cloud_free", 0)
                #     batt = f"{sat.dynamics.battery_charge_fraction:.0%}"
                #     suc_rate = f"{n_img/max(n_det,1):.0%}"
                #     last_evt = getattr(sat, "_last_dyn_event_log", None)
                #     evt_str = ""
                #     if last_evt:
                #         evt_str = f" | DYN: {last_evt.get('type','?')} {last_evt.get('lat',0):.1f}°N {last_evt.get('lon',0):.1f}°E prio={last_evt.get('priority',0):.2f} r={last_evt.get('reward',0):+.2f}"
                #     print(f" Ep {self._episode:4d} r={self._ep_reward:+8.3f} dyn={n_img}/{n_det}({suc_rate}) cf={n_cf} batt={batt}{evt_str}")
                # except Exception:
                #     pass

        # Periodic checkpoint
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._save_checkpoint()
            self._last_save = self.num_timesteps

        # Best model checkpoint
        if len(self._ep_rewards) >= 20:
            mean_r = float(np.mean(self._ep_rewards))
            if mean_r > self._best_mean:
                self._best_mean = mean_r
                best_path = os.path.join(self.save_dir, f"best_{self.exp_id}")
                self.model.save(best_path)
                if self.verbose >= 1:
                    print(f"  ★ New best: {mean_r:+.3f}  → {best_path}.zip")

        return True

    def _save_checkpoint(self) -> None:
        fname = f"ckpt_{self.exp_id}_step{self.num_timesteps}"
        path  = os.path.join(self.save_dir, fname)
        self.model.save(path)
        mean_r = float(np.mean(self._ep_rewards)) if self._ep_rewards else 0.0
        entry = {
            "step": int(self.num_timesteps),
            "path": path + ".zip",
            "mean_reward_100ep": round(mean_r, 4),
            **self.extra_meta
        }
        self._meta.append(entry)
        meta_path = os.path.join(self.save_dir, "checkpoint_meta.json")
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)
        if self.verbose >= 1:
            print(f"  [ckpt] step={self.num_timesteps:,}  mean_r={mean_r:+.3f}  → {path}.zip")

    def _on_training_end(self) -> None:
        self._save_checkpoint()

import math as _math

ACTION_NAMES = {
    **{i: f"STATIC-T{i:02d}" for i in range(20)},
    20: "DYN-slot0", 21: "DYN-slot1", 22: "DYN-slot2", 23: "DRIFT",
}
EVENT_ICONS = {"wildfire":"🔥","flood":"🌊","plume":"💨","earthquake":"⚡","eruption":"🌋"}

class VerboseActionLogger(BaseCallback):
    """
    Prints per-step action details: target name, cloud forecast, reward, success.
    Enable with --verbose-actions flag. Prints every `print_every` steps.
    """
    def __init__(self, print_every: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self._ep = 0
        self._step = 0
        self._print_every = print_every
        self._ep_reward = 0.0
        self._ep_static = 0
        self._ep_dyn_det = 0
        self._ep_dyn_img = 0
        self._ep_imaged = 0

    def _on_step(self) -> bool:
        actions = np.atleast_1d(self.locals.get("actions", []))
        rewards = np.atleast_1d(self.locals.get("rewards", [0.0]))
        dones = np.atleast_1d(self.locals.get("dones", [False]))
        infos = self.locals.get("infos", [{}])

        for i, (action, reward, done, info) in enumerate(zip(actions, rewards, dones, infos)):
            action = int(action)
            reward = float(reward)
            self._step += 1
            self._ep_reward += reward

            if self._step % self._print_every == 0:
                # Try to get satellite state for rich logging
                try:
                    env = self.training_env
                    sat = env.unwrapped.satellites[0] if hasattr(env.unwrapped, 'satellites') else None
                    if sat is None:
                        # Try unwrapping further
                        e = env
                        while hasattr(e, 'env'): e = e.env
                        while hasattr(e, 'envs'): e = e.envs[0]
                        if hasattr(e, 'unwrapped') and hasattr(e.unwrapped, 'satellites'):
                            sat = e.unwrapped.satellites[0]
                except Exception:
                    sat = None

                # Build action description
                if 20 <= action <= 22:
                    slot = action - 20
                    evt_str = f"DYN-slot{slot}"
                    try:
                        mgr = getattr(sat, '_event_manager', None)
                        now = float(sat.simulator.sim_time)
                        slots = mgr.get_slots(sat, now) if mgr else []
                        evt = slots[slot] if slot < len(slots) else None
                        if evt:
                            icon = EVENT_ICONS.get(evt.event_type, "📍")
                            lat = _math.degrees(evt.lat_rad)
                            lon = _math.degrees(evt.lon_rad)
                            cf = evt.cloud_cover_forecast
                            cloud_label = "CLEAR ✅" if cf < 0.3 else ("CLOUDY ❌" if cf > 0.7 else "PARTIAL ⚠️")
                            evt_str = (f"{icon} {evt.event_type.upper()} "
                                       f"lat={lat:.1f}°N lon={lon:.1f}°E "
                                       f"prio={evt.priority:.2f} "
                                       f"cloud_fcst={cf:.2f}({cloud_label})")
                    except Exception:
                        pass
                    print(f" [Step {self._step:5d}] ACT={action}({evt_str}) r={reward:+.3f}")

                elif action <= 19:
                    tgt_str = f"STATIC-T{action}"
                    try:
                        tgt = sat.scenario.targets[action]
                        cf = getattr(tgt, 'cloud_cover_forecast', 0.0)
                        cloud_label = "CLEAR ✅" if cf < 0.3 else ("CLOUDY ❌" if cf > 0.7 else "~")
                        tgt_str = (f"STATIC [{getattr(tgt,'name',f'T{action}')}] "
                                   f"prio={getattr(tgt,'priority',0):.2f} "
                                   f"cloud_fcst={cf:.2f}({cloud_label})")
                    except Exception:
                        pass
                    if reward > 0.01:
                        print(f" [Step {self._step:5d}] ACT={action}({tgt_str}) r={reward:+.3f} ✅")

                elif action == 23:
                    pass  # suppress drift steps

            if done:
                self._ep += 1
                m = info.get("episode_metrics", {}) if info else {}
                nd = m.get("n_dyn_detected", 0)
                nim = m.get("n_dyn_imaged", 0)
                ni = m.get("n_imaged", 0)
                nc = m.get("n_cloud_free", 0)
                ent = round(float(getattr(self.model, "ent_coef", 0)), 4)
                print(
                    f"\n {'═'*64}\n"
                    f" EPISODE {self._ep} DONE r={self._ep_reward:+.2f} "
                    f"static={ni-nim}/{ni-nim+20} "
                    f"dyn={nim}/{nd}({'0%' if nd==0 else f'{100*nim//nd}%'}) "
                    f"cf={nc}/{ni} ent={ent}\n"
                    f" {'═'*64}"
                )
                self._ep_reward = 0.0
                self._step = 0

        return True