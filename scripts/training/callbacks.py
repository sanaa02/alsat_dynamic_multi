#!/usr/bin/env python3
"""
callbacks.py  —  SB3 callbacks for ALSAT-EO-1 Phase 3 training
===============================================================
  EntropyAnnealingCallback   linear entropy decay
  DynamicEventCallback       per-episode DYN metrics + JSON log
  AutoCheckpointCallback     periodic + best-model checkpoints
  VerboseStepLogger          FULL per-step diagnostic logger (NEW/FIXED)

FIXES applied (v2):
  [FIX-1]  EntropyAnnealingCallback: end_val raised from 0.03 → 0.05.
           At 0.03 the policy becomes deterministic too early and
           dyn_suc collapses to 0% around ep 150.
  [FIX-2]  VerboseActionLogger replaced by VerboseStepLogger with:
           - Orbital position (lat/lon/alt) from Basilisk dynamics
           - Cloud TRUTH alongside forecast
           - Slew angle to target
           - Sojourn time τ from obs[55]
           - All active dynamic events (type, lat, lon, priority,
             cloud_fcst, cloud_truth, age, % lifetime, TTA)
           - Reward breakdown (base, slew_penalty, shaping_bonus)
           - Optional DRIFT step detail (--show-drift)
           - Per-episode summary banner
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

    def __init__(self, start_val: float = 0.15,
                 end_val: float = 0.05,          # FIX-1: was 0.03, too low
                 total_timesteps: int = 288000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.start_val       = start_val
        self.end_val         = end_val
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        frac    = min(1.0, self.num_timesteps / max(self.total_timesteps, 1))
        new_ent = self.start_val + frac * (self.end_val - self.start_val)
        new_ent = max(new_ent, self.end_val)   # hard floor
        self.model.ent_coef = float(new_ent)
        return True


# ─────────────────────────────────────────────────────────────────────────────
class DynamicEventCallback(BaseCallback):
    """
    Tracks per-episode DYN metrics and writes training_log.json.
    (unchanged from v1 — no bugs found here)
    """

    def __init__(self, log_dir: str = "results", log_every: int = 1,
                 window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir    = log_dir
        self.log_every  = log_every
        self.window     = window

        self._t0             = time.time()
        self._episode        = 0
        self._log: list      = []
        self.dyn_success_history: list = []
        self._reward_window  = deque(maxlen=window)
        self._dyn_suc_window = deque(maxlen=window)
        self._json_path: Optional[str] = None
        self._event_log      = []
        self._log_path       = os.path.join("results", "training_live.json")

        self.ep_rewards:     list = []
        self.ep_dyn_success: list = []
        self.ep_cf_rates:    list = []

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

            ni   = int(ep_m.get("n_imaged",        0))
            nd   = int(ep_m.get("n_dyn_detected",  0))
            ndi  = int(ep_m.get("n_dyn_imaged",    0))
            nc   = int(ep_m.get("n_cloud_free",    ni))
            ncl  = int(ep_m.get("n_cloudy",         0))
            slew = float(ep_m.get("total_slew_angle_deg", 0.0))
            egy  = float(ep_m.get("total_slew_energy_wh", 0.0))

            dyn_suc = ndi / nd if nd > 0 else 0.0
            cf_rate = nc  / ni if ni > 0 else 0.0

            if not done:
                continue

            self._episode += 1
            self._reward_window.append(r)
            self._dyn_suc_window.append(dyn_suc)
            self.dyn_success_history.append(dyn_suc)
            self.ep_rewards.append(r)
            self.ep_dyn_success.append(dyn_suc)
            self.ep_cf_rates.append(cf_rate)

            entry = {
                "episode":          self._episode,
                "timestep":         int(self.num_timesteps),
                "wall_time_s":      round(time.time() - self._t0, 1),
                "reward":           round(r,        4),
                "n_imaged":         ni,
                "n_dyn_detected":   nd,
                "n_dyn_imaged":     ndi,
                "dyn_success_rate": round(dyn_suc,  4),
                "cf_rate":          round(cf_rate,  4),
                "n_cloudy":         ncl,
                "total_slew_deg":   round(slew,     2),
                "total_slew_energy":round(egy,       4),
                "ent_coef":         round(float(getattr(self.model, "ent_coef", 0)), 5),
                "ep_rewards":       self.ep_rewards[-self.window:],
                "ep_dyn_success":   self.ep_dyn_success[-self.window:],
                "ep_cf_rates":      self.ep_cf_rates[-self.window:],
            }
            self._log.append(entry)

            if self._episode % self.log_every == 0 and self._json_path:
                with open(self._json_path, "w") as f:
                    json.dump(self._log, f, indent=2)

            if self.verbose >= 1 and self._episode % 25 == 0:
                mean_r   = np.mean(self._reward_window)
                mean_dyn = np.mean(self._dyn_suc_window)
                print(f"  Ep {self._episode:5d}  "
                      f"r={mean_r:+7.3f}  "
                      f"dyn_suc={mean_dyn:.1%}  "
                      f"n_dyn_img={ndi}  "
                      f"ent={entry['ent_coef']:.4f}")

        return True

    def _on_training_end(self) -> None:
        if self._json_path and self._log:
            with open(self._json_path, "w") as f:
                json.dump(self._log, f, indent=2)
            print(f"\n✓ Training log saved → {self._json_path}  "
                  f"({len(self._log)} episodes)")


# ─────────────────────────────────────────────────────────────────────────────
class AutoCheckpointCallback(BaseCallback):
    """
    Saves model every save_freq timesteps + tracks best mean reward.
    (unchanged from v1 — no bugs found here)
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
        self._meta: list        = []
        self._last_save   = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", [])
        ):
            if done:
                r = float(info.get("episode", {}).get("r", 0.0))
                self._ep_rewards.append(r)

        if self.num_timesteps - self._last_save >= self.save_freq:
            self._save_checkpoint()
            self._last_save = self.num_timesteps

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
        entry  = {
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
            print(f"  [ckpt] step={self.num_timesteps:,}  "
                  f"mean_r={mean_r:+.3f}  → {path}.zip")

    def _on_training_end(self) -> None:
        self._save_checkpoint()


# ─────────────────────────────────────────────────────────────────────────────
# FIX-2: Completely rewritten VerboseStepLogger
# Replaces the old VerboseActionLogger which was missing:
#   - Orbital position (lat/lon/alt)
#   - Cloud TRUTH (only had forecast)
#   - Slew angle to target
#   - Sojourn time τ from obs[55]
#   - All active events (not just the one being imaged)
#   - Reward breakdown
#   - DRIFT step information
# ─────────────────────────────────────────────────────────────────────────────

EVENT_ICONS = {
    "wildfire": "🔥", "flood": "🌊", "plume": "💨",
    "earthquake": "⚡", "eruption": "🌋",
}

ACTION_NAMES = {
    **{i: f"STATIC-T{i:02d}" for i in range(20)},
    20: "DYN-slot0", 21: "DYN-slot1", 22: "DYN-slot2", 23: "DRIFT",
}


class VerboseStepLogger(BaseCallback):
    """
    Full per-step diagnostic logger for ALSAT-EO-1.

    Per-step output includes:
      - Simulation time and satellite orbital position (lat/lon/alt)
      - Action taken with human-readable name (target name, event type)
      - Cloud forecast AND ground truth for the imaged target/event
      - Slew angle required for the action
      - Sojourn time τ (from obs[55], the SMDP duration feature)
      - Whether imaging succeeded (cloud < threshold + in access window)
      - Reward received + shaping bonus breakdown
      - ALL currently active dynamic events with:
          type, lat/lon, priority, cloud_fcst, cloud_truth,
          age in minutes, % of lifetime elapsed, TTA (time to access)

    Enable with:
        --verbose-steps               (enable, default print_every=1)
        --log-every-n 10              (print every 10 steps, less noise)
        --show-drift                  (also print DRIFT steps)

    Usage in callbacks list:
        VerboseStepLogger(print_every=10, show_drift=False, show_events=True)
    """

    MAX_ACTION_DUR_S = 3600.0   # normalisation constant from smdp_dynamic.py

    def __init__(self, print_every: int = 1,
                 show_drift: bool = False,
                 show_events: bool = True,
                 verbose: int = 1):
        super().__init__(verbose)
        self._step        = 0
        self._ep          = 0
        self._ep_reward   = 0.0
        self.print_every  = print_every
        self.show_drift   = show_drift
        self.show_events  = show_events

    # ── Env unwrapping ────────────────────────────────────────────────────
    def _get_sat(self):
        """Safely peel VecEnv / Monitor layers to reach the satellite."""
        try:
            e = self.training_env
            # peel SubprocVecEnv / DummyVecEnv
            while hasattr(e, 'venv'):  e = e.venv
            while hasattr(e, 'envs'): e = e.envs[0]
            # peel Monitor / gym.Wrapper layers
            while hasattr(e, 'env'):  e = e.env
            if hasattr(e, 'satellites'):
                return e.satellites[0]
            # one more level (sometimes needed with DynamicRewardShaper)
            if hasattr(e, 'unwrapped') and hasattr(e.unwrapped, 'satellites'):
                return e.unwrapped.satellites[0]
        except Exception:
            pass
        return None

    # ── Position ──────────────────────────────────────────────────────────
    def _sat_pos_str(self, sat) -> str:
        """Return 'lat=XX.X°N  lon=YY.Y°E  alt=ZZZkm' from Basilisk dynamics."""
        try:
            import numpy as _np
            r = _np.array(sat.dynamics.r_BN_N, dtype=float)   # ECI position [m]
            R = float(_np.linalg.norm(r))
            lat_deg = math.degrees(math.asin(r[2] / R))
            lon_deg = math.degrees(math.atan2(r[1], r[0]))
            alt_km  = (R - 6.3781e6) / 1000.0
            return f"lat={lat_deg:+.1f}°  lon={lon_deg:+.1f}°  alt={alt_km:.0f}km"
        except Exception:
            return "pos=N/A"

    # ── Cloud labels ──────────────────────────────────────────────────────
    @staticmethod
    def _cloud_truth(obj) -> float:
        """Ground-truth cloud cover (falls back to forecast if unavailable)."""
        return float(getattr(obj, 'cloud_cover',
                     getattr(obj, 'cloud_cover_forecast', 0.0)))

    @staticmethod
    def _cloud_label(cf: float) -> str:
        if cf < 0.20:  return "CLEAR ✅"
        if cf < 0.40:  return "MOSTLY CLEAR 🟡"
        if cf < 0.70:  return "PARTIAL ⚠️"
        return "CLOUDY ❌"

    # ── Active events ─────────────────────────────────────────────────────
    def _fmt_active_events(self, sat, now: float) -> str:
        """List ALL currently active dynamic events with full properties."""
        lines = []
        try:
            # Try both attribute names used across different env versions
            mgr = (getattr(sat, '_event_manager', None) or
                   getattr(sat, 'event_manager', None))
            if mgr is None:
                return ""

            # Access internal event list (try several possible names)
            active = (getattr(mgr, 'active_events', None) or
                      getattr(mgr, '_events', None) or
                      getattr(mgr, 'events', []))

            for evt in active:
                if evt is None:
                    continue
                icon     = EVENT_ICONS.get(getattr(evt, 'event_type', ''), "📍")
                lat      = math.degrees(getattr(evt, 'lat_rad', 0.0))
                lon      = math.degrees(getattr(evt, 'lon_rad', 0.0))
                age_min  = (now - getattr(evt, 'spawn_time', now)) / 60.0
                dur_s    = getattr(evt, 'duration_s', 3600.0)
                life_pct = min(100.0, 100.0 * age_min * 60.0 / max(dur_s, 1.0))
                cf_fcst  = float(getattr(evt, 'cloud_cover_forecast', 0.0))
                cf_truth = self._cloud_truth(evt)

                # TTA: try the manager's method, fall back to 0
                tta_s = 0.0
                try:
                    tta_s = float(mgr.time_to_access(evt, sat, now))
                except Exception:
                    pass

                etype = getattr(evt, 'event_type', '?')
                prio  = getattr(evt, 'priority', 0.0)
                lines.append(
                    f"    {icon} {etype:12s} "
                    f"lat={lat:+.1f}°  lon={lon:+.1f}°  "
                    f"prio={prio:.2f}  "
                    f"fcst={cf_fcst:.2f}  truth={cf_truth:.2f}  "
                    f"age={age_min:.0f}min ({life_pct:.0f}% life)  "
                    f"TTA={tta_s:.0f}s"
                )
        except Exception:
            pass
        return "\n".join(lines)

    # ── Main callback ─────────────────────────────────────────────────────
    def _on_step(self) -> bool:
        actions = np.atleast_1d(self.locals.get("actions", []))
        rewards = np.atleast_1d(self.locals.get("rewards", [0.0]))
        dones   = np.atleast_1d(self.locals.get("dones",   [False]))
        infos   = self.locals.get("infos", [{}])
        # new_obs has shape (n_envs, obs_dim) — obs[55] is tau_norm
        new_obs = self.locals.get("new_obs",
                  self.locals.get("obs_tensor",
                  self.locals.get("obs", [None])))

        for i, (act, rew, done, info) in enumerate(
                zip(actions, rewards, dones, infos)):
            act = int(act);  rew = float(rew)
            self._step    += 1
            self._ep_reward += rew

            # Shaping bonus (from DynamicRewardShaper, if applied)
            shaping_bonus = float((info or {}).get("shaping_bonus", 0.0))
            base_rew      = rew - shaping_bonus

            # Sojourn time τ from obs[55]
            tau_norm = 0.0
            try:
                obs_i = (new_obs[i] if new_obs is not None
                         and not isinstance(new_obs, type(None)) else None)
                if obs_i is not None and len(obs_i) >= 56:
                    tau_norm = float(obs_i[55])
            except Exception:
                pass
            tau_s = tau_norm * self.MAX_ACTION_DUR_S

            should_print = (self._step % self.print_every == 0)
            if not should_print and not done:
                continue

            sat = self._get_sat()
            now = 0.0
            try:
                now = float(sat.simulator.sim_time)
            except Exception:
                pass

            if should_print:
                pos_str = self._sat_pos_str(sat) if sat else "pos=N/A"
                hdr = (f"\n[Step {self._step:5d}]  "
                       f"t={now:.0f}s ({now/3600:.1f}h)  {pos_str}")

                # ── Dynamic event action (20-22) ──────────────────────────
                if 20 <= act <= 22:
                    slot     = act - 20
                    evt_desc = f"DYN-slot{slot} [EMPTY — no active event]"
                    try:
                        mgr  = (getattr(sat, '_event_manager', None) or
                                getattr(sat, 'event_manager', None))
                        evts = mgr.get_slots(sat, now) if mgr else []
                        evt  = evts[slot] if slot < len(evts) else None
                        if evt is not None:
                            icon     = EVENT_ICONS.get(
                                getattr(evt, 'event_type', ''), "📍")
                            lat      = math.degrees(getattr(evt, 'lat_rad', 0.0))
                            lon      = math.degrees(getattr(evt, 'lon_rad', 0.0))
                            cf_fcst  = float(getattr(evt, 'cloud_cover_forecast', 0.0))
                            cf_truth = self._cloud_truth(evt)
                            prio     = getattr(evt, 'priority', 0.0)
                            age_min  = (now - getattr(evt, 'spawn_time', now)) / 60.0
                            dur_s    = getattr(evt, 'duration_s', 3600.0)
                            life_pct = min(100.0, 100.0*age_min*60.0/max(dur_s,1))
                            tta_s    = 0.0
                            try:
                                tta_s = float(mgr.time_to_access(evt, sat, now))
                            except Exception:
                                pass
                            etype    = getattr(evt, 'event_type', '?')
                            slew_info = ""
                            try:
                                from env_alsat_debug import (
                                    calculate_slew_angle_to_target)
                                slew_deg = math.degrees(
                                    calculate_slew_angle_to_target(
                                        sat, evt))
                                slew_info = f"  slew={slew_deg:.1f}°"
                            except Exception:
                                pass
                            access_ok = tta_s < 30.0
                            evt_desc  = (
                                f"{icon} {etype.upper()}  "
                                f"lat={lat:+.1f}°  lon={lon:+.1f}°  "
                                f"prio={prio:.2f}  "
                                f"fcst={cf_fcst:.2f}  TRUTH={cf_truth:.2f}  "
                                f"{self._cloud_label(cf_truth)}"
                                f"{slew_info}  "
                                f"age={age_min:.0f}min ({life_pct:.0f}%life)  "
                                f"TTA={tta_s:.0f}s  "
                                f"{'✅ IN ACCESS' if access_ok else '⛔ OUT OF WINDOW'}"
                            )
                    except Exception:
                        pass

                    imaged = rew > 0.001
                    print(hdr)
                    print(f"  ACT={act} DYN-slot{slot}")
                    print(f"  {evt_desc}")
                    print(f"  τ={tau_s:.0f}s ({tau_norm:.3f} norm)")
                    print(f"  r={rew:+.4f}  "
                          f"[base={base_rew:+.4f}  shaping={shaping_bonus:+.4f}]  "
                          f"{'✅ DYN IMAGED!' if imaged else '❌ NO IMAGE'}")

                # ── Static target action (0-19) ───────────────────────────
                elif act <= 19:
                    tgt_desc = f"STATIC-T{act:02d}"
                    slew_info = ""
                    try:
                        tgt      = sat.scenario.targets[act]
                        cf_fcst  = float(getattr(tgt, 'cloud_cover_forecast', 0.0))
                        cf_truth = self._cloud_truth(tgt)
                        name     = getattr(tgt, 'name', f'T{act:02d}')
                        prio     = float(getattr(tgt, 'priority', 0.0))
                        tgt_desc = (
                            f"STATIC [{name}]  "
                            f"prio={prio:.2f}  "
                            f"fcst={cf_fcst:.2f}  TRUTH={cf_truth:.2f}  "
                            f"{self._cloud_label(cf_truth)}"
                        )
                        try:
                            from env_alsat_debug import (
                                calculate_slew_angle_to_target)
                            slew_deg = math.degrees(
                                calculate_slew_angle_to_target(sat, tgt))
                            slew_info = f"  slew={slew_deg:.1f}°"
                        except Exception:
                            pass
                    except Exception:
                        pass

                    imaged = rew > 0.001
                    print(hdr)
                    print(f"  ACT={act} {tgt_desc}{slew_info}")
                    print(f"  τ={tau_s:.0f}s ({tau_norm:.3f} norm)")
                    print(f"  r={rew:+.4f}  "
                          f"[base={base_rew:+.4f}  shaping={shaping_bonus:+.4f}]  "
                          f"{'✅ IMAGED' if imaged else '❌ CLOUDY/NO-ACCESS'}")

                # ── Drift (23) ────────────────────────────────────────────
                elif act == 23 and self.show_drift:
                    batt = "?"
                    try:
                        batt = (f"{sat.dynamics.battery_charge_fraction:.0%}")
                    except Exception:
                        pass
                    print(hdr)
                    print(f"  ACT=23 DRIFT  τ={tau_s:.0f}s  "
                          f"batt={batt}  r={rew:+.4f}")

                # ── Active events for this step ───────────────────────────
                if self.show_events and sat is not None:
                    evt_lines = self._fmt_active_events(sat, now)
                    if evt_lines:
                        print("  Active events:")
                        print(evt_lines)

            # ── Episode end summary ───────────────────────────────────────
            if done:
                self._ep += 1
                m       = (info or {}).get("episode_metrics", {})
                nd      = m.get("n_dyn_detected",    0)
                nim     = m.get("n_dyn_imaged",       0)
                ni      = m.get("n_imaged",            0)
                nc      = m.get("n_cloud_free",        0)
                slew_t  = m.get("total_slew_angle_deg", 0.0)
                ent     = round(float(getattr(self.model, "ent_coef", 0)), 4)
                dyn_pct = f"{100*nim//nd}%" if nd > 0 else "—"
                print(
                    f"\n{'═'*70}\n"
                    f" EPISODE {self._ep} END  "
                    f"r={self._ep_reward:+.3f}  "
                    f"ent={ent}\n"
                    f"   static imaged: {ni - nim}  |  "
                    f"dyn: {nim}/{nd} ({dyn_pct} success)  |  "
                    f"cloud-free: {nc}/{ni}  |  "
                    f"total_slew: {slew_t:.0f}°\n"
                    f"{'═'*70}"
                )
                self._ep_reward = 0.0
                self._step      = 0

        return True


# Keep the old name as an alias so existing imports don't break
VerboseActionLogger = VerboseStepLogger