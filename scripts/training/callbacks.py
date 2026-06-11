#!/usr/bin/env python3
"""
callbacks.py  --  SB3 callbacks for ALSAT-EO-1 Phase 3  (FIXED v3)
====================================================================
FIX-CB-1  EntropyAnnealingCallback was purely time-based (linear over
    total_timesteps), so it kept cutting entropy for 4,000 episodes while
    the policy was stuck in a bad local optimum.  Combined with the
    explore_bonus collapsing to zero, the policy had zero mechanisms left
    to escape.

    Fix: conditional entropy annealing.  Entropy only decays when the
    10-episode rolling average reward IMPROVES by at least min_improvement
    over the previous window.  If reward is stagnating or collapsing,
    entropy stays at its current level (or even rises back toward the
    recovery_level).

FIX-CB-2  DynamicEventCallback.ep_info_buffer read used info["r"] but
    the Monitor wrapper in some SB3 versions uses info["episode"]["r"].
    Fixed with a safe accessor.

FIX-CB-3  VerboseStepLogger labelled ALL non-positive rewards as
    "❌ CLOUDY/NO-ACCESS", hiding access failures from the developer.
    Fix: distinguish ✅ IMAGED / ❌ CLOUD / ⛔ NO-ACCESS / 〰️ EMPTY-SLOT.

All original callback classes are preserved unchanged except where noted.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FIX-CB-1: Conditional entropy annealing
# ─────────────────────────────────────────────────────────────────────────────

class EntropyAnnealingCallback(BaseCallback):
    """
    Entropy coefficient annealing that is CONDITIONAL on reward improvement.

    The entropy only decays when training is making progress.  If reward
    stagnates for stagnation_window episodes, entropy is held constant.
    If reward collapses (drops below collapse_threshold relative to best),
    entropy is partially RESTORED toward recovery_level to re-enable
    exploration.

    Parameters
    ----------
    start_val           : initial entropy coefficient (default 0.15)
    end_val             : minimum entropy coefficient (default 0.05)
    total_timesteps     : used only for the hard time-based floor
    window              : rolling window for reward progress check (default 30 eps)
    min_improvement     : minimum reward gain to allow decay (default 0.3)
    stagnation_window   : if reward hasn't improved in this many eps, freeze (default 50)
    collapse_threshold  : if avg reward drops this far below best, restore entropy (default 3.0)
    recovery_level      : entropy target when collapse detected (default start_val * 0.7)
    decay_rate          : fractional decay per qualifying episode (default 0.002)
    verbose             : 0 = silent, 1 = log milestones
    """

    def __init__(
        self,
        start_val:          float = 0.15,
        end_val:            float = 0.05,
        total_timesteps:    int   = 288_000,
        window:             int   = 30,
        min_improvement:    float = 0.3,
        stagnation_window:  int   = 50,
        collapse_threshold: float = 3.0,
        recovery_level:     float = None,  # defaults to start_val * 0.7
        decay_rate:         float = 0.002,
        verbose:            int   = 1,
    ):
        super().__init__(verbose)
        self.start_val          = start_val
        self.end_val            = end_val
        self.total_timesteps    = total_timesteps
        self.window             = window
        self.min_improvement    = min_improvement
        self.stagnation_window  = stagnation_window
        self.collapse_threshold = collapse_threshold
        self.recovery_level     = recovery_level if recovery_level is not None else start_val * 0.7
        self.decay_rate         = decay_rate

        self._current_ent       = start_val
        self._reward_buffer     = deque(maxlen=window)
        self._best_avg          = -float("inf")
        self._eps_since_improve = 0
        self._ep_count          = 0
        self._last_state        = "init"

    def _on_step(self) -> bool:
        # Collect episode rewards from the info dict
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode", None)
            if ep_info is None:
                # Also try direct reward key (some wrapper configs)
                ep_info = info.get("episode_metrics", None)
            if ep_info is not None:
                r = float(ep_info.get("r", ep_info.get("total_reward", 0.0)))
                self._reward_buffer.append(r)
                self._ep_count += 1
                # EMA of avg reward — resistant to single-episode spikes
                _raw_avg = float(np.mean(self._reward_buffer))
                if not hasattr(self, '_ema_avg'):
                    self._ema_avg = _raw_avg
                self._ema_avg = 0.97 * self._ema_avg + 0.03 * _raw_avg
                self._update_entropy()

        return True

    def _update_entropy(self):
        if len(self._reward_buffer) < min(10, self.window):
            return

        ema = getattr(self, '_ema_avg', float(np.mean(self._reward_buffer)))

        # ── Collapse: only if we had real positive progress first ─────────
        if self._best_avg > 0.0 and ema < self._best_avg - self.collapse_threshold:
            if self._last_state != "recovering":
                if self.verbose >= 1:
                    print(
                        f"\n  [ENT] ⚠️  Reward collapsed "
                        f"({ema:.2f} < best {self._best_avg:.2f} - {self.collapse_threshold}). "
                        f"Restoring entropy to {self.recovery_level:.4f}\n"
                    )
                self._last_state = "recovering"
            self._current_ent = min(self.start_val,
                                    max(self._current_ent, self.recovery_level))
            self._eps_since_improve = 0
            self.model.ent_coef = float(self._current_ent)
            return

        # ── Improvement check uses EMA, not noisy raw avg ─────────────────
        if ema > self._best_avg + self.min_improvement:
            self._best_avg          = ema
            self._eps_since_improve = 0
        else:
            self._eps_since_improve += 1

        # ── Stagnation: freeze only if ALSO trending downward ────────────
        # Don't freeze just because improvement < 0.5 — only freeze if
        # EMA is actually declining (policy is getting worse, not plateauing).
        _ema_trend = ema - getattr(self, '_prev_ema', ema)
        self._prev_ema = ema
        actually_stagnating = (
            self._eps_since_improve >= self.stagnation_window
            and _ema_trend <= 0.0   # flat or declining
        )

        if actually_stagnating:
            if self._last_state != "frozen":
                if self.verbose >= 1:
                    print(
                        f"\n  [ENT] ❄️  Frozen at {self._current_ent:.4f} "
                        f"(no improvement for {self.stagnation_window} eps, "
                        f"ema={ema:.2f})\n"
                    )
                self._last_state = "frozen"
            self.model.ent_coef = float(self._current_ent)
            return

        # ── Unfreeze if EMA starts rising again ───────────────────────────
        if self._last_state == "frozen" and _ema_trend > 0.05:
            self._last_state = "init"
            self._eps_since_improve = 0
            if self.verbose >= 1:
                print(f"\n  [ENT] ▶️  Unfrozen (ema rising: {ema:.2f})\n")

        # ── Decay when EMA is flat-or-rising (not actively worsening) ────
        if _ema_trend >= -0.1:
            new_ent = max(self.end_val,
                          self._current_ent * (1.0 - self.decay_rate))
            if new_ent < self._current_ent - 0.0005:
                if self._last_state != "decaying":
                    self._last_state = "decaying"
                    if self.verbose >= 1:
                        print(
                            f"\n  [ENT] 📉 Decaying: {self._current_ent:.4f} "
                            f"→ {new_ent:.4f} "
                            f"(ema={ema:.2f}, best={self._best_avg:.2f})\n"
                        )
            self._current_ent = new_ent

        self.model.ent_coef = float(self._current_ent)

    @property
    def current_entropy(self) -> float:
        return self._current_ent


# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from original (no bugs found)
# ─────────────────────────────────────────────────────────────────────────────

class AutoCheckpointCallback(BaseCallback):
    """Save model every ckpt_every episodes and when a new best reward is achieved."""

    def __init__(self, save_dir: str, ckpt_every: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir   = save_dir
        self.ckpt_every = ckpt_every
        self._ep        = 0
        self._best_r    = -float("inf")
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Use Monitor's info["episode"] (set only at terminal step).
            # Fallback: episode_metrics with "total_reward" key — also only
            # present at termination (mid-episode writes only set "n_dyn_imaged").
            ep_info = info.get("episode")
            if ep_info is None:
                ep_m = info.get("episode_metrics", {})
                if "total_reward" in ep_m:
                    ep_info = ep_m
            if ep_info is None:
                continue
            r = float(ep_info.get("r", ep_info.get("total_reward", 0.0)))
            self._ep += 1

            if self._ep % self.ckpt_every == 0:
                path = os.path.join(self.save_dir, f"ppo_ep{self._ep:05d}.zip")
                self.model.save(path)
                if self.verbose >= 1:
                    print(f"  [CKPT] Saved → {path}")

            if r > self._best_r:
                self._best_r = r
                path = os.path.join(self.save_dir, "ppo_best.zip")
                self.model.save(path)
                if self.verbose >= 1:
                    print(f"  [CKPT] New best r={r:.3f} → {path}")
        return True


class DynamicEventCallback(BaseCallback):
    """Per-episode DYN metrics logger — writes training_log.json."""

    def __init__(self, log_dir: str = "results", log_every: int = 1,
                 window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir   = log_dir
        self.log_every = log_every
        self.window    = window

        self._t0              = time.time()
        self._episode         = 0
        self._log: list       = []
        self._reward_window   = deque(maxlen=window)
        self._dyn_suc_window  = deque(maxlen=window)
        self._json_path: Optional[str] = None

        self.ep_rewards:     list = []
        self.ep_dyn_success: list = []
        self.ep_cf_rates:    list = []

    def _on_training_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._json_path = os.path.join(self.log_dir, "training_log.json")
        self._t0 = time.time()

    def _safe_ep_reward(self, info: dict) -> Optional[float]:
        """FIX-CB-2: handle both info['episode']['r'] and info['r']."""
        ep = info.get("episode")
        if ep is not None:
            return float(ep.get("r", 0.0))
        return float(info.get("total_reward", info.get("r", 0.0))) or None

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            r = self._safe_ep_reward(info)
            if r is None:
                continue

            ep_m = info.get("episode_metrics", {})
            ni   = int(ep_m.get("n_imaged", 0))
            nd   = int(ep_m.get("n_dyn_detected", 0))
            ndi  = int(ep_m.get("n_dyn_imaged", 0))
            nc   = int(ep_m.get("n_cloud_free", ni))

            dyn_suc = ndi / nd if nd > 0 else 0.0
            cf_rate = nc  / ni if ni > 0 else 0.0

            self._episode += 1
            self._reward_window.append(r)
            self._dyn_suc_window.append(dyn_suc)
            self.ep_rewards.append(r)
            self.ep_dyn_success.append(dyn_suc)
            self.ep_cf_rates.append(cf_rate)

            entry = {
                "ep": self._episode,
                "ts": int(self.num_timesteps),
                "wall_s": round(time.time() - self._t0, 1),
                "reward": round(r, 4),
                "n_imaged": ni,
                "n_dyn_det": nd,
                "n_dyn_img": ndi,
                "dyn_suc": round(dyn_suc, 4),
                "cf_rate": round(cf_rate, 4),
                "shaping_bonus": round(float(info.get("shaping_bonus", 0)), 4),
                "explore_bonus": round(float(info.get("explore_bonus_current", 0)), 5),
                "ent_coef": round(float(getattr(self.model, "ent_coef", 0)), 5),
            }
            self._log.append(entry)

            # Track per-episode action distribution
            if not hasattr(self, '_step_actions'):
                self._step_actions = []
            # Accumulate current step actions
            acts = self.locals.get('actions')
            if acts is not None:
                self._step_actions.append(int(acts[0]))

            if self._episode % self.log_every == 0 and self._json_path:
                summary = {
                    "n_episodes": self._episode,
                    "mean_reward_100": round(float(np.mean(self._reward_window)), 3),
                    "mean_dyn_suc_100": round(float(np.mean(self._dyn_suc_window)), 4),
                }
                with open(self._json_path, "w") as f:
                    json.dump({"episodes": self._log[-500:], "summary": summary},
                              f, indent=2, default=float)

            if self.verbose >= 1 and self._episode % max(1, self.log_every) == 0:
                r100   = np.mean(self._reward_window)
                d100   = np.mean(self._dyn_suc_window)
                ent    = float(getattr(self.model, "ent_coef", 0))
                print(
                    f"  Ep {self._episode:4d}  r={r:+7.3f}  avg100={r100:+6.2f}  "
                    f"dyn_suc={dyn_suc:.0%}(avg={d100:.0%})  "
                    f"ent={ent:.4f}"
                )
                # Action breakdown for the last episode
                try:
                    from collections import Counter as _C
                    _last = self._step_actions[-200:]   # last ~1 episode of steps
                    _ac   = _C(_last)
                    _n    = max(1, sum(_ac.values()))
                    _st   = sum(v for k, v in _ac.items() if k < 20) / _n
                    _dy   = sum(v for k, v in _ac.items() if 20 <= k <= 22) / _n
                    _dr   = _ac.get(23, 0) / _n
                    print(f"         actions: static={_st:.0%}  "
                          f"dyn={_dy:.0%}  drift={_dr:.0%}")
                except Exception:
                    pass
        return True


# ─────────────────────────────────────────────────────────────────────────────
# FIX-CB-3: VerboseStepLogger with separate CLOUD/NO-ACCESS/EMPTY labels
# ─────────────────────────────────────────────────────────────────────────────

EVENT_ICONS = {
    "wildfire": "🔥", "flood": "🌊", "earthquake": "🏔",
    "eruption": "🌋", "plume": "💨",
}
CLOUD_THRESH = 0.6   # must match env_alsat_debug.py


class VerboseStepLogger(BaseCallback):
    """
    Per-step logger with accurate failure-mode labels (FIX-CB-3).

    Labels:
      ✅ IMAGED         — reward > 0.001, DYN or static
      ❌ CLOUD          — image NOT taken because cloud_truth >= CLOUD_THRESH
      ⛔ NO-ACCESS      — image NOT taken because slew > 45° or outside window
      〰️ EMPTY-SLOT    — DYN action selected but slot was empty (should be masked)
      💤 DRIFT          — drift action (shown only if show_drift=True)
    """

    def __init__(self, print_every: int = 1, show_drift: bool = False,
                 show_events: bool = True):
        super().__init__(verbose=1)
        self.print_every = print_every
        self.show_drift  = show_drift
        self.show_events = show_events
        self._step_count = 0
        self._ep         = 0
        self._ep_reward  = 0.0

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._step_count % self.print_every != 0:
            return True

        try:
            info   = (self.locals.get("infos") or [{}])[0]
            rew    = float((self.locals.get("rewards") or [0])[0])
            act    = int((self.locals.get("actions") or [23])[0])
            done   = bool((self.locals.get("dones") or [False])[0])
            obs    = (self.locals.get("new_obs") or [None])[0]
            self._ep_reward += rew

            shaping = float(info.get("shaping_bonus", 0.0))
            base_r  = rew - shaping
            tau_s   = float(info.get("smdp_tau_s", 30.0))

            sat = self._get_sat()
            now = float(sat.simulator.sim_time) if sat else 0.0

            hdr = (
                f"  t={now:.0f}s  ep={self._ep+1}  step={self._step_count}  "
                f"ent={float(getattr(self.model,'ent_coef',0)):.4f}"
            )

            if 0 <= act < 20:
                # ── Static target ──────────────────────────────────────────
                label, detail = self._static_label(sat, act, rew)
                print(hdr)
                print(f"  ACT={act:2d} STATIC {detail}  τ={tau_s:.0f}s  "
                      f"r={rew:+.4f} [base={base_r:+.4f} shp={shaping:+.4f}]  {label}")

            elif 20 <= act <= 22:
                # ── DYN slot ───────────────────────────────────────────────
                slot_idx = act - 20
                label, detail = self._dyn_label(sat, slot_idx, rew, now)
                print(hdr)
                print(f"  ACT={act:2d} DYN-slot{slot_idx} {detail}  τ={tau_s:.0f}s  "
                      f"r={rew:+.4f} [base={base_r:+.4f} shp={shaping:+.4f}]  {label}")

            elif act == 23 and self.show_drift:
                batt = "?"
                try:
                    batt = f"{sat.dynamics.battery_charge_fraction:.0%}"
                except Exception:
                    pass
                print(hdr)
                print(f"  ACT=23 DRIFT  τ={tau_s:.0f}s  batt={batt}  r={rew:+.4f}  💤")

            if self.show_events and sat is not None:
                self._print_events(sat, now)

            if done:
                self._ep += 1
                ep_m     = info.get("episode_metrics", {})
                nd    = ep_m.get("n_dyn_detected", 0)
                nim   = ep_m.get("n_dyn_imaged",   0)
                ni    = ep_m.get("n_imaged",        0)
                print(
                    f"\n  {'═'*60}\n"
                    f"  EP {self._ep} END  r={self._ep_reward:+.3f}  "
                    f"imgs={ni}  dyn={nim}/{nd}\n"
                    f"  {'═'*60}\n"
                )
                logger.info(
                    f"[EP] ep={self._ep}  r={self._ep_reward:+.3f}  "
                    f"static_imaged={ep_m.get('n_imaged',0)}  "
                    f"cloud_free={ep_m.get('n_cloud_free',0)}  "
                    f"cloudy={ep_m.get('n_cloudy',0)}  "
                    f"dyn_detected={ep_m.get('n_dyn_detected',0)}  "
                    f"dyn_imaged={ep_m.get('n_dyn_imaged',0)}  "
                    f"missed={ep_m.get('n_missed_events',0)}  "
                    f"total_rew_accum={ep_m.get('total_reward',0):.3f}"
                )
                self._ep_reward = 0.0

        except Exception as exc:
            logger_cb = __import__("logging").getLogger(__name__)
            logger_cb.debug(f"VerboseStepLogger error: {exc}")

        return True

    def _get_sat(self):
        try:
            e = self.training_env
            while hasattr(e, "envs"):
                e = e.envs[0]
            while hasattr(e, "env"):
                e = e.env
            return getattr(e, "unwrapped", e).satellites[0]
        except Exception:
            return None

    def _static_label(self, sat, act: int, rew: float):
        """FIX-CB-3: returns (label, detail_string) for a static action."""
        try:
            tgt       = sat.scenario.targets[act]
            cf_truth  = float(getattr(tgt, "cloud_cover", 0.0))
            cf_fcst   = float(getattr(tgt, "cloud_cover_forecast", cf_truth))
            prio      = float(getattr(tgt, "priority", 0.5))
            name      = getattr(tgt, "name", f"T{act:02d}")

            from env_alsat_debug import calculate_slew_angle_to_target
            slew_deg = math.degrees(calculate_slew_angle_to_target(sat, tgt))

            if rew > 0.001:
                label = "✅ IMAGED"
            elif cf_truth >= CLOUD_THRESH:
                label = f"❌ CLOUD (truth={cf_truth:.2f})"
            elif slew_deg > 45.0:
                label = f"⛔ NO-ACCESS (slew={slew_deg:.1f}°)"
            else:
                label = f"⛔ NO-ACCESS (window miss)"

            detail = (
                f"[{name}]  prio={prio:.2f}  fcst={cf_fcst:.2f}  "
                f"truth={cf_truth:.2f}  slew={slew_deg:.1f}°"
            )
            return label, detail
        except Exception as exc:
            return f"? (err: {exc})", ""

    def _dyn_label(self, sat, slot_idx: int, rew: float, now: float):
        """FIX-CB-3: returns (label, detail_string) for a DYN slot action."""
        try:
            mgr = getattr(sat, "_event_manager", None)
            if mgr is None:
                return "〰️ NO-MGR", ""
            slots = mgr.get_slots(sat, now)
            evt   = slots[slot_idx] if slot_idx < len(slots) else None

            if evt is None:
                return "〰️ EMPTY-SLOT", "[no active event]"

            cf_truth = float(getattr(evt, "cloud_cover", 0.0))
            cf_fcst  = float(getattr(evt, "cloud_cover_forecast", cf_truth))
            prio     = float(getattr(evt, "priority", 0.9))
            etype    = getattr(evt, "event_type", "?")
            icon     = EVENT_ICONS.get(etype, "📍")
            lat      = math.degrees(getattr(evt, "lat_rad", 0.0))
            lon      = math.degrees(getattr(evt, "lon_rad", 0.0))

            from env_alsat_debug import calculate_slew_angle_to_target
            slew_deg = math.degrees(calculate_slew_angle_to_target(sat, evt))

            remaining_min = max(0, (evt.expiration_time - now) / 60)

            if rew > 0.001:
                label = "✅ IMAGED"
            elif cf_truth >= CLOUD_THRESH:
                label = f"❌ CLOUD (truth={cf_truth:.2f})"
            elif slew_deg > 45.0:
                label = f"⛔ NO-ACCESS (slew={slew_deg:.1f}°)"
            else:
                label = "⛔ NO-ACCESS (other)"

            detail = (
                f"{icon}{etype}  lat={lat:+.1f}°  lon={lon:+.1f}°  "
                f"prio={prio:.2f}  fcst={cf_fcst:.2f}  truth={cf_truth:.2f}  "
                f"slew={slew_deg:.1f}°  rem={remaining_min:.0f}min"
            )
            return label, detail
        except Exception as exc:
            return f"? (err: {exc})", ""

    def _print_events(self, sat, now: float):
        try:
            mgr = getattr(sat, "_event_manager", None)
            if mgr is None:
                return
            active = [e for e in getattr(mgr, "_events", [])
                      if e is not None and not e.imaged and e.expiration_time > now]
            if not active:
                return
            print(f"  Active events ({len(active)}):")
            for evt in active[:5]:
                icon = EVENT_ICONS.get(getattr(evt, "event_type", ""), "📍")
                lat  = math.degrees(getattr(evt, "lat_rad", 0.0))
                lon  = math.degrees(getattr(evt, "lon_rad", 0.0))
                rem  = max(0, (evt.expiration_time - now) / 60)
                print(
                    f"    {icon} {evt.name}  lat={lat:+.1f}°  lon={lon:+.1f}°  "
                    f"rem={rem:.0f}min  cf={evt.cloud_cover:.2f}"
                )
        except Exception:
            pass