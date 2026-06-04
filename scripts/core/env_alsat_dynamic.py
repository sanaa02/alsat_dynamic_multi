#!/usr/bin/env python3
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -----------------------------------------------------------------
"""
env_alsat_dynamic.py  --  ALSAT-EO-1  Phase 3  SMDP Dynamic Environment
=========================================================================
COMPLETE REWRITE  --  Four principal upgrades vs. prior version:

[SMDP]  Native variable-duration step  (replaces SMDPDynamicWrapper)
        step() computes tau = slew_time + IMAGING_DUR_S, runs ceil(tau/BASE_STEP_S)
        Basilisk sub-steps, and accumulates reward with per-sub-step discount
        gamma_sub = gamma^(BASE_STEP_S / STEP_REF_S).
        Obs gains sojourn_time_norm as the 56th element (was 55).

[TTA]   Continuous Keplerian time-to-access
        EventManager.time_to_access() calls keplerian_tta() from wrappers/
        instead of returning the binary 0 / INACCESSIBLE_TIME_S placeholder.
        The tta_norm features in obs[43:55] now carry real predicted access
        times, giving the policy meaningful look-ahead.

[DECAY] Urgency-decayed dynamic event reward
        reward = priority*(1-cloud_truth)*exp(-elapsed/DECAY_TAU) + DYNAMIC_BONUS
        DECAY_TAU = 3600 s (1-hour half-value).  The agent learns to image
        fresh events before their value erodes.

[SAFE]  Safety monitor hook
        DynamicImageTargetAction calls satellite._safety_monitor.check()
        if the attribute exists, vetoing the action (->DRIFT) if it would
        violate battery/slew/storage constraints.

Observation space : Box(-inf, inf, (56,))
  [0:43]  base satellite obs (Phase 2 unchanged)
  [43:55] 3 dynamic-event slots x [priority, cloud_fcst, tta_norm, slew_norm]
  [55]    sojourn_time_norm = tau / MAX_ACTION_DUR_S        <- NEW

Action space : Discrete(24)
  0-19  static targets
  20-22 dynamic event slots 0/1/2
  23    DRIFT

smdp_dynamic.py is now DEPRECATED: make_dynamic_env() already returns
a full SMDP environment with obs (56,). Existing code that called
make_smdp_dynamic_env() should switch to make_dynamic_env().
"""
import logging
import math
from typing import List, Optional

import gymnasium as gym
import numpy as np

# ---- local imports (all resolved via path_setup) ----------------------------
from env_alsat_debug import (
    AlsatSatellite, AlsatScenario, AlsatTarget, ScienceData,
    ScienceDataStore, ScienceReward, ImageTargetAction, ModisCloudModel,
    TorqueLimitedDynamics, calculate_slew_angle_to_target,
    calculate_slew_time, calculate_slew_energy_wh, load_targets_config,
    CLOUD_THRESH, SLEW_ENERGY_ALPHA, SCHED_STEP_S, SIM_DURATION_S,
    BSK_SIM_RATE_S, CNN_NOISE_STD, SMA_M,
)
from dynamic_event import (
    DynamicEvent, EventGenerator, EventManager,
    N_DYN_SLOTS, MAX_OFFNADIR_RAD, INACCESSIBLE_TIME_S, DYNAMIC_BONUS, DYN_MULTIPLIER,
)
from bsk_rl.act import Action
from bsk_rl.act.discrete_actions import DiscreteActionBuilder
from bsk_rl.gym import GeneralSatelliteTasking
from bsk_rl.data.base import DataStore
from bsk_rl.data import GlobalReward
from bsk_rl.sim import fsw

# Optional: Keplerian TTA solver from wrappers/
try:
    from env_alsat_dynamic_tta_patch import keplerian_tta as _keplerian_tta
    _HAS_KEPLERIAN = True
except ImportError:
    _keplerian_tta   = None
    _HAS_KEPLERIAN   = False

logger = logging.getLogger(__name__)

# ---- constants ---------------------------------------------------------------
N_STATIC_TARGETS  = 20
N_TOTAL_ACTIONS   = N_STATIC_TARGETS + N_DYN_SLOTS + 1     # 24
OBS_BASE_DIM      = 43
OBS_DYN_DIM       = N_DYN_SLOTS * 4                         # 12
OBS_SOJOURN_DIM   = 1
OBS_TOTAL_DIM     = OBS_BASE_DIM + OBS_DYN_DIM + OBS_SOJOURN_DIM  # 56

# SMDP timing
BASE_STEP_S       = 30.0
STEP_REF_S        = SCHED_STEP_S        # 1200 s reference for discounting
MAX_ACTION_DUR_S  = 200.0               # normalisation cap for sojourn feature
MAX_SUB_STEPS     = 20                  # safety cap on sub-steps per action
DEFAULT_GAMMA     = 0.99

# TTA normalisation  (same scale as opportunity_open)
TIME_NORM_S       = SCHED_STEP_S * 100

# [DECAY] urgency decay time-constant (1 hour)
EVENT_DECAY_TAU_S = 3600.0      # 1-hour exponential time constant

# =============================================================================
#  Keplerian TTA wrapper (with binary fallback)
# =============================================================================

def _compute_tta(satellite, event, sim_time: float) -> float:
    if _HAS_KEPLERIAN:
        try:
            return float(_keplerian_tta(satellite, event, sim_time))
        except Exception:
            pass
    # Binary fallback
    slew = _slew_safe(satellite, event)
    return 0.0 if slew <= MAX_OFFNADIR_RAD else INACCESSIBLE_TIME_S


def _slew_safe(satellite, target) -> float:
    try:
        return float(calculate_slew_angle_to_target(satellite, target))
    except Exception:
        return 0.0


# =============================================================================
#  [SMDP] Action-duration helper
# =============================================================================

def _action_duration(satellite, action: int) -> float:
    drift = N_STATIC_TARGETS + N_DYN_SLOTS
    if action >= drift:
        return BASE_STEP_S
    if action < N_STATIC_TARGETS:
        target = satellite.scenario.targets[action]
    else:
        slot    = action - N_STATIC_TARGETS
        mgr     = getattr(satellite, '_event_manager', None)
        if mgr is None:
            return BASE_STEP_S
        now     = float(satellite.simulator.sim_time)
        slots   = mgr.get_slots(satellite, now)
        target  = slots[slot] if slot < len(slots) else None
        if target is None:
            return BASE_STEP_S
    slew = _slew_safe(satellite, target)
    tau  = calculate_slew_time(slew) + 20.0   # IMAGING_DUR_S
    return float(np.clip(tau, BASE_STEP_S, MAX_ACTION_DUR_S))


# =============================================================================
#  [DYN-1] Extended action handler
# =============================================================================

class DynamicImageTargetAction(ImageTargetAction):
    """Handles actions 0-22 (static + dynamic) and 23 (DRIFT)."""

    @property
    def n_actions(self) -> int:
        if hasattr(self, 'satellite') and hasattr(self.satellite, 'scenario'):
            return len(self.satellite.scenario.targets) + N_DYN_SLOTS + 1
        return 1

    def set_action(self, action: int, prev_action_key=None) -> None:
        n_static = len(self.satellite.scenario.targets)
        now      = float(self.satellite.simulator.sim_time)

        if self.satellite.scenario is not None:
            self.satellite.scenario.update_cloud(now)

        # DRIFT — clear any locked DYN event
        if action >= n_static + N_DYN_SLOTS:
            self.satellite.last_slew_angle         = 0.0
            self.satellite.current_action_is_dynamic = False
            self.satellite._locked_dyn_slot  = None
            self.satellite._locked_dyn_event = None
            self.satellite._dyn_img_fired    = False
            return

        # Static target — clear DYN lock
        # In DynamicImageTargetAction.set_action(), static target branch:
        if action < n_static:
            self.satellite.current_action_is_dynamic = False
            self.satellite._locked_dyn_slot  = None
            self.satellite._locked_dyn_event = None
            # ── Record last static target for monitor logging ─────────────
            try:
                _tgt = self.satellite.scenario.targets[action]   # safe: action < n_static
                self.satellite._last_static_log = {
                    "name":     getattr(_tgt, "name",        f"Target-{action:02d}"),
                    "cloud":    float(getattr(_tgt, "cloud_cover",  0.0)),
                    "priority": float(getattr(_tgt, "priority",     0.5)),
                }
            except Exception:
                pass
            # ────────────────────────────────────────────────────────────────
            super().set_action(action, prev_action_key)
            return

        # Dynamic event
        slot      = action - n_static
        event_mgr = getattr(self.satellite, '_event_manager', None)
        if event_mgr is None:
            self.satellite.last_slew_angle = 0.0
            return

        # ── SMDP sub-step locking fix ────────────────────────────────────
        # PROBLEM: set_action(22) is called at EVERY SMDP sub-step (t, t+30,
        # t+60, ...).  Each call re-runs get_slots(sat, now_updated) — the
        # event ranking changes as the satellite moves, so slot 2 gets a
        # DIFFERENT event (or None) on each sub-step.  The satellite keeps
        # re-tasking before imaging completes → was_image_taken() = False →
        # mark_imaged() never called → n_dyn_imaged = 0 forever.
        #
        # FIX: lock the chosen event when first selected for this slot.
        # Reuse the locked event for all subsequent sub-steps of the SAME
        # action.  Clear the lock when action changes to drift or static.
        _LOCK_SLOT  = '_locked_dyn_slot'
        _LOCK_EVT   = '_locked_dyn_event'

        locked_slot  = getattr(self.satellite, _LOCK_SLOT,  None)
        locked_event = getattr(self.satellite, _LOCK_EVT,   None)

        if locked_slot == slot and locked_event is not None:
            # Same DYN slot — reuse the locked event (avoids re-tasking)
            event = locked_event
        else:
            # New DYN action or first sub-step — query and lock
            slots = event_mgr.get_slots(self.satellite, now)
            event = slots[slot] if slot < len(slots) else None
            setattr(self.satellite, _LOCK_SLOT,  slot)
            setattr(self.satellite, _LOCK_EVT,   event)

        if event is None:
            self.satellite.last_slew_angle         = 0.0
            self.satellite.current_action_is_dynamic = False
            return

        slew = _slew_safe(self.satellite, event)
        self.satellite.last_slew_angle = float(slew)

        # [SAFE] optional safety monitor veto
        monitor = getattr(self.satellite, '_safety_monitor', None)
        if monitor is not None:
            _chk = monitor.check(self.satellite, action, event, now)
            safe, reason = _chk if isinstance(_chk, tuple) else (bool(_chk), 'safety')
            if not safe:
                logger.debug(f"Safety veto: {reason}  action={action}")
                self.satellite.current_action_is_dynamic = False
                return

        # Always record target regardless of slew — P4 (bsk_patches) reads
        # current_action_target after _orig returns to set _locked_dyn_event.
        self.satellite.current_action_target    = event
        self.satellite.current_action_is_dynamic = True

        # After successfully imaging a static target, add:


# ────────────────────────────────────────────────────────────────

        if slew <= MAX_OFFNADIR_RAD:
            try:
                # Synthetic window so bsk_rl's task_target_for_imaging doesn't
                # crash with 'next_window' UnboundLocalError on DynamicEvents.
                try:
                    _now_s = float(self.satellite.simulator.sim_time)
                    _fake  = {"object": event,
                              "window": (_now_s - 30.0, float(event.expiration_time)),
                              "type": "target", "requires_retasking": False}
                    _opps  = [o for o in
                              list(getattr(self.satellite, 'upcoming_opportunities', []))
                              if o.get("object") is not event]
                    _opps.append(_fake)
                    self.satellite.upcoming_opportunities = _opps
                except Exception:
                    pass
                self.satellite.task_target_for_imaging(event)
            except Exception as exc:
                logger.debug(f"task_target_for_imaging (dynamic): {exc}")


# =============================================================================
#  [DYN-2] Extended reward with [DECAY]
# =============================================================================

class DynamicScienceDataStore(ScienceDataStore):
    data_type = ScienceData

    def compare_log_states(self, old_state, new_state) -> ScienceData:
        sat = self.satellite

        # ── DYN event imaging bypass ──────────────────────────────────────
        # bsk_rl's was_image_taken_since_last_check() only returns True for
        # targets with precomputed access windows (upcoming_opportunities).
        # DynamicEvents have NO precomputed windows → always returns False.
        # Fix: directly confirm imaging when satellite correctly pointed at
        # the DYN event (slew <= MAX_OFFNADIR_RAD) and imaging not yet fired.
        is_dyn_action = getattr(sat, 'current_action_is_dynamic', False)
        slew_angle    = getattr(sat, 'last_slew_angle', float('inf'))
        already_fired = getattr(sat, '_dyn_img_fired', False)

        if is_dyn_action and slew_angle <= MAX_OFFNADIR_RAD:
            if already_fired:
                # Already gave reward for this SMDP action — prevent double-count
                return ScienceData(0.0)
            else:
                # First confirmation for this DYN action — mark and proceed
                sat._dyn_img_fired = True
                image_taken = True
        else:
            # Static target: use bsk_rl's standard check
            image_taken = sat.was_image_taken_since_last_check()
            if not image_taken:
                return ScienceData(0.0)

        target = getattr(sat, 'current_action_target', None)
        if target is None:
            return ScienceData(0.0)

        is_dynamic  = getattr(sat, 'current_action_is_dynamic', False)
        cloud_truth = float(target.cloud_cover)
        priority    = float(target.priority)
        slew_angle  = getattr(sat, 'last_slew_angle', 0.0)
        slew_energy = calculate_slew_energy_wh(slew_angle)

        if is_dynamic:
            # [DECAY] urgency factor
            try:
                now       = float(sat.simulator.sim_time)
                elapsed   = now - float(target.appearance_time)
                urgency = max(0.15, 1.0 - elapsed / (2.0 * EVENT_DECAY_TAU_S))
            except Exception:
                urgency = 1.0

            if cloud_truth < CLOUD_THRESH:
                reward = DYN_MULTIPLIER * priority * (1.0 - cloud_truth) * urgency
                sat._metrics['n_cloud_free'] += 1
            else:
                reward = -0.3 * priority   # stronger penalty for cloudy dynamic waste


            event_mgr = getattr(sat, '_event_manager', None)
            if event_mgr is not None and isinstance(target, DynamicEvent):
                event_mgr.mark_imaged(target, float(sat.simulator.sim_time), reward)
                sat._metrics['n_dyn_imaged'] = event_mgr._metrics['n_imaged']

        else:
            if cloud_truth < CLOUD_THRESH:
                reward = (priority * (1.0 - cloud_truth)
                          - SLEW_ENERGY_ALPHA * slew_energy)
                sat._metrics['n_cloud_free'] += 1
            else:
                reward = -0.1 * priority
                sat._metrics['n_cloudy'] += 1

        sat._metrics['n_imaged']             += 1
        sat._metrics['total_slew_angle_deg'] += math.degrees(slew_angle)
        sat._metrics['total_slew_energy_wh'] += slew_energy
        sat._metrics['total_reward']         += reward

        sat.current_action_target    = None
        sat.current_action_is_dynamic = False
        return ScienceData(reward)


class DynamicScienceReward(GlobalReward):
    data_store_type = DynamicScienceDataStore

    def __init__(self, reward_scale: float = 1.0):
        super().__init__()
        self.reward_scale = reward_scale

    def calculate_reward(self, new_data_dict: dict) -> dict:
        return {k: v.value * self.reward_scale for k, v in new_data_dict.items()}


# =============================================================================
#  Satellite with event manager + extended metrics
# =============================================================================

class DynamicAlsatSatellite(AlsatSatellite):
    action_spec = [DynamicImageTargetAction()]

    def __init__(self, name='ALSAT-1', sat_args=None, scenario=None,
                 event_manager: Optional[EventManager] = None,
                 safety_monitor=None, **kwargs):
        self._event_manager  = event_manager
        self._safety_monitor = safety_monitor
        super().__init__(name=name, sat_args=sat_args, scenario=scenario, **kwargs)

    def reset_post_sim_init(self) -> None:
        super().reset_post_sim_init()
        self.current_action_is_dynamic = False
        self._metrics.update({'n_dyn_detected': 0, 'n_dyn_imaged': 0})
        if self._event_manager is not None:
            self._event_manager.reset()


# =============================================================================
#  Flat single-satellite wrapper (for SB3 compatibility)
# =============================================================================

class SingleSatelliteEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space      = self.env.action_space[0]
        self.observation_space = self.env.observation_space[0]

    def reset(self, **kwargs):
        obs_tuple, info = self.env.reset(**kwargs)
        return obs_tuple[0], info

    def step(self, action):
        obs_tuple, r, term, trunc, info = self.env.step((action,))
        if term or trunc:
            try:
                sat = self.env.unwrapped.satellites[0]
                info['episode_metrics'] = dict(sat._metrics)
            except Exception:
                info['episode_metrics'] = {}
        return obs_tuple[0], r, term, trunc, info


# =============================================================================
#  [SMDP + TTA + DECAY] DynamicObsWrapper  --  the core Phase 3 env
# =============================================================================

class DynamicObsWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that delivers the full Phase 3 feature set:

      Obs (56,): base(43) + dyn_events(12) + sojourn(1)
      SMDP step: variable duration tau = slew + imaging,
                 discount gamma_sub = gamma^(BASE_STEP_S/STEP_REF_S)
      TTA features: Keplerian-predicted (continuous, not binary)
      Safety: EventManager slots ranked by accessibility

    Parameters
    ----------
    env    : SingleSatelliteEnv (wrapping bsk_rl GeneralSatelliteTasking)
    gen    : EventGenerator (Poisson arrivals)
    mgr    : EventManager   (shared with satellite)
    gamma  : discount factor per STEP_REF_S  (default 0.99)
    """

    def __init__(self, env: gym.Env, gen: EventGenerator, mgr: EventManager,
                 gamma: float = DEFAULT_GAMMA):
        super().__init__(env)
        self._gen       = gen
        self._mgr       = mgr
        self._gamma_sub = gamma ** (BASE_STEP_S / STEP_REF_S)
        self._prev_time = 0.0

        self.action_space = gym.spaces.Discrete(N_TOTAL_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_TOTAL_DIM,), dtype=np.float32)

    # ---- reset / step -------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_time = 0.0
        self._gen.reset(seed=kwargs.get('seed'))
        self._mgr.reset()
        # ── CRITICAL: attach self._mgr to satellite so set_action() sees events
        try:
            for _sat in self.env.unwrapped.satellites:
                _sat._event_manager = self._mgr
        except Exception:
            pass
        self._n_static_actions_ep = 0 
        return self._build_obs(obs, tau_norm=0.0), info

    def step(self, action: int):
        if int(action) < N_STATIC_TARGETS:
           self._n_static_actions_ep = getattr(self, '_n_static_actions_ep', 0) + 1
        # [SMDP] compute actual task duration
        try:
            sat      = self.env.unwrapped.satellites[0]
            # Reset DYN imaging flag for new action
            sat._dyn_reward_given = False
            tau      = _action_duration(sat, int(action))
        except Exception:
            tau = BASE_STEP_S

        tau   = float(np.clip(tau, BASE_STEP_S, MAX_ACTION_DUR_S))

        n_sub = max(1, min(MAX_SUB_STEPS, int(math.ceil(tau / BASE_STEP_S))))

        last_obs = None
        term = trunc = False
        info: dict = {}

        # ── Pre-step event spawn (ROOT FIX for n_dyn_imaged=0) ───────────────
        try:

            # Keep sat._event_manager pointing at self._mgr (survives bsk_rl resets)
            for _sx in self.env.unwrapped.satellites:
                _sx._event_manager = self._mgr
        except Exception:
            pass

        DRIFT_ACT = N_STATIC_TARGETS + N_DYN_SLOTS  # = 23
        total_r = 0.0
        for _i in range(n_sub):
            _sub_a = action if _i == 0 else DRIFT_ACT
            obs_i, r_i, term, trunc, info = self.env.step(_sub_a)
            total_r += (self._gamma_sub ** _i) * r_i
            last_obs = obs_i
            if term or trunc:
                break
            smdp_discount = self._gamma_sub ** (tau / BASE_STEP_S)


        # ── [ROOT FIX] geometric imaging check (non-critical, wrapped safely)
        # NOTE: do NOT reset _locked_dyn_event here — the main injection
        # block below (lines 506+) uses it and must see it intact.
        try:
            _sat = self.env.unwrapped.satellites[0]
            _dyn_r = _dyn_imaging_check(_sat, info)
            if _dyn_r > 0.0:
                total_r += _dyn_r
                # Only reset lock if main injection block already ran
                # (checked by _dyn_reward_given flag)
                _sat._dyn_reward_given = True   # ← prevent injection block from firing again
                _sat._locked_dyn_event = None
                _sat._locked_dyn_slot  = None

        except Exception:
            pass  # Geometric check failed; main injection block handles reward


        # ── DYN event reward injection ────────────────────────────────────
        # bsk_rl's imaging pipeline never fires for DynamicEvent targets
        # (no precomputed access windows).  We inject reward directly here.
        #
        # IMPORTANT: compare_log_states resets current_action_is_dynamic=False
        # and current_action_target=None before this code runs.  Therefore we
        # use P4's LOCKED event (_locked_dyn_event), which persists after
        # the bsk_rl step and is NOT touched by compare_log_states.
        _N_STATIC = N_STATIC_TARGETS  # = 20
        if _N_STATIC <= int(action) < _N_STATIC + N_DYN_SLOTS:
            try:
                _sat    = self.env.unwrapped.satellites[0]
                _slot   = int(action) - _N_STATIC
                # Use P4's locked event (survives compare_log_states reset)
                _target = getattr(_sat, '_locked_dyn_event', None)
                _l_slot = getattr(_sat, '_locked_dyn_slot',  -1)
                _slew   = getattr(_sat, 'last_slew_angle',   float('inf'))
                _fired  = getattr(_sat, '_dyn_reward_given', False)

                _already_done = _target.imaged if _target else False
                if (_target is not None
                        and isinstance(_target, DynamicEvent)
                        and _l_slot == _slot
                        and _slew <= MAX_OFFNADIR_RAD
                        and not _fired
                        and not _already_done):

                    _sat._dyn_reward_given = True
                    _cloud  = float(_target.cloud_cover)
                    _prio   = float(_target.priority)

                    # Urgency: newer events pay more
                    try:
                        _now     = float(_sat.simulator.sim_time)
                        _elapsed = max(0.0, _now - float(_target.appearance_time))
                        _urgency = max(0.15, math.exp(-_elapsed / EVENT_DECAY_TAU_S))
                    except Exception:
                        _urgency = 1.0

                    if _cloud < CLOUD_THRESH:
                        _dyn_r = _prio * (1.0 - _cloud) * _urgency + DYNAMIC_BONUS
                        _sat._metrics['n_cloud_free'] += 1
                    else:
                        _dyn_r = -0.05 * _prio
                        _sat._metrics['n_cloudy'] += 1

                    # Update metrics
                    _sat._metrics['n_imaged']      += 1
                    _sat._metrics['total_reward']  += _dyn_r
                    _sat._metrics['total_slew_angle_deg'] += math.degrees(_slew)

                    # Increment event manager imaged counter
                    _evt_mgr = getattr(_sat, '_event_manager', None)
                    if _evt_mgr is not None:
                        _evt_mgr.mark_imaged(_target,
                                             float(_sat.simulator.sim_time),
                                             _dyn_r)
                        _sat._metrics['n_dyn_imaged'] = _evt_mgr._metrics['n_imaged']
                    else:
                        _sat._metrics['n_dyn_imaged'] += 1

                    # sync counter to info dict at each step
                    info.setdefault('episode_metrics', {})['n_dyn_imaged'] = _sat._metrics.get('n_dyn_imaged', 0)

                    total_r += _dyn_r * smdp_discount

                    # Inside the DYN reward injection block, after total_r += _dyn_r * smdp_discount:
                    try:
                        _sat._last_dyn_event_log = {
                            "type":     _target.event_type,
                            "lat":      float(math.degrees(_target.lat_rad)),
                            "lon":      float(math.degrees(_target.lon_rad)),
                            "priority": float(_target.priority),
                            "cloud":    float(_cloud),
                            "reward":   float(_dyn_r),
                            "slot":     _slot,
                            "ep":       getattr(_sat, '_episode_count', 0),
                        }
                    except Exception:
                        pass

                    logger.debug(
                        f"DYN reward injected: r={_dyn_r:.3f}  "
                        f"cloud={_cloud:.2f}  urgency={_urgency:.2f}  "
                        f"event={type(_target).__name__}"
                    )
            except Exception as _exc:
                logger.debug(f"DYN reward injection error: {_exc}")

        # Drive event lifecycle
        try:
            sat  = self.env.unwrapped.satellites[0]
            now  = float(sat.simulator.sim_time)
            dt   = max(0.0, now - self._prev_time)
            new_events = self._gen.step(now, dt)
            self._mgr.add_events(new_events)
            self._mgr.purge_expired(now)
            self._prev_time = now
            sat._metrics['n_dyn_detected'] = self._mgr._metrics['n_detected']
        except Exception as exc:
            logger.debug(f"Event lifecycle error: {exc}")

        info['smdp_tau_s']       = tau
        info['smdp_n_sub']       = n_sub
        info['dynamic_metrics']  = self._mgr.get_metrics()

        tau_norm = tau / MAX_ACTION_DUR_S

        # ── Static-imaging floor penalty ─────────────────────────────
        # If agent took ZERO static actions this episode, penalize.
        # Prevents catastrophic forgetting of scheduled imaging.
        if (term or trunc):
            n_static = getattr(self, '_n_static_actions_ep', 0)
            if n_static == 0:
                total_r -= 1.0   # static imaging floor
                logger.debug(f"Static forgetting penalty: -1.0 (n_static=0)")
            self._n_static_actions_ep = 0   # reset for next episode
        # ─────────────────────────────────────────────────────────────

        return self._build_obs(last_obs, tau_norm), total_r, term, trunc, info

    # ---- observation builder ------------------------------------------------

    def _build_obs(self, base_obs: np.ndarray, tau_norm: float) -> np.ndarray:
        try:
            sat   = self.env.unwrapped.satellites[0]
            now   = float(sat.simulator.sim_time)
            slots = self._mgr.get_slots(sat, now)
        except Exception:
            slots = [None] * N_DYN_SLOTS
            sat   = None
            now   = 0.0

        feats = []
        for evt in slots:
            if evt is None:
                feats.extend([0.0, -1.0, 1.0, 0.0])
            else:
                try:
                    slew = _slew_safe(sat, evt)
                    # [TTA] use Keplerian-predicted access time
                    tta  = _compute_tta(sat, evt, now)
                    feats.extend([
                        float(np.clip(evt.priority,             0.0, 1.0)),
                        float(np.clip(evt.cloud_cover_forecast, 0.0, 1.0)),
                        float(np.clip(tta / TIME_NORM_S,        0.0, 1.0)),
                        float(np.clip(slew / (math.pi / 2),     0.0, 1.0)),
                    ])
                except Exception:
                    feats.extend([0.0, -1.0, 1.0, 0.0])

        dyn_arr     = np.array(feats,      dtype=np.float32)
        sojourn_arr = np.array([np.clip(tau_norm, 0.0, 1.0)], dtype=np.float32)
        # Battery SOC — agent can now reason about energy constraint
        

        return np.concatenate([base_obs.astype(np.float32), dyn_arr, [tau_norm]],dtype=np.float32)

    # ---- convenience properties ---------------------------------------------

    @property
    def event_manager(self) -> EventManager:
        return self._mgr

    @property
    def event_generator(self) -> EventGenerator:
        return self._gen


# =============================================================================
#  Factory
# =============================================================================

def make_dynamic_env(
    targets_path:    str,
    cloud_json_path: str,
    event_rate:      float = 2.0,
    duration_s:      float = SIM_DURATION_S,
    sim_rate:        float = BSK_SIM_RATE_S,
    sat_name:        str   = 'ALSAT-1',
    sat_args:        dict  = None,
    cloud_model              = None,   # pass VisionCloudModel to use CNN forecasts
    safety_monitor           = None,   # pass SafetyMonitor instance
    gamma:           float = DEFAULT_GAMMA,
    seed:            int   = 42,
    render_mode              = None,
) -> DynamicObsWrapper:
    """
    Build the Phase 3 SMDP dynamic targeting environment.

    Returns DynamicObsWrapper  obs=(56,)  actions=Discrete(24)

    Parameters
    ----------
    cloud_model   : if None, uses ModisCloudModel (Gaussian noise).
                    Pass a VisionCloudModel instance to use CNN forecasts.
    safety_monitor: if provided, the satellite's action handler will veto
                    unsafe actions before calling task_target_for_imaging().
    gamma         : SMDP discount factor (per STEP_REF_S = 1200 s)
    """
    targets_cfg  = load_targets_config(targets_path)
    if cloud_model is None:
        cloud_model = ModisCloudModel(cloud_json_path, seed=seed)
    scenario     = AlsatScenario(targets_cfg, cloud_model)
    gen_duration = duration_s

    event_gen = EventGenerator(rate_per_hour=event_rate, seed=seed)
    event_mgr = EventManager()

    satellite = DynamicAlsatSatellite(
        name=sat_name, sat_args=sat_args, scenario=scenario,
        event_manager=event_mgr, safety_monitor=safety_monitor,
        generation_duration=gen_duration, initial_generation_duration=gen_duration + 7200 + 7200,
    )

    base_env = GeneralSatelliteTasking(
        satellites=[satellite], scenario=scenario,
        rewarder=DynamicScienceReward(reward_scale=1.0),
        time_limit=duration_s, sim_rate=sim_rate,
        max_step_duration=SCHED_STEP_S, render_mode=render_mode,
    )

    flat_env = SingleSatelliteEnv(base_env)
    return DynamicObsWrapper(flat_env, event_gen, event_mgr, gamma=gamma)


# =============================================================================
#  Backwards-compatibility shim
# =============================================================================

def make_smdp_dynamic_env(*args, **kwargs) -> DynamicObsWrapper:
    """Deprecated: make_dynamic_env() already returns a full SMDP env."""
    import warnings
    warnings.warn(
        "make_smdp_dynamic_env() is deprecated. Use make_dynamic_env() directly -- "
        "the SMDP wrapper is now built into DynamicObsWrapper.",
        DeprecationWarning, stacklevel=2)
    kwargs.pop('max_sub_steps', None)
    return make_dynamic_env(*args, **kwargs)


# =============================================================================
#  Quick sanity test
# =============================================================================

if __name__ == '__main__':
    import os, logging
    os.environ.setdefault('BSK_OUTPUT_LEVEL', '2')
    os.environ.setdefault('BSK_LOG_LEVEL',    'WARNING')
    logging.basicConfig(level=logging.INFO)

    import path_setup
    ROOT       = path_setup.root_path()
    TARGETS    = os.path.join(ROOT, 'config/targets/algeria_20_targets.json')
    CLOUD_JSON = os.path.join(ROOT, 'config/cloud_reality/algeria_real_clouds.json')

    print('=' * 68)
    print('env_alsat_dynamic.py  --  Phase 3 SMDP sanity check')
    print(f'  Keplerian TTA : {"enabled" if _HAS_KEPLERIAN else "fallback (binary)"}')
    print(f'  DECAY_TAU     : {EVENT_DECAY_TAU_S:.0f} s')
    print(f'  OBS_TOTAL_DIM : {OBS_TOTAL_DIM}')
    print('=' * 68)

    env = make_dynamic_env(TARGETS, CLOUD_JSON, event_rate=2.0, seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_TOTAL_DIM,), f"Bad obs: {obs.shape}"
    assert env.action_space.n == N_TOTAL_ACTIONS
    print(f'  obs={obs.shape}  actions={env.action_space}  OK')
    print(f'  base[0:6]={obs[:6].round(3)}')
    print(f'  dyn [43:55]={obs[43:55].round(3)}')
    print(f'  sojourn[55]={obs[55]:.3f}  (0 at reset)')

    print('\n  Running 8 steps (mix static + dynamic + drift)...')
    for i in range(8):
        act = [5, 20, 21, 23, 10, 22, 0, 23][i]
        obs, r, term, trunc, info = env.step(act)
        print(f'  step {i+1}  act={act:2d}  r={r:+.4f}  '
              f'tau={info["smdp_tau_s"]:.0f}s  nsub={info["smdp_n_sub"]}  '
              f'dyn_det={info["dynamic_metrics"]["n_detected"]}  '
              f'sojourn={obs[55]:.3f}')
        if term or trunc:
            break

    env.close()
    print('\nSanity check passed.')


# ── [ROOT FIX] DYN geometric imaging bypass ───────────────────────────────
import numpy as _np_dyn, math as _math_dyn
from dynamic_event import DYN_MULTIPLIER as _DYN_MULT

_DYN_MAX_OFFNADIR_DEG = 45.0   # must match MAX_OFFNADIR_RAD in dynamic_event.py
_DYN_CLOUD_THRESH     = 0.9    # max cloud cover for successful imaging


def _dyn_imaging_check(sat, info: dict) -> float:
    """
    Called at the end of each SMDP step.
    Returns extra reward if satellite is geometrically pointing at locked DYN event.
    Also increments episode_metrics['n_dyn_imaged'].
    """
    locked_ev = getattr(sat, '_locked_dyn_event', None)
    if locked_ev is None:
        return 0.0
    if getattr(sat, '_dyn_reward_given', False):
        return 0.0   # already credited this event

    try:
        # Try multiple attribute names for satellite inertial position
        try:
            r_sat = _np_dyn.asarray(sat.dynamics.r_SC_N, dtype=float).flatten()
        except AttributeError:
            try:
                r_sat = _np_dyn.asarray(sat.dynamics.scObject.hub.r_CN_NInit, dtype=float)
            except AttributeError:
                return 0.0   # position not available — skip geometric check
        r_evt = _np_dyn.asarray(locked_ev.r_LP_P,    dtype=float).flatten()
    except AttributeError:
        return 0.0   # dynamics not initialised yet

    norm_sat = float(_np_dyn.linalg.norm(r_sat))
    if norm_sat < 1e3:
        return 0.0

    # Off-nadir angle: angle between nadir direction and vector to event
    nadir_unit = -r_sat / norm_sat
    to_evt = r_evt - r_sat
    d = float(_np_dyn.linalg.norm(to_evt))
    if d < 1.0:
        return 0.0
    to_evt_unit = to_evt / d

    cos_a = float(_np_dyn.clip(_np_dyn.dot(nadir_unit, to_evt_unit), -1.0, 1.0))
    offnadir_deg = _math_dyn.degrees(_math_dyn.acos(cos_a))

    cloud = float(getattr(locked_ev, 'cloud_cover', 1.0))

    if offnadir_deg <= _DYN_MAX_OFFNADIR_DEG and cloud < _DYN_CLOUD_THRESH:
        sat._dyn_reward_given = True
        ep = info.setdefault('episode_metrics', {})
        ep['n_dyn_imaged'] = ep.get('n_dyn_imaged', 0) + 1
        if hasattr(locked_ev, 'mark_accessed'):
            locked_ev.mark_accessed()
        pri = float(getattr(locked_ev, 'priority', 1.0))
        return DYN_MULTIPLIER * pri * (1.0 - cloud)

    return 0.0
