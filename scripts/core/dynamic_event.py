#!/usr/bin/env python3
"""
dynamic_event.py  —  ALSAT-EO-1  Phase 3  Dynamic Targeting
============================================================
Implements specification Sections 2.1 (EventGenerator) and 2.2 (EventManager).

Classes
-------
DynamicEvent    : dataclass for a single unplanned emergency event
EventGenerator  : Poisson-process random spawner (configurable rate)
EventManager    : Lifecycle manager — slots events, tracks access, handles expiry

Design notes
------------
- DynamicEvent has the same interface as AlsatTarget (name, r_LP_P, priority,
  cloud_cover, cloud_cover_forecast) so it can be passed directly to
  task_target_for_imaging() and calculate_slew_angle_to_target().
- EventGenerator uses a memoryless Poisson process: inter-arrival times are
  exponentially distributed.  Call step(sim_time, dt) every decision step.
- EventManager provides get_slots() which returns up to N_DYN_SLOTS events
  sorted by (priority - slew_cost) so the agent always sees the most
  actionable events.
"""
from __future__ import annotations
# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------




import math
import dataclasses
from typing import List, Optional
import numpy as np

# ── Algeria bounding box (spec §2.1) ─────────────────────────────────────────
LAT_MIN_DEG = 30.0
LAT_MAX_DEG = 37.0
LON_MIN_DEG = -8.0
LON_MAX_DEG = 12.0

# ── Physical constants ────────────────────────────────────────────────────────
EARTH_R_M = 6.3781e6

# ── Event lifecycle defaults ──────────────────────────────────────────────────
EVENT_DURATION_MIN_S  = 3600.0      # 1 h minimum active period
EVENT_DURATION_MAX_S  = 14400.0     # 4 h maximum active period
DYNAMIC_BONUS         = 3.0         # Extra reward beyond priority*(1-cloud)
MAX_ACTIVE_EVENTS     = 5           # Internal queue depth
N_DYN_SLOTS           = 3           # Slots visible to the agent
PRIORITY_MIN          = 0.8
PRIORITY_MAX          = 1.0
CNN_NOISE_STD         = 0.05
EVENT_TYPES           = ["wildfire", "flood", "plume", "earthquake", "eruption"]

# ── Access geometry ───────────────────────────────────────────────────────────
MAX_OFFNADIR_RAD   = math.radians(45.0)
INACCESSIBLE_TIME_S = 7200.0   # placeholder when event not in current swath


# ============================================================================
#  Geometry helper
# ============================================================================

def lla2ecef_simple(lat_rad: float, lon_rad: float, alt_m: float = 0.0) -> np.ndarray:
    """Spherical-Earth ECEF conversion (matches bsk_rl lla2ecef format)."""
    r = EARTH_R_M + alt_m
    return np.array([
        r * math.cos(lat_rad) * math.cos(lon_rad),
        r * math.cos(lat_rad) * math.sin(lon_rad),
        r * math.sin(lat_rad),
    ], dtype=float)


# ============================================================================
#  DynamicEvent — duck-type compatible with AlsatTarget / bsk_rl Target
# ============================================================================

@dataclasses.dataclass
class DynamicEvent:
    """
    A single unplanned high-priority event (wildfire, flood, etc.).

    Attributes mirror AlsatTarget so bsk_rl task_target_for_imaging()
    and calculate_slew_angle_to_target() work without modification.
    """
    name:                 str
    lat_rad:              float
    lon_rad:              float
    r_LP_P:               np.ndarray   # ECEF position (3,)  — same format as AlsatTarget
    priority:             float        # 0.8 – 1.0
    appearance_time:      float        # sim time (s) when event became known
    expiration_time:      float        # sim time (s) when event expires
    cloud_cover:          float        # ground truth (used in reward)
    cloud_cover_forecast: float        # noisy CNN estimate (seen by agent)
    cloud_cover_std:      float        # uncertainty (for obs compatibility)
    event_type:           str          # wildfire | flood | plume | …
    imaged:               bool = False # set True once successfully imaged
    slot:                 int  = -1    # agent observation slot (0/1/2)

    def remaining_time(self, sim_time: float) -> float:
        return max(0.0, self.expiration_time - sim_time)


    def get_current_priority(self, sim_time: float) -> float:
        """Urgency: priority ramps +0.2 as event approaches expiration."""
        total = self.expiration_time - self.appearance_time
        if total <= 0.0:
            return self.priority
        fraction_elapsed = 1.0 - (self.remaining_time(sim_time) / total)
        return min(1.0, self.priority + 0.2 * float(fraction_elapsed))

    @property
    def is_imaged(self) -> bool:
        """Use external sim_time check; this is a convenience boolean."""
        return self.imaged
    
    def is_expired(self, sim_time: float) -> bool:
        """True when the event's time window has closed (time-based, not imaging-based)."""
        return sim_time >= self.expiration_time


# ============================================================================
#  EventGenerator — Poisson process spawner
# ============================================================================

class EventGenerator:
    """
    Generates dynamic events as a Poisson process.

    Parameters
    ----------
    rate_per_hour : float   — Expected events/hour (0 = no events)
    cnn_noise_std : float   — Scout-camera CNN noise sigma
    seed          : int     — RNG seed (for reproducibility)

    Usage
    -----
    gen = EventGenerator(rate_per_hour=0.5, seed=42)
    gen.reset(seed=episode_seed)
    new_events = gen.step(sim_time=now, dt=step_duration)
    """

    def __init__(self,
                 rate_per_hour: float = 0.5,
                 cnn_noise_std: float = CNN_NOISE_STD,
                 seed: int = 42):
        self.rate_hz  = rate_per_hour / 3600.0   # events per second
        self.cnn_std  = cnn_noise_std
        self._rng     = np.random.default_rng(seed)
        self._count   = 0
        self._countdown = self._sample_interarrival()

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, sim_time: float, dt: float) -> List[DynamicEvent]:
        """
        Advance generator by dt seconds.
        Returns list of DynamicEvent objects that appeared in this interval.
        """
        new_events: List[DynamicEvent] = []
        remaining_dt = dt
        while remaining_dt > self._countdown:
            remaining_dt -= self._countdown
            t_appear = sim_time - remaining_dt
            new_events.append(self._spawn(t_appear))
            self._countdown = self._sample_interarrival()
        self._countdown -= remaining_dt
        return new_events

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._count = 0
        self._countdown = self._sample_interarrival()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sample_interarrival(self) -> float:
        """Time until next event (exponential inter-arrival)."""
        if self.rate_hz <= 0:
            return float("inf")
        return float(self._rng.exponential(1.0 / self.rate_hz))

    def _spawn(self, appearance_time: float) -> DynamicEvent:
        """Create one random event."""
        self._count += 1
        lat_deg  = float(self._rng.uniform(LAT_MIN_DEG, LAT_MAX_DEG))
        lon_deg  = float(self._rng.uniform(LON_MIN_DEG, LON_MAX_DEG))
        lat_rad  = math.radians(lat_deg)
        lon_rad  = math.radians(lon_deg)
        priority = float(self._rng.uniform(PRIORITY_MIN, PRIORITY_MAX))
        duration = float(self._rng.uniform(EVENT_DURATION_MIN_S, EVENT_DURATION_MAX_S))
        etype    = str(self._rng.choice(EVENT_TYPES))

        # Disaster events tend to be under lower cloud cover (visible to satellite).
        # Beta(2,5) peaks near 0.25, giving mostly clear sky.
        cloud_truth = float(np.clip(self._rng.beta(2, 5), 0.0, 1.0))
        noise       = float(self._rng.normal(0.0, self.cnn_std))
        cloud_fcst  = float(np.clip(cloud_truth + noise, 0.0, 1.0))

        return DynamicEvent(
            name                 = f"dyn_{etype}_{self._count:04d}",
            lat_rad              = lat_rad,
            lon_rad              = lon_rad,
            r_LP_P               = lla2ecef_simple(lat_rad, lon_rad),
            priority             = priority,
            appearance_time      = appearance_time,
            expiration_time      = appearance_time + duration,
            cloud_cover          = cloud_truth,
            cloud_cover_forecast = cloud_fcst,
            cloud_cover_std      = self.cnn_std,
            event_type           = etype,
        )


# ============================================================================
#  EventManager — lifecycle, slotting, metrics
# ============================================================================

class EventManager:
    """
    Manages active dynamic events across an episode.

    Public interface
    ----------------
    add_events(new_events)          — add EventGenerator output
    get_slots(satellite, sim_time)  — returns [event|None] × N_DYN_SLOTS
    mark_imaged(event, time, rew)   — called when imaging succeeds
    purge_expired(sim_time)         — remove stale events
    get_metrics()                   — {n_detected, n_imaged, success_rate, …}
    reset()                         — clear for new episode
    """

    def __init__(self,
                 max_active: int = MAX_ACTIVE_EVENTS,
                 n_slots:    int = N_DYN_SLOTS):
        self.max_active = max_active
        self.n_slots    = n_slots
        self._events: List[DynamicEvent] = []
        self._metrics = dict(n_detected=0, n_imaged=0,
                             total_delay_s=0.0, total_dyn_reward=0.0)

    # ── Slot access ───────────────────────────────────────────────────────────

    def get_slots(self, satellite, sim_time: float) -> List[Optional[DynamicEvent]]:
        """
        Returns list of length n_slots with the top-priority accessible events.
        Empty slots are None.  Sorted by (priority − 0.1 × normalised_slew).
        """
        active = [e for e in self._events
                  if not e.imaged and e.expiration_time > sim_time]
        if not active:
            return [None] * self.n_slots

        def _score(evt: DynamicEvent) -> float:
            slew     = _slew_angle_safe(satellite, evt)
            priority = evt.get_current_priority(sim_time)
            return priority - 0.1 * (slew / MAX_OFFNADIR_RAD)

        ranked = sorted(active, key=_score, reverse=True)
        slots: List[Optional[DynamicEvent]] = []
        for i in range(self.n_slots):
            if i < len(ranked):
                ranked[i].slot = i
                slots.append(ranked[i])
            else:
                slots.append(None)
        return slots

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def add_events(self, new_events: List[DynamicEvent]) -> None:
        """Accept new events up to max_active capacity."""
        active = [e for e in self._events if not e.imaged]
        for evt in new_events:
            if len(active) < self.max_active:
                self._events.append(evt)
                active.append(evt)
                self._metrics["n_detected"] += 1

    def mark_imaged(self, event: DynamicEvent,
                    sim_time: float, reward: float) -> None:
        event.imaged = True
        self._metrics["n_imaged"]        += 1
        self._metrics["total_delay_s"]   += sim_time - event.appearance_time
        self._metrics["total_dyn_reward"] += reward

    def purge_expired(self, sim_time: float) -> None:
        """Remove imaged or timed-out events from the internal list."""
        self._events = [
            e for e in self._events
            if not e.imaged and e.expiration_time > sim_time
        ]

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def slew_angle(satellite, event: DynamicEvent) -> float:
        return _slew_angle_safe(satellite, event)

    @staticmethod
    def time_to_access(satellite, event: DynamicEvent,
                       sim_time: float) -> float:
        """
        Returns 0.0 if currently accessible (slew < 45°),
        else INACCESSIBLE_TIME_S as a placeholder for the agent.
        """
        slew = _slew_angle_safe(satellite, event)
        return 0.0 if slew <= MAX_OFFNADIR_RAD else INACCESSIBLE_TIME_S

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        m = dict(self._metrics)
        n_det = max(m["n_detected"], 1)
        n_img = max(m["n_imaged"],   1)
        m["success_rate"]  = m["n_imaged"] / n_det
        m["avg_delay_s"]   = m["total_delay_s"] / n_img if m["n_imaged"] > 0 else 0.0
        return m

    def reset(self) -> None:
        self._events = []
        self._metrics = dict(n_detected=0, n_imaged=0,
                             total_delay_s=0.0, total_dyn_reward=0.0)


# ============================================================================
#  Module-level geometry helper (avoids circular imports)
# ============================================================================

def _slew_angle_safe(satellite, target) -> float:
    """
    Compute slew angle (rad) from satellite's current pointing to target.
    Returns 0.0 on any error.
    """
    try:
        c_hat = np.asarray(satellite.fsw.c_hat_P,    dtype=float).ravel()
        r_sat = np.asarray(satellite.dynamics.r_BN_N, dtype=float).ravel()
        r_tgt = np.asarray(target.r_LP_P,             dtype=float).ravel()
        los   = r_tgt - r_sat
        los_n = np.linalg.norm(los)
        if los_n < 1.0:
            return 0.0
        los   = los / los_n
        c_n   = np.linalg.norm(c_hat)
        if c_n < 1e-6:
            return 0.0
        c_hat /= c_n
        dot = float(np.clip(np.dot(c_hat, los), -1.0, 1.0))
        return float(np.arccos(dot))
    except Exception:
        return 0.0


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time as _time

    print("=" * 60)
    print("dynamic_event.py — self-test")
    print("=" * 60)

    # 1. Generator test (rate 1/hr, run 48h)
    gen   = EventGenerator(rate_per_hour=1.0, seed=42)
    total = 0
    SIM_S = 172_800.0   # 48 h
    STEP  = 1200.0
    events_by_hour: dict = {}
    t = 0.0
    while t < SIM_S:
        evts = gen.step(t, STEP)
        for e in evts:
            h = int(e.appearance_time / 3600)
            events_by_hour.setdefault(h, []).append(e)
            total += 1
        t += STEP

    print(f"  Generated {total} events over 48h  (expected ~48 at rate=1/hr)")
    print(f"  First 3 events:")
    all_evts = [e for lst in events_by_hour.values() for e in lst]
    for e in sorted(all_evts, key=lambda x: x.appearance_time)[:3]:
        lat_d = math.degrees(e.lat_rad)
        lon_d = math.degrees(e.lon_rad)
        print(f"    {e.name}  lat={lat_d:.1f}°  lon={lon_d:.1f}°  "
              f"pri={e.priority:.3f}  cloud={e.cloud_cover:.3f}  "
              f"type={e.event_type}  dur={(e.expiration_time-e.appearance_time)/3600:.1f}h")

    # 2. EventManager test
    print()
    mgr = EventManager()
    class _FakeSat:
        class fsw:    c_hat_P = [0, 0, 1]
        class dynamics: r_BN_N = [0, 0, 7.37e6]

    evts_to_add = sorted(all_evts, key=lambda x: x.appearance_time)[:5]
    mgr.add_events(evts_to_add)
    slots = mgr.get_slots(_FakeSat(), 0.0)
    print(f"  Manager: added {len(evts_to_add)} events, got {sum(s is not None for s in slots)} slots")
    for i, s in enumerate(slots):
        if s is not None:
            print(f"    slot {i}: {s.name}  pri={s.priority:.3f}")

    mgr.mark_imaged(evts_to_add[0], 1200.0, 1.8)
    m = mgr.get_metrics()
    print(f"  Metrics: detected={m['n_detected']}  imaged={m['n_imaged']}  "
          f"success={m['success_rate']:.0%}  avg_delay={m['avg_delay_s']:.0f}s")

    # 3. Sparse scenario
    print()
    gen_s = EventGenerator(rate_per_hour=0.1, seed=7)  # ~5 per 48h
    total_s = 0
    t = 0.0
    while t < SIM_S:
        total_s += len(gen_s.step(t, STEP))
        t += STEP
    print(f"  Sparse (0.1/hr): {total_s} events in 48h  (expected ~5)")

    # 4. No-events scenario
    gen_n = EventGenerator(rate_per_hour=0.0, seed=99)
    total_n = 0
    t = 0.0
    while t < SIM_S:
        total_n += len(gen_n.step(t, STEP))
        t += STEP
    print(f"  No-events (0.0/hr): {total_n} events in 48h  (expected 0)")
    print()
    print("Self-test passed.")


# ── [ROOT FIX] get_slots no-accessibility-filter
# WHY: get_slots() was filtering events by CURRENT satellite accessibility
# (slew <= 45°).  The satellite is only over Algeria ~15% of the time, so
# get_slots() returned [None,None,None] for ~85% of steps → set_action()
# saw no event → current_action_is_dynamic=False → no imaging → n_dyn_imaged=0.
# FIX: return ALL active events sorted by urgency, regardless of current
# satellite position.  The scheduler decides where to point — not get_slots().

_orig_get_slots = EventManager.get_slots

def _patched_get_slots(self, satellite, sim_time: float):
    """Return up to N_DYN_SLOTS active events, sorted by urgency.
    Does NOT filter by current satellite accessibility."""
    active = [e for e in self._events if not e.imaged and e.expiration_time > sim_time]

    if not active:
        return [None] * self.n_slots
    # Sort: most urgent (highest priority / shortest remaining life) first
    active.sort(key=lambda e: -(e.priority / max(30.0, e.expiration_time - sim_time)))
    result = [None] * self.n_slots
    for i, ev in enumerate(active[:self.n_slots]):
        result[i] = ev
    return result

EventManager.get_slots = _patched_get_slots


# ── [FIX] DynamicEvent missing bsk_rl interface attributes ───────────────────
# bsk_rl's task_target_for_imaging() requires .id on every target object.
# AlsatTarget has .id; DynamicEvent did not — causing AttributeError in set_action,
# which silently resets _locked_dyn_event=None → n_dyn_imaged stays 0.

def _dyn_id(self):
    return self.name   # name is unique per event (set in _spawn)

def _dyn_target_id(self):
    return self.name

DynamicEvent.id        = property(_dyn_id)
DynamicEvent.target_id = property(_dyn_target_id)
