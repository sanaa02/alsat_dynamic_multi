#!/usr/bin/env python3
"""
safety_monitor.py  —  ALSAT-EO-1  Runtime Safety Shield
=========================================================
Implements a lightweight runtime monitor that vetoes unsafe actions
before they are sent to the satellite actuators.

Survey §8.4 "constrained safe RL"; Proposal §3.4 "resource constraints".

Three rule categories
---------------------
R1. Battery  — predicted SOC after action must stay above min_soc.
R2. Slew     — off-nadir angle must not exceed max_offnadir_deg.
R3. Storage  — at least min_storage_kb must remain in image buffer.

When a rule is violated the monitor replaces the action with DRIFT
(the last discrete action index).  All vetoes are logged for analysis.

Integration
-----------
Call SafetyMonitor.check(satellite, action) in
DynamicImageTargetAction.set_action() BEFORE task_target_for_imaging():

    if not monitor.check(sat, action, target):
        return   # veto → drift silently

Or wrap the entire env with SafetyWrapper (recommended for evaluation).
"""
from __future__ import annotations
# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------




import math, logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
try:
    from scripts.core.env_alsat_debug import calculate_slew_angle_to_target, calculate_slew_energy_wh
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False


logger = logging.getLogger(__name__)

# ── Physical constants (matches env_alsat_debug.py) ──────────────────────────
IMAGING_W         = 45.0
IMAGING_DUR_S     = 20.0
SLEW_PEAK_W       = 100.0
BATTERY_WH        = 300.0
SAFETY_SOC_MARGIN = 0.05    # 5 % extra margin above absolute minimum

# Drift action index (N_STATIC + N_DYN_SLOTS = 20+3=23)
DRIFT_ACTION      = 23


# ============================================================================
#  VetoRecord
# ============================================================================

@dataclass
class VetoRecord:
    """One instance of a safety veto."""
    sim_time:     float
    action:       int
    reason:       str
    soc_before:   float = 0.0
    slew_deg:     float = 0.0
    action_after: int   = DRIFT_ACTION


# ============================================================================
#  SafetyMonitor
# ============================================================================

class SafetyMonitor:
    """
    Rule-based runtime safety shield.

    Parameters
    ----------
    min_soc          : float  — minimum allowed battery SOC (default 0.15 = 15 %)
    max_offnadir_deg : float  — maximum slew angle in degrees (default 45°)
    min_storage_kb   : float  — minimum remaining image buffer (default 512 KB)
    log_vetoes       : bool   — whether to log veto events
    """

    def __init__(self,
                 min_soc:          float = 0.15,
                 max_offnadir_deg: float = 45.0,
                 min_storage_kb:   float = 512.0,
                 log_vetoes:       bool  = True):
        self.min_soc          = min_soc
        self.max_offnadir_rad = math.radians(max_offnadir_deg)
        self.min_storage_bytes= min_storage_kb * 1024
        self.log_vetoes       = log_vetoes
        self._vetoes: List[VetoRecord] = []
        self._n_checked = 0
        self._n_vetoed  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self,
              satellite,
              action:  int,
              target         = None,
              sim_time: float = 0.0) -> bool:
        """
        Returns True  if the action is safe (proceed).
        Returns False if the action is unsafe (caller should substitute DRIFT).

        Parameters
        ----------
        satellite : AlsatSatellite / DynamicAlsatSatellite
        action    : proposed discrete action index
        target    : optional Target/DynamicEvent (for slew check)
        sim_time  : current simulation time (for logging)
        """
        self._n_checked += 1
        drift_action = DRIFT_ACTION   # default safe fallback

        # ── R1. Battery ───────────────────────────────────────────────────────
        try:
            soc = float(satellite.dynamics.battery_charge_fraction)
        except Exception:
            soc = 1.0   # unknown → assume safe

        # Predict SOC after action: slew + imaging energy drain
        try:
            if target is not None:
                slew_rad = calculate_slew_angle_to_target(satellite, target)
                slew_e   = calculate_slew_energy_wh(slew_rad)
                img_e    = IMAGING_W * IMAGING_DUR_S / 3600.0
                delta_soc = (slew_e + img_e) / BATTERY_WH
            else:
                delta_soc = 0.0
        except Exception:
            delta_soc = 0.0
            slew_rad  = 0.0

        predicted_soc = soc - delta_soc
        if predicted_soc < (self.min_soc + SAFETY_SOC_MARGIN):
            return self._veto(action, "R1_battery", sim_time,
                              soc, math.degrees(locals().get("slew_rad", 0.0)))

        # ── R2. Slew angle ────────────────────────────────────────────────────
        if target is not None:
            try:
                from scripts.core.env_alsat_debug import calculate_slew_angle_to_target
                slew_rad = calculate_slew_angle_to_target(satellite, target)
                if slew_rad > self.max_offnadir_rad:
                    return self._veto(action, "R2_slew", sim_time,
                                      soc, math.degrees(slew_rad))
            except Exception:
                pass   # cannot check → allow

        # ── R3. Storage ───────────────────────────────────────────────────────
        try:
            msg        = satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
            used_bits  = float(msg.storedData[0])
            capacity   = float(satellite.dynamics.storageUnit.storageCapacity)
            free_bytes  = (capacity - used_bits) / 8.0
            if free_bytes < self.min_storage_bytes:
                return self._veto(action, "R3_storage", sim_time, soc, 0.0)
        except Exception:
            pass   # cannot read storage → allow

        return True   # all checks passed

    # ── Veto helper ───────────────────────────────────────────────────────────

    def _veto(self, action: int, reason: str,
               sim_time: float, soc: float, slew_deg: float) -> bool:
        rec = VetoRecord(sim_time=sim_time, action=action, reason=reason,
                         soc_before=soc, slew_deg=slew_deg,
                         action_after=DRIFT_ACTION)
        self._vetoes.append(rec)
        self._n_vetoed += 1
        if self.log_vetoes:
            logger.info(
                f"[SAFETY] Veto  action={action}  reason={reason}  "
                f"soc={soc:.2%}  slew={slew_deg:.1f}°  t={sim_time:.0f}s")
        return False

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        by_reason: dict = {}
        for v in self._vetoes:
            by_reason[v.reason] = by_reason.get(v.reason, 0) + 1
        return {
            "n_checked": self._n_checked,
            "n_vetoed":  self._n_vetoed,
            "veto_rate": self._n_vetoed / max(self._n_checked, 1),
            "by_reason": by_reason,
        }

    def reset(self) -> None:
        self._vetoes.clear()
        self._n_checked = 0
        self._n_vetoed  = 0


# ============================================================================
#  SafetyWrapper  (optional gymnasium wrapper for evaluation)
# ============================================================================

import gymnasium as gym

class SafetyWrapper(gym.Wrapper):
    """
    Wraps any DynamicObsWrapper / SMDPDynamicWrapper.
    Intercepts step() and silently replaces unsafe actions with DRIFT.
    Records veto statistics in info dict.

    Usage
    -----
    env = SafetyWrapper(make_dynamic_env(...))
    obs, info = env.reset()
    obs, r, done, trunc, info = env.step(action)
    # info["safety_vetoed"] == True if action was overridden
    # info["safety_stats"]  == SafetyMonitor.get_stats()
    """

    def __init__(self, env: gym.Env,
                 min_soc:          float = 0.15,
                 max_offnadir_deg: float = 45.0):
        super().__init__(env)
        self._monitor = SafetyMonitor(min_soc=min_soc,
                                      max_offnadir_deg=max_offnadir_deg)

    def reset(self, **kwargs):
        self._monitor.reset()
        return self.env.reset(**kwargs)

    def step(self, action: int):
        # Try to get satellite + target for the proposed action
        sat, target, sim_time = None, None, 0.0
        try:
            sat      = self.env.unwrapped.satellites[0]
            sim_time = float(sat.simulator.sim_time)
            n_static = len(sat.scenario.targets)
            if action < n_static:
                target = sat.scenario.targets[action]
            elif action < n_static + 3:
                slot      = action - n_static
                event_mgr = getattr(sat, "_event_manager", None)
                if event_mgr is not None:
                    slots  = event_mgr.get_slots(sat, sim_time)
                    target = slots[slot] if slot < len(slots) else None
        except Exception:
            pass

        safe = self._monitor.check(sat, action, target, sim_time)
        final_action = action if safe else DRIFT_ACTION

        obs, r, term, trunc, info = self.env.step(final_action)
        info["safety_vetoed"] = not safe
        info["safety_stats"]  = self._monitor.get_stats()
        return obs, r, term, trunc, info


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("safety_monitor.py — rule definition check")

    class _MockSat:
        class dynamics:
            battery_charge_fraction = 0.10   # below 15% min
        class simulator:
            sim_time = 3600.0

    monitor = SafetyMonitor(min_soc=0.15)
    safe    = monitor.check(_MockSat(), action=5, sim_time=3600.0)
    stats   = monitor.get_stats()
    print(f"  Low battery action safe={safe}  (expected False)")
    print(f"  Stats: {stats}")
    assert not safe
    assert stats["n_vetoed"] == 1
    assert stats["by_reason"]["R1_battery"] == 1
    print("  Test passed.")
