#!/usr/bin/env python3
"""
env_alsat_dynamic_tta_patch.py  —  Orbital TTA Drop-in for env_alsat_dynamic.py
================================================================================
Provides a Keplerian 2-body time-to-access (TTA) computation to replace
the INACCESSIBLE_TIME_S placeholder in EventManager.time_to_access().

Spec §4 "true time-to-access computation using Basilisk orbit propagator".

HOW TO APPLY
------------
1. Add this import at the top of env_alsat_dynamic.py:
       from env_alsat_dynamic_tta_patch import keplerian_tta

2. Replace the body of EventManager.time_to_access() with:
       return keplerian_tta(satellite, event, sim_time)

That's all.  Everything else stays identical.

Algorithm
---------
Uses simple leapfrog integration of the 2-body EOM (no J2, no drag)
for up to n_steps × dt_s seconds ahead.  At each step converts the
satellite ECI position to a subpoint (lat, lon) and computes the
Earth central angle to the event location.  The event is accessible
when the off-nadir angle implied by the central angle is ≤ 45°.

The conversion from ECEF event coordinates to ECI uses Earth's sidereal
rotation: θ_GAST ≈ Ω_E × sim_time  (accurate to ~0.02° over 20 min).

Returns
-------
float  — estimated seconds until next access window opens.
         0.0 if currently accessible.
         INACCESSIBLE_TIME_S if not found within horizon.
"""
from __future__ import annotations
# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------



import math
import numpy as np

# Physical constants
MU_EARTH       = 3.986004e14   # m^3/s^2
OMEGA_EARTH    = 7.292115e-5   # rad/s  (Earth rotation rate)
EARTH_R_M      = 6.3781e6      # m
ALT_KM         = 686.0
SAT_R_M        = EARTH_R_M + ALT_KM * 1e3
MAX_OFFNADIR_RAD = math.radians(45.0)
INACCESSIBLE_TIME_S = 7200.0   # placeholder when not found


def _central_angle_to_offnadir(central_angle_rad: float,
                                sat_r_m: float = SAT_R_M) -> float:
    """
    Convert Earth central angle φ (rad) to satellite off-nadir angle θ.
    Law of sines:  sin(θ) / R_E = sin(φ + θ) / r_sat
    Solved as: θ = atan2(R_E * sin(φ), r_sat - R_E * cos(φ))
    """
    return float(np.arctan2(
        EARTH_R_M * np.sin(central_angle_rad),
        sat_r_m   - EARTH_R_M * np.cos(central_angle_rad),
    ))


def keplerian_tta(satellite,
                  event,
                  sim_time:  float,
                  n_steps:   int   = 150,
                  dt_s:      float = 10.0) -> float:
    """
    Propagate the satellite orbit forward using leapfrog integration
    and return the time (s) until the event enters the 45° off-nadir cone.

    Parameters
    ----------
    satellite : AlsatSatellite with .dynamics.r_BN_N and .dynamics.v_BN_N
    event     : DynamicEvent or AlsatTarget with .r_LP_P (ECEF, metres)
    sim_time  : current simulation time (s) — used for ECEF→ECI rotation
    n_steps   : propagation horizon (default 150 × 10 s = 25 min)
    dt_s      : integration timestep (seconds)
    """
    try:
        r = np.asarray(satellite.dynamics.r_BN_N, dtype=float).ravel()
        v = np.asarray(satellite.dynamics.v_BN_N, dtype=float).ravel()
    except Exception:
        return INACCESSIBLE_TIME_S

    try:
        r_ecef = np.asarray(event.r_LP_P, dtype=float).ravel()
    except Exception:
        return INACCESSIBLE_TIME_S

    r_cur = r.copy()
    v_cur = v.copy()

    for step in range(n_steps):
        t_offset = step * dt_s

        # Convert event ECEF → ECI at (sim_time + t_offset)
        theta = OMEGA_EARTH * (sim_time + t_offset)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        r_evt_eci = np.array([
            r_ecef[0] * cos_t - r_ecef[1] * sin_t,
            r_ecef[0] * sin_t + r_ecef[1] * cos_t,
            r_ecef[2],
        ])

        # Earth central angle between satellite subpoint and event
        r_cur_n = np.linalg.norm(r_cur)
        if r_cur_n < 1.0:
            break
        r_evt_n = np.linalg.norm(r_evt_eci)
        if r_evt_n < 1.0:
            break

        cos_phi = float(np.clip(np.dot(r_cur / r_cur_n, r_evt_eci / r_evt_n),
                                -1.0, 1.0))
        phi     = float(np.arccos(cos_phi))   # central angle

        # Off-nadir angle at this future position
        off_nadir = _central_angle_to_offnadir(phi, r_cur_n)

        if off_nadir <= MAX_OFFNADIR_RAD:
            return float(t_offset)

        # Leapfrog propagation  (Störmer-Verlet, 2nd-order symplectic)
        accel  = -MU_EARTH / (r_cur_n ** 3) * r_cur
        v_half = v_cur + 0.5 * accel * dt_s
        r_cur  = r_cur + v_half * dt_s
        r_new_n = np.linalg.norm(r_cur)
        accel_new = -MU_EARTH / (r_new_n ** 3) * r_cur
        v_cur  = v_half + 0.5 * accel_new * dt_s

    return INACCESSIBLE_TIME_S


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Keplerian TTA — self test")

    # Fake satellite at ~686 km circular orbit, directly over Algeria
    # Event: near satellite subpoint → should be accessible now (TTA ≈ 0)
    r0 = np.array([SAT_R_M, 0.0, 0.0])          # at equator, prime meridian
    v0 = np.array([0.0, math.sqrt(MU_EARTH / SAT_R_M), 0.0])   # circular

    class FakeSat:
        class dynamics:
            r_BN_N = r0
            v_BN_N = v0

    class FakeEvent:
        r_LP_P = np.array([EARTH_R_M, 0.0, 0.0])  # directly below sat

    tta_near = keplerian_tta(FakeSat(), FakeEvent(), sim_time=0.0)
    print(f"  TTA (target below sat): {tta_near:.1f}s  (expected ~0)")

    class FarEvent:
        # Event on other side of Earth
        r_LP_P = np.array([-EARTH_R_M, 0.0, 0.0])

    tta_far = keplerian_tta(FakeSat(), FarEvent(), sim_time=0.0)
    print(f"  TTA (target on other side): {tta_far:.1f}s  "
          f"(expected large, ~{INACCESSIBLE_TIME_S:.0f} or orbit fraction)")

    assert tta_near < 50.0,  f"Near TTA too large: {tta_near}"
    assert tta_far  > 100.0, f"Far TTA too small: {tta_far}"
    print("  Test passed.")

# ── Integration instructions ────────────────────────────────────────────────
PATCH_INSTRUCTIONS = """
=== HOW TO INTEGRATE INTO env_alsat_dynamic.py ===

1) At the top of env_alsat_dynamic.py, add:
   from env_alsat_dynamic_tta_patch import keplerian_tta

2) Find the EventManager.time_to_access() static method:
   @staticmethod
   def time_to_access(satellite, event: DynamicEvent, sim_time: float) -> float:
       slew = _slew_angle_safe(satellite, event)
       return 0.0 if slew <= MAX_OFFNADIR_RAD else INACCESSIBLE_TIME_S

3) Replace the body with:
   @staticmethod
   def time_to_access(satellite, event: DynamicEvent, sim_time: float) -> float:
       return keplerian_tta(satellite, event, sim_time)

No other changes needed.  The SMDP observation [43:55] slot tta_norm
will now reflect the true predicted access time instead of the binary
0/INACCESSIBLE_TIME_S placeholder.
"""
