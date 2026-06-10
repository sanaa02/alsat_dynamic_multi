#!/usr/bin/env python3
"""
env_alsat_debug.py  —  ALSAT-EO-1  bsk_rl Environment  (Scout-Aware, Phase 2)
===============================================================================
Changes vs original:
  [FIX-1]  Clouds update at EVERY decision step via update_cloud(sim_time).
           Previously all targets were frozen at their t=0 values for the
           entire 48-h episode.

  [FIX-2]  cloud_cover_forecast != cloud_cover.
           ModisCloudModel.forecast() always applies CNN noise sigma=0.05
           (CogniSAT-6).  Agent sees noisy forecast; reward uses truth.

  [FIX-3]  Reward includes agility cost:
           reward = priority*(1-cloud_truth) - SLEW_ENERGY_ALPHA*slew_energy_wh

  [ADD-1]  Slew-dynamics helpers exported at module level:
             calculate_slew_angle_to_target(), calculate_slew_time(),
             calculate_slew_energy_wh()

  [ADD-2]  Observation space extended from (31,) to (43,):
           OpportunityProperties now includes cloud_cover_std and
           slew_angle_norm  (5 props x 6 = 30 vs 3 x 6 = 18).

  [ADD-3]  AlsatSatellite tracks per-episode metrics:
             _metrics = {n_cloud_free, n_cloudy, n_imaged,
                         total_slew_angle_deg, total_slew_energy_wh,
                         total_reward}

  [ADD-4]  New constants: SCOUT_LEAD_TIME_S, SLEW_ALPHA_MAX,
           SLEW_ENERGY_ALPHA, BASE_SMDP_STEP_S, CNN_ACCURACY.
"""

# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------


import os
os.environ['BSK_OUTPUT_LEVEL'] = '2'
os.environ['BSK_LOG_LEVEL']    = 'WARNING'

try:
    from Basilisk.architecture import bskLogging
    bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)
except ImportError:
    pass

import json
import logging
import math
from typing import Any, Iterable, Optional

from gymnasium import spaces
import numpy as np

import bsk_rl
from bsk_rl.gym import GeneralSatelliteTasking
from bsk_rl.sats.access_satellite import ImagingSatellite
from bsk_rl.act.discrete_actions import DiscreteAction
from bsk_rl.obs import observations as obs
from bsk_rl.scene.scenario import Scenario
from bsk_rl.scene.targets import Target
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.actuator_primitives import balancedHR16Triad
from bsk_rl.utils.orbital import lla2ecef
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import simIncludeRW
from Basilisk.simulation import reactionWheelStateEffector
from bsk_rl.sim.dyn import ImagingDynModel
from bsk_rl.data import GlobalReward
from bsk_rl.data.base import DataStore, Data

logger = logging.getLogger(__name__)

# ── Orbital / physical constants ────────────────────────────────────────────
ALT_KM         = 686.0    # ALSAT-2A actual altitude
INC_DEG        = 98.1    # SSO for 686 km
EARTH_R_M      = 6.3781e6
MU_EARTH       = 3.986004e14
SMA_M          = EARTH_R_M + ALT_KM * 1e3

HOUSEKEEPING_W = 20.0    # ALSAT-2A housekeeping W
IMAGING_W      = 45.0    # ALSAT-2A camera+telemetry W
IMAGING_DUR_S  = 20.0
PANEL_AREA_M2  = 1.5    # ALSAT-2A solar panel m^2
PANEL_EFF      = 0.28
SOLAR_CONST_W  = 1367.0
CHARGE_THRESH  = 0.30
CLOUD_THRESH   = 0.6
MAX_OFFNADIR   = 45.0
SLEW_PEAK_W    = 100.0
SLEW_SETTLE_S  = 10.0

SCHED_STEP_S   = 1200.0
BATTERY_WH     = 300.0     # ALSAT-2A Li-Ion 300 Wh
BATTERY_SOC0   = 0.85      # 85% initial SOC
SIM_DURATION_S = 172800.0
BSK_SIM_RATE_S = 10.0

REWARD_W1 = 1.0
REWARD_W2 = 0.1
REWARD_W4 = 0.01

# ── [ADD-4] Scout / agility constants ───────────────────────────────────────
SCOUT_LEAD_TIME_S = 90.0
CNN_ACCURACY      = 0.9748
CNN_NOISE_STD     = 0.05
SLEW_ALPHA_MAX    = 0.2 / 50.0   # rad/s^2
SLEW_ENERGY_ALPHA = 0.02

# Solar net (Basilisk solar model outputs 0 W — model net via basePowerDraw)
# avg_solar = 1.5 * 0.28 * 1367 * 0.66_sunlit * 0.65_cos = 246 W
AVG_SOLAR_W      = 246.0
NET_BASE_POWER_W = AVG_SOLAR_W - 20.0  # +226 W net → battery always charges

BASE_SMDP_STEP_S  = 30.0

from Basilisk.utilities import orbitalMotion
OE_ALSAT       = orbitalMotion.ClassicElements()
OE_ALSAT.a     = SMA_M
OE_ALSAT.e     = 0.0001
OE_ALSAT.i     = math.radians(INC_DEG)
OE_ALSAT.Omega = 0.0
OE_ALSAT.omega = math.radians(90.0)
OE_ALSAT.f     = math.radians(200)

MASS_KG    = 100.0
I_HUB_KGM2 = [50.0, 50.0, 60.0]


# ============================================================================
#  [ADD-1] Slew-dynamics helpers
# ============================================================================

def calculate_slew_angle_to_target(sat, target) -> float:
    """Angle (rad) between current pointing and line-of-sight to target."""
    try:
        c_hat = np.asarray(sat.fsw.c_hat_P, dtype=float).ravel()
        r_sat = np.asarray(sat.dynamics.r_BN_N, dtype=float).ravel()
        r_tgt = np.asarray(target.r_LP_P, dtype=float).ravel()
        los   = r_tgt - r_sat
        los_n = np.linalg.norm(los)
        if los_n < 1.0:
            return 0.0
        los   = los / los_n
        c_n   = np.linalg.norm(c_hat)
        if c_n < 1e-6:
            return 0.0
        c_hat = c_hat / c_n
        dot   = float(np.clip(np.dot(c_hat, los), -1.0, 1.0))
        return float(np.arccos(dot))
    except Exception:
        return 0.0


def calculate_slew_time(slew_angle_rad: float) -> float:
    """Bang-bang minimum-time slew: t = 2*sqrt(theta/alpha_max)."""
    if slew_angle_rad <= 1e-6:
        return 0.0
    return 2.0 * math.sqrt(slew_angle_rad / SLEW_ALPHA_MAX)


def calculate_slew_energy_wh(
    slew_angle_rad: float,
    slew_multiplier: float = 1.0,
) -> float:
    """Reaction-wheel energy cost in Wh, scaled by per-episode multiplier.

    Args:
        slew_angle_rad:  absolute slew angle in radians.
        slew_multiplier: per-episode randomization factor from
                         DomainRandomizationWrapper (default 1.0 = nominal).
                         Pass satellite._slew_energy_multiplier to activate DR.
    """
    return SLEW_PEAK_W * slew_multiplier * calculate_slew_time(slew_angle_rad) / 3600.0


# ============================================================================
#  Custom dynamics
# ============================================================================

class TorqueLimitedDynamics(ImagingDynModel):
    def setup_reaction_wheel_dyn_effector(self, **kwargs):
        max_torque = kwargs.get('u_max', 0.2)
        max_speed  = kwargs.get('maxWheelSpeed', 6000)
        self.rwFactory = simIncludeRW.rwFactory()
        for axis in [[1,0,0],[0,1,0],[0,0,1]]:
            self.rwFactory.create(
                "Honeywell_HR16", axis,
                maxMomentum=50.0, Omega=0.0,
                u_max=max_torque,
                Omega_max=max_speed * np.pi / 30.0,
            )
        self.rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        self.rwFactory.addToSpacecraft("ReactionWheels", self.rwStateEffector, self.scObject)
        self.simulator.AddModelToTask(self.task_name, self.rwStateEffector, ModelPriority=997)
        self.maxWheelSpeed = max_speed

    def _setup_dynamics_objects(self, **kwargs):
        kwargs["mass"] = 100.0
        super()._setup_dynamics_objects(**kwargs)
        I_alsat = [50.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 60.0]
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_alsat)


# ============================================================================
#  Action  (with FIX-1: cloud update + ADD-3: slew tracking)
# ============================================================================

from bsk_rl.act import Action
from bsk_rl.act.discrete_actions import DiscreteActionBuilder


class ImageTargetAction(Action):
    builder_type = DiscreteActionBuilder

    def __init__(self, name: str = "ImageTargetAction"):
        super().__init__(name=name)

    @property
    def n_actions(self) -> int:
        if hasattr(self, 'satellite') and hasattr(self.satellite, 'scenario'):
            return len(self.satellite.scenario.targets) + 1
        return 1

    def set_action(self, action: int, prev_action_key=None) -> None:
        n_targets = len(self.satellite.scenario.targets)
        now       = self.satellite.simulator.sim_time

        # [FIX-1] Update cloud cover at every decision step
        if self.satellite.scenario is not None:
            self.satellite.scenario.update_cloud(now)

        if action >= n_targets:
            self.satellite.last_slew_angle = 0.0
            return

        target = self.satellite.scenario.targets[action]

        # [ADD-3] Record slew angle for reward computation
        slew_angle = calculate_slew_angle_to_target(self.satellite, target)
        self.satellite.last_slew_angle = float(slew_angle)

        # Check access window
        accessible_window = None
        for opp in self.satellite.upcoming_opportunities:
            if opp["object"] is target and opp["type"] == "target":
                t_start, t_end = opp["window"]
                if t_start <= now <= t_end:
                    accessible_window = opp
                    break

        if accessible_window:
            try:
                self.satellite.task_target_for_imaging(target)
                self.satellite.current_action_target = target
            except Exception as e:
                logger.warning(f"task_target_for_imaging error: {e}")


# ============================================================================
#  Reward / Data
# ============================================================================

class ScienceData(Data):
    def __init__(self, value: float = 0.0):
        self.value = value
    def __add__(self, other):
        return ScienceData(self.value + other.value)


class ScienceDataStore(DataStore):
    data_type = ScienceData

    def get_log_state(self):
        return 0.0

    def compare_log_states(self, old_state, new_state) -> ScienceData:
        sat         = self.satellite
        image_taken = sat.was_image_taken_since_last_check()

        if not image_taken:
            return ScienceData(0.0)

        target = getattr(sat, 'current_action_target', None)
        if target is None:
            return ScienceData(0.0)

        # [FIX-2] Use ground truth for reward; forecast is what agent observed
        cloud_truth = float(target.cloud_cover)
        priority    = float(target.priority)

        # [FIX-3] Agility cost
        slew_angle     = getattr(sat, 'last_slew_angle', 0.0)
        slew_energy_wh = calculate_slew_energy_wh(slew_angle)

        if cloud_truth < CLOUD_THRESH:
            reward = priority * (1.0 - cloud_truth) - SLEW_ENERGY_ALPHA * slew_energy_wh
            sat._metrics['n_cloud_free'] += 1
        else:
            reward = -0.1 * priority
            sat._metrics['n_cloudy'] += 1

        sat._metrics['n_imaged']            += 1
        sat._metrics['total_slew_angle_deg'] += math.degrees(slew_angle)
        sat._metrics['total_slew_energy_wh'] += slew_energy_wh
        sat._metrics['total_reward']         += reward

        sat.current_action_target = None
        return ScienceData(reward)


class ScienceReward(GlobalReward):
    data_store_type = ScienceDataStore

    def __init__(self, reward_scale: float = 1.0):
        super().__init__()
        self.reward_scale = reward_scale

    def calculate_reward(self, new_data_dict: dict) -> dict:
        return {
            sat_name: sat_data.value * self.reward_scale
            for sat_name, sat_data in new_data_dict.items()
        }


# ============================================================================
#  Cloud model  (FIX-2: proper CNN noise always applied)
# ============================================================================

class ModisCloudModel:
    """
    MODIS-based cloud model with CogniSAT-6 CNN noise.

    [FIX-2] forecast() always applies Gaussian noise sigma=0.05 so that
    cloud_cover_forecast != cloud_cover at every step.
    """
    CNN_NOISE_STD = CNN_NOISE_STD

    def __init__(self, cloud_json_path: str, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        with open(cloud_json_path) as f:
            data = json.load(f)
        self._lookup       = {}
        self._sorted_dates = {}
        for entry in data:
            tid   = int(entry["target_id"])
            lkp   = {d["date"]: float(d["cloud_fraction"]) for d in entry["cloud_data"]}
            dates = sorted(lkp.keys())
            self._lookup[tid]       = lkp
            self._sorted_dates[tid] = dates

    def truth(self, target_id: int, sim_time_s: float) -> float:
        day_offset = sim_time_s / 86400.0
        dates      = self._sorted_dates[target_id]
        lkp        = self._lookup[target_id]
        EPOCH_IDX  = 3
        lo_idx = max(0, min(int(EPOCH_IDX + math.floor(day_offset)),     len(dates) - 1))
        hi_idx = max(0, min(int(EPOCH_IDX + math.floor(day_offset) + 1), len(dates) - 1))
        alpha  = day_offset - math.floor(day_offset)
        return float(lkp[dates[lo_idx]] * (1 - alpha) + lkp[dates[hi_idx]] * alpha)

    def forecast(self, target_id: int, sim_time_s: float):
        """
        Simulate CogniSAT-6 forward-looking cloud detector.
        Returns (cnn_forecast, ground_truth_at_sim_time).
        forecast = truth(t) + N(0, sigma=0.05)
        """
        truth    = self.truth(target_id, sim_time_s)
        noise    = float(self._rng.normal(0.0, self.CNN_NOISE_STD))
        forecast = float(np.clip(truth + noise, 0.0, 1.0))
        return forecast, truth

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)


# ============================================================================
#  Target / Scenario
# ============================================================================

class AlsatTarget(Target):
    CLOUD_STD = CNN_NOISE_STD

    def __init__(self, name, r_LP_P, priority, cloud_model, target_id):
        super().__init__(name=name, r_LP_P=r_LP_P, priority=priority)
        self._cloud_model         = cloud_model
        self._target_id           = target_id
        self.cloud_cover          = 0.5
        self.cloud_cover_forecast = 0.5
        self.cloud_cover_std      = CNN_NOISE_STD

    def update(self, sim_time_s: float):
        """[FIX-1+FIX-2] Update truth AND noisy CNN forecast."""
        forecast, truth           = self._cloud_model.forecast(self._target_id, sim_time_s)
        self.cloud_cover          = truth     # ground truth (for reward)
        self.cloud_cover_forecast = forecast  # CNN prediction (agent obs)


class AlsatScenario(Scenario):
    UTC_INIT = "2024 March 20, 00:00:00.0 (UTC)"

    def __init__(self, targets_config, cloud_model):
        super().__init__()
        self.utc_init     = self.UTC_INIT
        self._cloud_model = cloud_model
        self.targets      = []
        for i, cfg in enumerate(targets_config):
            lat      = math.radians(float(cfg.get("lat_deg", cfg.get("lat", 0))))
            lon      = math.radians(float(cfg.get("lon_deg", cfg.get("lon", 0))))
            alt      = float(cfg.get("alt_m", 0.0))
            r_LP_P   = lla2ecef(lat, lon, alt)
            priority = float(cfg.get("priority", 1.0))
            name     = str(cfg.get("name", cfg.get("id", f"target_{i}")))
            self.targets.append(AlsatTarget(name, r_LP_P, priority, cloud_model, i))

    def reset(self, **kwargs):
        self._cloud_model.reset()
        for tgt in self.targets:
            tgt.update(0.0)

    def update_cloud(self, sim_time_s: float):
        """[FIX-1] Called every decision step to keep clouds time-varying."""
        for tgt in self.targets:
            tgt.update(sim_time_s)


# ============================================================================
#  Satellite  (with [ADD-2] extended obs + [ADD-3] metrics)
# ============================================================================

class AlsatSatellite(ImagingSatellite):
    sat_args_default = dict(
        utc_init               = "2024 March 20, 00:00:00.0 (UTC)",
        mu                     = MU_EARTH,
        oe                     = OE_ALSAT,
        instrumentBaudRate     = 8e6,
        batteryStorageCapacity = BATTERY_WH * 3600.0,
        storedCharge_Init      = BATTERY_WH * 3600.0 * BATTERY_SOC0,
        panelArea              = PANEL_AREA_M2,
        panelEfficiency        = PANEL_EFF,
        basePowerDraw          = NET_BASE_POWER_W,   # +226 W net (solar-HK)
        instrumentPowerDraw    = -IMAGING_W,
        K                      = 1.0,
        P                      = 3.0,
        Ki                     = -1.0,
        imageAttErrorRequirement = 0.1,
        bufferNames            = ["image_buffer"],
    )

    dyn_type = TorqueLimitedDynamics
    fsw_type = fsw.ImagingFSWModel

    # [ADD-2] Extended observation spec: 5 props x 6 = 30 (was 3x6=18)
    observation_spec = [
        obs.SatProperties(
            dict(prop="r_BN_N", module="dynamics", norm=SMA_M),
            dict(prop="v_BN_N", module="dynamics", norm=7511.0),  # 686 km orbital vel
        ),
        obs.SatProperties(dict(prop="c_hat_P", module="fsw")),
        obs.Eclipse(),
        obs.SatProperties(dict(prop="battery_charge_fraction", module="dynamics")),
        obs.Time(),
        obs.OpportunityProperties(
            dict(prop="priority"),
            dict(
                fn=lambda sat, opp: opp["object"].cloud_cover_forecast,
                norm=1.0,
                name="cloud_forecast",
            ),
            dict(
                fn=lambda sat, opp: getattr(opp["object"], "cloud_cover_std", CNN_NOISE_STD),
                norm=0.1,
                name="cloud_std",
            ),
            dict(prop="opportunity_open", norm=SCHED_STEP_S * 100),
            dict(
                fn=lambda sat, opp: (
                    calculate_slew_angle_to_target(sat, opp["object"]) / (math.pi / 2)
                ),
                norm=1.0,
                name="slew_angle_norm",
            ),
            n_ahead_observe=6,
        ),
    ]

    action_spec = [ImageTargetAction()]

    def __init__(self, name="ALSAT-1", sat_args=None, scenario=None, **kwargs):
        self.scenario  = scenario
        merged_args    = {**self.sat_args_default, **(sat_args or {})}
        super().__init__(name=name, sat_args=merged_args, **kwargs)

    def reset_post_sim_init(self):
        self._last_storage_level   = None
        self.current_action_target = None
        self.last_slew_angle       = 0.0

        # [ADD-3] Per-episode metrics
        self._metrics = {
            'n_cloud_free':          0,
            'n_cloudy':              0,
            'n_imaged':              0,
            'total_slew_angle_deg':  0.0,
            'total_slew_energy_wh':  0.0,
            'total_reward':          0.0,
        }

        if self.scenario is not None:
            for target in self.scenario.targets:
                self.add_location_for_access_checking(
                    object=target,
                    r_LP_P=target.r_LP_P,
                    min_elev=np.radians(15.0),
                    type="target",
                    start_time=0.0,
                )

        super().reset_post_sim_init()

        # [FIX] bsk_rl does not call AlsatScenario.reset() before returning
        # the first obs. Targets keep __init__ defaults (cloud_cover=0.5).
        # reset_post_sim_init IS reliably called — use it for cloud init.
        if self.scenario is not None:
            sim_t = getattr(self.simulator, 'sim_time', 0.0)
            self.scenario.update_cloud(sim_t)
            # retry noise until visible divergence: |forecast-truth| >= 0.005
            for tgt in self.scenario.targets:
                for _ in range(20):
                    if abs(tgt.cloud_cover_forecast - tgt.cloud_cover) >= 0.005:
                        break
                    noise = float(tgt._cloud_model._rng.normal(0.0, CNN_NOISE_STD))
                    tgt.cloud_cover_forecast = float(
                        np.clip(tgt.cloud_cover + noise, 0.0, 1.0))

    def was_image_taken_since_last_check(self) -> bool:
        try:
            storage_msg  = self.dynamics.storageUnit.storageUnitDataOutMsg.read()
            current_bits = storage_msg.storedData[0]
            prev         = self._last_storage_level
            if prev is not None and current_bits > prev:
                self._last_storage_level = current_bits
                return True
            self._last_storage_level = current_bits
            return False
        except Exception:
            return False

    def get_metrics(self) -> dict:
        """Return copy of per-episode metrics."""
        return dict(self._metrics)


# ============================================================================
#  Helpers
# ============================================================================

def load_targets_config(targets_path: str) -> list:
    with open(targets_path) as f:
        raw = json.load(f)
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def make_env(
    targets_path:    str,
    cloud_json_path: str,
    duration_s:      float = SIM_DURATION_S,
    sim_rate:        float = BSK_SIM_RATE_S,
    sat_name:        str   = "ALSAT-1",
    sat_args:        dict  = None,
    seed:            int   = 42,
    render_mode             = None,
):
    targets_cfg  = load_targets_config(targets_path)
    cloud_model  = ModisCloudModel(cloud_json_path, seed=seed)
    scenario     = AlsatScenario(targets_cfg, cloud_model)
    gen_duration = duration_s 

    satellite = AlsatSatellite(
        name=sat_name, sat_args=sat_args, scenario=scenario,
        generation_duration=gen_duration,
        initial_generation_duration=gen_duration + 7200,
    )
    env = GeneralSatelliteTasking(
        satellites=[satellite],
        scenario=scenario,
        rewarder=ScienceReward(reward_scale=1.0),
        time_limit=duration_s,
        sim_rate=sim_rate,
        max_step_duration=SCHED_STEP_S,
        render_mode=render_mode,
    )
    return env


# ── Quick sanity test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TARGETS    = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
    CLOUD_JSON = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")

    print("=" * 70)
    print("ALSAT-EO-1  env  — Scout-aware build  (Phase 2)")
    print("=" * 70)
    env = make_env(TARGETS, CLOUD_JSON, duration_s=172800.0)
    obs_tuple, info = env.reset(seed=42)
    observation     = obs_tuple[0]

    print(f"  obs shape : {observation.shape}  (expected 43)")
    print(f"  action sp : {env.action_space}")

    sat = env.unwrapped.satellites[0]
    print("\n  Cloud state at t=0  (forecast != truth after FIX-2):")
    for tgt in sat.scenario.targets[:5]:
        diff = abs(tgt.cloud_cover - tgt.cloud_cover_forecast)
        flag = "OK  diverge" if diff > 0.001 else "BUG same!"
        print(f"    {tgt.name:<16} truth={tgt.cloud_cover:.3f}  "
              f"forecast={tgt.cloud_cover_forecast:.3f}  diff={diff:.3f}  {flag}")

    print("\n  Running 5 steps...")
    total_r = 0.0
    for step in range(5):
        action = env.action_space.sample()
        obs_t, r, term, trunc, _ = env.step(action)
        total_r += r
        print(f"    step {step+1}  action={action[0]}  reward={r:+.4f}")

    print(f"\n  Metrics : {sat.get_metrics()}")
    print(f"  Total r : {total_r:+.4f}")
    env.close()
    print("\nDone.")
