#!/usr/bin/env python3
"""
proper_evaluation.py  --  ALSAT-EO-1  Scientific Comparative Study
====================================================================
Implements a rigorous comparative evaluation matching the methodology of:

  Kangaslahti, Candela, Chien et al. (2024) ICRA
  Breitfeld, Candela, Chien et al. (2025) arXiv:2509.07997

Six policies evaluated:
  B1  Random              — uniform random action (lower bound)
  B2  Greedy-ignore       — greedy static only, no dynamic awareness
  B3  Greedy-dynamic-scout — greedy static + dynamic with CNN forecast
  B4  EDF-Greedy          — earliest-deadline-first for dynamic events
  B5  Oracle-Greedy       — greedy with ground-truth cloud (no CNN noise)
  A1  PPO (yours)         — trained SMDP-PPO agent

Evaluation protocol:
  - 30 episodes per policy
  - 3 evaluation seeds × 10 episodes each
  - Welch's t-test for statistical significance vs. each baseline
  - Cross-validated at 3 event rates: {0.5, 1.0, 2.0} events/hour

Outputs:
  - Console table (matches format of Kangaslahti 2024 Table 2)
  - LaTeX table (ready to paste into your thesis)
  - results/comparative_study.json  (all raw data)
  - plots/comparative_study.png     (reward bars + CI, dyn_suc bars)

Usage:
    # Baselines only (fast, no trained model needed):
    python proper_evaluation.py --baselines-only

    # Full study with your trained model:
    python proper_evaluation.py --model models/ppo_improved_final.zip \\
        --vecnorm models/vec_normalize.pkl \\
        --n-episodes 30 --seeds 42 123 456

    # Single event rate:
    python proper_evaluation.py --model models/ppo_improved_final.zip \\
        --event-rate 2.0 --n-episodes 30

    # Cross-validated event rate study:
    python proper_evaluation.py --model models/ppo_improved_final.zip \\
        --event-rates 0.5 1.0 2.0 --n-episodes 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in [
    os.path.join(_HERE, "..", "scripts", "core"),
    os.path.join(_HERE, "..", "scripts", "evaluation"),
    os.path.join(_HERE, "..", "scripts", "training"),
    os.path.join(_HERE, "..", "scripts"),
]:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import path_setup
    ROOT = path_setup.root_path()
except ImportError:
    ROOT = os.path.join(_HERE, "..")

DEFAULT_TARGETS    = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
DEFAULT_CLOUD_JSON = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")
DEFAULT_RESULTS    = os.path.join(ROOT, "results")
DEFAULT_PLOTS      = os.path.join(ROOT, "data/outputs/plots")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
#  Data classes
# =============================================================================

@dataclass
class EpisodeResult:
    policy:           str
    seed:             int
    event_rate:       float
    total_reward:     float = 0.0
    n_imaged:         int   = 0
    n_cloud_free:     int   = 0
    n_cloudy:         int   = 0
    n_dyn_detected:   int   = 0
    n_dyn_imaged:     int   = 0
    n_steps:          int   = 0
    total_slew_deg:   float = 0.0
    cloud_free_rate:  float = 0.0
    dyn_success_rate: float = 0.0
    avg_delay_s:      float = 0.0
    elapsed_s:        float = 0.0

    def finalise(self, sat_metrics: dict, dyn_metrics: dict) -> "EpisodeResult":
        n_img = sat_metrics.get("n_imaged", 0)
        n_cf  = sat_metrics.get("n_cloud_free", 0)
        self.n_imaged      = n_img
        self.n_cloud_free  = n_cf
        self.n_cloudy      = sat_metrics.get("n_cloudy", 0)
        self.cloud_free_rate  = n_cf / n_img if n_img > 0 else 0.0
        self.dyn_success_rate = dyn_metrics.get("success_rate", 0.0)
        self.avg_delay_s      = dyn_metrics.get("avg_delay_s",  0.0)
        self.n_dyn_detected   = dyn_metrics.get("n_detected",   0)
        self.n_dyn_imaged     = dyn_metrics.get("n_imaged",     0)
        return self


@dataclass
class PolicyStats:
    """Aggregated statistics for one policy at one event rate."""
    policy:      str
    event_rate:  float
    n_episodes:  int
    mean_reward: float
    std_reward:  float
    ci95_reward: float   # 1.96 × std / sqrt(n)
    mean_cf:     float
    std_cf:      float
    mean_dyn_suc: float
    std_dyn_suc:  float
    mean_dyn_det: float
    mean_dyn_img: float
    mean_delay_s: float
    raw_rewards:  List[float] = field(default_factory=list)

    @classmethod
    def from_episodes(cls, policy: str, event_rate: float,
                      episodes: List[EpisodeResult]) -> "PolicyStats":
        rewards  = [e.total_reward     for e in episodes]
        cf_rates = [e.cloud_free_rate  for e in episodes]
        dyn_suc  = [e.dyn_success_rate for e in episodes]
        delays   = [e.avg_delay_s      for e in episodes]
        det      = [e.n_dyn_detected   for e in episodes]
        img      = [e.n_dyn_imaged     for e in episodes]
        n        = len(rewards)
        std_r    = float(np.std(rewards))   if n > 1 else 0.0
        ci95     = 1.96 * std_r / math.sqrt(n) if n > 1 else 0.0
        return cls(
            policy        = policy,
            event_rate    = event_rate,
            n_episodes    = n,
            mean_reward   = float(np.mean(rewards)),
            std_reward    = std_r,
            ci95_reward   = ci95,
            mean_cf       = float(np.mean(cf_rates)),
            std_cf        = float(np.std(cf_rates)),
            mean_dyn_suc  = float(np.mean(dyn_suc)),
            std_dyn_suc   = float(np.std(dyn_suc)),
            mean_dyn_det  = float(np.mean(det)),
            mean_dyn_img  = float(np.mean(img)),
            mean_delay_s  = float(np.mean(delays)),
            raw_rewards   = rewards,
        )


# =============================================================================
#  Environment helpers
# =============================================================================

def _make_env(targets_path, cloud_path, event_rate, duration_s, seed):
    try:
        from env_dynamic_factory import Config, make_env as factory_make
        return factory_make(
            cfg             = Config.DYN_BASE,
            targets_path    = targets_path,
            cloud_json_path = cloud_path,
            event_rate      = event_rate,
            duration_s      = duration_s,
            seed            = seed,
            with_safety     = True,
        )
    except ImportError:
        from env_alsat_dynamic import make_dynamic_env
        return make_dynamic_env(
            targets_path    = targets_path,
            cloud_json_path = cloud_path,
            event_rate      = event_rate,
            duration_s      = duration_s,
            seed            = seed,
        )


def _get_sat(env):
    """Navigate wrapper stack to get the satellite object."""
    try:
        return env.env.unwrapped.satellites[0]
    except Exception:
        try:
            return env.unwrapped.satellites[0]
        except Exception:
            return None


def _get_metrics(env) -> Tuple[dict, dict]:
    """Return (sat_metrics, dyn_metrics) from env."""
    sat = _get_sat(env)
    sat_m = sat.get_metrics() if sat is not None else {}
    try:
        dyn_m = env.event_manager.get_metrics()
    except Exception:
        dyn_m = {}
    return sat_m, dyn_m


# =============================================================================
#  Policy implementations
# =============================================================================

class BasePolicy:
    """Abstract base class for all evaluation policies."""
    name: str = "abstract"

    def select_action(self, obs: np.ndarray, env, sat, sim_time: float) -> int:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    name = "B1-Random"
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._rng = np.random.default_rng(0)

    def select_action(self, obs, env, sat, sim_time):
        return int(self._rng.integers(0, self.n_actions))


class GreedyIgnoreDynamicPolicy(BasePolicy):
    """B2: Greedy on static targets only; ignores dynamic events."""
    name = "B2-Greedy-Ignore-Dyn"

    def select_action(self, obs, env, sat, sim_time):
        try:
            from env_alsat_debug import CLOUD_THRESH
        except ImportError:
            CLOUD_THRESH = 0.6
        n_static = len(sat.scenario.targets)
        drift    = env.action_space.n - 1
        best_act, best_val = drift, -1.0
        for tid, tgt in enumerate(sat.scenario.targets):
            if not _is_accessible_static(sat, tid, sim_time):
                continue
            fc = float(tgt.cloud_cover_forecast)
            if fc < CLOUD_THRESH:
                val = float(tgt.priority) * (1.0 - fc)
                if val > best_val:
                    best_val = val; best_act = tid
        return best_act


class GreedyDynamicScoutPolicy(BasePolicy):
    """B3: Greedy on static + dynamic events using CNN forecast."""
    name = "B3-Greedy-Dynamic-Scout"

    def select_action(self, obs, env, sat, sim_time):
        try:
            from env_alsat_debug import CLOUD_THRESH
            from dynamic_event import DYNAMIC_BONUS, MAX_OFFNADIR_RAD
        except ImportError:
            CLOUD_THRESH = 0.6; DYNAMIC_BONUS = 1.5
            MAX_OFFNADIR_RAD = math.radians(45)
        n_static = len(sat.scenario.targets)
        drift    = env.action_space.n - 1
        best_act, best_val = drift, -1.0

        for tid, tgt in enumerate(sat.scenario.targets):
            if not _is_accessible_static(sat, tid, sim_time):
                continue
            fc = float(tgt.cloud_cover_forecast)
            if fc < CLOUD_THRESH:
                val = float(tgt.priority) * (1.0 - fc)
                if val > best_val:
                    best_val = val; best_act = tid

        try:
            slots = env.event_manager.get_slots(sat, sim_time)
        except Exception:
            slots = []

        for si, evt in enumerate(slots):
            if evt is None: continue
            try:
                from env_alsat_debug import calculate_slew_angle_to_target
                slew = calculate_slew_angle_to_target(sat, evt)
                if slew > MAX_OFFNADIR_RAD: continue
            except Exception:
                pass
            fc  = float(evt.cloud_cover_forecast)
            val = float(evt.priority) * (1.0 - fc) + DYNAMIC_BONUS
            if val > best_val:
                best_val = val; best_act = n_static + si
        return best_act


class EDFGreedyPolicy(BasePolicy):
    """
    B4: Earliest-Deadline-First for dynamic events + greedy-priority for static.

    Stronger than B3 because:
    - Uses deadline urgency (EDF scheduling theory)
    - Accounts for remaining event lifetime
    - Cloud-weights static selection
    """
    name = "B4-EDF-Greedy"

    def select_action(self, obs, env, sat, sim_time):
        try:
            from env_alsat_debug import CLOUD_THRESH, calculate_slew_angle_to_target
            from dynamic_event import MAX_OFFNADIR_RAD, DYNAMIC_BONUS
        except ImportError:
            CLOUD_THRESH = 0.6; MAX_OFFNADIR_RAD = math.radians(45)
            DYNAMIC_BONUS = 1.5
        n_static = len(sat.scenario.targets)
        drift    = env.action_space.n - 1
        best_act, best_val = drift, -1.0

        # Dynamic: EDF — highest (priority / remaining_time)
        try:
            slots = env.event_manager.get_slots(sat, sim_time)
        except Exception:
            slots = []

        edf_best_act, edf_best_score = drift, -1.0
        for si, evt in enumerate(slots):
            if evt is None: continue
            try:
                slew = calculate_slew_angle_to_target(sat, evt)
                if slew > MAX_OFFNADIR_RAD: continue
            except Exception:
                pass
            remaining = max(1.0, evt.expiration_time - sim_time)
            fc        = float(evt.cloud_cover_forecast)
            urgency   = float(evt.priority) * (1.0 - fc) / remaining * 1000.0
            if urgency > edf_best_score:
                edf_best_score = urgency
                edf_best_act   = n_static + si

        if edf_best_act != drift:
            return edf_best_act

        # Static: greedy by priority × (1 - forecast_cloud)
        for tid, tgt in enumerate(sat.scenario.targets):
            if not _is_accessible_static(sat, tid, sim_time): continue
            fc = float(tgt.cloud_cover_forecast)
            if fc < CLOUD_THRESH:
                val = float(tgt.priority) * (1.0 - fc)
                if val > best_val:
                    best_val = val; best_act = tid
        return best_act


class OracleGreedyPolicy(BasePolicy):
    """
    B5: Greedy with ground-truth cloud cover (no CNN noise).

    This is the theoretical upper bound for cloud-aware scheduling.
    The gap between B5 and your PPO quantifies how much benefit
    remains from better cloud prediction.
    """
    name = "B5-Oracle-Greedy"

    def select_action(self, obs, env, sat, sim_time):
        try:
            from env_alsat_debug import CLOUD_THRESH, calculate_slew_angle_to_target
            from dynamic_event import MAX_OFFNADIR_RAD, DYNAMIC_BONUS
        except ImportError:
            CLOUD_THRESH = 0.6; MAX_OFFNADIR_RAD = math.radians(45)
            DYNAMIC_BONUS = 1.5
        n_static = len(sat.scenario.targets)
        drift    = env.action_space.n - 1
        best_act, best_val = drift, -1.0

        for tid, tgt in enumerate(sat.scenario.targets):
            if not _is_accessible_static(sat, tid, sim_time): continue
            cloud_truth = float(tgt.cloud_cover)   # ground truth, not forecast
            if cloud_truth < CLOUD_THRESH:
                val = float(tgt.priority) * (1.0 - cloud_truth)
                if val > best_val:
                    best_val = val; best_act = tid

        try:
            slots = env.event_manager.get_slots(sat, sim_time)
        except Exception:
            slots = []
        for si, evt in enumerate(slots):
            if evt is None: continue
            try:
                slew = calculate_slew_angle_to_target(sat, evt)
                if slew > MAX_OFFNADIR_RAD: continue
            except Exception:
                pass
            cloud_truth = float(evt.cloud_cover)   # ground truth
            val = float(evt.priority) * (1.0 - cloud_truth) + DYNAMIC_BONUS
            if val > best_val:
                best_val = val; best_act = n_static + si
        return best_act


# =============================================================================
#  Access helpers
# =============================================================================

def _is_accessible_static(satellite, target_idx: int, now: float) -> bool:
    try:
        target = satellite.scenario.targets[target_idx]
        for opp in satellite.upcoming_opportunities:
            if opp["object"] is target and opp["type"] == "target":
                t0, t1 = opp["window"]
                if t0 <= now <= t1:
                    return True
    except Exception:
        pass
    return False


# =============================================================================
#  Episode runner
# =============================================================================

def run_episode(
    policy:      "BasePolicy | Callable",
    policy_name: str,
    targets_path: str,
    cloud_path:   str,
    event_rate:   float,
    duration_s:   float,
    seed:         int,
    model=None,   # SB3 PPO model (for A1-PPO policy)
) -> EpisodeResult:
    """Run one episode with the given policy and return an EpisodeResult."""
    result = EpisodeResult(policy=policy_name, seed=seed, event_rate=event_rate)
    t0     = time.time()

    env = _make_env(targets_path, cloud_path, event_rate, duration_s, seed)
    obs, _ = env.reset(seed=seed)
    done   = False
    sat    = _get_sat(env)

    while not done:
        sim_time = float(sat.simulator.sim_time) if sat else 0.0

        if model is not None:
            # A1: PPO model
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif callable(policy):
            action = policy.select_action(obs, env, sat, sim_time)
        else:
            action = env.action_space.sample()

        obs, r, term, trunc, info = env.step(action)
        result.total_reward += r
        result.n_steps      += 1
        done = term or trunc

    sat_m, dyn_m = _get_metrics(env)
    result.elapsed_s = time.time() - t0
    env.close()
    return result.finalise(sat_m, dyn_m)


# =============================================================================
#  Full comparative study
# =============================================================================

def run_comparative_study(
    targets_path:   str,
    cloud_path:     str,
    event_rates:    List[float],
    n_episodes:     int         = 30,
    seeds:          List[int]   = (42, 123, 456),
    model_path:     Optional[str] = None,
    vecnorm_path:   Optional[str] = None,
    results_dir:    str         = DEFAULT_RESULTS,
    plots_dir:      str         = DEFAULT_PLOTS,
    duration_s:     float       = 172_800.0,
    baselines_only: bool        = False,
) -> Dict[str, Dict[float, PolicyStats]]:
    """
    Run the full comparative study.

    Returns nested dict: stats[policy_name][event_rate] = PolicyStats
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    # ── Load PPO model if provided ─────────────────────────────────────────
    ppo_model = None
    if model_path and not baselines_only:
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            ppo_model = PPO.load(model_path, device="cpu")
            logger.info(f"[eval] Loaded PPO model: {model_path}")
        except Exception as exc:
            logger.warning(f"[eval] Could not load PPO model: {exc}")

    # ── Policy registry ───────────────────────────────────────────────────
    policies: Dict[str, BasePolicy] = {
        "B1-Random":              RandomPolicy(n_actions=24),
        "B2-Greedy-Ignore-Dyn":   GreedyIgnoreDynamicPolicy(),
        "B3-Greedy-Dynamic-Scout": GreedyDynamicScoutPolicy(),
        "B4-EDF-Greedy":          EDFGreedyPolicy(),
        "B5-Oracle-Greedy":       OracleGreedyPolicy(),
    }
    if ppo_model is not None:
        policies["A1-PPO-SMDP"] = None   # sentinel → model path used

    # ── Main evaluation loop ──────────────────────────────────────────────
    all_episodes: Dict[str, Dict[float, List[EpisodeResult]]] = {
        pname: {er: [] for er in event_rates}
        for pname in policies
    }

    total_runs = len(policies) * len(event_rates) * n_episodes
    run_count  = 0

    for er in event_rates:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Event rate: {er:.1f} events/hr")
        logger.info(f"{'='*60}")

        for pname, policy in policies.items():
            logger.info(f"  Running {pname} ({n_episodes} episodes)...")
            model_for_policy = ppo_model if pname == "A1-PPO-SMDP" else None

            for ep_idx in range(n_episodes):
                # Distribute seeds evenly: cycle through seed list
                ep_seed = seeds[ep_idx % len(seeds)] * 1000 + ep_idx
                try:
                    result = run_episode(
                        policy       = policy,
                        policy_name  = pname,
                        targets_path = targets_path,
                        cloud_path   = cloud_path,
                        event_rate   = er,
                        duration_s   = duration_s,
                        seed         = ep_seed,
                        model        = model_for_policy,
                    )
                    all_episodes[pname][er].append(result)
                except Exception as exc:
                    logger.warning(f"    [{pname}] ep {ep_idx} failed: {exc}")

                run_count += 1
                if ep_idx % 5 == 0:
                    logger.info(f"    ep {ep_idx+1}/{n_episodes}  ({run_count}/{total_runs} total)")

            eps_here  = all_episodes[pname][er]
            if eps_here:
                r_mean = np.mean([e.total_reward for e in eps_here])
                d_mean = np.mean([e.dyn_success_rate for e in eps_here])
                logger.info(
                    f"    Done  mean_r={r_mean:+.2f}  dyn_suc={d_mean:.0%}"
                )

    # ── Compute statistics ────────────────────────────────────────────────
    stats: Dict[str, Dict[float, PolicyStats]] = {}
    for pname in policies:
        stats[pname] = {}
        for er in event_rates:
            eps = all_episodes[pname][er]
            if eps:
                stats[pname][er] = PolicyStats.from_episodes(pname, er, eps)

    # ── Statistical significance tests ────────────────────────────────────
    from scipy.stats import ttest_ind
    sig_tests: Dict[str, Dict] = {}

    if "A1-PPO-SMDP" in stats:
        for er in event_rates:
            if er not in stats["A1-PPO-SMDP"]:
                continue
            rl_rewards = stats["A1-PPO-SMDP"][er].raw_rewards
            for baseline_name in ["B2-Greedy-Ignore-Dyn", "B3-Greedy-Dynamic-Scout",
                                   "B4-EDF-Greedy", "B5-Oracle-Greedy"]:
                if baseline_name not in stats or er not in stats[baseline_name]:
                    continue
                bl_rewards = stats[baseline_name][er].raw_rewards
                if len(rl_rewards) < 2 or len(bl_rewards) < 2:
                    continue
                t_stat, p_val = ttest_ind(rl_rewards, bl_rewards, equal_var=False)
                key = f"A1-PPO vs {baseline_name} @ rate={er}"
                sig_tests[key] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "significant_0.05": bool(p_val < 0.05),
                    "rl_mean":  float(np.mean(rl_rewards)),
                    "bl_mean":  float(np.mean(bl_rewards)),
                    "improvement_pct": float(
                        (np.mean(rl_rewards) - np.mean(bl_rewards))
                        / (abs(np.mean(bl_rewards)) + 1e-8) * 100
                    ),
                }

    # ── Save raw results ──────────────────────────────────────────────────
    out_path = os.path.join(results_dir, "comparative_study.json")
    _save_results(all_episodes, stats, sig_tests, out_path)
    logger.info(f"\n  Raw results → {out_path}")

    # ── Print tables ──────────────────────────────────────────────────────
    for er in event_rates:
        print_comparison_table(stats, sig_tests, event_rate=er)

    # ── Print LaTeX ───────────────────────────────────────────────────────
    latex_path = os.path.join(results_dir, "table_comparison.tex")
    with open(latex_path, "w") as f:
        f.write(generate_latex_table(stats, sig_tests, event_rates))
    logger.info(f"  LaTeX table → {latex_path}")

    # ── Generate plots ────────────────────────────────────────────────────
    try:
        plot_comparison(stats, sig_tests, event_rates, plots_dir)
    except Exception as exc:
        logger.warning(f"  Plotting failed: {exc}")

    return stats


# =============================================================================
#  Output formatters
# =============================================================================

def print_comparison_table(
    stats:      Dict[str, Dict[float, PolicyStats]],
    sig_tests:  Dict[str, Dict],
    event_rate: float = 2.0,
) -> None:
    """Print a comparison table matching Kangaslahti et al. (2024) Table 2."""
    print()
    print("=" * 100)
    print(f"  COMPARATIVE STUDY — event_rate = {event_rate:.1f} events/hr")
    print("=" * 100)
    header = (
        f"  {'Policy':<28}  {'Reward':>8}  {'±95%CI':>7}  "
        f"{'CF%':>5}  {'DynSuc%':>8}  {'DynImg':>7}  "
        f"{'Delay(s)':>9}  {'vs B3 p':>8}"
    )
    print(header)
    print("  " + "-" * 96)

    for pname, er_stats in stats.items():
        if event_rate not in er_stats:
            continue
        s = er_stats[event_rate]
        sig_key = f"A1-PPO vs {pname} @ rate={event_rate}"
        p_str   = ""
        if sig_key in sig_tests:
            p = sig_tests[sig_key]["p_value"]
            p_str = f"{p:.3f}{'*' if p < 0.05 else ' '}"

        print(
            f"  {pname:<28}  "
            f"{s.mean_reward:>+8.2f}  "
            f"{s.ci95_reward:>7.2f}  "
            f"{s.mean_cf:>5.0%}  "
            f"{s.mean_dyn_suc:>8.0%}  "
            f"{s.mean_dyn_img:>7.1f}  "
            f"{s.mean_delay_s:>9.0f}  "
            f"{p_str:>8}"
        )

    print("=" * 100)

    # Value of dynamic targeting
    if "B3-Greedy-Dynamic-Scout" in stats and "B2-Greedy-Ignore-Dyn" in stats:
        if event_rate in stats["B3-Greedy-Dynamic-Scout"] and \
           event_rate in stats["B2-Greedy-Ignore-Dyn"]:
            gap = (stats["B3-Greedy-Dynamic-Scout"][event_rate].mean_reward
                   - stats["B2-Greedy-Ignore-Dyn"][event_rate].mean_reward)
            print(f"  Value of dynamic targeting (B3 - B2): {gap:+.2f}")

    # RL improvement over primary baseline (B3)
    if "A1-PPO-SMDP" in stats and "B3-Greedy-Dynamic-Scout" in stats:
        if event_rate in stats["A1-PPO-SMDP"] and \
           event_rate in stats["B3-Greedy-Dynamic-Scout"]:
            rl_r = stats["A1-PPO-SMDP"][event_rate].mean_reward
            b3_r = stats["B3-Greedy-Dynamic-Scout"][event_rate].mean_reward
            pct  = (rl_r - b3_r) / (abs(b3_r) + 1e-8) * 100
            print(f"  RL vs B3 improvement: {pct:+.1f}%")
    print()


def generate_latex_table(
    stats:      Dict[str, Dict[float, PolicyStats]],
    sig_tests:  Dict[str, Dict],
    event_rates: List[float],
) -> str:
    """Generate LaTeX table ready to paste into thesis."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Comparative study: SMDP-PPO vs. baselines across event rates. "
        r"Mean $\pm$ 95\% CI over 30 episodes (3 seeds $\times$ 10). "
        r"* denotes $p < 0.05$ vs.\ B3 (Welch's t-test).}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
    ]
    # Header with event rates
    er_str = " & ".join([f"${er:.1f}$ ev/hr" for er in event_rates])
    lines.append(
        r"\textbf{Policy} & \multicolumn{" + str(len(event_rates)) +
        r"}{c}{\textbf{Mean Total Reward} ($\pm$ 95\% CI)} \\"
    )
    lines.append(r" & " + er_str + r" \\")
    lines.append(r"\midrule")

    policy_display = {
        "B1-Random":              r"\textit{B1: Random}",
        "B2-Greedy-Ignore-Dyn":   r"\textit{B2: Greedy-Ignore-Dyn}",
        "B3-Greedy-Dynamic-Scout": r"\textit{B3: Greedy-Dynamic-Scout}",
        "B4-EDF-Greedy":          r"\textit{B4: EDF-Greedy}",
        "B5-Oracle-Greedy":       r"\textit{B5: Oracle-Greedy}",
        "A1-PPO-SMDP":            r"\textbf{A1: SMDP-PPO (ours)}",
    }

    for pname in ["B1-Random", "B2-Greedy-Ignore-Dyn", "B3-Greedy-Dynamic-Scout",
                  "B4-EDF-Greedy", "B5-Oracle-Greedy", "A1-PPO-SMDP"]:
        if pname not in stats:
            continue
        label = policy_display.get(pname, pname)
        cells = []
        for er in event_rates:
            if er not in stats[pname]:
                cells.append("---")
                continue
            s = stats[pname][er]
            sig_key = f"A1-PPO vs {pname} @ rate={er}"
            star    = ""
            if sig_key in sig_tests and sig_tests[sig_key]["significant_0.05"]:
                star = r"$^*$"
            cells.append(f"${s.mean_reward:+.1f} \\pm {s.ci95_reward:.1f}${star}")
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def plot_comparison(
    stats:      Dict[str, Dict[float, PolicyStats]],
    sig_tests:  Dict[str, Dict],
    event_rates: List[float],
    plots_dir:  str,
) -> None:
    """Generate comparison plots."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    n_er = len(event_rates)

    fig, axes = plt.subplots(1 + n_er, 2, figsize=(14, 5 + 4 * n_er))
    if not isinstance(axes[0], np.ndarray):
        axes = np.atleast_2d(axes)

    policy_colors = {
        "B1-Random":              "lightgray",
        "B2-Greedy-Ignore-Dyn":   "steelblue",
        "B3-Greedy-Dynamic-Scout": "cornflowerblue",
        "B4-EDF-Greedy":          "mediumpurple",
        "B5-Oracle-Greedy":       "mediumseagreen",
        "A1-PPO-SMDP":            "darkorange",
    }
    policy_names = list(policy_colors.keys())
    x = np.arange(len(policy_names))
    width = 0.6

    # Row 0: All event rates summary (mean reward bar chart)
    for col_idx, er in enumerate(event_rates[:2]):
        ax = axes[0][col_idx]
        means = []
        ci95s = []
        colors = []
        for pname in policy_names:
            if pname in stats and er in stats[pname]:
                s = stats[pname][er]
                means.append(s.mean_reward)
                ci95s.append(s.ci95_reward)
            else:
                means.append(0.0); ci95s.append(0.0)
            colors.append(policy_colors[pname])

        bars = ax.bar(x, means, width, color=colors, alpha=0.8, yerr=ci95s, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("B", "").replace("A", "").split("-")[0]
                            for p in policy_names], fontsize=8)
        ax.set_ylabel("Mean Total Reward")
        ax.set_title(f"Total Reward — {er:.1f} events/hr")
        ax.grid(axis="y", alpha=0.3)

    # Rows 1+: Per-event-rate detail (reward + dyn_suc)
    for row_idx, er in enumerate(event_rates):
        ax_r = axes[min(row_idx + 1, len(axes) - 1)][0]
        ax_d = axes[min(row_idx + 1, len(axes) - 1)][1]

        means_r, ci_r, means_d, ci_d, colors = [], [], [], [], []
        for pname in policy_names:
            if pname in stats and er in stats[pname]:
                s = stats[pname][er]
                means_r.append(s.mean_reward);   ci_r.append(s.ci95_reward)
                means_d.append(s.mean_dyn_suc);  ci_d.append(s.std_dyn_suc * 1.96 / math.sqrt(s.n_episodes))
            else:
                means_r.append(0.0); ci_r.append(0.0)
                means_d.append(0.0); ci_d.append(0.0)
            colors.append(policy_colors[pname])

        ax_r.bar(x, means_r, width, color=colors, alpha=0.8, yerr=ci_r, capsize=5)
        ax_r.set_xticks(x); ax_r.set_xticklabels(
            [p.split("-")[0] for p in policy_names], fontsize=8)
        ax_r.set_ylabel("Mean Reward"); ax_r.set_title(f"Reward @ {er:.1f} ev/hr")
        ax_r.grid(axis="y", alpha=0.3)

        ax_d.bar(x, means_d, width, color=colors, alpha=0.8, yerr=ci_d, capsize=5)
        ax_d.set_xticks(x); ax_d.set_xticklabels(
            [p.split("-")[0] for p in policy_names], fontsize=8)
        ax_d.set_ylabel("Dyn Success Rate"); ax_d.set_title(f"Dyn Success @ {er:.1f} ev/hr")
        ax_d.set_ylim(0, 1); ax_d.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "comparative_study.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"  Plot → {path}")


def _save_results(all_episodes, stats, sig_tests, out_path):
    """Save all raw results to JSON."""
    raw = {
        pname: {
            str(er): [asdict(e) for e in eps]
            for er, eps in er_dict.items()
        }
        for pname, er_dict in all_episodes.items()
    }
    agg = {}
    for pname, er_dict in stats.items():
        agg[pname] = {}
        for er, s in er_dict.items():
            d = asdict(s)
            d.pop("raw_rewards", None)  # omit large list
            agg[pname][str(er)] = d

    with open(out_path, "w") as f:
        json.dump({
            "raw_episodes":   raw,
            "aggregated":     agg,
            "significance":   sig_tests,
        }, f, indent=2, default=float)


# =============================================================================
#  CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ALSAT-EO-1 Proper Comparative Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--targets",       default=DEFAULT_TARGETS)
    ap.add_argument("--cloud",         default=DEFAULT_CLOUD_JSON)
    ap.add_argument("--model",         default=None,
                    help="Path to trained PPO .zip (omit for baselines-only)")
    ap.add_argument("--vecnorm",       default=None,
                    help="Path to VecNormalize .pkl (if used during training)")
    ap.add_argument("--n-episodes",    type=int,   default=30,
                    help="Episodes per policy (recommend 30 for publishable results)")
    ap.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 456])
    ap.add_argument("--event-rate",    type=float, default=None,
                    help="Single event rate (overrides --event-rates)")
    ap.add_argument("--event-rates",   type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--duration",      type=float, default=172_800.0)
    ap.add_argument("--baselines-only",action="store_true",
                    help="Skip PPO evaluation, run baselines only")
    ap.add_argument("--results-dir",   default=DEFAULT_RESULTS)
    ap.add_argument("--plots-dir",     default=DEFAULT_PLOTS)
    args = ap.parse_args()

    event_rates = [args.event_rate] if args.event_rate is not None else args.event_rates

    logger.info(f"Policies : B1-B5 (baselines) + A1 (PPO if --model provided)")
    logger.info(f"Episodes : {args.n_episodes} per policy per event rate")
    logger.info(f"Seeds    : {args.seeds}")
    logger.info(f"Rates    : {event_rates}")
    logger.info(f"Model    : {args.model or 'None (baselines only)'}")

    run_comparative_study(
        targets_path    = args.targets,
        cloud_path      = args.cloud,
        event_rates     = event_rates,
        n_episodes      = args.n_episodes,
        seeds           = args.seeds,
        model_path      = args.model,
        vecnorm_path    = args.vecnorm,
        results_dir     = args.results_dir,
        plots_dir       = args.plots_dir,
        duration_s      = args.duration,
        baselines_only  = args.baselines_only,
    )


if __name__ == "__main__":
    main()
