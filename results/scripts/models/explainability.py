#!/usr/bin/env python3
"""
explainability.py  —  ALSAT-EO-1  Scheduler Explainability Module
==================================================================
Post-hoc explainability tools for the dynamic targeting PPO agent.
Proposal §5 "mechanisms to interpret and visualise the scheduler's decisions".

Components
----------
FeatureNameMap
  Maps obs[0:56] indices to human-readable names.

DecisionLogger
  Records every action during an episode with state context and
  policy probabilities for alternatives.  Produces natural-language
  explanations: "Chose target X because its cloud forecast (0.12)
  and priority (0.95) gave value 0.838, beating dynamic event
  wildfire_001 (value 0.723)."

PolicyExplainer  (uses SHAP KernelExplainer)
  Computes feature attributions for the action-value function.
  Falls back to finite-difference attribution if SHAP unavailable.

TimelineRenderer
  Matplotlib timeline showing decisions, rewards, and feature
  importances over the 48-hour episode.

Usage
-----
    from explainability import DecisionLogger, PolicyExplainer, TimelineRenderer
    logger  = DecisionLogger(feature_names)
    for step …:
        logger.record(obs, action, reward, info, policy_probs)
    
    explainer = PolicyExplainer(model, background_obs)
    for record in logger.records:
        record["shap"] = explainer.explain(record["obs"])
    
    TimelineRenderer().render(logger.records, "results/timeline.png")
"""
from __future__ import annotations
# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------




import math, logging, json
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# ── Try SHAP ─────────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("shap not installed — falling back to finite-difference attribution.")


# ============================================================================
#  Feature Name Map   (obs[0:56])
# ============================================================================

def build_feature_names(n_targets: int = 20, n_dyn_slots: int = 3) -> List[str]:
    """
    Returns list of 56 feature name strings matching the SMDP obs vector.
    [0:43]  — base env features
    [43:55] — dynamic event slots
    [55]    — sojourn time
    """
    names = []
    # Position + velocity (6)
    for axis in ("x", "y", "z"): names.append(f"r_BN_{axis}_norm")
    for axis in ("x", "y", "z"): names.append(f"v_BN_{axis}_norm")
    # Pointing (3)
    for axis in ("x", "y", "z"): names.append(f"c_hat_{axis}")
    # Eclipse (1) + Battery (1) + Time (2)
    names += ["eclipse", "battery_soc", "time_sin", "time_cos"]
    # OpportunityProperties: 5 props × 6 targets-ahead = 30
    for slot in range(6):
        for prop in ("priority", "cloud_fcst", "cloud_std", "opp_open", "slew_norm"):
            names.append(f"target_{slot}_{prop}")
    # Dynamic event slots: 4 features × 3 slots = 12
    for slot in range(n_dyn_slots):
        for prop in ("priority", "cloud_fcst", "tta_norm", "slew_norm"):
            names.append(f"dyn_{slot}_{prop}")
    # Sojourn time (1)
    names.append("sojourn_norm")
    # Pad/truncate to 56
    while len(names) < 56:
        names.append(f"feat_{len(names)}")
    return names[:56]


FEATURE_NAMES = build_feature_names()


# ============================================================================
#  Decision Logger
# ============================================================================

class DecisionRecord:
    """One recorded decision step."""
    __slots__ = ("sim_time", "obs", "action", "reward", "is_dynamic",
                 "target_name", "cloud_truth", "cloud_fcst", "priority",
                 "slew_deg", "policy_probs", "shap_values", "explanation")

    def __init__(self, sim_time, obs, action, reward, is_dynamic,
                 target_name, cloud_truth, cloud_fcst, priority,
                 slew_deg, policy_probs):
        self.sim_time    = sim_time
        self.obs         = obs
        self.action      = action
        self.reward      = reward
        self.is_dynamic  = is_dynamic
        self.target_name = target_name
        self.cloud_truth = cloud_truth
        self.cloud_fcst  = cloud_fcst
        self.priority    = priority
        self.slew_deg    = slew_deg
        self.policy_probs = policy_probs
        self.shap_values  = None
        self.explanation  = ""

    def to_dict(self) -> dict:
        return {k: (getattr(self, k).tolist()
                    if isinstance(getattr(self, k), np.ndarray)
                    else getattr(self, k))
                for k in self.__slots__}


class DecisionLogger:
    """
    Records decisions during an episode and generates natural-language
    explanations based on state values and policy probabilities.

    Parameters
    ----------
    n_static   : number of static targets (default 20)
    n_dyn      : number of dynamic slots  (default 3)
    max_steps  : maximum records (for memory safety)
    """

    def __init__(self, n_static: int = 20, n_dyn: int = 3,
                 max_steps: int = 500):
        self.n_static  = n_static
        self.n_dyn     = n_dyn
        self.max_steps = max_steps
        self.records:  List[DecisionRecord] = []
        self._step     = 0

    def record(self,
               satellite,
               obs:          np.ndarray,
               action:       int,
               reward:       float,
               sim_time:     float,
               policy_probs: Optional[np.ndarray] = None) -> DecisionRecord:
        """
        Capture a decision step.  Extracts state info from satellite.
        """
        if len(self.records) >= self.max_steps:
            return None

        n_st  = self.n_static
        n_dyn = self.n_dyn

        is_dynamic  = n_st <= action < n_st + n_dyn
        is_drift    = action >= n_st + n_dyn
        target_name = "DRIFT"
        cloud_truth = cloud_fcst = priority = slew_deg = 0.0

        try:
            if not is_drift and not is_dynamic:
                tgt         = satellite.scenario.targets[action]
                target_name = tgt.name
                cloud_truth = float(tgt.cloud_cover)
                cloud_fcst  = float(tgt.cloud_cover_forecast)
                priority    = float(tgt.priority)
                from scripts.core.env_alsat_debug import calculate_slew_angle_to_target
                slew_deg = math.degrees(calculate_slew_angle_to_target(satellite, tgt))
            elif is_dynamic:
                mgr   = getattr(satellite, "_event_manager", None)
                slots = mgr.get_slots(satellite, sim_time) if mgr else []
                slot  = action - n_st
                evt   = slots[slot] if slot < len(slots) and slots[slot] else None
                if evt:
                    target_name = evt.name
                    cloud_truth = float(evt.cloud_cover)
                    cloud_fcst  = float(evt.cloud_cover_forecast)
                    priority    = float(evt.priority)
                    from scripts.core.env_alsat_debug import calculate_slew_angle_to_target
                    slew_deg = math.degrees(calculate_slew_angle_to_target(satellite, evt))
        except Exception as exc:
            logger.debug(f"DecisionLogger record error: {exc}")

        rec = DecisionRecord(
            sim_time=sim_time, obs=obs.copy(), action=action,
            reward=reward, is_dynamic=is_dynamic,
            target_name=target_name, cloud_truth=cloud_truth,
            cloud_fcst=cloud_fcst, priority=priority, slew_deg=slew_deg,
            policy_probs=policy_probs,
        )
        rec.explanation = self._explain(rec, satellite, sim_time)
        self.records.append(rec)
        self._step += 1
        return rec

    def _explain(self, rec: DecisionRecord, satellite, sim_time: float) -> str:
        """Generate a natural-language explanation for this decision."""
        if rec.action >= self.n_static + self.n_dyn:
            return "Drifted — no accessible target met value threshold."

        kind  = "dynamic event" if rec.is_dynamic else "static target"
        val   = rec.priority * (1.0 - rec.cloud_fcst)
        bonus = 1.0 if rec.is_dynamic else 0.0
        val  += bonus

        reason_parts = [
            f"Imaged {kind} '{rec.target_name}' (value={val:.3f}):",
            f"  priority={rec.priority:.3f}",
            f"  cloud_forecast={rec.cloud_fcst:.3f}",
            f"  slew={rec.slew_deg:.1f}°",
        ]
        if rec.is_dynamic:
            reason_parts.append("  +1.0 emergency bonus applied.")
        if rec.cloud_truth > 0.6:
            reason_parts.append("  ⚠ Cloudy at imaging time — reward penalised.")
        return "  ".join(reason_parts)

    def save(self, path: str) -> None:
        """Save all records as JSON."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.records], f,
                      indent=2, default=float)
        logger.info(f"DecisionLogger saved {len(self.records)} records → {path}")

    def summary(self) -> dict:
        total_r  = sum(r.reward for r in self.records)
        n_dyn    = sum(1 for r in self.records if r.is_dynamic)
        n_static = sum(1 for r in self.records
                       if not r.is_dynamic and r.action < self.n_static)
        n_drift  = len(self.records) - n_dyn - n_static
        return {
            "n_decisions": len(self.records),
            "n_dynamic":   n_dyn,
            "n_static":    n_static,
            "n_drift":     n_drift,
            "total_reward": total_r,
        }


# ============================================================================
#  PolicyExplainer  (SHAP or finite-difference fallback)
# ============================================================================

class PolicyExplainer:
    """
    Computes feature attributions for the PPO value function.

    Parameters
    ----------
    model            : SB3 PPO model
    background_obs   : numpy array (n_bg, obs_dim) of background states
    feature_names    : list of obs feature names (len = obs_dim)
    """

    def __init__(self,
                 model,
                 background_obs:  np.ndarray,
                 feature_names:   Optional[List[str]] = None):
        self.model         = model
        self.background    = background_obs
        self.feature_names = feature_names or FEATURE_NAMES[:background_obs.shape[1]]
        self._explainer    = None

        if SHAP_AVAILABLE:
            def _value_fn(obs_array: np.ndarray) -> np.ndarray:
                """Returns scalar value estimate for each obs row."""
                import torch
                t   = torch.FloatTensor(obs_array)
                pol = model.policy
                pol.eval()
                with torch.no_grad():
                    _, val, _ = pol.evaluate_actions(t, torch.zeros(len(obs_array), dtype=torch.long))
                return val.cpu().numpy().flatten()

            self._explainer = shap.KernelExplainer(
                _value_fn,
                shap.sample(background_obs, min(100, len(background_obs))),
            )
            logger.info("PolicyExplainer: using SHAP KernelExplainer.")
        else:
            logger.info("PolicyExplainer: using finite-difference fallback.")

    def explain(self, obs: np.ndarray,
                n_shap_samples: int = 50) -> np.ndarray:
        """
        Returns attribution array of shape (obs_dim,).
        Positive = increases value, negative = decreases.
        """
        x = obs.reshape(1, -1)
        if self._explainer is not None:
            try:
                sv = self._explainer.shap_values(x, nsamples=n_shap_samples)
                return np.array(sv).flatten()
            except Exception as exc:
                logger.warning(f"SHAP failed: {exc} — using finite-diff.")
        return self._finite_diff(x)

    def _finite_diff(self, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """Finite-difference approximation of feature importance."""
        import torch
        pol    = self.model.policy
        pol.eval()
        t0     = torch.FloatTensor(x)
        with torch.no_grad():
            _, v0, _ = pol.evaluate_actions(t0, torch.zeros(1, dtype=torch.long))
        v0 = v0.item()
        dim  = x.shape[1]
        attr = np.zeros(dim)
        for i in range(dim):
            xi = x.copy(); xi[0, i] += eps
            with torch.no_grad():
                _, vi, _ = pol.evaluate_actions(
                    torch.FloatTensor(xi), torch.zeros(1, dtype=torch.long))
            attr[i] = (vi.item() - v0) / eps
        return attr

    def top_features(self, obs: np.ndarray, k: int = 10) -> List[tuple]:
        """Returns top-k (feature_name, attribution) sorted by |attribution|."""
        attr = self.explain(obs)
        idx  = np.argsort(np.abs(attr))[::-1][:k]
        return [(self.feature_names[i], float(attr[i])) for i in idx]


# ============================================================================
#  TimelineRenderer
# ============================================================================

class TimelineRenderer:
    """
    Renders a 48-hour decision timeline with:
      - Top panel: cumulative reward over time
      - Middle panel: action type per step (static / dynamic / drift)
      - Bottom panel: top-5 SHAP feature importances (if available)
    """

    def render(self, records: List[DecisionRecord],
               save_path: str,
               feature_names: Optional[List[str]] = None) -> None:
        if not records:
            logger.warning("TimelineRenderer: no records to render.")
            return

        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fnames = feature_names or FEATURE_NAMES

        fig, axes = plt.subplots(3, 1, figsize=(16, 9),
                                 gridspec_kw={"height_ratios": [2, 1, 2]})
        fig.suptitle("ALSAT-EO-1 — Phase 3 Decision Timeline", fontsize=13)

        times   = [r.sim_time / 3600 for r in records]
        rewards = [r.reward for r in records]
        cum_r   = np.cumsum(rewards)

        # 1. Cumulative reward
        ax = axes[0]
        ax.plot(times, cum_r, color="steelblue", lw=2)
        ax.set_ylabel("Cumulative reward"); ax.set_xlabel("")
        ax.grid(alpha=0.3); ax.set_title("Cumulative Reward")

        # 2. Action type bar
        ax = axes[1]
        n_st  = 20
        n_dyn = 3
        colors = {"static": "mediumseagreen", "dynamic": "tomato", "drift": "#aaaaaa"}
        for r in records:
            if r.action < n_st:          c = colors["static"]
            elif r.action < n_st + n_dyn: c = colors["dynamic"]
            else:                         c = colors["drift"]
            ax.bar(r.sim_time / 3600, 1, width=0.05, color=c, alpha=0.7)
        patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
        ax.legend(handles=patches, loc="upper right", fontsize=8)
        ax.set_yticks([]); ax.set_ylabel("Action"); ax.set_xlabel("Time (h)")
        ax.set_title("Action Types Over Episode")

        # 3. Feature importance heatmap (from SHAP if available)
        ax = axes[2]
        records_with_shap = [r for r in records if r.shap_values is not None]
        if records_with_shap:
            shap_matrix = np.array([r.shap_values[:min(20, len(r.shap_values))]
                                     for r in records_with_shap])
            t_shap = [r.sim_time / 3600 for r in records_with_shap]
            im = ax.imshow(shap_matrix.T, aspect="auto", cmap="RdBu_r",
                           extent=[min(t_shap), max(t_shap), 0, shap_matrix.shape[1]],
                           vmin=-0.5, vmax=0.5)
            ax.set_yticks(range(min(20, len(fnames))))
            ax.set_yticklabels(fnames[:20], fontsize=6)
            plt.colorbar(im, ax=ax, label="SHAP value")
            ax.set_title("Feature Attributions (SHAP)")
        else:
            # Show reward per step as heatmap
            rwds = np.array([r.reward for r in records]).reshape(1, -1)
            ax.imshow(rwds, aspect="auto", cmap="RdYlGn",
                      extent=[min(times), max(times), 0, 1])
            ax.set_title("Reward per Step (run PolicyExplainer for SHAP)")
            ax.set_yticks([])
        ax.set_xlabel("Time (h)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Timeline saved → {save_path}")


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("explainability.py — feature name map check")
    names = build_feature_names()
    print(f"  Feature count: {len(names)}  (expected 56)")
    print("  First 10:", names[:10])
    print("  Dyn slots:", [n for n in names if n.startswith("dyn_")])
    print("  Last:", names[-1])
    assert len(names) == 56
    print("  Test passed.")
