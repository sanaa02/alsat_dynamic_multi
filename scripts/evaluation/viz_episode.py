#!/usr/bin/env python3
from __future__ import annotations
"""
viz_episode.py  --  ALSAT-EO-1  Vizard Visualization (single episode)
=====================================================================
Runs ONE evaluation episode with Basilisk Vizard streaming enabled.

🔴 CRITICAL ORDER FOR VIZARD — read this first:
────────────────────────────────────────────────
  1. Open Vizard app
  2. In Vizard: Connect tab → enter  tcp://localhost:5556  → click Connect
     (Vizard will now show "Waiting for connection..." — that is correct)
  3. THEN in terminal: python scripts/evaluation/viz_episode.py [--wait-vizard 15]
  4. Basilisk opens the port → Vizard receives the stream → 3D view appears

  If you run the Python script FIRST and then try to connect Vizard,
  it will time out because the port closes when the episode ends.

Usage
-----
    # With Vizard (open Vizard FIRST, then run this):
    python scripts/evaluation/viz_episode.py

    # With extra pause so you have time to switch to Vizard:
    python scripts/evaluation/viz_episode.py --wait-vizard 20

    # Without Vizard (just detailed step logs):
    python scripts/evaluation/viz_episode.py --no-vizard

    # Specify model explicitly:
    python scripts/evaluation/viz_episode.py \
        --model scripts/models/ppo_dynamic_final.zip
"""

# ---- ALSAT path-setup ------------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'
))
import path_setup  # noqa
# -----------------------------------------------------------------------

import argparse
import os
import sys
import time

import numpy as np

# ── Path helpers ──────────────────────────────────────────────────────────────
# _SCRIPTS is the alsat_dynamic/scripts/ directory
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# _ROOT is alsat_dynamic/ (one level above scripts/)
_ROOT    = os.path.dirname(_SCRIPTS)


def _inject_vizard_fallback(env) -> bool:
    """
    Direct Basilisk vizSupport injection — works on ALL bsk_rl versions.
    Call this AFTER env.reset() if render_mode='vizard' doesn't show anything.
    """
    try:
        from Basilisk.utilities import vizSupport
        # Walk wrappers to get GeneralSatelliteTasking
        raw = env
        while hasattr(raw, 'env'):
            raw = raw.env
        sat = raw.unwrapped.satellites[0]
        sim = sat.simulator
        scObj = sat.dynamics.scObject
        vizSupport.enableUnityVisualization(
            sim, "DynamicsTask", scObj,
            saveFile=None, liveStream=True)
        print("[VIZARD] vizSupport fallback injection OK — re-run env.reset() to apply")
        return True
    except Exception as exc:
        print(f"[VIZARD] fallback failed: {exc}")
        return False
    

def _find_model(hint: str) -> str:
    """
    Try multiple locations for a model file.
    hint can be absolute, relative to CWD, or relative to scripts/.
    """
    candidates = [
        hint,                                         # as-is (absolute or CWD-relative)
        os.path.join(_SCRIPTS, hint),                  # relative to scripts/
        os.path.join(_ROOT,    hint),                  # relative to project root
        # common default paths
        os.path.join(_SCRIPTS, "models", "ppo_dynamic_final.zip"),
        os.path.join(_ROOT,    "models", "ppo_dynamic_final.zip"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return hint  # return original so the warning message is informative

DEFAULT_MODEL      = os.path.join(_SCRIPTS, "models", "ppo_dynamic_final.zip")
DEFAULT_TARGETS    = os.path.join(_SCRIPTS, "config", "targets",
                                  "algeria_20_targets.json")
DEFAULT_CLOUD_JSON = os.path.join(_SCRIPTS, "config", "cloud_reality",
                                  "algeria_real_clouds.json")
SCHED_STEP_S       = 1200.0


def _action_name(action_idx: int, n_static: int = 20) -> str:
    if action_idx < n_static:
        return f"Image tgt-{action_idx+1:02d}"
    extra = action_idx - n_static
    return ["Task dyn-1", "Task dyn-2", "Task dyn-3", "Drift"][min(extra, 3)]


def _get_cloud_model(env):
    """Walk the wrapper chain to find _cloud_model."""
    obj = env
    for _ in range(8):  # max wrapper depth
        cm = getattr(obj, '_cloud_model', None)
        if cm is not None:
            return cm
        # try .env (gymnasium Wrapper), .base_env, .env_fns[0]()
        obj = getattr(obj, 'env', None) or getattr(obj, 'base_env', None)
        if obj is None:
            break
    return None


def run_vizard_episode(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from env_dynamic_factory import Config, make_env

    render_mode  = None if args.no_vizard else "vizard"
    model_path   = _find_model(args.model)

    print("=" * 70)
    print("  ALSAT-EO-1  Vizard Evaluation Episode")
    print("=" * 70)

    if render_mode == "vizard":
        print()
        print("  🔴 VIZARD — CORRECT ORDER:")
        print("     1. Open Vizard app  (if not already open)")
        print("     2. In Vizard: Connect → tcp://localhost:5556 → click Connect")
        print("        (Vizard shows 'Waiting for connection...'  ← that is CORRECT)")
        if args.wait_vizard > 0:
            print(f"     3. You have {args.wait_vizard}s after env.reset() to do step 2")
        print("     4. This script will then open the Basilisk stream")
        print()
    else:
        print("  Vizard OFF  (--no-vizard) — step logs only")

    print(f"  Model      : {model_path}")
    print(f"  Event rate : {args.event_rate:.1f} events/hr")
    print(f"  Duration   : {args.duration:.0f}s  ({args.duration/3600:.1f}h)")
    print()

    # Build environment
    print("[ENV] Building environment...")
    env = make_env(
        cfg             = Config.DYN_REAL_VISION,
        targets_path    = args.targets,
        cloud_json_path = args.cloud_json,
        event_rate      = args.event_rate,
        duration_s      = args.duration,
        seed            = args.seed,
        render_mode     = render_mode,
    )

    # Try to read cloud model info through wrapper layers
    cm      = _get_cloud_model(env)
    cmode   = getattr(cm, 'mode', 'unknown')
    npatches= getattr(getattr(cm, '_provider', None), 'n_patches', '?')
    print(f"[ENV] Cloud model : {cmode}  ({npatches} real MODIS patches)")
    print(f"[ENV] Obs space   : {env.observation_space.shape}  "
          f"Actions: {env.action_space.n}")

    # Load model (search multiple locations)
    if not os.path.exists(model_path):
        print(f"[WARN] Model not found at {model_path}")
        print("       Tip: model is saved to scripts/models/ — run with:")
        print(f"            --model scripts/models/ppo_dynamic_final.zip")
        print("       Using random policy for now.")
        model = None
    else:
        model = PPO.load(model_path, env=env)
        print(f"[MODEL] Loaded from {model_path}")

    # Reset and render
    # ── Vizard wait window MUST happen BEFORE env.reset() ─────────────────
    # Basilisk creates the ZMQ publisher inside reset() → InitializeSimulation().
    # Vizard must be subscribed BEFORE that call, or it misses the bind event.
    if render_mode == "vizard":
        print()
        print("  ══════════════════════════════════════════════════════")
        print("  VIZARD SETUP — do this NOW before the timer runs out:")
        print("  1. Open Vizard app")
        print("  2. File → Connect → enter:  tcp://localhost:5556")
        print("  3. Click Connect — Vizard shows 'Waiting…'  ← correct")
        print("  ══════════════════════════════════════════════════════")
        wait = getattr(args, 'wait_vizard', 15)
        for i in range(wait, 0, -1):
            print(f"     {i:2d}s remaining until simulation starts...", end='\r', flush=True)
            time.sleep(1)
        print("  ✅ Initialising simulation now...                       ")
        print()

    # Reset AFTER Vizard is connected
    obs, info = env.reset()

    _inject_vizard_fallback(env)

    # Trigger first frame
    if render_mode == "vizard":
        try:
            env.render()
        except Exception:
            pass  # rendering is automatic in env.step() — render() may be no-op

    # ── Episode loop ───────────────────────────────────────────────────────
    print("-" * 70)
    print("  EPISODE START")
    print("-" * 70)

    total_reward   = 0.0
    n_imaged       = 0
    n_cloud_free   = 0
    n_dyn_detected = 0
    n_dyn_imaged   = 0
    step           = 0
    t_start        = time.time()

    while True:
        step += 1
        sim_time_h = step * SCHED_STEP_S / 3600.0

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward

        if render_mode == "vizard":
            env.render()

        ep_m         = info.get("episode_metrics", {})
        n_imaged     = ep_m.get("n_imaged",       n_imaged)
        n_cloud_free = ep_m.get("n_cloud_free",   n_cloud_free)
        n_dyn_det    = ep_m.get("n_dyn_detected", 0)
        n_dyn_img    = ep_m.get("n_dyn_imaged",   0)
        n_dyn_detected = max(n_dyn_detected, n_dyn_det)
        n_dyn_imaged   = max(n_dyn_imaged,   n_dyn_img)

        act_lbl = _action_name(int(action))
        print(f"  t={sim_time_h:5.1f}h  step={step:4d}  act=[{act_lbl:16s}]  "
              f"r={reward:+6.3f}  tot={total_reward:+7.3f}  "
              f"img={n_imaged:2d}  cf={n_cloud_free:2d}  "
              f"dyn={n_dyn_imaged}/{n_dyn_detected}")

        if terminated or truncated:
            break

    elapsed   = time.time() - t_start
    cf_rate   = n_cloud_free   / max(n_imaged,       1)
    dyn_suc   = n_dyn_imaged   / max(n_dyn_detected, 1)

    print()
    print("=" * 70)
    print("  EPISODE COMPLETE")
    print("=" * 70)
    print(f"  Total reward  : {total_reward:+.3f}")
    print(f"  Images taken  : {n_imaged}   cloud-free: {n_cloud_free} ({cf_rate:.0%})")
    print(f"  Dynamic events: detected={n_dyn_detected}  imaged={n_dyn_imaged}  "
          f"success={dyn_suc:.0%}")
    print(f"  Steps         : {step}   Wall time: {elapsed:.1f}s")

    env.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ALSAT-EO-1 single episode with Vizard visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VIZARD ORDER (important!):
  1. Open Vizard and click Connect → tcp://localhost:5556 (Vizard waits)
  2. Then run this script (Basilisk opens the port → Vizard connects)
  3. Use --wait-vizard 20 to get a 20-second window to switch to Vizard

CORRECT MODEL PATH:
  Run from the alsat_dynamic/ folder:
    python scripts/evaluation/viz_episode.py --model scripts/models/ppo_dynamic_final.zip
""")
    ap.add_argument("--model",        default=DEFAULT_MODEL,
                    help="Path to trained PPO model (.zip)")
    ap.add_argument("--targets",      default=DEFAULT_TARGETS)
    ap.add_argument("--cloud-json",   default=DEFAULT_CLOUD_JSON)
    ap.add_argument("--event-rate",   type=float, default=2.0,
                    help="Dynamic events per hour (default 2.0)")
    ap.add_argument("--duration",     type=float, default=172800.0,
                    help="Episode duration in seconds (default 172800 = 48h)")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--no-vizard",    action="store_true",
                    help="Disable Vizard streaming (step logs only)")
    ap.add_argument("--wait-vizard",  type=int,   default=0,
                    help="Pause N seconds after reset so you can open Vizard "
                         "(e.g. --wait-vizard 20)")
    args = ap.parse_args()
    run_vizard_episode(args)


if __name__ == "__main__":
    main()
