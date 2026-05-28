#!/usr/bin/env python3
"""
debug_dyn_suc.py  --  Diagnose Why dyn_suc Is Always 0%
========================================================
Checks all 4 confirmed root causes and produces a clear report.

Usage
-----
    cd /home/sanaa/alsat_dynamic_improved
    python scripts/debug_dyn_suc.py --episodes 20 --random
    python scripts/debug_dyn_suc.py --episodes 20 --model models/ppo_smdp_full.zip
"""
from __future__ import annotations
import os, sys, argparse, json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import path_setup
ROOT = path_setup.root_path()
for _d in ["scripts/core", "scripts/training", "scripts/wrappers", "scripts"]:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_diagnostic(
    model_path: str  = None,
    n_episodes: int  = 20,
    seed:       int  = 42,
    event_rate: float = 2.0,
    verbose:    bool  = True,
) -> dict:
    from env_dynamic_factory import Config, make_env
    from env_alsat_debug import SCHED_STEP_S

    cnn_path = os.path.join(ROOT, "models/cloud_cnn_real.pt")
    targets  = os.path.join(ROOT, "config/targets/algeria_20_targets.json")
    cloud    = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")

    env = make_env(
        Config.DYN_REAL_VISION, targets, cloud,
        event_rate=event_rate, seed=seed,
        with_safety=True, cnn_path=cnn_path, with_action_mask=True,
    )
    n_actions = env.action_space.n          # should be 24
    N_STATIC  = 20
    DYN_IDS   = list(range(N_STATIC, N_STATIC + 3))  # [20, 21, 22]
    DRIFT_ID  = 23

    model = None
    if model_path and os.path.exists(model_path):
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=env, device="cpu")
        print(f"  Policy: {model_path}")
    else:
        print("  Policy: RANDOM")

    # ── Accumulators ──────────────────────────────────────────────────────
    action_counts     = Counter()
    ep_dyn_detected   = []
    ep_dyn_imaged     = []
    ep_rewards        = []
    metrics_key_union = set()
    obs_dyn_max       = []   # max |obs[35:56]| per step
    steps_with_event  = 0    # steps where at least one dyn event is visible

    print(f"\n{'='*60}")
    print(f"  Diagnostic: {n_episodes} episodes  event_rate={event_rate}/hr")
    print(f"{'='*60}")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_r = False, 0.0
        ep_acts = []

        while not done:
            action = int(model.predict(obs, deterministic=False)[0]
                         if model else env.action_space.sample())
            ep_acts.append(action)
            action_counts[action] += 1

            # Inspect dynamic slot obs (rough heuristic: features 35–55)
            dyn_obs = obs[35:56]
            obs_dyn_max.append(float(np.max(np.abs(dyn_obs))))
            if np.max(np.abs(dyn_obs)) > 0.01:
                steps_with_event += 1

            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            done = term or trunc

        m  = info.get("episode_metrics", {})
        metrics_key_union.update(m.keys())
        nd = m.get("n_dyn_detected", m.get("n_detected", 0))
        ni = m.get("n_dyn_imaged", 0)
        ep_dyn_detected.append(int(nd))
        ep_dyn_imaged.append(int(ni))
        ep_rewards.append(ep_r)

        if verbose:
            dyn_acts = [a for a in ep_acts if a in DYN_IDS]
            print(f"  Ep {ep+1:2d}: r={ep_r:+7.2f}  "
                  f"det={nd}  img={ni}  "
                  f"dyn_acts={len(dyn_acts)}/{len(ep_acts)}"
                  f"  ({','.join(map(str, dyn_acts[:6]))}{'…' if len(dyn_acts)>6 else ''})")

    env.close()

    # ── Report ────────────────────────────────────────────────────────────
    total_steps = max(sum(action_counts.values()), 1)
    dyn_total   = sum(action_counts.get(a, 0) for a in DYN_IDS)
    dyn_pct     = 100 * dyn_total / total_steps

    print(f"\n{'='*60}")
    print("  ACTION DISTRIBUTION")
    print(f"{'='*60}")
    for a in range(n_actions):
        cnt = action_counts.get(a, 0)
        pct = 100 * cnt / total_steps
        lbl = ("DRIFT" if a == DRIFT_ID else
               f"DYN-{a-N_STATIC}" if a in DYN_IDS else f"stat-{a:02d}")
        bar = "█" * max(0, int(pct / 2))
        print(f"  act {a:2d} [{lbl:7s}] {cnt:5d} ({pct:5.1f}%) {bar}")

    print(f"\n  ► Dynamic actions (20-22): {dyn_total}/{total_steps} = {dyn_pct:.2f}%")
    print(f"  ► Steps with visible event obs: {steps_with_event}/{total_steps} = "
          f"{100*steps_with_event/total_steps:.1f}%")

    print(f"\n{'='*60}")
    print("  EVENT METRICS PER EPISODE")
    print(f"{'='*60}")
    print(f"  n_dyn_detected  mean={np.mean(ep_dyn_detected):.2f}  "
          f"max={max(ep_dyn_detected)}  min={min(ep_dyn_detected)}")
    print(f"  n_dyn_imaged    mean={np.mean(ep_dyn_imaged):.2f}  "
          f"max={max(ep_dyn_imaged)}   min={min(ep_dyn_imaged)}")
    print(f"  episode_metrics keys: {sorted(metrics_key_union)}")

    print(f"\n{'='*60}")
    print("  ROOT CAUSE DIAGNOSIS")
    print(f"{'='*60}")

    causes = []

    # Cause A: entropy collapse
    if dyn_pct < 0.5:
        causes.append("A")
        print("  ✗ CAUSE A — ENTROPY COLLAPSE: Policy never takes dyn actions.")
        print("    FIX: Set ent_coef=0.05 at PPO start + entropy annealing callback")
        print("         Add dynamic-action entropy bonus (3× for actions 20-22)")
    else:
        print(f"  ✓ Cause A ok — policy does take dyn actions ({dyn_pct:.1f}%)")

    # Cause B: events not spawning
    if np.mean(ep_dyn_detected) < 0.5:
        causes.append("B")
        print("  ✗ CAUSE B — EVENTS NOT SPAWNING or not reaching metrics.")
        print("    FIX: Check EventGenerator.step() is called in env.step()")
        print("         Check event_rate parameter reaches EventGenerator")
        print("         Add: print(f'event spawned: {event}') in dynamic_event.py")
    else:
        print(f"  ✓ Cause B ok — events spawn ({np.mean(ep_dyn_detected):.1f}/ep avg)")

    # Cause C: dyn obs slots near zero
    if np.mean(obs_dyn_max) < 0.02:
        causes.append("C")
        print("  ✗ CAUSE C — DYNAMIC SLOT OBS NEAR ZERO (features 35-55).")
        print("    FIX: Check DynamicObsWrapper._build_obs() — event slots")
        print("         may be initialized but never written when events exist")
    else:
        print(f"  ✓ Cause C ok — dyn obs active ({np.mean(obs_dyn_max):.4f} avg)")

    # Cause D: events spawn but policy can't reach them
    if np.mean(ep_dyn_detected) > 0 and np.mean(ep_dyn_imaged) == 0 and dyn_pct > 1.0:
        causes.append("D")
        print("  ✗ CAUSE D — POLICY TRIES DYN ACTIONS but imaging always fails.")
        print("    FIX: Check CLOUD_THRESH vs event cloud_cover values")
        print("         Check slew angle feasibility: MAX_OFFNADIR_RAD vs event angles")
        print("         Check SafetyMonitor blocking dynamic actions")

    if not causes:
        print("  ✓ No root cause found at this stage — run with trained model after fixes")

    print(f"\n  Mean episode reward: {np.mean(ep_rewards):+.3f} ± {np.std(ep_rewards):.3f}")

    result = {
        "n_episodes": n_episodes, "event_rate": event_rate,
        "action_counts": dict(action_counts),
        "dyn_action_pct": dyn_pct,
        "dyn_detected_mean": float(np.mean(ep_dyn_detected)),
        "dyn_imaged_mean": float(np.mean(ep_dyn_imaged)),
        "obs_dyn_active_pct": float(100 * steps_with_event / total_steps),
        "episode_metrics_keys": sorted(metrics_key_union),
        "mean_reward": float(np.mean(ep_rewards)),
        "root_causes_found": causes,
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",   type=int,   default=20)
    ap.add_argument("--model",      type=str,   default=None)
    ap.add_argument("--random",     action="store_true")
    ap.add_argument("--event-rate", type=float, default=2.0)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--out",        type=str,   default="results/debug_dyn_suc.json")
    args = ap.parse_args()

    result = run_diagnostic(
        model_path  = None if args.random else args.model,
        n_episodes  = args.episodes,
        seed        = args.seed,
        event_rate  = args.event_rate,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Saved → {args.out}")
