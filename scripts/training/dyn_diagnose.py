#!/usr/bin/env python3
"""
dyn_diagnose.py — why are DYN actions failing?
Run: python scripts/training/dyn_diagnose.py
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import path_setup  # noqa

from env_dynamic_factory import make_env, Config, obs_dim
from env_alsat_debug import CLOUD_THRESH

ROOT      = path_setup.root_path()
CKPT      = os.path.join(ROOT, "models/checkpoints_v5_seed42/ppo_best.zip")
TARGETS   = os.path.join(ROOT, "config/targets/algeria_targets.json")
CLOUD     = os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json")
CNN_PATH  = os.path.join(ROOT, "models/cloud_cnn_real.pt")

def run():
    from stable_baselines3 import PPO
    import numpy as np

    cfg = Config.DYN_REAL_VISION
    env = make_env(cfg, TARGETS, CLOUD, event_rate=1.0,
                   duration_s=172800.0, seed=99, with_safety=False,
                   cnn_path=CNN_PATH)

    model = PPO.load(CKPT, env=env)
    print(f"Loaded: {CKPT}")
    print(f"obs_dim={obs_dim(cfg)}  n_actions={env.action_space.n}")
    print()

    # Counters
    dyn_attempts   = 0
    dyn_success    = 0
    dyn_cloudy     = 0
    dyn_no_access  = 0
    dyn_empty_slot = 0
    dyn_other      = 0

    n_episodes = 10
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=99 + ep)
        done   = False
        ep_r   = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)

            # Inspect env state BEFORE step
            try:
                obj = env
                while hasattr(obj, 'env'): obj = obj.env
                sat = obj.unwrapped.satellites[0]
                now = float(sat.simulator.sim_time)
                n_static = len(sat.scenario.targets)

                if action >= n_static and action < n_static + 3:
                    slot_idx = action - n_static
                    dyn_attempts += 1

                    # Check slot
                    mgr = getattr(env, '_mgr', None) or getattr(sat, '_event_manager', None)
                    if mgr is None:
                        obj2 = env
                        while hasattr(obj2, 'env'):
                            m = getattr(obj2, '_mgr', None)
                            if m: mgr = m; break
                            obj2 = obj2.env

                    slots = mgr.get_slots(sat, now) if mgr else []
                    evt   = slots[slot_idx] if slot_idx < len(slots) else None

                    if evt is None:
                        dyn_empty_slot += 1
                        print(f"  ep{ep+1} DYN slot{slot_idx}: EMPTY SLOT "
                              f"(action masked? check masking)")
                    else:
                        cloud = float(evt.cloud_cover)
                        prio  = float(evt.priority)
                        try:
                            from env_alsat_debug import calculate_slew_angle_to_target
                            slew = math.degrees(
                                calculate_slew_angle_to_target(sat, evt))
                        except Exception:
                            slew = -1.0

                        rem = (evt.expiration_time - now) / 60
                        print(f"  ep{ep+1} DYN slot{slot_idx}: "
                              f"cloud={cloud:.2f}  slew={slew:.1f}°  "
                              f"prio={prio:.2f}  rem={rem:.0f}min  "
                              f"thresh={CLOUD_THRESH}")

                        if cloud >= CLOUD_THRESH:
                            dyn_cloudy += 1
                        elif slew > 45.0:
                            dyn_no_access += 1
                        else:
                            dyn_other += 1

            except Exception as exc:
                print(f"  [diag error] {exc}")

            obs, rew, term, trunc, info = env.step(action)
            ep_r += rew
            done  = term or trunc

            if action >= n_static and action < n_static + 3 and rew > 0.01:
                dyn_success += 1
                print(f"  ✅ DYN SUCCESS  r={rew:+.3f}")

        print(f"Episode {ep+1}: r={ep_r:+.3f}")

    print()
    print("="*50)
    print(f"DYN attempts:    {dyn_attempts}")
    print(f"DYN success:     {dyn_success}  ({dyn_success/max(1,dyn_attempts):.0%})")
    print(f"  → empty slot:  {dyn_empty_slot}  ({dyn_empty_slot/max(1,dyn_attempts):.0%})")
    print(f"  → cloudy:      {dyn_cloudy}    ({dyn_cloudy/max(1,dyn_attempts):.0%})")
    print(f"  → no access:   {dyn_no_access} ({dyn_no_access/max(1,dyn_attempts):.0%})")
    print(f"  → other:       {dyn_other}     ({dyn_other/max(1,dyn_attempts):.0%})")
    env.close()

if __name__ == "__main__":
    run()