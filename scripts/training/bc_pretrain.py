#!/usr/bin/env python3
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -----------------------------------------------------------------
"""
bc_pretrain.py  --  ALSAT-EO-1  Behavioral Cloning Pretraining
==============================================================
collect_demonstrations  ->  behavioral_cloning  ->  PPO fine-tune
Uses env_dynamic_factory.make_env() (obs=56 SMDP env).
"""
import os, time, logging
import numpy as np

logger = logging.getLogger(__name__)

# Silence bsk_rl
import logging as _lg
_SKIP = frozenset(["Creating logger","Old environments","basePowerDraw"])
_orig = _lg.Logger.callHandlers
def _q(self, r):
    try:
        if any(s in r.getMessage() for s in _SKIP): return
    except Exception: pass
    _orig(self, r)
_lg.Logger.callHandlers = _q


def collect_demonstrations(targets_path, cloud_json_path, n_episodes=30,
                            event_rate=2.0, duration_s=172800.0, seed=42,
                            use_smdp=False, save_path="data/demos.npz",
                            verbose=True):
    from env_dynamic_factory import make_env, Config
    from env_alsat_debug import CLOUD_THRESH
    from dynamic_event import DYNAMIC_BONUS
    import path_setup; root=path_setup.root_path()

    cfg = Config.DYN_MODIS  # no CNN needed for demo collection
    obs_buf, act_buf = [], []
    n_total = 0

    for ep in range(n_episodes):
        env = make_env(cfg, targets_path, cloud_json_path, event_rate=event_rate,
                       duration_s=duration_s, seed=seed+ep, with_safety=False,
                       cnn_path=os.path.join(root,"models/cloud_cnn_real.pt"))
        obs, _ = env.reset(seed=seed+ep)
        done = False

        while not done:
            try:
                sat      = env.unwrapped.satellites[0]
                now      = float(sat.simulator.sim_time)
                n_static = len(sat.scenario.targets)
                best_act = n_static + 3   # drift
                best_val = -1.0

                for tid, tgt in enumerate(sat.scenario.targets):
                    accessible = any(
                        opp["object"] is tgt and opp["type"]=="target" and
                        opp["window"][0] <= now <= opp["window"][1]
                        for opp in sat.upcoming_opportunities)
                    if not accessible: continue
                    fc = float(tgt.cloud_cover_forecast)
                    if fc < CLOUD_THRESH:
                        v = float(tgt.priority)*(1-fc)
                        if v > best_val: best_val=v; best_act=tid

                mgr   = env.event_manager if hasattr(env,"event_manager") else None
                slots = mgr.get_slots(sat,now) if mgr else []
                for si, evt in enumerate(slots):
                    if evt is None: continue
                    from env_alsat_debug import calculate_slew_angle_to_target
                    from dynamic_event import MAX_OFFNADIR_RAD
                    if calculate_slew_angle_to_target(sat,evt) > MAX_OFFNADIR_RAD: continue
                    fc  = float(evt.cloud_cover_forecast)
                    val = float(evt.priority)*(1-fc)+DYNAMIC_BONUS
                    if val > best_val: best_val=val; best_act=n_static+si
            except Exception:
                best_act = getattr(env.action_space,'n',24)-1

            obs_buf.append(obs.copy()); act_buf.append(best_act)
            obs, _, t, tr, _ = env.step(best_act)
            done = t or tr; n_total += 1

        env.close()
        if verbose and (ep+1)%10==0:
            print(f"  Collected {ep+1}/{n_episodes} demo eps ({n_total:,} transitions)")

    obs_arr = np.array(obs_buf, dtype=np.float32)
    act_arr = np.array(act_buf, dtype=np.int64)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez_compressed(save_path, obs=obs_arr, actions=act_arr)
    if verbose:
        print(f"  Saved {len(obs_arr):,} transitions -> {save_path}")
    return obs_arr, act_arr




def behavioral_cloning(model, obs_array, act_array, n_epochs=50, batch_size=256,
                        lr=2e-3, verbose=True):
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from collections import Counter

    pol       = model.policy
    dev       = next(pol.parameters()).device
    n_actions = int(model.action_space.n)
    drift_act = n_actions - 1   # action 23

    # ── CHANGE A: exclude drift from BC ──────────────────────────────────
    # BC teaches target/event prioritisation — PPO learns drift on its own.
    # Including drift (38% of demos) just teaches the policy to do nothing.
    mask     = act_array != drift_act
    n_kept   = int(mask.sum())
    if n_kept < 50:          # safety: if barely any non-drift demos, use all
        mask = np.ones(len(act_array), dtype=bool)
        n_kept = len(act_array)
    obs_bc  = obs_array[mask]
    acts_bc = act_array[mask]
    if verbose:
        print(f"  BC: using {n_kept}/{len(act_array)} non-drift transitions "
              f"({n_kept/len(act_array):.0%} of demos)")

    # ── CHANGE B: SOFT class balancing — sqrt-inverse-freq, cap 2x ───────
    # sqrt dampens extreme imbalance; cap 2x means no single class dominates.
    counts  = Counter(acts_bc.tolist())
    weights = torch.ones(n_actions, dtype=torch.float32)
    for a, c in counts.items():
        weights[a] = min(2.0, (len(acts_bc) / (len(counts) * c)) ** 0.5)
    weights = weights.to(dev)
    if verbose:
        print(f"  BC weights: min={weights.min():.2f}  max={weights.max():.2f}  "
              f"(using {len(counts)} distinct actions)")

    obs_t = torch.FloatTensor(obs_bc).to(dev)
    act_t = torch.LongTensor(acts_bc).to(dev)
    dl    = DataLoader(TensorDataset(obs_t, act_t),
                       batch_size=min(batch_size, max(16, n_kept // 4)),
                       shuffle=True, drop_last=False)

    opt   = optim.Adam(pol.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    crit  = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    best  = float('inf')

    pol.train()
    for ep in range(n_epochs):
        total_loss = 0.0; correct = 0; total = 0
        for ob, ac in dl:
            opt.zero_grad()
            dist   = pol.get_distribution(ob)
            logits = dist.distribution.logits
            loss   = crit(logits, ac)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            opt.step()
            total_loss += loss.item() * len(ob)
            correct    += (logits.argmax(1) == ac).sum().item()
            total      += len(ob)
        sched.step()
        ml = total_loss / total; acc = correct / total
        if ml < best: best = ml
        if verbose and (ep + 1) % 10 == 0:
            print(f"    BC ep {ep+1:3d}/{n_epochs}  loss={ml:.4f}  acc={acc:.2%}")
    pol.set_training_mode(False)
    if verbose: print(f"  BC done.  Best loss={best:.4f}")
    return model