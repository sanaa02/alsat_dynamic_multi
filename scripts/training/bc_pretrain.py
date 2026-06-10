#!/usr/bin/env python3
"""
bc_pretrain.py  --  ALSAT-EO-1 Behavioral Cloning  (v3, FIX-BC-1)
===================================================================
FIX-BC-1  Added obs_wrapper_fn parameter to collect_demonstrations().
          When the caller passes obs_wrapper_fn=lambda env: TargetIDObsWrapper(env),
          demo observations are collected with the target_id feature injected
          into each obs slot.  This ensures demo obs matches training obs exactly.

          Without this fix, BC accuracy was stuck at ~39% (mode collapse into
          the single most frequent static target action).  With it, expect 60-70%.

All other fixes from v2 (FIX-1 through FIX-4) are preserved unchanged.
"""
import os, math, logging
import numpy as np

logger = logging.getLogger(__name__)

import logging as _lg
_SKIP = frozenset(["Creating logger", "Old environments", "basePowerDraw"])
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
                            verbose=True,
                            obs_wrapper_fn=None,
                            static_episode_fraction=0.5):   # NEW
    """
    static_episode_fraction: fraction of episodes collected at event_rate=0
    (static-only). The rest use the given event_rate. Default 0.5 = 50/50.
    """
    from env_dynamic_factory import make_env, Config
    from env_alsat_debug import CLOUD_THRESH
    from dynamic_event import DYN_MULTIPLIER
    import path_setup; root = path_setup.root_path()

    cnn_path = os.path.join(root, "models/cloud_cnn_real.pt")
    cfg = Config.DYN_REAL_VISION if os.path.exists(cnn_path) else Config.DYN_MODIS

    n_static_eps = int(n_episodes * static_episode_fraction)
    n_dyn_eps    = n_episodes - n_static_eps

    if verbose:
        print(f"  Demo config: {cfg.value}")
        print(f"  Episodes: {n_static_eps} static-only (rate=0) + "
              f"{n_dyn_eps} dynamic (rate={event_rate})")

    obs_buf, act_buf = [], []
    n_total = 0

    for ep in range(n_episodes):
        # First n_static_eps episodes: rate=0 (static only, guaranteed coverage)
        ep_rate = 0.0 if ep < n_static_eps else event_rate

        env = make_env(cfg, targets_path, cloud_json_path, event_rate=ep_rate,
                       duration_s=duration_s, seed=seed+ep, with_safety=False,
                       cnn_path=cnn_path)
        if obs_wrapper_fn is not None:
            env = obs_wrapper_fn(env)

        obs, _ = env.reset(seed=seed+ep)
        done = False

        while not done:
            try:
                obj = env
                while hasattr(obj, "env"): obj = obj.env
                sat      = getattr(obj, "unwrapped", obj).satellites[0]
                now      = float(sat.simulator.sim_time)
                n_static = len(sat.scenario.targets)
                best_act = n_static + 3
                best_val = -1.0

                for tid, tgt in enumerate(sat.scenario.targets):
                    from env_alsat_debug import calculate_slew_angle_to_target
                    try:
                        slew = calculate_slew_angle_to_target(sat, tgt)
                        if slew > math.radians(45.0): continue
                    except Exception:
                        pass
                    fc = float(tgt.cloud_cover_forecast)
                    if fc < CLOUD_THRESH:
                        v = float(tgt.priority) * (1 - fc)
                        if v > best_val: best_val = v; best_act = tid

                mgr = getattr(env, "event_manager", None)
                if mgr is None:
                    obj2 = env
                    while hasattr(obj2, "env"):
                        m = getattr(obj2, "_mgr", None)
                        if m is not None: mgr = m; break
                        obj2 = obj2.env

                slots = mgr.get_slots(sat, now) if mgr else []
                for si, evt in enumerate(slots):
                    if evt is None: continue
                    fc  = float(evt.cloud_cover_forecast)
                    val = DYN_MULTIPLIER * float(evt.priority) * (1.0 - fc)
                    if val > best_val: best_val = val; best_act = n_static + si

            except Exception:
                best_act = getattr(env.action_space, "n", 24) - 1

            obs_buf.append(obs.copy())
            act_buf.append(best_act)
            obs, _, t, tr, _ = env.step(best_act)
            done = t or tr
            n_total += 1

        env.close()
        if verbose and (ep+1) % 10 == 0:
            print(f"  Collected {ep+1}/{n_episodes} demo eps ({n_total:,} transitions)")

    obs_arr = np.array(obs_buf, dtype=np.float32)
    act_arr = np.array(act_buf, dtype=np.int64)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez_compressed(save_path, obs=obs_arr, actions=act_arr)

    from collections import Counter
    c = Counter(act_arr.tolist())
    if verbose:
        print(f"  Saved {len(obs_arr):,} transitions → {save_path}")
        print(f"  Action distribution (top 8): {c.most_common(8)}")
    return obs_arr, act_arr


def behavioral_cloning(model, obs_array, act_array,
                       n_epochs=60,
                       batch_size=256,
                       lr=7e-4,
                       verbose=True):
    import torch, torch.nn as nn, torch.optim as optim
    import copy
    from torch.utils.data import TensorDataset, DataLoader
    from collections import Counter

    pol       = model.policy
    dev       = next(pol.parameters()).device
    n_actions = int(model.action_space.n)
    drift_act = n_actions - 1

    # Cap drift at 20% of non-drift count — keeps it as "do nothing" signal
    # without letting it dominate. Remove entirely only if < 50 samples.
    non_drift_idx = np.where(act_array != drift_act)[0]
    drift_idx     = np.where(act_array == drift_act)[0]
    max_drift     = max(50, len(non_drift_idx) // 4)
    if len(drift_idx) > max_drift:
        rng_bc    = np.random.default_rng(42)
        drift_idx = rng_bc.choice(drift_idx, max_drift, replace=False)
    keep     = np.sort(np.concatenate([non_drift_idx, drift_idx]))
    obs_bc   = obs_array[keep]
    acts_bc  = act_array[keep]
    n_kept   = len(obs_bc)
    if verbose:
        from collections import Counter as _C
        _dist = _C(acts_bc.tolist())
        n_drift_kept = int((acts_bc == drift_act).sum())
        print(f"  BC: {n_kept} transitions kept  "
              f"(non-drift={len(non_drift_idx)}, drift_kept={n_drift_kept})")
        print(f"  BC action breakdown: " +
              "  ".join(f"a{a}={'DYN' if a>=20 else 'static' if a<20 else 'drift'}:{c}"
                        for a, c in sorted(_dist.items())))

    counts  = Counter(acts_bc.tolist())
    weights = torch.ones(n_actions, dtype=torch.float32)
    for a, c in counts.items():
        weights[a] = min(5.0, (len(acts_bc) / (len(counts) * c)) ** 0.5)

    weights = weights.to(dev)
    if verbose:
        print(f"  BC weights: min={weights.min():.2f}  max={weights.max():.2f}  "
              f"(using {len(counts)} distinct actions)")

    obs_t = torch.FloatTensor(obs_bc).to(dev)
    act_t = torch.LongTensor(acts_bc).to(dev)
    dl    = DataLoader(TensorDataset(obs_t, act_t),
                       batch_size=min(batch_size, max(64, n_kept // 4)),
                       shuffle=True, drop_last=False)

    opt  = optim.Adam(pol.parameters(), lr=lr, weight_decay=1e-5)

    def _lr_lambda(epoch):
        if epoch < 10:
            return epoch / 10.0
        cos_val = 0.5 * (1.0 + math.cos(math.pi * (epoch - 10) / max(n_epochs - 10, 1)))
        return max(0.05, cos_val)
    sched = optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best       = float("inf")
    best_ep    = 0
    patience   = 25
    best_state = None

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

        if ml < best:
            best = ml; best_ep = ep
            best_state = copy.deepcopy(pol.state_dict())

        if ep - best_ep >= patience:
            if verbose:
                print(f"  Early stop at ep {ep+1} (patience={patience}, best ep={best_ep+1})")
            break

        if verbose and (ep + 1) % 10 == 0:
            lr_now   = sched.get_last_lr()[0]
            improved = "★" if ep == best_ep else " "
            # Per-class accuracy
            with torch.no_grad():
                all_logits = pol.get_distribution(obs_t).distribution.logits
                all_preds  = all_logits.argmax(1).cpu().numpy()
            act_np = acts_bc  # numpy array
            static_mask = act_np < (n_actions - 4)   # actions 0-19
            dyn_mask    = (act_np >= n_actions - 4) & (act_np < n_actions - 1)
            drift_mask  = act_np == drift_act
            acc_static  = (all_preds[static_mask] == act_np[static_mask]).mean() if static_mask.any() else float("nan")
            acc_dyn     = (all_preds[dyn_mask]    == act_np[dyn_mask]   ).mean() if dyn_mask.any()    else float("nan")
            acc_drift   = (all_preds[drift_mask]  == act_np[drift_mask] ).mean() if drift_mask.any()  else float("nan")
            print(f"    BC ep {ep+1:3d}/{n_epochs}  loss={ml:.4f}  acc={acc:.2%}"
                  f"  [static={acc_static:.0%} dyn={acc_dyn:.0%} drift={acc_drift:.0%}]"
                  f"  lr={lr_now:.2e} {improved}")

    if best_state is not None:
        pol.load_state_dict(best_state)
        if verbose:
            print(f"  ✓ Restored best weights from ep {best_ep+1}  (loss={best:.4f})")

    pol.set_training_mode(False)
    if verbose:
        print(f"  BC done.  Best loss={best:.4f}  Best ep={best_ep+1}/{ep+1}")
    return model