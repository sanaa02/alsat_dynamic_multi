# ALSAT-EO-1 RL Scheduler — Research Analysis & Improvement Guide

## 1. Diagnosis of Your Current Training Results

### What the logs show

Your training log (`ent=0.0500` every single episode) reveals **three concrete problems**:

| Symptom | Root Cause | Severity |
|---|---|---|
| `ent=0.0500` locked for 1000+ episodes | Entropy collapsed to minimum floor — policy is stuck in a local optimum | **Critical** |
| `dyn_suc=12–16%` plateau (no upward trend) | Agent picks dynamic events it cannot physically reach (TTA not continuous) | **Critical** |
| `avg100` oscillates 12–18, no upward slope | Rollout buffer too small (n_steps=144), only 1 environment | **High** |
| High variance per episode (−8 to +42) | Single env, no VecNormalize, reward scale mismatch | **Medium** |

### Entropy collapse explanation

Your logs show `ent=0.0500` for **every episode from Ep 7130 to Ep 9120** — that is 2000 consecutive episodes with entropy never moving. This means:
- The entropy reached the minimum value set in `ent_coef=0.05` very early in training.
- The policy converged to a fixed action distribution (≈50/50 static/dynamic) from which it cannot escape.
- The 50% dynamic action selection is **not learning** — it's just a frozen policy that tries dynamic actions but almost always fails the geometry check.

### Dynamic success rate explanation

The agent selects dynamic actions 50–64% of the time but only successfully images 12–16% of them. The gap (50% selection → 12% success) means:
- The satellite is physically **not in range** of the dynamic event at the time of selection.
- Your `_HAS_KEPLERIAN` flag is likely `False` (Keplerian TTA patch not found), so `time_to_access()` returns the binary placeholder `INACCESSIBLE_TIME_S=7200` for ~85% of events (Algeria is under the satellite track for ~15% of each orbit).
- The agent has no continuous TTA signal to learn "wait until in range."

---

## 2. State-of-the-Art: What Comparable Papers Achieve

### Your direct reference papers

**Herrmann, Schaub et al. (2024) — bsk_rl framework paper (Acta Astronautica)**
- Algorithm: PPO (SB3), same bsk_rl framework you use
- `n_envs`: 8–16 parallel environments
- `n_steps`: 5,120 per update (35× yours)
- Total training: 5–20M timesteps (7–28× yours)
- Metric: total images per episode
- Key: VecNormalize on observations, linear LR decay

**Kangaslahti, Candela, Chien et al. (2024) — Dynamic Targeting (ICRA 2024)**
- This is the **direct prior work** your thesis extends.
- Algorithm: SMDP-PPO (same formulation as yours)
- Dynamic event capture rate reported: **~35–50%** (vs your 12–16%)
- Key differences from your setup:
  - Continuous Keplerian TTA used as observation (gives real look-ahead signal)
  - Behavioral cloning warmup from greedy expert
  - Trained for ~5M steps with 8 parallel environments
  - Separate evaluation over 50 episodes, 3 seeds, with 95% CI

**Breitfeld, Candela, Chien et al. (2025) — Learning-Based Planning (arXiv:2509.07997)**
- Algorithm: PPO + imitation learning hybrid
- Reports: 22–31% improvement in science return over greedy baseline
- Key: action masking (illegal actions get −∞ logits), shaped reward with urgency decay
- Comparison table: reports mean ± std over ≥30 episodes

**CogniSAT-6 Flight Experiment (2025 i-SAIRAS)**
- On-orbit demonstration of dynamic targeting
- Reports: 29% more high-quality images vs ground-scheduled plan
- Used EDF-priority composite baseline as primary comparison

### How they compare and what they report

All serious papers in this space report:
1. Mean total reward per episode ± std (over ≥30 eval episodes)
2. Cloud-free image rate (CF%)
3. Dynamic event capture rate (your `dyn_suc`)
4. Average response delay for dynamic events
5. Comparison against at least 3 baselines with statistical significance test

**Performance targets from literature:**
- Dynamic event capture rate: **30–50%** (you are at 12–16%)
- Improvement over greedy-dynamic baseline: **15–35%**
- Cloud-free rate: **65–80%** of images taken should be cloud-free

---

## 3. What You Need to Change (Prioritised)

### Priority 1 — Fix Entropy Collapse (Training Script)

Replace the static `ent_coef=0.05` with a **decaying schedule**:

```python
# In PPO constructor:
ent_coef = 0.15    # Start high to explore
# Then use EntropyAnnealCallback (see callbacks_improved.py)
# Decay: 0.15 → 0.01 over 80% of training
```

### Priority 2 — Fix n_steps and Add Parallel Environments

```python
# Current (broken):
n_steps = 144   # too small
DummyVecEnv([_make])   # 1 env

# Fixed:
n_steps = 1024              # 7× larger rollout buffer
n_envs  = 4                 # 4 parallel environments
batch_size = 512            # must divide n_steps × n_envs = 4096
DummyVecEnv([_make] * 4)   # 4 envs → 4096 samples per update
```

### Priority 3 — Add VecNormalize

Your observation contains position (7×10⁶ m), velocity (7500 m/s), and cloud cover [0,1]. Without normalization the network sees wildly different scales:

```python
from stable_baselines3.common.vec_env import VecNormalize
vec = DummyVecEnv([_make] * 4)
vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
```

### Priority 4 — Linear Learning Rate Decay

```python
def linear_schedule(initial_value, final_value=1e-5):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule

learning_rate = linear_schedule(3e-4, 1e-5)
```

### Priority 5 — Add Oracle Baseline for Upper Bound

You need an "oracle" baseline that knows true cloud cover at decision time. This gives your theoretical upper bound:

```python
# Oracle: same as greedy_dynamic_scout but uses cloud_truth instead of forecast
val_static  = priority * (1 - cloud_truth)    # not forecast
val_dynamic = priority * (1 - cloud_truth) + DYNAMIC_BONUS
```

---

## 4. Scientifically Correct Comparative Study

### Required baselines (minimum for a publishable comparison)

| # | Baseline Name | Description | Purpose |
|---|---|---|---|
| B1 | Random | Uniform random action selection | Lower bound |
| B2 | Greedy-ignore-dynamic | Greedy static only, ignores events | Legacy system without DT |
| B3 | Greedy-dynamic-scout | Greedy over static+dynamic using CNN forecast | Primary heuristic competitor |
| B4 | EDF-Greedy | Earliest-deadline-first for dynamic, greedy-priority for static | Scheduling theory baseline |
| B5 | Oracle-Greedy | Greedy with ground-truth cloud (no CNN noise) | Upper bound on information gain |
| A1 | **Your PPO** | SMDP-PPO with CNN look-ahead | Your approach |

### Evaluation protocol (matches Kangaslahti et al. 2024 methodology)

```
For each policy P in {B1, B2, B3, B4, B5, A1}:
  For each seed s in {42, 123, 456}:
    Run 30 episodes with seed offset s×1000
    Record: total_reward, cf_rate, dyn_success_rate, avg_delay_s, n_dyn_imaged
  
  Aggregate: mean ± std over 90 episodes (30 eps × 3 seeds)
```

### Statistical testing

Use **Welch's t-test** (not Student's t-test — your variances differ across policies):
```python
from scipy.stats import ttest_ind
t, p = ttest_ind(rl_rewards, baseline_rewards, equal_var=False)
```

Report: mean ± std, 95% CI, p-value vs each baseline. If p < 0.05, your result is statistically significant.

### Cross-validated event rate study

Run evaluation at 3 event rates: `{0.5, 1.0, 2.0}` events/hour. This shows how your policy scales and matches what papers like Kangaslahti et al. do (they vary observation request density).

---

## 5. The Files You Need to Modify/Add

See the companion Python files:

- `train_improved.py` — Drop-in replacement for `train_ppo_dynamic.py`. Fixes entropy, n_steps, VecNormalize, parallel envs, LR decay.
- `callbacks_improved.py` — `EntropyAnnealCallback`, `BestModelCallback`, `MultiSeedEvalCallback`.
- `proper_evaluation.py` — Full 6-policy comparative study with statistical tests, LaTeX table, and plots.

---

## 6. Mapping to Your Thesis Structure

For your thesis **Section 5 (Experimental Campaign)**:

```
5.1  Experimental Setup
     - Environment parameters (Table: your constants)
     - Training hyperparameters (Table: improved values)
     - Random seeds and reproducibility

5.2  Baseline Descriptions (B1–B5 above)

5.3  Training Analysis
     - Learning curves (reward, CF%, dyn_suc over episodes)
     - Entropy evolution (shows exploration vs. exploitation)
     - Action distribution shift over training

5.4  Results (Table: mean ± std, p-values)
     - Primary metric: total reward per 48h episode
     - Secondary: CF rate, dynamic event capture rate, avg delay
     - Cross-validated over 3 seeds × 30 episodes

5.5  Event Rate Sensitivity
     - Performance at {0.5, 1.0, 2.0} events/hour
     - Shows when RL outperforms heuristics (high event rates)

5.6  Ablation Study
     - Remove: (a) urgency shaping, (b) CNN forecast, (c) SMDP, (d) VecNorm
     - Quantifies contribution of each component
```

---

## 7. Expected Improvement After Fixes

Based on the literature and the identified root causes:

| Metric | Current | Expected After Fix | Reference |
|---|---|---|---|
| Dynamic success rate | 12–16% | 28–40% | Kangaslahti 2024: 35–50% |
| avg100 reward | 12–18 | 22–35 | Estimated from baselines |
| Training stability | High variance | Lower variance | VecNormalize effect |
| Reward vs greedy-dynamic | Unknown | +15–25% | Breitfeld 2025: 22–31% |
| Entropy during training | Stuck at 0.05 | 0.15→0.01 (healthy decay) | Standard PPO practice |

The biggest single gain will come from fixing entropy collapse + parallel environments. The dynamic success rate improvement requires the continuous TTA signal (Keplerian TTA patch) — without it, 30%+ dynamic success is not achievable because the agent has no geometric look-ahead.
