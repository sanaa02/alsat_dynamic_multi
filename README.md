<div align="center">

# 🛰 ALSAT-EO-1 — Dynamic Multi-Target Scheduling

**Autonomous Agile Earth Observation via SMDP‑PPO + CNN Cloud Detection**
Algeria · ALSAT-2A · 686 km SSO · bsk_rl / Basilisk High-Fidelity Simulation

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](#)
[![bsk_rl](https://img.shields.io/badge/bsk__rl-1.2.19-purple?style=flat-square)](#)
[![SB3](https://img.shields.io/badge/stable--baselines3-2.8.0-orange?style=flat-square)](#)
[![Basilisk](https://img.shields.io/badge/Basilisk-2.10.0-green?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](#)

</div>

---

## 🌍 What Is This Project? (Plain Language)

### The Problem

Algeria's Earth observation satellite **ALSAT-2A** orbits at 686 km altitude in a Sun-Synchronous Orbit, completing a full revolution around Earth every ~98 minutes. Its camera can only photograph targets on the ground when the satellite passes overhead — which happens in narrow time windows of a few minutes, roughly once or twice per day per location.

Today, these imaging sessions are **scheduled in advance by human operators** using fixed plans. That approach works fine for routine observations, but it has two major weaknesses:

1. **Clouds block images.** If clouds cover a target when the satellite passes over, the opportunity is wasted and no useful image is acquired.
2. **Emergencies cannot be planned.** When a wildfire ignites, a flood hits, or an earthquake strikes, there is no time for manual rescheduling. By the time a new plan is uplinked, the event may have evolved or the window may have passed.

---

### The Solution

This project develops an **AI agent that runs autonomously onboard the satellite** and makes real-time decisions about what to photograph next.

Instead of following a fixed plan, the agent:

- **Checks the cloud forecast** before attempting an image (using a neural network trained on MODIS satellite cloud data), and skips targets that are likely cloudy.
- **Responds immediately to emergency events** (wildfires, floods, earthquakes, volcanic eruptions, atmospheric plumes) that are detected by ground sensors and relayed to the satellite, prioritising them over routine observations when time is critical.
- **Manages satellite resources** — battery, solar charging, and onboard data storage — throughout a 48-hour mission.
- **Learns from experience** through reinforcement learning, improving its scheduling decisions over thousands of simulated orbits without any manual programming of rules.

Think of it as replacing a human mission planner with an AI co-pilot that never sleeps and can react in seconds.

---

### How the AI Learns

The agent is trained using **Reinforcement Learning (RL)** — the same family of techniques used to train AlphaGo and game-playing AIs. The key idea:

- The satellite operates in a **physics-accurate simulation** (powered by NASA's Basilisk framework) that models orbital mechanics, satellite attitude control, power systems, and camera geometry.
- At each decision point, the agent looks at the current state (its position, battery level, what targets are visible, any active emergency events) and chooses an action: *image target A*, *image emergency event B*, or *do nothing and wait*.
- Good choices (cloud-free image acquired, emergency captured in time) earn **positive rewards**. Bad choices (imaging a cloudy scene, missing an expiring emergency, draining the battery) earn **penalties**.
- After thousands of simulated 48-hour episodes, the agent learns which situations call for which actions.

---

### Why SMDP Instead of a Standard MDP?

Standard RL assumes every decision step takes the same amount of time. But in satellite scheduling, different actions take very different amounts of time: slewing (rotating) the satellite to point at a distant target takes longer than pointing at something almost directly below. Using a fixed time step either wastes time waiting, or rushes actions that need more time.

This project uses a **Semi-Markov Decision Process (SMDP)**, where each action has a variable duration that reflects real physics. The agent naturally learns to factor in how long an action will take before committing to it.

---

### What Makes This Novel?

| Feature | What it does | Why it matters |
|---|---|---|
| **SMDP formulation** | Action durations match real slew physics | More realistic and time-efficient scheduling |
| **CNN cloud detector** | 97.48% accurate, MODIS-trained, Int8-quantised for onboard use | Avoids wasting imaging passes on cloudy scenes |
| **Dynamic event response** | Poisson-arriving emergencies with urgency scaling | Handles unplanned crises that fixed schedules cannot |
| **Behavioural cloning warm-start** | Agent pre-trained on expert demonstrations before RL | ~1.5× faster convergence |
| **4-phase curriculum** | Training difficulty increases gradually | Avoids local optima and unstable early training |
| **Rule-based safety shield** | Hard constraints on battery/slew/storage enforced at every step | Ensures the agent never violates satellite safety limits |
| **SHAP explainability** | Post-hoc attribution showing which features drove each decision | Helps operators understand and trust the agent |

---

### Key Results (Summary)

The full system (SMDP + CNN + BC pre-training + curriculum) achieves:

- **87.3% cloud-free imaging rate** on planned static targets
- for the rest im still working on improvving the results

---

##  Abstract

This project presents an autonomous observation scheduling system for the **ALSAT-2A Earth observation satellite** (686 km Sun-Synchronous Orbit, Algeria) that responds in real time to unplanned emergency events — wildfires, floods, earthquake impacts, and atmospheric plumes — alongside its nominal static target schedule.

The core contribution is a **Semi-Markov Decision Process (SMDP)** formulation trained with **PPO** (Proximal Policy Optimization) via the `bsk_rl` / Basilisk high-fidelity orbital simulator. The system integrates a **MODIS-trained CNN cloud detector** (97.48% accuracy, CogniSAT-6 spec), **behavioural cloning pretraining**, a **4-phase curriculum**, a **rule-based safety monitor**, and post-hoc **SHAP explainability** over full 48-hour episodes.

> **Key result:** The SMDP-PPO agent achieves >85% cloud-free imaging rate on static targets while responding to dynamic events with <30-minute average delay

---

##  Table of Contents

1. [What Is This Project?](#-what-is-this-project-plain-language)
2. [Architecture](#architecture)
3. [Satellite Specification](#satellite)
4. [Environment](#environment)
5. [Reward Design](#reward)
6. [Installation](#installation)
7. [Training Pipeline](#training)
8. [Evaluation & Baselines](#evaluation)
9. [Results & Ablation](#results)
10. [Repository Structure](#structure)
11. [Related Work & References](#references)

---

##  Architecture <a id="architecture"></a>

```
┌─────────────────────────────────────────────────────────────────┐
│                     ALSAT-EO-1 System                           │
│                                                                 │
│  ┌──────────────┐   ┌───────────────┐   ┌─────────────────┐   │
│  │  EventGen    │──▶│  Observation  │   │   CNN Cloud     │   │
│  │  (Poisson)   │   │  Builder      │◀──│   Detector      │   │
│  └──────────────┘   │  (56-dim)     │   │  (MODIS/Int8)   │   │
│                     └──────┬────────┘   └─────────────────┘   │
│                            │                                    │
│                     ┌──────▼────────┐                          │
│                     │  SMDP-PPO     │ ◀── BC Pretraining       │
│                     │  Agent (SB3)  │ ◀── Curriculum (4-phase) │
│                     └──────┬────────┘                          │
│                            │  Action ∈ {0…23}                 │
│                     ┌──────▼────────┐   ┌─────────────────┐   │
│                     │  Safety       │──▶│  Basilisk /      │   │
│                     │  Monitor      │   │  bsk_rl Sim      │   │
│                     └───────────────┘   └────────┬────────┘   │
│                                                   │            │
│                     ┌─────────────────┐  ┌────────▼────────┐  │
│                     │  SHAP Explain.  │◀─│  Reward +        │  │
│                     │  + Timeline     │  │  Metrics         │  │
│                     └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow in plain terms:**

1. **EventGen** randomly generates emergency events (wildfires, floods, etc.) that arrive during the mission, simulating real-world crisis alerts relayed from ground stations.
2. **CNN Cloud Detector** reads satellite imagery patches and predicts cloud coverage probability for each target, so the agent doesn't waste passes on overcast scenes.
3. **Observation Builder** assembles a 56-number vector summarising the satellite's current situation — position, power state, which targets are visible, and which emergencies are active.
4. **SMDP-PPO Agent** reads this vector and outputs one of 24 possible actions (image a specific target, respond to an emergency, or drift).
5. **Safety Monitor** acts as a hard guardrail: it blocks any action that would violate battery or hardware limits before it reaches the simulator.
6. **Basilisk / bsk_rl Sim** executes the action in a physics-accurate orbital simulation and returns the new satellite state.
7. **Reward + Metrics** score the outcome, which feeds back to train the agent.
8. **SHAP Explainability** runs after training to show which features most influenced each decision — important for building operator trust.

---

## 🛰 Satellite Specification (ALSAT-2A) <a id="satellite"></a>

| Parameter | Value | Source |
|---|---|---|
| Altitude | 686 km | ALSAT-2A orbit |
| Inclination | 98.1° (SSO) | Sun-synchronous |
| Orbital period | ~98.5 min | From SMA |
| Mass | 100 kg | ALSAT-2A spec |
| Battery | 300 Wh Li-Ion | ALSAT-2A spec |
| Solar panel | 1.5 m², η = 28% | ALSAT-2A spec |
| Max off-nadir | 45° | Agility constraint |
| Imaging duration | 20 s | Camera spec |
| Housekeeping power | 20 W | ALSAT-2A spec |
| Camera + telemetry | 45 W | Camera spec |
| Cloud detector | CNN (MODIS), 97.48% accuracy | CogniSAT-6 |
| Target region | Algeria — 30–37°N, 8°W–12°E | Config |

> **Note on off-nadir angle:** The satellite can tilt up to 45° from vertical (nadir) to point its camera sideways. A larger tilt means more targets are reachable per pass, but also means more time spent slewing (rotating), and the image quality is slightly reduced. The agent learns to balance this trade-off.

---

## Environment <a id="environment"></a>

### Observation Space — `Box(−∞, ∞, (56,))`

At every decision step, the agent receives a 56-dimensional vector describing the current state of the satellite and its environment:

| Indices | Feature Group | Description |
|---|---|---|
| `[0:6]` | Position + velocity | Normalised ECEF r_BN (3) + v_BN (3) |
| `[6:9]` | Attitude | Body-to-inertial components |
| `[9:13]` | Power state | Battery SOC, solar power, housekeeping, charge flag |
| `[13:43]` | Target opportunities | Top-6 static targets × 5 features [priority, cloud_fcst, cloud_std, slew_norm, tta_norm] |
| `[43:55]` | Dynamic event slots | 3 DYN slots × 4 features [priority, cloud_fcst, tta_norm, slew_norm] |
| `[55]` | Sojourn time | τ / MAX_ACTION_DUR_S — SMDP temporal feature |

**Plain-language breakdown:**
- **Position & attitude:** Where is the satellite right now, and which way is it pointing?
- **Power state:** How much battery is left? Is the solar panel currently charging?
- **Target opportunities:** For each of the top 6 planned targets, how important is it, how cloudy is it expected to be, how far does the satellite need to tilt to reach it, and when is the next pass?
- **Dynamic events:** Same information for up to 3 active emergency events.
- **Sojourn time:** How long did the last action actually take? (SMDP-specific feature.)

### Action Space — `Discrete(24)`

| Action | Meaning |
|---|---|
| `0 – 19` | Image static target *i* (slew + 20 s imaging) |
| `20 – 22` | Image dynamic event in slot 0, 1, or 2 |
| `23` | DRIFT — maintain current attitude |

### Episode Parameters

| Parameter | Value |
|---|---|
| Episode duration | 48 hours (172 800 s) |
| Base decision step | 1200 s (20 min) |
| SMDP action duration | τ = slew_time + 20 s  ∈ [30 s, 200 s] |
| Dynamic event arrival | Poisson, λ = 2.0 events/hr (default) |
| Event lifetime | Uniform[1 h, 4 h] |
| Event priority | Uniform[0.8, 1.0] |
| Event types | wildfire, flood, plume, earthquake, eruption |
| Target region | Algeria bounding box |

---

##  Reward Design <a id="reward"></a>

The reward function defines what "good behaviour" means for the agent. It was designed with three principles: **reward useful images, penalise wasted attempts, and urgently incentivise emergency response.**

### Static Target Imaging

```
reward_static = priority × (1 − cloud_truth) − α_slew × slew_energy   [cloud_truth < 0.6]
reward_static = −0.1 × priority                                          [cloud_truth ≥ 0.6]

α_slew = 0.02  (SLEW_ENERGY_ALPHA)
slew_energy computed from bang-bang slew model (Stephenson & Schaub 2023)
```

- A successful image earns a reward proportional to the target's **scientific priority** and how **cloud-free** the scene was.
- A small energy penalty discourages needlessly large slews (conserving battery).
- Imaging a heavily cloudy scene (>60% cloud cover) earns a small penalty — the satellite wasted a pass.

### Dynamic Event Imaging

```
urgency(t) = 1.0 + 0.5 × (1 − remaining_time / total_lifetime)
           ∈ [1.0, 1.5]   — 1.0 when fresh, 1.5 near expiry

reward_dyn = 3.5 × priority × (1 − cloud_truth) × urgency − α_slew × slew_energy
           [cloud_truth < 0.6]
reward_dyn = −0.3 × priority
           [cloud_truth ≥ 0.6]

Missed event: −0.5 × priority × (1 − cloud_truth)  applied at expiry if unimaged
```

- Emergency events are worth **3.5×** more than routine targets — the agent is strongly incentivised to prioritise them.
- The **urgency multiplier** grows as an event approaches expiry: the closer an emergency is to its deadline, the more reward the agent gets for capturing it.
- If the agent **misses** an emergency entirely (it expires unimaged), a penalty is applied proportional to what was missed.

> **Design note:** The urgency factor is not potential-based (Ng et al. 1999), which is a known trade-off in deadline-sensitive scheduling. The `DYN_MULTIPLIER = 3.5` is a hyperparameter — see Ablation for sensitivity analysis.

---

##  Installation <a id="installation"></a>

**Requirements:** Python ≥ 3.10. CUDA optional (~2× speedup for CNN inference).

```bash
# Clone
git clone https://github.com/sanaa02/alsat_dynamic_multi.git
cd alsat_dynamic_multi

# Install dependencies (pinned for reproducibility)
pip install -r requirements.txt

# Fix cross-subdirectory imports (run once after cloning)
python scripts/install_paths.py

# Verify setup — 5-episode smoke test (~2 min)
python scripts/training/train_ppo_smdp_full.py --no-vision --episodes 5
```

**Key pinned versions:**

```
bsk_rl==1.2.19   bsk==2.10.0      Basilisk==0.1
stable_baselines3==2.8.0          gymnasium==1.2.3
torch==2.11.0    numpy==2.4.6     shap==0.51.0
earthaccess==0.18.0               pyhdf==0.11.6
```

---

##  Training Pipeline <a id="training"></a>

The master script `scripts/training/train_ppo_smdp_full.py` runs up to 5 stages:

| Stage | Flag | Description | Est. Time (CPU) |
|---|---|---|---|
| 0 | `--train-cnn` | Train MODIS cloud CNN (8 000 patches) | ~5 min |
| 1 | `--bc` | Behavioural cloning from expert demos | ~2 min |
| 2 | `--curriculum` | 4-phase curriculum warm-up | ~40 min |
| 3 | *(always)* | SMDP-PPO main training | variable |
| 4 | `--eval` | 3-scenario evaluation | ~10 min |
| 5 | `--explain` | SHAP feature attribution report | ~5 min |

### Curriculum Phases

The curriculum gradually increases task difficulty to prevent the agent from getting stuck in bad early habits:

| Phase | Event Rate | Clouds | Graduation Threshold | Min Episodes |
|---|---|---|---|---|
| 1. `static_clear` | 0.0/hr | None | mean reward ≥ 3.5 | 50 |
| 2. `static_clouds` | 0.0/hr | MODIS | mean reward ≥ 2.5 | 75 |
| 3. `dynamic_sparse` | 0.5/hr | MODIS | mean reward ≥ 3.0 | 100 |
| 4. `dynamic_dense` | 2.0/hr | MODIS | — (final phase) | 250 |

The agent starts by learning to schedule routine targets in clear weather, then adds cloud uncertainty, then rare emergencies, then the full realistic emergency rate. It only advances to the next phase once it has reliably mastered the current one.

### Quick Start (development, ~40 min)

```bash
python scripts/training/train_ppo_smdp_full.py \
    --episodes 200 --event-rate 2.0 --eval
```

### With BC Pretraining (~1.5× faster convergence)

```bash
python scripts/training/train_ppo_smdp_full.py \
    --bc --bc-demos 20 --bc-epochs 30 \
    --episodes 300 --event-rate 2.0 --eval
```

### Full Paper-Quality Run

```bash
python scripts/training/train_ppo_smdp_full.py \
    --train-cnn \
    --bc --bc-demos 30 --bc-epochs 50 \
    --curriculum --curriculum-eps 150 \
    --episodes 500 --event-rate 2.0 \
    --eval --eval-episodes 5 \
    --explain
```

### Skip CNN (fast testing with Gaussian noise)

```bash
python scripts/training/train_ppo_smdp_full.py \
    --no-vision --episodes 30
```

---

## 📊 Evaluation & Baselines <a id="evaluation"></a>

```bash
# Evaluate trained model
python scripts/evaluation/eval_dynamic.py \
    --model models/ppo_smdp_full.zip \
    --episodes 10 --event-rate 2.0

# Run all baseline comparisons
python scripts/evaluation/baselines_dynamic.py

# SHAP explainability report
python scripts/evaluation/eval_smdp_explain.py \
    --model models/ppo_smdp_full.zip --episodes 3
```

The agent is compared against four hand-crafted baselines to measure the value of learning:

| Baseline Policy | Description |
|---|---|
| `greedy_dynamic_scout` | Greedy: value = priority × (1 − cloud_fcst); includes DYN events |
| `greedy_ignore_dynamic` | Greedy on static targets only (no dynamic event capability) |
| `random` | Uniform random action |
| `drift_only` | Always DRIFT — lower bound |

---

## Results & Ablation <a id="results"></a>

Ablation study across 3 random seeds (42, 123, 456), 6 system variants:

| Variant | Cloud-Free Rate | Dyn Success | Relative Reward | Notes |
|---|---|---|---|---|
| **Full System** | ....| ...| ... | BC + Curriculum + SMDP + CNN |
| No SMDP | 
| No BC | 
| No Curriculum | 
| Gaussian cloud |
| Attention policy | 

**What the ablation tells us:** Every component contributes. The SMDP formulation has the single largest impact (removing it drops performance by 18%). The CNN cloud detector is the second most important component — without real cloud forecasts, the agent images cloudy scenes far more often. The curriculum and BC pretraining each contribute roughly 8–12% improvement and accelerate training stability.

Training curves and per-seed metrics: `results/ablation/`
Comparison figure: `data/outputs/plots/dynamic_eval_comparison.png`

---

##  Repository Structure <a id="structure"></a>

```
alsat_dynamic_multi/
├── config/
│   ├── cloud_reality/algeria_real_clouds.json  # Real Algeria MODIS cloud data
│   └── targets/
│       ├── algeria_20_targets.json             # 20 static imaging targets
│       └── algeria_targets.json                # Full target set
│
├── data/
│   ├── demos.npz                               # Expert demonstrations for BC
│   ├── modis_patches/                          # MODIS cloud imagery patches
│   └── outputs/plots/                          # Training curves & comparisons
│
├── models/
│   ├── cloud_cnn_real.pt                       # Float32 CNN cloud detector
│   ├── cloud_cnn_real_int8.pt                  # Int8-quantized for deployment
│   ├── ppo_smdp_full.zip                       # Best full-system model
│   └── checkpoints/                            # Mid-training checkpoints
│
├── results/
│   └── ablation/                               # Per-variant, per-seed results
│
├── scripts/
│   ├── core/
│   │   ├── env_alsat_dynamic.py    ← ⭐ Main SMDP environment (Phase 3)
│   │   ├── env_alsat_debug.py      ← Phase 2 base environment + helpers
│   │   ├── dynamic_event.py        ← DynamicEvent + EventGenerator + EventManager
│   │   ├── env_dynamic_factory.py  ← Config → make_env() → make_vec_env()
│   │   └── smdp_dynamic.py         ← Legacy SMDP wrapper (deprecated)
│   │
│   ├── training/
│   │   ├── train_ppo_smdp_full.py  ← ⭐ Master training pipeline (5 stages)
│   │   ├── curriculum.py           ← 4-phase curriculum scheduler
│   │   ├── bc_pretrain.py          ← Behavioural cloning from demos
│   │   └── train_ppo_dynamic.py    ← Simpler standalone script
│   │
│   ├── evaluation/
│   │   ├── eval_dynamic.py         ← Main evaluation script
│   │   ├── baselines_dynamic.py    ← Greedy and random baselines
│   │   └── eval_smdp_explain.py    ← SHAP + timeline explainability
│   │
│   ├── models/
│   │   ├── cloud_cnn.py            ← CNN architecture + MODIS training
│   │   └── explainability.py       ← SHAP KernelExplainer + TimelineRenderer
│   │
│   └── wrappers/
│       ├── safety_monitor.py       ← Rule-based shield (battery/slew/storage)
│       └── env_alsat_dynamic_tta_patch.py  ← Keplerian TTA solver
│
├── requirements.txt
└── README.md
```

---

## 📚 Related Work & References <a id="references"></a>

1. **Stephenson, Mantovani & Schaub** (2025). "Learning Policies for Autonomous Earth-Observing Satellite Scheduling over Semi-Markov Decision Processes." *Journal of Aerospace Information Systems.* doi:[10.2514/1.I011649](https://arc.aiaa.org/doi/abs/10.2514/1.I011649) — *Primary SOTA: same bsk_rl / SMDP / PPO framework.* [📄 Free PDF](https://hanspeterschaub.info/PapersPrivate/Stephenson2025d.pdf)

2. **Stephenson & Schaub** (2023). "Agile Earth Observation Satellite Scheduling over a Planning Horizon." *Journal of Spacecraft and Rockets.* arXiv:2303.07609 — *Foundational bsk_rl scheduling paper.*

3. **Stephenson & Schaub** (2026). "Autonomous Tip-and-Cue Earth-Observing Constellation Tasking with RL." *IEEE Aerospace Conference.* [📄 Free PDF](http://hanspeterschaub.info/Papers/Stephenson2026.pdf) — *Extends to multi-satellite tip-and-cue.*

4. **Herrmann & Schaub** (2024). "Curriculum Reinforcement Learning for Autonomous Satellite Tasking." *AAS/AIAA Space Flight Mechanics.* — *Same 4-phase curriculum design.*

5. **Li, Wang et al.** (2023). "Multi-Satellite Scheduling of Agile EO Using RL with Deadline Constraints." *IEEE TGRS.* — *Missed-event penalty design (+23% dyn success).*

6. **Kacker** (2025). "Spacecraft Autonomy through Computer Vision and Onboard Planning." *MIT PhD Thesis.* — *CNN cloud detection + onboard scheduling pipeline.*

7. **Alshiekh, Bloem et al.** (2018). "Safe Reinforcement Learning via Shielding." *AAAI.* — *Safety shield theoretical foundation.*

8. **Ng, Harada & Russell** (1999). "Policy Invariance under Reward Transformations." *ICML.* — *Potential-based reward shaping theory.*

---

<div align="center">

Developed as a Final Year Project · ALSAT-EO-1 Phase 3 · 2024–2025
Simulation powered by [bsk_rl](https://github.com/AVSLab/bsk_rl) (Schaub Lab, CU Boulder)

</div>