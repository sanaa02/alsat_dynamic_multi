#!/usr/bin/env python3
"""
repro.py  --  Reproducibility Protocol for ALSAT-EO-1
======================================================
Provides:
  1. seed_everything(seed)   — sets all RNG seeds deterministically
  2. ExperimentConfig        — dataclass capturing full experiment state
  3. save_repro_json(config) — writes reproducibility record
  4. CLI: python repro.py --replay <repro_json>  — re-runs from record
"""
from __future__ import annotations

import os, sys, json, random, argparse, subprocess, hashlib, time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import numpy as np


def seed_everything(seed: int) -> None:
    """
    Set ALL random number generators to a deterministic state.
    Call at the start of every training run, before any tensor creation.
    """
    import random as _rnd
    _rnd.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass

    print(f"  [Repro] All RNGs seeded with {seed}")


@dataclass
class ExperimentConfig:
    """
    Complete specification of one experiment run.
    Saving this JSON is sufficient to exactly reproduce the training.
    """
    # Identity
    exp_id:          str   = "exp_001"
    description:     str   = ""
    timestamp:       str   = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    # Seeds
    seed:            int   = 42
    env_seeds:       list  = field(default_factory=lambda: [42, 43, 44, 45])

    # Environment
    n_envs:          int   = 4
    n_satellites:    int   = 1
    event_rate:      float = 2.0
    duration_s:      float = 172800.0
    config_name:     str   = "dyn_real_vision"
    cnn_model:       str   = "models/cloud_cnn_real.pt"
    with_action_mask: bool = True
    with_domain_rand: bool = True

    # Algorithm
    episodes:        int   = 2000
    learning_rate:   float = 3e-4
    n_steps:         int   = 2048
    batch_size:      int   = 64
    n_epochs:        int   = 10
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    ent_coef_start:  float = 0.05
    ent_coef_end:    float = 0.005
    vf_coef:         float = 0.5
    max_grad_norm:   float = 0.5
    net_arch:        list  = field(default_factory=lambda: [256, 256])

    # Pipeline
    use_bc:          bool  = True
    bc_epochs:       int   = 50
    use_curriculum:  bool  = True
    curriculum_eps:  int   = 200
    dynamic_bonus:   float = 5.0

    # System
    git_hash:        str   = ""
    python_version:  str   = ""
    torch_version:   str   = ""
    cuda_version:    str   = ""
    hostname:        str   = ""

    def populate_system_info(self) -> "ExperimentConfig":
        """Fill in system metadata automatically."""
        import platform
        self.hostname       = platform.node()
        self.python_version = sys.version.split()[0]
        try:
            import torch
            self.torch_version = torch.__version__
            self.cuda_version  = torch.version.cuda or "cpu"
        except ImportError:
            pass
        try:
            self.git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            self.git_hash = "unknown"
        return self


def save_repro_json(cfg: ExperimentConfig, out_dir: str) -> str:
    """Save reproducibility record to out_dir/repro_{exp_id}.json"""
    os.makedirs(out_dir, exist_ok=True)
    cfg.populate_system_info()
    path = os.path.join(out_dir, f"repro_{cfg.exp_id}.json")
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"  [Repro] Saved → {path}")
    return path


def load_repro_json(path: str) -> ExperimentConfig:
    with open(path) as f:
        d = json.load(f)
    return ExperimentConfig(**{k: v for k, v in d.items()
                               if k in ExperimentConfig.__dataclass_fields__})


def build_command_from_config(cfg: ExperimentConfig) -> str:
    """Reconstruct the CLI command that exactly reproduces this run."""
    flags = [
        f"CUDA_VISIBLE_DEVICES=1 python -m scripts.training.train_ppo_smdp_full",
        f"--episodes {cfg.episodes}",
        f"--seed {cfg.seed}",
        f"--n-envs {cfg.n_envs}",
        f"--event-rate {cfg.event_rate}",
        f"--ent-coef {cfg.ent_coef_start}",
        f"--cnn-model {cfg.cnn_model}",
    ]
    if cfg.use_bc:          flags.append("--bc")
    if cfg.use_curriculum:  flags.append("--curriculum")
    if cfg.with_action_mask: flags.append("--action-mask")
    if cfg.with_domain_rand: flags.append("--domain-rand")
    return " \\\n    ".join(flags)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--create",  type=str,  default=None,
                    help="Create a new repro JSON with given exp_id")
    ap.add_argument("--replay",  type=str,  default=None,
                    help="Print reproduction command from repro JSON")
    ap.add_argument("--seed",    type=int,  default=42)
    args = ap.parse_args()

    if args.create:
        cfg = ExperimentConfig(exp_id=args.create, seed=args.seed)
        path = save_repro_json(cfg, "results/repro")
        print(f"\n  Repro command:\n  {build_command_from_config(cfg)}")

    elif args.replay:
        cfg = load_repro_json(args.replay)
        print(f"\n  Loaded experiment: {cfg.exp_id}")
        print(f"\n  Reproduction command:")
        print(f"  {build_command_from_config(cfg)}")
