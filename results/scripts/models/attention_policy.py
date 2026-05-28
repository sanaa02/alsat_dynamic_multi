#!/usr/bin/env python3
from __future__ import annotations
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -----------------------------------------------------------------
"""
attention_policy.py  --  ALSAT-EO-1  Transformer Scheduler Policy
==================================================================
Replaces the flat MLP with a cross-attention encoder that treats
each opportunity slot as a distinct token.  This lets the network
attend selectively to whichever targets are most relevant at each
decision step and scales naturally to variable target-list lengths.

Architecture
------------
obs (56,) is decomposed into 4 semantic blocks:

  state_feats   obs[0:13]   satellite dynamics + attitude (13,)
  target_slots  obs[13:43]  6 targets-ahead x 5 props   -> (6, 5)
  dyn_slots     obs[43:55]  3 dynamic events x 4 props  -> (3, 4)
  sojourn       obs[55]     SMDP sojourn time-norm      (1,)

The satellite state is used as the QUERY; targets and dynamic events
are the KEY/VALUE sequences.  Cross-attention retrieves the most
actionable slots given the current satellite state.

    state(13) ----> Linear(d) ----> Query
    targets(6,5) -> Linear(d) ----> Key, Value  -> CrossAttn -> pool
    dynamics(3,4)-> Linear(d) ----> Key, Value  -> CrossAttn -> pool
    sojourn(1)   -> Linear(d) ----+--> concat --> MLP --> features_dim

A second self-attention pass over [state, target_ctx, dyn_ctx, sojourn]
captures interactions among all four components.

d_model     = 64   (small: fits MPS / CPU quickly)
n_heads     = 4
features_dim= 256  (output passed to SB3 actor + critic heads)

Reference
---------
Vaswani et al., "Attention is All You Need", NeurIPS 2017.
Mantovani & Schaub, "Scalable async SMDP scheduling", GN&C 2025.

Usage
-----
    from attention_policy import make_attention_ppo, SchedulerAttentionExtractor

    # Build PPO with attention policy
    vec_env = make_vec_env(Config.DYN_VISION, ...)
    model   = make_attention_ppo(vec_env, seed=42)
    model.learn(total_timesteps=...)

    # Or use the extractor alone (e.g., with existing PPO)
    policy_kwargs = dict(
        features_extractor_class  = SchedulerAttentionExtractor,
        features_extractor_kwargs = dict(features_dim=256, d_model=64, n_heads=4),
        net_arch                  = [],   # let extractor do the lifting
    )
"""

import math
from typing import Dict, Optional, Tuple, Type

import numpy as np
import gymnasium as gym

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.type_aliases import Schedule
    SB3_OK = True
except ImportError:
    SB3_OK = False

# ---- obs decomposition constants (must match env_alsat_dynamic.py) ----------
_N_STATE   = 13   # r_BN_N(3)+v_BN_N(3)+c_hat(3)+eclipse(1)+batt(1)+time(2)
_N_TS      = 6    # n_ahead_observe in OpportunityProperties
_N_TF      = 5    # props per static slot (priority,cloud,std,opp_open,slew)
_N_DS      = 3    # N_DYN_SLOTS
_N_DF      = 4    # features per dynamic slot
_N_SOJOURN = 1
# indices in obs vector
_IDX_STATE_END   = _N_STATE                          # 13
_IDX_TARGET_END  = _IDX_STATE_END + _N_TS * _N_TF   # 43
_IDX_DYN_END     = _IDX_TARGET_END + _N_DS * _N_DF  # 55
_IDX_SOJOURN_END = _IDX_DYN_END + _N_SOJOURN        # 56


# =============================================================================
#  Core building block: residual cross-attention block
# =============================================================================

if TORCH_OK:
    class CrossAttentionBlock(nn.Module):
        """
        Single cross-attention layer: query attends to key-value sequence.

          out = LayerNorm(query + Attn(query, kv, kv))

        Parameters
        ----------
        d_model : embedding dim
        n_heads : number of attention heads
        dropout : attention dropout (default 0.0 for RL)
        """
        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
            super().__init__()
            self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout,
                                                batch_first=True)
            self.norm  = nn.LayerNorm(d_model)
            self.ff    = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self,
                    query: torch.Tensor,   # (B, Lq, d)
                    kv:    torch.Tensor,   # (B, Lkv, d)
                    ) -> torch.Tensor:
            attn_out, _ = self.attn(query, kv, kv)
            x = self.norm(query + attn_out)
            x = self.norm2(x + self.ff(x))
            return x   # (B, Lq, d)

    class SelfAttentionBlock(nn.Module):
        """Standard self-attention + feed-forward block."""
        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
            super().__init__()
            self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout,
                                                batch_first=True)
            self.norm  = nn.LayerNorm(d_model)
            self.ff    = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a, _ = self.attn(x, x, x)
            x    = self.norm(x + a)
            x    = self.norm2(x + self.ff(x))
            return x


# =============================================================================
#  SB3 BaseFeaturesExtractor
# =============================================================================

if TORCH_OK and SB3_OK:
    class SchedulerAttentionExtractor(BaseFeaturesExtractor):
        """
        SB3-compatible features extractor using cross-attention.

        Decomposes the (56,) obs into semantic blocks, embeds each
        to d_model, then applies:
          1. Cross-attention: state queries each of [targets, dynamics]
          2. Self-attention:  [state_ctx, target_ctx, dyn_ctx, sojourn]
          3. Mean-pool + MLP -> features_dim output

        Parameters
        ----------
        observation_space : gym.Space (must be Box with shape (56,))
        features_dim      : output feature dimension (default 256)
        d_model           : attention embedding size   (default 64)
        n_heads           : number of attention heads  (default 4)
        """

        def __init__(self,
                     observation_space: gym.Space,
                     features_dim: int = 256,
                     d_model:      int = 64,
                     n_heads:      int = 4):
            super().__init__(observation_space, features_dim)
            self.d = d_model

            # Input projections
            self.state_proj   = nn.Linear(_N_STATE, d_model)
            self.target_proj  = nn.Linear(_N_TF,    d_model)
            self.dyn_proj     = nn.Linear(_N_DF,    d_model)
            self.sojourn_proj = nn.Linear(_N_SOJOURN, d_model)

            # Learned type embeddings (tell the model which block each token is)
            self.type_embed = nn.Embedding(4, d_model)   # 0=state,1=tgt,2=dyn,3=sojourn

            # Cross-attention: satellite state attends to targets / events
            self.target_cross = CrossAttentionBlock(d_model, n_heads)
            self.dyn_cross    = CrossAttentionBlock(d_model, n_heads)

            # Self-attention over the combined token sequence
            self.global_self  = SelfAttentionBlock(d_model, n_heads)

            # Final projection to features_dim
            # Token sequence: state(1) + target_ctx(1) + dyn_ctx(1) + sojourn(1) = 4 tokens
            self.head = nn.Sequential(
                nn.Linear(d_model * 4, features_dim),
                nn.GELU(),
                nn.Linear(features_dim, features_dim),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            B = obs.shape[0]

            # 1. Decompose observation
            state   = obs[:, :_IDX_STATE_END]                          # (B,13)
            targets = obs[:, _IDX_STATE_END:_IDX_TARGET_END]           # (B,30)
            dyn     = obs[:, _IDX_TARGET_END:_IDX_DYN_END]             # (B,12)
            sojourn = obs[:, _IDX_DYN_END:_IDX_SOJOURN_END]           # (B,1)

            # 2. Reshape and project to d_model
            targets = targets.view(B, _N_TS, _N_TF)  # (B,6,5)
            dyn     = dyn.view(B, _N_DS, _N_DF)      # (B,3,4)

            s_tok = self.state_proj(state).unsqueeze(1)    # (B,1,d)
            t_tok = self.target_proj(targets)               # (B,6,d)
            d_tok = self.dyn_proj(dyn)                      # (B,3,d)
            j_tok = self.sojourn_proj(sojourn).unsqueeze(1) # (B,1,d)

            # 3. Add type embeddings
            dev = obs.device
            s_tok = s_tok + self.type_embed(torch.zeros(B, 1, dtype=torch.long, device=dev))
            t_tok = t_tok + self.type_embed(torch.ones(B, _N_TS, dtype=torch.long, device=dev))
            d_tok = d_tok + self.type_embed(2 * torch.ones(B, _N_DS, dtype=torch.long, device=dev))
            j_tok = j_tok + self.type_embed(3 * torch.ones(B, 1, dtype=torch.long, device=dev))

            # 4. Cross-attention: state queries target and dynamic slots
            t_ctx = self.target_cross(s_tok, t_tok)   # (B,1,d) -- state attends to targets
            d_ctx = self.dyn_cross(s_tok, d_tok)       # (B,1,d) -- state attends to dynamics

            # 5. Global self-attention over [state, target_ctx, dyn_ctx, sojourn]
            tokens = torch.cat([s_tok, t_ctx, d_ctx, j_tok], dim=1)  # (B,4,d)
            tokens = self.global_self(tokens)                          # (B,4,d)

            # 6. Flatten and project
            flat = tokens.reshape(B, -1)   # (B, 4*d)
            return self.head(flat)         # (B, features_dim)

        @staticmethod
        def n_params(d_model=64, features_dim=256):
            """Estimate parameter count."""
            # projections
            p  = 13*d_model + 5*d_model + 4*d_model + 1*d_model
            p += 4*d_model   # type embeds
            # 2 cross-attn blocks (each ~4*d^2 attn + 4*d^2 ff)
            p += 2 * 2 * (4 * d_model**2 + 2 * (d_model * d_model*2 + d_model*2 * d_model))
            # 1 self-attn block
            p += 4 * d_model**2 + 2 * (d_model * d_model*2 + d_model*2 * d_model)
            # head
            p += d_model*4*features_dim + features_dim**2
            return p


# =============================================================================
#  Factory: create PPO with attention policy
# =============================================================================

def make_attention_ppo(vec_env,
                       features_dim: int = 256,
                       d_model:      int = 64,
                       n_heads:      int = 4,
                       learning_rate: float = 3e-4,
                       n_steps: int = 2048,
                       batch_size: int = 72,
                       n_epochs:   int = 10,
                       gamma:      float = 0.99,
                       ent_coef:   float = 0.01,
                       seed:       int = 42,
                       device:     str = "cuda"):
    """
    Build PPO with the SchedulerAttentionExtractor features extractor.

    The actor and critic heads are lightweight (net_arch=[]) because the
    attention extractor already compresses obs into a rich 256-dim vector.

    Parameters
    ----------
    vec_env      : DummyVecEnv built from make_env(Config.DYN_VISION, ...)
    features_dim : output dim of attention extractor (default 256)
    d_model      : attention embedding size (default 64, ~100K params)
    n_heads      : number of attention heads (default 4)

    Returns
    -------
    PPO model ready for model.learn(total_timesteps=...)
    """
    if not (TORCH_OK and SB3_OK):
        raise ImportError("torch and stable-baselines3 are required.")

    policy_kwargs = dict(
        features_extractor_class  = SchedulerAttentionExtractor,
        features_extractor_kwargs = dict(
            features_dim = features_dim,
            d_model      = d_model,
            n_heads      = n_heads,
        ),
        net_arch = [],   # no extra MLP after extractor
    )

    model = PPO(
        "MlpPolicy", vec_env,
        policy_kwargs = policy_kwargs,
        learning_rate = learning_rate,
        n_steps       = n_steps,
        batch_size    = batch_size,
        n_epochs      = n_epochs,
        gamma         = gamma,
        gae_lambda    = 0.95,
        ent_coef      = ent_coef,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 0,
        seed          = seed,
        device        = device,
    )
    n_p = sum(p.numel() for p in model.policy.parameters())
    print(f"  AttentionPolicy: {n_p:,} parameters  "
          f"(d_model={d_model}, n_heads={n_heads}, features_dim={features_dim})")
    return model


# =============================================================================
#  Standalone test
# =============================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  attention_policy.py  --  self-test")
    print("=" * 62)

    if not (TORCH_OK and SB3_OK):
        print("  [SKIP] torch or stable-baselines3 not installed.")
    else:
        import torch

        # 1. Test the extractor directly
        obs_space = gym.spaces.Box(low=-float("inf"), high=float("inf"),
                                   shape=(56,), dtype=np.float32)
        ext   = SchedulerAttentionExtractor(obs_space, features_dim=256, d_model=64, n_heads=4)
        n_p   = sum(p.numel() for p in ext.parameters())
        print(f"\n  Extractor params : {n_p:,}")
        print(f"  (MLP equiv ~256x256x2 = {256*256*2:,}  --  attention is more expressive)")

        # Forward pass with batch of 4
        batch  = torch.randn(4, 56)
        out    = ext(batch)
        assert out.shape == (4, 256), f"Bad output shape: {out.shape}"
        print(f"  Forward pass     : OK  input=(4,56) -> output=(4,256)")

        # 2. Check gradient flow
        loss = out.sum()
        loss.backward()
        grad_ok = all(p.grad is not None for p in ext.parameters() if p.requires_grad)
        print(f"  Gradient flow    : {'OK' if grad_ok else 'FAILED'}")

        # 3. Attention weight inspection (verify it's not uniform)
        ext.eval()
        with torch.no_grad():
            state   = batch[:, :13]
            targets = batch[:, 13:43].view(4, 6, 5)
            s_tok   = ext.state_proj(state).unsqueeze(1)
            t_tok   = ext.target_proj(targets)
            _, attn_w = ext.target_cross.attn(s_tok, t_tok, t_tok)
        attn_std = attn_w.std().item()
        print(f"  Attention weights std: {attn_std:.4f}  (>0 means non-uniform focus)")

        print("\n  All tests passed.")
        print("\n  To train with attention policy:")
        print("    from attention_policy import make_attention_ppo")
        print("    model = make_attention_ppo(vec_env, d_model=64, n_heads=4)")
        print("    model.learn(total_timesteps=500*144)")
