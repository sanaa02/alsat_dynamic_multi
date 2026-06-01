#!/usr/bin/env python3
"""
real_vision_cloud_model.py  --  ALSAT-EO-1  Non-Circular Vision Cloud Model
=============================================================================
Replaces VisionCloudModel / CloudCNNPredictor so that the CNN's input is a
REAL MODIS patch sampled at random from disk, not a synthetic patch derived
from the ground-truth cloud fraction.

Architecture
------------
  truth   <- MODIS JSON (algeria_real_clouds.json)   [ground truth for reward]
  image   <- RealScoutImageProvider                   [random real patch, no CF]
  forecast<- CloudCNN(cloud_cnn_real.pt)(image)       [genuine CNN prediction]

This eliminates the circular dependency where:
    patch = SyntheticCloudPatch.generate(cloud_truth)
and therefore CNN(patch) ~ f(cloud_truth) trivially.

Drop-in replacement for ModisCloudModel
----------------------------------------
    forecast(target_id, sim_time_s) -> (cnn_forecast, ground_truth)

Usage in factory
----------------
    from real_vision_cloud_model import RealVisionCloudModel

    model = RealVisionCloudModel(
        cloud_json_path = "config/cloud_reality/algeria_real_clouds.json",
        patches_dir     = "data/modis_patches",
        cnn_path        = "models/cloud_cnn_real.pt",
    )
    forecast, truth = model.forecast(target_id=0, sim_time_s=86400.0)
"""

from __future__ import annotations

import logging
import math
import os
import sys
import types
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CNN noise fallback standard deviation (same as ModisCloudModel)
CNN_NOISE_STD = 0.05


class RealVisionCloudModel:
    """
    Non-circular cloud prediction model for ALSAT-EO-1.

    Parameters
    ----------
    cloud_json_path : str
        Path to algeria_real_clouds.json (provides ground truth).
    patches_dir : str
        Directory of real MODIS .npy patches (data/modis_patches/).
    cnn_path : str
        Path to the trained CNN weights (models/cloud_cnn_real.pt).
        Also accepts TorchScript (.ts) files — portable, no pickle issues.
    seed : int
        RNG seed (used for noise fallback and patch sampling).
    device : str
        PyTorch device string ("cpu" or "cuda").
    """

    def __init__(
        self,
        cloud_json_path: str,
        patches_dir:     str,
        cnn_path:        str = "models/cloud_cnn_real.pt",
        seed:            int = 42,
        device:          str = None,
    ) -> None:

        # ── 1. Ground-truth data from MODIS JSON ─────────────────────────
        import torch
        import json as _json
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available() and device != "cuda":
            logger.warning("RealVisionCloudModel: no CUDA GPU — running CNN on CPU.")
            # ^^^^ warn ONCE here in __init__, NOT in predict()

        self._device = torch.device(device)
        
        with open(cloud_json_path) as f:
            data = _json.load(f)
        self._lookup:       dict = {}
        self._sorted_dates: dict = {}
        for entry in data:
            tid  = int(entry["target_id"])
            lkp  = {d["date"]: float(d["cloud_fraction"])
                    for d in entry["cloud_data"]}
            self._lookup[tid]       = lkp
            self._sorted_dates[tid] = sorted(lkp.keys())
        logger.info(
            f"RealVisionCloudModel: loaded truth for {len(self._lookup)} targets "
            f"from {cloud_json_path}"
        )

        # ── 2. Real image provider (no label leakage) ─────────────────────
        _core_dir = os.path.dirname(os.path.abspath(__file__))
        if _core_dir not in sys.path:
            sys.path.insert(0, _core_dir)
        from scout_provider import RealScoutImageProvider
        self._provider = RealScoutImageProvider(patches_dir, seed=seed)

        # ── 3. CNN model ──────────────────────────────────────────────────
        self._model  = None
        self._device = device
        self._load_cnn(cnn_path)

        # ── 4. Noise fallback ─────────────────────────────────────────────
        self._noise_std = CNN_NOISE_STD
        self._rng       = np.random.default_rng(seed)

    # ──────────────────────────────────────────────────────────────────────
    # Public API  (matches ModisCloudModel interface exactly)
    # ──────────────────────────────────────────────────────────────────────

    def truth(self, target_id: int, sim_time_s: float) -> float:
        """
        MODIS ground truth — identical to ModisCloudModel.truth().
        Used by the reward function; NOT used as CNN input.
        """
        day_offset = sim_time_s / 86400.0
        dates      = self._sorted_dates[target_id]
        lkp        = self._lookup[target_id]
        EPOCH_IDX  = 3
        lo = max(0, min(int(EPOCH_IDX + math.floor(day_offset)),     len(dates) - 1))
        hi = max(0, min(int(EPOCH_IDX + math.floor(day_offset) + 1), len(dates) - 1))
        alpha = day_offset - math.floor(day_offset)
        return float(lkp[dates[lo]] * (1 - alpha) + lkp[dates[hi]] * alpha)

    def forecast(
        self, target_id: int, sim_time_s: float
    ) -> Tuple[float, float]:
        """
        Returns (cnn_forecast, ground_truth).

        The CNN operates on a randomly sampled real MODIS patch.
        Ground truth is looked up independently from the MODIS JSON.
        The two are *not* correlated by construction.
        """
        truth = self.truth(target_id, sim_time_s)

        if self._model is not None:
            try:
                import torch
                patch = self._provider.get_patch()   # (3, 64, 64) float32
                with torch.no_grad():
                    t       = torch.from_numpy(patch).unsqueeze(0).to(self._device)
                    cnn_out = self._model(t).item()
                forecast = float(np.clip(cnn_out, 0.0, 1.0))
            except Exception as exc:
                logger.warning(
                    f"RealVisionCloudModel CNN inference failed ({exc}); "
                    "using Gaussian noise fallback."
                )
                forecast = self._gaussian_fallback(truth)
        else:
            forecast = self._gaussian_fallback(truth)

        return forecast, truth

    def reset(self, seed: Optional[int] = None) -> None:
        """Called on env.reset() — re-seeds RNG and patch sampler."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._provider.reset(seed)

    @property
    def mode(self) -> str:
        """Returns 'real_cnn' if CNN is loaded, else 'gaussian_noise'."""
        return "real_cnn" if self._model is not None else "gaussian_noise"

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_cnn(self, cnn_path: str) -> None:
        """
        Load CloudCNN weights. Handles all three save formats:

          1. TorchScript (.ts)  — torch.jit.save()
             Portable; no pickle, no class import needed. Preferred for INT8.

          2. Full model (.pt)   — torch.save(model)
             Used by INT8 quantized models. Needs pickle namespace fix.
             INT8 models must stay on CPU (PyTorch quantization is CPU-only).

          3. State-dict (.pt)   — torch.save(model.state_dict())
             Used by standard FP32 training. Moved to self._device (GPU ok).
        """
        import torch

        # ── Step 1: add scripts/models to import path ─────────────────────
        _models_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models"
        ))
        if _models_dir not in sys.path:
            sys.path.insert(0, _models_dir)

        # ── Step 2: file existence check ──────────────────────────────────
        if not os.path.exists(cnn_path):
            logger.warning(
                f"RealVisionCloudModel: CNN weights not found at '{cnn_path}'. "
                "Will use Gaussian noise fallback."
            )
            return

        try:
            # ── Branch A: TorchScript (.ts) ───────────────────────────────
            # No class imports needed; loads on CPU (INT8 TS is CPU-only).
            if str(cnn_path).endswith(".ts"):
                m = torch.jit.load(cnn_path, map_location="cpu")
                m.eval()
                self._model  = m
                self._device = "cpu"
                logger.info(
                    f"RealVisionCloudModel: TorchScript CNN loaded from '{cnn_path}' "
                    "(device=cpu)"
                )
                return

            # ── Branch B: .pt file — need CloudCNN class ──────────────────
            from cloud_cnn import CloudCNN  # noqa: F401

            # Register CloudCNN in every module namespace that torch.save()
            # may have recorded as __module__ at save-time.  This lets
            # pickle find the class regardless of where the .pt was created.
            for _mname in [
                "scripts.training.train_ppo_smdp_full",
                "train_ppo_smdp_full",
                "__main__",
                "scripts.models.cloud_cnn",
                "cloud_cnn",
            ]:
                _mod = sys.modules.get(_mname)
                if _mod is None:
                    _mod = types.ModuleType(_mname)
                    sys.modules[_mname] = _mod
                setattr(_mod, "CloudCNN", CloudCNN)

            # Also patch the live __main__ module directly
            try:
                import __main__ as _main_mod
                setattr(_main_mod, "CloudCNN", CloudCNN)
            except Exception:
                pass

            # ── Load checkpoint ───────────────────────────────────────────
            checkpoint = torch.load(cnn_path, map_location="cpu",
                                    weights_only=False)

            # ── Detect format and build model ─────────────────────────────
            if isinstance(checkpoint, torch.nn.Module):
                # Full model (torch.save(model)) — typical for INT8 quantized.
                # INT8 quantized models CANNOT be moved to CUDA.
                m = checkpoint
                is_quantized = any(
                    "quantized" in type(mod).__name__.lower()
                    for mod in m.modules()
                )
                if is_quantized:
                    self._device = "cpu"
                    logger.info(
                        "RealVisionCloudModel: INT8 quantized model detected — "
                        "inference device forced to cpu."
                    )
                else:
                    m = m.to(self._device)

            elif isinstance(checkpoint, dict):
                # State-dict (torch.save(model.state_dict())) — standard FP32.
                m = CloudCNN().to(self._device)
                m.load_state_dict(checkpoint, strict=True)

            else:
                raise ValueError(
                    f"Unrecognised checkpoint type: {type(checkpoint)}. "
                    "Expected nn.Module or dict (state-dict)."
                )

            m.eval()
            self._model = m

            # Parameter count (may be 0 for fully quantized models — that's ok)
            try:
                n_params = sum(p.numel() for p in m.parameters())
            except Exception:
                n_params = 0

            logger.info(
                f"RealVisionCloudModel: CNN loaded from '{cnn_path}' "
                f"[{n_params:,} params, device={self._device}]"
            )

        except Exception as exc:
            logger.warning(
                f"RealVisionCloudModel: CNN unavailable ({exc}). "
                "Gaussian noise fallback active."
            )
            self._model = None

    def _gaussian_fallback(self, truth: float) -> float:
        """Gaussian noise around truth — same as ModisCloudModel.forecast()."""
        noise = float(self._rng.normal(0.0, self._noise_std))
        return float(np.clip(truth + noise, 0.0, 1.0))
