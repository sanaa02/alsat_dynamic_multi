#!/usr/bin/env python3
"""
cloud_cnn.py  —  ALSAT-EO-1  Vision-Based Scout Cloud Detector
===============================================================
Implements the lightweight CNN cloud-fraction estimator described in:
  Proposal §4  "compact CNN cloud detection (CogniSAT-6 / CloudScout)"
  Survey  §4   "vision-based look-ahead module"

Pipeline
--------
1. SyntheticCloudPatch
   Generates a realistic 64×64×3 satellite image patch from a given
   cloud_fraction scalar.  Uses spatially-correlated Gaussian fields
   (simulating mesoscale cloud structure) + CubeSat sensor noise and
   blur.  No real MODIS imagery needed — the CNN learns the relationship
   between spatial cloud patterns and cloud fraction from the synthetic
   distribution.

2. CloudCNN  (PyTorch, ~120 K parameters, <0.5 MB)
   3-conv + 2-FC regression network.  Output: scalar in [0, 1].
   Architecture:
     Conv(3→16,k=3,p=1) → BN → ReLU → MaxPool(2)   → 16×32×32
     Conv(16→32,k=3,p=1)→ BN → ReLU → MaxPool(2)   → 32×16×16
     Conv(32→32,k=3,p=1)→ BN → ReLU → MaxPool(2)   → 32×8×8
     Flatten → Linear(2048→128) → ReLU → Dropout(0.2) → Linear(128→1) → Sigmoid

3. CloudCNNTrainer
   Trains the CNN on n_samples synthetic patches and saves to disk.

4. CloudCNNPredictor
   Loads the trained model; provides the same interface as the Gaussian
   noise model (truth, forecast).

5. VisionCloudModel  (drop-in replacement for ModisCloudModel)
   Wraps an existing MODIS truth source + CloudCNNPredictor.
   Returns (cnn_cloud_forecast, cloud_truth) — identical signature
   to ModisCloudModel.forecast().

Accuracy targets
----------------
  ≥90% classification accuracy (cloud/clear at 0.5 threshold)
  RMSE ≤ 0.08 on fraction regression

Usage
-----
    # Train the CNN (one-time, ~2 min on CPU):
    python scripts/cloud_cnn.py --train --samples 8000

    # Use in training:
    from cloud_cnn import VisionCloudModel
    cloud_model = VisionCloudModel(cloud_json_path, cnn_path="models/cloud_cnn.pt")
"""
from __future__ import annotations
# ---- ALSAT path-setup --------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -------------------------------------------------------------------

import math, json, os, logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ── Try PyTorch import (graceful fallback to analytical noise) ────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found — CloudCNN falls back to analytical noise model.")

CNN_NOISE_STD   = 0.05
PATCH_SIZE      = 64
CNN_LATENCY_S   = 0.5    # inference latency subtracted from decision window


# ============================================================================
#  Synthetic Patch Generator
# ============================================================================

class SyntheticCloudPatch:
    """
    Generates realistic satellite image patches for CNN training.

    Parameters
    ----------
    patch_size : int   — spatial resolution (default 64×64 pixels)
    sigma_low  : float — large-scale cloud structure correlation length
    sigma_high : float — fine texture correlation length
    sensor_noise_std : float — CubeSat sensor readout noise
    """

    def __init__(self,
                 patch_size:       int   = PATCH_SIZE,
                 sigma_low:        float = 8.0,
                 sigma_high:       float = 2.0,
                 sensor_noise_std: float = 0.04):
        self.size        = patch_size
        self.sigma_low   = sigma_low
        self.sigma_high  = sigma_high
        self.noise_std   = sensor_noise_std

    def generate(self, cloud_fraction: float,
                 seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a (3, patch_size, patch_size) float32 image.
        Cloud fraction is encoded into the spatial coverage.
        """
        rng = np.random.default_rng(seed)
        s   = self.size

        # Multi-scale cloud field
        raw_low  = rng.standard_normal((s, s))
        raw_high = rng.standard_normal((s, s))

        # Spatial smoothing (simulates cloud meso/micro structure)
        from scipy.ndimage import gaussian_filter
        smooth_low  = gaussian_filter(raw_low,  sigma=self.sigma_low)
        smooth_high = gaussian_filter(raw_high, sigma=self.sigma_high)

        # Blend: large structure dominates
        combined = 0.7 * smooth_low + 0.3 * smooth_high
        # Normalize to [0,1]
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)

        # Threshold to achieve target cloud fraction
        # cloud_fraction of pixels should be "cloudy" (value > threshold)
        threshold = float(np.quantile(combined, 1.0 - cloud_fraction))
        cloud_mask = (combined >= threshold).astype(np.float32)

        # Build 3-channel image (approximate R, G, NIR channels)
        #   Cloud: bright + slightly blue-white (high in all channels)
        #   Clear: darker, browny-green surface
        surface_r  = 0.20 + 0.10 * rng.standard_normal((s, s)) * 0.05
        surface_g  = 0.22 + 0.10 * rng.standard_normal((s, s)) * 0.05
        surface_nir= 0.35 + 0.15 * rng.standard_normal((s, s)) * 0.05

        cloud_r   = 0.85 + 0.05 * rng.standard_normal((s, s)) * 0.03
        cloud_g   = 0.88 + 0.05 * rng.standard_normal((s, s)) * 0.03
        cloud_nir = 0.75 + 0.05 * rng.standard_normal((s, s)) * 0.03

        ch_r   = np.clip(cloud_mask * cloud_r   + (1 - cloud_mask) * surface_r,   0, 1)
        ch_g   = np.clip(cloud_mask * cloud_g   + (1 - cloud_mask) * surface_g,   0, 1)
        ch_nir = np.clip(cloud_mask * cloud_nir + (1 - cloud_mask) * surface_nir, 0, 1)

        # Add CubeSat sensor noise
        for ch in [ch_r, ch_g, ch_nir]:
            ch += rng.standard_normal((s, s)) * self.noise_std
            np.clip(ch, 0, 1, out=ch)

        # Optional mild blur (point-spread-function)
        from scipy.ndimage import gaussian_filter as gf
        ch_r   = gf(ch_r,   sigma=0.5)
        ch_g   = gf(ch_g,   sigma=0.5)
        ch_nir = gf(ch_nir, sigma=0.5)

        return np.stack([ch_r, ch_g, ch_nir], axis=0).astype(np.float32)

    def compute_actual_fraction(self, patch: np.ndarray) -> float:
        """Estimate cloud fraction from the generated patch (for QA)."""
        bright = patch.mean(axis=0)   # mean across channels
        return float((bright > 0.60).mean())


# ============================================================================
#  PyTorch Dataset
# ============================================================================

if TORCH_AVAILABLE:
    class CloudPatchDataset(Dataset):
        def __init__(self, n_samples: int = 5000, seed: int = 42):
            rng       = np.random.default_rng(seed)
            generator = SyntheticCloudPatch()
            self.patches: list = []
            self.labels:  list = []
            for i in range(n_samples):
                cf    = float(rng.uniform(0.0, 1.0))
                patch = generator.generate(cf, seed=int(rng.integers(0, 2**31)))
                self.patches.append(patch)
                self.labels.append(np.float32(cf))

        def __len__(self) -> int:
            return len(self.patches)

        def __getitem__(self, idx):
            return (torch.from_numpy(self.patches[idx]),
                    torch.tensor([self.labels[idx]]))


# ============================================================================
#  CloudCNN  (PyTorch)
# ============================================================================

if TORCH_AVAILABLE:
    class CloudCNN(nn.Module):
        """
        Lightweight 3-conv cloud fraction regression network.
        ~120 K parameters, <0.5 MB float32.

        Input  : (batch, 3, 64, 64)  — RGB or NIR+G+R patch
        Output : (batch, 1)          — cloud fraction ∈ [0,1]
        """
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                            # → 16×32×32

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                            # → 32×16×16

                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                            # → 32×8×8
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

        def count_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def size_mb(self) -> float:
            return self.count_params() * 4 / 1e6   # float32


# ============================================================================
#  Trainer
# ============================================================================

class CloudCNNTrainer:
    """Train CloudCNN on synthetic data and save to disk."""

    def __init__(self,
                 model_path: str   = "models/cloud_cnn_real.pt",
                 n_samples:  int   = 8000,
                 n_epochs:   int   = 25,
                 batch_size: int   = 64,
                 lr:         float = 1e-3,
                 seed:       int   = 42):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for CloudCNNTrainer.")
        self.model_path = model_path
        self.n_samples  = n_samples
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.seed       = seed

    def train(self) -> dict:
        """Train and save. Returns training history."""
        torch.manual_seed(self.seed)
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)

        print(f"  Generating {self.n_samples} synthetic cloud patches...")
        dataset    = CloudPatchDataset(n_samples=self.n_samples, seed=self.seed)
        val_split  = int(0.15 * len(dataset))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [len(dataset) - val_split, val_split],
            generator=torch.Generator().manual_seed(self.seed))
        train_dl   = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl     = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        model     = CloudCNN()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs)
        criterion = nn.MSELoss()

        print(f"  CloudCNN: {model.count_params():,} params  "
              f"({model.size_mb():.2f} MB)")

        history = {"train_loss": [], "val_rmse": [], "val_acc": []}
        best_val_rmse = float("inf")

        for epoch in range(self.n_epochs):
            model.train()
            total_loss = 0.0
            for patches, labels in train_dl:
                optimizer.zero_grad()
                pred = model(patches)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(patches)
            scheduler.step()
            train_loss = total_loss / len(train_ds)

            # Validation
            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for patches, labels in val_dl:
                    pred = model(patches).squeeze(1)
                    preds_all.append(pred.numpy())
                    labels_all.append(labels.squeeze(1).numpy())
            preds_arr  = np.concatenate(preds_all)
            labels_arr = np.concatenate(labels_all)
            rmse = float(np.sqrt(np.mean((preds_arr - labels_arr) ** 2)))
            # Classification accuracy at 0.5 threshold
            acc  = float(np.mean((preds_arr > 0.5) == (labels_arr > 0.5)))
            history["train_loss"].append(train_loss)
            history["val_rmse"].append(rmse)
            history["val_acc"].append(acc)

            if epoch % 5 == 0 or epoch == self.n_epochs - 1:
                print(f"    Epoch {epoch+1:3d}/{self.n_epochs}  "
                      f"loss={train_loss:.4f}  val_rmse={rmse:.4f}  "
                      f"val_acc={acc:.2%}")
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                torch.save(model.state_dict(), self.model_path)

        print(f"  Best val RMSE: {best_val_rmse:.4f}  "
              f"(target ≤0.08)  Acc: {max(history['val_acc']):.2%}  "
              f"(target ≥90%)")
        print(f"  Model saved → {self.model_path}")
        return history


# ============================================================================
#  Predictor
# ============================================================================

class CloudCNNPredictor:
    """
    Loads a trained CloudCNN and provides inference.
    Falls back to analytical Gaussian noise if model not found or PyTorch absent.
    """

    def __init__(self,
                 model_path:   str   = "models/cloud_cnn_real.pt",
                 noise_std:    float = CNN_NOISE_STD,
                 device:       str   = "cuda"):
        self.noise_std  = noise_std
        self._rng       = np.random.default_rng(42)
        self._model     = None
        self._device    = device
        self._generator = SyntheticCloudPatch()

        if TORCH_AVAILABLE and os.path.exists(model_path):
            try:
                m = CloudCNN()
                m.load_state_dict(torch.load(model_path, map_location=device))
                m.eval()
                self._model = m
                logger.info(f"CloudCNN loaded from {model_path}")
            except Exception as exc:
                logger.warning(f"CloudCNN load failed ({exc}) — using noise fallback.")

    @property
    def mode(self) -> str:
        return "cnn" if self._model is not None else "analytical"

    def predict(self, cloud_truth: float,
                lat_rad: float = 0.0,
                lon_rad: float = 0.0) -> float:
        """
        Returns CNN-predicted cloud fraction.
        If CNN unavailable, returns Gaussian-noise estimate.
        Inference latency: CNN_LATENCY_S = 0.5 s (not simulated here,
        subtracts from decision window in real flight software).
        """
        if self._model is not None:
            seed = int(abs(hash((cloud_truth, lat_rad, lon_rad))) % (2**31))
            patch = self._generator.generate(cloud_truth, seed=seed)
            with torch.no_grad():
                t   = torch.from_numpy(patch).unsqueeze(0)
                out = self._model(t).item()
            return float(np.clip(out, 0.0, 1.0))
        else:
            noise = float(self._rng.normal(0.0, self.noise_std))
            return float(np.clip(cloud_truth + noise, 0.0, 1.0))

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)


# ============================================================================
#  VisionCloudModel — drop-in replacement for ModisCloudModel
# ============================================================================

class VisionCloudModel:
    """
    Wraps an existing MODIS truth source (JSON) and replaces the
    Gaussian noise forecast with CNN inference.

    Drop-in replacement for ModisCloudModel.forecast():
      forecast(target_id, sim_time_s) → (cnn_forecast, ground_truth)

    Parameters
    ----------
    cloud_json_path : path to algeria_real_clouds.json
    cnn_path        : path to trained cloud_cnn.pt
    seed            : RNG seed
    """

    def __init__(self,
                 cloud_json_path: str,
                 cnn_path:        str   = "models/cloud_cnn_real.pt",
                 seed:            int   = 42):
        # Load MODIS truth data (same as ModisCloudModel)
        import json as _json
        with open(cloud_json_path) as f:
            data = _json.load(f)
        self._lookup:       dict = {}
        self._sorted_dates: dict = {}
        for entry in data:
            tid   = int(entry["target_id"])
            lkp   = {d["date"]: float(d["cloud_fraction"])
                     for d in entry["cloud_data"]}
            self._lookup[tid]       = lkp
            self._sorted_dates[tid] = sorted(lkp.keys())

        self._predictor = CloudCNNPredictor(model_path=cnn_path)
        self._rng       = np.random.default_rng(seed)
        logger.info(f"VisionCloudModel: predictor mode = {self._predictor.mode}")

    def truth(self, target_id: int, sim_time_s: float) -> float:
        """MODIS ground truth (unchanged from ModisCloudModel)."""
        import math as _math
        day_offset = sim_time_s / 86400.0
        dates      = self._sorted_dates[target_id]
        lkp        = self._lookup[target_id]
        EPOCH_IDX  = 3
        lo = max(0, min(int(EPOCH_IDX + _math.floor(day_offset)),     len(dates)-1))
        hi = max(0, min(int(EPOCH_IDX + _math.floor(day_offset) + 1), len(dates)-1))
        alpha = day_offset - _math.floor(day_offset)
        return float(lkp[dates[lo]] * (1 - alpha) + lkp[dates[hi]] * alpha)

    def forecast(self, target_id: int,
                 sim_time_s: float) -> Tuple[float, float]:
        """
        Returns (cnn_cloud_forecast, ground_truth).
        Signature identical to ModisCloudModel.forecast().
        """
        truth    = self.truth(target_id, sim_time_s)
        forecast = self._predictor.predict(truth)
        return forecast, truth

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._predictor.reset(seed)

def quantize_cloud_cnn(model_path, patches_dir=None, out_path=None, n_calib=200):
    """INT8 quantization for deployment-ready CNN (CogniSAT-6 style)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")
    import os as _os
    from torch.quantization import quantize_dynamic

    out_path = out_path or model_path.replace(".pt", "_int8.pt")
    model = CloudCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    fp32_kb = _os.path.getsize(model_path) / 1024

    model_int8 = quantize_dynamic(model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)

    if patches_dir and _os.path.isdir(patches_dir):
        cal_p, cal_l = [], []
        for f in sorted(x for x in _os.listdir(patches_dir) if x.endswith(".npy"))[:n_calib]:
            try:
                p = np.load(_os.path.join(patches_dir, f))
                if p.shape == (3, PATCH_SIZE, PATCH_SIZE):
                    cal_p.append(p.astype(np.float32))
                    cal_l.append(float(f.split("cf", 1)[1].split("_")[0]))
            except Exception: pass
        if cal_p:
            t_p = torch.FloatTensor(np.array(cal_p))
            t_l = torch.FloatTensor(cal_l)
            with torch.no_grad():
                p32 = model(t_p).squeeze(1); p8 = model_int8(t_p).squeeze(1)
            print(f"  FP32: RMSE={torch.sqrt(torch.mean((p32-t_l)**2)):.4f}  acc={((p32>0.5)==(t_l>0.5)).float().mean():.2%}")
            print(f"  INT8: RMSE={torch.sqrt(torch.mean((p8 -t_l)**2)):.4f}  acc={((p8 >0.5)==(t_l>0.5)).float().mean():.2%}")

    torch.save(model_int8, out_path)
    int8_kb = _os.path.getsize(out_path) / 1024
    print(f"  {fp32_kb:.0f} KB → {int8_kb:.0f} KB  ({100*(1-int8_kb/fp32_kb):.0f}% smaller)  →  {out_path}")
    return out_path


# ── CLI: train the CNN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap   = argparse.ArgumentParser(description="Cloud CNN trainer / tester")
    ap.add_argument("--train",   action="store_true", help="Train the CNN")
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--epochs",  type=int, default=25)
    ap.add_argument("--model",   default=os.path.join(ROOT, "models/cloud_cnn.pt"))
    ap.add_argument("--test",    action="store_true", help="Test patch generation")
    ap.add_argument("--quantize", action="store_true",
                help="Quantize trained model to INT8 (CogniSAT-6 style)")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data/modis_patches"),
                    help="Directory with .npy patches for calibration")
    ap.add_argument("--n-calib",  type=int, default=200,
                    help="Number of calibration patches for INT8 evaluation")
    args = ap.parse_args()

    if args.test:
        gen = SyntheticCloudPatch()
        print("Synthetic patch generation test:")
        for cf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p   = gen.generate(cf, seed=42)
            est = gen.compute_actual_fraction(p)
            print(f"  target_cf={cf:.2f}  patch_shape={p.shape}  "
                  f"estimated_cf={est:.2f}  min={p.min():.3f}  max={p.max():.3f}")

    if args.train:
        if not TORCH_AVAILABLE:
            print("[ERROR] PyTorch required for training.")
            sys.exit(1)
        print("=" * 60)
        print("  CloudCNN Training")
        print("=" * 60)
        trainer = CloudCNNTrainer(
            model_path=args.model,
            n_samples=args.samples,
            n_epochs=args.epochs,
        )
        history = trainer.train()

    if args.quantize:
        quantize_cloud_cnn(
            model_path=args.model,
            patches_dir=args.data_dir,
            out_path=args.model.replace(".pt", "_int8.pt"),
            n_calib=args.n_calib,
        )
    sys.exit(0)

    if not args.train and not args.test:
        # Quick inference demo
        predictor = CloudCNNPredictor(model_path=args.model)
        print(f"Predictor mode: {predictor.mode}")
        for truth in [0.1, 0.3, 0.6, 0.8]:
            pred = predictor.predict(truth)
            print(f"  truth={truth:.1f}  pred={pred:.3f}  diff={abs(pred-truth):.3f}")
