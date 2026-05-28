#!/usr/bin/env python3
from __future__ import annotations
# ---- ALSAT path-setup -------------------------------------------
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import path_setup  # noqa
# -----------------------------------------------------------------
"""
modis_real_data.py  --  ALSAT-EO-1  Real MODIS Imagery Integration  (v2)
=========================================================================
Fixed in v2:
  [FIX-1] _load_real_patches: silent exceptions replaced with explicit
           logging; NaN/inf patches handled with nan_to_num; all-black
           patches (no-data regions) are rejected.
  [FIX-2] CosineAnnealingLR.step() moved OUTSIDE the batch loop
           (was incorrectly called per batch, not per epoch).
  [FIX-3] Per-batch NaN guard: skip any batch where loss/pred is not
           finite instead of propagating NaN weights forward.
  [FIX-4] training lr reduced 3e-4 -> 1e-4 for stability with real data.
  [FIX-5] __getitem__ applies nan_to_num as a safety net so the
           DataLoader never returns a NaN tensor.

Usage (same as before)
------
    # Retrain on your extracted .npy patches:
    python scripts/models/modis_real_data.py --train \\
        --data-dir data/modis_patches \\
        --model-out models/cloud_cnn_real.pt

    # Quick demo (no internet, uses real cloud fractions):
    python scripts/models/modis_real_data.py --demo
"""


import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import earthaccess
    EARTHACCESS_OK = True
except ImportError:
    EARTHACCESS_OK = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import scipy.ndimage
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

AOI_LAT_MIN, AOI_LAT_MAX = 30.0, 37.0
AOI_LON_MIN, AOI_LON_MAX = -8.0, 12.0
PATCH_PX = 64


# =============================================================================
#  Enhanced synthetic patches (works with no download)
# =============================================================================

class EnhancedSyntheticPatchGenerator:
    """
    Generates synthetic cloud patches using REAL cloud fractions
    from algeria_real_clouds.json.
    """
    def __init__(self, cloud_json_path: str, patch_px: int = PATCH_PX,
                 noise_std: float = 0.04, seed: int = 42):
        self.patch_px  = patch_px
        self.noise_std = noise_std
        self._rng      = np.random.default_rng(seed)

        with open(cloud_json_path) as f:
            raw = json.load(f)
        self._cf_data: Dict[int, List[Tuple[str, float]]] = {}
        for entry in raw:
            tid  = int(entry["target_id"])
            data = sorted((d["date"], float(d["cloud_fraction"]))
                          for d in entry["cloud_data"])
            self._cf_data[tid] = data
        logger.info(f"EnhancedSyntheticPatchGenerator: {len(self._cf_data)} targets")

    def get_real_cf(self, target_id: int, date_str: str) -> float:
        if target_id not in self._cf_data:
            return 0.5
        data = self._cf_data[target_id]
        for d, cf in data:
            if d >= date_str:
                return cf
        return data[-1][1] if data else 0.5

    def generate(self, cloud_fraction: float, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(
            seed if seed is not None else self._rng.integers(0, 2**31))
        s   = self.patch_px
        raw_c = rng.standard_normal((s, s))
        raw_f = rng.standard_normal((s, s))
        if SCIPY_OK:
            from scipy.ndimage import gaussian_filter
            coarse = gaussian_filter(raw_c, sigma=8.0)
            fine   = gaussian_filter(raw_f, sigma=2.0)
        else:
            coarse, fine = raw_c, raw_f
        combined = 0.65 * coarse + 0.35 * fine
        rng_min, rng_max = combined.min(), combined.max()
        combined = (combined - rng_min) / (rng_max - rng_min + 1e-9)
        thr  = float(np.quantile(combined, 1.0 - cloud_fraction))
        mask = (combined >= thr).astype(np.float32)
        cloud_rgb   = np.array([0.86, 0.90, 0.82]) + rng.standard_normal((3,)) * 0.02
        surface_rgb = np.array([0.18, 0.21, 0.14]) + rng.standard_normal((3,)) * 0.03
        channels = []
        for ci in range(3):
            ch = mask * cloud_rgb[ci] + (1 - mask) * surface_rgb[ci]
            ch += rng.standard_normal((s, s)) * self.noise_std
            channels.append(np.clip(ch, 0, 1).astype(np.float32))
        return np.stack(channels, axis=0)


# =============================================================================
#  Dataset  [FIX-1] robust patch loading with logging
# =============================================================================

if TORCH_OK:
    class MODISCloudPatchDataset(Dataset):
        def __init__(self,
                     cloud_json_path: str,
                     n_samples:       int  = 8000,
                     real_patch_dir:  Optional[str] = None,
                     real_mix_ratio:  float = 0.5,   # fraction from real data
                     seed:            int  = 42):
            self.n_samples   = n_samples
            self.real_ratio  = real_mix_ratio
            self._gen        = EnhancedSyntheticPatchGenerator(cloud_json_path, seed=seed)
            self._rng        = np.random.default_rng(seed)
            self._real: List[Tuple[np.ndarray, float]] = []

            if real_patch_dir and os.path.isdir(real_patch_dir):
                self._load_real_patches(real_patch_dir)
            else:
                if real_patch_dir:
                    print(f"  [WARN] real_patch_dir '{real_patch_dir}' not found — "
                          "using enhanced synthetic data only.")

            n_real = len(self._real)
            if self.real_ratio >= 1.0:
                mode = "real-only"
            elif n_real > 0:
                mode = "real+synthetic"
            else:
                mode = "enhanced-synthetic-only"
                
            print(f"  Dataset: {n_samples} total  ({n_real} real  mode={mode})")

        def _load_real_patches(self, directory: str) -> None:
            loaded = 0; skip_shape = 0; skip_nan = 0; skip_parse = 0; skip_err = 0

            for fname in sorted(os.listdir(directory)):
                if not fname.endswith(".npy"):
                    continue

                # ── Load array ────────────────────────────────────────────
                fpath = os.path.join(directory, fname)
                try:
                    patch = np.load(fpath)
                except Exception as e:
                    logger.warning(f"Cannot load {fname}: {e}")
                    skip_err += 1; continue

                # ── Parse CF from filename ────────────────────────────────
                # Expected: cf<FLOAT>_<anything>.npy  e.g. cf0.350_MOD...npy
                try:
                    after_cf = fname.split("cf", 1)[1]      # "0.350_MOD..."
                    cf_str   = after_cf.split("_")[0]        # "0.350"
                    cf       = float(cf_str)
                    if not (0.0 <= cf <= 1.0):
                        raise ValueError(f"CF out of range: {cf}")
                except Exception as e:
                    logger.debug(f"Cannot parse CF from '{fname}': {e}")
                    skip_parse += 1; continue

                # ── Shape check ───────────────────────────────────────────
                if patch.shape != (3, PATCH_PX, PATCH_PX):
                    logger.debug(f"Wrong shape {patch.shape} in {fname}")
                    skip_shape += 1; continue

                # ── [FIX-1] NaN / inf handling ────────────────────────────
                patch = patch.astype(np.float32)
                if not np.isfinite(patch).all():
                    patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

                # Reject all-black patches (no-data fill regions)
                if patch.max() < 1e-4:
                    skip_nan += 1; continue

                # Ensure range [0, 1]
                patch = np.clip(patch, 0.0, 1.0)

                self._real.append((patch, cf))
                loaded += 1

            print(f"  Real patches: {loaded} loaded  "
                  f"(skipped: parse={skip_parse}, shape={skip_shape}, "
                  f"nan/black={skip_nan}, err={skip_err})")
            if loaded == 0:
                print("  [WARN] Zero real patches loaded!  Check that files "
                      "follow the naming convention:  cf<FLOAT>_<name>.npy  "
                      "and have shape (3, 64, 64).")

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            # Mix real and synthetic
            if self._real and self._rng.random() < self.real_ratio:
                i = int(self._rng.integers(len(self._real)))
                patch, cf = self._real[i]
            else:
                cf    = float(self._rng.uniform(0.0, 1.0))
                seed  = int(self._rng.integers(0, 2**31))
                patch = self._gen.generate(cf, seed=seed)

            # [FIX-5] Safety net: ensure no NaN reaches the model
            patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
            patch = np.clip(patch, 0.0, 1.0)

            return (torch.from_numpy(patch.copy()),
                    torch.tensor([cf], dtype=torch.float32))


# =============================================================================
#  Training  [FIX-2,3,4]
# =============================================================================

def train_cloud_cnn_real(
    cloud_json_path: str,
    model_out:       str   = "models/cloud_cnn_real.pt",
    n_samples:       int   = 10000,
    n_epochs:        int   = 30,
    batch_size:      int   = 64,
    lr:              float = 1e-4,      # [FIX-4] lowered for stability with real data
    real_patch_dir:  Optional[str] = None,
    real_mix_ratio:  float = 0.5,
    seed:            int   = 42,
    device: str = "auto",
) -> dict:
    if  device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        print(f"  CNN training device: {dev}")

    if not TORCH_OK:
        raise ImportError("PyTorch required.")

    from cloud_cnn import CloudCNN

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)

    print(f"  Building MODIS-calibrated dataset ({n_samples} samples)...")
    dataset = MODISCloudPatchDataset(
        cloud_json_path, n_samples=n_samples,
        real_patch_dir=real_patch_dir, real_mix_ratio=real_mix_ratio, seed=seed)

    val_n     = max(1, int(0.15 * n_samples))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_samples - val_n, val_n],
        generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = CloudCNN().to(dev) 
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # [FIX-2] scheduler updated ONCE per epoch, not per batch
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)
    crit  = nn.MSELoss()

    print(f"  Training CloudCNN  ({n_epochs} epochs, lr={lr})...")
    history   = {"train_loss": [], "val_rmse": [], "val_acc": []}
    best_rmse = float("inf")
    nan_batches = 0

    for ep in range(n_epochs):
        model.train()
        t_loss = 0.0
        t_count = 0

        for patches, labels in train_dl:
            patches = patches.to(dev, non_blocking=True)   # ADD
            labels  = labels.to(dev,  non_blocking=True)

            # [FIX-3] Pre-check for NaN in inputs
            if not (torch.isfinite(patches).all() and torch.isfinite(labels).all()):
                nan_batches += 1; continue

            opt.zero_grad()
            pred = model(patches)
            loss = crit(pred, labels)

            # [FIX-3] Post-check for NaN loss
            if not torch.isfinite(loss):
                nan_batches += 1; continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            t_loss  += loss.item() * len(patches)
            t_count += len(patches)

        # [FIX-2] scheduler step AFTER epoch, not inside batch loop
        sched.step()

        # Validation
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for patches, labels in val_dl:
                patches = patches.to(dev, non_blocking=True)   # ADD
                labels  = labels.to(dev,  non_blocking=True)   # ADD
                if not torch.isfinite(patches).all(): continue
                out = model(patches).squeeze(1)
                preds_all.append(out.cpu().numpy())            # ADD .cpu()
                labels_all.append(labels.squeeze(1).cpu().numpy()) 

        if not preds_all:
            print(f"    ep {ep+1:3d}  [WARN] all val batches had NaN — check data")
            continue

        preds  = np.concatenate(preds_all)
        labels = np.concatenate(labels_all)
        rmse   = float(np.sqrt(np.mean((preds - labels) ** 2)))
        acc    = float(np.mean((preds > 0.5) == (labels > 0.5)))
        mean_l = t_loss / max(t_count, 1)
        history["train_loss"].append(mean_l)
        history["val_rmse"].append(rmse)
        history["val_acc"].append(acc)

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), model_out)

        print(f"    ep {ep+1:3d}/{n_epochs}  "
              f"loss={mean_l:.4f}  rmse={rmse:.4f}  acc={acc:.2%}  "
              f"lr={sched.get_last_lr()[0]:.2e}")

    if nan_batches > 0:
        print(f"  [INFO] Skipped {nan_batches} NaN batches during training.")
    print(f"  Best val RMSE: {best_rmse:.4f}  Model -> {model_out}")
    return history


# =============================================================================
#  earthaccess downloader (unchanged)
# =============================================================================

class MODISEarthAccessDownloader:
    PRODUCT = "MOD09GA"; VERSION = "061"

    def __init__(self, username=None, password=None,
                 data_dir="data/modis_raw", verbose=True):
        if not EARTHACCESS_OK:
            raise ImportError("pip install earthaccess")
        self.data_dir = data_dir; self.verbose = verbose
        os.makedirs(data_dir, exist_ok=True)
        if username and password:
            os.environ["EARTHDATA_USERNAME"] = username
            os.environ["EARTHDATA_PASSWORD"] = password
            earthaccess.login(strategy="environment", persist=False)
        else:
            earthaccess.login(strategy="netrc")
        if verbose: print("  NASA Earthdata authentication OK.")

    def download(self, start_date="2024-03-20", end_date="2024-03-22"):
        results = earthaccess.search_data(
            short_name=self.PRODUCT, version=self.VERSION,
            temporal=(start_date, end_date),
            bounding_box=(AOI_LON_MIN, AOI_LAT_MIN, AOI_LON_MAX, AOI_LAT_MAX))
        if self.verbose:
            print(f"  Found {len(results)} granules ({start_date} to {end_date})")
        if not results: return []
        files = earthaccess.download(results, local_path=self.data_dir)
        if self.verbose: print(f"  Downloaded {len(files)} files -> {self.data_dir}/")
        return files


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import path_setup
    ROOT = path_setup.root_path()

    ap = argparse.ArgumentParser(description="MODIS real data integration (v2)")
    ap.add_argument("--download",   action="store_true")
    ap.add_argument("--username",   default="")
    ap.add_argument("--password",   default="")
    ap.add_argument("--start-date", default="2024-03-20")
    ap.add_argument("--end-date",   default="2024-03-22")
    ap.add_argument("--train",      action="store_true")
    ap.add_argument("--data-dir",   default=os.path.join(ROOT,"data/modis_patches"))
    ap.add_argument("--model-out",  default=os.path.join(ROOT,"models/cloud_cnn_real.pt"))
    ap.add_argument("--n-samples",  type=int,   default=10000)
    ap.add_argument("--n-epochs",   type=int,   default=30)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--cloud",      default=os.path.join(ROOT,"config/cloud_reality/algeria_real_clouds.json"))
    ap.add_argument("--demo",       action="store_true")
    ap.add_argument("--inspect",    action="store_true",
                    help="Inspect .npy files in --data-dir and report stats")
    ap.add_argument("--real-only", action="store_true",
                help="Use only real patches, no synthetic mixing")
    args = ap.parse_args()

    if args.inspect:
        # ── Diagnose what's in the folder ──────────────────────────────────
        print(f"\nInspecting: {args.data_dir}")
        npy_files = [f for f in os.listdir(args.data_dir) if f.endswith(".npy")]
        print(f"  Total .npy files: {len(npy_files)}")
        ok = bad_shape = bad_nan = bad_parse = 0
        for fname in npy_files[:200]:   # sample first 200
            fpath = os.path.join(args.data_dir, fname)
            try:
                p = np.load(fpath)
                if p.shape != (3, 64, 64):
                    bad_shape += 1; continue
                try:
                    cf_str = fname.split("cf", 1)[1].split("_")[0]
                    cf = float(cf_str)
                    if not (0 <= cf <= 1):
                        bad_parse += 1; continue
                except Exception:
                    bad_parse += 1; continue
                if not np.isfinite(p).all() or p.max() < 1e-4:
                    bad_nan += 1; continue
                ok += 1
            except Exception as e:
                bad_shape += 1
        print(f"  Valid : {ok}   bad_shape: {bad_shape}   "
              f"nan/black: {bad_nan}   bad_parse: {bad_parse}")
        if npy_files:
            sample = np.load(os.path.join(args.data_dir, npy_files[0]))
            print(f"  Sample file: {npy_files[0]}")
            print(f"    shape={sample.shape}  dtype={sample.dtype}  "
                  f"min={sample.min():.4f}  max={sample.max():.4f}  "
                  f"has_nan={not np.isfinite(sample).all()}")

    if args.download:
        if not EARTHACCESS_OK:
            print("[ERROR] pip install earthaccess")
        else:
            dl = MODISEarthAccessDownloader(
                username=args.username or None,
                password=args.password or None,
                data_dir=args.data_dir)
            dl.download(args.start_date, args.end_date)

    if args.train:
        history = train_cloud_cnn_real(
            cloud_json_path=args.cloud,
            model_out=args.model_out,
            n_samples=args.n_samples,
            n_epochs=args.n_epochs,
            lr=args.lr,
            real_patch_dir=args.data_dir if os.path.isdir(args.data_dir) else None,
            real_mix_ratio=1.0 if args.real_only else 0.5,
        )
        if history["val_rmse"]:
            print(f"  Final val RMSE: {history['val_rmse'][-1]:.4f}  "
                  f"Acc: {history['val_acc'][-1]:.2%}")

    if args.demo:
        gen = EnhancedSyntheticPatchGenerator(args.cloud, seed=42)
        print("Enhanced Synthetic Patch Demo:")
        print(f"  Target 0, 2024-03-20: CF={gen.get_real_cf(0, '2024-03-20'):.3f}")
        for cf in [0.1, 0.4, 0.7, 0.9]:
            p = gen.generate(cf, seed=int(cf*100))
            print(f"  cf={cf:.1f}  shape={p.shape}  "
                  f"mean={p.mean():.3f}  max={p.max():.3f}  has_nan={not np.isfinite(p).all()}")

    if not any([args.download, args.train, args.demo, args.inspect]):
        print("Use --demo, --train, --download, or --inspect")
