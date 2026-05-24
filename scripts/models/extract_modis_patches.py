#!/usr/bin/env python3
"""
extract_modis_patches.py  --  ALSAT-EO-1  MOD09GA -> .npy patch extractor
==========================================================================
KEY FIX: state_1km_1 is at 1 km resolution (1200x1200 pixels per tile),
         surface reflectance bands are at 500 m (2400x2400).
         Coordinates must be scaled by 0.5 when indexing the cloud mask.

Saves files as:  cf<CF:0.3f>_<hdf_stem>_<i:04d>.npy
  - patch: float32 (3, 64, 64) in range [0, 1]
  - CF   : cloud fraction from the 1 km state band (float in [0, 1])
"""
import os, sys, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import path_setup  # noqa

import argparse
ap = argparse.ArgumentParser(description="Extract 64x64 patches from MOD09GA HDF files")
ap.add_argument("--hdf-dir",   default="data/modis_patches",   help="Directory with .hdf files")
ap.add_argument("--out-dir",   default="data/modis_patches",   help="Output directory for .npy files")
ap.add_argument("--n-patches", type=int, default=30,            help="Patches per HDF file")
ap.add_argument("--patch-px",  type=int, default=64,            help="Patch size in 500m pixels")
ap.add_argument("--seed",      type=int, default=42)
args = ap.parse_args()

PATCH_PX = args.patch_px
os.makedirs(args.out_dir, exist_ok=True)

# HDF fill value for surface reflectance bands
FILL_VALUES = {-28672, -32768}   # cover different product versions

total_saved = 0
total_skip_nan = 0
total_skip_shape = 0

rng = np.random.default_rng(args.seed)

for hdf_name in sorted(os.listdir(args.hdf_dir)):
    if not hdf_name.endswith(".hdf"):
        continue
    hdf_path = os.path.join(args.hdf_dir, hdf_name)
    stem     = Path(hdf_name).stem
    print(f"\nProcessing {hdf_name} ...")

    try:
        import pyhdf.SD as SD
        sd = SD.SD(hdf_path)
    except Exception as e:
        print(f"  [ERROR] Cannot open HDF: {e}")
        continue

    try:
        # ── Surface reflectance at 500 m ─────────────────────────────────
        def read_band(name):
            arr = sd.select(name)[:].astype(np.float32)
            for fv in FILL_VALUES:
                arr[arr == fv] = 0.0       # replace fill with 0 (no data)
            arr = np.clip(arr * 0.0001, 0.0, 1.0)
            return arr

        r = read_band("sur_refl_b01_1")    # Red   (500 m)
        g = read_band("sur_refl_b04_1")    # Green (500 m)
        b = read_band("sur_refl_b03_1")    # Blue  (500 m)

        H500, W500 = r.shape   # typically 2400 x 2400

        # ── Cloud mask at 1 km ────────────────────────────────────────────
        qa_raw      = sd.select("state_1km_1")[:].astype(np.uint16)
        cloud_state = qa_raw & 0x0003           # bits 0-1: 0=clear,1=cloudy,2=mixed
        cloud_mask  = (cloud_state == 1) | (cloud_state == 2)   # True = cloud
        H1km, W1km  = cloud_mask.shape          # typically 1200 x 1200

        sd.end()

    except Exception as e:
        print(f"  [ERROR] Reading bands: {e}")
        try:
            sd.end()
        except Exception:
            pass
        continue

    # Check for NaN / inf in reflectance arrays (corrupted files)
    for arr, name in [(r,"R"), (g,"G"), (b,"B")]:
        if not np.isfinite(arr).all():
            arr[:] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            print(f"  [WARN] NaN/inf in {name} band — replaced with 0")

    # ── Sampling window: must be valid in BOTH grids ──────────────────────
    # At 500 m: patch covers [y0 : y0+PATCH_PX]
    # At 1 km : same region covers [y0//2 : y0//2 + PATCH_PX//2]
    PATCH_1KM = PATCH_PX // 2          # 32 pixels at 1 km
    y_max_500 = min(H500, H1km * 2) - PATCH_PX   # safe upper bound at 500m
    x_max_500 = min(W500, W1km * 2) - PATCH_PX

    if y_max_500 <= 0 or x_max_500 <= 0:
        print(f"  [SKIP] Tile too small: 500m={H500}x{W500}, 1km={H1km}x{W1km}")
        continue

    saved_this_file = 0
    for i in range(args.n_patches):
        y0 = int(rng.integers(0, y_max_500))
        x0 = int(rng.integers(0, x_max_500))

        # 500 m patch
        patch = np.stack([
            r[y0:y0+PATCH_PX, x0:x0+PATCH_PX],
            g[y0:y0+PATCH_PX, x0:x0+PATCH_PX],
            b[y0:y0+PATCH_PX, x0:x0+PATCH_PX],
        ], axis=0)   # (3, 64, 64)

        if patch.shape != (3, PATCH_PX, PATCH_PX):
            total_skip_shape += 1; continue

        # Validate patch (no NaN, not all-black)
        if not np.isfinite(patch).all():
            patch = np.nan_to_num(patch, nan=0.0)
        if patch.max() < 1e-4:          # all-zero = no-data tile region
            total_skip_nan += 1; continue

        # Cloud fraction from 1 km grid  (scale coordinates by 0.5)
        y0_1km = y0 // 2
        x0_1km = x0 // 2
        mask_p = cloud_mask[y0_1km:y0_1km+PATCH_1KM, x0_1km:x0_1km+PATCH_1KM]
        if mask_p.size == 0:
            total_skip_nan += 1; continue
        cf = float(np.mean(mask_p))     # always in [0, 1] since mask_p is bool

        # Skip if CF is somehow invalid (shouldn't happen but be safe)
        if not (0.0 <= cf <= 1.0):
            total_skip_nan += 1; continue

        fname = f"cf{cf:.3f}_{stem}_{i:04d}.npy"
        np.save(os.path.join(args.out_dir, fname), patch.astype(np.float32))
        saved_this_file += 1
        total_saved     += 1

    print(f"  Saved {saved_this_file}/{args.n_patches} patches")

print(f"\nDone.  Total saved: {total_saved}  "
      f"(skipped: shape={total_skip_shape}, invalid={total_skip_nan})")
print(f"Output: {args.out_dir}")
