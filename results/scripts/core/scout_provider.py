#!/usr/bin/env python3
"""
scout_provider.py  --  ALSAT-EO-1  Real MODIS Patch Sampler
============================================================
Provides a randomised stream of real MODIS 64×64 patches
**without exposing the label (cloud fraction)** embedded in
the filename.  This breaks the circular dependency that
existed in CloudCNNPredictor, where the synthetic patch
generator received cloud_truth as input and therefore the
CNN's input was derived from its own target label.

Usage
-----
    from scout_provider import RealScoutImageProvider

    provider = RealScoutImageProvider("data/modis_patches")
    patch = provider.get_patch()          # np.ndarray (3, 64, 64) float32
    patch = provider.get_patch_for(0.4)  # nearest-CF patch (optional)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RealScoutImageProvider:
    """
    Randomly samples a real MODIS patch (3, 64, 64) from a directory of
    .npy files extracted by extract_modis_patches.py.

    Filename convention:  cf{CF:.3f}_{stem}_{i:04d}.npy
    The CF encoded in the filename is *intentionally ignored* when returning
    the image so that no label information leaks into the CNN input pipeline.

    Parameters
    ----------
    patches_dir : str
        Directory containing the .npy patch files.
    seed : int
        RNG seed for reproducible sampling.
    """

    def __init__(self, patches_dir: str, seed: int = 42) -> None:
        patches_dir = os.path.expanduser(patches_dir)
        if not os.path.isdir(patches_dir):
            raise FileNotFoundError(
                f"RealScoutImageProvider: patches directory not found: {patches_dir}"
            )

        self._dir  = patches_dir
        self._rng  = np.random.default_rng(seed)

        # Index all .npy files (sorted for determinism across platforms)
        self._index: List[str] = sorted(
            str(Path(patches_dir) / f)
            for f in os.listdir(patches_dir)
            if f.endswith(".npy")
        )
        if not self._index:
            raise FileNotFoundError(
                f"RealScoutImageProvider: no .npy files found in {patches_dir}"
            )

        # Optional: build a CF-indexed lookup for get_patch_for()
        # Keys: cloud fraction (float), Values: list of file indices
        self._cf_index: dict[float, List[int]] = {}
        for i, path in enumerate(self._index):
            cf = self._parse_cf(Path(path).name)
            if cf is not None:
                key = round(cf, 2)
                self._cf_index.setdefault(key, []).append(i)

        logger.info(
            f"RealScoutImageProvider: {len(self._index)} patches loaded "
            f"from {patches_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_patch(self) -> np.ndarray:
        """
        Return a random (3, 64, 64) float32 patch.
        The cloud fraction of the selected patch is NOT revealed.
        """
        idx = int(self._rng.integers(0, len(self._index)))
        return self._load(self._index[idx])

    def get_patch_for(self, target_cf: float, tol: float = 0.1) -> np.ndarray:
        """
        Return a patch whose embedded CF is within *tol* of *target_cf*.
        Falls back to a purely random patch if none is close enough.

        Note: using this method re-introduces a *soft* correlation between
        the CNN input and ground truth.  Only use it if your ablation study
        explicitly requires it; for the main pipeline use get_patch().
        """
        best_key = None
        best_dist = float("inf")
        for key in self._cf_index:
            d = abs(key - target_cf)
            if d < best_dist:
                best_dist = d
                best_key = key

        if best_key is not None and best_dist <= tol:
            candidates = self._cf_index[best_key]
            idx = candidates[int(self._rng.integers(0, len(candidates)))]
            return self._load(self._index[idx])

        # Fallback: purely random
        logger.debug(
            f"get_patch_for({target_cf:.2f}): no patch within tol={tol}; "
            "returning random patch."
        )
        return self.get_patch()

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the RNG (called by cloud model on env reset)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @property
    def n_patches(self) -> int:
        """Number of patches in the pool."""
        return len(self._index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cf(filename: str) -> Optional[float]:
        """
        Parse the cloud fraction from the filename convention
        ``cf{CF:.3f}_{stem}_{i:04d}.npy``.
        Returns None if the filename does not match the convention.
        """
        try:
            if filename.startswith("cf"):
                return float(filename[2:7])
        except (ValueError, IndexError):
            pass
        return None

    def _load(self, path: str) -> np.ndarray:
        """
        Load a .npy patch and normalise shape to (3, 64, 64) float32.
        Handles edge cases from corrupted or differently-formatted files.
        """
        try:
            patch = np.load(path)
        except Exception as exc:
            logger.warning(f"Failed to load {path}: {exc}; returning zeros.")
            return np.zeros((3, 64, 64), dtype=np.float32)

        # Shape normalisation (defensive — all valid patches are already (3,64,64))
        if patch.ndim == 2:
            # Grayscale → replicate across 3 channels
            patch = np.stack([patch, patch, patch], axis=0)
        elif patch.ndim == 3:
            if patch.shape == (64, 64, 3):
                # HWC → CHW
                patch = patch.transpose(2, 0, 1)
            elif patch.shape[0] not in (1, 3):
                # Unexpected — return zeros
                logger.warning(
                    f"Unexpected patch shape {patch.shape} in {path}; "
                    "returning zeros."
                )
                return np.zeros((3, 64, 64), dtype=np.float32)
        else:
            logger.warning(
                f"Unexpected patch ndim={patch.ndim} in {path}; returning zeros."
            )
            return np.zeros((3, 64, 64), dtype=np.float32)

        patch = patch.astype(np.float32)

        # Sanity checks
        if not np.isfinite(patch).all():
            patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
        if patch.max() < 1e-4:
            logger.debug(f"All-black patch at {path}; still returning it.")

        return patch
