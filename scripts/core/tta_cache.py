#!/usr/bin/env python3
"""
tta_cache.py  --  Cached Keplerian TTA + Slew Computation
==========================================================
Verified bottleneck: keplerian_tta() runs 150 leapfrog integration steps
per call (×3 numpy ops each).  It is called from:
  1. _build_obs() → _compute_tta() for each active DYN event
  2. action_mask compute_action_mask() → get_slots() → sorting by urgency
  3. DynamicRewardShaper._slot_is_accessible() → _slew_safe()

Per SMDP step: ~3 call sites × 3 active events = 9 keplerian_tta calls.
Each call: ~150 iterations × numpy = ~0.4ms on CPU.
9 × 0.4ms × 144 steps × 700 eps = ~362 seconds (~6 min) in TTA alone.

FIX-TTA-1  Cache TTA results per (event_id, sim_time_bin) where bin = 30s.
           TTA changes by ~orbital_velocity × 30s / TTA_value ≈ tiny.
           Cache hit = one dict lookup vs 150 numpy iterations.

FIX-TTA-2  Cache slew angles per (event_id, attitude_hash).
           Slew angle doesn't change within a sub-step (attitude is constant
           during the slew action).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Cache bins — round sim_time to nearest TTA_BIN_S for cache key
TTA_BIN_S:  float = 30.0    # matches BASE_STEP_S; TTA changes < 0.1% per bin
SLEW_BIN_S: float = 30.0    # same bin for slew angle


class TTACache:
    """
    LRU-like cache for keplerian_tta and slew angle calculations.

    Usage:
        cache = TTACache()
        tta   = cache.get_tta(satellite, event, sim_time, keplerian_tta_fn)
        slew  = cache.get_slew(satellite, event, slew_fn)
    """

    def __init__(self, max_size: int = 512):
        self._tta_cache:  Dict[tuple, float] = {}
        self._slew_cache: Dict[tuple, float] = {}
        self._max_size   = max_size
        self._tta_hits   = 0
        self._tta_misses = 0
        self._slew_hits  = 0
        self._slew_misses = 0

    def get_tta(self, satellite, event, sim_time: float, tta_fn) -> float:
        """Return cached TTA or compute + cache."""
        event_id = id(event)
        t_bin    = round(sim_time / TTA_BIN_S) * TTA_BIN_S
        key      = (event_id, t_bin)

        if key in self._tta_cache:
            self._tta_hits += 1
            return self._tta_cache[key]

        self._tta_misses += 1
        try:
            val = float(tta_fn(satellite, event, sim_time))
        except Exception:
            val = 7200.0   # INACCESSIBLE fallback

        if len(self._tta_cache) >= self._max_size:
            # Simple eviction: remove oldest half
            keys = list(self._tta_cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._tta_cache[k]

        self._tta_cache[key] = val
        return val

    def get_slew(self, satellite, event, slew_fn) -> float:
        """Return cached slew angle or compute + cache."""
        event_id = id(event)
        # Use satellite attitude hash — changes when satellite actually slews
        try:
            c_hat = tuple(np.round(satellite.dynamics.c_hat_P, 3).tolist())
        except Exception:
            c_hat = (0.0,)
        key = (event_id, c_hat)

        if key in self._slew_cache:
            self._slew_hits += 1
            return self._slew_cache[key]

        self._slew_misses += 1
        try:
            val = float(slew_fn(satellite, event))
        except Exception:
            val = math.pi / 2  # 90° default (not accessible)

        if len(self._slew_cache) >= self._max_size:
            keys = list(self._slew_cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._slew_cache[k]

        self._slew_cache[key] = val
        return val

    def clear(self):
        """Clear all caches — call on env.reset()."""
        self._tta_cache.clear()
        self._slew_cache.clear()

    @property
    def stats(self) -> dict:
        return {
            "tta_hits":    self._tta_hits,
            "tta_misses":  self._tta_misses,
            "slew_hits":   self._slew_hits,
            "slew_misses": self._slew_misses,
            "tta_hit_pct": 100 * self._tta_hits  / max(self._tta_hits  + self._tta_misses,  1),
            "slew_hit_pct":100 * self._slew_hits / max(self._slew_hits + self._slew_misses, 1),
        }


# Module-level singleton — shared across all envs in the same process
_GLOBAL_TTA_CACHE = TTACache(max_size=1024)


def get_tta_cached(satellite, event, sim_time: float, tta_fn) -> float:
    """Module-level cached TTA lookup."""
    return _GLOBAL_TTA_CACHE.get_tta(satellite, event, sim_time, tta_fn)


def get_slew_cached(satellite, event, slew_fn) -> float:
    """Module-level cached slew lookup."""
    return _GLOBAL_TTA_CACHE.get_slew(satellite, event, slew_fn)


def clear_tta_cache():
    """Clear on env.reset() to avoid stale values across episodes."""
    _GLOBAL_TTA_CACHE.clear()


def tta_cache_stats() -> dict:
    return _GLOBAL_TTA_CACHE.stats


# ─────────────────────────────────────────────────────────────────────────────
# Patch _compute_tta and _slew_safe in env_alsat_dynamic module
# ─────────────────────────────────────────────────────────────────────────────

def patch_tta_and_slew():
    """
    Replace env_alsat_dynamic._compute_tta and _slew_safe with cached versions.
    Call once at startup (imported by env_alsat_dynamic_patch.py).
    """
    try:
        import env_alsat_dynamic as _mod

        _orig_tta  = _mod._compute_tta
        _orig_slew = _mod._slew_safe

        def _cached_compute_tta(satellite, event, sim_time: float) -> float:
            return get_tta_cached(satellite, event, sim_time, _orig_tta)

        def _cached_slew_safe(satellite, event) -> float:
            return get_slew_cached(satellite, event,
                                   lambda sat, evt: _orig_slew(sat, evt))

        _mod._compute_tta = _cached_compute_tta
        _mod._slew_safe   = _cached_slew_safe

        logger.info("[TTA-CACHE] _compute_tta and _slew_safe patched with caching")
        return True
    except Exception as exc:
        logger.warning(f"[TTA-CACHE] Could not patch TTA/slew: {exc}")
        return False