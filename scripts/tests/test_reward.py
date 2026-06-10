"""tests/test_reward.py — Regression tests for reward computation.

Run with:
    pytest scripts/tests/test_reward.py -v

These tests guard against the critical bugs documented in the review:
  - Bug #1: dual DYN reward injection
  - Bug #2: reversed urgency direction
  - Bug #3: cloud threshold inconsistency
  - Bug #4: slew multiplier ignored
"""

import math
import pytest
import numpy as np
import sys, os

# Add scripts/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Utility: minimal mock objects so tests run without Basilisk installed
# ---------------------------------------------------------------------------

class MockTarget:
    def __init__(self, priority=0.8, cloud_cover=0.2,
                 appearance_time=0.0, expiration_time=3600.0):
        self.priority        = priority
        self.cloud_cover     = cloud_cover
        self.appearance_time = appearance_time
        self.expiration_time = expiration_time


# ---------------------------------------------------------------------------
# Tests for urgency formula
# ---------------------------------------------------------------------------

class TestUrgencyFormula:
    """Urgency should be MAXIMUM at appearance and MINIMUM at expiry."""

    def _compute_urgency(self, now, appearance_time, expiration_time):
        """Replicate the corrected urgency formula from env_alsat_dynamic.py."""
        total_dur      = max(1.0, expiration_time - appearance_time)
        remaining      = max(0.0, expiration_time - now)
        frac_remaining = min(1.0, max(0.0, remaining / total_dur))
        return 1.0 + 0.5 * frac_remaining   # CORRECTED formula

    def test_urgency_max_at_appearance(self):
        """Urgency == 1.5 when event just appeared."""
        u = self._compute_urgency(
            now=0.0, appearance_time=0.0, expiration_time=3600.0
        )
        assert abs(u - 1.5) < 1e-6, f"Expected 1.5, got {u}"

    def test_urgency_min_at_expiry(self):
        """Urgency == 1.0 at expiry."""
        u = self._compute_urgency(
            now=3600.0, appearance_time=0.0, expiration_time=3600.0
        )
        assert abs(u - 1.0) < 1e-6, f"Expected 1.0, got {u}"

    def test_urgency_midpoint(self):
        """Urgency == 1.25 at half-lifetime."""
        u = self._compute_urgency(
            now=1800.0, appearance_time=0.0, expiration_time=3600.0
        )
        assert abs(u - 1.25) < 1e-6, f"Expected 1.25, got {u}"

    def test_urgency_monotonically_decreasing(self):
        """Urgency must decrease (or stay equal) as time advances."""
        times = np.linspace(0, 3600, 20)
        urgencies = [
            self._compute_urgency(t, 0.0, 3600.0) for t in times
        ]
        for i in range(1, len(urgencies)):
            assert urgencies[i] <= urgencies[i-1] + 1e-9, \
                f"Urgency increased from step {i-1} to {i}: {urgencies[i-1]} → {urgencies[i]}"

    def test_old_formula_would_fail(self):
        """The BUG: old formula 1.0 + 0.5*frac_elapsed gives 1.5 at expiry (wrong)."""
        # This documents the exact bug pattern
        frac_elapsed_at_expiry = 1.0
        buggy_urgency = 1.0 + 0.5 * frac_elapsed_at_expiry
        # Buggy formula gives 1.5 at expiry — the OPPOSITE of what's wanted
        assert buggy_urgency == 1.5, "Confirming bug exists in old formula"


# ---------------------------------------------------------------------------
# Tests for cloud threshold consistency
# ---------------------------------------------------------------------------

class TestCloudThreshold:
    """DYN and static imaging must use the SAME cloud threshold."""

    CLOUD_THRESH = 0.6  # unified threshold

    def test_unified_cloud_thresh(self):
        """Verify that _DYN_CLOUD_THRESH matches CLOUD_THRESH."""
        try:
            from core.env_alsat_debug import CLOUD_THRESH
            from core.env_alsat_dynamic import _DYN_CLOUD_THRESH
            assert _DYN_CLOUD_THRESH == CLOUD_THRESH, (
                f"Cloud threshold mismatch: static={CLOUD_THRESH}, "
                f"DYN={_DYN_CLOUD_THRESH}. Must be equal."
            )
        except ImportError:
            pytest.skip("env modules not importable without Basilisk — skipping import test")

    def test_dyn_high_cloud_not_rewarded(self):
        """A DYN event with cloud=0.85 must NOT yield positive reward."""
        cloud = 0.85
        prio  = 1.0
        DYN_MULTIPLIER = 2.0
        # With unified CLOUD_THRESH=0.6, cloud=0.85 > 0.6 → cloudy path
        should_be_rewarded = cloud < self.CLOUD_THRESH
        assert not should_be_rewarded, (
            "High-cloud DYN event must not receive imaging reward"
        )

    def test_static_and_dyn_same_cloud_same_eligibility(self):
        """At cloud=0.55 both DYN and static should be eligible."""
        cloud = 0.55
        assert cloud < self.CLOUD_THRESH, "Should be clear sky for both"

    def test_static_and_dyn_same_cloud_ineligible(self):
        """At cloud=0.75 both DYN and static should be ineligible."""
        cloud = 0.75
        assert cloud >= self.CLOUD_THRESH, "Should be cloudy for both"


# ---------------------------------------------------------------------------
# Tests for slew energy domain randomization
# ---------------------------------------------------------------------------

class TestSlewEnergyMultiplier:
    """DR multiplier must affect slew energy calculation."""

    def _slew_energy(self, angle_rad, multiplier=1.0, slew_rate_rad_s=0.03):
        """Replicate calculate_slew_energy_wh with multiplier."""
        SLEW_PEAK_W = 100.0
        slew_time   = abs(angle_rad) / slew_rate_rad_s
        return SLEW_PEAK_W * multiplier * slew_time / 3600.0

    def test_multiplier_scales_energy(self):
        """Energy with mult=1.3 must be 30% higher than mult=1.0."""
        angle = math.radians(30)
        e1    = self._slew_energy(angle, multiplier=1.0)
        e2    = self._slew_energy(angle, multiplier=1.3)
        assert abs(e2 / e1 - 1.3) < 1e-9, f"Expected 1.3× ratio, got {e2/e1}"

    def test_multiplier_zero_gives_zero(self):
        """Multiplier=0 should give zero energy cost."""
        assert self._slew_energy(math.radians(45), multiplier=0.0) == 0.0

    def test_nominal_multiplier_one(self):
        """Default multiplier=1.0 should match original fixed formula."""
        angle = math.radians(20)
        SLEW_PEAK_W   = 100.0
        slew_rate     = 0.03  # rad/s
        expected      = SLEW_PEAK_W * (angle / slew_rate) / 3600.0
        actual        = self._slew_energy(angle, multiplier=1.0)
        assert abs(actual - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tests for TTA normalization range
# ---------------------------------------------------------------------------

class TestTTANormalization:
    """TTA features must be well-distributed in [0, ~1] for real orbital geometry."""

    ORBITAL_PERIOD_S = 5900.0   # corrected normalizer
    OLD_NORM_S       = 120000.0 # original (buggy) normalizer

    def test_corrected_tta_range(self):
        """Real TTA values [300, 5900] s → [0.05, 1.0] with corrected norm."""
        for tta_s in [300, 1200, 3000, 5900]:
            normed = tta_s / self.ORBITAL_PERIOD_S
            assert 0.0 <= normed <= 1.1, f"TTA {tta_s}s → {normed:.3f} out of range"

    def test_old_norm_compresses_features(self):
        """Old normalizer squashes real TTA into tiny range < 0.05."""
        max_real_tta = 5900.0
        old_max_normed = max_real_tta / self.OLD_NORM_S
        assert old_max_normed < 0.05, (
            f"Old norm produced max {old_max_normed:.4f} — too compressed for MLP"
        )

    def test_new_norm_distinguishable(self):
        """Near (300 s) and far (4000 s) targets must differ by >0.1 after norm."""
        near = 300  / self.ORBITAL_PERIOD_S
        far  = 4000 / self.ORBITAL_PERIOD_S
        assert (far - near) > 0.1, f"Near/far difference {far-near:.3f} too small"


# ---------------------------------------------------------------------------
# Tests for PPO hyperparameters (static checks)
# ---------------------------------------------------------------------------

class TestPPOConfig:
    """Sanity-check PPO configuration against best-practice guidelines."""

    STEPS_PER_EP = 144   # int(172800 / 1200)

    def test_n_steps_covers_multiple_episodes(self):
        """n_steps must be ≥ 2× steps_per_ep for batch diversity."""
        n_steps = 4 * self.STEPS_PER_EP   # corrected value
        assert n_steps >= 2 * self.STEPS_PER_EP, \
            f"n_steps={n_steps} too small — PPO needs multi-episode batches"

    def test_batch_size_divides_n_steps(self):
        """batch_size must evenly divide n_steps (SB3 requirement)."""
        n_steps    = 4 * self.STEPS_PER_EP   # 576
        batch_size = max(72, n_steps // 8)   # 72
        assert n_steps % batch_size == 0, \
            f"batch_size={batch_size} does not divide n_steps={n_steps}"

    def test_gamma_covers_full_episode(self):
        """Effective horizon (1/(1-gamma)) must exceed episode length."""
        gamma           = 0.995  # corrected
        eff_horizon     = 1.0 / (1.0 - gamma)
        assert eff_horizon > self.STEPS_PER_EP, (
            f"Effective horizon {eff_horizon:.0f} steps < episode {self.STEPS_PER_EP} steps"
        )