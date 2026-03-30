"""
Tests for CREDIT IC-perturbation ensemble classes.

Covers:
  - GaussianNoise
  - ColorNoise
  - SphericalNoise
  - TemporalNoise
  - crps_spatial_avg (credit.verification.ensemble)

These tests run on CPU and do not require a GPU or any data files.
"""

import numpy as np
import pytest
import torch

from credit.ensemble.color import ColorNoise
from credit.ensemble.gaussian import GaussianNoise
from credit.ensemble.spherical import SphericalNoise
from credit.ensemble.temporal import TemporalNoise
from credit.verification.ensemble import crps_spatial_avg

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Representative state tensor: (batch=1, channels=4, time=1, lat=16, lon=32)
SHAPE = (1, 4, 1, 16, 32)


@pytest.fixture
def x():
    return torch.zeros(*SHAPE)


@pytest.fixture
def w_lat():
    lats = np.linspace(-90, 90, 16)
    w = np.cos(np.deg2rad(lats))
    return w / w.mean()


# ===========================================================================
# GaussianNoise
# ===========================================================================


class TestGaussianNoise:
    def test_output_shape(self, x):
        noise = GaussianNoise(amplitude=0.1)
        out = noise(x)
        assert out.shape == x.shape

    def test_zero_amplitude(self, x):
        noise = GaussianNoise(amplitude=0.0)
        out = noise(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_amplitude_scales_std(self, x):
        """Std of output should be ~ amplitude for large tensors."""
        big = torch.zeros(1, 1, 1, 256, 512)
        for amp in (0.05, 0.2):
            noise = GaussianNoise(amplitude=amp)
            out = noise(big)
            assert abs(out.std().item() - amp) < 0.01, (
                f"amplitude={amp}: expected std≈{amp}, got {out.std().item():.4f}"
            )

    def test_different_members_differ(self, x):
        noise = GaussianNoise(amplitude=0.1)
        a, b = noise(x), noise(x)
        assert not torch.allclose(a, b), "Two calls should produce different noise"

    def test_device_preserved(self, x):
        noise = GaussianNoise(amplitude=0.1)
        out = noise(x)
        assert out.device == x.device

    def test_repr(self):
        assert "GaussianNoise" in repr(GaussianNoise(amplitude=0.15))


# ===========================================================================
# ColorNoise
# ===========================================================================


class TestColorNoise:
    def test_output_shape(self, x):
        noise = ColorNoise(amplitude=0.1, reddening=2)
        out = noise(x)
        assert out.shape == x.shape

    def test_zero_amplitude(self, x):
        noise = ColorNoise(amplitude=0.0, reddening=2)
        out = noise(x)
        assert torch.allclose(out, torch.zeros_like(out))

    @pytest.mark.parametrize("reddening", [0, 1, 2, 3])
    def test_reddening_values(self, x, reddening):
        """All reddening values should produce finite output."""
        noise = ColorNoise(amplitude=0.1, reddening=reddening)
        out = noise(x)
        assert torch.isfinite(out).all(), f"reddening={reddening} produced non-finite output"

    def test_higher_reddening_smoother(self):
        """Higher reddening → lower high-frequency power."""
        torch.manual_seed(0)
        shape = (1, 1, 1, 64, 128)
        x = torch.zeros(*shape)

        white = ColorNoise(amplitude=1.0, reddening=0)(x)
        red = ColorNoise(amplitude=1.0, reddening=2)(x)

        # High-frequency variance: difference between adjacent points
        hf_white = (white[..., 1:] - white[..., :-1]).var().item()
        hf_red = (red[..., 1:] - red[..., :-1]).var().item()
        assert hf_red < hf_white, (
            f"Red noise should have less high-freq variance than white: white={hf_white:.4f}, red={hf_red:.4f}"
        )

    def test_device_preserved(self, x):
        noise = ColorNoise(amplitude=0.1, reddening=2)
        assert noise(x).device == x.device


# ===========================================================================
# SphericalNoise
# ===========================================================================


class TestSphericalNoise:
    def test_output_shape(self, x):
        noise = SphericalNoise(amplitude=0.1, smoothness=2.0, length_scale=3.0)
        out = noise(x)
        assert out.shape == x.shape

    def test_zero_amplitude(self, x):
        noise = SphericalNoise(amplitude=0.0, smoothness=2.0, length_scale=3.0)
        out = noise(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_finite_output(self, x):
        noise = SphericalNoise(amplitude=0.1, smoothness=2.0, length_scale=3.0)
        out = noise(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("smoothness,length_scale", [(1.5, 2.0), (2.0, 3.0), (3.0, 5.0)])
    def test_parameter_combinations(self, x, smoothness, length_scale):
        noise = SphericalNoise(amplitude=0.1, smoothness=smoothness, length_scale=length_scale)
        out = noise(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_higher_smoothness_smoother(self):
        """Higher smoothness → more spatially correlated (lower pointwise variance)."""
        torch.manual_seed(42)
        shape = (1, 1, 1, 32, 64)
        x = torch.zeros(*shape)

        rough = SphericalNoise(amplitude=1.0, smoothness=1.5, length_scale=3.0)(x)
        smooth = SphericalNoise(amplitude=1.0, smoothness=4.0, length_scale=3.0)(x)

        hf_rough = (rough[..., 1:] - rough[..., :-1]).var().item()
        hf_smooth = (smooth[..., 1:] - smooth[..., :-1]).var().item()
        assert hf_smooth < hf_rough, (
            f"Higher smoothness should reduce high-freq variance: rough={hf_rough:.4f}, smooth={hf_smooth:.4f}"
        )


# ===========================================================================
# TemporalNoise
# ===========================================================================


class TestTemporalNoise:
    def _make(self, rho=0.9):
        base = GaussianNoise(amplitude=0.1)
        return TemporalNoise(
            noise_generator=base,
            temporal_correlation=rho,
            hemispheric_rescale=False,
        )

    def test_output_is_tuple(self, x):
        tn = self._make()
        result = tn(x)
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_shapes(self, x):
        tn = self._make()
        perturbed, delta = tn(x)
        assert perturbed.shape == x.shape
        assert delta.shape == x.shape

    def test_step1_no_previous(self, x):
        """First step with no previous perturbation should still produce output."""
        tn = self._make()
        perturbed, delta = tn(x, previous_perturbation=None, forecast_step=1)
        assert perturbed.shape == x.shape

    def test_ar1_correlation(self, x):
        """Consecutive perturbations should be correlated (dot product > 0 on average)."""
        tn = self._make(rho=0.95)
        _, delta0 = tn(x, forecast_step=1)
        _, delta1 = tn(x, previous_perturbation=delta0, forecast_step=2)
        # δ₁ = ρ*δ₀ + ε → should be correlated with δ₀
        cos_sim = torch.nn.functional.cosine_similarity(delta0.flatten(), delta1.flatten(), dim=0).item()
        assert cos_sim > 0.5, f"Expected AR(1) correlation but got cosine_sim={cos_sim:.3f}"

    def test_zero_correlation_independent(self, x):
        """With rho=0, consecutive deltas should be near-independent."""
        tn = self._make(rho=0.0)
        _, delta0 = tn(x, forecast_step=1)
        _, delta1 = tn(x, previous_perturbation=delta0, forecast_step=2)
        cos_sim = abs(torch.nn.functional.cosine_similarity(delta0.flatten(), delta1.flatten(), dim=0).item())
        assert cos_sim < 0.3, f"rho=0 should give near-zero correlation, got {cos_sim:.3f}"


# ===========================================================================
# crps_spatial_avg
# ===========================================================================


class TestCRPSSpatialAvg:
    def test_perfect_forecast_crps_zero(self, w_lat):
        """Ensemble collapsed to truth → CRPS = 0."""
        n, lat, lon = 10, 16, 32
        truth = np.random.randn(lat, lon)
        pred = np.tile(truth, (n, 1, 1))  # all members = truth
        crps, _ = crps_spatial_avg(pred, truth, w_lat)
        assert abs(crps) < 1e-6, f"Perfect forecast should have CRPS≈0, got {crps}"

    def test_crps_nonnegative(self, w_lat):
        """CRPS is always ≥ 0."""
        rng = np.random.default_rng(0)
        pred = rng.standard_normal((20, 16, 32))
        truth = rng.standard_normal((16, 32))
        crps, _ = crps_spatial_avg(pred, truth, w_lat)
        assert crps >= 0.0, f"CRPS must be ≥ 0, got {crps}"

    def test_crps_improves_with_more_members(self, w_lat):
        """CRPS from a well-calibrated ensemble should decrease as ensemble size grows."""
        rng = np.random.default_rng(42)
        truth = np.zeros((16, 32))
        # Unit-normal ensemble — draws from same distribution as truth
        crps_small, _ = crps_spatial_avg(rng.standard_normal((5, 16, 32)), truth, w_lat)
        crps_large, _ = crps_spatial_avg(rng.standard_normal((100, 16, 32)), truth, w_lat)
        assert crps_large < crps_small, (
            f"Larger ensemble should give lower CRPS: small={crps_small:.4f}, large={crps_large:.4f}"
        )

    def test_spread_positive(self, w_lat):
        """Spread should be > 0 for a random ensemble."""
        rng = np.random.default_rng(1)
        pred = rng.standard_normal((20, 16, 32))
        truth = np.zeros((16, 32))
        _, spread = crps_spatial_avg(pred, truth, w_lat)
        assert spread > 0.0, "Spread must be > 0 for random ensemble"

    def test_spread_zero_for_constant_ensemble(self, w_lat):
        """Constant ensemble → spread = 0."""
        truth = np.zeros((16, 32))
        pred = np.ones((10, 16, 32))
        _, spread = crps_spatial_avg(pred, truth, w_lat)
        assert abs(spread) < 1e-6, f"Constant ensemble should have spread=0, got {spread}"

    def test_known_value(self, w_lat):
        """Verify sorted-ensemble formula against brute-force for a small case."""
        rng = np.random.default_rng(7)
        n, lat, lon = 8, 4, 8
        w = np.cos(np.deg2rad(np.linspace(-90, 90, lat)))
        w = w / w.mean()
        pred = rng.standard_normal((n, lat, lon))
        truth = rng.standard_normal((lat, lon))

        # Brute-force reference
        def sp(arr):
            return (arr * w[:, None]).sum() / (lat * lon)

        term1_ref = sp(np.abs(pred - truth[None]).mean(axis=0))
        term2_ref = 0.0
        for i in range(n):
            for j in range(n):
                term2_ref += sp(np.abs(pred[i] - pred[j]))
        term2_ref /= n * n
        crps_ref = term1_ref - 0.5 * term2_ref

        crps_val, _ = crps_spatial_avg(pred, truth, w)
        assert abs(crps_val - crps_ref) < 1e-10, (
            f"Sorted formula disagrees with brute-force: {crps_val:.8f} vs {crps_ref:.8f}"
        )

    def test_output_is_scalar(self, w_lat):
        rng = np.random.default_rng(0)
        crps, spread = crps_spatial_avg(
            rng.standard_normal((10, 16, 32)),
            rng.standard_normal((16, 32)),
            w_lat,
        )
        assert isinstance(crps, float)
        assert isinstance(spread, float)
