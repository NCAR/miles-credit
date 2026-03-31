"""
Tests for CREDIT IC-perturbation ensemble classes.

Covers:
  - GaussianNoise
  - ColorNoise
  - SphericalNoise
  - TemporalNoise
  - crps_spatial_avg (credit.verification.ensemble)
  - add_spatially_correlated_noise, hemispheric_rescale (credit.ensemble.utils)
  - calculate_crps_per_channel (credit.ensemble.crps)
  - BredVector init (credit.ensemble.bred_vector)

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


# ===========================================================================
# credit.ensemble.utils
# ===========================================================================


class TestEnsembleUtils:
    """Tests for add_spatially_correlated_noise and hemispheric_rescale."""

    from credit.ensemble.utils import add_spatially_correlated_noise, hemispheric_rescale

    SHAPE = (1, 2, 1, 16, 32)  # (B, C, T, H, W)

    def test_correlated_noise_output_shape(self):
        from credit.ensemble.utils import add_spatially_correlated_noise

        x = torch.zeros(*self.SHAPE)
        out = add_spatially_correlated_noise(x, correlation_scale=2)
        assert out.shape == x.shape

    def test_correlated_noise_is_nonzero(self):
        from credit.ensemble.utils import add_spatially_correlated_noise

        x = torch.zeros(*self.SHAPE)
        out = add_spatially_correlated_noise(x, correlation_scale=2)
        assert not torch.allclose(out, x), "Noise should be non-zero"

    def test_correlated_noise_is_finite(self):
        from credit.ensemble.utils import add_spatially_correlated_noise

        x = torch.zeros(*self.SHAPE)
        out = add_spatially_correlated_noise(x, correlation_scale=2)
        assert torch.isfinite(out).all()

    def test_higher_scale_smoother(self):
        """Higher correlation_scale → lower pointwise variance (smoother noise)."""
        torch.manual_seed(0)
        x = torch.zeros(1, 1, 1, 32, 64)
        from credit.ensemble.utils import add_spatially_correlated_noise

        rough = add_spatially_correlated_noise(x, correlation_scale=1)
        smooth = add_spatially_correlated_noise(x, correlation_scale=10)
        hf_rough = (rough[..., 1:] - rough[..., :-1]).var().item()
        hf_smooth = (smooth[..., 1:] - smooth[..., :-1]).var().item()
        assert hf_smooth < hf_rough, (
            f"Higher correlation_scale should be smoother: rough={hf_rough:.4f}, smooth={hf_smooth:.4f}"
        )

    def test_hemispheric_rescale_unity(self):
        """north_scale=south_scale=1 → output equals input."""
        from credit.ensemble.utils import hemispheric_rescale

        x = torch.ones(1, 2, 1, 8, 16)
        lats = torch.linspace(90, -90, 8)
        out = hemispheric_rescale(x, lats, north_scale=1.0, south_scale=1.0)
        assert torch.allclose(out, x)

    def test_hemispheric_rescale_shape(self):
        from credit.ensemble.utils import hemispheric_rescale

        x = torch.ones(*self.SHAPE)
        lats = torch.linspace(90, -90, self.SHAPE[3])
        out = hemispheric_rescale(x, lats)
        assert out.shape == x.shape

    def test_hemispheric_rescale_polarity(self):
        """north_scale=2, south_scale=0.5 → north pixels ≈ 2, south pixels ≈ 0.5."""
        from credit.ensemble.utils import hemispheric_rescale

        H = 16
        x = torch.ones(1, 1, 1, H, 4)
        lats = torch.linspace(90, -90, H)  # lat[0]=90 (north), lat[-1]=-90 (south)
        out = hemispheric_rescale(x, lats, north_scale=2.0, south_scale=0.5)
        # First row is lat=90 → north → scale=2
        assert torch.allclose(out[:, :, :, 0, :], torch.full_like(out[:, :, :, 0, :], 2.0))
        # Last row is lat=-90 → south → scale=0.5
        assert torch.allclose(out[:, :, :, -1, :], torch.full_like(out[:, :, :, -1, :], 0.5))

    def test_hemispheric_rescale_finite(self):
        from credit.ensemble.utils import hemispheric_rescale

        x = torch.randn(*self.SHAPE)
        lats = torch.linspace(90, -90, self.SHAPE[3])
        out = hemispheric_rescale(x, lats, north_scale=1.5, south_scale=0.8)
        assert torch.isfinite(out).all()


# ===========================================================================
# credit.ensemble.crps
# ===========================================================================


class TestCalculateCRPSPerChannel:
    """Tests for calculate_crps_per_channel."""

    # ensemble_predictions: [ensemble_size, 1, channels, 1, H, W]
    # y_true:               [1, channels, 1, H, W]

    @staticmethod
    def _make(ensemble_size=10, channels=3, H=8, W=16, fill=None, rng_seed=None):
        """Build synthetic ensemble and truth tensors."""
        if rng_seed is not None:
            torch.manual_seed(rng_seed)
        if fill is not None:
            ens = torch.full((ensemble_size, 1, channels, 1, H, W), float(fill))
            truth = torch.full((1, channels, 1, H, W), float(fill))
        else:
            ens = torch.randn(ensemble_size, 1, channels, 1, H, W)
            truth = torch.randn(1, channels, 1, H, W)
        return ens, truth

    def test_output_shape(self):
        from credit.ensemble.crps import calculate_crps_per_channel

        ens, truth = self._make(ensemble_size=8, channels=5)
        out = calculate_crps_per_channel(ens, truth)
        assert out.shape == (1, 5), f"Expected (1, 5), got {out.shape}"

    def test_perfect_forecast_zero_crps(self):
        """All ensemble members equal truth → CRPS ≈ 0 for every channel."""
        from credit.ensemble.crps import calculate_crps_per_channel

        ens, truth = self._make(fill=3.14)
        out = calculate_crps_per_channel(ens, truth)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), f"Perfect forecast should give CRPS≈0, got {out}"

    def test_crps_nonnegative(self):
        from credit.ensemble.crps import calculate_crps_per_channel

        ens, truth = self._make(rng_seed=42)
        out = calculate_crps_per_channel(ens, truth)
        assert (out >= 0).all(), f"CRPS must be ≥ 0, got {out}"

    def test_crps_finite(self):
        from credit.ensemble.crps import calculate_crps_per_channel

        ens, truth = self._make(rng_seed=7)
        out = calculate_crps_per_channel(ens, truth)
        assert torch.isfinite(out).all()

    def test_larger_ensemble_lower_crps(self):
        """Well-calibrated unit-normal ensemble: more members → lower or equal CRPS."""
        from credit.ensemble.crps import calculate_crps_per_channel

        torch.manual_seed(0)
        truth = torch.zeros(1, 1, 1, 8, 16)
        ens_small = torch.randn(5, 1, 1, 1, 8, 16)
        ens_large = torch.randn(200, 1, 1, 1, 8, 16)
        crps_small = calculate_crps_per_channel(ens_small, truth).mean().item()
        crps_large = calculate_crps_per_channel(ens_large, truth).mean().item()
        assert crps_large <= crps_small + 0.1, (
            f"Larger ensemble should give ≤ CRPS: small={crps_small:.4f}, large={crps_large:.4f}"
        )


# ===========================================================================
# credit.ensemble.bred_vector — BredVector init
# ===========================================================================


class TestBredVectorInit:
    """Tests for BredVector.__init__ with a simple toy model.

    BredVector requires a real model forward pass for __call__; we only test
    __init__ here (which is all CPU, no data required).
    """

    INPUT_STATIC_DIM = 3

    @staticmethod
    def _identity_model(x):
        """Mock model: return only dynamic channels (strips last INPUT_STATIC_DIM channels)."""
        return x[:, : -TestBredVectorInit.INPUT_STATIC_DIM, ...]

    def test_default_attributes(self):
        from credit.ensemble.bred_vector import BredVector

        bv = BredVector(model=self._identity_model)
        assert bv.noise_amplitude == 0.15
        assert bv.num_cycles == 5
        assert bv.integration_steps == 1
        assert bv.clamp is False
        assert bv.flag_mass_conserve is False
        assert bv.flag_water_conserve is False
        assert bv.flag_energy_conserve is False
        assert bv.use_post_block is False

    def test_custom_attributes(self):
        from credit.ensemble.bred_vector import BredVector

        bv = BredVector(
            model=self._identity_model,
            noise_amplitude=0.05,
            num_cycles=3,
            integration_steps=2,
            clamp=True,
            clamp_min=-1.0,
            clamp_max=1.0,
        )
        assert bv.noise_amplitude == 0.05
        assert bv.num_cycles == 3
        assert bv.integration_steps == 2
        assert bv.clamp is True
        assert bv.clamp_min == -1.0
        assert bv.clamp_max == 1.0

    def test_empty_post_conf_no_postblock(self):
        from credit.ensemble.bred_vector import BredVector

        bv = BredVector(model=self._identity_model, post_conf={})
        assert bv.use_post_block is False

    def test_hemispheric_rescale_false_by_default(self):
        from credit.ensemble.bred_vector import BredVector

        bv = BredVector(model=self._identity_model)
        # When hemispheric_rescale=False the attribute is set to False (not the fn)
        assert bv.hemispheric_rescale is False

    def test_perturb_output_shape(self):
        """perturb() returns a perturbation for dynamic channels only (C - input_static_dim)."""
        from credit.ensemble.bred_vector import BredVector

        C, static = 8, self.INPUT_STATIC_DIM
        bv = BredVector(model=self._identity_model, noise_amplitude=0.1, input_static_dim=static)
        x = torch.zeros(1, C, 1, 16, 32)
        out = bv.perturb(x)
        expected_C = C - static
        assert out.shape == (1, expected_C, 1, 16, 32), f"Expected {(1, expected_C, 1, 16, 32)}, got {out.shape}"

    def test_perturb_is_noisy(self):
        """perturb() perturbation should be non-zero for a non-zero state.

        Note: gamma_final = ||x_dyn|| / (||x_dyn + dx|| + eps), so an all-zero
        input state would collapse gamma to 0. Use randn input instead.
        """
        from credit.ensemble.bred_vector import BredVector

        torch.manual_seed(0)
        C, static = 8, self.INPUT_STATIC_DIM
        bv = BredVector(model=self._identity_model, noise_amplitude=0.5, input_static_dim=static)
        x = torch.randn(1, C, 1, 8, 16)
        out = bv.perturb(x)
        assert not torch.allclose(out, torch.zeros_like(out)), "perturb() should produce non-zero perturbation"
