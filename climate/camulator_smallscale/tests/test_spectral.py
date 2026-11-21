"""Spherical-harmonic spectra on the CAMulator grid (camulator_smallscale.spectral)."""

import math

import pytest
import torch

from camulator_smallscale.spectral import PARSEVAL, SphericalSpectra, band_sums, total_wavenumber_to_km

NLAT, NLON = 192, 288
MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.fixture(scope="module")
def spectra():
    return SphericalSpectra(NLAT, NLON, dtype=torch.float64)


def _coefficients(spectra, lmax_field, seed=0, slope=None):
    """Random real-field coefficients up to ``lmax_field``; with ``slope``, power falls off as ``l**slope``."""
    torch.manual_seed(seed)
    coeffs = torch.randn(spectra.lmax, spectra.mmax, dtype=torch.complex128)
    coeffs[lmax_field + 1 :] = 0
    coeffs = torch.tril(coeffs)  # m <= l
    coeffs[:, 0] = coeffs[:, 0].real.to(torch.complex128)
    if slope is not None:
        # power ~ (2l+1) |c|^2, so |c| ~ l**((slope - 1) / 2) gives power ~ l**slope
        coeffs = coeffs * torch.arange(1, spectra.lmax + 1, dtype=torch.float64)[:, None] ** ((slope - 1) / 2)
    return coeffs


def _field(spectra, coeffs):
    from torch_harmonics import InverseRealSHT

    isht = InverseRealSHT(NLAT, NLON, lmax=spectra.lmax, mmax=spectra.mmax, grid="equiangular").double()
    return isht(coeffs)


def _band_limited(spectra, lmax_field, seed=0, slope=None):
    return _field(spectra, _coefficients(spectra, lmax_field, seed, slope))


def test_default_truncation_is_triangular_at_the_zonal_nyquist(spectra):
    assert (spectra.lmax, spectra.mmax) == (145, 145)
    assert spectra.area_weights.mean().item() == pytest.approx(1.0)
    assert torch.allclose(spectra.area_weights, spectra.area_weights.flip(0))  # symmetric about the equator


def test_parseval(spectra):
    # Exact while every coefficient's quadrature is exact: l + 45 <= nlat - 1 for all l <= 144.
    f = _band_limited(spectra, 45)
    assert spectra.area_mean_square(f).item() == pytest.approx(PARSEVAL * spectra.power(f).sum().item(), rel=1e-9)


def test_red_spectrum_band_powers_are_recovered(spectra):
    # Up to the truncation, with an atmosphere-like slope: the quadrature leak stays negligible.
    coeffs = _coefficients(spectra, 144, seed=5, slope=-3.0)
    bands = [(1, 10), (11, 40), (41, 100), (101, 144)]
    true = band_sums(spectra.power_from(coeffs), bands)
    estimated = band_sums(spectra.power(_field(spectra, coeffs)), bands)
    torch.testing.assert_close(estimated, true, rtol=1e-3, atol=0)


def test_single_harmonic_lands_on_its_wavenumber(spectra):
    from torch_harmonics import InverseRealSHT

    isht = InverseRealSHT(NLAT, NLON, lmax=spectra.lmax, mmax=spectra.mmax, grid="equiangular").double()
    coeffs = torch.zeros(spectra.lmax, spectra.mmax, dtype=torch.complex128)
    coeffs[37, 11] = 1.0 + 0.5j
    power = spectra.power(isht(coeffs))
    assert power[37] / power.sum() > 1 - 1e-9


def test_power_invariant_to_longitude_shift_and_latitude_flip(spectra):
    torch.manual_seed(1)
    x = torch.randn(3, NLAT, NLON, dtype=torch.float64)
    power = spectra.power(x)
    torch.testing.assert_close(spectra.power(torch.roll(x, 17, dims=-1)), power, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(spectra.power(torch.flip(x, dims=(-2,))), power, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("slope", [None, -3.0])
def test_cross_and_coherence(spectra, slope):
    x = _band_limited(spectra, 120, seed=2, slope=slope)
    y = _band_limited(spectra, 120, seed=3, slope=slope)
    cx = spectra.transform(x)
    torch.testing.assert_close(spectra.cross_from(cx, cx), spectra.power_from(cx))
    coh_self = spectra.coherence(x, x, rel_eps=0)[1:121]
    coh_neg = spectra.coherence(x, -x, rel_eps=0)[1:121]
    torch.testing.assert_close(coh_self, torch.ones_like(coh_self), rtol=0, atol=1e-9)
    torch.testing.assert_close(coh_neg, -torch.ones_like(coh_neg), rtol=0, atol=1e-9)
    # The default relative floor barely moves it, even where a red spectrum's power is tiny.
    coh_floor = spectra.coherence(x, x)[1:121]
    torch.testing.assert_close(coh_floor, torch.ones_like(coh_floor), rtol=0, atol=1e-4)
    coh = spectra.coherence(x, y)
    assert coh.abs().max() <= 1 + 1e-9


def test_gradients_are_finite_at_zero_power(spectra):
    x = torch.zeros(NLAT, NLON, dtype=torch.float64, requires_grad=True)
    y = _band_limited(spectra, 60)
    spectra.coherence(x, y).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_band_sums():
    spectrum = torch.arange(10.0)
    torch.testing.assert_close(band_sums(spectrum, [(0, 1), (2, 4), (9, 9)]), torch.tensor([1.0, 9.0, 9.0]))


def test_wavelengths():
    assert total_wavenumber_to_km(40) == pytest.approx(1000.7, rel=1e-3)
    assert total_wavenumber_to_km(100) == pytest.approx(400.3, rel=1e-3)
    assert total_wavenumber_to_km(1) == pytest.approx(2 * math.pi * 6371.0)


@pytest.mark.skipif(not MPS, reason="needs Apple MPS")
def test_power_on_mps_matches_cpu():
    spectra = SphericalSpectra(NLAT, NLON)
    torch.manual_seed(4)
    x = torch.randn(2, 5, NLAT, NLON)
    cpu = spectra.power(x)
    mps = spectra.to("mps").power(x.to("mps")).cpu()
    torch.testing.assert_close(mps, cpu, rtol=1e-4, atol=1e-4 * cpu.abs().max().item())
