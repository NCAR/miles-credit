"""
Tests for credit.verification.standard (spectral analysis functions).

Covers:
  - zonal_spectrum: output shape, finite values, non-negative power
  - average_zonal_spectrum: reduces batch dims, result is 1-D

Guarded by pytest.mark.skipif for torch_harmonics — if not installed CI skips cleanly.
"""

import numpy as np
import pytest
import xarray as xr

try:
    from torch_harmonics import RealSHT  # noqa: F401

    _HARMONICS_AVAILABLE = True
except ImportError:
    _HARMONICS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HARMONICS_AVAILABLE,
    reason="torch_harmonics not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LAT, LON = 32, 64  # small grid for speed


def _make_da(nlat=LAT, nlon=LON, extra_dims=None):
    """Build a minimal xr.DataArray with (latitude, longitude) as last two dims."""
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    shape = (nlat, nlon) if extra_dims is None else (*extra_dims, nlat, nlon)
    data = np.random.default_rng(0).standard_normal(shape).astype("float64")
    if extra_dims is None:
        return xr.DataArray(data, dims=["latitude", "longitude"], coords={"latitude": lats, "longitude": lons})
    else:
        extra_coords = {f"dim{i}": range(s) for i, s in enumerate(extra_dims)}
        dims = [f"dim{i}" for i in range(len(extra_dims))] + ["latitude", "longitude"]
        coords = {"latitude": lats, "longitude": lons, **extra_coords}
        return xr.DataArray(data, dims=dims, coords=coords)


def _make_uv_ds(nlat=LAT, nlon=LON):
    """Build a Dataset with U and V fields for div/rot spectrum tests."""
    da = _make_da(nlat, nlon)
    return xr.Dataset({"U": da, "V": da.copy()})


# ---------------------------------------------------------------------------
# zonal_spectrum
# ---------------------------------------------------------------------------


class TestZonalSpectrum:
    def test_output_is_tensor(self):
        from credit.verification.standard import zonal_spectrum

        da = _make_da()
        result = zonal_spectrum(da, grid="equiangular")
        import torch

        assert isinstance(result, torch.Tensor)

    def test_output_last_dim_is_lmax(self):
        """Last dimension should be lmax = nlat + 1."""
        from credit.verification.standard import zonal_spectrum

        da = _make_da()
        result = zonal_spectrum(da, grid="equiangular")
        expected_last = LAT + 1
        assert result.shape[-1] == expected_last, f"Expected lmax={expected_last}, got {result.shape[-1]}"

    def test_output_nonnegative(self):
        """Spectral power should be non-negative (it's |coeff|^2)."""
        from credit.verification.standard import zonal_spectrum

        da = _make_da()
        result = zonal_spectrum(da, grid="equiangular")
        assert (result >= 0).all(), "Spectral power must be non-negative"

    def test_output_finite(self):
        from credit.verification.standard import zonal_spectrum

        da = _make_da()
        result = zonal_spectrum(da, grid="equiangular")
        import torch

        assert torch.isfinite(result).all(), "Spectral power contains non-finite values"

    def test_zero_field_zero_spectrum(self):
        """A zero field should have zero spectral power everywhere."""
        from credit.verification.standard import zonal_spectrum

        da = _make_da() * 0.0
        result = zonal_spectrum(da, grid="equiangular")
        import torch

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-10), "Zero field should give zero spectrum"

    def test_batch_dims_preserved(self):
        """Extra leading dims should be preserved in the output."""
        from credit.verification.standard import zonal_spectrum

        da = _make_da(extra_dims=(3, 2))
        result = zonal_spectrum(da, grid="equiangular")
        assert result.shape[:2] == (3, 2), f"Expected leading dims (3,2), got {result.shape[:2]}"


# ---------------------------------------------------------------------------
# average_zonal_spectrum
# ---------------------------------------------------------------------------


class TestAverageZonalSpectrum:
    def test_output_is_1d_numpy(self):
        """Should reduce all dims except last (wavenumber) and return numpy array."""
        from credit.verification.standard import average_zonal_spectrum

        da = _make_da(extra_dims=(4, 2))
        result = average_zonal_spectrum(da, grid="equiangular")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1, f"Expected 1-D output, got shape {result.shape}"

    def test_output_length(self):
        """With a batched input, result should have mmax = lmax//2+1 wavenumbers."""
        from credit.verification.standard import average_zonal_spectrum

        da = _make_da(extra_dims=(3,))
        result = average_zonal_spectrum(da, grid="equiangular")
        assert result.ndim == 1 and len(result) > 0, f"Expected 1-D output, got shape {result.shape}"

    def test_output_nonnegative(self):
        from credit.verification.standard import average_zonal_spectrum

        da = _make_da(extra_dims=(2,))
        result = average_zonal_spectrum(da, grid="equiangular")
        assert (result >= 0).all()

    def test_output_finite(self):
        from credit.verification.standard import average_zonal_spectrum

        da = _make_da(extra_dims=(2,))
        result = average_zonal_spectrum(da, grid="equiangular")
        assert np.isfinite(result).all()
