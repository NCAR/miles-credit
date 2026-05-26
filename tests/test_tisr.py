"""
test_tisr.py
------------
Pure PyTorch unit and integration tests for the TISR implementation
(credit/datasets/tisr.py). No JAX or GraphCast dependency — safe to run
as part of the normal CI test suite.

Usage
-----
Run from the repository root:

    pytest tests/test_tisr.py -v

Coverage
--------
To check test coverage:

    pytest tests/test_tisr.py --cov=credit.datasets.tisr --cov-report=term-missing -v
"""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from unittest.mock import MagicMock, patch

from credit.datasets.tisr import (
    _era5_tsi_data,
    _get_tsi,
    _get_j2000_days,
    _get_orbital_parameters,
    _get_solar_time,
    _get_hour_angle,
    _get_cosine_zenith_angle,
    _get_instantaneous_toa_tisr,
    _get_integrated_toa_tisr,
    _load_latlon_grid,
    _compute_tisr,
    TISRDataset,
)

# Representative timestamps — chosen to exercise diverse solar geometry:
#   * March equinox (near-zero declination)
#   * June solstice (max N declination)
#   * September equinox
#   * December solstice (max S declination)
#   * High noon on equator, midnight, and an arbitrary mid-day time
TIMESTAMPS = [
    "2020-03-20 12:00:00",  # March equinox
    "2020-06-21 00:00:00",  # June solstice midnight
    "2020-06-21 12:00:00",  # June solstice noon
    "2020-09-22 06:00:00",  # September equinox dawn
    "2020-12-21 18:00:00",  # December solstice dusk
    "2000-01-01 12:00:00",  # J2000 reference epoch
    "1989-11-08 21:00:00",  # Example from ERA5 docs
]


def to_np(x) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array."""
    return x.detach().cpu().numpy()


# =============================================================================
# NetCDF fixtures
# =============================================================================

def _write_rectangular_netcdf(path: str, ny: int = 4, nx: int = 8) -> None:
    """Write a minimal rectangular-grid NetCDF file with 1-D lat/lon coords."""
    lat = np.linspace(-90, 90, ny)
    lon = np.linspace(0, 360, nx, endpoint=False)
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})
    ds.to_netcdf(path)


def _write_curvilinear_netcdf(path: str, ny: int = 4, nx: int = 8) -> None:
    """Write a minimal curvilinear-grid NetCDF file with 2-D lat/lon variables."""
    lat_1d = np.linspace(-90, 90, ny)
    lon_1d = np.linspace(0, 360, nx, endpoint=False)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    ds = xr.Dataset(
        {
            "lat": (["y", "x"], lat_2d),
            "lon": (["y", "x"], lon_2d),
        }
    )
    ds.to_netcdf(path)


@pytest.fixture()
def rectangular_nc(tmp_path):
    """Temporary rectangular-grid NetCDF file (1-D lat/lon)."""
    path = str(tmp_path / "rect.nc")
    _write_rectangular_netcdf(path)
    return path


@pytest.fixture()
def curvilinear_nc(tmp_path):
    """Temporary curvilinear-grid NetCDF file (2-D lat/lon)."""
    path = str(tmp_path / "curv.nc")
    _write_curvilinear_netcdf(path)
    return path


# =============================================================================
# 1. TSI data integrity
# =============================================================================

class TestTSIDataIntegrity:
    """Checks on the ERA5 TSI lookup table itself."""

    def test_tsi_dataset_length(self):
        """ERA5 TSI dataset must cover exactly 84 years (1951.5–2034.5)."""
        times, values = _era5_tsi_data()
        assert len(times) == 84, f"Expected 84 TSI entries, got {len(times)}"
        assert len(values) == 84, f"Expected 84 TSI value entries, got {len(values)}"

    def test_tsi_times_are_monotonic(self):
        """TSI time axis must be strictly increasing (required for searchsorted)."""
        times, _ = _era5_tsi_data()
        diffs = times[1:] - times[:-1]
        assert (diffs > 0).all(), "TSI times are not strictly monotonically increasing"

    def test_tsi_times_start_and_end(self):
        """TSI data must span 1951.5 to 2034.5 exactly."""
        times, _ = _era5_tsi_data()
        assert abs(times[0].item() - 1951.5) < 1e-9, \
            f"TSI data should start at 1951.5, got {times[0].item()}"
        assert abs(times[-1].item() - 2034.5) < 1e-9, \
            f"TSI data should end at 2034.5, got {times[-1].item()}"

    def test_tsi_scale_factor_applied(self):
        """All TSI values must be scaled by 0.9965 (below ~1366 W/m²)."""
        _, values = _era5_tsi_data()
        assert (values < 1366.0).all(), \
            "TSI values should all be below 1366 W/m² (0.9965 scale not applied?)"

    def test_tsi_single_timestamp(self):
        """_get_tsi must handle a single pd.Timestamp (not just sequences)."""
        times, values = _era5_tsi_data()
        result = _get_tsi(pd.Timestamp("2020-06-21 12:00:00"), times, values)
        assert result.shape == (1,), \
            f"Single timestamp should return shape (1,), got {result.shape}"

    def test_tsi_leap_year(self):
        """TSI interpolation on Feb 29 of a leap year must not crash."""
        times, values = _era5_tsi_data()
        result = _get_tsi(["2020-02-29 12:00:00"], times, values)
        assert result.shape == (1,) and (result > 1358).all(), \
            f"Leap day TSI result unexpected: {result}"

    def test_tsi_year_boundary(self):
        """TSI on Dec 31 and Jan 1 of adjacent years must be close (smooth interpolation)."""
        times, values = _era5_tsi_data()
        dec31 = _get_tsi(["2020-12-31 23:00:00"], times, values)
        jan01 = _get_tsi(["2021-01-01 00:00:00"], times, values)
        assert abs(dec31.item() - jan01.item()) < 1.0, \
            f"TSI should be continuous across year boundary: {dec31.item():.4f} vs {jan01.item():.4f}"


# =============================================================================
# 2. TSI interpolation
# =============================================================================

class TestTSI:
    """Unit tests for _get_tsi."""

    def test_tsi_range(self):
        """All TSI values should be in physically plausible range ~1360–1362 W/m²."""
        times_torch, values_torch = _era5_tsi_data()
        result = _get_tsi(TIMESTAMPS, times_torch, values_torch)
        assert (result > 1358).all() and (result < 1364).all(), \
            f"TSI out of plausible range: {result}"

    def test_tsi_rejects_out_of_range_timestamp(self):
        """Timestamps outside 1951–2034 must raise ValueError."""
        times_torch, values_torch = _era5_tsi_data()
        with pytest.raises(ValueError):
            _get_tsi(["1800-01-01 00:00:00"], times_torch, values_torch)


# =============================================================================
# 3. J2000 day conversion
# =============================================================================

class TestJ2000Days:
    """Unit tests for _get_j2000_days."""

    def test_j2000_epoch_is_zero(self):
        """J2000 epoch (2000-01-01 12:00 TT) should give exactly 0.0 days."""
        result = _get_j2000_days(pd.Timestamp("2000-01-01 12:00:00"))
        assert abs(result.item()) < 1e-3, \
            f"J2000 epoch should be ~0, got {result.item()}"

    def test_j2000_days_batch_vs_scalar(self):
        """Batch input must give the same result as calling scalar inputs one by one."""
        batch_result = to_np(_get_j2000_days(pd.DatetimeIndex(TIMESTAMPS)))
        scalar_results = np.array([
            _get_j2000_days(pd.Timestamp(ts)).item()
            for ts in TIMESTAMPS
        ])
        np.testing.assert_array_equal(batch_result, scalar_results,
            err_msg="Batch J2000 days differ from scalar equivalents")

    def test_j2000_days_ordering(self):
        """Later timestamps must produce larger J2000 day values."""
        ts1 = _get_j2000_days(pd.Timestamp("2000-01-01 00:00:00")).item()
        ts2 = _get_j2000_days(pd.Timestamp("2020-06-21 12:00:00")).item()
        assert ts2 > ts1, "Later timestamp must have larger J2000 day value"

    def test_j2000_days_one_day_increment(self):
        """Two timestamps exactly 1 day apart must differ by exactly 1.0 J2000 days."""
        t1 = _get_j2000_days(pd.Timestamp("2020-06-21 00:00:00")).item()
        t2 = _get_j2000_days(pd.Timestamp("2020-06-22 00:00:00")).item()
        assert abs((t2 - t1) - 1.0) < 1e-9, \
            f"One day apart should differ by 1.0 J2000 days, got {t2 - t1}"

    def test_j2000_days_output_dtype(self):
        """Output dtype must be float64."""
        result = _get_j2000_days(pd.Timestamp("2020-06-21 12:00:00"))
        assert result.dtype == torch.float64, \
            f"Expected float64, got {result.dtype}"


# =============================================================================
# 4. Orbital parameters
# =============================================================================

class TestOrbitalParameters:
    """Unit tests for _get_orbital_parameters."""

    def test_solar_distance_near_one_au(self):
        """Earth-Sun distance should always be ~0.98–1.02 AU."""
        j2000_days = _get_j2000_days(pd.DatetimeIndex(TIMESTAMPS))
        op = _get_orbital_parameters(j2000_days)
        d = op["solar_distance_au"]
        assert (d > 0.98).all() and (d < 1.02).all(), \
            f"Solar distance out of expected range: {d}"

    def test_declination_in_valid_range(self):
        """sin(declination) must be in [-sin(23.44°), +sin(23.44°)] ≈ ±0.398."""
        j2000_days = _get_j2000_days(pd.DatetimeIndex(TIMESTAMPS))
        op = _get_orbital_parameters(j2000_days)
        limit = np.sin(np.radians(23.44))
        sd = op["sin_declination"].abs()
        assert (sd <= limit + 1e-3).all(), \
            f"sin_declination exceeds axial tilt bound: {sd}"


# =============================================================================
# 5. Solar time
# =============================================================================

class TestSolarTime:
    """Unit tests for _get_solar_time."""

    def test_solar_time_at_j2000_epoch(self):
        """At J2000 epoch, rotational_phase=0, eq_of_time≈-3s → solar_time≈0."""
        j2000 = torch.tensor([0.0], dtype=torch.float64)
        op = _get_orbital_parameters(j2000)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        assert abs(solar_time.item()) < 0.01, \
            f"Solar time at J2000 epoch should be near 0, got {solar_time.item()}"

    def test_solar_time_bounded(self):
        """Solar time (rotational_phase + eq_of_time correction) should stay near [0, 1)."""
        j2000_days = _get_j2000_days(pd.DatetimeIndex(TIMESTAMPS))
        op = _get_orbital_parameters(j2000_days)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        # eq_of_time is at most ~17 minutes = 0.012 days, so solar_time stays near [0, 1)
        assert (solar_time > -0.02).all() and (solar_time < 1.02).all(), \
            f"Solar time out of expected bounds: {solar_time}"


# =============================================================================
# 6. Solar geometry / hour angle
# =============================================================================

class TestSolarGeometry:
    """Unit tests for _get_hour_angle and _get_cosine_zenith_angle."""

    def test_hour_angle_uses_degrees(self):
        """
        The PyTorch version uses: hour_angle = 360 * solar_time + longitude_deg.
        At solar noon (solar_time=0.5) on the prime meridian (lon=0°), the hour
        angle should be 180°.
        """
        solar_time = torch.tensor([0.5], dtype=torch.float64)  # midday
        longitude = torch.tensor([0.0], dtype=torch.float64)   # prime meridian
        ha = _get_hour_angle(solar_time, longitude)
        assert abs(ha.item() - 180.0) < 1e-3, \
            f"Hour angle at solar noon / lon=0 should be 180°, got {ha.item()}"

    def test_cosine_zenith_equator_equinox_noon(self):
        """At solar noon on the equator with declination=0, cos_zenith should be 1.0."""
        cos_dec = torch.tensor([[[1.0]]], dtype=torch.float64)
        sin_dec = torch.tensor([[[0.0]]], dtype=torch.float64)
        lat = torch.tensor([[[0.0]]], dtype=torch.float64)   # equator, degrees
        ha = torch.tensor([[[0.0]]], dtype=torch.float64)    # solar noon, degrees

        cz = _get_cosine_zenith_angle(cos_dec, sin_dec, lat, ha)
        assert abs(cz.item() - 1.0) < 1e-5, \
            f"cos_zenith at equatorial noon/equinox should be 1.0, got {cz.item()}"

    def test_cosine_zenith_nightside_is_zero(self):
        """Below-horizon values must be clamped to 0 (not negative)."""
        cos_dec = torch.tensor([[[0.5]]], dtype=torch.float64)
        sin_dec = torch.tensor([[[0.866]]], dtype=torch.float64)
        lat = torch.tensor([[[-80.0]]], dtype=torch.float64)  # near south pole
        ha = torch.tensor([[[180.0]]], dtype=torch.float64)   # midnight

        cz = _get_cosine_zenith_angle(cos_dec, sin_dec, lat, ha)
        assert cz.item() >= 0.0, "Nightside cos_zenith must be >= 0"


# =============================================================================
# 7. Instantaneous TOA TISR
# =============================================================================

class TestInstantaneousFlux:
    """Unit tests for _get_instantaneous_toa_tisr."""

    def test_flux_nonnegative(self):
        """Radiation flux must always be non-negative."""
        tsi = torch.tensor([[[1361.0]]], dtype=torch.float64)
        solar_factor = torch.tensor([[[1.0]]], dtype=torch.float64)
        cos_zenith = torch.tensor([[[-0.5, 0.0, 0.5, 1.0]]], dtype=torch.float64)
        flux = _get_instantaneous_toa_tisr(tsi, solar_factor, cos_zenith)
        assert (flux >= 0).all(), "Flux must be non-negative everywhere"


# =============================================================================
# 8. Integrated TOA TISR
# =============================================================================

class TestIntegratedTISR:
    """Unit and integration tests for _get_integrated_toa_tisr."""

    def test_integration_shape(self):
        """Output of _get_integrated_toa_tisr must drop the time dimension."""
        inst = torch.rand(361, 5, 4, dtype=torch.float64)
        result = _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), 360)
        assert result.shape == (5, 4), \
            f"Expected shape (5, 4), got {result.shape}"

    def test_num_integration_steps_validation(self):
        """Non-positive or non-integer num_integration_steps must raise ValueError."""
        inst = torch.rand(361, 5, 4, dtype=torch.float64)
        with pytest.raises(ValueError):
            _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), 0)
        with pytest.raises(ValueError):
            _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), -1)

    def test_mismatched_steps_raises(self):
        """Passing wrong number of time steps must raise ValueError."""
        inst = torch.rand(100, 5, 4, dtype=torch.float64)  # 100 != 360 + 1
        with pytest.raises(ValueError):
            _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), 360)

    def test_output_dtype_is_float64(self):
        """Integrated TISR output dtype must be float64."""
        inst = torch.rand(361, 3, 4, dtype=torch.float64)
        result = _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), 360)
        assert result.dtype == torch.float64, \
            f"Expected float64 output, got {result.dtype}"

    @pytest.mark.parametrize("num_steps", [60, 180, 360, 720])
    def test_convergence_with_more_bins(self, num_steps):
        """Result should converge as num_steps increases; all bin counts should
        agree with the 360-bin reference to within 1%."""
        ts_str = "2020-06-21 12:00:00"
        lat_deg = np.array([0.0, 45.0])
        lon_deg = np.array([0.0, 90.0])

        def _run(n_steps):
            integration_period = pd.Timedelta(hours=1)
            t = pd.Timestamp(ts_str)
            ts = pd.date_range(end=t, periods=n_steps + 1,
                               freq=integration_period / n_steps)
            times_t, tsi_v = _era5_tsi_data()
            tsi = _get_tsi(ts, times_t, tsi_v).unsqueeze(-1).unsqueeze(-1)
            j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
            op = _get_orbital_parameters(j2000_days)
            lat_t = torch.tensor(lat_deg, dtype=torch.float64).reshape(1, -1, 1)
            lon_t = torch.tensor(lon_deg, dtype=torch.float64).reshape(1, 1, -1)
            solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
            ha = _get_hour_angle(solar_time, lon_t)
            cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
            inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
            return to_np(_get_integrated_toa_tisr(inst, integration_period, n_steps))

        result_360 = _run(360)
        result = _run(num_steps)
        np.testing.assert_allclose(result, result_360, rtol=1e-2,
            err_msg=f"num_steps={num_steps} diverges too much from 360-bin reference")

    def test_non_standard_integration_period(self):
        """A 6-hour integration period should give more energy than 1-hour
        at the same midday location."""
        ts_str = "2020-06-21 12:00:00"
        lat_deg = np.array([0.0])
        lon_deg = np.array([0.0])

        def _run(period_hours):
            integration_period = pd.Timedelta(hours=period_hours)
            num_steps = 360
            t = pd.Timestamp(ts_str)
            ts = pd.date_range(end=t, periods=num_steps + 1,
                               freq=integration_period / num_steps)
            times_t, tsi_v = _era5_tsi_data()
            tsi = _get_tsi(ts, times_t, tsi_v).unsqueeze(-1).unsqueeze(-1)
            j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
            op = _get_orbital_parameters(j2000_days)
            lat_t = torch.tensor(lat_deg, dtype=torch.float64).reshape(1, -1, 1)
            lon_t = torch.tensor(lon_deg, dtype=torch.float64).reshape(1, 1, -1)
            solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
            ha = _get_hour_angle(solar_time, lon_t)
            cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
            inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
            return _get_integrated_toa_tisr(inst, integration_period, num_steps).item()

        result_1h = _run(1)
        result_6h = _run(6)
        assert result_6h > result_1h, \
            f"6-hour TISR ({result_6h:.1f}) should exceed 1-hour TISR ({result_1h:.1f})"

    def test_polar_night_is_zero(self):
        """South pole in June: no sunlight -> integrated TISR must be 0 J/m2."""
        ts_str = "2020-06-21 12:00:00"
        lat_deg = np.array([-90.0])
        lon_deg = np.array([0.0])
        integration_period = pd.Timedelta(hours=1)
        num_steps = 360

        t = pd.Timestamp(ts_str)
        ts = pd.date_range(end=t, periods=num_steps + 1, freq=integration_period / num_steps)
        times_torch, tsi_values_torch = _era5_tsi_data()
        tsi = _get_tsi(ts, times_torch, tsi_values_torch).unsqueeze(-1).unsqueeze(-1)
        j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
        op = _get_orbital_parameters(j2000_days)
        lat_t = torch.tensor(lat_deg, dtype=torch.float64).reshape(1, -1, 1)
        lon_t = torch.tensor(lon_deg, dtype=torch.float64).reshape(1, 1, -1)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, lon_t)
        cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
        inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
        result = _get_integrated_toa_tisr(inst, integration_period, num_steps)

        assert result.item() == pytest.approx(0.0, abs=1.0), \
            f"South pole in June should have zero TISR, got {result.item()}"


# =============================================================================
# 9. Physical symmetry checks (full pipeline)
# =============================================================================

class TestPhysicalSymmetry:
    """Sanity checks based on known physical symmetries of solar geometry."""

    def test_longitude_periodicity(self):
        """lon=0 and lon=360 are the same point; TISR must be identical."""
        ts_str = "2020-06-21 12:00:00"
        integration_period = pd.Timedelta(hours=1)
        num_steps = 360

        def _tisr_at_lon(lon_val):
            t = pd.Timestamp(ts_str)
            ts = pd.date_range(end=t, periods=num_steps + 1,
                               freq=integration_period / num_steps)
            times_t, tsi_v = _era5_tsi_data()
            tsi = _get_tsi(ts, times_t, tsi_v).unsqueeze(-1).unsqueeze(-1)
            j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
            op = _get_orbital_parameters(j2000_days)
            lat_t = torch.tensor([45.0], dtype=torch.float64).reshape(1, -1, 1)
            lon_t = torch.tensor([lon_val], dtype=torch.float64).reshape(1, 1, -1)
            solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
            ha = _get_hour_angle(solar_time, lon_t)
            cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
            inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
            return _get_integrated_toa_tisr(inst, integration_period, num_steps).item()

        assert abs(_tisr_at_lon(0.0) - _tisr_at_lon(360.0)) < 1e-6, \
            "lon=0 and lon=360 must give identical TISR"

    def test_north_pole_midsummer_positive(self):
        """North pole in June has 24-hour daylight -> integrated TISR must be > 0."""
        ts_str = "2020-06-21 12:00:00"
        integration_period = pd.Timedelta(hours=1)
        num_steps = 360
        t = pd.Timestamp(ts_str)
        ts = pd.date_range(end=t, periods=num_steps + 1,
                           freq=integration_period / num_steps)
        times_t, tsi_v = _era5_tsi_data()
        tsi = _get_tsi(ts, times_t, tsi_v).unsqueeze(-1).unsqueeze(-1)
        j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
        op = _get_orbital_parameters(j2000_days)
        lat_t = torch.tensor([90.0], dtype=torch.float64).reshape(1, -1, 1)
        lon_t = torch.tensor([0.0], dtype=torch.float64).reshape(1, 1, -1)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, lon_t)
        cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
        inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
        result = _get_integrated_toa_tisr(inst, integration_period, num_steps).item()
        assert result > 0, \
            f"North pole in June should have positive TISR, got {result}"

    def test_tisr_upper_bound(self):
        """Integrated TISR cannot exceed TSI * solar_factor * integration_seconds.
        For a 1-hour window, that is ~1366 * 1.034 * 3600 ~ 5.08e6 J/m2."""
        ts_str = "2020-01-03 12:00:00"  # perihelion: max solar_factor ~1.034
        integration_period = pd.Timedelta(hours=1)
        num_steps = 360
        max_possible = 1366.0 * 1.034 * 3600.0

        t = pd.Timestamp(ts_str)
        ts = pd.date_range(end=t, periods=num_steps + 1,
                           freq=integration_period / num_steps)
        times_t, tsi_v = _era5_tsi_data()
        tsi = _get_tsi(ts, times_t, tsi_v).unsqueeze(-1).unsqueeze(-1)
        j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
        op = _get_orbital_parameters(j2000_days)
        lats = np.linspace(-90, 90, 19)
        lons = np.linspace(0, 360, 37)
        lat_t = torch.tensor(lats, dtype=torch.float64).reshape(1, -1, 1)
        lon_t = torch.tensor(lons, dtype=torch.float64).reshape(1, 1, -1)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, lon_t)
        cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
        inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
        result = to_np(_get_integrated_toa_tisr(inst, integration_period, num_steps))
        assert result.max() <= max_possible, \
            f"TISR exceeds physical upper bound: {result.max():.1f} > {max_possible:.1f}"


# =============================================================================
# 10. _load_latlon_grid
# =============================================================================

class TestLoadLatlonGrid:
    """Tests for _load_latlon_grid covering both grid types and error paths."""

    def test_rectangular_grid_shape(self, rectangular_nc):
        """Rectangular grid: output tensors must have shape (1, ny, nx)."""
        lat, lon = _load_latlon_grid(rectangular_nc)
        assert lat.shape == (1, 4, 8), f"Expected (1, 4, 8), got {lat.shape}"
        assert lon.shape == (1, 4, 8), f"Expected (1, 4, 8), got {lon.shape}"

    def test_rectangular_grid_dtype(self, rectangular_nc):
        """Rectangular grid: output tensors must be float64."""
        lat, lon = _load_latlon_grid(rectangular_nc)
        assert lat.dtype == torch.float64
        assert lon.dtype == torch.float64

    def test_rectangular_grid_values(self, rectangular_nc):
        """Rectangular grid: lat range must be [-90, 90], lon range [0, 360)."""
        lat, lon = _load_latlon_grid(rectangular_nc)
        assert lat.min().item() == pytest.approx(-90.0)
        assert lat.max().item() == pytest.approx(90.0)
        assert lon.min().item() == pytest.approx(0.0)

    def test_curvilinear_grid_shape(self, curvilinear_nc):
        """Curvilinear grid: output tensors must have shape (1, ny, nx)."""
        lat, lon = _load_latlon_grid(curvilinear_nc)
        assert lat.shape == (1, 4, 8), f"Expected (1, 4, 8), got {lat.shape}"
        assert lon.shape == (1, 4, 8), f"Expected (1, 4, 8), got {lon.shape}"

    def test_curvilinear_grid_dtype(self, curvilinear_nc):
        """Curvilinear grid: output tensors must be float64."""
        lat, lon = _load_latlon_grid(curvilinear_nc)
        assert lat.dtype == torch.float64
        assert lon.dtype == torch.float64

    def test_missing_file_raises(self, tmp_path):
        """Non-existent file must raise ValueError."""
        with pytest.raises(ValueError, match="Could not open"):
            _load_latlon_grid(str(tmp_path / "does_not_exist.nc"))

    def test_missing_latlon_raises(self, tmp_path):
        """NetCDF file with no recognisable lat/lon fields must raise ValueError."""
        path = str(tmp_path / "no_latlon.nc")
        ds = xr.Dataset({"temperature": (["x"], np.zeros(4))})
        ds.to_netcdf(path)
        with pytest.raises(ValueError, match="No latitude/longitude found"):
            _load_latlon_grid(path)


# =============================================================================
# 11. _compute_tisr
# =============================================================================

class TestComputeTISR:
    """Tests for _compute_tisr using a mocked NetCDF lat/lon grid."""

    def test_output_shape(self, rectangular_nc):
        """Output shape must be (ny, nx) — the spatial grid, no time dim."""
        result = _compute_tisr(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            integration_period=pd.Timedelta(hours=1),
            num_integration_steps=360,
            latlon_grid_path=rectangular_nc,
        )
        assert result.shape == (4, 8), f"Expected (4, 8), got {result.shape}"

    def test_output_dtype(self, rectangular_nc):
        """Output must be float64."""
        result = _compute_tisr(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            integration_period=pd.Timedelta(hours=1),
            num_integration_steps=360,
            latlon_grid_path=rectangular_nc,
        )
        assert result.dtype == torch.float64, \
            f"Expected float64, got {result.dtype}"

    def test_output_nonnegative(self, rectangular_nc):
        """All integrated TISR values must be non-negative."""
        result = _compute_tisr(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            integration_period=pd.Timedelta(hours=1),
            num_integration_steps=360,
            latlon_grid_path=rectangular_nc,
        )
        assert (result >= 0).all(), "Integrated TISR must be non-negative everywhere"

    def test_polar_night_via_compute_tisr(self, rectangular_nc):
        """South pole in June must receive zero TISR via the full _compute_tisr pipeline."""
        result = _compute_tisr(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            integration_period=pd.Timedelta(hours=1),
            num_integration_steps=360,
            latlon_grid_path=rectangular_nc,
        )
        # rectangular_nc has lat starting at -90; first row is the south pole
        south_pole_row = result[0, :]
        assert (south_pole_row.abs() <= 1.0).all(), \
            f"South pole row in June should be zero via _compute_tisr, got {south_pole_row}"

    def test_different_integration_period(self, rectangular_nc):
        """6-hour integration period must give more energy than 1-hour."""
        kwargs = dict(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            num_integration_steps=360,
            latlon_grid_path=rectangular_nc,
        )
        result_1h = _compute_tisr(integration_period=pd.Timedelta(hours=1), **kwargs)
        result_6h = _compute_tisr(integration_period=pd.Timedelta(hours=6), **kwargs)
        assert result_6h.sum() > result_1h.sum(), \
            "6-hour integrated TISR should exceed 1-hour integrated TISR"


# =============================================================================
# 12. TISRDataset
# =============================================================================

@pytest.fixture()
def tisr_dataset(rectangular_nc):
    """TISRDataset with BaseDataset.__init__ mocked out.

    Patches BaseDataset.__init__ to do nothing, then manually sets the
    attributes that TISRDataset.__init__ and its methods depend on, so that
    the full CREDIT config pipeline is not required.
    """
    with patch("credit.datasets.tisr.BaseDataset.__init__", return_value=None):
        ds = TISRDataset.__new__(TISRDataset)
        # Attributes normally provided by BaseDataset.__init__
        ds.curr_source_cfg = {
            "dataset_type": "tisr",
            "num_integration_steps": 360,
            "latlon_grid_path": rectangular_nc,
        }
        ds.dt = pd.Timedelta(hours=1)
        ds.var_dict = {"dynamic_forcing": {"vars_2D": ["tisr"]}}
        ds.init_register_all_fields = MagicMock()
        ds._get_field_name = MagicMock(return_value="src/dynamic_forcing/2d/tisr")
        # Run the real __init__ with a dummy data_config (BaseDataset.__init__ is mocked)
        TISRDataset.__init__(ds, data_config={}, return_target=False)
    return ds


class TestTISRDataset:
    """Tests for TISRDataset.__init__, _get_file_source, and _extract_field."""

    def test_init_sets_dataset_type(self, tisr_dataset):
        """__init__ must set dataset_type to 'tisr'."""
        assert tisr_dataset.dataset_type == "tisr"

    def test_init_sets_num_integration_steps(self, tisr_dataset):
        """__init__ must read num_integration_steps from config."""
        assert tisr_dataset.num_integration_steps == 360

    def test_init_default_num_integration_steps(self, rectangular_nc):
        """__init__ must default num_integration_steps to 360 when not in config."""
        with patch("credit.datasets.tisr.BaseDataset.__init__", return_value=None):
            ds = TISRDataset.__new__(TISRDataset)
            ds.curr_source_cfg = {
                "dataset_type": "tisr",
                "latlon_grid_path": rectangular_nc,
                # num_integration_steps intentionally omitted
            }
            ds.dt = pd.Timedelta(hours=1)
            ds.var_dict = {}
            ds.init_register_all_fields = MagicMock()
            ds._get_field_name = MagicMock()
            TISRDataset.__init__(ds, data_config={}, return_target=False)
        assert ds.num_integration_steps == 360

    def test_init_sets_latlon_grid_path(self, tisr_dataset, rectangular_nc):
        """__init__ must store latlon_grid_path from config."""
        assert tisr_dataset.latlon_grid_path == rectangular_nc

    def test_init_sets_static_metadata(self, tisr_dataset):
        """__init__ must set static_metadata with datetime_fmt."""
        assert tisr_dataset.static_metadata == {"datetime_fmt": "unix_ns"}

    def test_init_wrong_dataset_type_raises(self, rectangular_nc):
        """__init__ must assert-fail if dataset_type is not 'tisr'."""
        with patch("credit.datasets.tisr.BaseDataset.__init__", return_value=None):
            ds = TISRDataset.__new__(TISRDataset)
            ds.curr_source_cfg = {"dataset_type": "wrong"}
            ds.dt = pd.Timedelta(hours=1)
            ds.var_dict = {}
            ds.init_register_all_fields = MagicMock()
            ds._get_field_name = MagicMock()
            with pytest.raises(AssertionError):
                TISRDataset.__init__(ds, data_config={}, return_target=False)

    def test_get_file_source_returns_none(self, tisr_dataset):
        """_get_file_source must always return None."""
        assert tisr_dataset._get_file_source({}) is None

    def test_extract_field_no_var_dict(self, tisr_dataset):
        """_extract_field must do nothing when field_type has no var_dict entry."""
        sample = {}
        tisr_dataset._extract_field("prognostic", pd.Timestamp("2020-06-21 12:00:00"), sample)
        assert sample == {}, "sample must remain empty when field_type is not in var_dict"

    def test_extract_field_empty_vars_2d(self, tisr_dataset):
        """_extract_field must do nothing when vars_2D is empty."""
        tisr_dataset.var_dict = {"dynamic_forcing": {"vars_2D": []}}
        sample = {}
        tisr_dataset._extract_field("dynamic_forcing", pd.Timestamp("2020-06-21 12:00:00"), sample)
        assert sample == {}, "sample must remain empty when vars_2D is empty"

    def test_extract_field_wrong_variable_raises(self, tisr_dataset):
        """_extract_field must raise ValueError when vars_2D is not ['tisr']."""
        tisr_dataset.var_dict = {"dynamic_forcing": {"vars_2D": ["u10"]}}
        with pytest.raises(ValueError, match="TISRDataset only supports"):
            tisr_dataset._extract_field(
                "dynamic_forcing", pd.Timestamp("2020-06-21 12:00:00"), {}
            )

    def test_extract_field_writes_to_sample(self, tisr_dataset):
        """_extract_field must compute TISR and store it in the sample dict."""
        sample = {}
        tisr_dataset._extract_field(
            "dynamic_forcing", pd.Timestamp("2020-06-21 12:00:00"), sample
        )
        key = "src/dynamic_forcing/2d/tisr"
        assert key in sample, f"Expected key '{key}' in sample"
        assert sample[key].shape == (1, 1, 4, 8), \
            f"Expected shape (1, 1, 4, 8), got {sample[key].shape}"
        assert (sample[key] >= 0).all(), "TISR values must be non-negative"