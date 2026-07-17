"""
tests/test_tisr.py
------------
Unit tests for the TISRDataset class in credit.datasets.gen_2.tisr.py.

"""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from credit.datasets.gen_2.tisr import (
    _era5_tsi_data,  # pyright: ignore[reportPrivateUsage]
    _get_tsi,  # pyright: ignore[reportPrivateUsage]
    _get_j2000_days,  # pyright: ignore[reportPrivateUsage]
    _get_orbital_parameters,  # pyright: ignore[reportPrivateUsage]
    _get_solar_time,  # pyright: ignore[reportPrivateUsage]
    _get_hour_angle,  # pyright: ignore[reportPrivateUsage]
    _get_cosine_zenith_angle,  # pyright: ignore[reportPrivateUsage]
    _get_instantaneous_toa_tisr,  # pyright: ignore[reportPrivateUsage]
    _get_integrated_toa_tisr,  # pyright: ignore[reportPrivateUsage]
    _get_latlon_grid,  # pyright: ignore[reportPrivateUsage]
    _compute_tisr,  # pyright: ignore[reportPrivateUsage]
    TISRDataset,
)

# Representative timestamps — chosen to exercise diverse solar geometry:
#   * March equinox (near-zero declination)
#   * June solstice (max N declination)
#   * September equinox
#   * December solstice (max S declination)
#   * J2000 epoch (reference for day count)
TIMESTAMPS = [
    "2020-03-20 12:00:00",  # March equinox
    "2020-06-21 00:00:00",  # June solstice midnight
    "2020-06-21 12:00:00",  # June solstice noon
    "2020-12-21 18:00:00",  # December solstice dusk
    "2000-01-01 12:00:00",  # J2000 reference epoch
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
        """ERA5 TSI times and tsi_values must have the same length."""
        times, tsi_values = _era5_tsi_data()
        assert len(times) == len(tsi_values), f"times length {len(times)} != tsi_values length {len(tsi_values)}"

    def test_tsi_times_are_monotonic(self):
        """TSI time axis must be strictly increasing (required for searchsorted)."""
        times, _ = _era5_tsi_data()
        diffs = times[1:] - times[:-1]
        assert (diffs > 0).all(), "TSI times are not strictly monotonically increasing"

    def test_tsi_single_timestamp(self):
        """_get_tsi must handle a single pd.Timestamp (not just sequences)."""
        times, values = _era5_tsi_data()
        result = _get_tsi(pd.Timestamp("2020-06-21 12:00:00"), times, values)
        assert result.shape == (1,), f"Single timestamp should return shape (1,), got {result.shape}"

    def test_tsi_leap_year(self):
        """TSI interpolation on Feb 29 of a leap year must not crash."""
        times, values = _era5_tsi_data()
        result = _get_tsi(["2020-02-29 12:00:00"], times, values)
        assert result.shape == (1,) and (result > 1360).all(), f"Leap day TSI result unexpected: {result}"

    def test_tsi_year_boundary(self):
        """TSI on Dec 31 and Jan 1 of adjacent years must be close (smooth interpolation)."""
        times, values = _era5_tsi_data()
        dec31 = _get_tsi(["2020-12-31 23:00:00"], times, values)
        jan01 = _get_tsi(["2021-01-01 00:00:00"], times, values)
        assert abs(dec31.item() - jan01.item()) < 1.0, (
            f"TSI should be continuous across year boundary: {dec31.item():.4f} vs {jan01.item():.4f}"
        )


# =============================================================================
# 2. TSI interpolation
# =============================================================================


class TestTSI:
    """Unit tests for _get_tsi."""

    def test_tsi_range(self):
        """All TSI values should be in physically plausible range ~1360–1362 W/m²."""
        times_torch, values_torch = _era5_tsi_data()
        result = _get_tsi(TIMESTAMPS, times_torch, values_torch)
        assert (result > 1360).all() and (result < 1362).all(), f"TSI out of plausible range: {result}"

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
        assert abs(result.item()) < 1e-3, f"J2000 epoch should be ~0, got {result.item()}"

    def test_j2000_days_batch_vs_scalar(self):
        """Batch input must give the same result as calling scalar inputs one by one."""
        batch_result = to_np(_get_j2000_days(pd.DatetimeIndex(TIMESTAMPS)))
        scalar_results = np.array([_get_j2000_days(pd.Timestamp(ts)).item() for ts in TIMESTAMPS])
        np.testing.assert_array_equal(
            batch_result, scalar_results, err_msg="Batch J2000 days differ from scalar equivalents"
        )

    def test_j2000_days_ordering(self):
        """Later timestamps must produce larger J2000 day values."""
        ts1 = _get_j2000_days(pd.Timestamp("2000-01-01 00:00:00")).item()
        ts2 = _get_j2000_days(pd.Timestamp("2020-06-21 12:00:00")).item()
        assert ts2 > ts1, "Later timestamp must have larger J2000 day value"

    def test_j2000_days_one_day_increment(self):
        """Two timestamps exactly 1 day apart must differ by exactly 1.0 J2000 days."""
        t1 = _get_j2000_days(pd.Timestamp("2020-06-21 00:00:00")).item()
        t2 = _get_j2000_days(pd.Timestamp("2020-06-22 00:00:00")).item()
        assert abs((t2 - t1) - 1.0) < 1e-9, f"One day apart should differ by 1.0 J2000 days, got {t2 - t1}"

    def test_j2000_days_output_dtype(self):
        """Output dtype must be float32."""
        result = _get_j2000_days(pd.Timestamp("2020-06-21 12:00:00"))
        assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"


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
        assert (d > 0.98).all() and (d < 1.02).all(), f"Solar distance out of expected range: {d}"

    def test_declination_in_valid_range(self):
        """sin(declination) must be in [-sin(23.44°), +sin(23.44°)] ≈ ±0.398."""
        j2000_days = _get_j2000_days(pd.DatetimeIndex(TIMESTAMPS))
        op = _get_orbital_parameters(j2000_days)
        limit = 0.3979  # sin(23.44°)
        sd = op["sin_declination"].abs()
        assert (sd <= limit + 1e-3).all(), f"sin_declination exceeds axial tilt bound: {sd}"


# =============================================================================
# 5. Solar time
# =============================================================================


class TestSolarTime:
    """Unit tests for _get_solar_time."""

    def test_solar_time_at_j2000_epoch(self):
        """At J2000 epoch, rotational_phase=0, eq_of_time≈-3s → solar_time≈0."""
        j2000 = torch.tensor([0.0], dtype=torch.float32)
        op = _get_orbital_parameters(j2000)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        assert abs(solar_time.item()) < 0.01, f"Solar time at J2000 epoch should be near 0, got {solar_time.item()}"

    def test_solar_time_bounded(self):
        """Solar time (rotational_phase + eq_of_time correction) should stay near [0, 1)."""
        j2000_days = _get_j2000_days(pd.DatetimeIndex(TIMESTAMPS))
        op = _get_orbital_parameters(j2000_days)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        # eq_of_time is at most ~17 minutes = 0.012 days, so solar_time stays near [0, 1)
        assert (solar_time > -0.02).all() and (solar_time < 1.02).all(), (
            f"Solar time out of expected bounds: {solar_time}"
        )


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
        solar_time = torch.tensor([0.5], dtype=torch.float32)  # midday
        longitude = torch.tensor([0.0], dtype=torch.float32)  # prime meridian
        ha = _get_hour_angle(solar_time, longitude)
        assert abs(ha.item() - 180.0) < 1e-3, f"Hour angle at solar noon / lon=0 should be 180°, got {ha.item()}"

    def test_cosine_zenith_equator_equinox_noon(self):
        """At solar noon on the equator with declination=0, cos_zenith should be 1.0."""
        cos_dec = torch.tensor([[[1.0]]], dtype=torch.float32)
        sin_dec = torch.tensor([[[0.0]]], dtype=torch.float32)
        lat = torch.tensor([[[0.0]]], dtype=torch.float32)  # equator, degrees
        ha = torch.tensor([[[0.0]]], dtype=torch.float32)  # solar noon, degrees

        cz = _get_cosine_zenith_angle(cos_dec, sin_dec, lat, ha)
        assert abs(cz.item() - 1.0) < 1e-5, f"cos_zenith at equatorial noon/equinox should be 1.0, got {cz.item()}"

    def test_cosine_zenith_nightside_is_zero(self):
        """Below-horizon values must be clamped to 0 (not negative)."""
        cos_dec = torch.tensor([[[0.5]]], dtype=torch.float32)
        sin_dec = torch.tensor([[[0.866]]], dtype=torch.float32)
        lat = torch.tensor([[[-80.0]]], dtype=torch.float32)  # near south pole
        ha = torch.tensor([[[180.0]]], dtype=torch.float32)  # midnight

        cz = _get_cosine_zenith_angle(cos_dec, sin_dec, lat, ha)
        assert cz.item() >= 0.0, "Nightside cos_zenith must be >= 0"


# =============================================================================
# 7. Instantaneous TOA TISR
# =============================================================================


class TestInstantaneousFlux:
    """Unit tests for _get_instantaneous_toa_tisr."""

    def test_flux_nonnegative(self):
        """Radiation flux must always be non-negative."""
        tsi = torch.tensor([[[1361.0]]], dtype=torch.float32)
        solar_factor = torch.tensor([[[1.0]]], dtype=torch.float32)
        cos_zenith = torch.tensor([[[-0.5, 0.0, 0.5, 1.0]]], dtype=torch.float32)
        flux = _get_instantaneous_toa_tisr(tsi, solar_factor, cos_zenith)
        assert (flux >= 0).all(), "Flux must be non-negative everywhere"


# =============================================================================
# 8. Integrated TOA TISR
# =============================================================================


class TestIntegratedTISR:
    """Unit and integration tests for _get_integrated_toa_tisr."""

    def test_integration_shape(self):
        """Output of _get_integrated_toa_tisr must drop the time dimension."""
        inst = torch.rand(361, 5, 4, dtype=torch.float32)
        result = _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), 360)
        assert result.shape == (5, 4), f"Expected shape (5, 4), got {result.shape}"

    def test_num_integration_steps_validation(self):
        """Non-positive or non-integer num_integration_steps must raise ValueError."""
        inst = torch.rand(361, 5, 4, dtype=torch.float32)
        with pytest.raises(ValueError):
            _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), 0)
        with pytest.raises(ValueError):
            _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), -1)

    def test_mismatched_steps_raises(self):
        """Passing wrong number of time steps must raise ValueError."""
        inst = torch.rand(100, 5, 4, dtype=torch.float32)  # 100 != 360 + 1
        with pytest.raises(ValueError):
            _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), 360)

    def test_output_dtype_is_float32(self):
        """Integrated TISR output dtype must be float32."""
        inst = torch.rand(361, 3, 4, dtype=torch.float32)
        result = _get_integrated_toa_tisr(inst, pd.Timedelta(hours=1), 360)
        assert result.dtype == torch.float32, f"Expected float32 output, got {result.dtype}"

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
            ts = pd.date_range(end=t, periods=n_steps + 1, freq=integration_period / n_steps)
            times_t, tsi_v = _era5_tsi_data()
            tsi = _get_tsi(ts, times_t, tsi_v).unsqueeze(-1).unsqueeze(-1)
            j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
            op = _get_orbital_parameters(j2000_days)
            lat_t = torch.tensor(lat_deg, dtype=torch.float32).reshape(1, -1, 1)
            lon_t = torch.tensor(lon_deg, dtype=torch.float32).reshape(1, 1, -1)
            solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
            ha = _get_hour_angle(solar_time, lon_t)
            cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
            inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
            return to_np(_get_integrated_toa_tisr(inst, integration_period, n_steps))

        result_360 = _run(360)
        result = _run(num_steps)
        np.testing.assert_allclose(
            result, result_360, rtol=1e-2, err_msg=f"num_steps={num_steps} diverges too much from 360-bin reference"
        )

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
            ts = pd.date_range(end=t, periods=num_steps + 1, freq=integration_period / num_steps)
            times_t, tsi_v = _era5_tsi_data()
            tsi = _get_tsi(ts, times_t, tsi_v).unsqueeze(-1).unsqueeze(-1)
            j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
            op = _get_orbital_parameters(j2000_days)
            lat_t = torch.tensor(lat_deg, dtype=torch.float32).reshape(1, -1, 1)
            lon_t = torch.tensor(lon_deg, dtype=torch.float32).reshape(1, 1, -1)
            solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
            ha = _get_hour_angle(solar_time, lon_t)
            cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
            inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
            return _get_integrated_toa_tisr(inst, integration_period, num_steps).item()

        result_1h = _run(1)
        result_6h = _run(6)
        assert result_6h > result_1h, f"6-hour TISR ({result_6h:.1f}) should exceed 1-hour TISR ({result_1h:.1f})"

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
        lat_t = torch.tensor(lat_deg, dtype=torch.float32).reshape(1, -1, 1)
        lon_t = torch.tensor(lon_deg, dtype=torch.float32).reshape(1, 1, -1)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, lon_t)
        cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
        inst = _get_instantaneous_toa_tisr(tsi, 1.0 / op["solar_distance_au"] ** 2, cz)
        result = _get_integrated_toa_tisr(inst, integration_period, num_steps)

        assert result.item() == pytest.approx(0.0, abs=1.0), (
            f"South pole in June should have zero TISR, got {result.item()}"
        )


# =============================================================================
# 9. _get_latlon_grid
# =============================================================================


class TestGetLatlonGrid:
    """Tests for _get_latlon_grid covering file grids, spec grids, and error paths."""

    def test_rectangular_grid_shape(self, rectangular_nc):
        """Rectangular grid: output tensors must have shape (1, ny, nx)."""
        lat, lon = _get_latlon_grid(path=rectangular_nc)
        assert lat.shape == (1, 4, 8), f"Expected (1, 4, 8), got {lat.shape}"
        assert lon.shape == (1, 4, 8), f"Expected (1, 4, 8), got {lon.shape}"

    def test_rectangular_grid_dtype(self, rectangular_nc):
        """Rectangular grid: output tensors must be float32."""
        lat, lon = _get_latlon_grid(path=rectangular_nc)
        assert lat.dtype == torch.float32
        assert lon.dtype == torch.float32

    def test_rectangular_grid_values(self, rectangular_nc):
        """Rectangular grid: lat range must be [-90, 90], lon range [0, 360)."""
        lat, lon = _get_latlon_grid(path=rectangular_nc)
        assert lat.min().item() == pytest.approx(-90.0)
        assert lat.max().item() == pytest.approx(90.0)
        assert lon.min().item() == pytest.approx(0.0)

    def test_curvilinear_grid_shape(self, curvilinear_nc):
        """Curvilinear grid: output tensors must have shape (1, ny, nx)."""
        lat, lon = _get_latlon_grid(path=curvilinear_nc)
        assert lat.shape == (1, 4, 8), f"Expected (1, 4, 8), got {lat.shape}"
        assert lon.shape == (1, 4, 8), f"Expected (1, 4, 8), got {lon.shape}"

    def test_curvilinear_grid_dtype(self, curvilinear_nc):
        """Curvilinear grid: output tensors must be float32."""
        lat, lon = _get_latlon_grid(path=curvilinear_nc)
        assert lat.dtype == torch.float32
        assert lon.dtype == torch.float32

    def test_missing_file_raises(self, tmp_path):
        """Non-existent file must raise ValueError."""
        with pytest.raises(ValueError, match="Could not open"):
            _get_latlon_grid(path=str(tmp_path / "does_not_exist.nc"))

    def test_missing_latlon_raises(self, tmp_path):
        """NetCDF file with no recognisable lat/lon fields must raise ValueError."""
        path = str(tmp_path / "no_latlon.nc")
        ds = xr.Dataset({"temperature": (["x"], np.zeros(4))})
        ds.to_netcdf(path)
        with pytest.raises(ValueError, match="No latitude/longitude found"):
            _get_latlon_grid(path=path)

    def test_spec_grid_shape_and_values(self):
        """Spec mode: [start, end, num_points] endpoints inclusive, shape (1, ny, nx)."""
        lat, lon = _get_latlon_grid(lat_spec=[90, -90, 721], lon_spec=[0, 359.75, 1440])
        assert lat.shape == (1, 721, 1440), f"Got {lat.shape}"
        assert lon.shape == (1, 721, 1440), f"Got {lon.shape}"
        assert lat[0, 0, 0].item() == pytest.approx(90.0)
        assert lat[0, -1, 0].item() == pytest.approx(-90.0)
        assert lon[0, 0, 0].item() == pytest.approx(0.0)

    def test_spec_grid_dtype(self):
        """Spec mode: output tensors must be float32."""
        lat, lon = _get_latlon_grid(lat_spec=[90, -90, 3], lon_spec=[0, 180, 90])
        assert lat.dtype == torch.float32
        assert lon.dtype == torch.float32

    def test_neither_source_raises(self):
        """Supplying neither path nor specs must raise ValueError."""
        with pytest.raises(ValueError, match="exactly one grid source"):
            _get_latlon_grid()

    def test_both_sources_raises(self, rectangular_nc):
        """Supplying both path and specs must raise ValueError."""
        with pytest.raises(ValueError, match="exactly one grid source"):
            _get_latlon_grid(path=rectangular_nc, lat_spec=[90, -90, 3], lon_spec=[0, 180, 90])

    def test_one_sided_spec_raises(self):
        """Only one of lat_spec/lon_spec must raise ValueError."""
        with pytest.raises(ValueError, match="Both 'lat_spec' and 'lon_spec'"):
            _get_latlon_grid(lat_spec=[90, -90, 3])

    def test_bad_spec_length_raises(self):
        """Spec not of length 3 must raise ValueError."""
        with pytest.raises(ValueError, match="must be \\[start, end, num_points\\]"):
            _get_latlon_grid(lat_spec=[90, -90], lon_spec=[0, 180, 90])

    def test_non_int_num_points_raises(self):
        """num_points must be an int (float like 2.5 rejected)."""
        with pytest.raises(ValueError, match="num_points must be an int"):
            _get_latlon_grid(lat_spec=[90, -90, 2.5], lon_spec=[0, 180, 90])

    def test_bool_num_points_raises(self):
        """num_points must be a real int, not a bool (True is an int subclass)."""
        with pytest.raises(ValueError, match="num_points must be an int"):
            _get_latlon_grid(lat_spec=[90, -90, True], lon_spec=[0, 180, 90])

    def test_zero_num_points_raises(self):
        """num_points must be >= 1."""
        with pytest.raises(ValueError, match="num_points must be >= 1"):
            _get_latlon_grid(lat_spec=[90, -90, 0], lon_spec=[0, 180, 90])


# =============================================================================
# 10. _compute_tisr
# =============================================================================


class TestComputeTISR:
    """Tests for _compute_tisr using a lat/lon grid built from a NetCDF file."""

    def test_output_shape(self, rectangular_nc):
        """Output shape must be (ny, nx) — the spatial grid, no time dim."""
        lat, lon = _get_latlon_grid(path=rectangular_nc)
        result = _compute_tisr(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            integration_period=pd.Timedelta(hours=1),
            num_integration_steps=360,
            latitude=lat,
            longitude=lon,
        )
        assert result.shape == (4, 8), f"Expected (4, 8), got {result.shape}"

    def test_output_dtype(self, rectangular_nc):
        """Output must be float32."""
        lat, lon = _get_latlon_grid(path=rectangular_nc)
        result = _compute_tisr(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            integration_period=pd.Timedelta(hours=1),
            num_integration_steps=360,
            latitude=lat,
            longitude=lon,
        )
        assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"

    def test_output_nonnegative(self, rectangular_nc):
        """All integrated TISR values must be non-negative."""
        lat, lon = _get_latlon_grid(path=rectangular_nc)
        result = _compute_tisr(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            integration_period=pd.Timedelta(hours=1),
            num_integration_steps=360,
            latitude=lat,
            longitude=lon,
        )
        assert (result >= 0).all(), "Integrated TISR must be non-negative everywhere"

    def test_polar_night_via_compute_tisr(self, rectangular_nc):
        """South pole in June must receive zero TISR via the full _compute_tisr pipeline."""
        lat, lon = _get_latlon_grid(path=rectangular_nc)
        result = _compute_tisr(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            integration_period=pd.Timedelta(hours=1),
            num_integration_steps=360,
            latitude=lat,
            longitude=lon,
        )
        # rectangular_nc has lat starting at -90; first row is the south pole
        south_pole_row = result[0, :]
        assert (south_pole_row.abs() <= 1.0).all(), (
            f"South pole row in June should be zero via _compute_tisr, got {south_pole_row}"
        )

    def test_different_integration_period(self, rectangular_nc):
        """6-hour integration period must give more energy than 1-hour."""
        lat, lon = _get_latlon_grid(path=rectangular_nc)
        kwargs = dict(
            t=pd.Timestamp("2020-06-21 12:00:00"),
            num_integration_steps=360,
            latitude=lat,
            longitude=lon,
        )
        result_1h = _compute_tisr(integration_period=pd.Timedelta(hours=1), **kwargs)
        result_6h = _compute_tisr(integration_period=pd.Timedelta(hours=6), **kwargs)
        assert result_6h.sum() > result_1h.sum(), "6-hour integrated TISR should exceed 1-hour integrated TISR"


# =============================================================================
# 11. TISRDataset
# =============================================================================


def _make_tisr_data_config(grid_source: dict, num_integration_steps: int | None = 360) -> dict:
    """Build a minimal but complete data_config for a single TISR source.

    Args:
        grid_source: Grid keys to merge into the source config, e.g.
            ``{"latlon_grid_path": ...}`` or ``{"lat_spec": [...], "lon_spec": [...]}``.
        num_integration_steps: Value for the config key, or None to omit it
            (exercises the default).
    """
    source_cfg: dict = {
        "dataset_type": "tisr",
        "variables": {
            "prognostic": None,
            "diagnostic": None,
            "dynamic_forcing": {"vars_2D": ["tisr"]},
        },
        **grid_source,
    }
    if num_integration_steps is not None:
        source_cfg["num_integration_steps"] = num_integration_steps

    return {
        "source": {"TISR": source_cfg},
        "timestep": "1h",  # -> self.dt = 1 hour, as the tests assume
        "forecast_len": 1,
        "start_datetime": "2020-06-01T00:00:00",
        "end_datetime": "2020-06-04T00:00:00",
    }


@pytest.fixture()
def tisr_dataset(rectangular_nc):
    """A real TISRDataset built through the actual BaseDataset.__init__ path (no mocking)."""
    data_config = _make_tisr_data_config({"latlon_grid_path": rectangular_nc})
    return TISRDataset(data_config, return_target=False)


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
        data_config = _make_tisr_data_config({"latlon_grid_path": rectangular_nc}, num_integration_steps=None)
        ds = TISRDataset(data_config, return_target=False)
        assert ds.num_integration_steps == 360

    def test_init_sets_latlon_grid_path(self, tisr_dataset, rectangular_nc):
        """__init__ must store latlon_grid_path from config."""
        assert tisr_dataset.latlon_grid_path == rectangular_nc

    def test_init_sets_static_metadata(self, tisr_dataset):
        """__init__ must set static_metadata with datetime_fmt."""
        assert tisr_dataset.static_metadata == {"datetime_fmt": "unix_ns"}

    def test_init_wrong_dataset_type_raises(self, rectangular_nc):
        """__init__ must assert-fail if dataset_type is not 'tisr'."""
        data_config = _make_tisr_data_config({"latlon_grid_path": rectangular_nc})
        data_config["source"]["TISR"]["dataset_type"] = "wrong"
        with pytest.raises(AssertionError):
            TISRDataset(data_config, return_target=False)

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
            tisr_dataset._extract_field("dynamic_forcing", pd.Timestamp("2020-06-21 12:00:00"), {})

    def test_extract_field_writes_to_sample(self, tisr_dataset):
        """_extract_field must compute TISR and store it in the sample dict."""
        sample = {}
        tisr_dataset._extract_field("dynamic_forcing", pd.Timestamp("2020-06-21 12:00:00"), sample)
        key = "TISR/dynamic_forcing/2d/tisr"
        assert key in sample, f"Expected key '{key}' in sample"
        assert sample[key].shape == (1, 1, 4, 8), f"Expected shape (1, 1, 4, 8), got {sample[key].shape}"
        assert (sample[key] >= 0).all(), "TISR values must be non-negative"


# =============================================================================
# GraphCast / JAX parity tests
# Each test calls pytest.importorskip("graphcast.solar_radiation") as its
# first line, so the test is skipped automatically when the package is absent.
# Install graphcast with: pip install --upgrade "https://github.com/deepmind/graphcast/archive/master.zip"
# =============================================================================

_PARITY_RTOL = 1e-6
_PARITY_ATOL = 1e-7


class TestTSIParity:
    """_get_tsi must return identical interpolated TSI values as the JAX reference."""

    def test_tsi_values_match(self):
        gc_solar = pytest.importorskip("graphcast.solar_radiation")
        tsi_data_jax = gc_solar.era5_tsi_data()
        times_torch, values_torch = _era5_tsi_data()

        jax_result = gc_solar.get_tsi(TIMESTAMPS, tsi_data_jax)
        torch_result = _get_tsi(TIMESTAMPS, times_torch, values_torch)

        np.testing.assert_allclose(
            to_np(torch_result),
            np.array(jax_result),
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
            err_msg="TSI interpolation diverges between implementations",
        )


class TestJ2000DaysParity:
    """_get_j2000_days must agree with the JAX reference."""

    @pytest.mark.parametrize("ts", TIMESTAMPS)
    def test_j2000_days_match(self, ts):
        gc_solar = pytest.importorskip("graphcast.solar_radiation")
        jax_val = float(gc_solar._get_j2000_days(pd.Timestamp(ts)))
        torch_val = float(_get_j2000_days(pd.Timestamp(ts)).item())
        assert abs(jax_val - torch_val) < 1e-4, f"J2000 days differ for {ts}: JAX={jax_val}, PyTorch={torch_val}"


class TestOrbitalParametersParity:
    """Every orbital parameter must match the JAX reference."""

    KEYS = [
        "sin_declination",
        "cos_declination",
        "eq_of_time_seconds",
        "solar_distance_au",
    ]

    @pytest.mark.parametrize("ts", TIMESTAMPS)
    def test_orbital_params_match(self, ts):
        gc_solar = pytest.importorskip("graphcast.solar_radiation")
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        j2000 = float(gc_solar._get_j2000_days(pd.Timestamp(ts)))
        jax_op = gc_solar._get_orbital_parameters(jnp.array(j2000))
        torch_op = _get_orbital_parameters(torch.tensor([j2000], dtype=torch.float64))

        for key in self.KEYS:
            jax_val = float(np.asarray(getattr(jax_op, key)))
            torch_val = float(torch_op[key].item())
            assert abs(jax_val - torch_val) < _PARITY_ATOL, (
                f"{key} mismatch at {ts}: JAX={jax_val:.6f}, PyTorch={torch_val:.6f}"
            )


class TestSolarTimeParity:
    """Solar time derived from PyTorch orbital params must match JAX."""

    def test_solar_time_matches_jax(self):
        gc_solar = pytest.importorskip("graphcast.solar_radiation")
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        for ts_str in TIMESTAMPS:
            j2000 = float(gc_solar._get_j2000_days(pd.Timestamp(ts_str)))
            jax_op = gc_solar._get_orbital_parameters(jnp.array(j2000))
            jax_solar_time = float(jax_op.rotational_phase + jax_op.eq_of_time_seconds / 86400.0)

            torch_op = _get_orbital_parameters(torch.tensor([j2000], dtype=torch.float64))
            torch_solar_time = _get_solar_time(torch_op["rotational_phase"], torch_op["eq_of_time_seconds"]).item()

            assert abs(jax_solar_time - torch_solar_time) < 1e-9, (
                f"Solar time mismatch at {ts_str}: JAX={jax_solar_time:.10f}, PyTorch={torch_solar_time:.10f}"
            )


class TestSolarGeometryParity:
    """cos_zenith (PyTorch) must match sin_altitude (JAX) — they are the same quantity."""

    def test_cosine_zenith_matches_jax_sin_altitude(self):
        gc_solar = pytest.importorskip("graphcast.solar_radiation")
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        ts = "2020-06-21 12:00:00"
        lat_deg = np.array([0.0, 45.0, -45.0, 90.0, -90.0])
        lon_deg = np.array([0.0, 90.0, -90.0, 180.0, 45.0])

        j2000 = float(gc_solar._get_j2000_days(pd.Timestamp(ts)))
        jax_op = gc_solar._get_orbital_parameters(jnp.array(j2000))

        jax_vals = np.asarray(
            gc_solar._get_solar_sin_altitude(
                jax_op,
                jnp.sin(jnp.radians(lat_deg)),
                jnp.cos(jnp.radians(lat_deg)),
                jnp.radians(lon_deg),
            )
        )
        jax_vals_clamped = np.maximum(jax_vals, 0.0)

        torch_op = _get_orbital_parameters(torch.tensor([[j2000]], dtype=torch.float64))
        solar_time = _get_solar_time(torch_op["rotational_phase"], torch_op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, torch.tensor(lon_deg, dtype=torch.float64))
        cos_zenith = _get_cosine_zenith_angle(
            cos_declination=torch_op["cos_declination"],
            sin_declination=torch_op["sin_declination"],
            latitude=torch.tensor(lat_deg, dtype=torch.float64),
            hour_angle=ha,
        )

        np.testing.assert_allclose(
            to_np(cos_zenith.squeeze()),
            jax_vals_clamped,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
            err_msg="cos_zenith (PyTorch) vs sin_altitude (JAX) mismatch",
        )


class TestInstantaneousFluxParity:
    """Instantaneous flux (W/m²) must match the JAX _get_radiation_flux."""

    @pytest.mark.parametrize("ts", TIMESTAMPS)
    def test_instantaneous_flux_matches_jax(self, ts):
        gc_solar = pytest.importorskip("graphcast.solar_radiation")
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        lat_deg = np.array([[-90.0], [-45.0], [0.0], [45.0], [90.0]])
        lon_deg = np.array([0.0, 90.0, 180.0, 270.0])

        tsi_data_jax = gc_solar.era5_tsi_data()
        tsi_scalar = float(gc_solar.get_tsi([ts], tsi_data_jax)[0])
        j2000 = float(gc_solar._get_j2000_days(pd.Timestamp(ts)))

        jax_flux = np.asarray(
            gc_solar._get_radiation_flux(
                j2000_days=jnp.array(j2000),
                sin_latitude=jnp.sin(jnp.radians(lat_deg)),
                cos_latitude=jnp.cos(jnp.radians(lat_deg)),
                longitude=jnp.radians(lon_deg),
                tsi=jnp.array(tsi_scalar),
            )
        )

        torch_op = _get_orbital_parameters(torch.tensor([[[j2000]]], dtype=torch.float64))
        lat_t = torch.tensor(lat_deg, dtype=torch.float64).unsqueeze(0)
        lon_t = torch.tensor(lon_deg, dtype=torch.float64).unsqueeze(0)
        solar_time = _get_solar_time(torch_op["rotational_phase"], torch_op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, lon_t)
        cz = _get_cosine_zenith_angle(torch_op["cos_declination"], torch_op["sin_declination"], lat_t, ha)
        solar_factor = 1.0 / torch_op["solar_distance_au"] ** 2
        torch_flux = _get_instantaneous_toa_tisr(
            tsi=torch.tensor([[[tsi_scalar]]], dtype=torch.float64),
            solar_factor=solar_factor,
            cos_zenith=cz,
        ).squeeze(0)

        np.testing.assert_allclose(
            to_np(torch_flux),
            jax_flux,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
            err_msg=f"Instantaneous TOA flux mismatch at {ts}",
        )


class TestIntegratedTISRParity:
    """Integrated TISR must match the JAX get_toa_incident_solar_radiation."""

    @pytest.fixture(autouse=True)
    def _force_float64(self, monkeypatch):
        # Parity is measured against float64 JAX, so run the port in float64
        # too — otherwise we'd be measuring the float32/float64 gap, not
        # algorithmic fidelity. Production code stays float32 (see functional tests above).
        monkeypatch.setattr("credit.datasets.gen_2.tisr._TORCH_DTYPE", torch.float64)

    @pytest.mark.parametrize("ts_str", TIMESTAMPS)
    def test_integrated_tisr_matches_jax(self, ts_str):
        gc_solar = pytest.importorskip("graphcast.solar_radiation")
        lat_deg = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
        lon_deg = np.array([0.0, 90.0, 180.0, 270.0])
        integration_period = pd.Timedelta(hours=1)
        num_steps = 360

        jax_result = np.asarray(
            gc_solar.get_toa_incident_solar_radiation(
                timestamps=[ts_str],
                latitude=lat_deg,
                longitude=lon_deg,
                integration_period=integration_period,
                num_integration_bins=num_steps,
                use_jit=False,
            )
        )[0]  # shape (lat, lon)

        t = pd.Timestamp(ts_str)
        ts = pd.date_range(
            end=t,
            periods=num_steps + 1,
            freq=integration_period / num_steps,
        )
        times_torch, tsi_values_torch = _era5_tsi_data()
        tsi = _get_tsi(ts, times_torch, tsi_values_torch).unsqueeze(-1).unsqueeze(-1)
        j2000_days = _get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
        op = _get_orbital_parameters(j2000_days)
        lat_t = torch.tensor(lat_deg, dtype=torch.float64).reshape(1, -1, 1)
        lon_t = torch.tensor(lon_deg, dtype=torch.float64).reshape(1, 1, -1)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, lon_t)
        cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
        solar_factor = 1.0 / op["solar_distance_au"] ** 2
        inst = _get_instantaneous_toa_tisr(tsi, solar_factor, cz)
        torch_result = to_np(_get_integrated_toa_tisr(inst, integration_period, num_steps))  # shape (lat, lon)

        np.testing.assert_allclose(
            torch_result,
            jax_result,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
            err_msg=f"Integrated TISR mismatch at {ts_str}",
        )
