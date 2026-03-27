"""
Tests for credit.verification.deterministic and applications.eval_weatherbench.

Coverage:
  - Unit tests for rmse, bias, acc with known mathematical results
  - Regional breakdown consistency
  - CSV fast path (integration, uses real 2020 metrics data)
  - NetCDF full path (integration, uses synthetic forecast/ERA5 netCDFs)
  - Climatology loading and ACC alignment helpers
"""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from credit.verification.deterministic import (
    acc,
    acc_by_region,
    bias,
    deterministic_scores,
    latitude_slices,
    latitude_weights,
    rmse,
    rmse_by_region,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LAT = np.linspace(-90, 90, 37)
LON = np.linspace(0, 360, 72, endpoint=False)
TIME = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")


def _make_da(data, name="x"):
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": TIME, "latitude": LAT, "longitude": LON},
        name=name,
    )


def _zeros():
    return _make_da(np.zeros((3, 37, 72)))


def _ones():
    return _make_da(np.ones((3, 37, 72)))


def _constant(c):
    return _make_da(np.full((3, 37, 72), c))


# ---------------------------------------------------------------------------
# Unit tests: rmse
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_zero_when_perfect(self):
        da = _ones()
        assert rmse(da, da) == pytest.approx(0.0, abs=1e-6)

    def test_known_uniform_error(self):
        """When pred = truth + c everywhere, RMSE = c."""
        truth = _zeros()
        pred = _constant(3.0)
        assert rmse(pred, truth) == pytest.approx(3.0, rel=1e-4)

    def test_symmetric(self):
        """RMSE(a, b) == RMSE(b, a)."""
        rng = np.random.default_rng(0)
        a = _make_da(rng.standard_normal((3, 37, 72)))
        b = _make_da(rng.standard_normal((3, 37, 72)))
        assert rmse(a, b) == pytest.approx(rmse(b, a), rel=1e-6)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        a = _make_da(rng.standard_normal((3, 37, 72)))
        b = _make_da(rng.standard_normal((3, 37, 72)))
        assert rmse(a, b) >= 0.0

    def test_per_timestep_shape(self):
        a = _make_da(np.random.default_rng(2).standard_normal((3, 37, 72)))
        b = _make_da(np.zeros((3, 37, 72)))
        result = rmse(a, b, time_mean=False)
        assert result.dims == ("time",)
        assert len(result) == 3

    def test_latitude_weighting_matters(self):
        """Polar errors should count less than equatorial errors."""
        truth = _zeros()
        # Error only at poles (high latitudes)
        data_pole = np.zeros((3, 37, 72))
        data_pole[:, :3, :] = 10.0  # south pole rows
        data_pole[:, -3:, :] = 10.0  # north pole rows
        pred_pole = _make_da(data_pole)

        # Error only at equator
        data_equator = np.zeros((3, 37, 72))
        mid = 37 // 2
        data_equator[:, mid - 2 : mid + 2, :] = 10.0
        pred_equator = _make_da(data_equator)

        rmse_pole = rmse(pred_pole, truth)
        rmse_equator = rmse(pred_equator, truth)
        assert rmse_equator > rmse_pole


# ---------------------------------------------------------------------------
# Unit tests: bias
# ---------------------------------------------------------------------------


class TestBias:
    def test_zero_when_perfect(self):
        da = _ones()
        assert bias(da, da) == pytest.approx(0.0, abs=1e-6)

    def test_positive_bias(self):
        pred = _constant(5.0)
        truth = _constant(2.0)
        assert bias(pred, truth) == pytest.approx(3.0, rel=1e-4)

    def test_negative_bias(self):
        pred = _constant(1.0)
        truth = _constant(4.0)
        assert bias(pred, truth) == pytest.approx(-3.0, rel=1e-4)

    def test_antisymmetric(self):
        """bias(a, b) == -bias(b, a)."""
        rng = np.random.default_rng(3)
        a = _make_da(rng.standard_normal((3, 37, 72)))
        b = _make_da(rng.standard_normal((3, 37, 72)))
        assert bias(a, b) == pytest.approx(-bias(b, a), rel=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: acc
# ---------------------------------------------------------------------------


class TestACC:
    def test_perfect_anomaly_correlation(self):
        """If pred anomaly == truth anomaly, ACC = 1."""
        clim = _zeros()
        signal = _make_da(np.random.default_rng(4).standard_normal((3, 37, 72)))
        assert acc(signal, signal, clim) == pytest.approx(1.0, abs=1e-5)

    def test_anti_correlated(self):
        """If pred anomaly == -truth anomaly, ACC = -1."""
        clim = _zeros()
        signal = _make_da(np.random.default_rng(5).standard_normal((3, 37, 72)))
        assert acc(signal, -signal, clim) == pytest.approx(-1.0, abs=1e-5)

    def test_climatology_shifted_out(self):
        """ACC should be 1 regardless of climatology offset if anomalies match."""
        clim = _constant(100.0)
        signal = _make_da(np.random.default_rng(6).standard_normal((3, 37, 72)))
        pred = signal + clim
        truth = signal + clim
        assert acc(pred, truth, clim) == pytest.approx(1.0, abs=1e-5)

    def test_range_minus_one_to_one(self):
        rng = np.random.default_rng(7)
        pred = _make_da(rng.standard_normal((3, 37, 72)))
        truth = _make_da(rng.standard_normal((3, 37, 72)))
        clim = _zeros()
        result = acc(pred, truth, clim)
        assert -1.0 <= result <= 1.0

    def test_per_timestep_shape(self):
        clim = _zeros()
        pred = _make_da(np.random.default_rng(8).standard_normal((3, 37, 72)))
        truth = _make_da(np.random.default_rng(9).standard_normal((3, 37, 72)))
        result = acc(pred, truth, clim, time_mean=False)
        assert result.dims == ("time",)
        assert len(result) == 3

    def test_acc_vs_pearson_differ_with_bias(self):
        """ACC and Pearson give different results when there is a mean bias.

        With clim=0, a large additive bias inflates the denominator of ACC so
        true_acc << 1.  Pearson subtracts the spatial mean first, so it stays
        close to 1 even when a uniform offset is present.
        """
        rng = np.random.default_rng(10)
        signal = rng.standard_normal((3, 37, 72))
        truth = _make_da(signal)
        pred = _make_da(signal + 50.0)  # large uniform bias
        clim = _zeros()

        # True ACC with clim=0 is severely degraded by the 50-unit bias
        true_acc = acc(pred, truth, clim)

        # Pearson removes the spatial mean so is unaffected by uniform bias
        pred_anom = pred - pred.mean(dim=["latitude", "longitude"])
        truth_anom = truth - truth.mean(dim=["latitude", "longitude"])
        w = np.cos(np.deg2rad(truth.latitude))
        num = (w * pred_anom * truth_anom).sum(dim=["latitude", "longitude"])
        denom = np.sqrt(
            (w * pred_anom**2).sum(dim=["latitude", "longitude"])
            * (w * truth_anom**2).sum(dim=["latitude", "longitude"])
        )
        pearson = float((num / (denom + 1e-12)).mean())

        # Pearson ≈ 1 (bias is uniform, spatial structure unchanged)
        assert pearson == pytest.approx(1.0, abs=1e-3)
        # True WB2 ACC is much lower than Pearson when there is a mean bias
        assert true_acc < pearson - 0.5


# ---------------------------------------------------------------------------
# Unit tests: regional breakdown
# ---------------------------------------------------------------------------


class TestRegions:
    def test_rmse_by_region_keys(self):
        pred = _make_da(np.random.default_rng(11).standard_normal((3, 37, 72)))
        truth = _zeros()
        result = rmse_by_region(pred, truth)
        for region in latitude_slices:
            assert f"rmse_{region}" in result

    def test_rmse_global_consistent(self):
        """rmse_by_region global should match rmse() global."""
        pred = _make_da(np.random.default_rng(12).standard_normal((3, 37, 72)))
        truth = _zeros()
        by_region = rmse_by_region(pred, truth)
        direct = rmse(pred, truth)
        assert by_region["rmse_global"] == pytest.approx(direct, rel=1e-4)

    def test_acc_by_region_keys(self):
        rng = np.random.default_rng(13)
        pred = _make_da(rng.standard_normal((3, 37, 72)))
        truth = _make_da(rng.standard_normal((3, 37, 72)))
        clim = _zeros()
        result = acc_by_region(pred, truth, clim)
        for region in latitude_slices:
            assert f"acc_{region}" in result

    def test_deterministic_scores_all_keys(self):
        rng = np.random.default_rng(14)
        pred = _make_da(rng.standard_normal((3, 37, 72)))
        truth = _make_da(rng.standard_normal((3, 37, 72)))
        clim = _zeros()
        result = deterministic_scores(pred, truth, da_clim=clim)
        for region in latitude_slices:
            assert f"rmse_{region}" in result
            assert f"bias_{region}" in result
            assert f"acc_{region}" in result

    def test_deterministic_scores_no_clim(self):
        """Without climatology, ACC keys should be absent."""
        rng = np.random.default_rng(15)
        pred = _make_da(rng.standard_normal((3, 37, 72)))
        truth = _make_da(rng.standard_normal((3, 37, 72)))
        result = deterministic_scores(pred, truth, da_clim=None)
        for region in latitude_slices:
            assert f"rmse_{region}" in result
            assert f"acc_{region}" not in result


# ---------------------------------------------------------------------------
# Unit tests: latitude_weights
# ---------------------------------------------------------------------------


class TestLatitudeWeights:
    def test_equator_heavier_than_poles(self):
        da = _zeros()
        w = latitude_weights(da)
        equator_idx = np.argmin(np.abs(LAT))
        pole_idx = 0
        assert float(w[equator_idx]) > float(w[pole_idx])

    def test_mean_is_one(self):
        da = _zeros()
        w = latitude_weights(da)
        assert float(w.mean()) == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Integration test: CSV fast path (uses real 2020 data if available)
# ---------------------------------------------------------------------------

REAL_CSV_DIR = "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/single_step/metrics/"


@pytest.mark.skipif(
    not os.path.isdir(REAL_CSV_DIR),
    reason="Real metrics CSVs not available",
)
class TestCSVFastPath:
    def test_loads_and_aggregates(self):
        from applications.eval_weatherbench import load_csv_metrics

        df = load_csv_metrics(REAL_CSV_DIR)
        assert len(df) > 0
        assert "forecast_step" in df.columns

    def test_scores_have_expected_columns(self):
        from applications.eval_weatherbench import csv_to_wb2_scores, load_csv_metrics

        df = load_csv_metrics(REAL_CSV_DIR)
        scores = csv_to_wb2_scores(df, lead_time_hours=6)
        assert "lead_time_hours" in scores.columns
        assert "rmse_Z500" in scores.columns
        assert "acc_Z500" in scores.columns
        assert "rmse_t2m" in scores.columns

    def test_z500_rmse_day5_reasonable(self):
        """Z500 RMSE at day 5 should be between 200 and 500 m² for a good model."""
        from applications.eval_weatherbench import csv_to_wb2_scores, load_csv_metrics

        df = load_csv_metrics(REAL_CSV_DIR)
        scores = csv_to_wb2_scores(df, lead_time_hours=6)
        day5 = scores[scores["forecast_step"] == 20]
        assert len(day5) == 1
        rmse_val = day5["rmse_Z500"].values[0]
        assert 200 < rmse_val < 500, f"Z500 RMSE at day 5 = {rmse_val:.1f}, outside expected range"

    def test_t2m_rmse_day5_reasonable(self):
        """t2m RMSE at day 5 should be between 0.5 and 4 K for a good model."""
        from applications.eval_weatherbench import csv_to_wb2_scores, load_csv_metrics

        df = load_csv_metrics(REAL_CSV_DIR)
        scores = csv_to_wb2_scores(df, lead_time_hours=6)
        day5 = scores[scores["forecast_step"] == 20]
        rmse_val = day5["rmse_t2m"].values[0]
        assert 0.5 < rmse_val < 4.0, f"t2m RMSE at day 5 = {rmse_val:.3f} K, outside expected range"

    def test_acc_z500_monotone_decrease(self):
        """Z500 ACC should decrease with lead time."""
        from applications.eval_weatherbench import csv_to_wb2_scores, load_csv_metrics

        df = load_csv_metrics(REAL_CSV_DIR)
        scores = csv_to_wb2_scores(df, lead_time_hours=6)
        acc_vals = scores["acc_Z500"].values[:20]  # first 20 steps
        # allow small non-monotone wobbles (noise), check overall trend
        assert acc_vals[0] > acc_vals[-1], "ACC should decrease from step 1 to step 20"
        assert acc_vals[0] > 0.99, "ACC at step 1 should be > 0.99"

    def test_rmse_z500_monotone_increase(self):
        """Z500 RMSE should increase with lead time (at least for first 10 days)."""
        from applications.eval_weatherbench import csv_to_wb2_scores, load_csv_metrics

        df = load_csv_metrics(REAL_CSV_DIR)
        scores = csv_to_wb2_scores(df, lead_time_hours=6)
        rmse_vals = scores["rmse_Z500"].values[:20]
        assert rmse_vals[-1] > rmse_vals[0], "RMSE should increase from step 1 to step 20"


# ---------------------------------------------------------------------------
# Integration test: NetCDF full path (synthetic data)
# ---------------------------------------------------------------------------


def _make_synthetic_forecast_nc(path, init_time, step, lat, lon, rng):
    """Write a minimal CREDIT-format forecast netCDF."""
    valid_time = pd.Timestamp(init_time) + pd.Timedelta(hours=step * 6)
    level = np.arange(18)

    ds = xr.Dataset(
        {
            "Z500": (["time", "latitude", "longitude"], rng.standard_normal((1, len(lat), len(lon))).astype("float32")),
            "T500": (
                ["time", "latitude", "longitude"],
                (280 + rng.standard_normal((1, len(lat), len(lon)))).astype("float32"),
            ),
            "t2m": (
                ["time", "latitude", "longitude"],
                (290 + rng.standard_normal((1, len(lat), len(lon)))).astype("float32"),
            ),
            "SP": (
                ["time", "latitude", "longitude"],
                (101325 + 100 * rng.standard_normal((1, len(lat), len(lon)))).astype("float32"),
            ),
            "U": (
                ["time", "level", "latitude", "longitude"],
                rng.standard_normal((1, len(level), len(lat), len(lon))).astype("float32"),
            ),
        },
        coords={
            "time": [np.datetime64(valid_time)],
            "level": level.astype("float32"),
            "latitude": lat.astype("float32"),
            "longitude": lon.astype("float32"),
        },
        attrs={"Conventions": "CF-1.11"},
    )
    ds.to_netcdf(path)
    return ds


def _make_synthetic_era5_zarr(path, times, lat, lon, rng):
    """Write a minimal ERA5-format zarr store."""
    level = np.arange(18).astype("float32")
    ds = xr.Dataset(
        {
            "u_component_of_wind": (
                ["time", "level", "latitude", "longitude"],
                rng.standard_normal((len(times), len(level), len(lat), len(lon))).astype("float32"),
            ),
            "VAR_2T": (
                ["time", "latitude", "longitude"],
                (290 + rng.standard_normal((len(times), len(lat), len(lon)))).astype("float32"),
            ),
            "SP": (
                ["time", "latitude", "longitude"],
                (101325 + 100 * rng.standard_normal((len(times), len(lat), len(lon)))).astype("float32"),
            ),
        },
        coords={
            "time": times,
            "level": level,
            "latitude": lat.astype("float32"),
            "longitude": lon.astype("float32"),
        },
    )
    ds.to_zarr(path, mode="w")
    return ds


class TestNetCDFPath:
    """Integration tests using synthetic forecast netCDFs and ERA5 zarr."""

    @pytest.fixture
    def synthetic_data(self, tmp_path):
        rng = np.random.default_rng(42)
        lat = np.linspace(-90, 90, 19)  # coarse for speed
        lon = np.linspace(0, 360, 36, endpoint=False)

        init = "2020-06-01"
        steps = [4, 8]  # day 1 and day 2

        # build forecast dir: forecast_dir/2020-06-01T00Z/pred_*_SSS.nc
        init_dir = tmp_path / "forecasts" / "2020-06-01T00Z"
        init_dir.mkdir(parents=True)
        for step in steps:
            nc_path = init_dir / f"pred_2020-06-01T00Z_{step:03d}.nc"
            _make_synthetic_forecast_nc(str(nc_path), init, step, lat, lon, rng)

        # build ERA5: one zarr covering all valid times
        valid_times = [np.datetime64(pd.Timestamp(init) + pd.Timedelta(hours=s * 6)) for s in steps]
        era5_path = str(tmp_path / "era5_2020.zarr")
        _make_synthetic_era5_zarr(era5_path, valid_times, lat, lon, rng)

        return {
            "forecast_dir": str(tmp_path / "forecasts"),
            "era5_glob": era5_path,
            "lat": lat,
            "lon": lon,
            "steps": steps,
        }

    def test_compute_netcdf_scores_runs(self, synthetic_data):
        from applications.eval_weatherbench import compute_netcdf_scores

        scores = compute_netcdf_scores(
            synthetic_data["forecast_dir"],
            synthetic_data["era5_glob"],
            clim_path=None,
            lead_time_hours=6,
        )
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) == len(synthetic_data["steps"])

    def test_scores_have_lead_time_column(self, synthetic_data):
        from applications.eval_weatherbench import compute_netcdf_scores

        scores = compute_netcdf_scores(
            synthetic_data["forecast_dir"],
            synthetic_data["era5_glob"],
            clim_path=None,
            lead_time_hours=6,
        )
        assert "lead_time_hours" in scores.columns
        assert list(scores["lead_time_hours"]) == [s * 6 for s in synthetic_data["steps"]]

    def test_rmse_columns_present_and_finite(self, synthetic_data):
        from applications.eval_weatherbench import compute_netcdf_scores

        scores = compute_netcdf_scores(
            synthetic_data["forecast_dir"],
            synthetic_data["era5_glob"],
            clim_path=None,
            lead_time_hours=6,
        )
        for region in latitude_slices:
            col = f"rmse_t2m_{region}"
            if col in scores.columns:
                assert scores[col].notna().all(), f"{col} has NaNs"
                assert (scores[col] >= 0).all(), f"{col} has negative values"

    def test_rmse_zero_when_perfect(self):
        """If forecast == ERA5, all RMSEs should be 0."""
        from applications.eval_weatherbench import _add_scores

        rng = np.random.default_rng(99)
        lat = np.linspace(-90, 90, 19)
        lon = np.linspace(0, 360, 36, endpoint=False)
        time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")

        data = rng.standard_normal((2, 19, 36)).astype("float32")
        da = xr.DataArray(
            data, dims=["time", "latitude", "longitude"], coords={"time": time, "latitude": lat, "longitude": lon}
        )

        row = {}
        _add_scores(row, da, da, "test_var")
        assert row["rmse_test_var"] == pytest.approx(0.0, abs=1e-5)
        assert row["bias_test_var"] == pytest.approx(0.0, abs=1e-5)

    def test_acc_one_when_perfect_with_clim(self):
        """If forecast == ERA5 and we have climatology, ACC should be 1."""
        from applications.eval_weatherbench import _add_scores

        rng = np.random.default_rng(100)
        lat = np.linspace(-90, 90, 19)
        lon = np.linspace(0, 360, 36, endpoint=False)
        time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")

        signal = rng.standard_normal((2, 19, 36)).astype("float32")
        clim_data = rng.standard_normal((2, 19, 36)).astype("float32")

        da_signal = xr.DataArray(
            signal, dims=["time", "latitude", "longitude"], coords={"time": time, "latitude": lat, "longitude": lon}
        )
        da_clim = xr.DataArray(
            clim_data, dims=["time", "latitude", "longitude"], coords={"time": time, "latitude": lat, "longitude": lon}
        )

        row = {}
        _add_scores(row, da_signal, da_signal, "test_var", da_clim=da_clim)
        assert row["acc_test_var"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Integration test: climatology loading
# ---------------------------------------------------------------------------

REAL_CLIM_PATH = "/glade/campaign/cisl/aiml/akn7/CREDIT_CESM/VERIF/ERA5_clim/ERA5_clim_1990_2019_6h_cesm_interp.nc"


@pytest.mark.skipif(
    not os.path.exists(REAL_CLIM_PATH),
    reason="ERA5 climatology not available on this system",
)
class TestClimatologyLoading:
    def test_loads_without_error(self):
        from applications.eval_weatherbench import load_climatology

        ds = load_climatology()
        assert ds is not None

    def test_has_expected_dims(self):
        from applications.eval_weatherbench import load_climatology

        ds = load_climatology()
        assert "dayofyear" in ds.dims
        assert "hour" in ds.dims
        assert "latitude" in ds.dims
        assert "longitude" in ds.dims

    def test_select_clim_aligns_to_times(self):
        from applications.eval_weatherbench import load_climatology, select_clim

        ds = load_climatology()
        var = "2m_temperature"
        if var not in ds:
            pytest.skip(f"{var} not in climatology dataset")

        times = np.array(["2020-01-15T00", "2020-06-21T12", "2020-12-31T18"], dtype="datetime64[ns]")
        da = select_clim(ds, times, var)
        assert da.dims[0] == "time"
        assert len(da.time) == 3

    def test_clim_values_physically_reasonable(self):
        """2m temperature climatology should be between 180 K and 330 K."""
        from applications.eval_weatherbench import load_climatology

        ds = load_climatology()
        var = "2m_temperature"
        if var not in ds:
            pytest.skip(f"{var} not in climatology dataset")

        sample = float(ds[var].isel(dayofyear=0, hour=0).mean())
        assert 180 < sample < 330, f"Mean 2m_temperature = {sample:.1f} K, seems wrong"
