"""
Tests for regional (WRF/downscaling) dataset classes.

Synthetic in-memory netCDF fixtures (via tmp_path) — no real data needed.
"""

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_times(n, start="1980-01-01", step_h=3):
    return np.array([np.datetime64(start) + np.timedelta64(step_h * i, "h") for i in range(n)])


def _make_combined_nc(path, ua_vars, surf_vars, n_time=10, n_lat=8, n_lon=8, n_level=3):
    """
    Write a netCDF with mixed-dim variables:
      upper-air: (time, level, lat, lon)
      surface:   (time, lat, lon)
    Both in the same file, mimicking Kyle's merged zarr layout.
    """
    times = _make_times(n_time)
    lats = np.linspace(30.0, 35.0, n_lat)
    lons = np.linspace(-110.0, -105.0, n_lon)
    levels = np.arange(n_level)
    rng = np.random.default_rng(0)

    data_vars = {}
    for v in ua_vars:
        data_vars[v] = xr.DataArray(
            rng.standard_normal((n_time, n_level, n_lat, n_lon)).astype("float32"),
            dims=["time", "level", "lat", "lon"],
        )
    for v in surf_vars:
        data_vars[v] = xr.DataArray(
            rng.standard_normal((n_time, n_lat, n_lon)).astype("float32"),
            dims=["time", "lat", "lon"],
        )
    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "level": levels, "lat": lats, "lon": lons},
    )
    ds.to_netcdf(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# WRF_Dataset tests
# ---------------------------------------------------------------------------


class TestWRFDataset:
    """Unit tests for WRF_Dataset (single-step, dual-domain)."""

    INT_UA = ["T", "U", "V"]
    BND_UA = ["T", "U"]
    BND_SURF = ["MSL", "T2"]

    @pytest.fixture
    def interior_file(self, tmp_path):
        # Interior: upper-air only (no surface, forcing, static, diag)
        return _make_combined_nc(tmp_path / "interior.nc", self.INT_UA, [], n_time=10)

    @pytest.fixture
    def boundary_file(self, tmp_path):
        # Boundary: UA + surface in one file (both extracted from same xr.Dataset)
        return _make_combined_nc(tmp_path / "boundary.nc", self.BND_UA, self.BND_SURF, n_time=10)

    @pytest.fixture
    def dataset(self, interior_file, boundary_file):
        from credit.datasets.wrf_singlestep import WRF_Dataset

        param_interior = {
            "varname_upper_air": self.INT_UA,
            "varname_surface": None,
            "varname_dyn_forcing": None,
            "varname_forcing": None,
            "varname_static": None,
            "varname_diagnostic": None,
            "filenames": [interior_file],
            "filename_surface": None,
            "filename_dyn_forcing": None,
            "filename_forcing": None,
            "filename_static": None,
            "filename_diagnostic": None,
            "history_len": 1,
            "forecast_len": 1,
        }
        param_outside = {
            "varname_upper_air": self.BND_UA,
            "varname_surface": self.BND_SURF,
            "filenames": [boundary_file],
            "filename_surface": boundary_file,  # flag: surface IS present (file is same combined file)
            "lead_time_periods": 1,
            "history_len": 1,
            "forecast_len": 0,
        }
        return WRF_Dataset(param_interior, param_outside)

    def test_import(self):
        from credit.datasets.wrf_singlestep import WRF_Dataset  # noqa: F401

    def test_init(self, dataset):
        assert dataset is not None

    def test_len_positive(self, dataset):
        assert len(dataset) > 0

    def test_len_formula(self, dataset):
        # total_seq_len overwritten by boundary block (1+0=1) so len = 10-1+1 = 10
        assert len(dataset) == 10

    def test_getitem_returns_dict(self, dataset):
        sample = dataset[0]
        assert isinstance(sample, dict)

    def test_getitem_keys(self, dataset):
        sample = dataset[0]
        assert "WRF_input" in sample
        assert "WRF_target" in sample
        assert "boundary_input" in sample
        assert "time_encode" in sample
        assert "datetime_index" in sample

    def test_getitem_wrf_input_is_dataset(self, dataset):
        sample = dataset[0]
        assert isinstance(sample["WRF_input"], xr.Dataset)

    def test_getitem_wrf_target_is_dataset(self, dataset):
        sample = dataset[0]
        assert isinstance(sample["WRF_target"], xr.Dataset)

    def test_getitem_boundary_is_dataset(self, dataset):
        sample = dataset[0]
        assert isinstance(sample["boundary_input"], xr.Dataset)

    def test_getitem_wrf_input_has_vars(self, dataset):
        sample = dataset[0]
        for v in self.INT_UA:
            assert v in sample["WRF_input"].data_vars

    def test_getitem_boundary_has_ua_vars(self, dataset):
        sample = dataset[0]
        for v in self.BND_UA:
            assert v in sample["boundary_input"].data_vars

    def test_getitem_boundary_has_surf_vars(self, dataset):
        sample = dataset[0]
        for v in self.BND_SURF:
            assert v in sample["boundary_input"].data_vars

    def test_getitem_time_encode_nonempty(self, dataset):
        sample = dataset[0]
        assert sample["time_encode"].size > 0

    def test_getitem_index_assigned(self, dataset):
        sample = dataset[0]
        assert sample["index"] == 0

    def test_getitem_last_index(self, dataset):
        last = len(dataset) - 1
        sample = dataset[last]
        assert sample["index"] == last

    def test_multiple_interior_files(self, tmp_path, boundary_file):
        """Two interior files → length doubles."""
        from credit.datasets.wrf_singlestep import WRF_Dataset

        f1 = _make_combined_nc(tmp_path / "a.nc", self.INT_UA, [], n_time=10)
        f2 = _make_combined_nc(tmp_path / "b.nc", self.INT_UA, [], n_time=10)
        param_interior = {
            "varname_upper_air": self.INT_UA,
            "varname_surface": None,
            "varname_dyn_forcing": None,
            "varname_forcing": None,
            "varname_static": None,
            "varname_diagnostic": None,
            "filenames": [f1, f2],
            "filename_surface": None,
            "filename_dyn_forcing": None,
            "filename_forcing": None,
            "filename_static": None,
            "filename_diagnostic": None,
            "history_len": 1,
            "forecast_len": 1,
        }
        param_outside = {
            "varname_upper_air": self.BND_UA,
            "varname_surface": self.BND_SURF,
            "filenames": [boundary_file],
            "filename_surface": boundary_file,
            "lead_time_periods": 1,
            "history_len": 1,
            "forecast_len": 0,
        }
        ds = WRF_Dataset(param_interior, param_outside)
        # each file: 10 - 1 + 1 = 10 (total_seq_len=1 from boundary block), two files → 20
        assert len(ds) == 20


# ---------------------------------------------------------------------------
# Dscale_Dataset tests
# ---------------------------------------------------------------------------


class TestDscaleDataset:
    """Unit tests for Dscale_Dataset (single-step, HR/LR)."""

    HR_UA = ["T", "Q"]
    LR_UA = ["T", "U"]
    LR_SURF = ["MSL"]

    @pytest.fixture
    def hr_file(self, tmp_path):
        return _make_combined_nc(tmp_path / "hr.nc", self.HR_UA, [], n_time=8)

    @pytest.fixture
    def lr_file(self, tmp_path):
        return _make_combined_nc(tmp_path / "lr.nc", self.LR_UA, self.LR_SURF, n_time=8)

    @pytest.fixture
    def dataset(self, hr_file, lr_file):
        from credit.datasets.dscale_singlestep import Dscale_Dataset

        param_HR = {
            "varname_upper_air": self.HR_UA,
            "varname_surface": None,
            "varname_dyn_forcing": None,
            "varname_forcing": None,
            "varname_static": None,
            "varname_diagnostic": None,
            "filenames": [hr_file],
            "filename_surface": None,
            "filename_dyn_forcing": None,
            "filename_forcing": None,
            "filename_static": None,
            "filename_diagnostic": None,
            "history_len": 1,
            "forecast_len": 1,
            "levels": 3,
            "level_pick": None,
        }
        param_LR = {
            "varname_upper_air": self.LR_UA,
            "varname_surface": self.LR_SURF,
            "filenames": [lr_file],
            "filename_surface": lr_file,  # flag: surface present
            "lead_time_periods": 1,
            "history_len": 1,
            "forecast_len": 0,
            "levels": 3,
            "level_pick": None,
        }
        return Dscale_Dataset(param_HR, param_LR)

    def test_import(self):
        from credit.datasets.dscale_singlestep import Dscale_Dataset  # noqa: F401

    def test_init(self, dataset):
        assert dataset is not None

    def test_len_positive(self, dataset):
        assert len(dataset) > 0

    def test_len_formula(self, dataset):
        # total_seq_len overwritten by LR block (1+0=1), so len = 8-1+1 = 8
        assert len(dataset) == 8

    def test_getitem_returns_dict(self, dataset):
        sample = dataset[0]
        assert isinstance(sample, dict)

    def test_getitem_index_assigned(self, dataset):
        sample = dataset[0]
        assert sample["index"] == 0

    def test_getitem_last_index(self, dataset):
        last = len(dataset) - 1
        sample = dataset[last]
        assert sample["index"] == last


# ---------------------------------------------------------------------------
# WRF_MultiStep tests
# ---------------------------------------------------------------------------


class TestWRFMultiStep:
    """Smoke tests for WRF_MultiStep (multi-step rollout dataset)."""

    INT_UA = ["T", "U"]
    BND_UA = ["T"]
    BND_SURF = ["MSL"]

    def test_import(self):
        from credit.datasets.wrf_multistep import WRF_MultiStep  # noqa: F401

    def test_init_and_len(self, tmp_path):
        from credit.datasets.wrf_multistep import WRF_MultiStep

        int_file = _make_combined_nc(tmp_path / "int.nc", self.INT_UA, [], n_time=12)
        bnd_file = _make_combined_nc(tmp_path / "bnd.nc", self.BND_UA, self.BND_SURF, n_time=12)

        param_interior = {
            "varname_upper_air": self.INT_UA,
            "varname_surface": None,
            "varname_dyn_forcing": None,
            "varname_forcing": None,
            "varname_static": None,
            "varname_diagnostic": None,
            "filenames": [int_file],
            "filename_surface": None,
            "filename_dyn_forcing": None,
            "filename_forcing": None,
            "filename_static": None,
            "filename_diagnostic": None,
            "history_len": 1,
            "forecast_len": 2,
        }
        param_outside = {
            "varname_upper_air": self.BND_UA,
            "varname_surface": self.BND_SURF,
            "filenames": [bnd_file],
            "filename_surface": bnd_file,
            "lead_time_periods": 1,
            "lead_time_periods_outside": 1,
            "history_len": 1,
            "forecast_len": 0,
        }
        ds = WRF_MultiStep(param_interior, param_outside)
        assert ds is not None
        assert len(ds) > 0
