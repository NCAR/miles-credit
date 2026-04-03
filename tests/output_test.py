"""
output_test.py
-------------------------------------------------------
Unit tests for credit.output:
    - load_metadata()
    - make_xarray()
"""

import textwrap
from datetime import datetime

import numpy as np
import pytest

from credit.output import load_metadata, make_xarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conf_with_metadata(value):
    """Minimal config dict with only the predict.metadata key set."""
    return {"predict": {"metadata": value}}


def _make_pred_conf(n_vars=2, n_levels=3, n_surface=2, n_diag=1, height=4, width=8):
    """
    Build a minimal (conf, pred) pair for make_xarray.

    Layout of pred channel dim: (n_vars * n_levels) upper-air + (n_surface + n_diag) single-level.
    """
    upper_ch = n_vars * n_levels
    single_ch = n_surface + n_diag
    total_ch = upper_ch + single_ch

    pred = np.random.default_rng(0).standard_normal((1, total_ch, 1, height, width)).astype(np.float32)

    conf = {
        "model": {"levels": n_levels},
        "data": {
            "variables": [f"UA{i}" for i in range(n_vars)],
            "surface_variables": [f"SFC{i}" for i in range(n_surface)],
            "diagnostic_variables": [f"DIAG{i}" for i in range(n_diag)],
        },
    }
    return pred, conf


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_bare_filename_resolves_to_package(self):
        """A bare filename like 'era5.yaml' is resolved from credit.metadata."""
        conf = _conf_with_metadata("era5.yaml")
        meta = load_metadata(conf)
        assert isinstance(meta, dict)
        # era5.yaml must contain at least one variable entry
        assert len(meta) > 0

    def test_bare_filename_cam(self):
        """cam.yaml is another bundled metadata file."""
        conf = _conf_with_metadata("cam.yaml")
        meta = load_metadata(conf)
        assert isinstance(meta, dict)

    def test_full_path_custom_file(self, tmp_path):
        """A full path bypasses the package lookup and reads the file directly."""
        custom = tmp_path / "custom_meta.yaml"
        custom.write_text(
            textwrap.dedent("""\
            MYVAR:
                long_name: "My variable"
                units: "K"
        """)
        )
        conf = _conf_with_metadata(str(custom))
        meta = load_metadata(conf)
        assert "MYVAR" in meta
        assert meta["MYVAR"]["units"] == "K"

    def test_env_var_in_full_path(self, tmp_path, monkeypatch):
        """$ENV_VAR expansion works for full paths."""
        custom = tmp_path / "env_meta.yaml"
        custom.write_text("ENVVAR:\n  units: Pa\n")
        monkeypatch.setenv("TEST_META_DIR", str(tmp_path))
        conf = _conf_with_metadata("$TEST_META_DIR/env_meta.yaml")
        meta = load_metadata(conf)
        assert "ENVVAR" in meta

    def test_false_metadata_defaults_to_era5(self):
        """When metadata is False, era5.yaml is loaded as the default."""
        conf = _conf_with_metadata(False)
        meta = load_metadata(conf)
        assert isinstance(meta, dict)
        assert len(meta) > 0

    def test_missing_file_raises(self):
        """Referencing a non-existent bare filename raises FileNotFoundError."""
        conf = _conf_with_metadata("does_not_exist.yaml")
        with pytest.raises(FileNotFoundError):
            load_metadata(conf)

    def test_missing_full_path_raises(self):
        """Referencing a non-existent full path raises FileNotFoundError."""
        conf = _conf_with_metadata("/tmp/definitely_not_here_abc123.yaml")
        with pytest.raises(FileNotFoundError):
            load_metadata(conf)


# ---------------------------------------------------------------------------
# make_xarray
# ---------------------------------------------------------------------------


class TestMakeXarray:
    def test_returns_tuple_when_surface_vars_present(self):
        """make_xarray returns (upper_air, single_level) when surface_variables is non-empty."""
        pred, conf = _make_pred_conf(n_vars=2, n_levels=3, n_surface=2, n_diag=1)
        lat = np.linspace(-90, 90, 4)
        lon = np.linspace(0, 360, 8, endpoint=False)
        dt = datetime(2020, 1, 1)
        result = make_xarray(pred, dt, lat, lon, conf)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_single_array_when_no_surface_vars(self):
        """make_xarray returns only the upper-air DataArray when no surface/diag variables."""
        pred, conf = _make_pred_conf(n_vars=2, n_levels=3, n_surface=0, n_diag=0)
        lat = np.linspace(-90, 90, 4)
        lon = np.linspace(0, 360, 8, endpoint=False)
        dt = datetime(2020, 1, 1)
        result = make_xarray(pred, dt, lat, lon, conf)
        # Should be a single DataArray, not a tuple
        import xarray as xr

        assert isinstance(result, xr.DataArray)

    def test_upper_air_shape(self):
        n_vars, n_levels, height, width = 3, 4, 5, 10
        pred, conf = _make_pred_conf(
            n_vars=n_vars, n_levels=n_levels, n_surface=1, n_diag=0, height=height, width=width
        )
        lat = np.linspace(-90, 90, height)
        lon = np.linspace(0, 360, width, endpoint=False)
        upper, _ = make_xarray(pred, datetime(2021, 6, 1), lat, lon, conf)
        assert upper.dims == ("time", "vars", "level", "latitude", "longitude")
        assert upper.shape == (1, n_vars, n_levels, height, width)

    def test_single_level_shape(self):
        n_surface, n_diag, height, width = 2, 1, 5, 10
        pred, conf = _make_pred_conf(
            n_vars=2, n_levels=3, n_surface=n_surface, n_diag=n_diag, height=height, width=width
        )
        lat = np.linspace(-90, 90, height)
        lon = np.linspace(0, 360, width, endpoint=False)
        _, single = make_xarray(pred, datetime(2021, 6, 1), lat, lon, conf)
        assert single.dims == ("time", "vars", "latitude", "longitude")
        assert single.shape == (1, n_surface + n_diag, height, width)

    def test_upper_air_coords(self):
        """vars and level coordinates carry the right labels."""
        pred, conf = _make_pred_conf(n_vars=2, n_levels=3, n_surface=1, n_diag=0)
        lat = np.linspace(-90, 90, 4)
        lon = np.linspace(0, 360, 8, endpoint=False)
        upper, _ = make_xarray(pred, datetime(2022, 3, 15), lat, lon, conf)
        assert list(upper.coords["vars"].values) == conf["data"]["variables"]
        assert len(upper.coords["level"]) == 3

    def test_single_level_coords(self):
        """Single-level vars coordinate contains surface + diagnostic variable names."""
        pred, conf = _make_pred_conf(n_vars=2, n_levels=3, n_surface=2, n_diag=1)
        lat = np.linspace(-90, 90, 4)
        lon = np.linspace(0, 360, 8, endpoint=False)
        _, single = make_xarray(pred, datetime(2022, 3, 15), lat, lon, conf)
        expected_vars = conf["data"]["surface_variables"] + conf["data"]["diagnostic_variables"]
        assert list(single.coords["vars"].values) == expected_vars

    def test_time_coordinate(self):
        """The time coordinate holds the forecast_datetime."""
        pred, conf = _make_pred_conf()
        lat = np.linspace(-90, 90, 4)
        lon = np.linspace(0, 360, 8, endpoint=False)
        dt = datetime(2023, 7, 4, 12, 0, 0)
        upper, _ = make_xarray(pred, dt, lat, lon, conf)
        assert upper.coords["time"].values[0] == np.datetime64(dt)

    def test_custom_level_ids(self):
        """level_ids from conf['data'] override the default range."""
        pred, conf = _make_pred_conf(n_vars=2, n_levels=3)
        conf["data"]["level_ids"] = [500, 850, 1000]
        lat = np.linspace(-90, 90, 4)
        lon = np.linspace(0, 360, 8, endpoint=False)
        upper, _ = make_xarray(pred, datetime(2020, 1, 1), lat, lon, conf)
        assert list(upper.coords["level"].values) == [500, 850, 1000]

    def test_values_preserved(self):
        """Numeric values in the prediction array are not altered by make_xarray."""
        pred, conf = _make_pred_conf(n_vars=1, n_levels=2, n_surface=1, n_diag=0, height=3, width=3)
        lat = np.linspace(-90, 90, 3)
        lon = np.linspace(0, 360, 3, endpoint=False)
        upper, _ = make_xarray(pred, datetime(2020, 1, 1), lat, lon, conf)
        # upper has shape (time=1, vars=1, level=2, lat=3, lon=3)
        expected = pred[0, :2, 0].reshape(1, 1, 2, 3, 3)
        np.testing.assert_array_equal(upper.values, expected)
