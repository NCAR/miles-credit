"""Unit tests for the ``credit begin`` wizard's pure helpers (credit/cli/_begin.py).

The wizard itself is interactive, but everything between the prompts is a pure
function: preset -> config dict.  These tests build the same ``state`` dict the
prompts would produce and check that the emitted configs pass ``credit check``
and that the non-interactive helpers behave — no data files, no network, no GPU.
"""

import socket
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from credit.cli import _Report, _run_checks
from credit.cli._begin import (
    _WB2_GRIDS,
    _build_config,
    _detect_system,
    _make_data,
    _padding_totals,
    _parse_level,
    _pbs_config,
    _read_level_values,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(preset, save_loc, system="local"):
    """The state dict ``_collect_state`` would produce for *preset* with defaults."""
    return {
        "system": system,
        "experiment": "test_exp",
        "save_loc": str(save_loc),
        "seed": 3141,
        "dataset": preset["dataset_type"],
        "preset": preset,
        "start": preset["start"],
        "end": preset["end"],
        "timestep": preset["timestep"],
        "valid_start": "2021-01-01",
        "valid_end": "2021-12-31",
        "vars_3D": preset["vars_3D"],
        "vars_2D": preset["vars_2D"],
        "batch_size": 1,
        "batches_per_epoch": 1,
        "parallelism_data": "none",
        "nodes": 1,
        "gpus": 1,
        "pbs": None,
    }


def _arco_preset():
    return {
        "dataset_type": "arco_era5",
        "level_coord": "hybrid",
        "levels": list(range(1, 138)),
        "vars_3D": ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"],
        "vars_2D": ["surface_pressure", "2m_temperature"],
        "static": ["geopotential_at_surface", "land_sea_mask"],
        "timestep": "6h",
        "start": "1979-01-01",
        "end": "2018-12-31",
        "height": 721,
        "width": 1440,
        "tisr": ([90, -90, 721], [0, 359.75, 1440]),
    }


def _wb2_preset(resolution):
    height, width, lat_spec, lon_spec = _WB2_GRIDS[resolution]
    return {
        "dataset_type": "weatherbench2_era5",
        "level_coord": "level",
        "levels": [500, 700, 850],
        "vars_3D": ["temperature", "specific_humidity"],
        "vars_2D": ["surface_pressure"],
        "static": ["land_sea_mask"],
        "timestep": "6h",
        "start": "1959-01-01",
        "end": "2022-12-31",
        "height": height,
        "width": width,
        "resolution": resolution,
        "tisr": (lat_spec, lon_spec),
    }


def _errors(conf):
    rep = _Report("begin-test.yml")
    _run_checks(conf, rep, deep=False)
    return [f for f in rep.findings if f.severity == "error"]


# ---------------------------------------------------------------------------
# Generated configs pass `credit check`
# ---------------------------------------------------------------------------


def test_arco_config_passes_check(tmp_path):
    conf = _build_config(_state(_arco_preset(), tmp_path))
    assert _errors(conf) == []
    assert conf["model"]["output_only_channels"] == 0
    # static fields + tisr are input-only channels
    assert conf["model"]["input_only_channels"] == 3


@pytest.mark.parametrize("resolution", sorted(_WB2_GRIDS))
def test_wb2_config_passes_check(tmp_path, resolution):
    conf = _build_config(_state(_wb2_preset(resolution), tmp_path))
    assert _errors(conf) == []


# ---------------------------------------------------------------------------
# WB2 TISR grid specs match the real store coordinates
# ---------------------------------------------------------------------------

# Verified against the public WeatherBench2 zarr stores: the regridded 240x121
# and 64x32 stores use ascending latitude (64x32 pole-offset), and no store
# wraps longitude back to the prime meridian.
_WB2_TRUTH = {
    "1440x721": ((90.0, -90.0, 721), (0.0, 359.75, 1440)),
    "240x121": ((-90.0, 90.0, 121), (0.0, 358.5, 240)),
    "64x32": ((-87.1875, 87.1875, 32), (0.0, 354.375, 64)),
    "full": ((90.0, -90.0, 721), (0.0, 359.75, 1440)),
}


@pytest.mark.parametrize("resolution", sorted(_WB2_GRIDS))
def test_wb2_grid_specs_match_store_coordinates(resolution):
    height, width, lat_spec, lon_spec = _WB2_GRIDS[resolution]
    lat_truth, lon_truth = _WB2_TRUTH[resolution]
    assert tuple(lat_spec) == pytest.approx(lat_truth)
    assert tuple(lon_spec) == pytest.approx(lon_truth)
    assert lat_spec[2] == height and lon_spec[2] == width
    # inclusive-endpoint linspace must never duplicate the 0/360 meridian
    assert (lon_spec[1] - lon_spec[0]) % 360 != 0


# ---------------------------------------------------------------------------
# _make_data: local static/diagnostic wiring
# ---------------------------------------------------------------------------


def _local_preset(**overrides):
    preset = {
        "dataset_type": "local",
        "level_coord": "level",
        "levels": [100, 500, 1000],
        "vars_3D": ["T", "Q"],
        "vars_2D": ["SP"],
        "static": [],
        "diagnostic": [],
        "timestep": "6h",
        "start": "2020-01-01",
        "end": "2020-12-31",
        "height": 181,
        "width": 360,
        "local_paths": {"prognostic": "/data/prog_*.nc", "diagnostic": "", "static": ""},
    }
    preset.update(overrides)
    return preset


def test_make_data_local_disabled_groups(tmp_path):
    state = _state(_local_preset(), tmp_path)
    data, _, _, _, _, input_only, output_only = _make_data(state)
    variables = data["source"]["ERA5"]["variables"]
    assert variables["static"] is None
    assert variables["diagnostic"] is None
    assert input_only == 0 and output_only == 0


def test_make_data_local_with_static_and_diagnostic(tmp_path):
    preset = _local_preset(
        static=["LSM"],
        diagnostic=["TP"],
        local_paths={
            "prognostic": "/data/prog_*.nc",
            "diagnostic": "/data/diag_*.nc",
            "static": "/data/static.nc",
        },
    )
    state = _state(preset, tmp_path)
    data, _, _, _, _, input_only, output_only = _make_data(state)
    variables = data["source"]["ERA5"]["variables"]
    assert variables["static"] == {"vars_2D": ["LSM"], "path": "/data/static.nc"}
    assert variables["diagnostic"] == {"vars_2D": ["TP"], "path": "/data/diag_*.nc"}
    assert input_only == 1 and output_only == 1
    conf = _build_config(state)
    assert conf["model"]["output_only_channels"] == 1


def test_make_data_local_never_emits_pathless_static(tmp_path):
    # A static group without a path is invalid for LocalDataset; when the
    # static path is disabled the group must collapse to None even if the
    # preset still carries variable names.
    preset = _local_preset(static=["LSM"])
    data, _, _, _, _, input_only, _ = _make_data(_state(preset, tmp_path))
    assert data["source"]["ERA5"]["variables"]["static"] is None
    assert input_only == 0


# ---------------------------------------------------------------------------
# Smaller helpers
# ---------------------------------------------------------------------------


def test_padding_totals_known_grid():
    assert _padding_totals(181, 360) == (75, 24)
    h_total, w_total = _padding_totals(721, 1440)
    assert h_total >= 0 and w_total >= 0


def test_pbs_config_derecho():
    result = _pbs_config("ncar", "exp", nodes=2, gpus=4)
    assert result["nodes"] == 2 and result["ngpus"] == 4
    assert {"queue", "walltime", "mem", "job_name"} <= set(result)


@pytest.mark.parametrize(
    ("hostname", "expected"),
    [
        ("derecho7", "derecho"),
        ("dec0123.hsn.de.hpc.ucar.edu", "ncar"),
        ("casper-login1", "casper"),
        ("nid001234", "perlmutter"),
        ("decode-box.local", "local"),
        ("unidata.ucar.edu", "local"),
        ("my-laptop", "local"),
    ],
)
def test_detect_system_hostnames(monkeypatch, hostname, expected):
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)
    assert _detect_system()["system"] == expected


def test_parse_level():
    assert _parse_level("100") == 100 and isinstance(_parse_level("100"), int)
    assert _parse_level("87.1875") == pytest.approx(87.1875)
    assert _parse_level("surface") == "surface"


def test_read_level_values(tmp_path):
    path = tmp_path / "prog_2020.nc"
    ds = xr.Dataset(
        {"T": (("level", "lat", "lon"), np.zeros((3, 2, 2)))},
        coords={"level": [100.0, 500.0, 1000.0], "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    ds.to_netcdf(path)
    assert _read_level_values(str(tmp_path / "prog_*.nc"), "level") == [100.0, 500.0, 1000.0]
    assert _read_level_values(str(tmp_path / "prog_*.nc"), "missing_coord") is None
    assert _read_level_values(str(tmp_path / "nope_*.nc"), "level") is None
    assert _read_level_values(str(tmp_path / "prog_*.nc"), "") is None


def test_python_dash_m_credit_cli_runs():
    # `credit begin` shells out to `python -m credit.cli check`; guard the
    # __main__ entry so that invocation keeps working.
    result = subprocess.run(
        [sys.executable, "-m", "credit.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "check" in result.stdout
