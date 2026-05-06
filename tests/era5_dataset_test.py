"""
era5_dataset_test.py
--------------------
Tests for the ERA5Dataset (credit.datasets.era5)

Dataset output format
---------------------------------
Samples are nested dicts::

    {
        "input": {"era5/{field_type}/{dim}/{varname}": tensor, ...},
        "target": {"era5/{field_type}/{dim}/{varname}": tensor, ...},  # return_target only
        "metadata": {"input_datetime": int, "target_datetime": int},
    }

Tensor shapes (single sample, no batch dim):
    3D variable: (n_levels, 1, lat, lon)
    2D variable: (1, 1, lat, lon)
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import pytest
import torch
from torch.utils.data import DataLoader

from credit.datasets.era5 import ERA5Dataset, ARCOERA5Dataset
from credit.samplers import DistributedMultiStepBatchSampler


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def annual_xr_dataset():
    """
    Return a dict mapping year -> xr.Dataset,
    each with time coords restricted to that year.
    """

    def make_ds(start: str, end: str) -> xr.Dataset:
        time = pd.date_range(start, end, freq="6h")
        level = [1000, 850, 500, 300]
        lat = np.linspace(-90, 90, 21)
        lon = np.linspace(-180, 180, 41)

        return xr.Dataset(
            data_vars={
                "T": (
                    ("time", "level", "latitude", "longitude"),
                    np.random.rand(len(time), len(level), len(lat), len(lon)),
                ),
                "U": (
                    ("time", "level", "latitude", "longitude"),
                    np.random.rand(len(time), len(level), len(lat), len(lon)),
                ),
                "SP": (("time", "latitude", "longitude"), np.random.rand(len(time), len(lat), len(lon))),
                "tsi": (("time", "latitude", "longitude"), np.random.rand(len(time), len(lat), len(lon))),
                "TP": (("time", "latitude", "longitude"), np.random.rand(len(time), len(lat), len(lon))),
                "LSM": (("latitude", "longitude"), np.random.rand(len(lat), len(lon))),
            },
            coords={
                "time": time,
                "level": level,
                "latitude": lat,
                "longitude": lon,
            },
        )

    return {
        2022: make_ds("2022-12-01", "2022-12-31 18:00"),
        2023: make_ds("2023-01-01", "2023-01-31"),
    }


@pytest.fixture
def patch_era5_io_multiyear(
    monkeypatch: pytest.MonkeyPatch, annual_xr_dataset: dict[int, xr.Dataset]
) -> dict[int, xr.Dataset]:
    """
    Patch glob + xarray open so ERA5Dataset sees
    multiple yearly files and routes correctly.
    """

    def fake_glob(pattern: str) -> list[str]:
        if pattern == "/fake/*.zarr":
            return ["/fake/era5_2022.zarr", "/fake/era5_2023.zarr"]
        raise ValueError(f"Unexpected glob pattern: {pattern}")

    monkeypatch.setattr("credit.datasets.base_dataset.glob", fake_glob)

    def fake_open_dataset(path: str, **kwargs) -> xr.Dataset:
        for year in (2022, 2023):
            if str(year) in path:
                return annual_xr_dataset[year]
        raise ValueError(f"Unexpected path: {path}")

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    return annual_xr_dataset


@pytest.fixture
def patch_refactor_io_multiyear(
    monkeypatch: pytest.MonkeyPatch, annual_xr_dataset: dict[int, xr.Dataset]
) -> dict[int, xr.Dataset]:
    """
    Same patching as above but targets the refactored module path.
    """

    def fake_glob(pattern: str) -> list[str]:
        if pattern == "/fake/*.zarr":
            return ["/fake/era5_2022.zarr", "/fake/era5_2023.zarr"]
        raise ValueError(f"Unexpected glob pattern: {pattern}")

    monkeypatch.setattr("credit.datasets.base_dataset.glob", fake_glob)

    def fake_open_dataset(path: str, **kwargs) -> xr.Dataset:
        for year in (2022, 2023):
            if str(year) in path:
                return annual_xr_dataset[year]
        raise ValueError(f"Unexpected path: {path}")

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    return annual_xr_dataset


@pytest.fixture
def minimal_config() -> dict[str, Any]:
    return {
        "timestep": "6h",
        "forecast_len": 6,
        "start_datetime": "2022-12-25",
        "end_datetime": "2023-01-05",
        "source": {
            "ERA5": {
                "dataset_name": "era5",
                "level_coord": "level",
                "levels": [1000, 850, 500, 300],
                "variables": {
                    "prognostic": {
                        "vars_3D": ["T", "U"],
                        "vars_2D": ["SP"],
                        "path": "/fake/*.zarr",
                    },
                    "dynamic_forcing": {
                        "vars_2D": ["tsi"],
                        "path": "/fake/*.zarr",
                    },
                    "static": {
                        "vars_2D": ["LSM"],
                        "path": "/fake/*.zarr",
                    },
                    "diagnostic": {
                        "vars_2D": ["TP"],
                        "path": "/fake/*.zarr",
                    },
                },
            }
        },
    }


@pytest.fixture
def minimal_arco_era5_config() -> dict[str, Any]:
    return {
        "timestep": "6h",
        "forecast_len": 6,
        "start_datetime": "2022-12-25",
        "end_datetime": "2023-01-05",
        "source": {
            "ARCO_ERA5": {
                "dataset_name": "arco_era5",
                "level_coord": "level",
                "levels": [1000, 850],
                "variables": {
                    "prognostic": {
                        "vars_3D": ["temperature"],
                        "vars_2D": ["surface_pressure"],
                    },
                    "dynamic_forcing": {
                        "vars_2D": ["toa_incident_solar_radiation"],
                    },
                    "static": {
                        "vars_2D": ["land_sea_mask"],
                    },
                    "diagnostic": {
                        "vars_2D": ["total_precipitation"],
                    },
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Original ERA5Dataset tests (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_dataset_len(minimal_config: dict[str, Any], patch_era5_io_multiyear: dict[int, xr.Dataset]):
    ds: ERA5Dataset = ERA5Dataset(minimal_config)
    assert len(ds) > 0


def test_return_target(minimal_config: dict[str, Any], patch_era5_io_multiyear: dict[int, xr.Dataset]):
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=True)
    t: pd.Timestamp = ds.datetimes[0]

    sample: dict[str, Any] = ds[(t, 0)]

    assert "target" in sample, "Expected 'target' key in sample"
    assert isinstance(sample["target"], dict), "Expected 'target' to be a dictionary"


def test_tensor_shapes(minimal_config: dict[str, Any], patch_era5_io_multiyear: dict[int, xr.Dataset]):
    ds = ERA5Dataset(minimal_config, return_target=True)
    t = ds.datetimes[0]

    sample = ds[(t, 0)]
    print(sample)
    x = sample["input"]
    y = sample["target"]

    assert x["era5/prognostic/3d/T"].ndim == 4
    assert x["era5/prognostic/3d/T"].shape == (4, 1, 21, 41)
    assert y["era5/prognostic/3d/T"].ndim == x["era5/prognostic/3d/T"].ndim
    assert y["era5/prognostic/3d/T"].shape == (4, 1, 21, 41)


def test_datetimes(minimal_config: dict[str, Any], patch_era5_io_multiyear: dict[int, xr.Dataset]):
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=True)
    x_times: list[int] = []
    y_times: list[int] = []

    for _, t in enumerate(ds.datetimes):
        sample = ds[(t, 1)]
        x_times.append(sample["metadata"]["input_datetime"])
        y_times.append(sample["metadata"]["target_datetime"])

    assert (pd.to_datetime(x_times) == pd.to_datetime(ds.datetimes)).all()
    assert (pd.to_datetime(y_times) == (pd.to_datetime(ds.datetimes) + ds.dt)).all()


# ---------------------------------------------------------------------------
# Refactored ERA5Dataset tests
# ---------------------------------------------------------------------------


def test_refactor_dataset_len(minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]):
    ds: ERA5Dataset = ERA5Dataset(minimal_config)
    assert len(ds) > 0


def test_refactor_key_format_step0(minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]):
    """Step 0 input should contain per-variable keys for prognostic, static, and dynamic_forcing."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=False)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 0)]

    inp = sample["input"]
    assert "era5/prognostic/3d/T" in inp
    assert "era5/prognostic/3d/U" in inp
    assert "era5/prognostic/2d/SP" in inp
    assert "era5/dynamic_forcing/2d/tsi" in inp
    assert "era5/static/2d/LSM" in inp
    assert "metadata" in sample


def test_refactor_key_format_step1(minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]):
    """Step > 0 input should only contain dynamic_forcing keys (no prognostic/static)."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=False)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 1)]

    inp = sample["input"]
    assert "era5/dynamic_forcing/2d/tsi" in inp
    assert "era5/prognostic/3d/T" not in inp
    assert "era5/prognostic/2d/SP" not in inp
    assert "era5/static/2d/LSM" not in inp
    assert "metadata" in sample


def test_refactor_3d_tensor_shape(minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]):
    """3D variables should have shape (n_levels, 1, lat, lon)."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=False)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 0)]

    n_levels = len(minimal_config["source"]["ERA5"]["levels"])  # 4
    lat, lon = 21, 41

    t_tensor = sample["input"]["era5/prognostic/3d/T"]
    assert t_tensor.shape == (n_levels, 1, lat, lon), f"Expected ({n_levels}, 1, {lat}, {lon}), got {t_tensor.shape}"
    assert t_tensor.dtype == torch.float32


def test_refactor_2d_tensor_shape(minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]):
    """2D variables should have shape (1, 1, lat, lon) — singleton level dim."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=False)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 0)]

    lat, lon = 21, 41

    for key in ("era5/prognostic/2d/SP", "era5/dynamic_forcing/2d/tsi", "era5/static/2d/LSM"):
        assert key in sample["input"]
        tensor = sample["input"][key]
        assert tensor.shape == (1, 1, lat, lon), f"{key}: expected (1, 1, {lat}, {lon}), got {tensor.shape}"
        assert tensor.dtype == torch.float32


def test_refactor_all_tensors_same_ndim(
    minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]
):
    """All variable tensors in input must have exactly 4 dimensions."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=False)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 0)]

    for key, tensor in sample["input"].items():
        assert tensor.ndim == 4, f"{key} has {tensor.ndim} dims, expected 4"


def test_refactor_target_is_dict(minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]):
    """Target should be a dict of per-variable tensors, not a concatenated tensor."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=True)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 0)]

    assert "target" in sample
    assert isinstance(sample["target"], dict)


def test_refactor_target_keys(minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]):
    """Target should contain prognostic + diagnostic keys (T, U, SP, TP)."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=True)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 0)]

    tgt = sample["target"]
    assert "era5/prognostic/3d/T" in tgt
    assert "era5/prognostic/3d/U" in tgt
    assert "era5/prognostic/2d/SP" in tgt
    assert "era5/diagnostic/2d/TP" in tgt
    # static and dynamic_forcing should not appear in target
    assert "era5/static/2d/LSM" not in tgt
    assert "era5/dynamic_forcing/2d/tsi" not in tgt


def test_refactor_target_tensor_shapes(
    minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]
):
    """Each tensor in target should have correct shape."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=True)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 0)]

    n_levels = len(minimal_config["source"]["ERA5"]["levels"])  # 4
    lat, lon = 21, 41

    tgt = sample["target"]
    assert tgt["era5/prognostic/3d/T"].shape == (n_levels, 1, lat, lon)
    assert tgt["era5/prognostic/2d/SP"].shape == (1, 1, lat, lon)
    assert tgt["era5/diagnostic/2d/TP"].shape == (1, 1, lat, lon)


def test_refactor_metadata_datetimes(
    minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]
):
    """metadata['input_datetime'] should match the timestamp passed to __getitem__."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=True)
    x_times: list[int] = []
    y_times: list[int] = []
    for t in ds.datetimes:
        sample = ds[(t, 1)]
        assert isinstance(sample["metadata"]["input_datetime"], (int, np.integer)), "Expected input_datetime to be int"
        assert isinstance(sample["metadata"]["target_datetime"], (int, np.integer)), (
            "Expected target_datetime to be int"
        )
        x_times.append(sample["metadata"]["input_datetime"])
        y_times.append(sample["metadata"]["target_datetime"])

    assert (pd.to_datetime(x_times) == pd.to_datetime(ds.datetimes)).all()
    assert (pd.to_datetime(y_times) == (pd.to_datetime(ds.datetimes) + ds.dt)).all()


def test_refactor_static_metadata(minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]):
    """static_metadata should contain levels and datetime_fmt."""
    ds: ERA5Dataset = ERA5Dataset(minimal_config, return_target=False)

    assert hasattr(ds, "static_metadata")
    assert ds.static_metadata["levels"] == minimal_config["source"]["ERA5"]["levels"]
    assert ds.static_metadata["datetime_fmt"] == "unix_ns"


def test_refactor_null_diagnostic(
    minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]
) -> None:
    """Setting diagnostic: null in config should produce an empty target for that field."""
    cfg = dict(minimal_config)
    cfg["source"] = dict(minimal_config["source"])
    cfg["source"]["ERA5"] = dict(minimal_config["source"]["ERA5"])
    cfg["source"]["ERA5"]["variables"] = dict(minimal_config["source"]["ERA5"]["variables"])
    cfg["source"]["ERA5"]["variables"]["diagnostic"] = None

    ds: ERA5Dataset = ERA5Dataset(cfg, return_target=True)
    t: pd.Timestamp = ds.datetimes[0]
    sample: dict[str, Any] = ds[(t, 0)]

    assert "era5/diagnostic/2d/TP" not in sample["target"]


def test_refactor_dataloader_default_collate(
    minimal_config: dict[str, Any], patch_refactor_io_multiyear: dict[int, xr.Dataset]
):
    """
    Dataset + DistributedMultiStepBatchSampler + DataLoader should work
    without a custom collate_fn.
    Validates that tensors for the same key are stacked correctly under input/target.
    """  # noqa: E501
    ds = ERA5Dataset(minimal_config, return_target=True)
    sampler = DistributedMultiStepBatchSampler(
        ds,
        batch_size=2,
        num_forecast_steps=minimal_config["forecast_len"],
        shuffle=False,
        num_replicas=1,
        rank=0,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
    )

    batch = next(iter(loader))

    n_levels = len(minimal_config["source"]["ERA5"]["levels"])  # 4
    lat, lon = 21, 41

    if "era5/prognostic/3d/T" in batch["input"]:
        assert batch["input"]["era5/prognostic/3d/T"].shape == (2, n_levels, 1, lat, lon)
    if "era5/dynamic_forcing/2d/tsi" in batch["input"]:
        assert batch["input"]["era5/dynamic_forcing/2d/tsi"].shape == (2, 1, 1, lat, lon)
    if "era5/prognostic/3d/T" in batch["target"]:
        assert batch["target"]["era5/prognostic/3d/T"].shape == (2, n_levels, 1, lat, lon)


def test_arco_era5_single_load(minimal_arco_era5_config):
    arco_ds: ARCOERA5Dataset = ARCOERA5Dataset(minimal_arco_era5_config, return_target=True)
    sample = arco_ds[(pd.Timestamp("2022-12-31 00:00"), 0)]
    assert isinstance(sample, dict)
    assert "input" in sample
    assert "metadata" in sample
    assert "target" in sample
    assert sample["input"]["arco_era5/prognostic/3d/temperature"].shape == (
        len(minimal_arco_era5_config["source"]["ARCO_ERA5"]["levels"]),
        1,
        721,
        1440,
    )
    assert sample["target"]["arco_era5/prognostic/3d/temperature"].min() > 200
    assert ~torch.any(torch.isnan(sample["target"]["arco_era5/prognostic/3d/temperature"]))
