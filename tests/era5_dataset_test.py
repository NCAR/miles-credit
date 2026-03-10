"""
era5_dataset_test.py
--------------------
Tests for both the original ERA5Dataset (credit.datasets.era5) and the
refactored version (credit.datasets.era5_refactor).

Refactored dataset output format
---------------------------------
Samples are nested dicts::

    {
        "input":  {"era5/{field_type}/{dim}/{varname}": tensor, ...},
        "target": {"era5/{field_type}/{dim}/{varname}": tensor, ...},  # return_target only
        "metadata": {"input_datetime": int, "target_datetime": int},
    }

Tensor shapes (single sample, no batch dim):
    3D variable : (n_levels, 1, lat, lon)
    2D variable : (1, 1, lat, lon)
"""

import numpy as np
import pandas as pd
import xarray as xr
import pytest
import torch
from torch.utils.data import DataLoader

from credit.datasets.era5_old import ERA5Dataset
from credit.datasets.era5 import ERA5Dataset as ERA5DatasetRefactored
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

    def make_ds(start, end):
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
def patch_era5_io_multiyear(monkeypatch, annual_xr_dataset):
    """
    Patch glob + xarray open so ERA5Dataset sees
    multiple yearly files and routes correctly.
    """
    ERA5_MODULE = "credit.datasets.era5"

    monkeypatch.setattr(f"{ERA5_MODULE}.glob", lambda pattern: ["/fake/era5_2022.zarr", "/fake/era5_2023.zarr"])

    def fake_open_dataset(path, **kwargs):
        for year in (2022, 2023):
            if str(year) in path:
                return annual_xr_dataset[year]
        raise ValueError(f"Unexpected path: {path}")

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    return annual_xr_dataset


@pytest.fixture
def patch_refactor_io_multiyear(monkeypatch, annual_xr_dataset):
    """
    Same patching as above but targets the refactored module path.
    """
    ERA5_MODULE = "credit.datasets.era5_refactor"

    monkeypatch.setattr(f"{ERA5_MODULE}.glob", lambda pattern: ["/fake/era5_2022.zarr", "/fake/era5_2023.zarr"])

    def fake_open_dataset(path, **kwargs):
        for year in (2022, 2023):
            if str(year) in path:
                return annual_xr_dataset[year]
        raise ValueError(f"Unexpected path: {path}")

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    return annual_xr_dataset


@pytest.fixture
def minimal_config():
    return {
        "timestep": "6h",
        "forecast_len": 6,
        "start_datetime": "2022-12-25",
        "end_datetime": "2023-01-05",
        "source": {
            "ERA5": {
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


# ---------------------------------------------------------------------------
# Original ERA5Dataset tests (unchanged behaviour)
# ---------------------------------------------------------------------------

def test_dataset_len(minimal_config, patch_era5_io_multiyear):
    ds = ERA5Dataset(minimal_config)
    assert len(ds) > 0


def test_step0_contains_prognostic(minimal_config, patch_era5_io_multiyear):
    ds = ERA5Dataset(minimal_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]
    assert "dynamic_forcing" in sample
    assert "prognostic" in sample
    assert isinstance(sample["dynamic_forcing"], torch.Tensor)
    assert isinstance(sample["prognostic"], torch.Tensor)


def test_step1_skips_prognostic(minimal_config, patch_era5_io_multiyear):
    ds = ERA5Dataset(minimal_config, return_target=False)
    t = ds.datetimes[0]

    sample = ds[(t, 1)]

    assert "metadata" in sample
    assert "dynamic_forcing" in sample
    assert "prognostic" not in sample


def test_return_target(minimal_config, patch_era5_io_multiyear):
    ds = ERA5Dataset(minimal_config, return_target=True)
    t = ds.datetimes[0]

    sample = ds[(t, 0)]

    assert "target" in sample
    assert isinstance(sample["target"], torch.Tensor)


def test_tensor_shapes(minimal_config, patch_era5_io_multiyear):
    ds = ERA5Dataset(minimal_config, return_target=True)
    t = ds.datetimes[0]

    sample = ds[(t, 0)]
    x = sample["prognostic"]
    y = sample["target"]

    assert x.ndim == 4
    assert x.shape == (9, 1, 21, 41)
    assert y.ndim == x.ndim
    assert y.shape == (10, 1, 21, 41)


def test_datetimes(minimal_config, patch_era5_io_multiyear):
    ds = ERA5Dataset(minimal_config, return_target=True)
    x_times, y_times = [], []
    for i, t in enumerate(ds.datetimes):
        sample = ds[(t, 1)]
        x_times.append(sample["metadata"]["input_datetime"])
        y_times.append(sample["metadata"]["target_datetime"])

    assert (pd.to_datetime(x_times) == pd.to_datetime(ds.datetimes)).all()
    assert (pd.to_datetime(y_times) == (pd.to_datetime(ds.datetimes) + ds.dt)).all()


# ---------------------------------------------------------------------------
# Refactored ERA5Dataset tests
# ---------------------------------------------------------------------------

def test_refactor_dataset_len(minimal_config, patch_refactor_io_multiyear):
    ds = ERA5DatasetRefactored(minimal_config)
    assert len(ds) > 0


def test_refactor_key_format_step0(minimal_config, patch_refactor_io_multiyear):
    """Step 0 input should contain per-variable keys for prognostic, static, and dynamic_forcing."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    inp = sample["input"]
    assert "era5/prognostic/3d/T" in inp
    assert "era5/prognostic/3d/U" in inp
    assert "era5/prognostic/2d/SP" in inp
    assert "era5/dynamic_forcing/2d/tsi" in inp
    assert "era5/static/2d/LSM" in inp
    assert "metadata" in sample


def test_refactor_key_format_step1(minimal_config, patch_refactor_io_multiyear):
    """Step > 0 input should only contain dynamic_forcing keys (no prognostic/static)."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 1)]

    inp = sample["input"]
    assert "era5/dynamic_forcing/2d/tsi" in inp
    assert "era5/prognostic/3d/T" not in inp
    assert "era5/prognostic/2d/SP" not in inp
    assert "era5/static/2d/LSM" not in inp
    assert "metadata" in sample


def test_refactor_3d_tensor_shape(minimal_config, patch_refactor_io_multiyear):
    """3D variables should have shape (n_levels, 1, lat, lon)."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    n_levels = len(minimal_config["source"]["ERA5"]["levels"])  # 4
    lat, lon = 21, 41

    t_tensor = sample["input"]["era5/prognostic/3d/T"]
    assert t_tensor.shape == (n_levels, 1, lat, lon), (
        f"Expected ({n_levels}, 1, {lat}, {lon}), got {t_tensor.shape}"
    )
    assert t_tensor.dtype == torch.float32


def test_refactor_2d_tensor_shape(minimal_config, patch_refactor_io_multiyear):
    """2D variables should have shape (1, 1, lat, lon) — singleton level dim."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    lat, lon = 21, 41

    for key in ("era5/prognostic/2d/SP", "era5/dynamic_forcing/2d/tsi", "era5/static/2d/LSM"):
        assert key in sample["input"]
        tensor = sample["input"][key]
        assert tensor.shape == (1, 1, lat, lon), (
            f"{key}: expected (1, 1, {lat}, {lon}), got {tensor.shape}"
        )
        assert tensor.dtype == torch.float32


def test_refactor_all_tensors_same_ndim(minimal_config, patch_refactor_io_multiyear):
    """All variable tensors in input must have exactly 4 dimensions."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key, tensor in sample["input"].items():
        assert tensor.ndim == 4, f"{key} has {tensor.ndim} dims, expected 4"


def test_refactor_target_is_dict(minimal_config, patch_refactor_io_multiyear):
    """Target should be a dict of per-variable tensors, not a concatenated tensor."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "target" in sample
    assert isinstance(sample["target"], dict)


def test_refactor_target_keys(minimal_config, patch_refactor_io_multiyear):
    """Target should contain prognostic + diagnostic keys (T, U, SP, TP)."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    tgt = sample["target"]
    assert "era5/prognostic/3d/T" in tgt
    assert "era5/prognostic/3d/U" in tgt
    assert "era5/prognostic/2d/SP" in tgt
    assert "era5/diagnostic/2d/TP" in tgt
    # static and dynamic_forcing should not appear in target
    assert "era5/static/2d/LSM" not in tgt
    assert "era5/dynamic_forcing/2d/tsi" not in tgt


def test_refactor_target_tensor_shapes(minimal_config, patch_refactor_io_multiyear):
    """Each tensor in target should have correct shape."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    n_levels = len(minimal_config["source"]["ERA5"]["levels"])  # 4
    lat, lon = 21, 41

    tgt = sample["target"]
    assert tgt["era5/prognostic/3d/T"].shape == (n_levels, 1, lat, lon)
    assert tgt["era5/prognostic/2d/SP"].shape == (1, 1, lat, lon)
    assert tgt["era5/diagnostic/2d/TP"].shape == (1, 1, lat, lon)


def test_refactor_metadata_datetimes(minimal_config, patch_refactor_io_multiyear):
    """metadata['input_datetime'] should match the timestamp passed to __getitem__."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=True)
    x_times, y_times = [], []
    for t in ds.datetimes:
        sample = ds[(t, 1)]
        x_times.append(sample["metadata"]["input_datetime"])
        y_times.append(sample["metadata"]["target_datetime"])

    assert (pd.to_datetime(x_times) == pd.to_datetime(ds.datetimes)).all()
    assert (pd.to_datetime(y_times) == (pd.to_datetime(ds.datetimes) + ds.dt)).all()


def test_refactor_static_metadata(minimal_config, patch_refactor_io_multiyear):
    """static_metadata should contain levels and datetime_fmt."""
    ds = ERA5DatasetRefactored(minimal_config, return_target=False)

    assert hasattr(ds, "static_metadata")
    assert ds.static_metadata["levels"] == minimal_config["source"]["ERA5"]["levels"]
    assert ds.static_metadata["datetime_fmt"] == "unix_ns"


def test_refactor_null_diagnostic(minimal_config, patch_refactor_io_multiyear):
    """Setting diagnostic: null in config should produce an empty target for that field."""
    cfg = dict(minimal_config)
    cfg["source"] = dict(minimal_config["source"])
    cfg["source"]["ERA5"] = dict(minimal_config["source"]["ERA5"])
    cfg["source"]["ERA5"]["variables"] = dict(minimal_config["source"]["ERA5"]["variables"])
    cfg["source"]["ERA5"]["variables"]["diagnostic"] = None

    ds = ERA5DatasetRefactored(cfg, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "era5/diagnostic/2d/TP" not in sample["target"]


def test_refactor_dataloader_default_collate(minimal_config, patch_refactor_io_multiyear):
    """
    Dataset + DistributedMultiStepBatchSampler + DataLoader should work
    without a custom collate_fn.
    Validates that tensors for the same key are stacked correctly under input/target.
    """
    ds = ERA5DatasetRefactored(minimal_config, return_target=True)
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
