import numpy as np
import pandas as pd
import xarray as xr
import pytest
import torch
from credit.datasets.era5 import ERA5Dataset


@pytest.fixture
def annual_xr_dataset():
    """
    Return a dict mapping year -> xr.Dataset,
    each with time coords restricted to that year.
    """

    def make_ds(start, end):
        time = pd.date_range(start, end, freq="6h")
        level = [100, 85, 50, 30]
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

    # 1) glob returns two "files", one per year
    monkeypatch.setattr(f"{ERA5_MODULE}.glob", lambda pattern: ["/fake/era5_2022.zarr", "/fake/era5_2023.zarr"])

    # 2) open_dataset returns dataset based on year in filename
    def fake_open_dataset(path):
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
        "forecast_len": 5,
        "start_datetime": "2022-12-25",
        "end_datetime": "2023-01-05",
        "source": {
            "ERA5": {
                "level_coord": "level",
                "levels": [100, 85, 50, 30],
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
            }
        },
    }


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
