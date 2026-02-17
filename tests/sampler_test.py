import numpy as np
import pandas as pd
import xarray as xr
import pytest
from credit.datasets.era5 import ERA5Dataset
from credit.samplers import DistributedMultiStepBatchSampler
from torch.utils.data import DataLoader


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


def test_sampler_multistep(minimal_config, patch_era5_io_multiyear):

    dataset = ERA5Dataset(minimal_config, return_target=True)
    batch_size = 4
    sampler = DistributedMultiStepBatchSampler(
        dataset=dataset, batch_size=batch_size, rank=0, num_replicas=1, shuffle=True
    )
    loader = iter(DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=0, pin_memory=True))
    time_steps = []

    for i in range(dataset.num_forecast_steps):
        batch = next(loader)
        ts = pd.to_datetime(batch["metadata"]["input_datetime"][0])
        time_steps.append(ts)

    assert (time_steps == pd.date_range(time_steps[0], time_steps[-1], freq="6h")).all()
    batch = next(loader)
    assert not ts == (pd.to_datetime(batch["metadata"]["input_datetime"][0]) + pd.Timedelta("6h"))
