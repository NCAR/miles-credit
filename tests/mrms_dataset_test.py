"""
mrms_dataset_test.py
--------------------
Tests for MRMSDataset (credit.datasets.MRMS).

Output format
-------------
Samples are nested dicts::

    {
        "input":  {"mrms/{field_type}/2d/{varname}": tensor, ...},
        "target": {"mrms/{field_type}/2d/{varname}": tensor, ...},  # return_target only
        "metadata": {"input_datetime": int, "target_datetime": int},
    }

Field type semantics:
    prognostic     — in input at step 0 AND in target (autoregressive rollout)
    dynamic_forcing — in input at every step; never in target
    diagnostic     — in target only

Tensor shapes (single sample, no batch dim):
    All MRMS variables are 2D: (1, 1, lat, lon)
"""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from credit.datasets.MRMS import MRMSDataset
from credit.samplers import DistributedMultiStepBatchSampler

QPE_1H = "MultiSensor_QPE_01H_Pass2_00.00"
QPE_6H = "MultiSensor_QPE_06H_Pass2_00.00"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mrms_xr_dataset():
    """Single xr.Dataset covering the full test date range."""
    time = pd.date_range("2024-06-01", "2024-06-03", freq="6h")
    lat = np.linspace(20.0, 55.0, 50)
    lon = np.linspace(230.0, 300.0, 100)  # 0-360 convention

    return xr.Dataset(
        data_vars={
            QPE_1H: (
                ("time", "lat", "lon"),
                np.random.rand(len(time), len(lat), len(lon)).astype(np.float32),
            ),
            QPE_6H: (
                ("time", "lat", "lon"),
                np.random.rand(len(time), len(lat), len(lon)).astype(np.float32),
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


@pytest.fixture
def patch_mrms_io(monkeypatch, mrms_xr_dataset):
    """Patch glob + xr.open_dataset so MRMSDataset loads from the fake dataset."""
    MRMS_MODULE = "credit.datasets.MRMS"
    monkeypatch.setattr(
        f"{MRMS_MODULE}.glob",
        lambda pattern: ["/fake/MRMS_20240601-000000.nc"],
    )
    monkeypatch.setattr(xr, "open_dataset", lambda path, **kw: mrms_xr_dataset)
    return mrms_xr_dataset


@pytest.fixture
def minimal_config():
    """Config with only prognostic field type."""
    return {
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2024-06-01",
        "end_datetime": "2024-06-02",
        "source": {
            "MRMS": {
                "mode": "local",
                "variables": {
                    "prognostic": {
                        "vars_2D": [QPE_1H, QPE_6H],
                        "path": "/fake/MRMS_*.nc",
                        "filename_time_format": "%Y%m%d-%H%M%S",
                    }
                },
            }
        },
    }


@pytest.fixture
def config_with_forcing():
    """Config with both prognostic and dynamic_forcing field types."""
    return {
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2024-06-01",
        "end_datetime": "2024-06-02",
        "source": {
            "MRMS": {
                "mode": "local",
                "variables": {
                    "prognostic": {
                        "vars_2D": [QPE_1H],
                        "path": "/fake/MRMS_*.nc",
                        "filename_time_format": "%Y%m%d-%H%M%S",
                    },
                    "dynamic_forcing": {
                        "vars_2D": [QPE_6H],
                        "path": "/fake/MRMS_*.nc",
                        "filename_time_format": "%Y%m%d-%H%M%S",
                    },
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mrms_dataset_len(minimal_config, patch_mrms_io):
    ds = MRMSDataset(minimal_config)
    assert len(ds) > 0


def test_mrms_key_format(minimal_config, patch_mrms_io):
    """Both QPE variables should appear under mrms/prognostic/2d/."""
    ds = MRMSDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    inp = sample["input"]
    assert f"mrms/prognostic/2d/{QPE_1H}" in inp
    assert f"mrms/prognostic/2d/{QPE_6H}" in inp
    assert "metadata" in sample


def test_mrms_prognostic_loaded_at_step0(minimal_config, patch_mrms_io):
    """Prognostic variables should appear in input at step i=0."""
    ds = MRMSDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert f"mrms/prognostic/2d/{QPE_1H}" in sample["input"]
    assert f"mrms/prognostic/2d/{QPE_6H}" in sample["input"]


def test_mrms_prognostic_absent_at_step1(minimal_config, patch_mrms_io):
    """Prognostic variables should NOT appear in input at step i > 0."""
    ds = MRMSDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 1)]

    # Only dynamic_forcing would be loaded at i > 0, but none is configured
    # here — so input is empty
    assert len(sample["input"]) == 0


def test_mrms_dynamic_forcing_every_step(config_with_forcing, patch_mrms_io):
    """Dynamic forcing should appear in input at both i=0 and i=1."""
    ds = MRMSDataset(config_with_forcing)
    t = ds.datetimes[0]

    sample0 = ds[(t, 0)]
    sample1 = ds[(t, 1)]

    forcing_key = f"mrms/dynamic_forcing/2d/{QPE_6H}"
    prog_key = f"mrms/prognostic/2d/{QPE_1H}"

    # Dynamic forcing present at every step
    assert forcing_key in sample0["input"]
    assert forcing_key in sample1["input"]

    # Prognostic only at step 0
    assert prog_key in sample0["input"]
    assert prog_key not in sample1["input"]


def test_mrms_tensor_shape(minimal_config, patch_mrms_io):
    """All tensors should have shape (1, 1, lat, lon)."""
    ds = MRMSDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    lat, lon = 50, 100

    for key, tensor in sample["input"].items():
        assert tensor.shape == (1, 1, lat, lon), (
            f"{key}: expected (1, 1, {lat}, {lon}), got {tensor.shape}"
        )
        assert tensor.dtype == torch.float32


def test_mrms_all_tensors_ndim4(minimal_config, patch_mrms_io):
    """All input tensors must have exactly 4 dimensions."""
    ds = MRMSDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key, tensor in sample["input"].items():
        assert tensor.ndim == 4, f"{key} has {tensor.ndim} dims, expected 4"


def test_mrms_target_is_dict(minimal_config, patch_mrms_io):
    """Target should be a dict, not a tensor."""
    ds = MRMSDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "target" in sample
    assert isinstance(sample["target"], dict)


def test_mrms_target_keys(minimal_config, patch_mrms_io):
    """Target should contain prognostic variable keys."""
    ds = MRMSDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert f"mrms/prognostic/2d/{QPE_1H}" in sample["target"]
    assert f"mrms/prognostic/2d/{QPE_6H}" in sample["target"]


def test_mrms_target_tensor_shapes(minimal_config, patch_mrms_io):
    """Target tensors should have the same shape as the corresponding input tensors."""
    ds = MRMSDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key in sample["target"]:
        assert sample["target"][key].shape == sample["input"][key].shape


def test_mrms_metadata_datetimes(minimal_config, patch_mrms_io):
    """metadata datetimes should match the sampler-generated timestamps."""
    ds = MRMSDataset(minimal_config, return_target=True)
    x_times, y_times = [], []
    for t in ds.datetimes:
        sample = ds[(t, 0)]
        x_times.append(sample["metadata"]["input_datetime"])
        y_times.append(sample["metadata"]["target_datetime"])

    assert (pd.to_datetime(x_times) == pd.to_datetime(ds.datetimes)).all()
    assert (pd.to_datetime(y_times) == (pd.to_datetime(ds.datetimes) + ds.dt)).all()


def test_mrms_extent_applied(minimal_config, patch_mrms_io):
    """With extent set, lat/lon dims should be smaller than the full grid."""
    cfg = dict(minimal_config)
    cfg["source"] = dict(minimal_config["source"])
    cfg["source"]["MRMS"] = dict(minimal_config["source"]["MRMS"])
    # Subset to roughly half the lon range (230-265 out of 230-300)
    cfg["source"]["MRMS"]["extent"] = [230, 265, 20, 55]

    ds = MRMSDataset(cfg)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    key = f"mrms/prognostic/2d/{QPE_1H}"
    tensor = sample["input"][key]

    # Full lon span is 100 points (230-300); subset (230-265) should be fewer
    assert tensor.shape[-1] < 100, (
        f"Expected lon dim < 100 after extent subsetting, got {tensor.shape[-1]}"
    )


def test_mrms_dataloader_default_collate(minimal_config, patch_mrms_io):
    """DataLoader + DistributedMultiStepBatchSampler should work without custom collate."""
    ds = MRMSDataset(minimal_config, return_target=True)
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

    key = f"mrms/prognostic/2d/{QPE_1H}"
    assert batch["input"][key].shape == (2, 1, 1, 50, 100)
    assert batch["target"][key].shape == (2, 1, 1, 50, 100)
