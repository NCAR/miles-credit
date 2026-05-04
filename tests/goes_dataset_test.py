"""
goes_dataset_test.py
--------------------
Tests for GOESDataset (credit.datasets.goes).

"""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from credit.datasets.goes import GOESDataset
from credit.samplers import DistributedMultiStepBatchSampler

CMI_C04 = "CMI_C04"
CMI_C07 = "CMI_C07"

# Fake grid dimensions (must match the lat/lon fixture)
NY = 40  # rows  (latitude  axis)
NX = 80  # cols  (longitude axis)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def goes_xr_dataset():
    """Single xr.Dataset on a fake curvilinear grid."""

    return xr.Dataset(
        data_vars={
            CMI_C04: (
                ("y", "x"),
                np.random.rand(NY, NX).astype(np.float32),
            ),
            CMI_C07: (
                ("y", "x"),
                np.random.rand(NY, NX).astype(np.float32),
            ),
        },
        coords={
            "y": np.arange(NY),
            "x": np.arange(NX),
        },
    )


@pytest.fixture
def latlon_xr_dataset():
    """Fake 2-D lat/lon grid matching goes_xr_dataset dimensions."""
    lat2d = np.linspace(20.0, 55.0, NY * NX).reshape(NY, NX).astype(np.float32)
    lon2d = np.linspace(-130.0, -60.0, NY * NX).reshape(NY, NX).astype(np.float32)

    return xr.Dataset(
        data_vars={
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
        },
    )


@pytest.fixture
def patch_goes_io(monkeypatch, goes_xr_dataset, latlon_xr_dataset):
    """Patch file-system and xr.open_dataset so GOESDataset uses fake data."""

    # --- patch _collect_GOES_file_path to return a fixed file map ---
    def fake_collect(self, base_dir="", verbose=False):
        extended = pd.date_range(
            self.datetimes[0],
            self.datetimes[-1] + self.dt,
            freq=self.dt,
        )
        result = []
        for t in extended:
            start = t
            end = t + self.dt - pd.Timedelta("1ns")
            result.append((start, end, "/fake/goes16_fake.nc"))
        return result

    monkeypatch.setattr(GOESDataset, "_collect_GOES_file_path", fake_collect)

    # --- patch xr.open_dataset to return fake GOES or latlon dataset ---
    original_open = xr.open_dataset

    def fake_open_dataset(path_or_obj, **kwargs):
        if isinstance(path_or_obj, str) and "lat_lon" in path_or_obj:
            return latlon_xr_dataset
        return goes_xr_dataset

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    return goes_xr_dataset


@pytest.fixture
def minimal_config():
    """Config with only prognostic field type."""
    return {
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2021-06-01",
        "end_datetime": "2021-06-02",
        "source": {
            "GOES": {
                "goes_id": "goes16",
                "mode": "local",
                "product": "ABI-L2-MCMIPC",
                "latlon2d_dir": "/fake/latlon/",
                "variables": {
                    "prognostic": {
                        "vars_2D": [CMI_C04, CMI_C07],
                        "path": "/fake/goes/",
                    },
                    "diagnostic": None,
                    "dynamic_forcing": None,
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Tests — basic construction
# ---------------------------------------------------------------------------


def test_goes_dataset_len(minimal_config, patch_goes_io):
    ds = GOESDataset(minimal_config)
    assert len(ds) > 0


def test_goes_dataset_datetimes_type(minimal_config, patch_goes_io):
    ds = GOESDataset(minimal_config)
    assert isinstance(ds.datetimes, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Tests — key format
# ---------------------------------------------------------------------------


def test_goes_key_format(minimal_config, patch_goes_io):
    """Both CMI variables should appear under goes16/prognostic/2d/."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    inp = sample["input"]
    assert f"goes16/prognostic/2d/{CMI_C04}" in inp
    assert f"goes16/prognostic/2d/{CMI_C07}" in inp
    assert "metadata" in sample


# ---------------------------------------------------------------------------
# Tests — step semantics
# ---------------------------------------------------------------------------


def test_goes_prognostic_loaded_at_step0(minimal_config, patch_goes_io):
    """Prognostic variables should appear in input at step i=0."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert f"goes16/prognostic/2d/{CMI_C04}" in sample["input"]
    assert f"goes16/prognostic/2d/{CMI_C07}" in sample["input"]


def test_goes_prognostic_absent_at_step1(minimal_config, patch_goes_io):
    """Prognostic variables should NOT appear in input at step i > 0."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 1)]

    # No dynamic_forcing configured here — input should be empty at i > 0
    assert len(sample["input"]) == 0


# ---------------------------------------------------------------------------
# Tests — tensor shape and dtype
# ---------------------------------------------------------------------------


def test_goes_tensor_shape(minimal_config, patch_goes_io):
    """All input tensors should have shape (1, 1, NY, NX)."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key, tensor in sample["input"].items():
        assert tensor.shape == (1, 1, NY, NX), f"{key}: expected (1, 1, {NY}, {NX}), got {tensor.shape}"
        assert tensor.dtype == torch.float32


def test_goes_all_tensors_ndim4(minimal_config, patch_goes_io):
    """All input tensors must have exactly 4 dimensions."""
    ds = GOESDataset(minimal_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key, tensor in sample["input"].items():
        assert tensor.ndim == 4, f"{key} has {tensor.ndim} dims, expected 4"


# ---------------------------------------------------------------------------
# Tests — target
# ---------------------------------------------------------------------------


def test_goes_target_is_dict(minimal_config, patch_goes_io):
    """Target should be a dict, not a tensor."""
    ds = GOESDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "target" in sample
    assert isinstance(sample["target"], dict)


def test_goes_target_keys(minimal_config, patch_goes_io):
    """Target should contain prognostic variable keys."""
    ds = GOESDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert f"goes16/prognostic/2d/{CMI_C04}" in sample["target"]
    assert f"goes16/prognostic/2d/{CMI_C07}" in sample["target"]


def test_goes_target_tensor_shapes(minimal_config, patch_goes_io):
    """Target tensors should have the same shape as the corresponding input tensors."""
    ds = GOESDataset(minimal_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for key in sample["target"]:
        inp_key = key  # prognostic keys are shared between input and target
        if inp_key in sample["input"]:
            assert sample["target"][key].shape == sample["input"][inp_key].shape


def test_goes_no_target_without_flag(minimal_config, patch_goes_io):
    """'target' key should be absent when return_target=False (default)."""
    ds = GOESDataset(minimal_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "target" not in sample


# ---------------------------------------------------------------------------
# Tests — metadata
# ---------------------------------------------------------------------------


def test_goes_metadata_input_datetime(minimal_config, patch_goes_io):
    """metadata['input_datetime'] should match the sampled timestamp (nanoseconds)."""
    ds = GOESDataset(minimal_config)
    for t in ds.datetimes:
        sample = ds[(t, 0)]
        assert sample["metadata"]["input_datetime"] == int(t.value)


def test_goes_metadata_target_datetime(minimal_config, patch_goes_io):
    """metadata['target_datetime'] should equal input_datetime + dt."""
    ds = GOESDataset(minimal_config, return_target=True)
    for t in ds.datetimes:
        sample = ds[(t, 0)]
        expected = int((t + ds.dt).value)
        assert sample["metadata"]["target_datetime"] == expected


def test_goes_metadata_datetimes_all_samples(minimal_config, patch_goes_io):
    """input and target datetimes should be consistent across the full dataset."""
    ds = GOESDataset(minimal_config, return_target=True)
    x_times, y_times = [], []
    for t in ds.datetimes:
        sample = ds[(t, 0)]
        x_times.append(sample["metadata"]["input_datetime"])
        y_times.append(sample["metadata"]["target_datetime"])

    assert (pd.to_datetime(x_times) == pd.to_datetime(ds.datetimes)).all()
    assert (pd.to_datetime(y_times) == (pd.to_datetime(ds.datetimes) + ds.dt)).all()


# ---------------------------------------------------------------------------
# Tests — spatial extent
# ---------------------------------------------------------------------------


def test_goes_extent_applied(minimal_config, patch_goes_io, monkeypatch, latlon_xr_dataset):
    """With extent set, spatial dims should be smaller than the full grid."""
    cfg = {
        **minimal_config,
        "source": {
            "GOES": {
                **minimal_config["source"]["GOES"],
                "extent": [-130, -95, 20, 55],  # roughly half the fake lon range
            }
        },
    }

    ds = GOESDataset(cfg)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    key = f"goes16/prognostic/2d/{CMI_C04}"
    tensor = sample["input"][key]

    # At least one spatial dim should be strictly smaller than the full grid
    assert tensor.shape[-1] < NX or tensor.shape[-2] < NY, (
        f"Expected cropped spatial dims, got full shape {tensor.shape}"
    )


# ---------------------------------------------------------------------------
# Tests — DataLoader integration
# ---------------------------------------------------------------------------


def test_goes_dataloader_default_collate(minimal_config, patch_goes_io):
    """DataLoader + DistributedMultiStepBatchSampler should collate correctly."""
    ds = GOESDataset(minimal_config, return_target=True)
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

    key = f"goes16/prognostic/2d/{CMI_C04}"
    assert batch["input"][key].shape == (2, 1, 1, NY, NX)
    assert batch["target"][key].shape == (2, 1, 1, NY, NX)
