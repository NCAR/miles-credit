"""
multi_source_dataset_test.py
----------------------------
Tests for MultiSourceDataset (credit.datasets.multi_source).

Output format
-------------
Samples are nested dicts keyed by lowercase source name::

    {
        "era5": {"input": {...}, "target": {...}, "metadata": {...}},
        "mrms": {"input": {...}, "target": {...}, "metadata": {...}},
    }

Sub-dataset IO is replaced with lightweight fakes via monkeypatching so these
tests exercise the wrapper logic only (timestamp intersection, delegation,
output structure) without touching disk.
"""

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from credit.datasets.multi_source import MultiSourceDataset
from credit.samplers import DistributedMultiStepBatchSampler

# Shared constants
DATETIMES = pd.date_range("2024-06-01", "2024-06-02", freq="6h")
ERA5_KEY = "era5/prognostic/3d/T"
MRMS_KEY = "mrms/prognostic/2d/QPE"
ERA5_SHAPE = (4, 1, 8, 8)   # (n_levels, 1, lat, lon)
MRMS_SHAPE = (1, 1, 50, 100)


# ---------------------------------------------------------------------------
# Fake sub-datasets
# ---------------------------------------------------------------------------

class _FakeERA5:
    """Minimal ERA5Dataset stand-in."""

    datetimes = DATETIMES

    def __init__(self, config, return_target=False):
        self.return_target = return_target
        self.static_metadata = {"levels": [1000, 850, 500, 300], "datetime_fmt": "unix_ns"}

    def __len__(self):
        return len(self.datetimes)

    def __getitem__(self, args):
        sample = {
            "input": {ERA5_KEY: torch.zeros(*ERA5_SHAPE)},
            "metadata": {"input_datetime": 0},
        }
        if self.return_target:
            sample["target"] = {ERA5_KEY: torch.zeros(*ERA5_SHAPE)}
            sample["metadata"]["target_datetime"] = 1
        return sample


class _FakeMRMS:
    """Minimal MRMSDataset stand-in."""

    datetimes = DATETIMES

    def __init__(self, config, return_target=False):
        self.return_target = return_target
        self.static_metadata = {"datetime_fmt": "unix_ns"}

    def __len__(self):
        return len(self.datetimes)

    def __getitem__(self, args):
        sample = {
            "input": {MRMS_KEY: torch.zeros(*MRMS_SHAPE)},
            "metadata": {"input_datetime": 0},
        }
        if self.return_target:
            sample["target"] = {MRMS_KEY: torch.zeros(*MRMS_SHAPE)}
            sample["metadata"]["target_datetime"] = 1
        return sample


class _FakeMRMSSubset:
    """MRMS stand-in whose datetimes are a strict subset of ERA5's."""

    datetimes = DATETIMES[::2]  # every other timestamp

    def __init__(self, config, return_target=False):
        self.return_target = return_target
        self.static_metadata = {"datetime_fmt": "unix_ns"}

    def __len__(self):
        return len(self.datetimes)

    def __getitem__(self, args):
        sample = {
            "input": {MRMS_KEY: torch.zeros(*MRMS_SHAPE)},
            "metadata": {"input_datetime": 0},
        }
        if self.return_target:
            sample["target"] = {MRMS_KEY: torch.zeros(*MRMS_SHAPE)}
            sample["metadata"]["target_datetime"] = 1
        return sample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_sources(monkeypatch):
    """Replace _SOURCE_REGISTRY with lightweight fakes."""
    import credit.datasets.multi_source as ms
    monkeypatch.setattr(ms, "_SOURCE_REGISTRY", {"ERA5": _FakeERA5, "MRMS": _FakeMRMS})


@pytest.fixture
def patch_sources_subset(monkeypatch):
    """MRMS has a strict subset of ERA5 timestamps — tests intersection."""
    import credit.datasets.multi_source as ms
    monkeypatch.setattr(ms, "_SOURCE_REGISTRY", {"ERA5": _FakeERA5, "MRMS": _FakeMRMSSubset})


@pytest.fixture
def both_config():
    return {
        "source": {"ERA5": {}, "MRMS": {}},
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2024-06-01",
        "end_datetime": "2024-06-02",
    }


@pytest.fixture
def era5_only_config():
    return {
        "source": {"ERA5": {}},
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2024-06-01",
        "end_datetime": "2024-06-02",
    }


@pytest.fixture
def mrms_only_config():
    return {
        "source": {"MRMS": {}},
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2024-06-01",
        "end_datetime": "2024-06-02",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_multi_source_len(patch_sources, both_config):
    ds = MultiSourceDataset(both_config)
    assert len(ds) > 0
    assert len(ds) == len(DATETIMES)


def test_multi_source_output_source_keys(patch_sources, both_config):
    """Top-level keys should be lowercase source names."""
    ds = MultiSourceDataset(both_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert set(sample.keys()) == {"era5", "mrms"}


def test_multi_source_nested_structure(patch_sources, both_config):
    """Each source dict should contain 'input' and 'metadata'."""
    ds = MultiSourceDataset(both_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for source in ("era5", "mrms"):
        assert "input" in sample[source]
        assert "metadata" in sample[source]


def test_multi_source_target_present(patch_sources, both_config):
    """With return_target=True, each source should have a 'target' key."""
    ds = MultiSourceDataset(both_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    for source in ("era5", "mrms"):
        assert "target" in sample[source]
        assert isinstance(sample[source]["target"], dict)


def test_multi_source_variable_keys(patch_sources, both_config):
    """ERA5 and MRMS variable keys should appear under the correct source."""
    ds = MultiSourceDataset(both_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert ERA5_KEY in sample["era5"]["input"]
    assert MRMS_KEY in sample["mrms"]["input"]
    # No cross-contamination
    assert MRMS_KEY not in sample["era5"]["input"]
    assert ERA5_KEY not in sample["mrms"]["input"]


def test_multi_source_era5_only(patch_sources, era5_only_config):
    """With only ERA5 in config, output should have only 'era5' key."""
    ds = MultiSourceDataset(era5_only_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert set(sample.keys()) == {"era5"}


def test_multi_source_mrms_only(patch_sources, mrms_only_config):
    """With only MRMS in config, output should have only 'mrms' key."""
    ds = MultiSourceDataset(mrms_only_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert set(sample.keys()) == {"mrms"}


def test_multi_source_static_metadata(patch_sources, both_config):
    """static_metadata should aggregate sub-dataset static_metadata by source name."""
    ds = MultiSourceDataset(both_config)

    assert hasattr(ds, "static_metadata")
    assert "era5" in ds.static_metadata
    assert "mrms" in ds.static_metadata
    assert ds.static_metadata["era5"]["levels"] == [1000, 850, 500, 300]
    assert ds.static_metadata["era5"]["datetime_fmt"] == "unix_ns"
    assert ds.static_metadata["mrms"]["datetime_fmt"] == "unix_ns"


def test_multi_source_timestamp_intersection(patch_sources_subset, both_config):
    """MultiSourceDataset.datetimes should be the intersection of sub-dataset timestamps."""
    ds = MultiSourceDataset(both_config)
    # _FakeMRMSSubset has every other timestamp from DATETIMES
    expected = set(DATETIMES) & set(DATETIMES[::2])
    assert set(ds.datetimes) == expected
    assert len(ds) == len(expected)


def test_multi_source_dataloader_default_collate(patch_sources, both_config):
    """DataLoader + DistributedMultiStepBatchSampler should work without custom collate."""
    ds = MultiSourceDataset(both_config, return_target=True)
    sampler = DistributedMultiStepBatchSampler(
        ds,
        batch_size=2,
        num_forecast_steps=both_config["forecast_len"],
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

    # ERA5 tensor: batch dim prepended → (2, 4, 1, 8, 8)
    assert batch["era5"]["input"][ERA5_KEY].shape == (2, *ERA5_SHAPE)
    assert batch["era5"]["target"][ERA5_KEY].shape == (2, *ERA5_SHAPE)

    # MRMS tensor: batch dim prepended → (2, 1, 1, 50, 100)
    assert batch["mrms"]["input"][MRMS_KEY].shape == (2, *MRMS_SHAPE)
    assert batch["mrms"]["target"][MRMS_KEY].shape == (2, *MRMS_SHAPE)
