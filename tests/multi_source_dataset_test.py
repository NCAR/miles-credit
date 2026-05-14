"""
multi_source_dataset_test.py
----------------------------
Tests for MultiSourceDataset (credit.datasets.multi_source).

Output format
-------------
Samples are nested dicts keyed first by data type, then by source name::

    {
        "input":    {"era5": {...}, "mrms": {...}},
        "target":   {"era5": {...}, "mrms": {...}},   # return_target only
        "metadata": {"era5": {...}, "mrms": {...}},
    }

Sub-dataset IO is replaced with lightweight fakes via monkeypatching so these
tests exercise the wrapper logic only (timestamp intersection, delegation,
output structure) without touching disk.
"""

from typing import Any

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from credit.datasets.base_dataset import AbstractBaseDataset, BaseDataset
from credit.datasets.multi_source import MultiSourceDataset
from credit.samplers import DistributedMultiStepBatchSampler

# Shared constants
DATETIMES = pd.date_range("2024-06-01", "2026-06-02", freq="6h")
DATETIMES_SUBSET = pd.date_range("2020-06-01", "2022-12-31", freq="12h")
BASE1_KEYS = ["Base1/prognostic/3d/T", "Base1/prognostic/3d/U", "Base1/prognostic/2d/t2m"]
BASE2_KEYS = ["Base2/dynamic_forcing/2d/d2m", "Base2/diagnostic/3d/V"]
BASE3_KEYS = ["Base3/diagnostic/3d/V", "Base3/static/2d/orog"]


@pytest.fixture
def multi_config() -> dict[str, Any]:
    return {
        "source": {
            "Base1": {
                "dataset_type": "base",
                "variables": {
                    "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
                },
            },
            "Base2": {
                "dataset_type": "base",
                "variables": {"dynamic_forcing": {"vars_2D": ["d2m"]}, "diagnostic": {"vars_3D": ["V"]}},
            },
            "Base3": {
                "dataset_type": "base",
                "variables": {
                    "diagnostic": {"vars_3D": ["V"]},
                    "static": {"vars_2D": ["orog"]},
                },
            },
        },
        "timestep": "6h",
        "forecast_len": 2,
        "start_datetime": "2024-06-01",
        "end_datetime": "2026-06-02",
    }


@pytest.fixture
def multi_config_time_subsets() -> dict[str, Any]:
    return {
        "source": {
            "Base1": {
                "dataset_type": "base",
                "variables": {
                    "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
                },
                "start_datetime": "2020-06-01",
                "end_datetime": "2022-12-31",
            },
            "Base2": {
                "dataset_type": "base",
                "variables": {"dynamic_forcing": {"vars_2D": ["d2m"]}, "diagnostic": {"vars_3D": ["V"]}},
                "timestep": "12h",
            },
            "Base3": {
                "dataset_type": "base",
                "variables": {
                    "static": {"vars_2D": ["orog"]},
                },
            },
        },
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2000-01-01",
        "end_datetime": "2026-06-02",
    }


@pytest.fixture
def one_source_config() -> dict[str, Any]:
    return {
        "source": {
            "Single_Base": {
                "dataset_type": "base",
                "variables": {
                    "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
                    "dynamic_forcing": {"vars_2D": ["d2m"]},
                    "diagnostic": {"vars_3D": ["V"]},
                    "static": {"vars_2D": ["orog"]},
                },
            }
        },
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2024-06-01",
        "end_datetime": "2024-06-02",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multi_source_len(multi_config):
    """Basic type and length checks."""
    ds = MultiSourceDataset(multi_config)
    assert isinstance(ds, MultiSourceDataset)
    assert isinstance(ds, AbstractBaseDataset)
    assert len(ds) > 0
    # This is a simplification for these parameters!
    assert len(ds) == (len(DATETIMES) - 2), (
        f"DS has {len(ds)} samples but expected {len(DATETIMES)} based on datetimes fixture"
    )


def test_multi_source_output_source_keys(multi_config):
    """Check that we can sample and that the source names are correct."""
    ds = MultiSourceDataset(multi_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "input" in sample
    assert "metadata" in sample

    for source in ("Base1", "Base2", "Base3"):
        assert source in sample["input"]
        assert source in sample["metadata"]


def test_multi_source_target_present(multi_config):
    """With return_target=True, each source should appear in sample['target']."""
    ds = MultiSourceDataset(multi_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "target" in sample
    for source in ("Base1", "Base2", "Base3"):
        assert source in sample["target"]
        assert isinstance(sample["target"][source], dict)


def test_multi_source_variable_keys_no_return_type(multi_config):
    """Multisource variable keys should appear in input only when return_target=False."""
    ds = MultiSourceDataset(multi_config, return_target=False)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "target" not in sample

    for base, b_keys in zip(("Base1", "Base2", "Base3"), (BASE1_KEYS, BASE2_KEYS, BASE3_KEYS)):
        for b_key in b_keys:
            if "diagnostic" in b_key:
                assert b_key not in sample["input"][base]
            else:
                assert b_key in sample["input"][base]
            assert base in sample["metadata"]

            # Ensure no cross-talk between sources
            for other_base in ("Base1", "Base2", "Base3"):
                if other_base == base:
                    continue
                assert b_key not in sample["input"][other_base]


def test_multi_source_variable_keys_no_return_type_step_index(multi_config):
    """Step index should not affect variable keys when return_target=False."""
    ds = MultiSourceDataset(multi_config, return_target=False)
    t = ds.datetimes[0]
    sample_1 = ds[(t, 1)]

    assert "target" not in sample_1

    for base, b_keys in zip(("Base1", "Base2", "Base3"), (BASE1_KEYS, BASE2_KEYS, BASE3_KEYS)):
        for b_key in b_keys:
            if "dynamic_forcing" in b_key:
                assert b_key in sample_1["input"][base]
            else:
                assert b_key not in sample_1["input"][base]
            assert base in sample_1["metadata"]


def test_multi_source_variable_keys_with_return_type(multi_config):
    """Multisource variable keys should be split between input and target when return_target=True."""
    ds = MultiSourceDataset(multi_config, return_target=True)
    t = ds.datetimes[0]
    sample = ds[(t, 2)]

    for base, b_keys in zip(("Base1", "Base2", "Base3"), (BASE1_KEYS, BASE2_KEYS, BASE3_KEYS)):
        for b_key in b_keys:
            if "static" in b_key:
                assert b_key not in sample["input"][base]
                assert b_key not in sample["target"][base]
            elif "dynamic_forcing" in b_key:
                assert b_key in sample["input"][base]
                assert b_key not in sample["target"][base]
            else:
                assert b_key not in sample["input"][base]
                assert b_key in sample["target"][base]

            # Ensure no cross-talk between sources
            for other_base in ("Base1", "Base2", "Base3"):
                if other_base == base:
                    continue
                assert b_key not in sample["input"][other_base]
                assert b_key not in sample["target"][other_base]


def test_multi_source_dataset_type(multi_config):
    """Input and target values should be tensors."""
    ds = MultiSourceDataset(multi_config, return_target=True)
    for sub_ds in ds.datasets.values():
        assert isinstance(sub_ds, BaseDataset)


def test_multi_source_single_dataset(one_source_config):
    """With only one source in config, output should have 'Single_Base' under 'input'."""
    ds = MultiSourceDataset(one_source_config)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]

    assert "input" in sample
    assert set(sample["input"].keys()) == {"Single_Base"}


def test_multi_source_static_metadata(multi_config):
    """static_metadata should aggregate sub-dataset static_metadata by source name."""
    ds = MultiSourceDataset(multi_config)

    assert hasattr(ds, "static_metadata")
    assert "Base1" in ds.static_metadata
    assert "Base2" in ds.static_metadata
    assert "Base3" in ds.static_metadata


def test_multi_source_timestamp_intersection_time_subsets(multi_config_time_subsets):
    ds = MultiSourceDataset(multi_config_time_subsets)

    assert isinstance(ds.datetimes, pd.DatetimeIndex)
    assert len(ds.datetimes) > 0
    # This is a simplification for these parameters!
    assert len(ds) == len(DATETIMES_SUBSET) - 1
    assert ds.datetimes[0] == DATETIMES_SUBSET[0]
    assert ds.datetimes[-1] == DATETIMES_SUBSET[-2]


def test_multi_source_dataloader_default_collate(multi_config):
    """DataLoader + DistributedMultiStepBatchSampler should work without custom collate."""
    batch_size = 7  # Something different than the other dimensions

    ds = MultiSourceDataset(multi_config, return_target=True)
    sampler = DistributedMultiStepBatchSampler(
        ds,
        batch_size=batch_size,
        num_forecast_steps=multi_config["forecast_len"],
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

    assert isinstance(batch, dict)
    assert "input" in batch
    assert "target" in batch
    assert "metadata" in batch

    for source in ("Base1", "Base2", "Base3"):
        assert source in batch["input"]
        assert source in batch["target"]
        assert source in batch["metadata"]

    for base, b_keys in zip(("Base1", "Base2", "Base3"), (BASE1_KEYS, BASE2_KEYS, BASE3_KEYS)):
        for entry_key, entry_val in batch["input"][base].items():
            assert entry_key in b_keys
            assert isinstance(entry_val, torch.Tensor)
            # should be (batch, n_levels, 1, lat, lon)
            assert len(entry_val.shape) == 5, (
                f"Expected 5D tensor for {entry_key} in {base}, got shape {entry_val.shape}"
            )
            # batch dim should be prepended
            assert entry_val.shape[0] == batch_size, (
                f"Batch size {batch_size} should be the zeroth dimension of tensor for {entry_key} in {base} with shape {entry_val.shape}"
            )
