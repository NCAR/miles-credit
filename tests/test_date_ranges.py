"""Tests for non-contiguous date_ranges (train/validate on disjoint date blocks,
e.g. 1950-1965 + 1970-1985, skipping 1966-1969).

Reuses the RecordingDataset registered by test_cyclic_temporal_mode.py (records
every timestamp _extract_field is called with, instead of touching real files).
"""

from typing import Any

import pandas as pd
import pytest

from credit.datasets.gen_2.multi_source import MultiSourceDataset

# Importing this module registers "recording_base" via @register_dataset --
# reused rather than re-registering to avoid the "overwriting registry entry" warning.
from tests.test_cyclic_temporal_mode import RecordingDataset  # noqa: F401


BLOCKS = [["1950-01-01", "1965-12-31"], ["1970-01-01", "1985-12-31"], ["1990-01-01", "2005-12-31"]]


def _single_source_config(**source_overrides) -> dict[str, Any]:
    source = {
        "dataset_type": "recording_base",
        "mode": "remote",
        "variables": {"prognostic": {"vars_2D": ["t2m"]}},
        **source_overrides,
    }
    return {"source": {"Src": source}, "timestep": "6h", "forecast_len": 1, "date_ranges": BLOCKS}


# --------------------------------------------------------------------------- #
# BaseDataset-level: date_ranges wiring + precedence
# --------------------------------------------------------------------------- #


class TestDateRangesBaseDataset:
    def test_gaps_excluded_from_source_datetimes(self):
        ds = RecordingDataset(_single_source_config())
        gap = ds.datetimes[(ds.datetimes > pd.Timestamp("1965-12-31")) & (ds.datetimes < pd.Timestamp("1970-01-01"))]
        assert len(gap) == 0
        assert ds.datetimes[0] >= pd.Timestamp("1950-01-01")
        assert ds.datetimes[-1] <= pd.Timestamp("2005-12-31")

    def test_boundary_init_never_produces_out_of_block_window(self):
        """The exact property the whole design depends on: a sampler-style init
        time near a block's tail must never let t + i*dt (or t - k*dt for
        history) land inside an excluded gap."""
        cfg = _single_source_config()
        cfg["source"]["Src"]["history_len"] = 4
        cfg["history_len"] = 4
        ds = RecordingDataset(cfg)

        # last available init time before the first gap
        t = ds.datetimes[ds.datetimes <= pd.Timestamp("1965-12-31")][-1]
        forecast_len = 1
        for i in range(forecast_len + 1):
            probe = t + i * ds.dt
            assert not (pd.Timestamp("1965-12-31") < probe < pd.Timestamp("1970-01-01"))
        # history window must also stay inside the same block
        for k in range(cfg["history_len"]):
            probe = t - k * ds.dt
            assert probe >= pd.Timestamp("1950-01-01")

    def test_backward_compatible_when_date_ranges_absent(self):
        """A config using only start_datetime/end_datetime (no date_ranges at
        all) behaves exactly as before -- untouched by this feature."""
        cfg = {
            "source": {
                "Src": {
                    "dataset_type": "recording_base",
                    "mode": "remote",
                    "variables": {"prognostic": {"vars_2D": ["t2m"]}},
                }
            },
            "timestep": "6h",
            "forecast_len": 1,
            "start_datetime": "2000-01-01",
            "end_datetime": "2000-01-05",
        }
        ds = RecordingDataset(cfg)
        assert ds.date_ranges is None
        expected = pd.date_range("2000-01-01", pd.Timestamp("2000-01-05") - pd.Timedelta("6h"), freq="6h")
        assert ds.datetimes.equals(expected)

    def test_source_level_scalar_overrides_data_level_date_ranges(self):
        """A source with its own start_datetime/end_datetime is unaffected by a
        data-level date_ranges -- this is the normal persist/cyclic pattern."""
        cfg = _single_source_config(start_datetime="2010-01-01", end_datetime="2010-01-05")
        ds = RecordingDataset(cfg)
        assert ds.date_ranges is None
        expected = pd.date_range("2010-01-01", pd.Timestamp("2010-01-05") - pd.Timedelta("6h"), freq="6h")
        assert ds.datetimes.equals(expected)

    def test_source_level_date_ranges_overrides_data_level(self):
        own_blocks = [["2010-01-01", "2010-01-03"]]
        cfg = _single_source_config(date_ranges=own_blocks)
        ds = RecordingDataset(cfg)
        assert ds.date_ranges == own_blocks
        assert ds.datetimes[0] >= pd.Timestamp("2010-01-01")
        assert ds.datetimes[-1] <= pd.Timestamp("2010-01-03")

    def test_source_level_scalar_and_date_ranges_together_raises(self):
        cfg = _single_source_config(start_datetime="2010-01-01", date_ranges=[["2010-01-01", "2010-01-03"]])
        with pytest.raises(ValueError, match="date_ranges cannot be combined"):
            RecordingDataset(cfg)

    def test_data_level_scalar_and_date_ranges_together_raises(self):
        cfg = _single_source_config()
        cfg["start_datetime"] = "1950-01-01"
        with pytest.raises(ValueError, match="date_ranges cannot be combined"):
            RecordingDataset(cfg)


# --------------------------------------------------------------------------- #
# MultiSourceDataset-level: master clock + persist/cyclic interaction
# --------------------------------------------------------------------------- #


@pytest.fixture
def date_ranges_multi_config() -> dict[str, Any]:
    return {
        "source": {
            "Real": {
                "dataset_type": "recording_base",
                "mode": "remote",
                "variables": {"prognostic": {"vars_2D": ["t2m"]}},
            },
            "Clim": {
                "dataset_type": "recording_base",
                "mode": "remote",
                "variables": {"dynamic_forcing": {"vars_2D": ["sst"]}},
                "temporal_mode": "cyclic",
                "cycle_year": 2000,
                "timestep": "1D",
                "start_datetime": "2000-01-01",
                "end_datetime": "2000-12-31",
            },
            "Persist1": {
                "dataset_type": "recording_base",
                "mode": "remote",
                "variables": {"static": {"vars_2D": ["orog"]}},
                "temporal_mode": "persist",
                "timestep": "1D",
                "start_datetime": "1950-01-01",
                "end_datetime": "2005-12-31",
            },
        },
        "timestep": "6h",
        "forecast_len": 1,
        "date_ranges": BLOCKS,
    }


class TestDateRangesMultiSource:
    def test_master_clock_excludes_gaps(self, date_ranges_multi_config):
        ds = MultiSourceDataset(date_ranges_multi_config)
        gap1 = ds.datetimes[(ds.datetimes > pd.Timestamp("1965-12-31")) & (ds.datetimes < pd.Timestamp("1970-01-01"))]
        gap2 = ds.datetimes[(ds.datetimes > pd.Timestamp("1985-12-31")) & (ds.datetimes < pd.Timestamp("1990-01-01"))]
        assert len(gap1) == 0
        assert len(gap2) == 0

    def test_normal_source_inherits_data_level_date_ranges(self, date_ranges_multi_config):
        """A source with no overrides of its own picks up the data-level
        date_ranges automatically."""
        ds = MultiSourceDataset(date_ranges_multi_config)
        real_ds = ds.datasets["Real"]
        assert real_ds.date_ranges == BLOCKS
        gap = real_ds.datetimes[
            (real_ds.datetimes > pd.Timestamp("1965-12-31")) & (real_ds.datetimes < pd.Timestamp("1970-01-01"))
        ]
        assert len(gap) == 0

    def test_cyclic_source_unaffected_by_data_level_date_ranges(self, date_ranges_multi_config):
        """A cyclic source keeps its own independent (single-year) coverage
        and answers correctly regardless of the outer date_ranges."""
        ds = MultiSourceDataset(date_ranges_multi_config)
        clim_ds = ds.datasets["Clim"]
        assert clim_ds.date_ranges is None
        assert clim_ds.datetimes[0] >= pd.Timestamp("2000-01-01")
        assert clim_ds.datetimes[-1] <= pd.Timestamp("2000-12-31")

        # a tick right at the tail of the first block still resolves correctly
        t = ds.datetimes[ds.datetimes <= pd.Timestamp("1965-12-31")][-1]
        ds[(t, 0)]
        field_type, resolved = clim_ds.recorded_calls[-1]
        assert (resolved.month, resolved.day) == (t.month, t.day)
        assert resolved.year == 2000

    def test_persist_source_unaffected_by_data_level_date_ranges(self, date_ranges_multi_config):
        ds = MultiSourceDataset(date_ranges_multi_config)
        persist_ds = ds.datasets["Persist1"]
        assert persist_ds.date_ranges is None
        # Persist's own full-span coverage is untouched by the master's gaps
        # (only trimmed by the ordinary forecast_len margin at the tail, same
        # as any single-range source -- unrelated to date_ranges).
        assert persist_ds.datetimes[0] == pd.Timestamp("1950-01-01")
        assert persist_ds.datetimes[-1] == pd.Timestamp("2005-12-31") - persist_ds.num_forecast_steps * persist_ds.dt

    def test_data_level_scalar_and_date_ranges_together_raises(self, date_ranges_multi_config):
        date_ranges_multi_config["start_datetime"] = "1950-01-01"
        with pytest.raises(ValueError, match="date_ranges cannot be combined"):
            MultiSourceDataset(date_ranges_multi_config)
