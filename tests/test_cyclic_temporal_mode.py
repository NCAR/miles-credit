"""Tests for temporal_mode: "cyclic" (credit.datasets.gen_2.base_dataset / multi_source).

Uses a lightweight ``RecordingDataset`` (registered via ``register_dataset``,
the same mechanism external/custom datasets use) that records every timestamp
``_extract_field`` is actually called with, instead of touching real files --
this is what lets these tests assert on exactly which timestamp (real vs.
cycle-remapped) reached the read call, which a bare dataset_type: "base"
source (constant output regardless of *t*) can't distinguish.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from credit.datasets import register_dataset
from credit.datasets.gen_2.base_dataset import BaseDataset
from credit.datasets.gen_2.local import LocalDataset
from credit.datasets.gen_2.multi_source import MultiSourceDataset


@register_dataset("recording_base")
class RecordingDataset(BaseDataset):
    """Test-only BaseDataset subclass: records (field_type, t) for every
    _extract_field call and returns constant data (values are never
    what's under test here -- only which timestamp was requested is)."""

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        super().__init__(data_config, return_target)
        self.dataset_type = "recording_base"
        self.recorded_calls: list[tuple[str, Any]] = []
        self.init_register_all_fields()

    def _extract_field(self, field_type: str, t, sample: dict[str, Any]) -> None:
        self.recorded_calls.append((field_type, t))
        if field_type in self.var_dict:
            for var_2d in self.var_dict[field_type].get("vars_2D", []):
                key = self._get_field_name(field_type, "2d", var_2d)
                sample[key] = torch.ones(1, 1, 3, 3)
            for var_3d in self.var_dict[field_type].get("vars_3D", []):
                key = self._get_field_name(field_type, "3d", var_3d)
                sample[key] = torch.ones(2, 1, 3, 3)


def _single_source_config(**source_overrides) -> dict[str, Any]:
    source = {
        "dataset_type": "recording_base",
        "mode": "remote",
        "variables": {"dynamic_forcing": {"vars_2D": ["sst"]}},
        "temporal_mode": "cyclic",
        "cycle_year": 2000,
        "timestep": "1D",
        "start_datetime": "2000-01-01",
        "end_datetime": "2000-12-31",
        **source_overrides,
    }
    return {
        "source": {"Clim": source},
        "timestep": "1D",
        "forecast_len": 1,
        "start_datetime": "2000-01-01",
        "end_datetime": "2000-12-31",
    }


# --------------------------------------------------------------------------- #
# BaseDataset-level: _resolve_cyclic_timestamp / _load_sample wiring
# --------------------------------------------------------------------------- #


class TestCyclicBaseDataset:
    def test_basic_lookup_resolves_into_cycle_year(self):
        """A real timestamp from an arbitrary year resolves to the matching
        day within cycle_year, regardless of which real year it came from."""
        ds = RecordingDataset(_single_source_config())
        ds[(pd.Timestamp("1985-07-15 12:00"), 0)]
        recorded_ts = [t for _, t in ds.recorded_calls]
        assert all(t == pd.Timestamp("2000-07-15") for t in recorded_ts)

    def test_same_calendar_day_every_real_year(self):
        """The same (month, day) from different real years all resolve to the
        identical on-disk cycle_year record -- no per-year duplication needed."""
        ds = RecordingDataset(_single_source_config())
        for year in (1979, 1999, 2018):
            ds.recorded_calls.clear()
            ds[(pd.Timestamp(f"{year}-07-15 00:00"), 0)]
            assert all(t == pd.Timestamp("2000-07-15") for _, t in ds.recorded_calls)

    def test_metadata_reports_real_timestamp_not_cycle_remap(self):
        """input_datetime must reflect the real master timestamp, not the
        internal cycle-year lookup key."""
        ds = RecordingDataset(_single_source_config())
        sample = ds[(pd.Timestamp("1985-07-15 12:00"), 0)]
        from credit.datasets.gen_2._utils import decode_time

        assert decode_time(sample["metadata"]["input_datetime"]) == pd.Timestamp("1985-07-15 12:00")

    def test_history_window_crosses_real_year_boundary(self):
        """history_len=4 at a real Jan 1 spans real Dec 31 (previous real year)
        through Jan 1 -- each entry must resolve independently into cycle_year
        (all landing near its end/start), with no attempt to look outside it."""
        cfg = _single_source_config(
            timestep="6h",
            # A few days' buffer beyond the calendar year on each side: this
            # source's own self.datetimes is trimmed by (history_len-1)*dt at
            # the start and forecast_len*dt at the end (ordinary BaseDataset
            # behavior, unrelated to cyclic mode) -- the buffer keeps Dec 31 /
            # Jan 1 of the actual cycle_year available after that trim.
            start_datetime="1999-12-28",
            end_datetime="2001-01-03",
        )
        cfg["source"]["Clim"]["history_len"] = 4
        cfg["history_len"] = 4
        ds = RecordingDataset(cfg)

        ds[(pd.Timestamp("1986-01-01 00:00"), 0)]
        recorded_ts = [t for _, t in ds.recorded_calls]
        assert recorded_ts == [
            pd.Timestamp("2000-12-31 06:00"),
            pd.Timestamp("2000-12-31 12:00"),
            pd.Timestamp("2000-12-31 18:00"),
            pd.Timestamp("2000-01-01 00:00"),
        ]

    def test_wraps_to_last_entry_when_before_first_native_record(self):
        """A remap landing before the cycle's first native timestamp wraps to
        the cycle's last entry instead of raising (periodic, not bounded)."""
        cfg = _single_source_config(timestep="1D")
        # Native records start at 06:00, not 00:00 -- start_datetime shifts the clock.
        cfg["source"]["Clim"]["start_datetime"] = "2000-01-01 06:00"
        cfg["source"]["Clim"]["end_datetime"] = "2000-12-31 06:00"
        ds = RecordingDataset(cfg)

        # A real Jan 1 00:00 remaps to cycle_year Jan 1 00:00, before the first
        # native record (Jan 1 06:00) -> should wrap to the last entry (Dec 31 06:00).
        ds[(pd.Timestamp("1990-01-01 00:00"), 0)]
        recorded_ts = [t for _, t in ds.recorded_calls]
        assert recorded_ts == [ds.datetimes[-1]]

    def test_missing_cycle_year_raises_clear_error(self):
        cfg = _single_source_config()
        del cfg["source"]["Clim"]["cycle_year"]
        with pytest.raises(ValueError, match="cycle_year"):
            RecordingDataset(cfg)

    def test_non_cyclic_source_unaffected(self):
        """temporal_mode left at the default ("exact") behaves exactly as
        before -- no cyclic remapping applied."""
        cfg = _single_source_config()
        del cfg["source"]["Clim"]["temporal_mode"]
        del cfg["source"]["Clim"]["cycle_year"]
        ds = RecordingDataset(cfg)
        t = pd.Timestamp("2000-07-15 00:00")
        ds[(t, 0)]
        assert all(recorded_t == t for _, recorded_t in ds.recorded_calls)


# --------------------------------------------------------------------------- #
# MultiSourceDataset-level: master clock / calendar exemptions
# --------------------------------------------------------------------------- #


@pytest.fixture
def cyclic_multi_config() -> dict[str, Any]:
    """A multi-decade master clock with one ordinary ("exact") source spanning
    the whole run, and one cyclic climatology source spanning a single year."""
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
        },
        "timestep": "1D",
        "forecast_len": 1,
        "start_datetime": "1979-01-01",
        "end_datetime": "1979-01-05",
    }


class TestCyclicMultiSourceDataset:
    def test_master_clock_not_clipped_to_cyclic_source_range(self, cyclic_multi_config):
        """The master clock spans 1979, entirely outside the cyclic source's own
        (2000-01-01..2000-12-31) coverage -- it must not be clipped or filtered
        against that range at all."""
        ds = MultiSourceDataset(cyclic_multi_config)
        expected = pd.date_range("1979-01-01", pd.Timestamp("1979-01-05") - pd.Timedelta("1D"), freq="1D")
        assert ds.datetimes.equals(expected)

    def test_cyclic_source_answers_for_out_of_range_real_tick(self, cyclic_multi_config):
        """A real 1979 master tick -- far outside the cyclic source's own
        2000-only coverage -- still returns data for it via MultiSourceDataset."""
        ds = MultiSourceDataset(cyclic_multi_config)
        t = ds.datetimes[0]
        sample = ds[(t, 0)]
        assert "Clim" in sample["input"]
        clim_ds = ds.datasets["Clim"]
        assert len(clim_ds.recorded_calls) == 1
        assert clim_ds.recorded_calls[0][1] == pd.Timestamp(f"2000-{t.month:02d}-{t.day:02d}")

    def test_nonstandard_cyclic_calendar_does_not_constrain_master_clock(self, cyclic_multi_config):
        """A cyclic noleap source must not remove Feb 29 from a standard real clock."""
        cyclic_multi_config["source"]["Clim"]["calendar"] = "noleap"
        cyclic_multi_config["start_datetime"] = "2000-02-28"
        cyclic_multi_config["end_datetime"] = "2000-03-02"

        ds = MultiSourceDataset(cyclic_multi_config)

        assert ds.calendar == "standard"
        assert pd.Timestamp("2000-02-29") in ds.datetimes


# --------------------------------------------------------------------------- #
# LocalDataset: inferring cycle_year/start_datetime/end_datetime/timestep
# from the data itself, when the config omits them
# --------------------------------------------------------------------------- #


def _write_climatology_file(path, times):
    ds = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.random.rand(len(times), 4, 4).astype(np.float32))},
        coords={"time": times, "latitude": np.linspace(-60, 60, 4), "longitude": np.linspace(0, 300, 4)},
    )
    ds.to_netcdf(path)
    return str(path)


def _cyclic_local_config(path, **source_overrides):
    source = {
        "dataset_type": "local",
        "temporal_mode": "cyclic",
        "variables": {"dynamic_forcing": {"vars_2D": ["sst"], "path": path}},
        **source_overrides,
    }
    return {"source": {"SST_clim": source}, "forecast_len": 1}


class TestCyclicLocalDatasetInference:
    def test_infers_dt_start_end_cycle_year_from_file(self, tmp_path):
        times = pd.date_range("2000-01-01", "2000-12-31", freq="1D")
        path = _write_climatology_file(tmp_path / "sst_clim.nc", times)
        ds = LocalDataset(_cyclic_local_config(path))

        assert ds.cycle_year == 2000
        assert ds.dt == pd.Timedelta("1D")
        assert ds.start_datetime == pd.Timestamp("2000-01-01")
        assert ds.end_datetime == pd.Timestamp("2000-12-31")

    def test_inferred_source_answers_for_an_arbitrary_real_year(self, tmp_path):
        times = pd.date_range("2000-01-01", "2000-12-31", freq="1D")
        path = _write_climatology_file(tmp_path / "sst_clim.nc", times)
        ds = LocalDataset(_cyclic_local_config(path))

        sample = ds[(pd.Timestamp("1985-07-15"), 0)]
        assert "SST_clim/dynamic_forcing/2d/sst" in sample["input"]

    def test_explicit_cycle_year_overrides_inference(self, tmp_path):
        times = pd.date_range("2000-01-01", "2000-12-31", freq="1D")
        path = _write_climatology_file(tmp_path / "sst_clim.nc", times)
        ds = LocalDataset(_cyclic_local_config(path, cycle_year=2004))
        assert ds.cycle_year == 2004

    def test_explicit_start_end_timestep_override_inference(self, tmp_path):
        times = pd.date_range("2000-01-01", "2000-12-31", freq="1D")
        path = _write_climatology_file(tmp_path / "sst_clim.nc", times)
        ds = LocalDataset(
            _cyclic_local_config(path, start_datetime="2000-01-01", end_datetime="2000-06-01", timestep="1D")
        )
        assert ds.start_datetime == pd.Timestamp("2000-01-01")
        assert ds.end_datetime == pd.Timestamp("2000-06-01")

    def test_file_spanning_multiple_years_raises(self, tmp_path):
        times = pd.date_range("2000-06-01", "2001-06-01", freq="1D")
        path = _write_climatology_file(tmp_path / "sst_clim.nc", times)
        with pytest.raises(ValueError, match="spans more than one year"):
            LocalDataset(_cyclic_local_config(path))

    def test_single_timestamp_file_falls_back_to_required_config_error(self, tmp_path):
        """No timestep can be inferred from one timestamp -- falls back to the
        ordinary required-config error rather than guessing."""
        path = _write_climatology_file(tmp_path / "sst_clim.nc", pd.date_range("2000-01-01", periods=1, freq="1D"))
        with pytest.raises(KeyError, match="timestep"):
            LocalDataset(_cyclic_local_config(path))

    def test_cyclic_source_cannot_combine_with_date_ranges(self, tmp_path):
        times = pd.date_range("2000-01-01", "2000-12-31", freq="1D")
        path = _write_climatology_file(tmp_path / "sst_clim.nc", times)
        with pytest.raises(ValueError, match="cannot be combined with"):
            LocalDataset(_cyclic_local_config(path, date_ranges=[["2000-01-01", "2000-06-01"]]))
