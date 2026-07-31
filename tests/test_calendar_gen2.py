"""Calendar support tests for the gen2 pipeline (noleap / cftime data).

Covers, per the gen2 calendar plan:
  1. Helper unit tests (build_time_index, encode/decode_time, to_calendar, ...)
  2. Golden regression: standard-calendar behavior is identical to plain pandas
  3. LocalDataset + sampler integration on synthetic noleap data, with a
     multistep rollout crossing the Feb 28 -> Mar 1 leap boundary
  4. MultiSourceDataset master-clock rules: most-restrictive calendar,
     mixed-source validation errors, injected-datetimes conversion
  5. batch_init_times / metadata round-trips for inference
"""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from credit.datasets.gen_2._utils import (
    _find_file,
    build_time_index,
    build_time_index_multi,
    decode_time,
    encode_time,
    filter_index_by_labels,
    is_standard_calendar,
    most_restrictive_calendar,
    normalize_calendar,
    to_calendar,
    to_cycle_year,
)
from credit.datasets.gen_2.local import LocalDataset
from credit.datasets.gen_2.multi_source import MultiSourceDataset
from credit.samplers import MultiStepBatchSamplerSubset
from credit.trainers.rollout_utils import batch_init_times

# --------------------------------------------------------------------------- #
# 1. Helper unit tests
# --------------------------------------------------------------------------- #


class TestBuildTimeIndex:
    def test_standard_is_identical_to_pd_date_range(self):
        idx = build_time_index("2000-02-27", "2000-03-02", "6h", calendar="standard")
        expected = pd.date_range("2000-02-27", "2000-03-02", freq="6h")
        assert type(idx) is type(expected)
        assert idx.equals(expected)

    def test_none_calendar_is_standard(self):
        idx = build_time_index("2000-01-01", "2000-01-02", "6h", calendar=None)
        assert isinstance(idx, pd.DatetimeIndex)

    def test_timedelta_freq_matches_string_freq(self):
        a = build_time_index("2000-01-01", "2000-01-03", pd.Timedelta("6h"))
        b = build_time_index("2000-01-01", "2000-01-03", "6h")
        assert a.equals(b)

    def test_noleap_skips_feb29(self):
        idx = build_time_index("2000-02-27", "2000-03-02", "6h", calendar="noleap")
        assert isinstance(idx, xr.CFTimeIndex)
        assert not any(t.month == 2 and t.day == 29 for t in idx)
        # 27, 28, Mar 1, Mar 2 -> 3 full days of 4 steps + the final 00Z tick
        assert len(idx) == 13

    def test_noleap_year_has_1460_steps(self):
        idx = build_time_index("2000-01-01", "2000-12-31T18:00", "6h", calendar="noleap")
        assert len(idx) == 1460

    def test_standard_leap_year_has_1464_steps(self):
        idx = build_time_index("2000-01-01", "2000-12-31T18:00", "6h", calendar="standard")
        assert len(idx) == 1464

    def test_noleap_timedelta_freq(self):
        idx = build_time_index("2000-01-01", "2000-01-02", pd.Timedelta("6h"), calendar="noleap")
        assert isinstance(idx, xr.CFTimeIndex)
        assert len(idx) == 5

    def test_360_day_raises(self):
        with pytest.raises(NotImplementedError, match="360_day"):
            build_time_index("2000-01-01", "2000-02-01", "6h", calendar="360_day")

    def test_sampler_style_arithmetic_crosses_leap_boundary(self):
        """The exact operations MultiStepBatchSamplerSubset performs."""
        idx = build_time_index("2000-02-27", "2000-03-02", "6h", calendar="noleap")
        batch = idx[[4, 5]]  # positional fancy indexing, 2000-02-28 00Z / 06Z
        stepped = batch + 4 * pd.Timedelta("6h")  # one day forward
        assert stepped[0] == cftime.DatetimeNoLeap(2000, 3, 1, 0)
        assert stepped[1] == cftime.DatetimeNoLeap(2000, 3, 1, 6)


class TestBuildTimeIndexMulti:
    """Non-contiguous date_ranges support (build_time_index_multi)."""

    BLOCKS = [("1950-01-01", "1965-12-31"), ("1970-01-01", "1985-12-31"), ("1990-01-01", "2005-12-31")]

    def test_gaps_are_entirely_excluded(self):
        idx = build_time_index_multi(
            self.BLOCKS, "6h", "standard", start_margin=pd.Timedelta("18h"), end_margin=pd.Timedelta("6h")
        )
        gap1 = idx[(idx > pd.Timestamp("1965-12-31")) & (idx < pd.Timestamp("1970-01-01"))]
        gap2 = idx[(idx > pd.Timestamp("1985-12-31")) & (idx < pd.Timestamp("1990-01-01"))]
        assert len(gap1) == 0
        assert len(gap2) == 0

    def test_per_block_margin_matches_single_block_formula(self):
        """A single-block call must produce exactly what build_time_index does
        with the same history/forecast margin -- date_ranges is a pure
        generalization, not a different formula."""
        margin_start, margin_end = pd.Timedelta("18h"), pd.Timedelta("6h")
        multi = build_time_index_multi(
            [("1950-01-01", "1965-12-31")], "6h", "standard", start_margin=margin_start, end_margin=margin_end
        )
        single = build_time_index(
            pd.Timestamp("1950-01-01") + margin_start, pd.Timestamp("1965-12-31") - margin_end, "6h", "standard"
        )
        assert multi.equals(single)

    def test_boundary_ticks_are_exactly_the_trimmed_edges(self):
        idx = build_time_index_multi(
            self.BLOCKS, "6h", "standard", start_margin=pd.Timedelta("18h"), end_margin=pd.Timedelta("6h")
        )
        assert idx[0] == pd.Timestamp("1950-01-01 18:00")
        last_before_gap1 = idx[idx <= pd.Timestamp("1965-12-31")][-1]
        first_after_gap1 = idx[idx >= pd.Timestamp("1970-01-01")][0]
        assert last_before_gap1 == pd.Timestamp("1965-12-30 18:00")
        assert first_after_gap1 == pd.Timestamp("1970-01-01 18:00")

    def test_empty_blocks_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_time_index_multi([], "6h", "standard")

    def test_block_with_end_before_start_raises(self):
        with pytest.raises(ValueError, match="end <= start"):
            build_time_index_multi([("1965-12-31", "1950-01-01")], "6h", "standard")

    def test_overlapping_blocks_raise(self):
        with pytest.raises(ValueError, match="overlap"):
            build_time_index_multi([("1950-01-01", "1965-12-31"), ("1960-01-01", "1985-12-31")], "6h", "standard")

    def test_unsorted_input_is_sorted_before_validation(self):
        """Blocks need not be given in order -- they're sorted internally."""
        idx = build_time_index_multi([("1970-01-01", "1985-12-31"), ("1950-01-01", "1965-12-31")], "6h", "standard")
        assert idx[0] < idx[-1]
        assert idx[0] == pd.Timestamp("1950-01-01")

    def test_block_too_short_for_margin_raises(self):
        with pytest.raises(ValueError, match="too short"):
            build_time_index_multi(
                [("1950-01-01", "1950-01-01 12:00")], "6h", "standard", start_margin=pd.Timedelta("1D")
            )

    def test_noleap_calendar(self):
        idx = build_time_index_multi([("2000-01-01", "2000-12-30"), ("2002-01-01", "2002-12-30")], "1D", "noleap")
        assert all(isinstance(t, cftime.datetime) for t in idx)
        assert not any(t.year == 2001 for t in idx)


class TestToCalendar:
    def test_standard_passthrough(self):
        t = pd.Timestamp("2000-06-01T12")
        assert to_calendar(t, "standard") == t

    def test_pd_to_noleap(self):
        t = to_calendar(pd.Timestamp("2000-03-01T06"), "noleap")
        assert t == cftime.DatetimeNoLeap(2000, 3, 1, 6)

    def test_noleap_to_standard(self):
        t = to_calendar(cftime.DatetimeNoLeap(2000, 3, 1, 6), "standard")
        assert t == pd.Timestamp("2000-03-01T06")

    def test_same_calendar_passthrough(self):
        t = cftime.DatetimeNoLeap(2000, 3, 1)
        assert to_calendar(t, "noleap") is t

    def test_alias_365_day(self):
        t = to_calendar(pd.Timestamp("2000-03-01"), "365_day")
        assert not t.calendar == "standard" and t.year == 2000

    def test_feb29_to_noleap_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            to_calendar(pd.Timestamp("2000-02-29"), "noleap")

    def test_string_input(self):
        assert to_calendar("2000-03-01", "noleap") == cftime.DatetimeNoLeap(2000, 3, 1)


class TestEncodeDecodeTime:
    def test_standard_is_unix_ns(self):
        t = pd.Timestamp("2000-06-01T12:00:00")
        assert encode_time(t) == int(t.value)
        assert decode_time(encode_time(t)) == t

    def test_noleap_round_trip(self):
        t = cftime.DatetimeNoLeap(2000, 3, 1, 6)
        rt = decode_time(encode_time(t), "noleap")
        assert rt == t and rt.calendar == t.calendar

    def test_noleap_elapsed_time_across_leap_boundary(self):
        """Differencing encoded values gives true elapsed model time: 6h, no gap."""
        a = encode_time(cftime.DatetimeNoLeap(2000, 2, 28, 18))
        b = encode_time(cftime.DatetimeNoLeap(2000, 3, 1, 0))
        assert b - a == int(pd.Timedelta("6h").value)

    def test_int64_collate_safe(self):
        import torch

        v = encode_time(cftime.DatetimeNoLeap(2050, 1, 1))
        assert torch.tensor([v]).item() == v


class TestCalendarResolutionHelpers:
    def test_normalize_aliases(self):
        assert normalize_calendar("365_day") == "noleap"
        assert normalize_calendar(None) == "standard"
        assert normalize_calendar("Gregorian") == "gregorian"

    def test_is_standard(self):
        assert is_standard_calendar("proleptic_gregorian")
        assert not is_standard_calendar("noleap")

    def test_most_restrictive(self):
        assert most_restrictive_calendar(["standard", "gregorian"]) == "standard"
        assert most_restrictive_calendar(["standard", "noleap"]) == "noleap"
        assert most_restrictive_calendar(["365_day", "noleap"]) == "noleap"
        with pytest.raises(ValueError, match="multiple non-standard"):
            most_restrictive_calendar(["noleap", "all_leap"])


class TestFileLookupAndLabels:
    def test_find_file_accepts_cftime(self):
        intervals = [
            (pd.Timestamp("1999-01-01"), pd.Timestamp("1999-12-31T23:59:59"), "/d/1999.nc"),
            (pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31T23:59:59"), "/d/2000.nc"),
        ]
        assert _find_file(intervals, cftime.DatetimeNoLeap(2000, 3, 1)) == "/d/2000.nc"

    def test_filter_index_by_labels_mixed_types(self):
        master = build_time_index("2000-02-28", "2000-03-01", "6h", calendar="noleap")
        source = pd.date_range("2000-02-28", "2000-03-01", freq="6h")  # includes Feb 29
        kept = filter_index_by_labels(master, source)
        # noleap master has no Feb 29; every remaining label exists in the source
        assert len(kept) == len(master)
        # reverse: standard master filtered by noleap source drops Feb 29
        kept_rev = filter_index_by_labels(source, master)
        assert not any(t.month == 2 and t.day == 29 for t in kept_rev)


# --------------------------------------------------------------------------- #
# Synthetic data fixtures
# --------------------------------------------------------------------------- #

NLAT, NLON, NLEV = 5, 7, 3


def _write_source_file(path, times, with_3d=True):
    """Write a small netcdf with the given time coordinate (list of datetimes)."""
    nt = len(times)
    data_vars = {
        "SP": (("time", "latitude", "longitude"), np.random.rand(nt, NLAT, NLON).astype(np.float32)),
    }
    if with_3d:
        data_vars["T"] = (
            ("time", "level", "latitude", "longitude"),
            np.random.rand(nt, NLEV, NLAT, NLON).astype(np.float32),
        )
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "level": np.arange(NLEV),
            "latitude": np.linspace(-60, 60, NLAT),
            "longitude": np.linspace(0, 300, NLON),
        },
    )
    ds.to_netcdf(path)
    return str(path)


def _source_config(path, name="CESM", **extra):
    src = {
        "dataset_type": "local",
        "level_coord": "level",
        "variables": {
            "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"], "path": str(path)},
            "dynamic_forcing": None,
            "static": None,
            "diagnostic": None,
        },
        **extra,
    }
    return {name: src}


def _data_config(source, start, end, forecast_len=1, **extra):
    return {
        "source": source,
        "start_datetime": start,
        "end_datetime": end,
        "timestep": "6h",
        "forecast_len": forecast_len,
        **extra,
    }


@pytest.fixture
def noleap_file(tmp_path):
    times = xr.date_range("2000-02-26", "2000-03-04", freq="6h", calendar="noleap", use_cftime=True)
    return _write_source_file(tmp_path / "noleap_2000.nc", list(times))


@pytest.fixture
def standard_file(tmp_path):
    times = pd.date_range("2000-02-26", "2000-03-04", freq="6h")
    return _write_source_file(tmp_path / "standard_2000.nc", list(times))


# --------------------------------------------------------------------------- #
# 2 + 3. LocalDataset integration: golden standard path and noleap path
# --------------------------------------------------------------------------- #


class TestLocalDatasetStandardGolden:
    """The standard-calendar path must be bit-identical to plain pandas."""

    def test_clock_type_and_values(self, standard_file):
        cfg = _data_config(_source_config(standard_file), "2000-02-26", "2000-03-04")
        ds = LocalDataset(cfg, return_target=True)
        assert ds.calendar == "standard"
        assert type(ds.datetimes) is pd.DatetimeIndex
        assert ds.datetimes.equals(
            pd.date_range("2000-02-26", pd.Timestamp("2000-03-04") - pd.Timedelta("6h"), freq="6h")
        )

    def test_metadata_is_plain_unix_ns(self, standard_file):
        cfg = _data_config(_source_config(standard_file), "2000-02-26", "2000-03-04")
        ds = LocalDataset(cfg, return_target=True)
        t = pd.Timestamp("2000-02-28T06")
        sample = ds[(t, 0)]
        assert sample["metadata"]["input_datetime"] == int(t.value)
        assert sample["metadata"]["target_datetime"] == int((t + pd.Timedelta("6h")).value)
        assert ds.static_metadata["datetime_fmt"] == "unix_ns"
        assert ds.static_metadata["calendar"] == "standard"


class TestLocalDatasetNoleap:
    def test_finds_calendar_and_builds_cftime_clock(self, noleap_file):
        cfg = _data_config(_source_config(noleap_file), "2000-02-26", "2000-03-04")
        ds = LocalDataset(cfg, return_target=True)
        assert ds.calendar == "noleap"
        assert isinstance(ds.datetimes, xr.CFTimeIndex)
        assert not any(t.month == 2 and t.day == 29 for t in ds.datetimes)
        assert ds.static_metadata["datetime_fmt"] == "cf_ns:noleap"

    def test_explicit_config_calendar_wins_over_found_calendar(self, noleap_file):
        source = _source_config(noleap_file, calendar="noleap")
        cfg = _data_config(source, "2000-02-26", "2000-03-04")
        ds = LocalDataset(cfg, return_target=False)
        assert ds.calendar == "noleap"

    def test_getitem_across_leap_boundary(self, noleap_file):
        """Feb 28 18Z input -> Mar 1 00Z target: the pair the old code crashed on."""
        cfg = _data_config(_source_config(noleap_file), "2000-02-26", "2000-03-04")
        ds = LocalDataset(cfg, return_target=True)
        t = cftime.DatetimeNoLeap(2000, 2, 28, 18)
        sample = ds[(t, 0)]
        assert sample["input"]["CESM/prognostic/3d/T"].shape == (NLEV, 1, NLAT, NLON)
        target_dt = decode_time(sample["metadata"]["target_datetime"], "noleap")
        assert target_dt == cftime.DatetimeNoLeap(2000, 3, 1, 0)

    def test_standard_master_timestamp_feb29_raises_loudly(self, noleap_file):
        cfg = _data_config(_source_config(noleap_file), "2000-02-26", "2000-03-04")
        ds = LocalDataset(cfg, return_target=False)
        with pytest.raises(ValueError, match="does not exist"):
            ds[(pd.Timestamp("2000-02-29T00"), 0)]

    def test_sampler_multistep_rollout_crosses_boundary(self, noleap_file):
        """Full sampler -> dataset loop over the leap boundary, 5-step rollout."""
        n_steps = 5
        cfg = _data_config(_source_config(noleap_file), "2000-02-26", "2000-03-04", forecast_len=n_steps)
        ds = LocalDataset(cfg, return_target=True)
        # init at 2000-02-28 06Z: steps land on 12Z, 18Z, Mar 1 00Z, 06Z
        init_pos = list(ds.datetimes).index(cftime.DatetimeNoLeap(2000, 2, 28, 6))
        sampler = MultiStepBatchSamplerSubset(ds, batch_size=1, index_subset=[init_pos], num_forecast_steps=n_steps)

        seen = []
        for batch in sampler:
            (t, i) = batch[0]
            sample = ds[(t, i)]  # must load without error at every step
            seen.append(decode_time(sample["metadata"]["input_datetime"], "noleap"))
        assert len(seen) == n_steps
        assert seen[0] == cftime.DatetimeNoLeap(2000, 2, 28, 6)
        assert seen[-1] == cftime.DatetimeNoLeap(2000, 3, 1, 6)
        assert not any(t.month == 2 and t.day == 29 for t in seen)
        # consecutive steps are exactly 6h of model time apart
        encoded = [encode_time(t) for t in seen]
        assert all(b - a == int(pd.Timedelta("6h").value) for a, b in zip(encoded, encoded[1:]))


# --------------------------------------------------------------------------- #
# 4. MultiSourceDataset master-clock rules
# --------------------------------------------------------------------------- #


class TestMultiSourceCalendar:
    def test_noleap_source_sets_master_calendar(self, noleap_file):
        cfg = _data_config(_source_config(noleap_file), "2000-02-26", "2000-03-04")
        msd = MultiSourceDataset(cfg, return_target=False)
        assert msd.calendar == "noleap"
        assert isinstance(msd.datetimes, xr.CFTimeIndex)
        assert not any(t.month == 2 and t.day == 29 for t in msd.datetimes)

    def test_mixed_noleap_and_standard_sources(self, noleap_file, standard_file):
        source = {
            **_source_config(noleap_file, name="CESM"),
            **_source_config(standard_file, name="OBS"),
        }
        cfg = _data_config(source, "2000-02-26", "2000-03-04")
        msd = MultiSourceDataset(cfg, return_target=False)
        assert msd.calendar == "noleap"
        # master clock intersected across calendars by label; Feb 29 absent,
        # everything else present in both sources
        assert len(msd.datetimes) > 0
        assert not any(t.month == 2 and t.day == 29 for t in msd.datetimes)
        # both sources can serve a post-boundary timestamp
        sample = msd[(cftime.DatetimeNoLeap(2000, 3, 1, 0), 0)]
        assert "CESM" in sample["input"] and "OBS" in sample["input"]

    def test_explicit_standard_master_with_noleap_source_raises(self, noleap_file):
        cfg = _data_config(_source_config(noleap_file), "2000-02-26", "2000-03-04", calendar="standard")
        with pytest.raises(ValueError, match="most restrictive"):
            MultiSourceDataset(cfg, return_target=False)

    def test_injected_datetimes_converted_to_master_calendar(self, noleap_file):
        """The rollout_gen2 inference path: init times injected as `datetimes`."""
        inits = [pd.Timestamp("2000-02-27T00"), pd.Timestamp("2000-03-01T00")]
        cfg = _data_config(_source_config(noleap_file), "2000-02-26", "2000-03-04", datetimes=inits)
        msd = MultiSourceDataset(cfg, return_target=False)
        assert isinstance(msd.datetimes, xr.CFTimeIndex)
        assert msd.datetimes[1] == cftime.DatetimeNoLeap(2000, 3, 1, 0)

    def test_injected_feb29_init_raises(self, noleap_file):
        cfg = _data_config(
            _source_config(noleap_file), "2000-02-26", "2000-03-04", datetimes=[pd.Timestamp("2000-02-29T00")]
        )
        with pytest.raises(ValueError, match="does not exist"):
            MultiSourceDataset(cfg, return_target=False)

    def test_standard_only_golden(self, standard_file):
        """Standard-calendar multi-source master clock is unchanged."""
        cfg = _data_config(_source_config(standard_file), "2000-02-26", "2000-03-04")
        msd = MultiSourceDataset(cfg, return_target=False)
        assert msd.calendar == "standard"
        assert type(msd.datetimes) is pd.DatetimeIndex
        expected = pd.date_range("2000-02-26", pd.Timestamp("2000-03-04") - pd.Timedelta("6h"), freq="6h")
        assert msd.datetimes.equals(expected)


# --------------------------------------------------------------------------- #
# 5. Inference init schedule
# --------------------------------------------------------------------------- #


class TestBatchInitTimes:
    def test_standard_unchanged(self):
        conf = {"first_init_date": "2000-02-27", "last_init_date": "2000-03-02", "init_interval": "1d"}
        inits = batch_init_times(conf)
        assert inits == [pd.Timestamp("2000-02-27") + i * pd.Timedelta("1d") for i in range(5)]

    def test_noleap_schedule_skips_feb29(self):
        conf = {"first_init_date": "2000-02-27", "last_init_date": "2000-03-02", "init_interval": "1d"}
        inits = batch_init_times(conf, calendar="noleap")
        assert all(isinstance(t, cftime.datetime) for t in inits)
        assert not any(t.month == 2 and t.day == 29 for t in inits)
        # elapsed stepping: 27, 28, Mar 1, Mar 2 -> 4 inits (vs 5 Gregorian)
        assert len(inits) == 4


# --------------------------------------------------------------------------- #
# 6. to_cycle_year (temporal_mode: cyclic)
# --------------------------------------------------------------------------- #


class TestToCycleYear:
    def test_basic_remap_preserves_month_day_time(self):
        """Only the year changes; month/day/hour/minute/second are preserved."""
        t = pd.Timestamp("1985-07-15 12:30:05")
        result = to_cycle_year(t, 2000, "standard")
        assert result == pd.Timestamp("2000-07-15 12:30:05")

    def test_feb29_clamped_when_cycle_year_not_leap(self):
        """A real Feb 29 (from an actual leap real year) remapped onto a non-leap
        cycle_year clamps to Feb 28 instead of raising."""
        t = pd.Timestamp("1988-02-29 06:00")
        result = to_cycle_year(t, 2021, "standard")
        assert result == pd.Timestamp("2021-02-28 06:00")

    def test_feb29_preserved_when_cycle_year_is_leap(self):
        t = pd.Timestamp("1988-02-29 06:00")
        result = to_cycle_year(t, 2000, "standard")
        assert result == pd.Timestamp("2000-02-29 06:00")

    def test_feb28_unaffected_by_clamping(self):
        """A real Feb 28 (non-leap real year) is never touched by the clamp logic."""
        t = pd.Timestamp("1990-02-28 18:00")
        result = to_cycle_year(t, 2021, "standard")
        assert result == pd.Timestamp("2021-02-28 18:00")

    def test_noleap_calendar_clamps_regardless_of_cycle_year(self):
        """noleap never has a Feb 29, for any cycle_year."""
        t = pd.Timestamp("1988-02-29 06:00")
        result = to_cycle_year(t, 2000, "noleap")
        assert isinstance(result, cftime.datetime)
        assert (result.month, result.day) == (2, 28)
        assert normalize_calendar(result.calendar) == "noleap"

    def test_all_leap_calendar_keeps_feb29_regardless_of_cycle_year(self):
        t = pd.Timestamp("1988-02-29 06:00")
        result = to_cycle_year(t, 2021, "all_leap")
        assert isinstance(result, cftime.datetime)
        assert (result.month, result.day) == (2, 29)

    def test_julian_calendar_leap_rule_has_no_century_exception(self):
        """Julian: leap every 4 years, no Gregorian century exception -- 1900 is leap."""
        t = pd.Timestamp("1988-02-29 06:00")
        result = to_cycle_year(t, 1900, "julian")
        assert (result.month, result.day) == (2, 29)
