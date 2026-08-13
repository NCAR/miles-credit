"""Unit tests for the file-mapping helpers in credit/datasets/gen_2/_utils.py."""

import pandas as pd
import pytest

from credit.datasets.gen_2._utils import _extract_time_fmt, _find_file, _map_files


class TestMapFiles:
    def test_single_file_covers_all_time(self):
        intervals = _map_files(["/data/era5.zarr"], "%Y")
        assert intervals == [(pd.Timestamp.min, pd.Timestamp.max, "/data/era5.zarr")]

    def test_plain_year_files(self):
        files = ["/data/era5_2020.zarr", "/data/era5_2019.zarr"]
        intervals = _map_files(files, "%Y", "/data/era5_%Y.zarr")
        assert [iv[2] for iv in intervals] == ["/data/era5_2019.zarr", "/data/era5_2020.zarr"]
        assert intervals[0][0] == pd.Timestamp("2019-01-01")
        assert intervals[0][1].year == 2019

    def test_template_with_glob_suffix(self):
        """Regression: glob wildcards in the template must not become literal regex chars.

        example-end-to-end.yml uses '..._subset_%Y*' matching files like
        '..._subset_1979_conserve.zarr'.
        """
        template = "/data/ERA5_subset_%Y*"
        files = ["/data/ERA5_subset_1979_conserve.zarr", "/data/ERA5_subset_1980_conserve.zarr"]
        intervals = _map_files(files, _extract_time_fmt(template), template)
        assert intervals[0][0] == pd.Timestamp("1979-01-01")
        assert intervals[1][0] == pd.Timestamp("1980-01-01")

    def test_template_with_glob_question_mark(self):
        template = "/data/run?_%Y.nc"
        files = ["/data/runA_2001.nc", "/data/runB_2002.nc"]
        intervals = _map_files(files, "%Y", template)
        assert [iv[0].year for iv in intervals] == [2001, 2002]

    def test_anchoring_skips_literal_digits_before_placeholder(self):
        """The case the anchored branch exists for: literal year in the prefix."""
        template = "/data/branch_1980_%Y_data.zarr"
        files = ["/data/branch_1980_1990_data.zarr", "/data/branch_1980_1991_data.zarr"]
        intervals = _map_files(files, "%Y", template)
        assert [iv[0].year for iv in intervals] == [1990, 1991]

    def test_glob_and_literal_digits_combined(self):
        template = "/data/branch_1980_%Y_*.zarr"
        files = ["/data/branch_1980_1990_a.zarr", "/data/branch_1980_1991_b.zarr"]
        intervals = _map_files(files, "%Y", template)
        assert [iv[0].year for iv in intervals] == [1990, 1991]

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="did not match path"):
            _map_files(["/data/a.nc", "/data/b.nc"], "%Y", "/data/era5_%Y.nc")

    def test_monthly_granularity(self):
        files = ["/data/solar_2021_06.nc", "/data/solar_2021_07.nc"]
        intervals = _map_files(files, "%Y_%m", "/data/solar_%Y_%m.nc")
        assert intervals[0][0] == pd.Timestamp("2021-06-01")
        assert intervals[0][1] < pd.Timestamp("2021-07-01")


class TestExtractTimeFmt:
    def test_simple(self):
        assert _extract_time_fmt("/data/era5_%Y.nc") == "%Y"

    def test_spanning_literals(self):
        assert _extract_time_fmt("/data/%Y/%m/era5.nc") == "%Y/%m"

    def test_glob_after_code_not_included(self):
        assert _extract_time_fmt("/data/era5_%Y*") == "%Y"


class TestFindFile:
    def test_lookup_and_out_of_range(self):
        intervals = _map_files(["/data/era5_2019.zarr", "/data/era5_2020.zarr"], "%Y", "/data/era5_%Y.zarr")
        assert _find_file(intervals, pd.Timestamp("2019-06-01")) == "/data/era5_2019.zarr"
        assert _find_file(intervals, pd.Timestamp("2020-12-31T23:00")) == "/data/era5_2020.zarr"
        with pytest.raises(KeyError):
            _find_file(intervals, pd.Timestamp("2021-01-01"))
