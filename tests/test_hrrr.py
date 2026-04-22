"""
tests/test_hrrr.py
------------------
Unit tests for credit/datasets/hrrr.py covering path helpers, .idx parsers,
and product-specific entry-map functions.

Remote/dataset integration tests are skipped unless the environment variable
``HRRR_TEST_REMOTE=1`` is set (they hit real AWS endpoints).
"""

import os
import textwrap

import pandas as pd
import pytest

from credit.datasets.hrrr import (
    VALID_PRODUCTS,
    _build_nat_entry_map,
    _build_prs_entry_map,
    _find_subhf_entry,
    _hrrr_local_path,
    _hrrr_s3_uri,
    _parse_idx,
    _resolve_nat_levels,
    _resolve_pressure_levels,
    HRRRDataset,
)

# ---------------------------------------------------------------------------
# Constants / product registry
# ---------------------------------------------------------------------------


def test_valid_products():
    assert VALID_PRODUCTS == {
        "HRRR": "wrfprsf", 
        "HRRR_NAT": "wrfnatf", 
        "HRRR_SUBH": "wrfsubhf"
    }


# ---------------------------------------------------------------------------
# Path helpers — all three products
# ---------------------------------------------------------------------------

_T_V3 = pd.Timestamp("2022-01-01 06:00")  # v3/v4 (after cutoff)
_T_V2 = pd.Timestamp("2018-01-01 12:00")  # v1/v2 (before cutoff)


def test_s3_uri_wrfprsf_v3():
    uri = _hrrr_s3_uri(_T_V3, forecast_hour=0, product="wrfprsf")
    assert uri == "s3://noaa-hrrr-bdp-pds/hrrr.20220101/conus/hrrr.t06z.wrfprsf00.grib2"


def test_s3_uri_wrfnatf_v3():
    uri = _hrrr_s3_uri(_T_V3, forecast_hour=1, product="wrfnatf")
    assert uri == "s3://noaa-hrrr-bdp-pds/hrrr.20220101/conus/hrrr.t06z.wrfnatf01.grib2"


def test_s3_uri_wrfsubhf_v3():
    uri = _hrrr_s3_uri(_T_V3, forecast_hour=2, product="wrfsubhf")
    assert uri == "s3://noaa-hrrr-bdp-pds/hrrr.20220101/conus/hrrr.t06z.wrfsubhf02.grib2"


def test_s3_uri_v2_no_conus():
    uri = _hrrr_s3_uri(_T_V2, forecast_hour=0, product="wrfprsf")
    assert "conus" not in uri
    assert "hrrr.20180101/hrrr.t12z.wrfprsf00.grib2" in uri


def test_s3_uri_default_product():
    """Default product is wrfprsf."""
    uri_explicit = _hrrr_s3_uri(_T_V3, forecast_hour=0, product="wrfprsf")
    uri_default = _hrrr_s3_uri(_T_V3, forecast_hour=0)
    assert uri_explicit == uri_default


def test_local_path_wrfnatf():
    path = _hrrr_local_path("/data/hrrr", _T_V3, forecast_hour=0, product="wrfnatf")
    assert path.endswith("hrrr.t06z.wrfnatf00.grib2")
    assert "conus" in path


def test_local_path_wrfsubhf():
    path = _hrrr_local_path("/data/hrrr", _T_V3, forecast_hour=1, product="wrfsubhf")
    assert path.endswith("hrrr.t06z.wrfsubhf01.grib2")


# ---------------------------------------------------------------------------
# _parse_idx — step field added
# ---------------------------------------------------------------------------

_IDX_TEXT = textwrap.dedent("""\
    1:0:d=2022010106:TMP:500 mb:anl:
    2:12345:d=2022010106:TMP:700 mb:anl:
    3:23456:d=2022010106:UGRD:500 mb:anl:
    4:34567:d=2022010106:TMP:2 m above ground:anl:
""")


def test_parse_idx_basic():
    entries = _parse_idx(_IDX_TEXT)
    assert len(entries) == 4
    assert entries[0]["var"] == "TMP"
    assert entries[0]["level"] == "500 mb"
    assert entries[0]["step"] == "anl"
    assert entries[0]["byte_start"] == 0
    assert entries[0]["byte_end"] == 12344  # next entry start - 1
    assert entries[-1]["byte_end"] is None  # last entry


def test_parse_idx_step_field():
    text = textwrap.dedent("""\
        1:0:d=2022010106:TMP:2 m above ground:15 min fcst:
        2:5000:d=2022010106:TMP:2 m above ground:30 min fcst:
    """)
    entries = _parse_idx(text)
    assert entries[0]["step"] == "15 min fcst"
    assert entries[1]["step"] == "30 min fcst"


# ---------------------------------------------------------------------------
# _build_prs_entry_map / _resolve_pressure_levels
# ---------------------------------------------------------------------------


def _make_prs_entries():
    return [
        {"var": "TMP", "level": "500 mb", "step": "anl", "byte_start": 0, "byte_end": 100},
        {"var": "TMP", "level": "700 mb", "step": "anl", "byte_start": 101, "byte_end": 200},
        {"var": "TMP", "level": "850 mb", "step": "anl", "byte_start": 201, "byte_end": 300},
        {"var": "UGRD", "level": "500 mb", "step": "anl", "byte_start": 301, "byte_end": 400},
    ]


def test_build_prs_entry_map():
    entries = _make_prs_entries()
    prs_map = _build_prs_entry_map(entries, "TMP")
    assert set(prs_map.keys()) == {500.0, 700.0, 850.0}
    assert prs_map[500.0]["byte_start"] == 0


def test_resolve_pressure_levels_all():
    prs_map = _build_prs_entry_map(_make_prs_entries(), "TMP")
    levels = _resolve_pressure_levels(None, prs_map, "T")
    assert levels == sorted(prs_map.keys(), reverse=True)


def test_resolve_pressure_levels_subset():
    prs_map = _build_prs_entry_map(_make_prs_entries(), "TMP")
    levels = _resolve_pressure_levels([500, 850], prs_map, "T")
    assert 500.0 in levels and 850.0 in levels


def test_resolve_pressure_levels_missing():
    prs_map = _build_prs_entry_map(_make_prs_entries(), "TMP")
    with pytest.raises(ValueError, match="999"):
        _resolve_pressure_levels([999], prs_map, "T")


# ---------------------------------------------------------------------------
# _build_nat_entry_map / _resolve_nat_levels
# ---------------------------------------------------------------------------


def _make_nat_entries():
    return [
        {"var": "TMP", "level": "10 hybrid level", "step": "anl", "byte_start": 0, "byte_end": 100},
        {"var": "TMP", "level": "20 hybrid level", "step": "anl", "byte_start": 101, "byte_end": 200},
        {"var": "TMP", "level": "30 hybrid level", "step": "anl", "byte_start": 201, "byte_end": 300},
        {"var": "UGRD", "level": "10 hybrid level", "step": "anl", "byte_start": 301, "byte_end": 400},
        {"var": "TMP", "level": "500 mb", "step": "anl", "byte_start": 401, "byte_end": 500},
    ]


def test_build_nat_entry_map():
    entries = _make_nat_entries()
    nat_map = _build_nat_entry_map(entries, "TMP")
    assert set(nat_map.keys()) == {10, 20, 30}
    # pressure-level entry must be excluded
    assert 500 not in nat_map
    assert nat_map[10]["byte_start"] == 0


def test_build_nat_entry_map_other_var():
    entries = _make_nat_entries()
    nat_map = _build_nat_entry_map(entries, "UGRD")
    assert set(nat_map.keys()) == {10}


def test_resolve_nat_levels_all():
    nat_map = _build_nat_entry_map(_make_nat_entries(), "TMP")
    levels = _resolve_nat_levels(None, nat_map, "T")
    assert levels == [10, 20, 30]


def test_resolve_nat_levels_subset():
    nat_map = _build_nat_entry_map(_make_nat_entries(), "TMP")
    levels = _resolve_nat_levels([10, 30], nat_map, "T")
    assert levels == [10, 30]


def test_resolve_nat_levels_missing():
    nat_map = _build_nat_entry_map(_make_nat_entries(), "TMP")
    with pytest.raises(ValueError, match="99"):
        _resolve_nat_levels([99], nat_map, "T")


# ---------------------------------------------------------------------------
# _find_subhf_entry
# ---------------------------------------------------------------------------


def _make_subhf_entries():
    return [
        {"var": "TMP", "level": "2 m above ground", "step": "15 min fcst", "byte_start": 0, "byte_end": 100},
        {"var": "TMP", "level": "2 m above ground", "step": "30 min fcst", "byte_start": 101, "byte_end": 200},
        {"var": "TMP", "level": "2 m above ground", "step": "45 min fcst", "byte_start": 201, "byte_end": 300},
        {"var": "TMP", "level": "2 m above ground", "step": "60 min fcst", "byte_start": 301, "byte_end": 400},
    ]


def test_find_subhf_entry_hit():
    entries = _make_subhf_entries()
    e = _find_subhf_entry(entries, "TMP", "2 m above ground", 30)
    assert e["byte_start"] == 101


def test_find_subhf_entry_all_steps():
    entries = _make_subhf_entries()
    for step in [15, 30, 45, 60]:
        e = _find_subhf_entry(entries, "TMP", "2 m above ground", step)
        assert e["step"] == f"{step} min fcst"


def test_find_subhf_entry_miss():
    entries = _make_subhf_entries()
    with pytest.raises(KeyError, match="75 min fcst"):
        _find_subhf_entry(entries, "TMP", "2 m above ground", 75)


# ---------------------------------------------------------------------------
# HRRRDataset instantiation (no I/O)
# ---------------------------------------------------------------------------


def _make_config(source_key="HRRR", **extra_source):
    source_defaults = {
        "mode": "remote",
        "forecast_hour": 0,
        "levels": [500, 700, 850],
        "variables": {
            "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
        },
    }
    source_defaults.update(extra_source)
    return {
        "source": {source_key: source_defaults},
        "start_datetime": "2022-01-01 00:00",
        "end_datetime": "2022-01-02 00:00",
        "timestep": "1h",
        "forecast_len": 0,
    }


def test_hrrr_dataset_wrfprsf_defaults():
    cfg = _make_config("HRRR")
    ds = HRRRDataset(cfg)
    assert ds.product == "wrfprsf"
    assert ds.source_name == "hrrr"
    assert len(ds) == 25  # 2022-01-01 00:00 … 2022-01-02 00:00 inclusive (25 h)


def test_hrrr_dataset_wrfnatf():
    cfg = _make_config("HRRR_NAT", variables={"prognostic": {"vars_3D": ["T", "U"], "vars_2D": []}})
    ds = HRRRDataset(cfg)
    assert ds.product == "wrfnatf"
    assert ds.source_name == "hrrr_nat"


def test_hrrr_dataset_wrfsubhf():
    cfg = _make_config(
        "HRRR_SUBH",
        variables={"prognostic": {"vars_3D": [], "vars_2D": ["t2m"]}},
    )
    cfg["timestep"] = "15min"
    ds = HRRRDataset(cfg)
    assert ds.product == "wrfsubhf"
    assert ds.source_name == "hrrr_subh"


def test_hrrr_dataset_invalid_product():
    cfg = _make_config("HRRR_BADPRODUCT")
    with pytest.raises(ValueError, match="Unknown HRRR product"):
        HRRRDataset(cfg)


def test_hrrr_dataset_missing_source():
    cfg = _make_config("HRRR_MISSING")
    del cfg["source"]
    with pytest.raises(ValueError, match="Missing 'source' key in config"):
        HRRRDataset(cfg)


def test_hrrr_dataset_wrong_config_hierarchy_passed_higher():
    cfg = _make_config("HRRR")
    data_cfg = {"data": cfg}

    with pytest.raises(ValueError, match="Missing 'source' key in config"):
        HRRRDataset(data_cfg)


def test_hrrr_dataset_wrong_config_hierarchy_passed_lower():
    cfg = _make_config("HRRR")
    with pytest.raises(ValueError, match="Missing 'source' key in config"):
        HRRRDataset(cfg["source"]["HRRR"])


def test_hrrr_dataset_only_one_of_multiple_sources():
    cfg = _make_config("HRRR")
    rest_cfg = {k: v for k, v in cfg.items() if k != "source"}

    other_dataset_name = ["ERA5", "MRMS", "NOT_VALID_DATASET"]

    for other in other_dataset_name:
        multi_source_cfg = {
            "source": {
                other: cfg["source"]["HRRR"],
                "HRRR": cfg["source"]["HRRR"],
            },
            **rest_cfg,
        }
        with pytest.raises(ValueError, match="Expected exactly one source in config"):
            HRRRDataset(multi_source_cfg)


# ---------------------------------------------------------------------------
# Remote integration tests (skipped unless HRRR_TEST_REMOTE=1)
# ---------------------------------------------------------------------------

SKIP_REMOTE = not os.getenv("HRRR_TEST_REMOTE")


@pytest.mark.skipif(SKIP_REMOTE, reason="Set HRRR_TEST_REMOTE=1 to run remote tests")
def test_hrrr_remote_wrfprsf_getitem():
    cfg = _make_config(
        "HRRR", 
        levels=[500, 700], 
        variables={"prognostic": {"vars_3D": ["T"], "vars_2D": ["t2m"]}})
    ds = HRRRDataset(cfg)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]
    assert "hrrr/prognostic/3d/T" in sample["input"]
    assert sample["input"]["hrrr/prognostic/3d/T"].shape == (2, 1, *sample["input"]["hrrr/prognostic/3d/T"].shape[2:])


@pytest.mark.skipif(SKIP_REMOTE, reason="Set HRRR_TEST_REMOTE=1 to run remote tests")
def test_hrrr_remote_wrfnatf_getitem():
    cfg = _make_config(
        "HRRR_NAT", 
        levels=[10, 20], 
        variables={"prognostic": {"vars_3D": ["T"], "vars_2D": []}})
    ds = HRRRDataset(cfg)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]
    assert "hrrr_nat/prognostic/3d/T" in sample["input"]
