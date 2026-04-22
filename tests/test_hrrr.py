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
import numpy as np

import pytest

from credit.datasets.hrrr import (
    VALID_PRODUCTS,
    _HRRR_HTTPS_BASE,
    _build_nat_entry_map,
    _build_prs_entry_map,
    _find_subhf_entry,
    _hrrr_local_path,
    _hrrr_s3_uri,
    _parse_idx,
    _resolve_nat_levels,
    _resolve_pressure_levels,
    _s3_uri_to_https,
    HRRRDataset,
)

# ---------------------------------------------------------------------------
# Constants / product registry
# ---------------------------------------------------------------------------


def test_valid_products():
    assert VALID_PRODUCTS == {"HRRR": "wrfprsf", "HRRR_NAT": "wrfnatf", "HRRR_SUBH": "wrfsubhf"}


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
    assert "conus" in path


def test_local_path_v2_no_conus():
    path = _hrrr_local_path("/data/hrrr", _T_V2, forecast_hour=0, product="wrfprsf")
    assert path.endswith("hrrr.t12z.wrfprsf00.grib2")
    assert "conus" not in path


# ---------------------------------------------------------------------------
# s3 uri to https


def test_s3_uri_to_https():
    s3_uri = _hrrr_s3_uri(_T_V3, forecast_hour=0, product="wrfprsf")
    assert s3_uri.startswith("s3://")
    https_url = _s3_uri_to_https(s3_uri)
    assert https_url.startswith(_HRRR_HTTPS_BASE + "/")
    start_len = len(_HRRR_HTTPS_BASE) + 1
    assert https_url[start_len + 1] != "/"  # no double slashes in path


# ---------------------------------------------------------------------------
# _parse_idx — step field added
# ---------------------------------------------------------------------------

_IDX_TEXT = textwrap.dedent("""\
    1:0:d=2022010106:TMP:500 mb:anl:
    2:12345:d=2022010106:TMP:700 mb:anl:
    3:23456:d=2022010106:UGRD:500 mb:anl:
    4:34567:d=2022010106:DPT:2 m above ground:anl:
""")


def test_parse_idx_basic():
    entries = _parse_idx(_IDX_TEXT)
    assert len(entries) == 4
    # First entry
    assert entries[0]["var"] == "TMP"
    assert entries[0]["level"] == "500 mb"
    assert entries[0]["step"] == "anl"
    assert entries[0]["byte_start"] == 0
    assert entries[0]["byte_end"] == 12344  # next entry start - 1
    # Last entry
    assert entries[-1]["var"] == "DPT"
    assert entries[-1]["level"] == "2 m above ground"
    assert entries[-1]["step"] == "anl"
    assert entries[-1]["byte_start"] == 34567
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


def test_hrrr_local_no_base_path():
    cfg = _make_config("HRRR", mode="local")
    assert "base_path" not in cfg["source"]["HRRR"]
    with pytest.raises(ValueError, match="Missing 'base_path'"):
        HRRRDataset(cfg)


# ---------------------------------------------------------------------------
# HRRRDataset variable types
# ---------------------------------------------------------------------------


def test_hrrr_dataset_variable_types():
    cfg = _make_config(
        "HRRR",
        variables={
            "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m", "d2m"]},
            "diagnostic": {"vars_3D": ["RH"], "vars_2D": ["sp"]},
            "dynamic_forcing": {"vars_3D": ["Q"], "vars_2D": ["dswrf"]},
        },
    )
    ds = HRRRDataset(cfg)
    var_dict = ds.var_dict
    assert var_dict["prognostic"]["vars_3D"] == ["T", "U"]
    assert var_dict["prognostic"]["vars_2D"] == ["t2m", "d2m"]
    assert var_dict["diagnostic"]["vars_3D"] == ["RH"]
    assert var_dict["diagnostic"]["vars_2D"] == ["sp"]
    assert var_dict["dynamic_forcing"]["vars_3D"] == ["Q"]
    assert var_dict["dynamic_forcing"]["vars_2D"] == ["dswrf"]


def test_hrrr_dataset_unsupported_static_variables():
    cfg = _make_config(
        "HRRR",
        variables={
            "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
            "static": {
                "vars_3D": ["orog"],
                "vars_2D": [],
            },  # Currently unsupported field type for HRRR datasets, but can change in the future
        },
    )
    with pytest.raises(KeyError, match="Unknown field_type 'static'"):
        HRRRDataset(cfg)


def test_hrrr_dataset_unsupported_no_variables():
    cfg = _make_config("HRRR", variables={})
    del cfg["source"]["HRRR"]["variables"]  # Simulate user forgetting to include "variables" key
    with pytest.raises(KeyError, match="Missing 'variables' key"):
        HRRRDataset(cfg)


def test_hrrr_dataset_unsupported_empty_variables():
    cfg = _make_config("HRRR", variables={})
    with pytest.raises(ValueError, match="No variables specified"):
        HRRRDataset(cfg)


# ---------------------------------------------------------------------------
# Spatial Slicing
# ---------------------------------------------------------------------------


def _make_large_extent_dict():
    # Larger than HRRR CONUS Domain
    return {
        "lon_min": -125,
        "lon_max": -60,
        "lat_min": 21,
        "lat_max": 49,
    }


def _make_small_inner_extent_dict():
    # Specifically chosen since this exhibits the overbounding used in the current spatial slicing
    return {
        "lon_min": -100,
        "lon_max": -80,
        "lat_min": 30,
        "lat_max": 45,
    }


def _make_extent_from_dict(extent_dict):
    return [extent_dict["lon_min"], extent_dict["lon_max"], extent_dict["lat_min"], extent_dict["lat_max"]]


def _make_example_lat_lon_array_from_northwest_corner():
    # Pulled from HRRR pygrib file
    lat_array = np.array(
        [
            [21.138123, 21.14511004, 21.1520901, 21.1590632, 21.16602932, 21.17298847, 21.17994064],
            [21.16299459, 21.1699845, 21.17696744, 21.18394341, 21.1909124, 21.19787441, 21.20482944],
            [21.18786863, 21.19486142, 21.20184723, 21.20882607, 21.21579793, 21.22276281, 21.22972071],
            [21.21274513, 21.21974079, 21.22672948, 21.23371119, 21.24068592, 21.24765367, 21.25461443],
            [21.23762407, 21.24462262, 21.25161418, 21.25859877, 21.26557637, 21.27254698, 21.2795106],
        ]
    )
    lon_array = np.array(
        [
            [-122.719528, -122.69286132, -122.6661903, -122.63951495, -122.61283526, -122.58615124, -122.5594629],
            [-122.72702499, -122.70035119, -122.67367305, -122.64699057, -122.62030375, -122.59361261, -122.56691713],
            [-122.73452632, -122.7078454, -122.68116014, -122.65447053, -122.62777658, -122.6010783, -122.57437568],
            [-122.74203201, -122.71534397, -122.68865157, -122.66195483, -122.63525374, -122.60854832, -122.58183856],
            [-122.74954205, -122.72284688, -122.69614735, -122.66944347, -122.64273525, -122.61602268, -122.58930577],
        ]
    )

    assert lat_array.shape == lon_array.shape == (5, 7)

    return lat_array, lon_array


def make_example_lat_lon_array_from_southeast_corner():
    # Pulled from HRRR pygrib file
    lat_array = np.array(
        [
            [47.82395705, 47.81369093, 47.80341474, 47.79312849],
            [47.84850201, 47.83823259, 47.8279531, 47.81766354],
            [47.87304341, 47.86277069, 47.85248789, 47.84219502],
        ]
    )
    lon_array = np.array(
        [
            [-61.05747982, -61.02092594, -60.9843842, -60.94785462],
            [-61.04219088, -61.00562502, -60.96907132, -60.93252978],
            [-61.02688979, -60.99031194, -60.95374627, -60.91719277],
        ]
    )

    assert lat_array.shape == lon_array.shape == (3, 4)

    return lat_array, lon_array


def make_example_sparse_lat_lon_array():
    # Pulled from HRRR pygrib file with [::200]
    lat_array = np.array(
        [
            [
                21.138123,
                22.39363054,
                23.35211458,
                23.9989997,
                24.32429928,
                24.3229457,
                23.99496013,
                23.34545175,
                22.38444668,
            ],
            [
                26.15567125,
                27.51704653,
                28.55681548,
                29.25878727,
                29.6118577,
                29.61038848,
                29.25440314,
                28.54958622,
                27.50708576,
            ],
            [
                31.23524502,
                32.7064578,
                33.83064323,
                34.58986908,
                34.97181768,
                34.97022817,
                34.58512669,
                33.82282544,
                32.69569056,
            ],
            [
                36.33549131,
                37.91987254,
                39.13117389,
                39.94956089,
                40.36137462,
                40.35966068,
                39.94444814,
                39.12274831,
                37.90827365,
            ],
            [
                41.41038658,
                43.11071088,
                44.41149614,
                45.29078069,
                45.73337849,
                45.73153623,
                45.28528634,
                44.40244548,
                43.09825876,
            ],
            [
                46.41017285,
                48.22884146,
                49.6213434,
                50.56325709,
                51.03758491,
                51.03561028,
                50.55736972,
                49.61165081,
                48.21551648,
            ],
        ]
    )
    lon_array = np.array(
        [
            [
                -122.719528,
                -117.3045922,
                -111.74689876,
                -106.08259595,
                -100.35241767,
                -94.60006322,
                -88.87025317,
                -83.20666997,
                -77.65001567,
            ],
            [
                -124.31052716,
                -118.58097871,
                -112.68038876,
                -106.65130907,
                -100.54250721,
                -94.40681151,
                -88.29845598,
                -82.27024578,
                -76.37090505,
            ],
            [
                -126.10886706,
                -120.0297058,
                -113.74342493,
                -107.30043563,
                -100.7597301,
                -94.18597618,
                -87.64581876,
                -81.20389314,
                -74.91913112,
            ],
            [
                -128.15621863,
                -121.68731687,
                -114.96463159,
                -108.04824745,
                -101.01033929,
                -93.93120087,
                -86.89397596,
                -79.97891151,
                -73.25809648,
            ],
            [
                -130.50562862,
                -123.60114758,
                -116.38160307,
                -108.91897449,
                -101.30266779,
                -93.63401497,
                -86.01857485,
                -78.55761016,
                -71.34040163,
            ],
            [
                -133.22540236,
                -125.83346482,
                -118.04464678,
                -109.9454323,
                -101.64807138,
                -93.28287546,
                -84.98663607,
                -76.88955879,
                -69.10370576,
            ],
        ]
    )

    assert lat_array.shape == lon_array.shape == (6, 9)

    return lat_array, lon_array


def test_hrrr_dataset_spatial_slicing_with_large_extent():
    nw_arrays = _make_example_lat_lon_array_from_northwest_corner()
    se_arrays = make_example_lat_lon_array_from_southeast_corner()

    for lat_array, lon_array in [nw_arrays, se_arrays]:
        cfg = _make_config("HRRR", extent=_make_extent_from_dict(_make_large_extent_dict()))
        ds = HRRRDataset(cfg)
        curr_slice = ds._get_spatial_slice(lat_array, lon_array)

        assert len(curr_slice) == 2
        assert isinstance(curr_slice[0], slice) and isinstance(curr_slice[1], slice)

        # Because the extent is larger than CONUS, we expect the full range of the lat/lon arrays
        assert curr_slice[0] == slice(0, lat_array.shape[0])
        assert curr_slice[1] == slice(0, lon_array.shape[1])

        # If we slice again, we should get the same result (idempotent)
        curr_slice_2 = ds._get_spatial_slice(lat_array, lon_array)
        assert curr_slice == curr_slice_2


def test_hrrr_dataset_spatial_slicing_with_small_inner_extent():
    lat_array, lon_array = make_example_sparse_lat_lon_array()

    cfg = _make_config("HRRR", extent=_make_extent_from_dict(_make_small_inner_extent_dict()))
    ds = HRRRDataset(cfg)
    curr_slice = ds._get_spatial_slice(lat_array, lon_array)
    print(curr_slice)

    assert len(curr_slice) == 2
    assert isinstance(curr_slice[0], slice) and isinstance(curr_slice[1], slice)
    assert curr_slice[0] == slice(2, 4, None)
    assert curr_slice[1] == slice(5, 8, None)


def test_hrrr_dataset_spatial_slicing_no_extent():
    cfg = _make_config("HRRR")
    assert "extent" not in cfg["source"]["HRRR"]
    ds = HRRRDataset(cfg)
    lat_array, lon_array = _make_example_lat_lon_array_from_northwest_corner()

    curr_slice = ds._get_spatial_slice(lat_array, lon_array)
    assert len(curr_slice) == 2
    assert isinstance(curr_slice[0], slice) and isinstance(curr_slice[1], slice)
    assert curr_slice[0] == slice(None) and curr_slice[1] == slice(None)


# ---------------------------------------------------------------------------
# Getting Dataset Items
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Remote integration tests (skipped unless HRRR_TEST_REMOTE=1)
# ---------------------------------------------------------------------------

SKIP_REMOTE = not os.getenv("HRRR_TEST_REMOTE")


@pytest.mark.skipif(SKIP_REMOTE, reason="Set HRRR_TEST_REMOTE=1 to run remote tests")
def test_hrrr_remote_wrfprsf_getitem():
    cfg = _make_config("HRRR", levels=[500, 700], variables={"prognostic": {"vars_3D": ["T"], "vars_2D": ["t2m"]}})
    ds = HRRRDataset(cfg)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]
    assert "hrrr/prognostic/3d/T" in sample["input"]
    assert sample["input"]["hrrr/prognostic/3d/T"].shape == (2, 1, *sample["input"]["hrrr/prognostic/3d/T"].shape[2:])


@pytest.mark.skipif(SKIP_REMOTE, reason="Set HRRR_TEST_REMOTE=1 to run remote tests")
def test_hrrr_remote_wrfnatf_getitem():
    cfg = _make_config("HRRR_NAT", levels=[10, 20], variables={"prognostic": {"vars_3D": ["T"], "vars_2D": []}})
    ds = HRRRDataset(cfg)
    t = ds.datetimes[0]
    sample = ds[(t, 0)]
    assert "hrrr_nat/prognostic/3d/T" in sample["input"]
