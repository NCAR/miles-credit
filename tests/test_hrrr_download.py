"""
tests/test_hrrr_download.py
------------------
Unit tests for credit/datasets/hrrr_download.py.

Remote/dataset integration tests are skipped unless the environment variable
``HRRR_TEST_REMOTE=1`` is set (they hit real AWS endpoints).
"""

import os
import pathlib
import time

import pytest

from credit.datasets.hrrr_download import download_hrrr, get_specific_product_config

# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


def _make_multisource_config():
    return {
        "source": {
            "HRRR": {
                "mode": "local",
                "base_path": "/path/to/hrrr",
                "forecast_hour": 0,
                "levels": [500, 700, 850],
                "variables": {
                    "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
                },
            },
            "HRRR_NAT": {
                "mode": "local",
                "base_path": "/path/to/hrrr_nat",
                "forecast_hour": 0,
                "levels": [500, 700, 850],
                "variables": {
                    "dynamic_forcing": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
                },
            },
            "HRRR_SUBH": {
                "mode": "local",
                "base_path": "/path/to/hrrr_subhf",
                "forecast_hour": 0,
                "levels": [500, 700, 850],
                "variables": {
                    "prognostic": {"vars_2D": ["t2m"]},
                },
            },
        },
        "start_datetime": "2022-01-01 01:00",
        "end_datetime": "2022-01-01 02:00",
        "timestep": "1h",
        "forecast_len": 0,
    }


def test_get_specific_product_config():
    config = _make_multisource_config()

    # Main HRRR
    hrrr_paired_opts = [("HRRR", "wrfprsf"), ("HRRR_NAT", "wrfnatf"), ("HRRR_SUBH", "wrfsubhf")]
    for product_key_tuple in hrrr_paired_opts:
        main_key = product_key_tuple[0]
        for product in product_key_tuple:
            subconfig = get_specific_product_config(config, product)
            assert len(subconfig["source"]) == 1
            assert main_key in subconfig["source"]
            assert subconfig["source"][main_key] == config["source"][main_key]


# ---------------------------------------------------------------------------
# Remote integration tests (skipped unless HRRR_TEST_REMOTE=1)
# ---------------------------------------------------------------------------


SKIP_REMOTE = not os.getenv("HRRR_TEST_REMOTE")
REASON_SKIP_REMOTE = "Set HRRR_TEST_REMOTE=1 to run remote tests"


@pytest.fixture(scope="session")
def _make_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("hrrr_test")


@pytest.fixture(scope="session")
def _make_config_download_factory(_make_base_path):
    """
    Set up a factory since we will want to modify the config for different tests,
    but we want to share the same base path across tests.
    """

    def _make_config_download(source_key="HRRR", **extra_source):
        source_defaults = {
            "mode": "local",
            "base_path": _make_base_path,
            "forecast_hour": 0,
            "levels": [500, 700, 850],
            "variables": {
                "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
            },
        }
        source_defaults.update(extra_source)
        resulting_config = {
            "source": {source_key: source_defaults},
            "start_datetime": "2022-01-01 01:00",
            "end_datetime": "2022-01-01 02:00",
            "timestep": "1h",
            "forecast_len": 0,
        }
        return resulting_config

    return _make_config_download


@pytest.mark.skipif(SKIP_REMOTE, reason=REASON_SKIP_REMOTE)
def test_download_hrrr(_make_config_download_factory):
    time_start_make_config = time.time()
    config = _make_config_download_factory(source_key="HRRR")
    time_end_make_config = time.time()
    # Will print if test fails
    print(f"\nMake config took {time_end_make_config - time_start_make_config:.2f} seconds")

    time_start_download = time.time()
    download_hrrr(config=config, overwrite=True, num_workers=1)
    time_end_download = time.time()
    # Will print if test fails
    print(f"Download took {time_end_download - time_start_download:.2f} seconds")

    # Check the temporary directory for the expected files.
    tmp_dir = pathlib.Path(config["source"]["HRRR"]["base_path"])
    assert tmp_dir.exists()
    assert tmp_dir.is_dir()

    # We expect a structure like:
    # tmp_dir/
    #   hrrr.20220101/
    #     conus/
    #       hrrr.t01z.wrfprsf00.grib2
    #       hrrr.t01z.wrfprsf00.grib2.idx
    #       hrrr.t02z.wrfprsf00.grib2
    #       hrrr.t02z.wrfprsf00.grib2.idx

    print("Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"Contents of {tmp_dir}:")

    for path in sorted(tmp_dir.rglob("*")):
        # Calculate depth by checking distance from the root directory
        depth = len(path.relative_to(tmp_dir).parts)
        spacer = "  " * depth
        file_size = path.stat().st_size
        print(f"{spacer}+ ({file_size} bytes) {path.name} ")

    file1_path = tmp_dir / "hrrr.20220101" / "conus" / "hrrr.t01z.wrfprsf00.grib2"
    file1_idx_path = file1_path.with_suffix(file1_path.suffix + ".idx")
    file2_path = tmp_dir / "hrrr.20220101" / "conus" / "hrrr.t02z.wrfprsf00.grib2"
    file2_idx_path = file2_path.with_suffix(file2_path.suffix + ".idx")

    assert file1_path.exists(), f"Expected file not found: {file1_path}"
    assert file1_idx_path.exists(), f"Expected file not found: {file1_idx_path}"
    assert file2_path.exists(), f"Expected file not found: {file2_path}"
    assert file2_idx_path.exists(), f"Expected file not found: {file2_idx_path}"


@pytest.mark.skipif(SKIP_REMOTE, reason=REASON_SKIP_REMOTE)
def test_download_hrrr_subhf(_make_config_download_factory):
    time_start_make_config = time.time()
    config = _make_config_download_factory(source_key="HRRR_SUBH")
    time_end_make_config = time.time()
    # Will print if test fails
    print(f"\nMake config took {time_end_make_config - time_start_make_config:.2f} seconds")

    time_start_download = time.time()
    download_hrrr(config=config, overwrite=True, num_workers=2)
    time_end_download = time.time()
    # Will print if test fails
    print(f"Download took {time_end_download - time_start_download:.2f} seconds")

    # Check the temporary directory for the expected files.
    tmp_dir = pathlib.Path(config["source"]["HRRR_SUBH"]["base_path"])
    assert tmp_dir.exists()
    assert tmp_dir.is_dir()

    # We expect a structure like:
    # tmp_dir/
    #   hrrr.20220101/
    #     conus/
    #       hrrr.t00z.wrfsubhf01.grib2
    #       hrrr.t00z.wrfsubhf01.grib2.idx
    #       hrrr.t01z.wrfsubhf01.grib2
    #       hrrr.t01z.wrfsubhf01.grib2.idx

    print("Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"Contents of {tmp_dir}:")

    for path in sorted(tmp_dir.rglob("*")):
        # Calculate depth by checking distance from the root directory
        depth = len(path.relative_to(tmp_dir).parts)
        spacer = "  " * depth
        file_size = path.stat().st_size
        print(f"{spacer}+ ({file_size} bytes) {path.name} ")

    file1_path = tmp_dir / "hrrr.20220101" / "conus" / "hrrr.t00z.wrfsubhf01.grib2"
    file1_idx_path = file1_path.with_suffix(file1_path.suffix + ".idx")
    file2_path = tmp_dir / "hrrr.20220101" / "conus" / "hrrr.t01z.wrfsubhf01.grib2"
    file2_idx_path = file2_path.with_suffix(file2_path.suffix + ".idx")

    assert file1_path.exists(), f"Expected file not found: {file1_path}"
    assert file1_idx_path.exists(), f"Expected file not found: {file1_idx_path}"
    assert file2_path.exists(), f"Expected file not found: {file2_path}"
    assert file2_idx_path.exists(), f"Expected file not found: {file2_idx_path}"
