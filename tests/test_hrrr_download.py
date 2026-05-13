"""
tests/test_hrrr_download.py
------------------
Unit tests for credit/datasets/hrrr_download.py.

Remote/dataset integration tests are run unless the environment variable
``SKIP_REMOTE=1`` is set (they hit real AWS endpoints).
"""

from typing import Any

import os
import pathlib
import time

import pytest

from credit.datasets.hrrr_download import download_hrrr
from credit.datasets.hrrr import VALID_PRODUCTS
from credit.datasets.multi_source import make_single_source_subconfig

# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


def _make_multisource_config() -> dict[str, Any]:
    return {
        "source": {
            "Test_HRRR": {
                "dataset_type": "hrrr",
                "product": "wrfprsf",
                "mode": "local",
                "base_path": "/path/to/hrrr",
                "forecast_hour": 0,
                "levels": [500, 700, 850],
                "variables": {
                    "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
                },
            },
            "Test_HRRR_NAT": {
                "dataset_type": "hrrr",
                "product": "wrfnatf",
                "mode": "local",
                "base_path": "/path/to/hrrr_nat",
                "forecast_hour": 0,
                "levels": [500, 700, 850],
                "variables": {
                    "dynamic_forcing": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
                },
            },
            "Test_HRRR_SUBH": {
                "dataset_type": "hrrr",
                "product": "wrfsubhf",
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
    data_config = _make_multisource_config()

    dataset_types_check = "hrrr"
    product_check = ["wrfprsf", "wrfnatf", "wrfsubhf"]

    for source, product in zip(data_config["source"].keys(), product_check):
        subconfig = make_single_source_subconfig(data_config, source)
        assert len(subconfig["source"]) == 1
        assert source in subconfig["source"]
        assert subconfig["source"][source] == data_config["source"][source]

        assert subconfig["source"][source]["dataset_type"] == dataset_types_check
        assert subconfig["source"][source]["product"] == product


# ---------------------------------------------------------------------------
# Remote integration tests (skipped unless HRRR_TEST_REMOTE=1)
# ---------------------------------------------------------------------------

SKIP_REMOTE = bool(os.getenv("SKIP_HRRR_REMOTE"))
REASON_SKIP_REMOTE = "Set SKIP_HRRR_REMOTE=1 to skip remote tests"


@pytest.fixture(scope="session")
def _make_base_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return tmp_path_factory.mktemp("hrrr_test")


@pytest.fixture(scope="session")
def _make_config_download_factory(_make_base_path: pathlib.Path) -> Any:
    """
    Set up a factory since we will want to modify the config for different tests,
    but we want to share the same base path across tests.
    """

    def _make_config_download(product: VALID_PRODUCTS = "wrfprsf", **extra_source) -> dict[str, Any]:
        source_defaults: dict[str, Any] = {
            "dataset_type": "hrrr",
            "product": product,
            "mode": "local",
            "base_path": _make_base_path,
            "forecast_hour": 0,
            "levels": [500, 700, 850],
            "variables": {
                "prognostic": {"vars_3D": ["T", "U"], "vars_2D": ["t2m"]},
            },
        }
        source_defaults.update(extra_source)
        resulting_config: dict[str, Any] = {
            "source": {f"Test_HRRR_{product}": source_defaults},
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
    data_config = _make_config_download_factory(product="wrfprsf")
    time_end_make_config = time.time()
    # Will print if test fails
    print(f"\nMake config took {time_end_make_config - time_start_make_config:.2f} seconds")

    time_start_download = time.time()
    download_hrrr(data_config=data_config, overwrite=True, num_workers=1)
    time_end_download = time.time()
    # Will print if test fails
    print(f"Download took {time_end_download - time_start_download:.2f} seconds")

    # Check the temporary directory for the expected files.
    tmp_dir = pathlib.Path(data_config["source"]["Test_HRRR_wrfprsf"]["base_path"])
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

    print("Data config:")
    for key, value in data_config.items():
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
    data_config = _make_config_download_factory(product="wrfsubhf")
    time_end_make_config = time.time()
    # Will print if test fails
    print(f"\nMake config took {time_end_make_config - time_start_make_config:.2f} seconds")

    time_start_download = time.time()
    download_hrrr(data_config=data_config, overwrite=True, num_workers=2)
    time_end_download = time.time()
    # Will print if test fails
    print(f"Download took {time_end_download - time_start_download:.2f} seconds")

    # Check the temporary directory for the expected files.
    tmp_dir = pathlib.Path(data_config["source"]["Test_HRRR_wrfsubhf"]["base_path"])
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

    print("Data config:")
    for key, value in data_config.items():
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
