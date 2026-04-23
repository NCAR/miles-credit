"""
tests/test_hrrr_download.py
------------------
Unit tests for credit/datasets/hrrr_download.py.

Remote/dataset integration tests are skipped unless the environment variable
``HRRR_TEST_REMOTE=1`` is set (they hit real AWS endpoints).
"""

import os

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
        "end_datetime": "2022-01-01 03:00",
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
            "end_datetime": "2022-01-01 03:00",
            "timestep": "1h",
            "forecast_len": 0,
        }
        return resulting_config

    return _make_config_download


@pytest.mark.skipif(SKIP_REMOTE, reason=REASON_SKIP_REMOTE)
def test_download_hrrr(_make_config_download_factory):
    config = _make_config_download_factory(source_key="HRRR")

    download_hrrr(config=config, overwrite=True, num_workers=2)


@pytest.mark.skipif(SKIP_REMOTE, reason=REASON_SKIP_REMOTE)
def test_download_hrrr_subhf(_make_config_download_factory):
    config = _make_config_download_factory(source_key="HRRR_SUBH")

    download_hrrr(config=config, overwrite=True, num_workers=1)
