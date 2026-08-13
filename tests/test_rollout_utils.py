import pandas as pd
from credit.trainers.rollout_utils import apply_inference_overrides, with_inference_datetime_bounds


def test_apply_inference_overrides_replaces_sections_independently():
    conf = {
        "data": {"source": {"training": {}}},
        "preblocks": {"ic_only": {"training": {"type": "noop"}}},
        "postblocks": {"per_step": {"training": {"type": "noop"}}},
        "inference": {
            "data": {"source": {"inference": {}}},
            "postblocks": {"per_step": {"inference": {"type": "noop"}}},
        },
    }
    original_data = conf["data"]
    original_preblocks = conf["preblocks"]

    schema_conf = apply_inference_overrides(conf)

    assert schema_conf["data"] is original_data
    assert conf["data"] == {"source": {"inference": {}}}
    assert conf["preblocks"] is original_preblocks
    assert conf["postblocks"]["per_step"]["inference"]["type"] == "noop"


def test_with_inference_datetime_bounds_derives_missing_bounds():
    data_conf = {"timestep": "6h", "source": {"era5": {}}}
    init_times = [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01")]

    result = with_inference_datetime_bounds(data_conf, init_times, n_steps=4, timestep="6h")

    assert result["start_datetime"] == pd.Timestamp("2020-01-01")
    assert result["end_datetime"] == pd.Timestamp("2020-01-03")
    assert result["source"] == data_conf["source"]


def test_with_inference_datetime_bounds_preserves_explicit_bounds():
    data_conf = {
        "start_datetime": "2019-12-01",
        "end_datetime": "2019-12-02",
        "source": {"era5": {}},
    }

    result = with_inference_datetime_bounds(
        data_conf,
        [pd.Timestamp("2020-01-01")],
        n_steps=4,
        timestep="6h",
    )

    assert result["start_datetime"] == "2019-12-01"
    assert result["end_datetime"] == "2019-12-02"
