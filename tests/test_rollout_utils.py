"""Tests for credit.trainers.rollout_utils config helpers."""

import pandas as pd

from credit.trainers.rollout_utils import apply_inference_overrides, with_inference_datetime_bounds


def _conf(**inference_overrides):
    return {
        "data": {"source": {"ERA5": {"dataset_type": "local"}}, "timestep": "6h"},
        "preblocks": {"per_step": {"concat": {"type": "concat"}}},
        "postblocks": {"per_step": {"reconstruct": {"type": "reconstruct"}}},
        "inference": {"run_mode": "batch", **inference_overrides},
    }


class TestApplyInferenceOverrides:
    def test_no_inference_overrides_leaves_top_level_untouched(self):
        conf = _conf()
        original_data, original_pre, original_post = conf["data"], conf["preblocks"], conf["postblocks"]
        apply_inference_overrides(conf)
        assert conf["data"] is original_data
        assert conf["preblocks"] is original_pre
        assert conf["postblocks"] is original_post

    def test_inference_data_replaces_top_level_data(self):
        override_data = {"source": {"HRRR": {"dataset_type": "local"}}, "timestep": "1h"}
        conf = _conf(data=override_data)
        apply_inference_overrides(conf)
        assert conf["data"] == override_data
        assert conf["data"] is conf["inference"]["data"]

    def test_inference_preblocks_replaces_top_level_preblocks(self):
        override_pre = {"per_step": {"scaler": {"type": "bridgescaler_transform"}}}
        conf = _conf(preblocks=override_pre)
        apply_inference_overrides(conf)
        assert conf["preblocks"] == override_pre

    def test_inference_postblocks_replaces_top_level_postblocks(self):
        override_post = {"post_rollout": {"laplace_filter": {"type": "laplace_filter"}}}
        conf = _conf(postblocks=override_post)
        apply_inference_overrides(conf)
        assert conf["postblocks"] == override_post

    def test_keys_are_independent(self):
        """Only inference.data is set -- preblocks/postblocks fall back to top-level."""
        override_data = {"source": {"HRRR": {"dataset_type": "local"}}, "timestep": "1h"}
        conf = _conf(data=override_data)
        original_pre, original_post = conf["preblocks"], conf["postblocks"]
        apply_inference_overrides(conf)
        assert conf["data"] == override_data
        assert conf["preblocks"] is original_pre
        assert conf["postblocks"] is original_post

    def test_all_three_overridden_independently(self):
        override_data = {"source": {"HRRR": {}}, "timestep": "1h"}
        override_pre = {"per_step": {"a": {"type": "concat"}}}
        override_post = {"per_step": {"b": {"type": "reconstruct"}}}
        conf = _conf(data=override_data, preblocks=override_pre, postblocks=override_post)
        apply_inference_overrides(conf)
        assert conf["data"] == override_data
        assert conf["preblocks"] == override_pre
        assert conf["postblocks"] == override_post

    def test_no_inference_section_is_a_noop(self):
        conf = _conf()
        del conf["inference"]
        original_data = conf["data"]
        apply_inference_overrides(conf)
        assert conf["data"] is original_data

    def test_inference_data_null_is_treated_as_absent(self):
        """YAML `data:` with no value parses to None -- must not replace with None."""
        conf = _conf(data=None)
        original_data = conf["data"]
        apply_inference_overrides(conf)
        assert conf["data"] is original_data

    def test_data_only_override_warns(self, caplog):
        """Overriding data alone -- the risky combination -- logs a warning."""
        override_data = {"source": {"HRRR": {"dataset_type": "local"}}, "timestep": "1h"}
        conf = _conf(data=override_data)
        with caplog.at_level("WARNING"):
            apply_inference_overrides(conf)
        assert any("preblocks/postblocks were not overridden" in r.message for r in caplog.records)

    def test_data_and_preblocks_override_does_not_warn(self, caplog):
        """Overriding preblocks alongside data is the safe combination -- no warning."""
        override_data = {"source": {"HRRR": {"dataset_type": "local"}}, "timestep": "1h"}
        override_pre = {"per_step": {"rename": {"type": "rename"}}}
        conf = _conf(data=override_data, preblocks=override_pre)
        with caplog.at_level("WARNING"):
            apply_inference_overrides(conf)
        assert not any("preblocks/postblocks were not overridden" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# with_inference_datetime_bounds
# ---------------------------------------------------------------------------


class TestWithInferenceDatetimeBounds:
    def test_single_forecast_style_one_init_time(self):
        """Mirrors inference.single_forecast: one init time, forecast_length via n_steps."""
        t0 = pd.Timestamp("2020-06-01T00:00")
        result = with_inference_datetime_bounds({}, [t0], n_steps=4, timestep="6h")
        assert result["start_datetime"] == t0
        assert result["end_datetime"] == t0 + 4 * pd.Timedelta("6h")

    def test_batch_forecast_style_multiple_init_times(self):
        """Mirrors inference.batch_forecast: spans first_init_date..last_init_date."""
        times = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10")]
        result = with_inference_datetime_bounds({}, times, n_steps=40, timestep="6h")
        assert result["start_datetime"] == pd.Timestamp("2020-01-01")
        assert result["end_datetime"] == pd.Timestamp("2020-01-10") + 40 * pd.Timedelta("6h")

    def test_init_times_need_not_be_sorted(self):
        times = [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05")]
        result = with_inference_datetime_bounds({}, times, n_steps=1, timestep="6h")
        assert result["start_datetime"] == pd.Timestamp("2020-01-01")
        assert result["end_datetime"] == pd.Timestamp("2020-01-10") + pd.Timedelta("6h")

    def test_explicit_values_always_win(self):
        data_conf = {"start_datetime": "1999-01-01", "end_datetime": "1999-12-31"}
        result = with_inference_datetime_bounds(data_conf, [pd.Timestamp("2020-01-01")], n_steps=4, timestep="6h")
        assert result["start_datetime"] == "1999-01-01"
        assert result["end_datetime"] == "1999-12-31"

    def test_partial_override_only_fills_missing(self):
        data_conf = {"start_datetime": "1999-01-01"}
        result = with_inference_datetime_bounds(data_conf, [pd.Timestamp("2020-01-01")], n_steps=4, timestep="6h")
        assert result["start_datetime"] == "1999-01-01"
        assert result["end_datetime"] == pd.Timestamp("2020-01-01") + 4 * pd.Timedelta("6h")

    def test_does_not_mutate_input(self):
        data_conf = {"source": {"ERA5": {}}}
        with_inference_datetime_bounds(data_conf, [pd.Timestamp("2020-01-01")], n_steps=1, timestep="6h")
        assert "start_datetime" not in data_conf

    def test_other_keys_preserved(self):
        data_conf = {"source": {"ERA5": {}}, "timestep": "6h"}
        result = with_inference_datetime_bounds(data_conf, [pd.Timestamp("2020-01-01")], n_steps=1, timestep="6h")
        assert result["source"] == {"ERA5": {}}
        assert result["timestep"] == "6h"
