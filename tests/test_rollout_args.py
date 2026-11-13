"""Tests for credit.rollout_args: --init-time parsing and rollout CLI overrides (CPU only)."""

import argparse

import pytest

from credit.rollout_args import apply_cli_overrides, check_rollout_flags, init_time_arg


def _ns(**kw):
    base = dict(run_mode=None, init_time=None, save_dir=None, mode=None, noise_scale=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2020-06-01T00", "2020-06-01T00:00:00"),
        ("2020-06-01 06:00", "2020-06-01T06:00:00"),
        ("2020-06-01T00Z", "2020-06-01T00:00:00"),
        ("2020-06-01T12+00:00", "2020-06-01T12:00:00"),
    ],
)
def test_init_time_arg_normalizes_to_naive_utc(value, expected):
    assert init_time_arg(value) == expected


@pytest.mark.parametrize("value", ["2020060100", "tomorrow-ish", "", "2020-06-01T00-07:00"])
def test_init_time_arg_rejects(value):
    with pytest.raises(argparse.ArgumentTypeError):
        init_time_arg(value)


def test_check_rollout_flags():
    check_rollout_flags(None, "2020-06-01T00:00:00")
    check_rollout_flags("single", "2020-06-01T00:00:00")
    check_rollout_flags("batch", None)
    with pytest.raises(ValueError, match="conflicts"):
        check_rollout_flags("batch", "2020-06-01T00:00:00")


def test_no_flags_leaves_config_untouched():
    conf = {"run_mode": "batch", "mode": "ddp", "save_forecast": "/a"}
    assert apply_cli_overrides(dict(conf), _ns()) == conf


def test_init_time_implies_single():
    conf = apply_cli_overrides({"run_mode": "batch"}, _ns(init_time="2020-06-01T00:00:00"))
    assert conf["run_mode"] == "single"
    assert conf["single_forecast"]["start_datetime"] == "2020-06-01T00:00:00"


def test_init_time_keeps_other_single_forecast_keys():
    conf = {"single_forecast": {"start_datetime": "2019-01-01", "forecast_length": "5d"}}
    apply_cli_overrides(conf, _ns(init_time="2020-06-01T00:00:00"))
    assert conf["single_forecast"] == {"start_datetime": "2020-06-01T00:00:00", "forecast_length": "5d"}


def test_mode_save_dir_noise_scale_override():
    conf = apply_cli_overrides(
        {"mode": "ddp", "save_forecast": "/a"}, _ns(mode="none", save_dir="/b", noise_scale=0.0, run_mode="batch")
    )
    assert conf == {"mode": "none", "save_forecast": "/b", "noise_scale": 0.0, "run_mode": "batch"}


def test_conflicting_flags_raise():
    with pytest.raises(ValueError):
        apply_cli_overrides({}, _ns(run_mode="batch", init_time="2020-06-01T00:00:00"))
