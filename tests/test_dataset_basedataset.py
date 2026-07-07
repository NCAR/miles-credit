"""
tests/test_dataset_basedataset.py
-------------------------------
Unit tests for the BaseDataset class in credit.datasets.base_dataset.py.

"""

import pytest
import pandas as pd
from typing import Any, Dict, List

from credit.datasets.base_dataset import AbstractBaseDataset, BaseDataset

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Provides a minimal, valid configuration dictionary for BaseDataset."""
    return {
        "timestep": "6h",
        "forecast_len": 1,
        "start_datetime": "2022-12-31T18:00:00Z",
        "end_datetime": "2023-01-02T00:00:00Z",
        "source": {
            "TestSource_Base": {
                "variables": {
                    "prognostic": {
                        "vars_3D": ["T", "U", "V", "Q"],
                        "vars_2D": ["t2m"],
                        "path": "/fake/prognostic/*.nc",
                    },
                    "dynamic_forcing": {
                        "vars_2D": ["msl"],
                        "path": "/fake/dyn/*.nc",
                    },
                    "static": {
                        "vars_2D": ["lsm"],
                        "path": "/fake/static.nc",
                    },
                    "diagnostic": {
                        "vars_2D": ["tp"],
                        "path": "/fake/diagnostic/*.nc",
                    },
                }
            }
        },
    }


@pytest.fixture
def multi_source_config(minimal_config: Dict[str, Any]) -> Dict[str, Any]:
    """Provides a config with multiple sources to test error handling."""
    config = minimal_config.copy()
    config["source"]["Another_Source"] = {
        "variables": {"prognostic": {"vars_2D": ["t2m"], "path": "/fake/prognostic/*.nc"}}
    }
    return config


@pytest.fixture
def patch_base_dataset_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches glob and _map_files for BaseDataset tests."""

    def fake_glob(pattern: str) -> List[str]:
        return ["/fake/file1.nc", "/fake/file2.nc"]

    monkeypatch.setattr("credit.datasets.base_dataset.glob", fake_glob)

    def fake_map_files(files: List[str], time_fmt: str) -> List[tuple[pd.Timestamp, pd.Timestamp, str]]:
        return [
            (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), "/fake/file1.nc"),
            (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), "/fake/file2.nc"),
        ]

    monkeypatch.setattr("credit.datasets.base_dataset._map_files", fake_map_files)


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_success(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test successful initialization with a valid config."""
    ds = BaseDataset(minimal_config)
    assert ds is not None
    assert ds.return_target is False
    assert len(ds.datetimes) > 0


def test_init_invalid_config_type() -> None:
    """Test that __init__ raises TypeError for non-dict config."""
    with pytest.raises(TypeError, match="Expected data_config to be a dict"):
        BaseDataset("not a dict")  # pyright: ignore[reportArgumentType]


def test_init_invalid_return_target_type(minimal_config: Dict[str, Any]) -> None:
    """Test that __init__ raises TypeError for non-bool return_target."""
    with pytest.raises(TypeError, match="Expected return_target to be a bool"):
        BaseDataset(minimal_config, return_target="not a bool")  # pyright: ignore[reportArgumentType]


def test_init_missing_source_key(minimal_config: Dict[str, Any]) -> None:
    """Test that __init__ raises KeyError if 'source' key is missing."""
    config = minimal_config.copy()
    del config["source"]
    with pytest.raises(KeyError, match="Expected 'source' key in data_config"):
        BaseDataset(config)


def test_init_empty_source_dict(minimal_config: Dict[str, Any]) -> None:
    """Test that __init__ raises ValueError if 'source' is an empty dict."""
    config = minimal_config.copy()
    config["source"] = {}
    with pytest.raises(ValueError, match="Expected 'source' key in data_config to be non-empty"):
        BaseDataset(config)


def test_init_multiple_sources_raises_error(multi_source_config: Dict[str, Any]) -> None:
    """Test that BaseDataset (by default) raises ValueError for multiple sources."""
    with pytest.raises(ValueError, match="Multiple sources found in config"):
        BaseDataset(multi_source_config)


def test_init_missing_variables_key(minimal_config: Dict[str, Any]) -> None:
    """Test that __init__ raises AssertionError if 'variables' is missing."""
    config = minimal_config.copy()
    del config["source"]["TestSource_Base"]["variables"]
    with pytest.raises(KeyError, match="Expected 'variables' key in source config"):
        BaseDataset(config)


# ---------------------------------------------------------------------------
# Clock parameter loading tests
# ---------------------------------------------------------------------------


def test_load_clock_params_from_main_config(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test that clock parameters are loaded correctly from the main config."""
    ds = BaseDataset(minimal_config)
    assert ds.dt == pd.Timedelta("6h")
    assert ds.num_forecast_steps == 1
    assert ds.start_datetime == pd.Timestamp("2022-12-31T18:00:00Z")
    assert ds.end_datetime == pd.Timestamp("2023-01-02T00:00:00Z")


def test_load_clock_params_override_from_source(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test that source-specific clock parameters override main config."""
    config = minimal_config.copy()
    config["source"]["TestSource_Base"]["timestep"] = "12h"
    config["source"]["TestSource_Base"]["forecast_len"] = 5
    config["source"]["TestSource_Base"]["start_datetime"] = "2023-01-01T00:00:00Z"
    config["source"]["TestSource_Base"]["end_datetime"] = "2023-01-01T12:00:00Z"

    ds = BaseDataset(config)
    assert ds.dt == pd.Timedelta("12h")
    assert ds.num_forecast_steps == 5
    assert ds.start_datetime == pd.Timestamp("2023-01-01T00:00:00Z")
    assert ds.end_datetime == pd.Timestamp("2023-01-01T12:00:00Z")


def test_load_clock_params_missing_raises_keyerror(minimal_config: Dict[str, Any]) -> None:
    """Test that missing required clock params raises KeyError."""
    for key in ["timestep", "forecast_len", "start_datetime", "end_datetime"]:
        config = minimal_config.copy()
        del config[key]
        with pytest.raises(KeyError, match=f"{key} must be specified"):
            BaseDataset(config)


def test_load_dt_warning(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test warning when source timestep is smaller than data timestep."""
    config = minimal_config.copy()
    original_dt = config["timestep"]
    config["source"]["TestSource_Base"]["timestep"] = "1h"  # smaller than 6h
    # Check smaller to ensure the test is valid
    assert config["source"]["TestSource_Base"]["timestep"] < original_dt
    BaseDataset(config)
    assert "is smaller than" in caplog.text


def test_load_forecast_len_warning(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test warning when source forecast_len is greater than data forecast_len."""
    config = minimal_config.copy()
    original_forecast_len = config["forecast_len"]
    config["source"]["TestSource_Base"]["forecast_len"] = 5  # greater than 1
    assert config["source"]["TestSource_Base"]["forecast_len"] > original_forecast_len
    BaseDataset(config)
    assert "is greater than" in caplog.text


def test_load_start_datetime_warning(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test warning when source start_datetime is earlier than data start_datetime."""
    config = minimal_config.copy()
    original_start = config["start_datetime"]
    config["source"]["TestSource_Base"]["start_datetime"] = "2000-01-01T00:00:00Z"  # earlier than 2022-12-31
    assert config["source"]["TestSource_Base"]["start_datetime"] < original_start
    BaseDataset(config)
    assert "is earlier than" in caplog.text


def test_load_end_datetime_warning(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test warning when source end_datetime is later than data end_datetime."""
    config = minimal_config.copy()
    original_end = config["end_datetime"]
    config["source"]["TestSource_Base"]["end_datetime"] = "2100-01-01T12:00:00Z"  # later than 2023-01-02
    assert config["source"]["TestSource_Base"]["end_datetime"] > original_end
    BaseDataset(config)
    assert "is later than" in caplog.text


# ---------------------------------------------------------------------------
# _register_field tests
# ---------------------------------------------------------------------------


def test_register_field_success(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test successful field registration."""
    config = minimal_config.copy()
    config["source"]["TestSource_Base"]["mode"] = "remote"
    ds = BaseDataset(config)
    assert "prognostic" in ds.var_dict, (
        f"Expected 'prognostic' in var_dict, got {ds.var_dict.keys()}. \nFull var_dict: \n{ds.var_dict}"
    )
    assert "prognostic" in ds.file_dict, (
        f"Expected 'prognostic' in file_dict, got {ds.file_dict.keys()}. \nFull file_dict: \n{ds.file_dict.keys()}"
    )
    assert ds.var_dict["prognostic"]["vars_2D"] == ["t2m"], (
        f"Expected vars_2D ['t2m'], got {ds.var_dict['prognostic']['vars_2D']}"
    )
    assert ds.file_dict["prognostic"] is not None, "Expected file_dict['prognostic'] to be set, got None"


def test_register_field_invalid_type(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test that an invalid field type raises KeyError."""
    config = minimal_config.copy()
    config["source"]["TestSource_Base"]["variables"]["invalid_field"] = {"vars_2D": ["x"]}
    with pytest.raises(KeyError, match="Unknown field_type 'invalid_field'"):
        BaseDataset(config)


def test_register_field_null_config(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test that a null field config is handled correctly."""
    config = minimal_config.copy()
    config["source"]["TestSource_Base"]["variables"]["prognostic"] = None
    ds = BaseDataset(config)
    assert ds.file_dict["prognostic"] is None
    assert "prognostic" not in ds.var_dict


def test_register_field_missing_vars(minimal_config: Dict[str, Any]) -> None:
    """Test that a field with no vars_2D or vars_3D raises ValueError."""
    config = minimal_config.copy()
    config["source"]["TestSource_Base"]["variables"]["prognostic"] = {"path": "/fake/*.nc"}
    with pytest.raises(ValueError, match="must define at least one of vars_3D or vars_2D"):
        BaseDataset(config)


def test_mode_setting(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test that mode is set correctly based on config."""
    config = minimal_config.copy()
    # Default should be local
    ds_local = BaseDataset(config)
    assert ds_local.mode == "local"

    # Explicitly set to remote
    config["source"]["TestSource_Base"]["mode"] = "remote"
    ds_remote = BaseDataset(config)
    assert ds_remote.mode == "remote"

    # Set to something that is not 'local' or 'remote' should raise ValueError
    config["source"]["TestSource_Base"]["mode"] = "invalid_mode"
    with pytest.raises(ValueError, match="Unknown mode 'invalid_mode'"):
        BaseDataset(config)


# ---------------------------------------------------------------------------
# __len__ and __getitem__ tests
# ---------------------------------------------------------------------------


def test_len(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test that len(dataset) is correct."""
    ds = BaseDataset(minimal_config)
    # start="2022-12-31T18:00", end="2023-01-02T00:00", freq="6h", forecast_len=1
    # Valid start times: 18:00, 00:00, 06:00, 12:00, 18:00. (5)
    # end_datetime - num_forecast_steps * dt = 2023-01-02T00:00 - 6h = 2023-01-01T18:00
    # pd.date_range('2022-12-31 18:00', '2023-01-01 18:00', freq='6h') -> 5 timestamps
    assert len(ds) == 5


def test_getitem_step0(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test __getitem__ at step i=0 loads prognostic, static, and dynamic_forcing."""
    ds = BaseDataset(minimal_config)

    with caplog.at_level("ERROR"):
        sample = ds[(ds.datetimes[0], 0)]
        assert "You are using the default _extract_field method in BaseDataset" in caplog.text

    assert "input" in sample
    inp = sample["input"]
    assert "TestSource_Base/prognostic/2d/t2m" in inp
    assert "TestSource_Base/prognostic/3d/T" in inp
    assert "TestSource_Base/prognostic/3d/U" in inp
    assert "TestSource_Base/static/2d/lsm" in inp
    assert "TestSource_Base/dynamic_forcing/2d/msl" in inp
    assert "metadata" in sample
    assert "input_datetime" in sample["metadata"]


def test_getitem_step1(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test __getitem__ at step i>0 only loads dynamic_forcing."""
    ds = BaseDataset(minimal_config)

    with caplog.at_level("ERROR"):
        sample = ds[(ds.datetimes[0], 1)]
        assert "You are using the default _extract_field method in BaseDataset" in caplog.text

    assert "input" in sample
    inp = sample["input"]
    assert "TestSource_Base/prognostic/2d/t2m" not in inp
    assert "TestSource_Base/prognostic/3d/T" not in inp
    assert "TestSource_Base/static/2d/lsm" not in inp
    assert "TestSource_Base/dynamic_forcing/2d/msl" in inp


def test_getitem_return_target_true(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test __getitem__ with return_target=True."""
    config = minimal_config.copy()

    for mode in ["local", "remote"]:
        config["source"]["TestSource_Base"]["mode"] = mode
        ds = BaseDataset(config, return_target=True)

        with caplog.at_level("ERROR"):
            sample = ds[(ds.datetimes[0], 0)]
            assert "You are using the default _extract_field method in BaseDataset" in caplog.text

        assert "target" in sample
        tgt = sample["target"]
        assert "TestSource_Base/prognostic/2d/t2m" in tgt
        assert "TestSource_Base/prognostic/3d/T" in tgt
        assert "TestSource_Base/prognostic/3d/U" in tgt
        assert "TestSource_Base/diagnostic/2d/tp" in tgt
        # Static and dynamic forcing should not be in target
        assert "TestSource_Base/static/2d/lsm" not in tgt
        assert "TestSource_Base/dynamic_forcing/2d/msl" not in tgt
        assert "target_datetime" in sample["metadata"]


def test_getitem_return_target_false(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Test __getitem__ with return_target=False."""
    config = minimal_config.copy()
    config["source"]["TestSource_Base"]["mode"] = "remote"
    ds = BaseDataset(config, return_target=False)
    sample = ds[(ds.datetimes[0], 0)]
    assert "target" not in sample
    assert "target_datetime" not in sample["metadata"]


# ---------------------------------------------------------------------------
# history_len > 1 (multi-step input window) tests
# ---------------------------------------------------------------------------
#
# The generic multi-step reader BaseDataset._extract_field_window calls the
# single-step _extract_field once per history timestamp and stacks along the
# time dimension (dim=1). Using the BaseDataset placeholder _extract_field
# (which emits (n_levels, 1, lat, lon) / (1, 1, lat, lon)) the time dim of the
# stacked result must equal history_len. The target is always single-step.


def test_history_len_defaults_to_one(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """With no history_len key, history_len defaults to 1 and input is single-step."""
    ds = BaseDataset(minimal_config)
    assert ds.history_len == 1
    inp = ds[(ds.datetimes[0], 0)]["input"]
    # Time dim (axis 1) is 1 on every input field — identical to legacy behaviour.
    assert inp["TestSource_Base/prognostic/3d/T"].shape[1] == 1
    assert inp["TestSource_Base/prognostic/2d/t2m"].shape[1] == 1
    assert inp["TestSource_Base/static/2d/lsm"].shape[1] == 1
    assert inp["TestSource_Base/dynamic_forcing/2d/msl"].shape[1] == 1


@pytest.mark.parametrize("history_len", [2, 3])
def test_history_len_input_window_stacks_in_time(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None, history_len: int
) -> None:
    """history_len > 1 stacks prognostic/static/dynamic_forcing over the window at i=0."""
    config = minimal_config.copy()
    config["history_len"] = history_len
    ds = BaseDataset(config, return_target=True)
    assert ds.history_len == history_len

    sample = ds[(ds.datetimes[0], 0)]
    inp = sample["input"]
    # Every history-carrying input field has time dim == history_len.
    assert inp["TestSource_Base/prognostic/3d/T"].shape[1] == history_len
    assert inp["TestSource_Base/prognostic/2d/t2m"].shape[1] == history_len
    assert inp["TestSource_Base/static/2d/lsm"].shape[1] == history_len
    assert inp["TestSource_Base/dynamic_forcing/2d/msl"].shape[1] == history_len
    # Non-time dims are untouched (3D keeps n_levels, 2D keeps singleton level).
    assert inp["TestSource_Base/prognostic/3d/T"].shape[0] == 7
    assert inp["TestSource_Base/prognostic/2d/t2m"].shape[0] == 1

    # The target is always a single step, regardless of history_len.
    tgt = sample["target"]
    assert tgt["TestSource_Base/prognostic/3d/T"].shape[1] == 1
    assert tgt["TestSource_Base/diagnostic/2d/tp"].shape[1] == 1


def test_history_len_rollout_step_is_single_step(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """At rollout steps (i > 0) only the newest dynamic_forcing step is loaded, single-step."""
    config = minimal_config.copy()
    config["history_len"] = 3
    ds = BaseDataset(config)

    inp = ds[(ds.datetimes[0], 1)]["input"]
    # Prognostic/static are not loaded at i > 0; dynamic_forcing is single-step.
    assert "TestSource_Base/prognostic/3d/T" not in inp
    assert "TestSource_Base/static/2d/lsm" not in inp
    assert inp["TestSource_Base/dynamic_forcing/2d/msl"].shape[1] == 1


def test_history_len_shifts_timestamps_forward(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """The sampling clock starts (history_len-1)*dt later so every sample has full history."""
    config = minimal_config.copy()
    config["history_len"] = 2
    ds = BaseDataset(config)
    ds1 = BaseDataset(minimal_config)  # history_len == 1
    # With H=2 the first valid timestamp is one dt later than with H=1.
    assert ds.datetimes[0] == ds1.datetimes[0] + ds.dt


@pytest.mark.parametrize("bad_value", [0, -1, 2.0, "2"])
def test_history_len_invalid_raises(minimal_config: Dict[str, Any], bad_value: Any) -> None:
    """history_len must be a positive int."""
    config = minimal_config.copy()
    config["history_len"] = bad_value
    with pytest.raises(ValueError, match="history_len must be a positive int"):
        BaseDataset(config)


def test_extract_field_window_matches_repeated_single_step(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None
) -> None:
    """_extract_field_window equals stacking independent single-step reads along time."""
    import torch

    ds = BaseDataset(minimal_config)
    t_history = pd.date_range(ds.datetimes[0] - ds.dt, ds.datetimes[0], freq=ds.dt)  # length 2

    windowed: Dict[str, Any] = {}
    ds._extract_field_window("prognostic", t_history, windowed)  # pyright: ignore[reportPrivateUsage]

    # Build the expected result by hand: single-step read per timestamp, cat in time.
    per_step = []
    for tk in t_history:
        s: Dict[str, Any] = {}
        ds._extract_field("prognostic", pd.Timestamp(tk), s)  # pyright: ignore[reportPrivateUsage]
        per_step.append(s)
    for key in per_step[0]:
        expected = torch.cat([step[key] for step in per_step], dim=1)
        assert torch.equal(windowed[key], expected)


# ---------------------------------------------------------------------------
# temporal_mode: persist tests
# ---------------------------------------------------------------------------


def test_temporal_mode_default_is_none(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """temporal_mode is None by default (no key in source config)."""
    ds = BaseDataset(minimal_config)
    assert ds.temporal_mode == "exact"


def test_temporal_mode_set_from_config(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """temporal_mode is read from the source config block."""
    minimal_config["source"]["TestSource_Base"]["temporal_mode"] = "persist"
    ds = BaseDataset(minimal_config)
    assert ds.temporal_mode == "persist"


def test_persist_snaps_to_last_native_timestamp(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """__getitem__ with temporal_mode=persist snaps fine-resolution t to last native timestamp."""
    minimal_config["source"]["TestSource_Base"]["temporal_mode"] = "persist"
    ds = BaseDataset(minimal_config)
    t_native = ds.datetimes[0]

    # A timestamp 1 hour after the first native tick (within the same 6h native interval)
    t_fine = t_native + pd.Timedelta("1h")
    sample_native = ds[(t_native, 0)]
    sample_fine = ds[(t_fine, 0)]

    # Both should resolve to the same native timestamp
    assert sample_fine["metadata"]["input_datetime"] == sample_native["metadata"]["input_datetime"]


def test_persist_before_range_raises(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """_resolve_persist_timestamp raises ValueError when t < first native timestamp."""
    minimal_config["source"]["TestSource_Base"]["temporal_mode"] = "persist"
    ds = BaseDataset(minimal_config)
    t_before = ds.datetimes[0] - pd.Timedelta("1h")

    with pytest.raises(ValueError, match="before the first available native timestamp"):
        ds._resolve_persist_timestamp(t_before)  # pyright: ignore[reportPrivateUsage]


def test_persist_cache_hit_returns_same_object(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Two calls with t values that resolve to the same native timestamp return the same cached dict."""
    minimal_config["source"]["TestSource_Base"]["temporal_mode"] = "persist"
    ds = BaseDataset(minimal_config)
    t_native = ds.datetimes[0]
    t_fine = t_native + pd.Timedelta("1h")

    sample_a = ds[(t_native, 0)]
    sample_b = ds[(t_fine, 0)]

    assert sample_a is sample_b


def test_persist_new_interval_evicts_cache(minimal_config: Dict[str, Any], patch_base_dataset_io: None) -> None:
    """Moving to a new native interval evicts the previous cached entries."""
    minimal_config["source"]["TestSource_Base"]["temporal_mode"] = "persist"
    ds = BaseDataset(minimal_config)
    t0 = ds.datetimes[0]
    t1 = ds.datetimes[1]  # next native tick

    sample_t0 = ds[(t0, 0)]
    assert len(ds._persist_cache) == 1  # pyright: ignore[reportPrivateUsage]

    # Advance to next native interval — cache should evict t0 entry
    ds[(t1, 0)]
    cached_keys = list(ds._persist_cache.keys())  # pyright: ignore[reportPrivateUsage]
    assert all(k[0] == t1 for k in cached_keys)
    assert sample_t0 is not ds._persist_cache.get((t0, True))  # pyright: ignore[reportPrivateUsage]


def test_persist_i0_and_i1_are_separate_cache_entries(
    minimal_config: Dict[str, Any], patch_base_dataset_io: None
) -> None:
    """i==0 and i>0 are cached under separate keys for the same native timestamp."""
    minimal_config["source"]["TestSource_Base"]["temporal_mode"] = "persist"
    ds = BaseDataset(minimal_config)
    t = ds.datetimes[0]

    sample_i0 = ds[(t, 0)]
    sample_i1 = ds[(t, 1)]

    assert sample_i0 is not sample_i1
    assert len(ds._persist_cache) == 2  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# AbstractBaseDataset method tests
# ---------------------------------------------------------------------------


def test_abstract_base_dataset_methods_raise_error() -> None:
    """Test that methods of AbstractBaseDataset raise NotImplementedError."""

    dataset = AbstractBaseDataset(data_config={})

    with pytest.raises(NotImplementedError):
        len(dataset)
    with pytest.raises(NotImplementedError):
        dataset[(pd.Timestamp("2023-01-01"), 0)]
    with pytest.raises(NotImplementedError):
        dataset._build_timestamps()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(NotImplementedError):
        dataset._get_field_name("diagnostic", "2d", "t2m")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(NotImplementedError):
        dataset.init_register_all_fields()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(NotImplementedError):
        dataset._register_field("prognostic", {})  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(NotImplementedError):
        dataset._get_file_source({})  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(NotImplementedError):
        dataset._extract_field("prognostic", pd.Timestamp("2023-01-01"), {})  # pyright: ignore[reportPrivateUsage]
