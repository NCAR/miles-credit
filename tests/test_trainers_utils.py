"""Tests for credit.trainers.utils.load_dataset validation_data config handling."""

from unittest.mock import patch

from credit.trainers.utils import load_dataset

_BASE_DATA_CONF = {
    "source": "WB2",
    "years": [2020],
    "variables": ["temperature"],
}


def _make_conf(validation_data=None):
    conf = {"data": dict(_BASE_DATA_CONF)}
    if validation_data is not None:
        conf["validation_data"] = validation_data
    return conf


# Case: validation_data present but incomplete — should fail loudly
def test_load_dataset_validation_missing_keys_raises():
    """Incomplete validation_data (has source but missing other keys) raises ValueError."""
    conf = _make_conf(validation_data={"source": "WB2"})
    try:
        load_dataset(conf, is_train=False)
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        assert "validation_data is missing keys" in str(exc)


# Case: validation_data present but no source — source inherited from training
def test_load_dataset_validation_inherits_source():
    """validation_data without 'source' inherits it from conf['data']."""
    conf = _make_conf(validation_data={"years": [2021], "variables": ["temperature"]})
    with patch("credit.trainers.utils.MultiSourceDataset") as mock_ds:
        load_dataset(conf, is_train=False)
        called_conf = mock_ds.call_args[0][0]
        assert called_conf["source"] == _BASE_DATA_CONF["source"]
        assert called_conf["years"] == [2021]  # validation value, not training


# Case: no validation_data — falls back to training config entirely
def test_load_dataset_no_validation_data_uses_training_conf():
    """Absent validation_data passes conf['data'] to MultiSourceDataset unchanged."""
    conf = _make_conf()
    with patch("credit.trainers.utils.MultiSourceDataset") as mock_ds:
        load_dataset(conf, is_train=False)
        called_conf = mock_ds.call_args[0][0]
        assert called_conf == conf["data"]


# Case: empty validation_data {} — treated same as absent, falls back to training config
def test_load_dataset_empty_validation_data_uses_training_conf():
    """Empty validation_data dict passes conf['data'] to MultiSourceDataset unchanged."""
    conf = _make_conf(validation_data={})
    with patch("credit.trainers.utils.MultiSourceDataset") as mock_ds:
        load_dataset(conf, is_train=False)
        called_conf = mock_ds.call_args[0][0]
        assert called_conf == conf["data"]


# Case: full validation_data — used as-is, training source must not bleed in
def test_load_dataset_validation_full_uses_validation_source():
    """Full validation_data uses its own source, not training's."""
    conf = _make_conf(validation_data={"source": "WB2_VAL", "years": [2021], "variables": ["temperature"]})
    with patch("credit.trainers.utils.MultiSourceDataset") as mock_ds:
        load_dataset(conf, is_train=False)
        called_conf = mock_ds.call_args[0][0]
        assert called_conf["source"] == "WB2_VAL"  # not overwritten by training source
