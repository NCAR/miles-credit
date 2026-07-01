import sys

import pytest
import yaml

from credit.applications import train_gen2
from credit.losses.base_losses import CRPS_LOSSES, is_crps_loss


def _write_minimal_gen2_config(tmp_path, loss_name, ensemble_size):
    config = {
        "save_loc": str(tmp_path / "run"),
        "data": {"source": {"ERA5": {}}},
        "loss": {"training_loss": loss_name},
        "trainer": {
            "type": "era5-gen2",
            "parallelism": {"data": "ddp", "tensor": 1, "domain": 1},
            "ensemble_size": ensemble_size,
        },
    }
    config_path = tmp_path / f"{loss_name}.yml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@pytest.mark.parametrize("loss_name", sorted(CRPS_LOSSES))
def test_gen2_rejects_registered_crps_loss_with_single_member(tmp_path, monkeypatch, loss_name):
    config_path = _write_minimal_gen2_config(tmp_path, loss_name, ensemble_size=1)

    monkeypatch.setattr(sys, "argv", ["train_gen2.py", "-c", str(config_path)])
    with pytest.raises(ValueError, match="requires trainer.ensemble_size > 1"):
        train_gen2.main_cli()


@pytest.mark.parametrize("loss_name", sorted(CRPS_LOSSES))
def test_registered_crps_losses_are_discovered_from_loss_registry(loss_name):
    assert is_crps_loss(loss_name)


def test_non_crps_loss_is_not_crps():
    assert not is_crps_loss("mse")


def test_gen2_ring_crps_requires_configured_ensemble_to_match_dp_world_size(tmp_path, monkeypatch):
    config_path = _write_minimal_gen2_config(tmp_path, "ring-crps", ensemble_size=2)

    monkeypatch.setattr(sys, "argv", ["train_gen2.py", "-c", str(config_path)])
    with pytest.raises(ValueError, match="data-parallel world size"):
        train_gen2.main_cli()
