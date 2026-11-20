"""HFLoss against the HF gen1 VariableTotalLoss2D and trainer behavior."""

import numpy as np
import pytest
import torch
import xarray as xr

from camulator_gen2.losses import HFLoss
from credit.losses.base import BaseLoss

B, C, H, W = 2, 5, 7, 6


@pytest.fixture
def lat_file(tmp_path):
    # includes both poles, like the CAMulator statics file (-90 ... 90)
    ds = xr.Dataset(
        coords={"latitude": np.linspace(-90.0, 90.0, H).astype(np.float32), "longitude": np.arange(W) * 60.0}
    )
    path = tmp_path / "statics.nc"
    ds.to_netcdf(path)
    return str(path)


def _hf_lat_weights(lat_file):
    """HF credit/losses/weighted_loss.py:get_lat_weights."""
    ds = xr.open_dataset(lat_file)
    lat = torch.from_numpy(ds["latitude"].values).float()
    weights = torch.cos(torch.deg2rad(lat)).clamp(min=1e-4) ** 1.0
    weights = weights / weights.mean()
    return weights.unsqueeze(1).expand(-1, ds["longitude"].shape[0])


def _hf_loss(target, pred, lat_weights):
    """HF VariableTotalLoss2D.forward with mse, latitude weights, no variable weights."""
    loss = torch.nn.MSELoss(reduction="none")(target, pred)
    per_channel = [torch.mul(loss[:, i], lat_weights).mean() for i in range(loss.shape[1])]
    return torch.mean(torch.stack(per_channel))


def _batch():
    g = torch.Generator().manual_seed(0)
    cmap = {
        "CESM/prognostic/3d/T": {"slice": slice(0, 3)},
        "CESM/prognostic/2d/PS": {"slice": slice(3, 4)},
        "CESM/diagnostic/2d/PRECT": {"slice": slice(4, 5)},
    }
    return {
        "y": torch.randn(B, C, H, W, generator=g),
        "y_pred_hf": torch.randn(B, C, H, W, generator=g),
        "y_pred": torch.randn(B, C, H, W, generator=g),
        "water_conservation_loss": torch.tensor(0.3),
        "metadata": {"target": {"_channel_map": cmap}},
    }


def test_is_a_baseloss_for_trainer_dispatch(lat_file):
    loss = HFLoss(lat_file)
    assert isinstance(loss, BaseLoss)
    assert loss.log_variance is None


def test_pole_rows_use_the_hf_clamp(lat_file):
    loss = HFLoss(lat_file)
    expected = _hf_lat_weights(lat_file)
    assert torch.equal(loss.lat_weights, expected)
    assert (loss.lat_weights[0] > 0).all() and (loss.lat_weights[-1] > 0).all()


def test_training_scores_post_mass_prediction_plus_penalty(lat_file):
    batch = _batch()
    loss = HFLoss(lat_file, water_penalty_weight=0.05)
    got = loss(batch)
    expected = _hf_loss(batch["y"], batch["y_pred_hf"], _hf_lat_weights(lat_file)) + 0.05 * 0.3
    assert torch.equal(got, expected)


def test_validation_scores_fixed_prediction_without_penalty(lat_file):
    batch = _batch()
    loss = HFLoss(lat_file, water_penalty_weight=0.05)
    with torch.no_grad():
        got = loss(batch)
    expected = _hf_loss(batch["y"], batch["y_pred"], _hf_lat_weights(lat_file))
    assert torch.equal(got, expected)


def test_missing_penalty_is_an_error_unless_disabled(lat_file):
    batch = _batch()
    del batch["water_conservation_loss"]
    with pytest.raises(KeyError, match="HFWaterFixer"):
        HFLoss(lat_file)(batch)
    HFLoss(lat_file, water_penalty_weight=0)(batch)


def test_missing_prediction_key_names_the_flatten_block(lat_file):
    batch = _batch()
    del batch["y_pred_hf"]
    with pytest.raises(KeyError, match="flatten_to_tensor"):
        HFLoss(lat_file)(batch)


def test_per_variable_losses(lat_file):
    batch = _batch()
    loss = HFLoss(lat_file)
    loss(batch)
    w = _hf_lat_weights(lat_file)
    sq = (batch["y"] - batch["y_pred_hf"]) ** 2
    channel = torch.stack([(sq[:, i] * w).mean() for i in range(C)])
    assert set(loss.last_var_losses) == {"T", "PS", "PRECT"}
    assert loss.last_var_losses["T"] == pytest.approx(channel[:3].mean().item())
    assert loss.last_var_losses["PRECT"] == pytest.approx(channel[4].item())


def test_gradient_flows_to_scored_prediction_and_penalty(lat_file):
    batch = _batch()
    batch["y_pred_hf"].requires_grad_(True)
    penalty_source = torch.tensor(0.5, requires_grad=True)
    batch["water_conservation_loss"] = penalty_source**2
    HFLoss(lat_file)(batch).backward()
    assert batch["y_pred_hf"].grad.abs().sum() > 0
    assert penalty_source.grad == pytest.approx(0.05 * 2 * 0.5)
