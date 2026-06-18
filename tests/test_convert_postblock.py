"""test_convert_postblock.py — integration: credit convert → postblocks → rollout simulation.

Tests that `credit convert` on a v1 flat-schema config:
  1. Writes the correct postblocks section (reconstruct then bridgescaler inverse).
  2. Produces JSON scaler files that build into working postblocks.
  3. apply_postblocks correctly inverse-transforms a synthetic prediction tensor,
     simulating the rollout loop without needing a real model or dataset.
"""

import argparse

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

try:
    from credit.cli._convert import _build_bridgescaler_jsons
    from credit.postblock import build_postblocks, apply_postblocks

    _BS_AVAILABLE = True
except (ImportError, Exception):
    _BS_AVAILABLE = False

skip_bs = pytest.mark.skipif(not _BS_AVAILABLE, reason="bridgescaler not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mean_std_nc(tmp_path):
    """Synthetic per-level mean/std NC files: T (3D, 4 levels) and SP (2D)."""
    n_levels = 4
    mean_T = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    std_T = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    mean_SP = np.float32(1013.25)
    std_SP = np.float32(5.0)

    xr.Dataset({"T": xr.DataArray(mean_T, dims=("level",)), "SP": xr.DataArray(mean_SP)}).to_netcdf(
        tmp_path / "mean.nc"
    )
    xr.Dataset({"T": xr.DataArray(std_T, dims=("level",)), "SP": xr.DataArray(std_SP)}).to_netcdf(tmp_path / "std.nc")

    return str(tmp_path / "mean.nc"), str(tmp_path / "std.nc"), n_levels, mean_T, std_T, mean_SP, std_SP


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_bs
def test_convert_writes_postblocks(tmp_path, mean_std_nc):
    """credit convert on a flat v1 config produces reconstruct + bridgescaler postblocks."""
    import credit.cli as cli

    mean_path, std_path, *_ = mean_std_nc

    v1_conf = {
        "trainer": {"type": "era5"},
        "data": {
            "variables": ["T"],
            "surface_variables": ["SP"],
            "mean_path": mean_path,
            "std_path": std_path,
            "forecast_len": 0,
            "valid_forecast_len": 0,
            "backprop_on_timestep": [0, 1],
        },
        "loss": {"training_loss": "mse"},
        "pbs": {"project": "NAML0001", "job_name": "test", "walltime": "01:00:00", "conda": "credit"},
    }
    v1_path = str(tmp_path / "v1.yml")
    out_path = str(tmp_path / "v2.yml")
    with open(v1_path, "w") as f:
        yaml.dump(v1_conf, f)

    args = argparse.Namespace(config=v1_path, output=out_path, defaults=True, level_coord="level")
    cli._convert(args)

    with open(out_path) as f:
        result = yaml.safe_load(f)

    pb = result.get("postblocks", {})
    per_step = pb.get("per_step", {})
    keys = list(per_step.keys())
    assert "reconstruct" in keys
    assert "scaler" in keys
    assert keys.index("reconstruct") < keys.index("scaler"), "reconstruct must come before scaler"
    assert per_step["reconstruct"]["type"] == "reconstruct"
    assert per_step["scaler"]["type"] == "bridgescaler_transformer"
    assert per_step["scaler"]["args"]["method"] == "inverse_transform"
    assert "key" not in per_step["scaler"]["args"], "key should not be set; default y_processed is correct"

    # The generated JSON must load into working postblocks
    postblocks = build_postblocks({"postblocks": pb})
    assert "reconstruct" in postblocks
    assert "scaler" in postblocks


@skip_bs
def test_convert_postblocks_inverse_transform(tmp_path, mean_std_nc):
    """Postblocks built from credit convert output correctly inverse-transform a synthetic prediction.

    Simulates the rollout loop: normalized flat tensor → apply_postblocks →
    nested physical-space dict, then flatten back to (1, C_out, H, W).
    """
    mean_path, std_path, n_levels, mean_T, std_T, mean_SP, std_SP = mean_std_nc

    var_groups = {("prognostic", "3d"): ["T"], ("prognostic", "2d"): ["SP"]}
    pre_json = str(tmp_path / "pre.json")
    post_json = str(tmp_path / "post.json")
    _, post_vars = _build_bridgescaler_jsons(mean_path, std_path, var_groups, pre_json, post_json)

    postblocks_cfg = {
        "per_step": {
            "reconstruct": {"type": "reconstruct"},
            "scaler": {
                "type": "bridgescaler_transformer",
                "args": {
                    "scaler_path": post_json,
                    "variables": post_vars,
                    "method": "inverse_transform",
                },
            },
        }
    }
    postblocks = build_postblocks({"postblocks": postblocks_cfg})

    # Synthetic normalized prediction: (B=1, C_out, T=1, H=4, W=4)
    # All-zeros normalized → inverse should recover the mean values.
    B, n_time, H, W = 1, 1, 4, 4
    C_T, C_SP = n_levels, 1
    y_pred_norm = torch.zeros(B, C_T + C_SP, n_time, H, W)

    # channel_map mirrors what ConcatToTensor / Reconstruct expects
    channel_map = {
        "ERA5/prognostic/3d/T": {"slice": slice(0, C_T * n_time), "orig_shape": (C_T, n_time)},
        "ERA5/prognostic/2d/SP": {"slice": slice(C_T * n_time, (C_T + C_SP) * n_time), "orig_shape": (C_SP, n_time)},
    }
    meta = {"target": {"_channel_map": channel_map}}

    batch_dict = apply_postblocks(postblocks, {"y_pred": y_pred_norm, "metadata": meta})
    pred = batch_dict["y_processed"]

    assert isinstance(pred, dict), "Reconstruct should split y_pred into nested y_processed dict"
    assert "ERA5" in pred
    T_out = pred["ERA5"]["ERA5/prognostic/3d/T"]
    SP_out = pred["ERA5"]["ERA5/prognostic/2d/SP"]

    assert T_out.shape == (B, C_T, n_time, H, W)
    assert SP_out.shape == (B, C_SP, n_time, H, W)

    # Zero normalized → output equals mean
    expected_T = torch.tensor(mean_T, dtype=torch.float32).view(1, C_T, 1, 1, 1).expand_as(T_out)
    expected_SP = torch.tensor([mean_SP], dtype=torch.float32).view(1, C_SP, 1, 1, 1).expand_as(SP_out)
    assert torch.allclose(T_out, expected_T, atol=1e-4), f"T max diff {(T_out - expected_T).abs().max()}"
    assert torch.allclose(SP_out, expected_SP, atol=1e-4), f"SP max diff {(SP_out - expected_SP).abs().max()}"

    # Simulate rollout extraction: flatten nested dict back to (1, C_out, H, W)
    y_pred_phys = torch.cat([pred[k.split("/")[0]][k].flatten(1, 2) for k in channel_map], dim=1)
    assert y_pred_phys.shape == (B, C_T + C_SP, H, W)
