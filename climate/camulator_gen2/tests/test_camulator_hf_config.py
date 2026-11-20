"""The shipped config's postblock chains, built and run end to end on synthetic files.

Every file path in ``configs/camulator_hf.yml`` is swapped for a small synthetic
stand-in (4 x 6 grid, the real 32 levels and 147 output channels), then the
training chain, HFLoss and the rollout chain run on one batch.
"""

import copy
import os

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from camulator_gen2.make_scaler import SOURCE, VARIABLES, main as make_scaler
from credit.losses import load_loss
from credit.postblock import apply_postblocks, build_postblocks

B, L, H, W = 2, 32, 4, 6
CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "camulator_hf.yml")

PHYSICAL = {
    "U": 5.0,
    "V": 2.0,
    "T": 250.0,
    "Qtot": 5e-3,
    "PS": 1.0e5,
    "TREFHT": 288.0,
    "PRECT": 2e-3,
    "QFLX": -1e-3,
    "TS": 289.0,
    "CLDHGH": 0.4,
    "CLDLOW": 0.4,
    "CLDMED": 0.4,
    "TAUX": 0.1,
    "TAUY": 0.1,
    "U10": 6.0,
    "FSDS_J": 4e6,
    "FLDS_J": 7e6,
    "FSUS": 1e6,
    "FLUS": 8e6,
    "SHFLX": -4e5,
    "LHFLX": -2e6,
    "FSUTOA": 100.0,
    "FLUT": 240.0,
    "SOLIN": 340.0,
    "SST": 290.0,
    "ICEFRAC": 0.1,
    "co2vmr_3d": 3.7e-4,
}


def _key(field_type, dim, name):
    return f"{SOURCE}/{field_type}/{dim}/{name}"


@pytest.fixture
def files(tmp_path):
    mean, std = {}, {}
    for (_, dim), names in VARIABLES.items():
        for name in names:
            m, s = PHYSICAL[name], max(abs(PHYSICAL[name]) * 0.05, 1e-6)
            if name == "PS":
                mean[name] = (("latitude", "longitude"), np.full((H, W), m))
                std[name] = (("latitude", "longitude"), np.full((H, W), s))
            elif dim == "3d":
                mean[name] = (("level",), np.full(L, m))
                std[name] = (("level",), np.full(L, s))
            else:
                mean[name], std[name] = ((), np.float64(m)), ((), np.float64(s))
    paths = {k: str(tmp_path / f"{k}.nc") for k in ("mean", "std", "statics")}
    xr.Dataset(mean).to_netcdf(paths["mean"])
    xr.Dataset(std).to_netcdf(paths["std"])

    lat, lon = np.linspace(-90.0, 90.0, H), np.linspace(0.0, 300.0, W)
    lon2d, lat2d = np.meshgrid(lon, lat)
    xr.Dataset(
        {
            "lon2d": (("y", "x"), lon2d),
            "lat2d": (("y", "x"), lat2d),
            "hyai": (("ilev",), np.linspace(200.0, 0.0, L + 1)),
            "hybi": (("ilev",), np.linspace(0.0, 1.0, L + 1)),
            "PHIS": (("y", "x"), np.full((H, W), 1000.0)),
        },
        coords={"latitude": lat, "longitude": lon},
    ).to_netcdf(paths["statics"])

    paths["scaler"] = str(tmp_path / "scaler.json")
    make_scaler(["--mean", paths["mean"], "--std", paths["std"], "--out", paths["scaler"]])
    return paths


def _swap_paths(node, files):
    """Point every file path in the config at the synthetic stand-ins."""
    replace = {
        "scaler_path": "scaler",
        "mean_path": "mean",
        "std_path": "std",
        "save_loc_physics": "statics",
        "latitude_weights": "statics",
    }
    if isinstance(node, dict):
        return {k: files[replace[k]] if k in replace else _swap_paths(v, files) for k, v in node.items()}
    if isinstance(node, list):
        return [_swap_paths(v, files) for v in node]
    return node


@pytest.fixture
def conf(files):
    with open(CONFIG) as f:
        return _swap_paths(yaml.safe_load(f), files)


def _channel_map():
    cmap, cursor = {}, 0
    for (field_type, dim), names in VARIABLES.items():
        if field_type == "dynamic_forcing":
            continue
        for name in names:
            n = L if dim == "3d" else 1
            cmap[_key(field_type, dim, name)] = {"slice": slice(cursor, cursor + n), "orig_shape": (n, 1)}
            cursor += n
    return cmap, cursor


def _batch(requires_grad):
    cmap, n_ch = _channel_map()
    assert n_ch == 147
    g = torch.Generator().manual_seed(0)
    y_pred = (0.5 * torch.randn(B, n_ch, H, W, generator=g)).requires_grad_(requires_grad)
    x_physical = {SOURCE: {}}
    for (field_type, dim), names in VARIABLES.items():
        if field_type == "diagnostic":
            continue
        for name in names:
            n = L if dim == "3d" else 1
            x_physical[SOURCE][_key(field_type, dim, name)] = torch.full((B, n, 1, H, W), PHYSICAL[name])
    return {
        "y_pred": y_pred,
        "y": 0.5 * torch.randn(B, n_ch, H, W, generator=g),
        "x_physical": x_physical,
        "metadata": {"target": {"_channel_map": cmap}},
    }, cmap


def test_training_chain_and_loss(conf):
    blocks = build_postblocks(conf, phase="per_step")
    assert list(blocks) == [
        "reconstruct",
        "tracer_fixer",
        "inverse_scale",
        "mass_fixer",
        "flatten_hf",
        "water_fixer",
        "energy_fixer_updown",
        "flatten",
        "reconstruct_target",
        "scaler_target",
    ]

    batch, cmap = _batch(requires_grad=True)
    y_pred_in = batch["y_pred"]
    out = apply_postblocks(blocks, batch)
    for key in ("y_pred_hf", "y_pred", "water_conservation_loss", "y_processed", "y_target_processed"):
        assert key in out, key
    assert torch.isfinite(out["y_pred_hf"]).all() and torch.isfinite(out["y_pred"]).all()

    # y_pred_hf is the state before the water and energy fixers: they changed PRECT and T only.
    hf, fixed = out["y_pred_hf"], out["y_pred"]
    precip, temp, wind = (
        cmap[_key(*k)]["slice"]
        for k in [("diagnostic", "2d", "PRECT"), ("prognostic", "3d", "T"), ("prognostic", "3d", "U")]
    )
    assert not torch.allclose(hf[:, precip], fixed[:, precip])
    assert not torch.allclose(hf[:, temp], fixed[:, temp])
    torch.testing.assert_close(hf[:, wind], fixed[:, wind])

    criterion = load_loss(conf)
    loss = criterion(out)
    loss.backward()
    assert torch.isfinite(loss)
    assert y_pred_in.grad is not None and y_pred_in.grad.abs().sum() > 0


def test_rollout_chain(conf):
    rollout_conf = copy.deepcopy(conf)
    rollout_conf["postblocks"] = conf["inference"]["postblocks"]
    blocks = build_postblocks(rollout_conf, phase="per_step")
    assert list(blocks) == [
        "reconstruct",
        "tracer_fixer",
        "wind_filter",
        "inverse_scale",
        "mass_fixer",
        "water_fixer",
        "energy_fixer_updown",
    ]

    batch, _ = _batch(requires_grad=False)
    out = apply_postblocks(blocks, batch)
    for var_tensors in out["y_processed"].values():
        for key, t in var_tensors.items():
            assert torch.isfinite(t).all(), key
