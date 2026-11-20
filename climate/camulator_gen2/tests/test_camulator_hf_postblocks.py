"""HFTracerFixer and HFWaterFixer against the HF (gen1 fork) behavior they reproduce."""

import os

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from camulator_gen2.postblocks import HFTracerFixer, HFWaterFixer
from credit.postblock.base import BasePostblock
from credit.postblock.conservation import GlobalWaterFixer

B, L, H, W = 2, 4, 5, 6
SRC = "CESM"
QTOT = f"{SRC}/prognostic/3d/Qtot"
PRECT = f"{SRC}/diagnostic/2d/PRECT"
CLDHGH = f"{SRC}/diagnostic/2d/CLDHGH"
TRACERS = [QTOT, PRECT, CLDHGH]
THRES = [0.0, 0.0, 0.0]
THRES_MAX = [None, None, 1.0]
CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "camulator_hf.yml")


@pytest.fixture
def stats_files(tmp_path):
    """gen1-style mean/std files: per-level Qtot (very different per level), scalar 2-D stats."""
    mean = xr.Dataset(
        {
            "Qtot": (("level",), np.array([3e-6, 5e-4, 4e-3, 1e-2])),
            "PRECT": ((), np.float64(2e-4)),
            "CLDHGH": ((), np.float64(0.4)),
        }
    )
    std = xr.Dataset(
        {
            "Qtot": (("level",), np.array([3e-7, 4e-4, 3e-3, 6e-3])),
            "PRECT": ((), np.float64(5e-4)),
            "CLDHGH": ((), np.float64(0.3)),
        }
    )
    mean_path, std_path = tmp_path / "mean.nc", tmp_path / "std.nc"
    mean.to_netcdf(mean_path)
    std.to_netcdf(std_path)
    return str(mean_path), str(std_path)


def _hf_channel_stats(mean_path, std_path, bug):
    """HF parser + TracerFixer.__init__: one name per output channel, Qtot repeated per level.

    The HF code took ``values.flatten()[0]`` for every channel (the bug); the fixed
    version takes level k for the k-th repeat.
    """
    mean_ds, std_ds = xr.open_dataset(mean_path), xr.open_dataset(std_path)
    names = ["Qtot"] * L + ["PRECT", "CLDHGH"]
    means, stds, seen = [], [], {}
    for name in names:
        k = seen.get(name, 0)
        seen[name] = k + 1
        m = np.array(mean_ds[name].values).flatten()
        s = np.array(std_ds[name].values).flatten()
        idx = 0 if (bug or m.size == 1) else k
        means.append(float(m[idx]))
        stds.append(float(s[idx]))
    return torch.tensor(means).float(), torch.tensor(stds).float()


def _hf_tracer_forward(y_pred, means, stds):
    """Line-by-line port of the HF TracerFixer.forward denorm branch on a flat (B, C, T, H, W) tensor."""
    y_pred = y_pred.clone()
    thres = [0.0] * L + [0.0, 0.0]
    thres_max = [None] * L + [None, 1.0]
    orig_dtype = y_pred.dtype
    for i in range(y_pred.shape[1]):
        m, s = means[i], stds[i]
        chan = y_pred[:, i, ...].float() * s + m
        chan[chan < thres[i]] = thres[i]
        if thres_max[i] is not None:
            chan[chan >= thres_max[i]] = thres_max[i]
        y_pred[:, i, ...] = ((chan - m) / s).to(dtype=orig_dtype)
    return y_pred


def _nested(flat):
    """Split a flat (B, C, 1, H, W) prediction the way reconstruct does."""
    return {
        "y_processed": {
            SRC: {
                QTOT: flat[:, :L],
                PRECT: flat[:, L : L + 1],
                CLDHGH: flat[:, L + 1 : L + 2],
            }
        }
    }


def _flat(batch):
    p = batch["y_processed"][SRC]
    return torch.cat([p[QTOT], p[PRECT], p[CLDHGH]], dim=1)


@pytest.mark.parametrize("bug", [True, False])
def test_tracer_fixer_matches_hf_bitwise(stats_files, bug):
    torch.manual_seed(0)
    # normalized values wide enough that every channel has points below 0 and clouds above 1
    y = torch.randn(B, L + 2, 1, H, W) * 40.0
    means, stds = _hf_channel_stats(*stats_files, bug=bug)
    expected = _hf_tracer_forward(y, means, stds)

    block = HFTracerFixer(*stats_files, TRACERS, THRES, THRES_MAX, replicate_level0_stats_bug=bug)
    got = _flat(block(_nested(y.clone())))

    assert torch.equal(got, expected)
    assert not torch.equal(got, y)  # something was clipped


def test_level0_bug_leaves_lower_levels_unclipped(stats_files):
    """With the bug, a Qtot value that is negative in physical units survives on lower levels."""
    mean_ds, std_ds = xr.open_dataset(stats_files[0]), xr.open_dataset(stats_files[1])
    m, s = mean_ds["Qtot"].values, std_ds["Qtot"].values
    phys = np.full(L, -1e-3)  # -1 g/kg on every level
    y = torch.zeros(1, L + 2, 1, 1, 1)
    y[0, :L, 0, 0, 0] = torch.tensor((phys - m) / s, dtype=torch.float32)

    buggy = HFTracerFixer(*stats_files, TRACERS, THRES, THRES_MAX, replicate_level0_stats_bug=True)
    fixed = HFTracerFixer(*stats_files, TRACERS, THRES, THRES_MAX, replicate_level0_stats_bug=False)
    q_buggy = buggy(_nested(y.clone()))["y_processed"][SRC][QTOT][0, :, 0, 0, 0].numpy() * s + m
    q_fixed = fixed(_nested(y.clone()))["y_processed"][SRC][QTOT][0, :, 0, 0, 0].numpy() * s + m

    assert np.all(q_fixed > -1e-9)
    assert np.all(q_buggy[1:] < -9e-4)  # levels 1.. keep their negative water


def test_tracer_fixer_gradient(stats_files):
    y = (torch.randn(B, L + 2, 1, H, W) * 40.0).requires_grad_(True)
    block = HFTracerFixer(*stats_files, TRACERS, THRES, THRES_MAX)
    out = _flat(block(_nested(y)))
    out.sum().backward()
    grad = y.grad
    # unclipped points pass the gradient through (s / s = 1), clipped ones get none
    assert torch.allclose(grad[grad != 0], torch.ones_like(grad[grad != 0]), atol=1e-5)
    assert (grad == 0).any()


def test_tracer_fixer_rejects_level_mismatch(stats_files):
    block = HFTracerFixer(*stats_files, TRACERS, THRES, THRES_MAX)
    batch = _nested(torch.zeros(B, L + 2, 1, H, W))
    batch["y_processed"][SRC][QTOT] = torch.zeros(B, L + 1, 1, H, W)
    with pytest.raises(ValueError, match="levels"):
        block(batch)


def test_tracer_fixer_rejects_wrong_threshold_count(stats_files):
    with pytest.raises(ValueError, match="tracer_thres_max"):
        HFTracerFixer(*stats_files, TRACERS, 0.0, [None, 1.0])


# --------------------------------------------------------------------------- #
# HFWaterFixer
# --------------------------------------------------------------------------- #
@pytest.fixture
def statics_file(tmp_path):
    lat = np.linspace(-80.0, 80.0, H)
    lon = np.linspace(0.0, 300.0, W)
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds = xr.Dataset(
        {
            "lon2d": (("y", "x"), lon2d),
            "lat2d": (("y", "x"), lat2d),
            "hyai": (("ilev",), np.linspace(1000.0, 0.0, L + 1)),
            "hybi": (("ilev",), np.linspace(0.0, 1.0, L + 1)),
        }
    )
    path = tmp_path / "statics.nc"
    ds.to_netcdf(path)
    return str(path)


def _water_kwargs(statics_file):
    return dict(
        save_loc_physics=statics_file,
        lon_lat_level_name=["lon2d", "lat2d", "hyai", "hybi"],
        grid_type="sigma",
        midpoint=True,
        lead_time_periods=6,
        q_var=QTOT,
        sp_var=f"{SRC}/prognostic/2d/PS",
        precip_var=PRECT,
        evapor_var=f"{SRC}/diagnostic/2d/QFLX",
    )


def _water_batch(requires_grad=False):
    g = torch.Generator().manual_seed(1)
    ps = f"{SRC}/prognostic/2d/PS"
    qflx = f"{SRC}/diagnostic/2d/QFLX"
    y = {
        QTOT: 0.005 + 0.002 * torch.rand(B, L, 1, H, W, generator=g),
        ps: 1.0e5 + 500.0 * torch.rand(B, 1, 1, H, W, generator=g),
        PRECT: 1.0e-3 * torch.rand(B, 1, 1, H, W, generator=g),
        qflx: -1.0e-3 * torch.rand(B, 1, 1, H, W, generator=g),
    }
    if requires_grad:
        for t in y.values():
            t.requires_grad_(True)
    x = {QTOT: torch.full((B, L, 1, H, W), 0.005), ps: torch.full((B, 1, 1, H, W), 1.0e5)}
    return {"y_processed": {SRC: dict(y)}, "x_physical": {SRC: x}}, y


def test_water_fixer_correction_unchanged_and_penalty(statics_file):
    hf = HFWaterFixer(**_water_kwargs(statics_file))
    stock = GlobalWaterFixer(**_water_kwargs(statics_file))

    batch_hf, y = _water_batch()
    batch_stock, _ = _water_batch()
    precip_before = y[PRECT].clone()
    out_hf = hf(batch_hf)
    out_stock = stock(batch_stock)

    # the correction is the stock gen2 water fixer's, unchanged
    assert torch.equal(out_hf["y_processed"][SRC][PRECT], out_stock["y_processed"][SRC][PRECT])

    # penalty = mean over the batch of (P_correct_ratio - 1)^2, the HF definition
    ratio = (out_hf["y_processed"][SRC][PRECT] / precip_before).flatten(1).mean(1)
    expected = ((ratio - 1.0) ** 2).mean()
    assert out_hf["water_conservation_loss"].ndim == 0
    assert torch.allclose(out_hf["water_conservation_loss"], expected, rtol=1e-4)
    assert out_hf["water_conservation_loss"] > 0


def test_water_penalty_has_gradient(statics_file):
    hf = HFWaterFixer(**_water_kwargs(statics_file))
    batch, y = _water_batch(requires_grad=True)
    hf(batch)["water_conservation_loss"].backward()
    assert y[PRECT].grad is not None and y[PRECT].grad.abs().sum() > 0
    assert y[QTOT].grad is not None and y[QTOT].grad.abs().sum() > 0


def test_custom_objects_register():
    """The config's custom_objects block registers all three classes."""
    from credit.losses import _load_loss_entry
    from credit.postblock import _load_postblock_entry
    from credit.registry import load_custom_objects

    with open(CONFIG) as f:
        conf = yaml.safe_load(f)
    load_custom_objects({"custom_objects": conf["custom_objects"]})

    assert _load_postblock_entry("hf_tracer_fixer") is HFTracerFixer
    assert _load_postblock_entry("hf_water_fixer") is HFWaterFixer
    assert issubclass(HFWaterFixer, BasePostblock)
    from camulator_gen2.losses import HFLoss

    assert _load_loss_entry("hf_loss") is HFLoss
