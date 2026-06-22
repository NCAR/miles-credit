"""
Tests for the gen2 (name-based) conservation postblocks in
``credit/postblock/conservation.py`` and the ``Reconstruct`` / ``FlattenToTensor``
round trip in ``credit/postblock/reconstruct.py``.

Everything is self-contained: a tiny synthetic statics file is written to a tmp
path, and synthetic ``y_processed`` / ``ic_raw`` super-dicts are built by hand on
a small grid. No campaign-storage paths, no network.

The t0 input state is supplied under ``x_physical`` (the per-step physical input
key the trainer stashes), which is the default ``input_source_key`` the fixers
read.
"""

import numpy as np
import torch
import xarray as xr
import pytest

from credit.postblock.reconstruct import Reconstruct, FlattenToTensor
from credit.postblock.conservation import (
    TracerFixer,
    GlobalMassFixer,
    GlobalWaterFixer,
    GlobalEnergyFixerUpDown,
)

SRC = "ERA5"
B, L, H, W = 2, 4, 4, 8

VARS_3D = ["U", "V", "T", "Qtot"]
VARS_2D_PROG = ["PS", "TREFHT"]
VARS_2D_DIAG = [
    "PRECT",
    "QFLX",
    "FSUTOA",
    "FLUT",
    "FSDS_J",
    "FLDS_J",
    "FSUS",
    "FLUS",
    "SHFLX",
    "LHFLX",
    "TS",
]


def key(field_type, dim, name):
    return f"{SRC}/{field_type}/{dim}/{name}"


def _physical_value(name, shape):
    """Return a physically plausible tensor for a variable."""
    if name == "T":
        return torch.full(shape, 250.0)
    if name == "Qtot":
        return torch.full(shape, 0.005)
    if name in ("U", "V"):
        return torch.full(shape, 5.0)
    if name == "PS":
        return torch.full(shape, 1.0e5)
    if name == "TREFHT":
        return torch.full(shape, 288.0)
    if name == "PRECT":
        return torch.full(shape, 1.0e-4)
    if name == "QFLX":
        return torch.full(shape, 1.0e-5)
    # radiation / heat fluxes (J m-2 over the step) and clouds
    return torch.full(shape, 100.0)


def build_channel_map():
    """Output channel map in canonical order: prognostic 3d, prognostic 2d, diagnostic 2d."""
    cmap = {}
    cursor = 0
    for name in VARS_3D:
        k = key("prognostic", "3d", name)
        cmap[k] = {"slice": slice(cursor, cursor + L), "orig_shape": (L, 1)}
        cursor += L
    for name in VARS_2D_PROG:
        k = key("prognostic", "2d", name)
        cmap[k] = {"slice": slice(cursor, cursor + 1), "orig_shape": (1, 1)}
        cursor += 1
    for name in VARS_2D_DIAG:
        k = key("diagnostic", "2d", name)
        cmap[k] = {"slice": slice(cursor, cursor + 1), "orig_shape": (1, 1)}
        cursor += 1
    return cmap, cursor


def build_super_dict(requires_grad=False):
    """Build y_processed (physical), ic_raw (physical), and metadata."""
    cmap, n_ch = build_channel_map()
    y_processed = {SRC: {}}
    for k, info in cmap.items():
        nl = info["orig_shape"][0]
        name = k.split("/")[-1]
        t = _physical_value(name, (B, nl, 1, H, W)).clone()
        t.requires_grad_(requires_grad)
        y_processed[SRC][k] = t

    # input (t0) state: prognostic vars + SOLIN forcing, under the per-step
    # physical input key the fixers default to.
    x_physical = {SRC: {}}
    for name in VARS_3D:
        x_physical[SRC][key("prognostic", "3d", name)] = _physical_value(name, (B, L, 1, H, W))
    for name in VARS_2D_PROG:
        x_physical[SRC][key("prognostic", "2d", name)] = _physical_value(name, (B, 1, 1, H, W))
    x_physical[SRC][key("dynamic_forcing", "2d", "SOLIN")] = torch.full((B, 1, 1, H, W), 400.0)

    metadata = {"target": {"_channel_map": cmap}}
    return {"y_processed": y_processed, "x_physical": x_physical, "metadata": metadata}, cmap, n_ch


@pytest.fixture
def statics_file(tmp_path):
    lat = np.linspace(-80.0, 80.0, H)
    lon = np.linspace(0.0, 315.0, W)
    lon2d, lat2d = np.meshgrid(lon, lat)
    hyai = np.linspace(1000.0, 0.0, L + 1)  # Pa, top -> surface
    hybi = np.linspace(0.0, 1.0, L + 1)  # unitless
    phis = np.full((H, W), 1000.0)
    ds = xr.Dataset(
        {
            "lon2d": (("y", "x"), lon2d),
            "lat2d": (("y", "x"), lat2d),
            "hyai": (("ilev",), hyai),
            "hybi": (("ilev",), hybi),
            "PHIS": (("y", "x"), phis),
        }
    )
    path = tmp_path / "statics.nc"
    ds.to_netcdf(path)
    return str(path)


def _physics_args(statics_file):
    return dict(
        save_loc_physics=statics_file,
        lon_lat_level_name=["lon2d", "lat2d", "hyai", "hybi"],
        grid_type="sigma",
        midpoint=True,
    )


# --------------------------------------------------------------------------- #
# Reconstruct <-> FlattenToTensor round trip + gradient
# --------------------------------------------------------------------------- #
def test_reconstruct_flatten_roundtrip_and_grad():
    cmap, n_ch = build_channel_map()
    y_pred = torch.randn(B, n_ch, H, W, requires_grad=True)
    batch = {"y_pred": y_pred, "metadata": {"target": {"_channel_map": cmap}}}

    recon = Reconstruct(detach=False)
    batch = recon(batch)
    flat = FlattenToTensor()  # no scaler -> pure flatten
    batch = flat(batch)

    out = batch["y_pred"]
    assert out.shape == y_pred.shape
    assert torch.allclose(out, y_pred, atol=1e-6)

    # gradient flows end to end
    out.sum().backward()
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


def test_reconstruct_detach_true_blocks_grad():
    cmap, n_ch = build_channel_map()
    y_pred = torch.randn(B, n_ch, H, W, requires_grad=True)
    batch = {"y_pred": y_pred, "metadata": {"target": {"_channel_map": cmap}}}
    batch = Reconstruct(detach=True)(batch)
    leaf = batch["y_processed"][SRC][key("prognostic", "3d", "U")]
    assert not leaf.requires_grad


# --------------------------------------------------------------------------- #
# TracerFixer
# --------------------------------------------------------------------------- #
def test_tracer_fixer_clamps_and_grad():
    batch, cmap, _ = build_super_dict(requires_grad=False)
    qk = key("prognostic", "3d", "Qtot")
    # leaf tensor with a mix of negative and positive values
    vals = torch.linspace(-1.0, 1.0, B * L * H * W).reshape(B, L, 1, H, W).clone()
    leaf = vals.requires_grad_(True)
    batch["y_processed"][SRC][qk] = leaf
    fixer = TracerFixer(tracer_vars=[qk], tracer_thres=0.0)
    batch = fixer(batch)
    out = batch["y_processed"][SRC][qk]
    assert (out >= 0.0).all()
    out.sum().backward()
    assert leaf.grad is not None
    # gradient passes through where the value was above threshold, zero where clamped
    assert torch.isfinite(leaf.grad).all()


# --------------------------------------------------------------------------- #
# Mass fixer: surface pressure correction conserves dry-air mass
# --------------------------------------------------------------------------- #
def test_mass_fixer_conserves_dry_air(statics_file):
    batch, cmap, _ = build_super_dict(requires_grad=True)
    spk = key("prognostic", "2d", "PS")
    # perturb predicted surface pressure so it is initially off
    batch["y_processed"][SRC][spk] = batch["y_processed"][SRC][spk] * 1.05

    fixer = GlobalMassFixer(
        q_var=key("prognostic", "3d", "Qtot"),
        sp_var=spk,
        **_physics_args(statics_file),
    )
    core = fixer.core

    def dry_mass(sp_5d):
        q = batch["y_processed"][SRC][key("prognostic", "3d", "Qtot")][:, :, 0, ...]
        return core.total_dry_air_mass(q, sp_5d[:, 0, 0, ...])

    sp_before = batch["y_processed"][SRC][spk].detach().clone()
    q_input = batch["x_physical"][SRC][key("prognostic", "3d", "Qtot")][:, :, -1, ...]
    sp_input = batch["x_physical"][SRC][spk][:, 0, -1, ...]
    target_mass = core.total_dry_air_mass(q_input, sp_input)

    batch = fixer(batch)
    sp_after = batch["y_processed"][SRC][spk]
    assert sp_after.shape == (B, 1, 1, H, W)
    assert torch.isfinite(sp_after).all()

    mass_before = dry_mass(sp_before)
    mass_after = dry_mass(sp_after)
    # correction should drive predicted dry-air mass to the t0 target
    err_before = (mass_before - target_mass).abs()
    err_after = (mass_after - target_mass).abs()
    assert (err_after <= err_before + 1e-6).all()
    assert torch.allclose(mass_after, target_mass, rtol=1e-4)

    # gradient flows
    sp_after.sum().backward()
    grad = batch["y_processed"][SRC][key("prognostic", "3d", "Qtot")]
    # Qtot leaf participated in the sp correction
    assert grad.grad is not None


# --------------------------------------------------------------------------- #
# Water fixer
# --------------------------------------------------------------------------- #
def test_water_fixer_runs_and_grad(statics_file):
    batch, cmap, _ = build_super_dict(requires_grad=True)
    pk = key("diagnostic", "2d", "PRECT")
    fixer = GlobalWaterFixer(
        q_var=key("prognostic", "3d", "Qtot"),
        sp_var=key("prognostic", "2d", "PS"),
        precip_var=pk,
        evapor_var=key("diagnostic", "2d", "QFLX"),
        lead_time_periods=6,
        **_physics_args(statics_file),
    )
    precip_in = batch["y_processed"][SRC][pk]
    batch = fixer(batch)
    out = batch["y_processed"][SRC][pk]
    assert out.shape == (B, 1, 1, H, W)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert precip_in.grad is not None


# --------------------------------------------------------------------------- #
# Energy fixer (up/down)
# --------------------------------------------------------------------------- #
def test_energy_fixer_updown_runs_and_grad(statics_file):
    batch, cmap, _ = build_super_dict(requires_grad=True)
    tk = key("prognostic", "3d", "T")
    fixer = GlobalEnergyFixerUpDown(
        T_var=tk,
        q_var=key("prognostic", "3d", "Qtot"),
        U_var=key("prognostic", "3d", "U"),
        V_var=key("prognostic", "3d", "V"),
        sp_var=key("prognostic", "2d", "PS"),
        surface_geopotential_name="PHIS",
        toa_down_solar_input_var=key("dynamic_forcing", "2d", "SOLIN"),
        toa_up_solar_var=key("diagnostic", "2d", "FSUTOA"),
        toa_up_olr_var=key("diagnostic", "2d", "FLUT"),
        surf_down_solar_var=key("diagnostic", "2d", "FSDS_J"),
        surf_up_solar_var=key("diagnostic", "2d", "FSUS"),
        surf_down_lw_var=key("diagnostic", "2d", "FLDS_J"),
        surf_up_lw_var=key("diagnostic", "2d", "FLUS"),
        surf_sh_var=key("diagnostic", "2d", "SHFLX"),
        surf_lh_var=key("diagnostic", "2d", "LHFLX"),
        lead_time_periods=6,
        **_physics_args(statics_file),
    )
    t_in = batch["y_processed"][SRC][tk]
    batch = fixer(batch)
    out = batch["y_processed"][SRC][tk]
    assert out.shape == (B, L, 1, H, W)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert t_in.grad is not None


# --------------------------------------------------------------------------- #
# Full chain: fixers leave y_processed physical; flatten rebuilds y_pred
# --------------------------------------------------------------------------- #
def test_full_chain_flatten_preserves_physical(statics_file):
    batch, cmap, n_ch = build_super_dict(requires_grad=False)
    # run the conservation fixers
    GlobalMassFixer(
        q_var=key("prognostic", "3d", "Qtot"),
        sp_var=key("prognostic", "2d", "PS"),
        **_physics_args(statics_file),
    )(batch)

    phys_snapshot = batch["y_processed"][SRC][key("prognostic", "3d", "T")].detach().clone()
    flat = FlattenToTensor()  # no scaler
    batch = flat(batch)
    assert batch["y_pred"].shape == (B, n_ch, H, W)
    # y_processed untouched by a no-scaler flatten
    assert torch.allclose(batch["y_processed"][SRC][key("prognostic", "3d", "T")], phys_snapshot)
