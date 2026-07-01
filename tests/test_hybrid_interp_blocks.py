"""Tests for the hybrid-to-hybrid level interpolation pre/postblocks.

Covers the coefficient loader, parity of the vmapped torch column function
with the numba reference used by ``credit.gefs.interpolate_vertical_levels``
(``create_pressure_grid`` + ``interp_hybrid_to_hybrid_levels``), and both
block protocols.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from credit.interp import create_pressure_grid, interp_hybrid_to_hybrid_levels
from credit.postblock import POSTBLOCK_REGISTRY
from credit.postblock._interp_utils import load_hybrid_level_coefficients
from credit.postblock.hybrid_interp import HybridLevelInterp as HybridLevelInterpPostblock
from credit.postblock.hybrid_interp import interp_column_hybrid_to_hybrid
from credit.preblock import PREBLOCK_REGISTRY
from credit.preblock.hybrid_interp import HybridLevelInterp as HybridLevelInterpPreblock

# Two distinct, monotonic hybrid level sets (interface coefficients, top → surface)
N_SRC_HALF = 25  # 24 midpoint levels
N_DST_HALF = 17  # 16 midpoint levels
_S_SRC = np.linspace(0.0, 1.0, N_SRC_HALF)
_S_DST = np.linspace(0.0, 1.0, N_DST_HALF)
SRC_A_HALF = 60_000.0 * _S_SRC * (1.0 - _S_SRC)
SRC_B_HALF = _S_SRC**2
DST_A_HALF = 40_000.0 * _S_DST * (1.0 - _S_DST)
DST_B_HALF = _S_DST**1.5


def _write_level_file(path, a_half, b_half):
    xr.Dataset({"a_half": ("half_level", a_half), "b_half": ("half_level", b_half)}).to_netcdf(path)
    return str(path)


@pytest.fixture(scope="module")
def level_files(tmp_path_factory):
    """Write source and destination interface-coefficient files."""
    tmp_dir = tmp_path_factory.mktemp("level_info")
    return {
        "source": _write_level_file(tmp_dir / "source_levels.nc", SRC_A_HALF, SRC_B_HALF),
        "dest": _write_level_file(tmp_dir / "dest_levels.nc", DST_A_HALF, DST_B_HALF),
    }


# ---------------------------------------------------------------------------
# load_hybrid_level_coefficients
# ---------------------------------------------------------------------------


def test_loader_interfaces_to_midpoints(level_files):
    """Interface coefficients are averaged to midpoints (create_pressure_grid convention)."""
    a, b = load_hybrid_level_coefficients(level_files["source"], "a_half", "b_half", on_interfaces=True)
    assert a.shape == (N_SRC_HALF - 1,)
    np.testing.assert_allclose(a.numpy(), 0.5 * (SRC_A_HALF[:-1] + SRC_A_HALF[1:]), rtol=1e-6)
    np.testing.assert_allclose(b.numpy(), 0.5 * (SRC_B_HALF[:-1] + SRC_B_HALF[1:]), rtol=1e-6)


def test_loader_midpoints_and_level_subset(level_files):
    """on_interfaces=False uses values directly; levels subsets 1-based midpoints."""
    a, b = load_hybrid_level_coefficients(
        level_files["source"], "a_half", "b_half", on_interfaces=False, levels=[1, 5, 10]
    )
    np.testing.assert_allclose(a.numpy(), SRC_A_HALF[[0, 4, 9]], rtol=1e-6)
    np.testing.assert_allclose(b.numpy(), SRC_B_HALF[[0, 4, 9]], rtol=1e-6)


def test_loader_vcoord_convention(tmp_path):
    """A 2D variable with a_var == b_var splits into a (row 0) and b (row 1) like GFS vcoord."""
    vcoord = np.stack([SRC_A_HALF, SRC_B_HALF])
    path = tmp_path / "gfs_ctrl.nc"
    xr.Dataset({"vcoord": (("nvcoord", "levp"), vcoord)}).to_netcdf(path)
    a, b = load_hybrid_level_coefficients(str(path), "vcoord", "vcoord", on_interfaces=True)
    np.testing.assert_allclose(a.numpy(), 0.5 * (SRC_A_HALF[:-1] + SRC_A_HALF[1:]), rtol=1e-6)
    np.testing.assert_allclose(b.numpy(), 0.5 * (SRC_B_HALF[:-1] + SRC_B_HALF[1:]), rtol=1e-6)


# ---------------------------------------------------------------------------
# Parity with credit.gefs.interpolate_vertical_levels internals
# ---------------------------------------------------------------------------


def test_parity_with_interp_py_reference():
    """The vmapped column function matches create_pressure_grid + interp_hybrid_to_hybrid_levels."""
    rng = np.random.default_rng(7)
    heights_m = np.array([[0.0, 500.0, 1500.0], [2500.0, 3500.0, 100.0]])
    sp = 101_325.0 * np.exp(-heights_m / 8000.0)  # (y, x)
    n_src = N_SRC_HALF - 1
    temperature = 216.0 + 72.0 * np.linspace(0.0, 1.0, n_src)[:, None, None] + rng.normal(0.0, 1.0, (n_src, 2, 3))
    u_wind = rng.normal(0.0, 15.0, (n_src, 2, 3))

    src_pressure, _ = create_pressure_grid(sp, SRC_A_HALF, SRC_B_HALF)
    dst_pressure, _ = create_pressure_grid(sp, DST_A_HALF, DST_B_HALF)
    ref_t = interp_hybrid_to_hybrid_levels(temperature, src_pressure, dst_pressure)
    ref_u = interp_hybrid_to_hybrid_levels(u_wind, src_pressure, dst_pressure)

    def _flat(arr_3d):  # (levels, y, x) → (y*x, levels)
        return torch.tensor(arr_3d.reshape(arr_3d.shape[0], -1).T, dtype=torch.float64)

    src_a_mid = torch.tensor(0.5 * (SRC_A_HALF[:-1] + SRC_A_HALF[1:]), dtype=torch.float64)
    src_b_mid = torch.tensor(0.5 * (SRC_B_HALF[:-1] + SRC_B_HALF[1:]), dtype=torch.float64)
    dst_a_mid = torch.tensor(0.5 * (DST_A_HALF[:-1] + DST_A_HALF[1:]), dtype=torch.float64)
    dst_b_mid = torch.tensor(0.5 * (DST_B_HALF[:-1] + DST_B_HALF[1:]), dtype=torch.float64)
    vinterp = torch.vmap(interp_column_hybrid_to_hybrid, in_dims=(0, 0, None, None, None, None), chunk_size=2)
    out = vinterp(
        torch.stack([_flat(temperature), _flat(u_wind)], dim=1),
        torch.tensor(sp.ravel(), dtype=torch.float64),
        src_a_mid,
        src_b_mid,
        dst_a_mid,
        dst_b_mid,
    )  # (N, 2, n_dst)

    n_dst = N_DST_HALF - 1
    np.testing.assert_allclose(out[:, 0].T.reshape(n_dst, 2, 3).numpy(), ref_t, rtol=1e-8)
    np.testing.assert_allclose(out[:, 1].T.reshape(n_dst, 2, 3).numpy(), ref_u, rtol=1e-8)


# ---------------------------------------------------------------------------
# Postblock and preblock protocols
# ---------------------------------------------------------------------------

SRC = "GFS"
T_VAR = f"{SRC}/prognostic/3d/temperature"
Q_VAR = f"{SRC}/prognostic/3d/specific_humidity"
SP_VAR = f"{SRC}/prognostic/2d/surface_pressure"

N_SRC_MID = N_SRC_HALF - 1
N_DST_MID = N_DST_HALF - 1


def _engine_args(level_files, **overrides):
    args = {
        "variables": [T_VAR, Q_VAR],
        "surface_pressure_var": SP_VAR,
        "source_level_info_file": level_files["source"],
        "dest_level_info_file": level_files["dest"],
    }
    args.update(overrides)
    return args


def _nested_state(B=2, n_time=1, H=4, W=6, n_levels=N_SRC_MID):
    torch.manual_seed(0)
    return {
        SRC: {
            T_VAR: 220.0 + 70.0 * torch.rand(B, n_levels, n_time, H, W),
            Q_VAR: 0.012 * torch.rand(B, n_levels, n_time, H, W),
            SP_VAR: torch.full((B, 1, n_time, H, W), 95_000.0),
        }
    }


def test_postblock_output_shapes(level_files):
    """Interpolated variables land on the destination level count, in place."""
    batch = {"y_processed": _nested_state()}
    mod = HybridLevelInterpPostblock(**_engine_args(level_files))
    out = mod(batch)
    for var in (T_VAR, Q_VAR):
        assert out["y_processed"][SRC][var].shape == (2, N_DST_MID, 1, 4, 6)
        assert torch.isfinite(out["y_processed"][SRC][var]).all()
    assert out["y_processed"][SRC][SP_VAR].shape == (2, 1, 1, 4, 6)  # untouched


def test_postblock_identity_when_levels_match(level_files):
    """Interpolating onto the same level set returns the input values."""
    batch = {"y_processed": _nested_state()}
    original = batch["y_processed"][SRC][T_VAR].clone()
    mod = HybridLevelInterpPostblock(**_engine_args(level_files, dest_level_info_file=level_files["source"]))
    out = mod(batch)
    assert torch.allclose(out["y_processed"][SRC][T_VAR], original, rtol=1e-6)


def test_postblock_monotone_profile_stays_in_range(level_files):
    """Interpolation of a monotone profile is bounded by its end values."""
    batch = {"y_processed": _nested_state()}
    t_min, t_max = 200.0, 290.0
    batch["y_processed"][SRC][T_VAR] = (
        torch.linspace(t_min, t_max, N_SRC_MID).view(1, -1, 1, 1, 1).expand(2, -1, 1, 4, 6).clone()
    )
    mod = HybridLevelInterpPostblock(**_engine_args(level_files))
    out = mod(batch)
    t_out = out["y_processed"][SRC][T_VAR]
    assert (t_out >= t_min).all() and (t_out <= t_max).all()


def test_postblock_missing_variables_skipped_and_level_mismatch_raises(level_files):
    """Absent variables are skipped silently; wrong level counts raise ValueError."""
    batch = {"y_processed": {SRC: {SP_VAR: torch.full((1, 1, 1, 4, 6), 95_000.0)}}}
    mod = HybridLevelInterpPostblock(**_engine_args(level_files))
    mod(batch)  # no interp variables present — no-op

    bad = {"y_processed": _nested_state(n_levels=N_SRC_MID + 3)}
    with pytest.raises(ValueError, match="levels"):
        mod(bad)


def test_preblock_transforms_requested_data_types(level_files):
    """The preblock interpolates input and target without mutating the caller's batch."""
    batch = {"input": _nested_state(), "target": _nested_state()}
    original_input_t = batch["input"][SRC][T_VAR]
    mod = HybridLevelInterpPreblock(**_engine_args(level_files))
    out = mod(batch)
    assert out["input"][SRC][T_VAR].shape == (2, N_DST_MID, 1, 4, 6)
    assert out["target"][SRC][Q_VAR].shape == (2, N_DST_MID, 1, 4, 6)
    # caller's dict still holds the original source-level tensor
    assert batch["input"][SRC][T_VAR] is original_input_t
    assert batch["input"][SRC][T_VAR].shape[1] == N_SRC_MID


def test_preblock_data_type_selection(level_files):
    """data_types=["input"] leaves target untouched; invalid types raise."""
    batch = {"input": _nested_state(), "target": _nested_state()}
    mod = HybridLevelInterpPreblock(data_types=["input"], **_engine_args(level_files))
    out = mod(batch)
    assert out["input"][SRC][T_VAR].shape[1] == N_DST_MID
    assert out["target"][SRC][T_VAR].shape[1] == N_SRC_MID

    with pytest.raises(ValueError, match="data_types"):
        HybridLevelInterpPreblock(data_types=["metadata"], **_engine_args(level_files))


def test_registered_in_both_registries():
    """The block is reachable from both factory registries."""
    assert POSTBLOCK_REGISTRY["hybrid_level_interp"] is HybridLevelInterpPostblock
    assert PREBLOCK_REGISTRY["hybrid_level_interp"] is HybridLevelInterpPreblock
