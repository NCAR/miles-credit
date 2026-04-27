"""test_preblock.py — unit tests for credit.preblock modules."""

import numpy as np
import pytest
import torch
import xarray as xr

try:
    from bridgescaler.distributed_tensor import DStandardScalerTensor
    from bridgescaler import save_scaler_dict, scale_var_dict
    from credit.preblock.scaler import BridgeScalerTransformer

    _BRIDGESCALER_AVAILABLE = True
except (ImportError, Exception):
    _BRIDGESCALER_AVAILABLE = False

from credit.preblock.regrid import Regridder


def create_synthetic_data() -> dict:
    """
    Creates synthetic data as a nested dictionary of torch tensors.

    Structure: data[source][split][var_name]
    - source: "ERA5"
    - split: "input" | "target"
    - var_name: "era5/pronostic/3d/T" | "era5/pronostic/3d/U" | "era5/pronostic/3d/V"
    - tensor shape: (100, 16, 1, 8, 8)
    """
    shape = (100, 16, 1, 8, 8)
    var_names = [
        "era5/pronostic/3d/T",
        "era5/pronostic/3d/U",
        "era5/pronostic/3d/V",
    ]

    return {"era5": {split: {var: torch.randn(*shape) for var in var_names} for split in ("input", "target")}}


# ---------------------------------------------------------------------------
# Fixture — synthetic ESMF weight file (384×576 → 192×288)
# ---------------------------------------------------------------------------


@pytest.fixture
def weight_file(tmp_path):
    """Write a minimal ESMF-compatible weight file and return its path.

    Represents a 2:1 block-average downsampling from an 8×8 source grid to a
    4×4 destination grid, matching create_synthetic_data().  Each destination
    cell is the average of the 4 corresponding source cells (weight = 0.25 each).

    Only the variables read by Regrid.__init__ are written:
      - dst_grid_dims: [nlon, nlat] SCRIP/ESMF convention; Regrid reverses with [::-1]
      - row, col, S:   1-based COO sparse entries
      - mask_a/b:      stubs so xarray exposes the n_a / n_b dimension sizes
    """
    n_src_lat, n_src_lon = 8, 8
    n_dst_lat, n_dst_lon = 4, 4

    # Build the COO sparse entries with numpy (vectorized, no Python loop)
    j_dst, i_dst = np.indices((n_dst_lat, n_dst_lon))
    j_dst, i_dst = j_dst.ravel(), i_dst.ravel()  # (n_b,)
    dst_idx = j_dst * n_dst_lon + i_dst + 1  # 1-based

    dj = np.array([0, 0, 1, 1])
    di = np.array([0, 1, 0, 1])
    j_src = j_dst[:, None] * 2 + dj  # (n_b, 4)
    i_src = i_dst[:, None] * 2 + di  # (n_b, 4)
    src_idx = j_src * n_src_lon + i_src + 1  # 1-based

    rows = np.repeat(dst_idx, 4)  # (n_s,)
    cols = src_idx.ravel()  # (n_s,)
    vals = np.full(len(rows), 0.25)

    n_a = n_src_lat * n_src_lon
    n_b = n_dst_lat * n_dst_lon

    ds = xr.Dataset(
        {
            "dst_grid_dims": xr.DataArray(np.array([n_dst_lon, n_dst_lat], dtype=np.int32), dims=("dst_grid_rank",)),
            "mask_a": xr.DataArray(np.ones(n_a, dtype=np.int32), dims=("n_a",)),
            "mask_b": xr.DataArray(np.ones(n_b, dtype=np.int32), dims=("n_b",)),
            "row": xr.DataArray(rows.astype(np.int32), dims=("n_s",)),
            "col": xr.DataArray(cols.astype(np.int32), dims=("n_s",)),
            "S": xr.DataArray(vals.astype(np.float64), dims=("n_s",)),
        }
    )
    path = tmp_path / "weights.nc"
    ds.to_netcdf(path)
    return str(path), n_src_lat, n_src_lon, n_dst_lat, n_dst_lon


# ---------------------------------------------------------------------------
# Regrid tests
# ---------------------------------------------------------------------------


def test_regrid_output_shape(weight_file):
    """Downsampling regrid produces the destination grid shape for all splits."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["era5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    batch = create_synthetic_data()
    result = regrid(batch)
    for split in ("input", "target"):
        assert result["era5"][split]["era5/pronostic/3d/T"].shape == (100, 16, 1, n_dst_lat, n_dst_lon)


def test_regrid_uniform_input(weight_file):
    """Block-average regrid: uniform input maps to uniform output of the same value."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["era5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    batch = {"era5": {"input": {"era5/pronostic/3d/T": torch.ones(1, 1, 1, n_src_lat, n_src_lon)}}}
    result = regrid(batch)
    assert torch.allclose(
        result["era5"]["input"]["era5/pronostic/3d/T"],
        torch.ones(1, 1, 1, n_dst_lat, n_dst_lon),
        atol=1e-5,
    )


def test_regrid_reshape_false(weight_file):
    """reshape_to_xy=False returns a flat (prod(lead_dims), n_b) tensor."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["era5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables, reshape_to_xy=False)
    batch = create_synthetic_data()
    result = regrid(batch)
    assert result["era5"]["input"]["era5/pronostic/3d/T"].shape == (100 * 16 * 1, n_dst_lat * n_dst_lon)


def test_regrid_flip_axis(weight_file):
    """flip_axis is applied to the input before regridding."""
    import copy

    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["era5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    regrid_flip = Regridder(path, variables=variables, flip_axis=[-1])
    batch = create_synthetic_data()
    result = regrid(copy.deepcopy(batch))
    result_flip = regrid_flip(copy.deepcopy(batch))
    assert not torch.allclose(
        result["era5"]["input"]["era5/pronostic/3d/T"],
        result_flip["era5"]["input"]["era5/pronostic/3d/T"],
    )


_skip_bridgescaler = pytest.mark.skipif(
    not _BRIDGESCALER_AVAILABLE,
    reason="bridgescaler not available in this environment",
)


# ---------------------------------------------------------------------------
# Fixture — real DStandardScalerTensor fit on random data
# ---------------------------------------------------------------------------


@pytest.fixture
def scaler_file(tmp_path):
    """Fit a DStandardScalerTensor on random data, save to JSON, return path.

    Uses 16 channels to match typical CREDIT usage.  Spatial size is kept small
    (8×8) so the fixture stays fast.
    """
    x_dict = create_synthetic_data()
    variables = x_dict["era5"]["input"].keys()
    scaler = DStandardScalerTensor(channels_last=False)
    scaler_dict = scale_var_dict(x_dict, scaler, method="fit")
    path = str(tmp_path / "scaler.json")
    save_scaler_dict(scaler_dict, path)
    return path, variables, x_dict


# ---------------------------------------------------------------------------
# Scaler tests
# ---------------------------------------------------------------------------

# VAR_NAMES = ["era5/pronostic/3d/T", "era5/pronostic/3d/U", "era5/pronostic/3d/V"]


@_skip_bridgescaler
def test_scaler_output_shape(scaler_file):
    """Transform preserves the input tensor shape for every variable."""
    path, variables, data = scaler_file
    scaler = BridgeScalerTransformer(scaler_path=path, variables=list(variables), method="transform")
    original_shapes = {v: data["era5"]["input"][v].shape for v in variables}
    result = scaler(data)
    for v in variables:
        assert result["era5"]["input"][v].shape == original_shapes[v]


@_skip_bridgescaler
def test_scaler_transform_changes_values(scaler_file):
    """Transform produces different values than the raw input."""
    path, variables, data = scaler_file
    scaler = BridgeScalerTransformer(scaler_path=path, variables=list(variables), method="transform")
    var = list(variables)[0]
    original = data["era5"]["input"][var].clone()
    result = scaler(data)
    assert not torch.allclose(result["era5"]["input"][var].float(), original.float())


@_skip_bridgescaler
def test_scaler_round_trip(scaler_file):
    """transform followed by inverse recovers the original tensor."""
    path, variables, data = scaler_file
    var_list = list(variables)
    fwd = BridgeScalerTransformer(scaler_path=path, variables=var_list, method="transform")
    inv = BridgeScalerTransformer(scaler_path=path, variables=var_list, method="inverse_transform")
    var = var_list[0]
    original = data["era5"]["input"][var].clone()
    data = fwd(data)
    data = inv(data)
    assert torch.allclose(data["era5"]["input"][var].float(), original.float(), atol=1e-5)
