"""test_preblock.py — unit tests for credit.preblock modules."""

import copy

import numpy as np
import pytest
import torch
import xarray as xr

try:
    from bridgescaler.distributed_tensor import DStandardScalerTensor
    from bridgescaler import save_scaler_dict, scale_var_dict
    from credit.preblock.scaler import BridgeScalerTransformer
    from credit.preblock.norm import ERA5Normalizer

    _BRIDGESCALER_AVAILABLE = True
except (ImportError, Exception):
    _BRIDGESCALER_AVAILABLE = False

from credit.preblock.regrid import Regridder


def create_synthetic_data() -> dict:
    """
    Creates synthetic data as a nested dictionary of torch tensors.

    Structure: data[data_type][source][var_name]
    - data_type: "input" | "target"
    - source: "Test_ERA5"
    - var_name: "Test_ERA5/pronostic/3d/T" | ...
    - tensor shape: (100, 16, 1, 8, 8)
    """
    shape = (100, 16, 1, 8, 8)
    var_names = [
        "Test_ERA5/pronostic/3d/T",
        "Test_ERA5/pronostic/3d/U",
        "Test_ERA5/pronostic/3d/V",
    ]

    return {split: {"Test_ERA5": {var: torch.randn(*shape) for var in var_names}} for split in ("input", "target")}


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
    variables = ["Test_ERA5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    batch = create_synthetic_data()
    result = regrid(batch)
    for split in ("input", "target"):
        assert result[split]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"].shape == (100, 16, 1, n_dst_lat, n_dst_lon)


def test_regrid_uniform_input(weight_file):
    """Block-average regrid: uniform input maps to uniform output of the same value."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/pronostic/3d/T": torch.ones(1, 1, 1, n_src_lat, n_src_lon)}}}
    result = regrid(batch)
    assert torch.allclose(
        result["input"]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"],
        torch.ones(1, 1, 1, n_dst_lat, n_dst_lon),
        atol=1e-5,
    )


def test_regrid_reshape_false(weight_file):
    """reshape_to_xy=False returns a flat (prod(lead_dims), n_b) tensor."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables, reshape_to_xy=False)
    batch = create_synthetic_data()
    result = regrid(batch)
    assert result["input"]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"].shape == (100 * 16 * 1, n_dst_lat * n_dst_lon)


def test_regrid_flip_axis(weight_file):
    """flip_axis is applied to the input before regridding."""
    import copy

    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    regrid_flip = Regridder(path, variables=variables, flip_axis=[-1])
    batch = create_synthetic_data()
    result = regrid(copy.deepcopy(batch))
    result_flip = regrid_flip(copy.deepcopy(batch))
    assert not torch.allclose(
        result["input"]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"],
        result_flip["input"]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"],
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
    variables = x_dict["input"]["Test_ERA5"].keys()
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
    original_shapes = {v: data["input"]["Test_ERA5"][v].shape for v in variables}
    result = scaler(data)
    for v in variables:
        assert result["input"]["Test_ERA5"][v].shape == original_shapes[v]


@_skip_bridgescaler
def test_scaler_transform_changes_values(scaler_file):
    """Transform produces different values than the raw input."""
    path, variables, data = scaler_file
    scaler = BridgeScalerTransformer(scaler_path=path, variables=list(variables), method="transform")
    var = list(variables)[0]
    original = data["input"]["Test_ERA5"][var].clone()
    result = scaler(data)
    assert not torch.allclose(result["input"]["Test_ERA5"][var].float(), original.float())


@_skip_bridgescaler
def test_scaler_round_trip(scaler_file):
    """transform followed by inverse recovers the original tensor."""
    path, variables, data = scaler_file
    var_list = list(variables)
    fwd = BridgeScalerTransformer(scaler_path=path, variables=var_list, method="transform")
    inv = BridgeScalerTransformer(scaler_path=path, variables=var_list, method="inverse_transform")
    var = var_list[0]
    original = data["input"]["Test_ERA5"][var].clone()
    data = fwd(data)
    data = inv(data)
    assert torch.allclose(data["input"]["Test_ERA5"][var].float(), original.float(), atol=1e-5)


# ---------------------------------------------------------------------------
# Parity tests: ERA5Normalizer (old) vs BridgeScalerTransformer (new)
# ---------------------------------------------------------------------------
# The converter reads the same mean/std NC files that ERA5Normalizer uses and
# stores them in DStandardScalerTensor objects (var = std**2).  Both normalize
# as (x - mean) / std so the outputs must agree to float32 precision.


@pytest.fixture
def zscore_nc_files(tmp_path):
    """Write synthetic mean/std NetCDF files and return their paths.

    3-D variable 'T': 4 levels, non-trivial per-level mean/std.
    2-D variable 'SP': scalar mean/std.
    """
    n_levels = 4
    mean_T = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    std_T = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    mean_SP = np.float32(1013.25)
    std_SP = np.float32(5.0)

    ds_mean = xr.Dataset(
        {
            "T": xr.DataArray(mean_T, dims=("level",)),
            "SP": xr.DataArray(mean_SP),
        }
    )
    ds_std = xr.Dataset(
        {
            "T": xr.DataArray(std_T, dims=("level",)),
            "SP": xr.DataArray(std_SP),
        }
    )
    mean_path = str(tmp_path / "mean.nc")
    std_path = str(tmp_path / "std.nc")
    ds_mean.to_netcdf(mean_path)
    ds_std.to_netcdf(std_path)
    return mean_path, std_path, n_levels, mean_T, std_T, mean_SP, std_SP


@pytest.fixture
def zscore_batch():
    """Batch dict with 3-D and 2-D prognostic variables matching zscore_nc_files."""
    B, L, T, H, W = 2, 4, 1, 8, 8
    return {
        "input": {
            "era5": {
                "era5/prognostic/3d/T": torch.randn(B, L, T, H, W),
                "era5/prognostic/2d/SP": torch.randn(B, 1, T, H, W),
            }
        },
        "target": {
            "era5": {
                "era5/prognostic/3d/T": torch.randn(B, L, T, H, W),
                "era5/prognostic/2d/SP": torch.randn(B, 1, T, H, W),
            }
        },
    }


@_skip_bridgescaler
def test_preblock_era5normalizer_vs_bridgescaler_3d(tmp_path, zscore_nc_files, zscore_batch):
    """ERA5Normalizer and BridgeScalerTransformer produce identical output for 3-D vars."""
    from credit.cli._convert import _build_bridgescaler_jsons

    mean_path, std_path, n_levels, mean_T, std_T, _, _ = zscore_nc_files
    var_groups = {("prognostic", "3d"): ["T"], ("prognostic", "2d"): ["SP"]}
    pre_json = str(tmp_path / "pre_scaler.json")
    post_json = str(tmp_path / "post_scaler.json")
    pre_keys, _ = _build_bridgescaler_jsons(mean_path, std_path, var_groups, pre_json, post_json)

    batch_old = copy.deepcopy(zscore_batch)
    batch_new = copy.deepcopy(zscore_batch)

    # Old path
    old_norm = ERA5Normalizer(mean_path=mean_path, std_path=std_path)
    batch_old = old_norm(batch_old)

    # New path
    new_norm = BridgeScalerTransformer(scaler_path=pre_json, variables=pre_keys, method="transform")
    batch_new = new_norm(batch_new)

    for split in ("input", "target"):
        old_T = batch_old[split]["era5"]["era5/prognostic/3d/T"]
        new_T = batch_new[split]["era5"]["era5/prognostic/3d/T"]
        assert torch.allclose(old_T, new_T, atol=1e-5), (
            f"3D var mismatch in {split}: max diff {(old_T - new_T).abs().max()}"
        )


@_skip_bridgescaler
def test_preblock_era5normalizer_vs_bridgescaler_2d(tmp_path, zscore_nc_files, zscore_batch):
    """ERA5Normalizer and BridgeScalerTransformer produce identical output for 2-D vars."""
    from credit.cli._convert import _build_bridgescaler_jsons

    mean_path, std_path, *_ = zscore_nc_files
    var_groups = {("prognostic", "3d"): ["T"], ("prognostic", "2d"): ["SP"]}
    pre_json = str(tmp_path / "pre_scaler.json")
    post_json = str(tmp_path / "post_scaler.json")
    pre_keys, _ = _build_bridgescaler_jsons(mean_path, std_path, var_groups, pre_json, post_json)

    batch_old = copy.deepcopy(zscore_batch)
    batch_new = copy.deepcopy(zscore_batch)

    old_norm = ERA5Normalizer(mean_path=mean_path, std_path=std_path)
    batch_old = old_norm(batch_old)

    new_norm = BridgeScalerTransformer(scaler_path=pre_json, variables=pre_keys, method="transform")
    batch_new = new_norm(batch_new)

    for split in ("input", "target"):
        old_SP = batch_old[split]["era5"]["era5/prognostic/2d/SP"]
        new_SP = batch_new[split]["era5"]["era5/prognostic/2d/SP"]
        assert torch.allclose(old_SP, new_SP, atol=1e-5), (
            f"2D var mismatch in {split}: max diff {(old_SP - new_SP).abs().max()}"
        )


@_skip_bridgescaler
def test_postblock_bridgescaler_inverse_matches_manual(tmp_path, zscore_nc_files):
    """BridgeScalerTransformer postblock inverse_transform matches (x*std + mean) applied manually."""
    from credit.cli._convert import _build_bridgescaler_jsons
    from credit.postblock.scaler import BridgeScalerTransformer as PostBridgeScaler

    mean_path, std_path, n_levels, mean_T, std_T, mean_SP, std_SP = zscore_nc_files
    var_groups = {("prognostic", "3d"): ["T"], ("prognostic", "2d"): ["SP"]}
    pre_json = str(tmp_path / "pre_scaler.json")
    post_json = str(tmp_path / "post_scaler.json")
    _, post_prog_vars = _build_bridgescaler_jsons(mean_path, std_path, var_groups, pre_json, post_json)

    B, T, H, W = 2, 1, 8, 8
    # Simulate a normalized prediction dict (post-Reconstruct structure)
    pred_T = torch.randn(B, n_levels, T, H, W)
    pred_SP = torch.randn(B, 1, T, H, W)
    prediction = {"era5": {"prognostic": {"3d": {"T": pred_T.clone()}, "2d": {"SP": pred_SP.clone()}}}}
    batch_dict = {"prediction": prediction}

    postblock = PostBridgeScaler(scaler_path=post_json, variables=post_prog_vars, method="inverse_transform")
    result = postblock(batch_dict)

    # Manual inverse: x * std + mean
    mean_T_t = torch.tensor(mean_T, dtype=torch.float32).view(1, n_levels, 1, 1, 1)
    std_T_t = torch.tensor(std_T, dtype=torch.float32).view(1, n_levels, 1, 1, 1)
    expected_T = pred_T * std_T_t + mean_T_t

    mean_SP_t = torch.tensor([mean_SP], dtype=torch.float32).view(1, 1, 1, 1, 1)
    std_SP_t = torch.tensor([std_SP], dtype=torch.float32).view(1, 1, 1, 1, 1)
    expected_SP = pred_SP * std_SP_t + mean_SP_t

    got_T = result["prediction"]["era5"]["prognostic"]["3d"]["T"]
    got_SP = result["prediction"]["era5"]["prognostic"]["2d"]["SP"]

    assert torch.allclose(got_T, expected_T, atol=1e-4), (
        f"T inverse mismatch: max diff {(got_T - expected_T).abs().max()}"
    )
    assert torch.allclose(got_SP, expected_SP, atol=1e-4), (
        f"SP inverse mismatch: max diff {(got_SP - expected_SP).abs().max()}"
    )
