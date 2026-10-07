"""Tests for credit.postblock.regrid_se_to_latlon.SEToLatLonPostBlock.

Includes the end-to-end Reconstruct -> SEToLatLonPostBlock path exercised by
config/gen_2/examples/wxformer_cubesphere_next_wb2.yml (PR #478 review):
a cube-sphere/SE-grid model's raw (B, C, 1, ncol) output must reconstruct and
regrid to lat-lon with exactly the same rank a 2D-grid model produces, or the
netCDF writer in credit/output_gen2.py breaks on the extra dimension.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from credit.postblock.reconstruct import Reconstruct
from credit.postblock.regrid_se_to_latlon import SEToLatLonPostBlock


@pytest.fixture
def identity_weight_file(tmp_path):
    """An ESMF-format weight file mapping an ncol-length SE grid 1:1 onto an
    (nlat, nlon) lat-lon grid of the same size (identity regrid, weight=1)."""
    nlat, nlon = 3, 4
    ncol = nlat * nlon

    ds = xr.Dataset(
        {
            "dst_grid_dims": xr.DataArray(np.array([nlon, nlat], dtype=np.int32), dims=("dst_grid_rank",)),
            "row": xr.DataArray(np.arange(1, ncol + 1, dtype=np.int32), dims=("n_s",)),
            "col": xr.DataArray(np.arange(1, ncol + 1, dtype=np.int32), dims=("n_s",)),
            "S": xr.DataArray(np.ones(ncol, dtype=np.float64), dims=("n_s",)),
            "mask_a": xr.DataArray(np.ones(ncol, dtype=np.int32), dims=("n_a",)),
            "mask_b": xr.DataArray(np.ones(ncol, dtype=np.int32), dims=("n_b",)),
        }
    )
    path = tmp_path / "weights.nc"
    ds.to_netcdf(path)
    return str(path), nlat, nlon, ncol


def test_regrid_tensor_identity(identity_weight_file):
    """Identity weights: regridding reshapes (..., ncol) -> (..., nlat, nlon)
    without changing any values."""
    path, nlat, nlon, ncol = identity_weight_file
    block = SEToLatLonPostBlock(weight_file=path, keys=["y_pred"])

    x = torch.randn(2, 5, ncol)
    y = block._regrid_tensor(x)
    assert tuple(y.shape) == (2, 5, nlat, nlon)
    assert torch.equal(y.reshape(2, 5, ncol), x)


def test_forward_skips_missing_key(identity_weight_file):
    """A key absent from batch_dict is silently skipped, not an error."""
    path, _, _, ncol = identity_weight_file
    block = SEToLatLonPostBlock(weight_file=path, keys=["not_present"])
    batch = {"y_pred": torch.randn(1, 2, ncol)}
    out = block(batch)
    assert out["y_pred"].shape == (1, 2, ncol)  # untouched


def test_reconstruct_then_regrid_matches_2d_grid_rank(identity_weight_file):
    """A cube-sphere model's raw (B, C, 1, ncol) output, after Reconstruct and
    then SEToLatLonPostBlock(keys=["y_processed"]), ends up at exactly the same
    rank a 2D lat-lon-grid model's (B, C, H, W) output would after the same
    Reconstruct step alone -- no extra dimension survives from the frame axis
    or from the 1D -> 2D spatial regrid.
    """
    path, nlat, nlon, ncol = identity_weight_file
    source = "Test_ERA5"
    key_3d = f"{source}/prognostic/3d/T"
    key_2d = f"{source}/prognostic/2d/SP"
    output_map = {
        key_3d: {"slice": slice(0, 4), "orig_shape": (4, 1)},
        key_2d: {"slice": slice(4, 5), "orig_shape": (1, 1)},
    }
    metadata = {"target": {"_channel_map": output_map}}

    B = 2
    y_pred_flat = torch.randn(B, 5, ncol)  # (B, C, ncol) -- no frame dim
    y_pred_cube = y_pred_flat.unsqueeze(2)  # (B, C, 1, ncol) -- what CubedWXFormer/WXFormerColumn actually return

    reconstruct = Reconstruct()
    regrid = SEToLatLonPostBlock(weight_file=path, keys=["y_processed"])

    for y_pred in (y_pred_flat, y_pred_cube):
        batch = {"y_pred": y_pred, "metadata": metadata}
        out = regrid(reconstruct(batch))
        pred = out["y_processed"][source]

        # (B, n_levels, n_time, nlat, nlon) -- 5D, same rank output_gen2._to_dataset
        # expects from a 2D-grid model, never 6D.
        assert pred[key_3d].shape == (B, 4, 1, nlat, nlon)
        assert pred[key_2d].shape == (B, 1, 1, nlat, nlon)
