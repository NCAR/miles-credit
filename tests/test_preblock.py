"""test_preblock.py — unit tests for credit.preblock modules."""

import numpy as np
import pytest
import torch
import xarray as xr
from bridgescaler.distributed_tensor import DStandardScalerTensor
from bridgescaler import save_scaler

from credit.preblock.regrid import Regrid
from credit.preblock.scaler import Scaler


# ---------------------------------------------------------------------------
# Fixture — synthetic ESMF weight file (384×576 → 192×288)
# ---------------------------------------------------------------------------


@pytest.fixture
def weight_file(tmp_path):
    """Write a minimal ESMF-compatible weight file and return its path.

    Represents a 2:1 block-average downsampling from a 384×576 source grid to the
    192×288 CREDIT destination grid.  Each destination cell is the average of the
    4 corresponding source cells (weight = 0.25 each).

    Only the variables read by Regrid.__init__ are written:
      - dst_grid_dims: [nlon, nlat] SCRIP/ESMF convention; Regrid reverses with [::-1]
      - row, col, S:   1-based COO sparse entries
      - mask_a/b:      stubs so xarray exposes the n_a / n_b dimension sizes
    """
    n_src_lat, n_src_lon = 384, 576
    n_dst_lat, n_dst_lon = 192, 288

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
    """Downsampling regrid produces the destination grid shape."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    regrid = Regrid(path)
    x = torch.randn(2, n_src_lat, n_src_lon)
    assert regrid(x).shape == (2, n_dst_lat, n_dst_lon)


def test_regrid_uniform_input(weight_file):
    """Block-average regrid: uniform input maps to uniform output of the same value."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    regrid = Regrid(path)
    x = torch.ones(n_src_lat, n_src_lon)
    assert torch.allclose(regrid(x), torch.ones(n_dst_lat, n_dst_lon), atol=1e-5)


def test_regrid_reshape_false(weight_file):
    """reshape_to_xy=False returns a flat (batch, n_b) tensor."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    regrid = Regrid(path, reshape_to_xy=False)
    x = torch.randn(2, n_src_lat, n_src_lon)
    assert regrid(x).shape == (2, n_dst_lat * n_dst_lon)


def test_regrid_flip_axis(weight_file):
    """flip_axis is applied to the input before regridding."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    regrid = Regrid(path)
    regrid_flip = Regrid(path, flip_axis=[-1])
    x = torch.randn(n_src_lat, n_src_lon)
    # For a non-uniform input, flipping before regridding should give a different result
    assert not torch.allclose(regrid(x), regrid_flip(x))


# ---------------------------------------------------------------------------
# Fixture — real DStandardScalerTensor fit on random data
# ---------------------------------------------------------------------------


@pytest.fixture
def scaler_file(tmp_path):
    """Fit a DStandardScalerTensor on random data, save to JSON, return path.

    Uses 16 channels to match typical CREDIT usage.  Spatial size is kept small
    (8×8) so the fixture stays fast.
    """
    n_channels = 16
    x_fit = torch.from_numpy(np.random.random((100, n_channels, 1, 8, 8)))
    scaler = DStandardScalerTensor(channels_last=False)
    scaler.fit(x_fit)
    path = str(tmp_path / "scaler.json")
    save_scaler(scaler, path)
    return path, n_channels


# ---------------------------------------------------------------------------
# Scaler tests
# ---------------------------------------------------------------------------


def test_scaler_output_shape(scaler_file):
    """Transform preserves the input tensor shape."""
    path, n_channels = scaler_file
    scaler = Scaler(path)
    x = torch.from_numpy(np.random.random((2, n_channels, 1, 8, 8)))
    assert scaler(x).shape == x.shape


def test_scaler_transform_changes_values(scaler_file):
    """Transform produces different values than the raw input."""
    path, n_channels = scaler_file
    scaler = Scaler(path)
    x = torch.from_numpy(np.random.random((2, n_channels, 1, 8, 8)))
    assert not torch.allclose(scaler(x).float(), x.float())


def test_scaler_round_trip(scaler_file):
    """transform followed by inverse_transform recovers the original tensor."""
    path, n_channels = scaler_file
    fwd = Scaler(path, inverse=False)
    inv = Scaler(path, inverse=True)
    x = torch.from_numpy(np.random.random((2, n_channels, 1, 8, 8)))
    assert torch.allclose(inv(fwd(x)).float(), x.float(), atol=1e-5)
