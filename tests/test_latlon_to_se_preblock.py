import torch
import xarray as xr

from credit.preblock.latlon_to_se import TripoleToSEPreBlock
from credit.preblock.norm import ERA5Normalizer


def _write_weight_file(path):
    ds = xr.Dataset(
        {
            "row": ("n_s", [1, 2, 3]),
            "col": ("n_s", [1, 2, 3]),
            "S": ("n_s", [1.0, 1.0, 1.0]),
        },
        attrs={"n_src": 4, "n_dst": 3},
    )
    ds.to_netcdf(path)


def test_tripole_to_se_regrids_source_grid_tensor(tmp_path):
    weight_file = tmp_path / "weights.nc"
    _write_weight_file(weight_file)
    block = TripoleToSEPreBlock(weight_file)

    x = torch.tensor([[[[[10.0, 20.0], [30.0, 40.0]]]]])
    out = block._regrid_tensor(x)

    assert out.shape == (1, 1, 1, 3)
    torch.testing.assert_close(out, torch.tensor([[[[10.0, 20.0, 30.0]]]]))


def test_tripole_to_se_passes_through_already_se_tensor(tmp_path):
    weight_file = tmp_path / "weights.nc"
    _write_weight_file(weight_file)
    block = TripoleToSEPreBlock(weight_file)

    x = torch.randn(2, 4, 1, 1, 3)
    out = block._regrid_tensor(x)

    assert out.shape == (2, 4, 1, 3)
    torch.testing.assert_close(out, x.reshape(2, 4, 1, 3))


def test_era5_normalizer_passes_through_already_se_rollout_tensor(tmp_path):
    mean_path = tmp_path / "mean.nc"
    std_path = tmp_path / "std.nc"
    xr.Dataset({"T": ((), 10.0)}).to_netcdf(mean_path)
    xr.Dataset({"T": ((), 2.0)}).to_netcdf(std_path)
    norm = ERA5Normalizer(str(mean_path), str(std_path))

    se_tensor = torch.full((1, 1, 1, 1, 3), 5.0)
    raw_tensor = torch.full((1, 1, 1, 2, 2), 14.0)

    assert norm._normalize_tensor("ERA5/prognostic/2d/T", se_tensor) is se_tensor
    torch.testing.assert_close(
        norm._normalize_tensor("ERA5/prognostic/2d/T", raw_tensor),
        torch.full_like(raw_tensor, 2.0),
    )
