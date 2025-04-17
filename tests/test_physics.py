import os

import xarray as xr

import torch
from credit.physics_core import ModelLevelPressures

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]),
                      "config")

def test_pressure_on_mlevs():
    # test on gh actions
    level_info = xr.open_dataset(os.path.join(TEST_FILE_DIR, "data/level_info_test.nc"))
    a_vals, b_vals = level_info["a_model"], level_info["b_model"]
    
    device = "cpu"
    a_tensor = torch.tensor(a_vals.values).to(device) / 100
    b_tensor = torch.tensor(b_vals.values).to(device)
    levels = len(a_tensor)

    batch, ntime, nlat, nlon = 2, 2, 90, 180

    y_pred = torch.ones((batch, 62, ntime, nlat, nlon)) * 1013.
    sp_index = 32

    a_tensor = a_tensor.view(1, levels, 1, 1, 1)
    b_tensor = b_tensor.view(1, levels, 1, 1, 1)
    plev_converter = ModelLevelPressures(a_tensor,
                                               b_tensor,
                                               plev_dim=1)
    thickness = plev_converter.compute_mlev_thickness(y_pred[:, sp_index: sp_index + 1])
    
    assert torch.all(thickness >= 1e-6)
    assert thickness.shape == (batch, levels, ntime, nlat, nlon)

if __name__ == "__main__":
    test_pressure_on_mlevs()