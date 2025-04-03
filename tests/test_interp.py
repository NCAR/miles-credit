"""Tests interp.py."""

from credit.interp import full_state_pressure_interpolation
import xarray as xr
import os
import numpy as np


def test_full_state_pressure_interpolation():
    """Tests full state pressure interpolation function."""
    path_to_test = os.path.abspath(os.path.dirname(__file__))
    input_file = os.path.join(path_to_test, "data/test_interp.nc")
    ds = xr.open_dataset(input_file)
    pressure_levels = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
    height_levels = np.arange(0, 5500.0, 500.0)
    interp_ds = full_state_pressure_interpolation(
        ds,
        ds["Z_GDS4_SFC"].values,
        pressure_levels=pressure_levels,
        height_levels=height_levels,
        lat_var="lat",
        lon_var="lon",
    )
    for var in ["U", "V", "T", "Q"]:
        assert (
            interp_ds[f"{var}_PRES"].shape[1] == pressure_levels.size
        ), "Pressure level mismatch"
        assert ~np.any(np.isnan(interp_ds[f"{var}_PRES"])), "NaN found"
    return
