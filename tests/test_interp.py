"""Tests interp.py."""

from credit.interp import full_state_pressure_interpolation, interp_pressure_to_hybrid_levels
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
        assert interp_ds[f"{var}_PRES"].shape[1] == pressure_levels.size, "Pressure level mismatch"
        assert ~np.any(np.isnan(interp_ds[f"{var}_PRES"])), "NaN found"
    return


def test_interp_pressure_to_hybrid_levels():
    """Tests that a field linear in log(pressure) is recovered exactly on hybrid levels."""
    ny, nx = 2, 3
    pressure_levels = np.array([100.0, 300.0, 500.0, 700.0, 850.0, 1000.0])
    model_pressure_1d = np.array([150.0, 400.0, 600.0, 800.0, 950.0])
    model_pressure = np.broadcast_to(model_pressure_1d.reshape(-1, 1, 1), (model_pressure_1d.size, ny, nx)).copy()
    surface_pressure = np.full((ny, nx), 1013.0)

    # A field defined as log(pressure) is interpolated exactly by log-space linear interpolation,
    # giving an analytic ground truth (expected output is simply log(model_pressure)).
    log_pressure_levels = np.log(pressure_levels)
    pressure_var = np.broadcast_to(log_pressure_levels.reshape(-1, 1, 1), (pressure_levels.size, ny, nx)).copy()

    model_var = interp_pressure_to_hybrid_levels(pressure_var, pressure_levels, model_pressure, surface_pressure)

    assert model_var.shape == model_pressure.shape
    np.testing.assert_allclose(model_var, np.log(model_pressure), rtol=1e-6)


def test_interp_pressure_to_hybrid_levels_below_surface_masking():
    """Tests that only pressure levels above the local surface pressure are used for interpolation."""
    pressure_levels = np.array([100.0, 300.0, 500.0, 700.0, 850.0, 1000.0])
    model_pressure = np.array([150.0, 400.0, 600.0]).reshape(-1, 1, 1)
    log_pressure_levels = np.log(pressure_levels)
    pressure_var = log_pressure_levels.reshape(-1, 1, 1).copy()
    # Only the 100, 300, and 500 hPa levels are above ground at this point.
    surface_pressure = np.array([[600.0]])

    model_var = interp_pressure_to_hybrid_levels(pressure_var, pressure_levels, model_pressure, surface_pressure)

    air_levels = pressure_levels < surface_pressure[0, 0]
    expected = np.interp(
        np.log(model_pressure[:, 0, 0]),
        log_pressure_levels[air_levels],
        pressure_var[air_levels, 0, 0],
    )
    np.testing.assert_allclose(model_var[:, 0, 0], expected, rtol=1e-6)
