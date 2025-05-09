"""Functions to interpolate data to pressure and height coordinates."""

import numpy as np
from numba import njit
import xarray as xr
from .physics_constants import RDGAS, GRAVITY
import os


def full_state_pressure_interpolation(
    state_dataset: xr.Dataset,
    surface_geopotential: np.ndarray,
    pressure_levels: np.ndarray = np.array([500.0, 850.0]),
    interp_fields: tuple[str] = ("U", "V", "T", "Q"),
    pres_ending: str = "_PRES",
    height_levels: np.ndarray = None,
    height_ending: str = "_HEIGHT",
    temperature_var: str = "T",
    q_var: str = "Q",
    surface_pressure_var: str = "SP",
    geopotential_var: str = "Z",
    time_var: str = "time",
    lat_var: str = "latitude",
    lon_var: str = "longitude",
    pres_var: str = "pressure",
    level_var: str = "level",
    height_var: str = "height_agl",
    model_level_file: str = "../credit/metadata/ERA5_Lev_Info.nc",
    verbose: int = 1,
    a_model_name: str = "a_model",
    b_model_name: str = "b_model",
    a_half_name: str = "a_half",
    b_half_name: str = "b_half",
) -> xr.Dataset:
    """Interpolate the full state of the model to pressure and height coordinates.

    Interpolate full model state variables from model levels to pressure levels and height levels. The raw CREDIT
    model output are on hybrid sigma-pressure vertical levels, which start as terrain following near the surface
    and relax to constant pressure levels aloft. The state variables for CREDIT models (and hydrostatic models
    more generally) are u, v, temperature, specific humidity, and surface pressure, with surface geopotential as
    a static variable. To perform pressure and height interpolation, the following steps happen:

    1. Pressure is calculated on every full model level (middle of the vertical grid box) and half level (top and
        bottom of the vertical grid box starting at the surface and ending at the model top of the atmospshere). This
        requires knowing the a and b coefficients for the model levels.
    2.  Geopotential on each hybrid sigma-pressure level is calculated from surface geopotential, pressure, temperature,
        and specific humidity as a vertical integral calculation. The calculation is sensitive to numerical precision,
        so data are cast to float64 before calling the geopotential calculation.
    3. Interpolation is done from hybrid sigma-pressure levels to fixed pressure levels. Everything is interpolated
        linearly with log(pressure) as the x coordinate. For pressure levels below ground (where pressure of level > surface
        pressure), special extrapolation routines are done for temperature and geopotential while constant extrapolation
        is assumed for u, v, and q.
    4. Interpolation to height above ground level is also performed at the end. Heights are defined in meters.

    Args:
        state_dataset (xr.Dataset): state variables being interpolated
        surface_geopotential (np.ndarray): surface geopotential levels in units m^2/s^2.
        pressure_levels (np.ndarray): pressure levels for interpolation in hPa.
        interp_fields (tuple[str]): fields to be interpolated.
        pres_ending (str): ending string to attach to pressure interpolated variables.
        height_levels (np.ndarray): height levels for interpolation to height above ground level in meters.
        height_ending (str): ending string to attach to height interpolated variables.
        temperature_var (str): temperature variable to be interpolated (units K).
        q_var (str): mixing ratio/specific humidity variable to be interpolated (units kg/kg).
        surface_pressure_var (str): surface pressure variable (units Pa).
        geopotential_var (str): geopotential variable being derived (units m^2/s^2).
        time_var (str): time coordinate
        lat_var (str): latitude coordinate
        lon_var (str): longitude coordinate
        pres_var (str): pressure coordinate
        level_var (str): name of level coordinate
        height_var (str): height coordinate
        model_level_file (str): relative path to file containing model levels.
        verbose (int): verbosity level. If verbose > 0, print progress.
        a_model_name (str): Name of A weight at level midpoints in sigma coordinate formula. 'a_model' by default.
        b_model_name (str): Name of B weight at level midpoints in sigma coordinate formula. 'b_model' by default.
        a_half_name (str): Name of A weight at level interfaces in sigma coordinate formula. 'a_half' by default.
        b_half_name (str): Name of B weight at level interfaces in sigma coordinate formula. 'b_half' by default.

    Returns:
        pressure_ds (xr.Dataset): Dataset containing pressure interpolated variables.

    """
    path_to_file = os.path.abspath(os.path.dirname(__file__))
    model_level_file = os.path.join(path_to_file, model_level_file)
    pressure_levels = np.array(pressure_levels)
    with xr.open_dataset(model_level_file) as mod_lev_ds:
        valid_levels = np.isin(
            mod_lev_ds[level_var].values, state_dataset[level_var].values
        )
        a_model = mod_lev_ds[a_model_name].values[valid_levels]
        b_model = mod_lev_ds[b_model_name].values[valid_levels]
        a_half_full = mod_lev_ds[a_half_name].values
        b_half_full = mod_lev_ds[b_half_name].values

    pres_dims = (time_var, pres_var, lat_var, lon_var)
    surface_dims = (time_var, lat_var, lon_var)
    coords = {
        time_var: state_dataset[time_var],
        pres_var: pressure_levels,
        lat_var: state_dataset[lat_var],
        lon_var: state_dataset[lon_var],
    }
    coords_surface = {
        time_var: state_dataset[time_var],
        lat_var: state_dataset[lat_var],
        lon_var: state_dataset[lon_var],
    }

    pressure_ds = xr.Dataset(
        data_vars={
            f + pres_ending: xr.DataArray(
                coords=coords,
                dims=pres_dims,
                name=f + pres_ending,
                attrs=state_dataset[f].attrs,
            )
            for f in interp_fields
        },
        coords=coords,
    )
    pressure_ds[geopotential_var] = xr.DataArray(
        coords=state_dataset[temperature_var].coords,
        dims=state_dataset[temperature_var].dims,
        name=geopotential_var,
    )
    pressure_ds["P"] = xr.DataArray(
        coords=state_dataset[temperature_var].coords,
        dims=state_dataset[temperature_var].dims,
        name="P",
    )
    pressure_ds[geopotential_var + pres_ending] = xr.DataArray(
        coords=coords, dims=pres_dims, name=geopotential_var + pres_ending
    )
    pressure_ds["mean_sea_level_" + pres_var] = xr.DataArray(
        coords=coords_surface, dims=surface_dims, name="mean_sea_level_" + pres_var
    )
    if height_levels is not None:
        height_levels = np.array(height_levels)
        coords_height = {
            time_var: state_dataset[time_var],
            height_var: height_levels,
            lat_var: state_dataset[lat_var],
            lon_var: state_dataset[lon_var],
        }
        height_dims = (time_var, height_var, lat_var, lon_var)
        height_shape = (
            coords_height[time_var].size,
            coords_height[height_var].size,
            coords_height[lat_var].size,
            coords_height[lon_var].size,
        )
        for var in interp_fields:
            pressure_ds[var + height_ending] = xr.DataArray(
                data=np.zeros(height_shape, dtype=np.float32),
                coords=coords_height,
                dims=height_dims,
                name=var + height_ending,
            )
        pressure_ds["P" + height_ending] = xr.DataArray(
            data=np.zeros(height_shape, dtype=np.float32),
            coords=coords_height,
            dims=height_dims,
            name=var + height_ending,
        )

    for t, time in enumerate(state_dataset[time_var]):
        interp_full_data = {}
        surface_pressure_data = state_dataset[surface_pressure_var][t].values.astype(
            np.float64
        )

        pressure_grid, half_pressure_grid = create_reduced_pressure_grid(
            surface_pressure_data, a_model, b_model
        )

        interp_full_data["P"], full_half_pressure_grid = create_pressure_grid(
            surface_pressure_data, a_half_full, b_half_full
        )

        interp_full_data[temperature_var] = interp_hybrid_to_hybrid_levels(
            state_dataset[temperature_var][t].values.astype(np.float64),
            pressure_grid,
            interp_full_data["P"],
        )
        interp_full_data[q_var] = interp_hybrid_to_hybrid_levels(
            state_dataset[q_var][t].values.astype(np.float64),
            pressure_grid,
            interp_full_data["P"],
        )
        for interp_field in interp_fields:
            if interp_field not in interp_full_data.keys():
                interp_full_data[interp_field] = interp_hybrid_to_hybrid_levels(
                    state_dataset[interp_field][t].values.astype(np.float64),
                    pressure_grid,
                    interp_full_data["P"],
                )

        geopotential_full_grid = geopotential_from_model_vars(
            surface_geopotential.astype(np.float64),
            surface_pressure_data,
            interp_full_data[temperature_var],
            interp_full_data[q_var],
            full_half_pressure_grid,
        )
        pressure_ds["P"][t] = pressure_grid
        pressure_ds[geopotential_var][t] = geopotential_full_grid[valid_levels]
        for interp_field in interp_fields:
            if interp_field == temperature_var:
                pressure_ds[interp_field + pres_ending][t] = (
                    interp_temperature_to_pressure_levels(
                        interp_full_data[interp_field],
                        interp_full_data["P"] / 100.0,
                        pressure_levels,
                        state_dataset[surface_pressure_var][t].values / 100.0,
                        surface_geopotential,
                        geopotential_full_grid,
                    )
                )
            else:
                pressure_ds[interp_field + pres_ending][t] = (
                    interp_hybrid_to_pressure_levels(
                        interp_full_data[interp_field],
                        interp_full_data["P"] / 100.0,
                        pressure_levels,
                    )
                )
        pressure_ds[geopotential_var + pres_ending][t] = (
            interp_geopotential_to_pressure_levels(
                geopotential_full_grid,
                interp_full_data["P"] / 100.0,
                pressure_levels,
                state_dataset[surface_pressure_var][t].values / 100.0,
                surface_geopotential,
                interp_full_data[temperature_var],
            )
        )
        pressure_ds["mean_sea_level_" + pres_var][t] = mean_sea_level_pressure(
            state_dataset[surface_pressure_var][t].values,
            interp_full_data[temperature_var],
            interp_full_data["P"],
            surface_geopotential,
            geopotential_full_grid,
        )
        if height_levels is not None:
            for interp_field in interp_full_data.keys():
                height_var = interp_field + height_ending
                pressure_ds[height_var][t] = interp_hybrid_to_height_agl(
                    interp_full_data[interp_field],
                    height_levels,
                    geopotential_full_grid,
                    surface_geopotential,
                )
    return pressure_ds


@njit
def create_pressure_grid(surface_pressure, model_a_half, model_b_half):
    """Create a pressure 3D grid from a full set of vertical levels.

    Create a 3D pressure field at model levels from the surface pressure field and the hybrid sigma-pressure
    coefficients from ECMWF. Conversion is `pressure_3d = a + b * SP`.

    Args:
        surface_pressure (np.ndarray): (time, latitude, longitude) or (latitude, longitude) grid in units of Pa.
        model_a_half (np.ndarray): a coefficients at each model level being used in units of Pa.
        model_b_half (np.ndarray): b coefficients at each model level being used (unitness).

    Returns:
        pressure_3d: 3D pressure field with dimensions of surface_pressure and number of levels from model_a and model_b.

    """
    assert (
        model_a_half.size == model_b_half.size
    ), "Model pressure coefficient arrays do not match."
    if surface_pressure.ndim == 3:
        # Generate the 3D pressure field for a time series of surface pressure grids
        pressure_3d = np.zeros(
            (
                surface_pressure.shape[0],
                model_a_half.shape[0] - 1,
                surface_pressure.shape[1],
                surface_pressure.shape[2],
            ),
            dtype=surface_pressure.dtype,
        )
        pressure_3d_half = np.zeros(
            (
                surface_pressure.shape[0],
                model_a_half.shape[0],
                surface_pressure.shape[1],
                surface_pressure.shape[2],
            ),
            dtype=surface_pressure.dtype,
        )
        model_a_3d = model_a_half.reshape(-1, 1, 1)
        model_b_3d = model_b_half.reshape(-1, 1, 1)
        for i in range(surface_pressure.shape[0]):
            pressure_3d_half[i] = model_a_3d + model_b_3d * surface_pressure[i]
            pressure_3d[i] = 0.5 * (pressure_3d_half[:-1] + pressure_3d_half[1:])
    else:
        # Generate the 3D pressure field for a single surface pressure grid.
        model_a_3d = model_a_half.reshape(-1, 1, 1)
        model_b_3d = model_b_half.reshape(-1, 1, 1)
        pressure_3d_half = model_a_3d + model_b_3d * surface_pressure
        pressure_3d = 0.5 * (pressure_3d_half[:-1] + pressure_3d_half[1:])
    return pressure_3d, pressure_3d_half


@njit
def create_reduced_pressure_grid(surface_pressure, model_a_full, model_b_full):
    """Create a pressure 3D grid using sparse vertical levels.

    Create a 3D pressure field at model levels from the surface pressure field and the reduced set of hybrid sigma-
    pressure levels used in the CREDIT models. This function assumes that the coefficients for the full levels are
    being passed and then derives the half levels by taking the geometric means of the a and b coefficients on full
    levels. Conversion is `pressure_3d = a + b * SP`.

    Args:
        surface_pressure (np.ndarray): (time, latitude, longitude) or (latitude, longitude) grid in units of Pa.
        model_a_full (np.ndarray): a coefficients at each model level being used in units of Pa.
        model_b_full (np.ndarray): b coefficients at each model level being used (unitless).

    Returns:
        pressure_3d: 3D pressure field with dimensions of surface_pressure and number of levels from model_a and model_b.

    """
    assert (
        model_a_full.size == model_b_full.size
    ), "Model pressure coefficient arrays do not match."
    model_a_half_mid = np.sqrt(model_a_full[1:] * model_a_full[:-1])
    model_a_half = np.zeros(model_a_half_mid.size + 2)
    model_a_half[1:-1] = model_a_half_mid
    model_b_half_mid = np.sqrt(model_b_full[1:] * model_b_full[:-1])
    model_b_half = np.zeros(model_b_half_mid.size + 2)
    model_b_half[1:-1] = model_b_half_mid
    model_b_half[-1] = 1.0
    model_a_half_3d = model_a_half.reshape(-1, 1, 1)
    model_b_half_3d = model_b_half.reshape(-1, 1, 1)
    model_a_full_3d = model_a_full.reshape(-1, 1, 1)
    model_b_full_3d = model_b_full.reshape(-1, 1, 1)
    if surface_pressure.ndim == 3:
        # Generate the 3D pressure field for a time series of surface pressure grids
        pressure_3d = np.zeros(
            (
                surface_pressure.shape[0],
                model_a_full.shape[0],
                surface_pressure.shape[1],
                surface_pressure.shape[2],
            ),
            dtype=surface_pressure.dtype,
        )
        pressure_3d_half = np.zeros(
            (
                surface_pressure.shape[0],
                model_a_full.shape[0] + 1,
                surface_pressure.shape[1],
                surface_pressure.shape[2],
            ),
            dtype=surface_pressure.dtype,
        )

        for i in range(surface_pressure.shape[0]):
            pressure_3d_half[i] = (
                model_a_half_3d + model_b_half_3d * surface_pressure[i]
            )
            pressure_3d[i] = model_a_full_3d + model_b_full_3d * surface_pressure[i]
    else:
        pressure_3d_half = model_a_half_3d + model_b_half_3d * surface_pressure
        pressure_3d = model_a_full_3d + model_b_full_3d * surface_pressure
    return pressure_3d, pressure_3d_half


@njit
def geopotential_from_model_vars(
    surface_geopotential,
    surface_pressure,
    temperature,
    specific_humidity,
    half_pressure,
):
    """Calculate geopotential from model level data.

    Calculate geopotential from the base state variables. Geopotential height is calculated by adding thicknesses
    calculated within each half-model-level to account for variations in temperature and moisture between grid cells.
    Note that this function is calculating geopotential in units of (m^2 s^-2) not geopential height.

    To convert geopotential to geopotential height, divide geopotential by g (9.806 m s^-2).

    Geopotential height is defined as the height above mean sea level. To get height above ground level, substract
    the surface geoptential height field from the 3D geopotential height field.

    Args:
        surface_geopotential (np.ndarray): Surface geopotential in shape (y,x) and units m^2 s^-2.
        surface_pressure (np.ndarray): Surface pressure in shape (y, x) and units Pa
        temperature (np.ndarray): temperature in shape (levels, y, x) and units K
        specific_humidity (np.ndarray): mixing ratio in shape (levels, y, x) and units kg/kg.

    Returns:
        model_geoptential (np.ndarray): geopotential on model levels in shape (levels, y, x)

    """
    RDGAS = 287.06
    gamma = 0.609133  # from MetView
    model_geopotential = np.zeros(
        (half_pressure.shape[0] - 1, half_pressure.shape[1], half_pressure.shape[2]),
        dtype=surface_pressure.dtype,
    )
    half_geopotential = np.zeros(half_pressure.shape, dtype=surface_pressure.dtype)
    half_geopotential[-1] = surface_geopotential
    virtual_temperature = temperature * (1.0 + gamma * specific_humidity)
    m = model_geopotential.shape[-3] - 1
    for i in range(0, model_geopotential.shape[-3]):
        if m == 0:
            dlog_p = np.log(half_pressure[m + 1] / 0.1)
            alpha = np.ones(half_pressure[m + 1].shape) * np.log(2)
        else:
            dlog_p = np.log(half_pressure[m + 1] / half_pressure[m])
            alpha = 1.0 - (
                (half_pressure[m] / (half_pressure[m + 1] - half_pressure[m])) * dlog_p
            )
        model_geopotential[m] = (
            half_geopotential[m + 1] + RDGAS * virtual_temperature[m] * alpha
        )
        half_geopotential[m] = (
            half_geopotential[m + 1] + RDGAS * virtual_temperature[m] * dlog_p
        )
        m -= 1
    return model_geopotential


@njit
def interp_hybrid_to_pressure_levels(
    model_var, model_pressure, interp_pressures, use_log=True
):
    """Interpolate to pressure levels.

    Interpolate data field from hybrid sigma-pressure vertical coordinates to pressure levels.
    `model_pressure` and `interp_pressure` should have consistent units with each other.

    Args:
        model_var (np.ndarray): 3D field on hybrid sigma-pressure levels with shape (levels, y, x).
        model_pressure (np.ndarray): 3D pressure field with shape (levels, y, x) in units Pa or hPa
        interp_pressures: (np.ndarray): pressure levels for interpolation in units Pa or hPa.
        use_log (bool): If True, use the natural logarithm of the pressure as the interpolation coordinate.
            Otherwise, use the pressure.

    Returns:
        pressure_var (np.ndarray): 3D field on pressure levels with shape (len(interp_pressures), y, x).

    """
    pressure_var = np.zeros(
        (interp_pressures.shape[0], model_var.shape[1], model_var.shape[2]),
        dtype=model_var.dtype,
    )
    if use_log:
        interp_pres_coord = np.log(interp_pressures)
    else:
        interp_pres_coord = interp_pressures
    for (i, j), v in np.ndenumerate(model_var[0]):
        if use_log:
            pres_coord = np.log(model_pressure[:, i, j])
        else:
            pres_coord = model_pressure[:, i, j]
        pressure_var[:, i, j] = np.interp(
            interp_pres_coord, pres_coord, model_var[:, i, j]
        )
    return pressure_var


@njit
def interp_pressure_to_hybrid_levels(
    pressure_var, pressure_levels, model_pressure, surface_pressure
):
    """Interpolate fields on pressure levels to hybrid levels.

    Interpolate data field from hybrid sigma-pressure vertical coordinates to pressure levels.
    `model_pressure` and `pressure_levels` and 'surface_pressure' should have consistent units with each other.

    Args:
        pressure_var (np.ndarray): 3D field on pressure levels with shape (levels, y, x).
        pressure_levels (np.double): pressure levels for interpolation in units Pa or hPa.
        model_pressure (np.ndarray): 3D pressure field with shape (levels, y, x) in units Pa or hPa
        surface_pressure (np.ndarray): pressure at the surface in units Pa or hPa.

    Returns:
        model_var (np.ndarray): 3D field on hybrid sigma-pressure levels with shape (model_pressure.shape[0], y, x).

    """
    model_var = np.zeros(model_pressure.shape, dtype=model_pressure.dtype)
    log_interp_pressures = np.log(pressure_levels)
    for (i, j), v in np.ndenumerate(model_var[0]):
        air_levels = np.where(pressure_levels < surface_pressure[i, j])[0]
        model_var[:, i, j] = np.interp(
            np.log(model_pressure[:, i, j]),
            log_interp_pressures[air_levels],
            pressure_var[air_levels, i, j],
        )
    return pressure_var


@njit
def interp_hybrid_to_hybrid_levels(hybrid_var, hybrid_pressure, target_pressure):
    """
    Interpolate fields on hybrid levels to hybrid levels via pressure.

    Interpolate data from hybrid sigma-pressure vertical coordinates to other hybrid levels.

    Args:
        hybrid_var (np.ndarray): 3D field on hybrid sigma-pressure levels with shape (levels, y, x).
        hybrid_pressure (np.double): pressure levels for interpolation in units Pa or hPa.
        target_pressure (np.ndarray): 3D target pressure fields with shape (levels, y, x) in units Pa or hPa

    Returns:
        model_var (np.ndarray): 3D field on hybrid sigma-pressure levels with shape (target_pressure.shape[0], y, x).

    """
    model_var = np.zeros(target_pressure.shape, dtype=target_pressure.dtype)
    for (i, j), v in np.ndenumerate(hybrid_var[0]):
        model_var[:, i, j] = np.interp(
            np.log(target_pressure[:, i, j]),
            np.log(hybrid_pressure[:, i, j]),
            hybrid_var[:, i, j],
        )
    return model_var


@njit
def interp_geopotential_to_pressure_levels(
    geopotential,
    model_pressure,
    interp_pressures,
    surface_pressure,
    surface_geopotential,
    temperature_k,
    temp_height=150,
):
    """Interpolate geopotential field to pressure levels.

    Interpolate geopotential field from hybrid sigma-pressure vertical coordinates to pressure levels.
    `model_pressure` and `interp_pressure` should have consistent units of hPa or Pa. Geopotential height is extrapolated
    below the surface based on Eq. 15 in Trenberth et al. (1993).

    Args:
        geopotential (np.ndarray): geopotential in units m^2/s^2.
        model_pressure (np.ndarray): 3D pressure field with shape (levels, y, x) in units Pa or hPa
        interp_pressures (np.ndarray): pressure levels for interpolation in units Pa or hPa.
        surface_pressure (np.ndarray): pressure at the surface in units Pa or hPa.
        surface_geopotential (np.ndarray): geopotential at the surface in units m^2/s^2.
        temperature_k (np.ndarray): temperature  in units K.
        temp_height (float): height above ground of nearest vertical grid cell.

    Returns:
        pressure_var (np.ndarray): 3D field on pressure levels with shape (len(interp_pressures), y, x).

    """
    LAPSE_RATE = 0.0065  # K / m
    ALPHA = LAPSE_RATE * RDGAS / GRAVITY
    pressure_var = np.zeros(
        (interp_pressures.shape[0], geopotential.shape[1], geopotential.shape[2]),
        dtype=geopotential.dtype,
    )
    log_interp_pressures = np.log(interp_pressures)
    for (i, j), v in np.ndenumerate(geopotential[0]):
        pressure_var[:, i, j] = np.interp(
            log_interp_pressures, np.log(model_pressure[:, i, j]), geopotential[:, i, j]
        )
        for pl, interp_pressure in enumerate(interp_pressures):
            if interp_pressure > surface_pressure[i, j]:
                height_agl = (
                    geopotential[:, i, j] - surface_geopotential[i, j]
                ) / GRAVITY
                h = np.argmin(np.abs(height_agl - temp_height))
                temp_surface_k = temperature_k[h, i, j] + ALPHA * temperature_k[
                    h, i, j
                ] * (surface_pressure[i, j] / model_pressure[h, i, j] - 1)
                surface_height = surface_geopotential[i, j] / GRAVITY
                temp_sea_level_k = temp_surface_k + LAPSE_RATE * surface_height
                temp_pl = np.minimum(temp_sea_level_k, 298.0)
                if surface_height > 2500.0:
                    gamma = (
                        GRAVITY
                        / surface_geopotential[i, j]
                        * np.maximum(temp_pl - temp_surface_k, 0)
                    )

                elif 2000.0 <= surface_height <= 2500.0:
                    t_adjusted = 0.002 * (
                        (2500 - surface_height) * temp_sea_level_k
                        + (surface_height - 2000.0) * temp_pl
                    )
                    gamma = (
                        GRAVITY
                        / surface_geopotential[i, j]
                        * (t_adjusted - temp_surface_k)
                    )
                else:
                    gamma = LAPSE_RATE
                a_ln_p = (
                    gamma
                    * RDGAS
                    / GRAVITY
                    * np.log(interp_pressure / surface_pressure[i, j])
                )
                ln_p_ps = np.log(interp_pressure / surface_pressure[i, j])
                pressure_var[pl, i, j] = surface_geopotential[
                    i, j
                ] - RDGAS * temp_surface_k * ln_p_ps * (
                    1 + a_ln_p / 2.0 + a_ln_p**2 / 6.0
                )
    return pressure_var


@njit
def interp_temperature_to_pressure_levels(
    model_var,
    model_pressure,
    interp_pressures,
    surface_pressure,
    surface_geopotential,
    geopotential,
    temp_height=150,
):
    """Interpolate the temperature field to pressure levels.

    Interpolate temperature field from hybrid sigma-pressure vertical coordinates to pressure levels.
    `model_pressure` and `interp_pressure` should have consistent units of hPa or Pa. Temperature is extrapolated
    below the surface based on Eq. 16 in Trenberth et al. (1993).

    Args:
        model_var (np.ndarray): 3D field on hybrid sigma-pressure levels with shape (levels, y, x).
        model_pressure (np.ndarray): 3D pressure field with shape (levels, y, x) in units Pa
        interp_pressures: (np.ndarray): pressure levels for interpolation in units Pa or.
        surface_pressure (np.ndarray): pressure at the surface in units Pa or hPa.
        surface_geopotential (np.ndarray): geopotential at the surface in units m^2/s^2.
        temp_height (float): height above ground of nearest vertical grid cell.

    Returns:
        pressure_var (np.ndarray): 3D field on pressure levels with shape (len(interp_pressures), y, x).

    """
    LAPSE_RATE = 0.0065  # K / m
    ALPHA = LAPSE_RATE * RDGAS / GRAVITY
    pressure_var = np.zeros(
        (interp_pressures.shape[0], model_var.shape[1], model_var.shape[2]),
        dtype=model_var.dtype,
    )
    log_interp_pressures = np.log(interp_pressures)
    for (i, j), v in np.ndenumerate(model_var[0]):
        pressure_var[:, i, j] = np.interp(
            log_interp_pressures, np.log(model_pressure[:, i, j]), model_var[:, i, j]
        )
        for pl, interp_pressure in enumerate(interp_pressures):
            if interp_pressure > surface_pressure[i, j]:
                # The height above ground of each sigma level varies, especially in complex terrain
                # To minimize extrapolation error, pick the level closest to 150 m AGL, which is the ECMWF standard.
                height_agl = (
                    geopotential[:, i, j] - surface_geopotential[i, j]
                ) / GRAVITY
                h = np.argmin(np.abs(height_agl - temp_height))
                temp_surface_k = model_var[h, i, j] + ALPHA * model_var[h, i, j] * (
                    surface_pressure[i, j] / model_pressure[h, i, j] - 1
                )
                surface_height = surface_geopotential[i, j] / GRAVITY
                temp_sea_level_k = temp_surface_k + LAPSE_RATE * surface_height
                temp_pl = np.minimum(temp_sea_level_k, 298.0)
                if surface_height > 2500.0:
                    gamma = (
                        GRAVITY
                        / surface_geopotential[i, j]
                        * np.maximum(temp_pl - temp_surface_k, 0)
                    )

                elif 2000.0 <= surface_height <= 2500.0:
                    t_adjusted = 0.002 * (
                        (2500 - surface_height) * temp_sea_level_k
                        + (surface_height - 2000.0) * temp_pl
                    )
                    gamma = (
                        GRAVITY
                        / surface_geopotential[i, j]
                        * (t_adjusted - temp_surface_k)
                    )
                else:
                    gamma = LAPSE_RATE
                a_ln_p = (
                    gamma
                    * RDGAS
                    / GRAVITY
                    * np.log(interp_pressure / surface_pressure[i, j])
                )
                pressure_var[pl, i, j] = temp_surface_k * (
                    1 + a_ln_p + 0.5 * a_ln_p**2 + 1 / 6.0 * a_ln_p**3
                )
    return pressure_var


@njit
def interp_hybrid_to_height_agl(
    model_var: np.ndarray,
    interp_heights_m: np.ndarray,
    geopotential: np.ndarray,
    surface_geopotential: np.ndarray,
):
    """Interpolate data on hybrid sigma-pressure levels to heights above ground level in meters.

    Args:
        model_var (np.ndarray): State variable of shape [levels, lat, lon]
        interp_heights_m (np.ndarray): 1D array of height levels in meters above ground level.
        geopotential (np.ndarray): geopotential on model levels in units of m^2/s^2.
        surface_geopotential: geopotential at the surface in units of m^2/s^2.

    Returns:
        height_var (np.ndarray): State variable on height above ground levels in shape [interp_heights, lat, lon].

    """
    model_height_agl = (geopotential - surface_geopotential) / GRAVITY
    height_var = np.zeros(
        (interp_heights_m.shape[0], model_var.shape[1], model_var.shape[2]),
        dtype=model_var.dtype,
    )
    for (i, j), v in np.ndenumerate(model_var[0]):
        height_var[:, i, j] = np.interp(
            interp_heights_m, model_height_agl[::-1, i, j], model_var[::-1, i, j]
        )
    return height_var


@njit
def mean_sea_level_pressure(
    surface_pressure_pa,
    temperature_k,
    pressure_pa,
    surface_geopotential,
    geopotential,
    temp_height=150.0,
):
    """Calculate the mean sea level pressure.

    Calculate mean sea level pressure from surface pressure, lowest model level temperature,
    the pressure of the lowest model level (derived from create_pressure_grid), and surface_geopotential.
    This calculation is based on the procedure from Trenberth et al. (1993) implemented in CESM CAM.

    Trenberth, K., J. Berry , and L. Buja, 1993: Vertical Interpolation and Truncation of Model-Coordinate,
    University Corporation for Atmospheric Research, https://doi.org/10.5065/D6HX19NH.

    CAM implementation: https://github.com/ESCOMP/CAM/blob/cam_cesm2_2_rel/src/physics/cam/cpslec.F90

    Args:
        surface_pressure_pa: surface pressure in Pascals
        temperature_k: Temperature at the lowest model level in Kelvin.
        pressure_pa: Pressure at the lowest model level in Pascals.
        surface_geopotential: Geopotential of the surface in m^2 s^-2.
        geopotential: Geopotential at all levels.
        temp_height: height of nearest vertical grid cell

    Returns:
        mslp: Mean sea level pressure in Pascals.

    """
    LAPSE_RATE = 0.0065  # K / m
    ALPHA = LAPSE_RATE * RDGAS / GRAVITY
    mslp = np.zeros(surface_pressure_pa.shape, dtype=surface_pressure_pa.dtype)
    for (i, j), p in np.ndenumerate(mslp):
        if np.abs(surface_geopotential[i, j] / GRAVITY) < 1e-4:
            mslp[i, j] = surface_pressure_pa[i, j]
        else:
            height_agl = (geopotential[:, i, j] - surface_geopotential[i, j]) / GRAVITY
            h = np.argmin(np.abs(height_agl - temp_height))
            temp_surface_k = temperature_k[h, i, j] + ALPHA * temperature_k[h, i, j] * (
                surface_pressure_pa[i, j] / pressure_pa[h, i, j] - 1
            )
            temp_sealevel_k = (
                temp_surface_k + LAPSE_RATE * surface_geopotential[i, j] / GRAVITY
            )

            if (temp_surface_k <= 290.5) and (temp_sealevel_k > 290.5):
                gamma = GRAVITY / surface_geopotential[i, j] * (290.5 - temp_surface_k)
            elif (temp_surface_k > 290.5) and (temp_sealevel_k > 290.5):
                gamma = 0.0
                temp_surface_k = 0.5 * (290.5 + temp_surface_k)
            else:
                gamma = LAPSE_RATE
                if temp_surface_k < 255:
                    temp_surface_k = 0.5 * (255.0 + temp_surface_k)
            beta = surface_geopotential[i, j] / (RDGAS * temp_surface_k)
            x = gamma * surface_geopotential[i, j] / (GRAVITY * temp_surface_k)
            mslp[i, j] = surface_pressure_pa[i, j] * np.exp(
                beta * (1.0 - x / 2.0 + x**2 / 3.0)
            )
    return mslp
