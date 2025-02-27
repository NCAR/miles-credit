import numpy as np
from numba import njit
import xarray as xr
from tqdm import tqdm
from .physics_constants import RDGAS, GRAVITY
import os


def full_state_pressure_interpolation(
    state_dataset: xr.Dataset,
    surface_geopotential: np.ndarray,
    pressure_levels: np.ndarray = np.array([500.0, 850.0]),
    interp_fields: tuple[str] = ("U", "V", "T", "Q"),
    pres_ending: str = "_PRES",
    temperature_var: str = "T",
    q_var: str = "Q",
    surface_pressure_var: str = "SP",
    geopotential_var: str = "Z",
    time_var: str = "time",
    lat_var: str = "latitude",
    lon_var: str = "longitude",
    pres_var: str = "pressure",
    level_var: str = "level",
    model_level_file: str = "../credit/metadata/ERA5_Lev_Info.nc",
    verbose: int = 1,
    a_coord: str = "a_model",
    b_coord: str = "b_model",
    a_half: str = "a_half",
    b_half: str = "b_half",
    temp_level_index: int = -2,
) -> xr.Dataset:
    """
    Interpolate full model state variables from model levels to pressure levels.

    Args:
        state_dataset (xr.Dataset): state variables being interpolated
        surface_geopotential (np.ndarray): surface geopotential levels in units m^2/s^2.
        pressure_levels (np.ndarray): pressure levels for interpolation in hPa.
        interp_fields (tuple[str]): fields to be interpolated.
        pres_ending (str): ending string to attach to pressure interpolated variables.
        temperature_var (str): temperature variable to be interpolated (units K).
        q_var (str): mixing ratio/specific humidity variable to be interpolated (units kg/kg).
        surface_pressure_var (str): surface pressure variable (units Pa).
        geopotential_var (str): geopotential variable being derived (units m^2/s^2).
        time_var (str): time coordinate
        lat_var (str): latitude coordinate
        lon_var (str): longitude coordinate
        pres_var (str): pressure coordinate
        level_var (str): name of level coordinate
        model_level_file (str): relative path to file containing model levels.
        verbose (int): verbosity level. If verbose > 0, print progress.
        a_coord (str): Name of A weight in sigma coordinate formula. 'a_model' by default.
        b_coord (str): Name of B weight in sigma coordinate formula. 'b_model' by default.
        a_half (str): Name of A weight in sigma coordinate formula at half levels. 'a_half' by default.
        b_half (str): Name of B weight in sigma coordinate formula at half levels. 'b_half' by default.
        temp_level_index (int): vertical index of the temperature level used for interpolation below ground.
    Returns:
        pressure_ds (xr.Dataset): Dataset containing pressure interpolated variables.
    """
    path_to_file = os.path.abspath(os.path.dirname(__file__))
    model_level_file = os.path.join(path_to_file, model_level_file)
    with xr.open_dataset(model_level_file) as mod_lev_ds:
        a_half = mod_lev_ds[a_half].values
        b_half = mod_lev_ds[b_half].values
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
    pressure_ds[geopotential_var + pres_ending] = xr.DataArray(
        coords=coords, dims=pres_dims, name=geopotential_var + pres_ending
    )
    pressure_ds["mean_sea_level_" + pres_var] = xr.DataArray(
        coords=coords_surface, dims=surface_dims, name="mean_sea_level_" + pres_var
    )
    disable = False
    if verbose == 0:
        disable = True
    sub_half_levels = np.concatenate([state_dataset[level_var].values, [138]])
    sub_levels = state_dataset[level_var].values
    for t, time in tqdm(enumerate(state_dataset[time_var]), disable=disable):
        pressure_grid, half_pressure_grid = create_pressure_grid(
            state_dataset[surface_pressure_var][t].values.astype(np.float64), a_half, b_half
        )
        pressure_sub_grid = pressure_grid[sub_levels - 1]
        half_pressure_sub_grid = half_pressure_grid[sub_half_levels - 1]
        geopotential_grid = geopotential_from_model_vars(
            surface_geopotential.astype(np.float64),
            state_dataset[surface_pressure_var][t].values.astype(np.float64),
            state_dataset[temperature_var][t].values.astype(np.float64),
            state_dataset[q_var][t].values.astype(np.float64),
            half_pressure_sub_grid,
        )
        for interp_field in interp_fields:
            if interp_field == temperature_var:
                pressure_ds[interp_field + pres_ending][t] = interp_temperature_to_pressure_levels(
                    state_dataset[interp_field][t].values,
                    pressure_sub_grid / 100.0,
                    pressure_levels,
                    state_dataset[surface_pressure_var][t].values / 100.0,
                    surface_geopotential,
                    geopotential_grid,
                )
            else:
                pressure_ds[interp_field + pres_ending][t] = interp_hybrid_to_pressure_levels(
                    state_dataset[interp_field][t].values,
                    pressure_sub_grid / 100.0,
                    pressure_levels,
                )
        pressure_ds[geopotential_var + pres_ending][t] = interp_geopotential_to_pressure_levels(
            geopotential_grid,
            pressure_sub_grid / 100.0,
            pressure_levels,
            state_dataset[surface_pressure_var][t].values / 100.0,
            surface_geopotential,
            state_dataset[temperature_var][t].values,
        )
        pressure_ds["mean_sea_level_" + pres_var][t] = mean_sea_level_pressure(
            state_dataset[surface_pressure_var][t].values,
            state_dataset[temperature_var][t].values,
            pressure_sub_grid,
            surface_geopotential,
            geopotential_grid,
        )
    return pressure_ds


@njit
def create_pressure_grid(surface_pressure, model_a_half, model_b_half):
    """
    Create a 3D pressure field at model levels from the surface pressure field and the hybrid sigma-pressure
    coefficients from ECMWF. Conversion is `pressure_3d = a + b * SP`.

    Args:
        surface_pressure (np.ndarray): (time, latitude, longitude) or (latitude, longitude) grid in units of Pa.
        model_a_half (np.ndarray): a coefficients at each model level being used in units of Pa.
        model_b_half (np.ndarray): b coefficients at each model level being used (unitness).

    Returns:
        pressure_3d: 3D pressure field with dimensions of surface_pressure and number of levels from model_a and model_b.
    """
    assert model_a_half.size == model_b_half.size, "Model pressure coefficient arrays do not match."
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
            pressure_3d_half = model_a_3d + model_b_3d * surface_pressure[i]
            pressure_3d[i] = 0.5 * (pressure_3d_half[:-1] + pressure_3d_half[1:])
    else:
        # Generate the 3D pressure field for a single surface pressure grid.
        model_a_3d = model_a_half.reshape(-1, 1, 1)
        model_b_3d = model_b_half.reshape(-1, 1, 1)
        pressure_3d_half = model_a_3d + model_b_3d * surface_pressure
        pressure_3d = 0.5 * (pressure_3d_half[:-1] + pressure_3d_half[1:])
    return pressure_3d, pressure_3d_half


@njit
def geopotential_from_model_vars(surface_geopotential, surface_pressure, temperature, mixing_ratio, half_pressure):
    """
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
        mixing_ratio (np.ndarray): mixing ratio in shape (levels, y, x) and units kg/kg.
        a_half (np.ndarray): a coefficients at each model half level being used in units of Pa.
        b_half (np.ndarray): b coefficients at each model half level being used (unitness).

    Returns:
        model_geoptential (np.ndarray): geopotential on model levels in shape (levels, y, x)
    """
    RDGAS = 287.06
    gamma = 0.609133  # from MetView
    model_geopotential = np.zeros(
        (half_pressure.shape[0] - 1, half_pressure.shape[1], half_pressure.shape[2]), dtype=surface_pressure.dtype
    )
    half_geopotential = np.zeros(half_pressure.shape, dtype=surface_pressure.dtype)
    half_geopotential[-1] = surface_geopotential
    virtual_temperature = temperature * (1.0 + gamma * mixing_ratio)
    m = model_geopotential.shape[-3] - 1
    for i in range(0, model_geopotential.shape[-3]):
        if m == 0:
            dlog_p = np.log(half_pressure[m + 1] / 0.1)
            alpha = np.ones(half_pressure[m + 1].shape) * np.log(2)
        else:
            dlog_p = np.log(half_pressure[m + 1] / half_pressure[m])
            alpha = 1.0 - ((half_pressure[m] / (half_pressure[m + 1] - half_pressure[m])) * dlog_p)
        model_geopotential[m] = half_geopotential[m + 1] + RDGAS * virtual_temperature[m] * alpha
        half_geopotential[m] = half_geopotential[m + 1] + RDGAS * virtual_temperature[m] * dlog_p
        m -= 1
    return model_geopotential


@njit
def interp_hybrid_to_pressure_levels(model_var, model_pressure, interp_pressures):
    """
    Interpolate data field from hybrid sigma-pressure vertical coordinates to pressure levels.
    `model_pressure` and `interp_pressure` should have consistent units with each other.

    Args:
        model_var (np.ndarray): 3D field on hybrid sigma-pressure levels with shape (levels, y, x).
        model_pressure (np.ndarray): 3D pressure field with shape (levels, y, x) in units Pa or hPa
        interp_pressures: (np.ndarray): pressure levels for interpolation in units Pa or hPa.

    Returns:
        pressure_var (np.ndarray): 3D field on pressure levels with shape (len(interp_pressures), y, x).
    """
    pressure_var = np.zeros(
        (interp_pressures.shape[0], model_var.shape[1], model_var.shape[2]),
        dtype=model_var.dtype,
    )
    log_interp_pressures = np.log(interp_pressures)
    for (i, j), v in np.ndenumerate(model_var[0]):
        pressure_var[:, i, j] = np.interp(log_interp_pressures, np.log(model_pressure[:, i, j]), model_var[:, i, j])
    return pressure_var


@njit
def interp_pressure_to_hybrid_levels(pressure_var, pressure_levels, model_pressure, surface_pressure):
    """
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
def interp_geopotential_to_pressure_levels(
    geopotential,
    model_pressure,
    interp_pressures,
    surface_pressure,
    surface_geopotential,
    temperature_k,
    temp_height=150,
):
    """
    Interpolate geopotential field from hybrid sigma-pressure vertical coordinates to pressure levels.
    `model_pressure` and `interp_pressure` should have consistent units of hPa or Pa. Geopotential height is extrapolated
    below the surface based on Eq. 15 in Trenberth et al. (1993).

    Args:
        geopotential (np.ndarray): geopotential in units m^2/s^2.
        model_pressure (np.ndarray): 3D pressure field with shape (levels, y, x) in units Pa or hPa
        interp_pressures (np.ndarray): pressure levels for interpolation in units Pa or hPa.
        surface_pressure (np.ndarray): pressure at the surface in units Pa or hPa.
        surface_geopotential (np.ndarray): geopotential at the surface in units m^2/s^2.
        temperaure_k (np.ndarray): temperature  in units K.
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
        pressure_var[:, i, j] = np.interp(log_interp_pressures, np.log(model_pressure[:, i, j]), geopotential[:, i, j])
        for pl, interp_pressure in enumerate(interp_pressures):
            if interp_pressure > surface_pressure[i, j]:
                height_agl = (geopotential[:, i, j] - surface_geopotential[i, j]) / GRAVITY
                h = np.argmin(np.abs(height_agl - temp_height))
                temp_surface_k = temperature_k[h, i, j] + ALPHA * temperature_k[h, i, j] * (
                    surface_pressure[i, j] / model_pressure[h, i, j] - 1
                )
                surface_height = surface_geopotential[i, j] / GRAVITY
                temp_sea_level_k = temp_surface_k + LAPSE_RATE * surface_height
                temp_pl = np.minimum(temp_sea_level_k, 298.0)
                if surface_height > 2500.0:
                    gamma = GRAVITY / surface_geopotential[i, j] * np.maximum(temp_pl - temp_surface_k, 0)

                elif 2000.0 <= surface_height <= 2500.0:
                    t_adjusted = 0.002 * (
                        (2500 - surface_height) * temp_sea_level_k + (surface_height - 2000.0) * temp_pl
                    )
                    gamma = GRAVITY / surface_geopotential[i, j] * (t_adjusted - temp_surface_k)
                else:
                    gamma = LAPSE_RATE
                a_ln_p = gamma * RDGAS / GRAVITY * np.log(interp_pressure / surface_pressure[i, j])
                ln_p_ps = np.log(interp_pressure / surface_pressure[i, j])
                pressure_var[pl, i, j] = surface_geopotential[i, j] - RDGAS * temp_surface_k * ln_p_ps * (
                    1 + a_ln_p / 2.0 + a_ln_p**2 / 6.0
                )
    return pressure_var


@njit
def interp_temperature_to_pressure_levels(
    model_var, model_pressure, interp_pressures, surface_pressure, surface_geopotential, geopotential, temp_height=150
):
    """
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
        pressure_var[:, i, j] = np.interp(log_interp_pressures, np.log(model_pressure[:, i, j]), model_var[:, i, j])
        for pl, interp_pressure in enumerate(interp_pressures):
            if interp_pressure > surface_pressure[i, j]:
                # The height above ground of each sigma level varies, especially in complex terrain
                # To minimize extrapolation error, pick the level closest to 150 m AGL, which is the ECMWF standard.
                height_agl = (geopotential[:, i, j] - surface_geopotential[i, j]) / GRAVITY
                h = np.argmin(np.abs(height_agl - temp_height))
                temp_surface_k = model_var[h, i, j] + ALPHA * model_var[h, i, j] * (
                    surface_pressure[i, j] / model_pressure[h, i, j] - 1
                )
                surface_height = surface_geopotential[i, j] / GRAVITY
                temp_sea_level_k = temp_surface_k + LAPSE_RATE * surface_height
                temp_pl = np.minimum(temp_sea_level_k, 298.0)
                if surface_height > 2500.0:
                    gamma = GRAVITY / surface_geopotential[i, j] * np.maximum(temp_pl - temp_surface_k, 0)

                elif 2000.0 <= surface_height <= 2500.0:
                    t_adjusted = 0.002 * (
                        (2500 - surface_height) * temp_sea_level_k + (surface_height - 2000.0) * temp_pl
                    )
                    gamma = GRAVITY / surface_geopotential[i, j] * (t_adjusted - temp_surface_k)
                else:
                    gamma = LAPSE_RATE
                a_ln_p = gamma * RDGAS / GRAVITY * np.log(interp_pressure / surface_pressure[i, j])
                pressure_var[pl, i, j] = temp_surface_k * (1 + a_ln_p + 0.5 * a_ln_p**2 + 1 / 6.0 * a_ln_p**3)
    return pressure_var


@njit
def mean_sea_level_pressure(
    surface_pressure_pa, temperature_k, pressure_pa, surface_geopotential, geopotential, temp_height=150.0
):
    """
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
            temp_sealevel_k = temp_surface_k + LAPSE_RATE * surface_geopotential[i, j] / GRAVITY

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
            mslp[i, j] = surface_pressure_pa[i, j] * np.exp(beta * (1.0 - x / 2.0 + x**2 / 3.0))
    return mslp
