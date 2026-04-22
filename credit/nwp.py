import numpy as np
import pandas as pd
from os.path import join, exists, expandvars
import xarray as xr
from functools import partial
from multiprocessing import Pool
from importlib.resources import files
from credit.interp import (
    geopotential_from_model_vars,
    create_pressure_grid,
    create_reduced_pressure_grid,
    interp_hybrid_to_hybrid_levels,
    interp_hybrid_to_pressure_levels,
)
from credit.physics_constants import GRAVITY
import datetime
import time
import traceback
import yaml

gfs_map = {}
level_map = {}
upper_air = {}
surface = {}


def build_GFS_init(
    output_grid: xr.Dataset,
    date: pd.Timestamp,
    variables: list,
    model_level_file: str = "",
    model_levels: np.ndarray = None,
    gdas_base_path: str = "gs://global-forecast-system/",
    variable_mapping: str = "wchapmanera5",
    n_procs: int = 1,
):
    """
    Download GFS model level initial conditions, regrid to CREDIT model's grid, and vertically interpolate
    3D variables to CREDIT model levels.

    Args:
        output_grid (xr.Dataset): netCDF file containing latitude and longitude coordinates of CREDIT model grid.
        date (pd.Timestamp): date of GFS initialization
        variables (list): list of variable names.
        model_level_file (str): Path to file containing output model level a and b coefficients on full and half levels.
        model_levels (np.ndarray): Subset of model levels to interpolate to.
        gdas_base_path (str): Path to GFS base directory on NOMADS (archives last 10 days) or Google Cloud (since 2021)
        variable_mapping (str):
        n_procs (int): Number of processors to use in pool.

    Returns:
        (xr.Dataset) Interpolated GFS initial conditions
    """

    required_variables = [
        "pressfc",
        "tmp",
        "spfh",
        "hgtsfc",
    ]  # required for calculating pressure and geopotential
    _get_gfs_maps(variable_mapping)
    gfs_variables = list(set([k for k, v in gfs_map.items() if v in variables]).union(required_variables))

    pool = Pool(n_procs)
    atm_full_path = _build_file_path(date, gdas_base_path, file_type="atm")
    sfc_full_path = _build_file_path(date, gdas_base_path, file_type="sfc")
    print("Download GFS atmospheric data")
    start = time.perf_counter()
    gfs_atm_data = _load_gfs_data(atm_full_path, gfs_variables, pool=pool)
    end = time.perf_counter()
    dur = end - start
    print(f"Elapsed: {dur:0.6f}")
    print("Download GFS surface data")
    start = time.perf_counter()
    gfs_sfc_data = _load_gfs_data(sfc_full_path, gfs_variables, pool=pool)
    end = time.perf_counter()
    dur = end - start
    print(f"Elapsed: {dur:0.6f}")
    gfs_data = _combine_data(gfs_atm_data, gfs_sfc_data, gfs_map["tmp"], gfs_map["spfh"])
    print("Regrid data")
    start = time.perf_counter()
    regridded_gfs = _regrid(gfs_data, output_grid, pool=pool)
    end = time.perf_counter()
    dur = end - start
    print(f"Elapsed: {dur:0.6f}")
    print("Interpolate to model levels")
    start = time.perf_counter()
    # interpolated_gfs = _interpolate_to_model_level(regridded_gfs, output_grid, model_level_indices, variables)
    interpolated_gfs = _vertical_interpolation(
        regridded_gfs, model_level_file, model_levels, variable_mapping, variables, gfs_map["pressfc"]
    )
    end = time.perf_counter()
    dur = end - start
    print(f"Elapsed: {dur:0.6f}")
    final_data = _format_data(interpolated_gfs, regridded_gfs, model_levels)

    return final_data


def _get_gfs_maps(variable_mapping_type: str):
    meta_path = str(files("credit.metadata"))
    mapping_file = join(meta_path, f"gfs_to_{variable_mapping_type}.yml")
    global gfs_map, level_map, upper_air, surface
    if not exists(mapping_file):
        raise FileNotFoundError(f"{variable_mapping_type} does not have a valid GFS mapping file in credit.metadata.")
    with open(mapping_file) as mapping_obj:
        mapping = yaml.safe_load(mapping_obj)
    gfs_map = mapping["gfs_map"]
    level_map = mapping["level_map"]
    upper_air = mapping["upper_air"]
    surface = mapping["surface"]
    return


def _add_pressure_and_geopotential(data, temperature_var, specific_humidity_var):
    """
    Derive pressure and geopotential fields from model level data and to dataset
    Args:
        data: (xr.Dataset) GFS model level data

    Returns:
        xr.Dataset
    """
    sfc_pressure = data["SP"].values.squeeze()
    sfc_gpt = data["hgtsfc"].values.squeeze() * GRAVITY
    level_T = data[temperature_var].values.squeeze()
    level_Q = data[specific_humidity_var].values.squeeze()
    a_coeff = data.attrs["ak"]
    b_coeff = data.attrs["bk"]

    full_prs_grid, half_prs_grid = create_pressure_grid(sfc_pressure, a_coeff, b_coeff)
    geopotential = geopotential_from_model_vars(
        sfc_gpt.astype(np.float64),
        sfc_pressure.astype(np.float64),
        level_T.astype(np.float64),
        level_Q.astype(np.float64),
        half_prs_grid.astype(np.float64),
    )
    data["Z"] = (data[temperature_var].dims, np.expand_dims(geopotential, axis=0))
    data["P"] = (data[temperature_var].dims, np.expand_dims(full_prs_grid, axis=0))

    return data


def _build_file_path(date, base_path, file_type="atm", step="f000"):
    """
    Create NOMADS filepaths for etiher upper air or surface data
    Args:
        date: (pd.Timestamp) date of GFS initialization
        base_path: (str) NOMADS base directory (archives last 10 days)
        file_type: (str) Type of analysis data (supports 'atm' or 'sfc')
        step: (str) "anl" or "f000" to "f009". f times have additional diagnostics
            like ugrd10 and vgrd10 not found in the analysis files.
    Returns:
        (str) NOMADS or Google Cloud filepaths that can be read in xarray with the h5netcdf engine
    """
    dir_path = date.strftime("gdas.%Y%m%d/%H/atmos/")
    file_name = date.strftime(f"gdas.t%Hz.{file_type}{step}.nc")

    return join(base_path, dir_path, file_name)


def _load_gfs_variable(variable, full_file_path=None):
    try:
        print("Loading ", variable)
        with xr.open_dataset(full_file_path, engine="h5netcdf", storage_options={"token": "anon"}) as full_ds:
            sub_ds = full_ds[variable].load()
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return sub_ds


def _load_gfs_data(full_file_path, variables, pool=None):
    """
    Load GFS data directly from Nomads or Google Cloud server
    Args:
        full_file_path: (str) NOMADS filepath
        variables: (list) list of variable names

    Returns:
        xr.Dataset
    """
    ds = xr.open_dataset(full_file_path, engine="h5netcdf", storage_options={"token": "anon"})
    available_vars = list(ds.data_vars)
    sub_variables = [v for v in variables if v in available_vars]
    load_vars = partial(_load_gfs_variable, full_file_path=full_file_path)
    var_ds_list = pool.map(load_vars, sub_variables)
    full_ds = xr.merge(var_ds_list)
    full_ds = full_ds.rename({"grid_xt": "longitude", "grid_yt": "latitude"})
    full_ds.attrs = ds.attrs
    return full_ds


def _combine_data(atm_data, sfc_data, temperature_var, specific_humidity_var):
    """
    Merge upper air and surface data
    Args:
        atm_data: (xr.Dataset) GFS upper air data
        sfc_data: (xr.Dataset) GFS surface data

    Returns:
        xr.Dataset
    """
    for var in sfc_data.data_vars:
        atm_data[var] = (sfc_data[var].dims, sfc_data[var].values)

    for var in atm_data.data_vars:
        if var in gfs_map.keys():
            atm_data = atm_data.rename({var: gfs_map[var]})

    data = _add_pressure_and_geopotential(atm_data, temperature_var, specific_humidity_var)

    return data


def _regrid_variable(variable_data, regridder):
    try:
        regridded_data = regridder(variable_data)
        regridded_data.name = variable_data.name
        return regridded_data
    except Exception as e:
        print(traceback.format_exc())
        raise e


def _regrid(nwp_data, output_grid, method="bilinear", pool=None):
    """
    Spatially regrid (interpolate) from GFS grid to CREDIT grid
    Args:
        nwp_data (xr.Dataset): GFS initial conditions
        output_grid (xr.Dataset): CREDIT grid
        method (str): conservative or bilinear

    Returns:
        (xr.Dataset): Regridded GFS initial conditions
    """
    try:
        import xesmf as xe
    except (ImportError, ModuleNotFoundError) as e:
        message = """xesmf not installed.\n
                Install esmf with conda first to prevent conda from overwriting numpy.\n
                `conda install -c conda-forge esmf esmpy`
                Then install xesmf with pip.\n
                `pip install xesmf`
                """
        raise e(message)

    if "time" in output_grid.variables.keys():
        ds_out = output_grid[["longitude", "latitude"]].drop_vars(["time"]).load()
    else:
        ds_out = output_grid[["longitude", "latitude"]].load()
    in_grid = nwp_data[["longitude", "latitude"]].load()
    regridder = xe.Regridder(in_grid, ds_out, method=method)
    results = []
    for variable in list(nwp_data.data_vars):
        da = nwp_data[variable]
        da.name = variable
        results.append(pool.apply_async(_regrid_variable, (da, regridder)))
    ds_re_list = []
    for result in results:
        ds_re_list.append(result.get())
    ds_regridded = xr.merge(ds_re_list)
    return ds_regridded.squeeze()


def _vertical_interpolation(
    state_dataset, model_level_file, model_levels, variable_mapping, variables, surface_pressure_var
):
    upper_vars = [var for var in variables if var in upper_air]
    surface_vars = [var for var in variables if var in surface]
    vars_500 = [var for var in variables if "500" in var]
    if "/" not in model_level_file:
        meta_path = str(files("credit.metadata"))
        model_level_file_path = join(meta_path, model_level_file)
    else:
        model_level_file_path = expandvars(model_level_file)
    a_model_name = "a_model"
    b_model_name = "b_model"
    level_var = "level"
    if "cam" in variable_mapping.lower():
        a_model_name = "hyam"
        b_model_name = "hybm"
        level_var = "level"
    with xr.open_dataset(model_level_file_path) as mod_lev_ds:
        valid_levels = np.isin(mod_lev_ds[level_var].values, model_levels)
        if a_model_name == "hyam":
            P0 = 1.0
            a_model = mod_lev_ds[a_model_name].values[valid_levels] * P0
        else:
            a_model = mod_lev_ds[a_model_name].values[valid_levels]
        b_model = mod_lev_ds[b_model_name].values[valid_levels]
    out_pressure, _ = create_reduced_pressure_grid(state_dataset[surface_pressure_var].values, a_model, b_model)
    interpolated_data = {}
    for var in upper_vars:
        interpolated_data[var] = {
            "dims": ["level", "latitude", "longitude"],
            "data": interp_hybrid_to_hybrid_levels(state_dataset[var].values, state_dataset["P"].values, out_pressure),
        }
    for var in vars_500:
        interpolated_data[var] = {
            "dims": ["latitude", "longitude"],
            "data": np.squeeze(
                interp_hybrid_to_pressure_levels(
                    state_dataset[level_map[var]].values,
                    state_dataset["P"].values,
                    np.array([50000.0], dtype=np.float32),
                )
            ),
        }
    for var in surface_vars:
        interpolated_data[var] = {
            "dims": state_dataset[var].dims,
            "data": state_dataset[var].values,
        }
    return interpolated_data


def _format_data(data_dict, regridded_data, model_levels):
    """
    Format data for CREDIT model ingestion
    Args:
        data_dict: (dict) Dictionary of xr.DataArrays of interpolated GFS model level data
        regridded_data: (xr.Dataset) GFS initial conditions on CREDIT grid
        model_levels: (list) list of model level indices to extract from L137 model levels

    Returns:
        xr.Dataset of GFS initial conditions interpolated to CREDIT grid and model levels
    """
    data = xr.Dataset.from_dict(data_dict).transpose("level", "latitude", "longitude", ...).expand_dims("time")
    data = data.assign_coords(
        level=model_levels,
        latitude=regridded_data["latitude"].values,
        longitude=regridded_data["longitude"].values,
        time=[pd.to_datetime(regridded_data["time"].values.astype(str))],
    )

    return data


def _format_datetime(init_time):
    """
    Format datetime string from CREDIT configuration file
    Args:
        init_time: (dict) Dictionary of Forecast times from configuration file

    Returns:
        pd.Timestamp of initialization time
    """
    dt = datetime.datetime(
        init_time["start_year"],
        init_time["start_month"],
        init_time["start_day"],
        init_time["start_hours"][0],
    )

    return pd.Timestamp(dt)
