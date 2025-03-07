import numpy as np
import pandas as pd
from os.path import join
import xarray as xr
import fsspec
import xesmf as xe
from credit.interp import geopotential_from_model_vars, create_pressure_grid

gfs_map = {'tmp': 'T', 'ugrd': 'U', 'vgrd': 'V', 'spfh': 'Q', 'pressfc': 'SP', 'tmp2m': 't2m'}
level_map = {'T500': 'T', 'U500': 'U', 'V500': 'V', 'Q500': 'Q', 'Z500': 'Z'}
upper_air = ['T', 'U', 'V', 'Q', 'Z']
surface = ['SP', 't2m']
STANDARD_GRAVITY = 9.80665

def build_GFS_init(output_grid, gdas_base_path="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/",
                   date="202503030600", variables=[], model_level_indices=[]):

    required_variables = ['pressfc', 'tmp', 'spfh', 'hgtsfc'] # required for calculating pressure and geopotential
    gfs_variables = list(set([k for k, v in gfs_map.items() if v in variables]).union(required_variables))
    atm_full_path = build_file_path(date, gdas_base_path, file_type='atm')
    sfc_full_path = build_file_path(date, gdas_base_path, file_type='sfc')
    gfs_atm_data = load_gfs_data(atm_full_path, gfs_variables)
    gfs_sfc_data = load_gfs_data(sfc_full_path, gfs_variables)
    gfs_data = combine_data(gfs_atm_data, gfs_sfc_data)
    regridded_gfs = regrid(gfs_data, output_grid)
    interpolated_gfs = interpolate_to_model_level(regridded_gfs, output_grid, model_level_indices, variables)
    final_data = format_data(interpolated_gfs, regridded_gfs, model_level_indices)

    return final_data


def add_pressure_and_geopotntial(data):

    sfc_pressure = data['SP'].values.squeeze()
    sfc_gpt = data['hgtsfc'].values.squeeze() * STANDARD_GRAVITY
    level_T = data['T'].values.squeeze()
    level_Q = data['Q'].values.squeeze()
    a_coeff = data.attrs['ak']
    b_coeff = data.attrs['bk']

    full_prs_grid, half_prs_grid = create_pressure_grid(sfc_pressure, a_coeff, b_coeff)
    geopotential = geopotential_from_model_vars(sfc_gpt, sfc_pressure, level_T, level_Q, half_prs_grid)
    data['Z'] = (data['T'].dims, np.expand_dims(geopotential, axis=0))
    data['P'] = (data['T'].dims, np.expand_dims(full_prs_grid, axis=0))

    return data


def build_file_path(date, base_path, file_type='atm'):

    date_obj = pd.Timestamp(date)
    dir_path = date_obj.strftime("gdas.%Y%m%d/%H/atmos/")
    file_name = date_obj.strftime(f"gdas.t%Hz.{file_type}anl.nc")

    return join(base_path, dir_path, file_name)


def load_gfs_data(full_file_path, variables):

    ds = xr.open_dataset(fsspec.open(full_file_path).open())
    available_vars = ds.data_vars
    vars = [v for v in variables if v in available_vars]
    ds = ds[vars].rename({'grid_xt': 'longitude', 'grid_yt': 'latitude'}).load()

    return ds


def combine_data(atm_data, sfc_data):

    for var in sfc_data.data_vars:
        atm_data[var] = (sfc_data[var].dims, sfc_data[var].values)

    for var in atm_data.data_vars:
        if var in gfs_map.keys():
            atm_data = atm_data.rename({var: gfs_map[var]})

    data = add_pressure_and_geopotntial(atm_data)

    return data


def regrid(nwp_data, output_grid, method="conservative"):

    ds_out = output_grid[['longitude', 'latitude']].drop_vars(['time']).load()
    in_grid = nwp_data[['longitude', 'latitude']].load()
    regridder = xe.Regridder(in_grid, ds_out, method=method)
    ds_regridded = regridder(nwp_data)

    return ds_regridded.squeeze()


def interpolate_to_model_level(regridded_nwp_data, output_grid, model_level_indices, variables):

    upper_vars = [var for var in variables if var in upper_air]
    surface_vars = [var for var in variables if var in surface]
    vars_500 = [var for var in variables if '500' in var]

    xp = regridded_nwp_data['P'].values
    fp = regridded_nwp_data
    output_pressure = (output_grid['a_half'] + output_grid['b_half'] * regridded_nwp_data['SP'])
    sampled_output_pressure = output_pressure[model_level_indices].values
    ny, nx = regridded_nwp_data.sizes['latitude'], regridded_nwp_data.sizes['longitude']
    interpolated_data = {}
    for var in upper_vars:
        fp_data = fp[var].values
        interpolated_data[var] = {'dims': ['latitude', 'longitude', 'level'],
                                  'data': np.array([np.interp(sampled_output_pressure[:, j, i], xp[:, j, i], fp_data[:, j, i])
                                                    for j in range(ny) for i in range(nx)]).reshape(ny, nx,
                                                                                                    len(model_level_indices))}
    for var in vars_500:
        prs = 50000 # 500mb
        fp_data = fp[level_map[var]].values
        interpolated_data[var] = {'dims': ['latitude', 'longitude'],
                                  'data': np.array([np.interp([prs], xp[:, j, i], fp_data[:, j, i])
                                                    for j in range(ny) for i in range(nx)]).reshape(ny, nx)}
    for var in surface_vars:
        interpolated_data[var] = {'dims': regridded_nwp_data[var].dims, 'data': regridded_nwp_data[var].values}

    return interpolated_data


def format_data(data_dict, regridded_data, model_levels):

    data = xr.Dataset.from_dict(data_dict).transpose('level', 'latitude', 'longitude', ...).expand_dims('time')
    data = data.assign_coords(level=model_levels,
                              latitude=regridded_data['latitude'].values,
                              longitude=regridded_data['longitude'].values,
                              time=[pd.to_datetime(regridded_data['time'].values.astype(str))])

    return data




