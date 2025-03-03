import numpy as np
import pandas as pd
from os.path import join
import xarray as xr
import fsspec
import xesmf as xe

gfs_map = {'tmp': 'T', 'ugrd': 'U', 'vgrd': 'V', 'spfh': 'Q', 'pressfc': 'SP', 'tmp2m': 't2m', 'delz': 'Z500'}
level_map = {'T500': 'T', 'U500': 'U', 'V500': 'V', 'Q500': 'Q', 'Z500': 'Z'}
upper_air = ['T', 'U', 'V', 'Q', 'Z']
surface = ['SP', 't2m']

def build_GFS_init(output_grid, gdas_base_path="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/",
                   date="202502200600", variables=[], model_level_indices=[]):

    gfs_variables = [k for k, v in gfs_map.items() if v in variables]
    atm_full_path = build_file_path(date, gdas_base_path, file_type='atm')
    sfc_full_path = build_file_path(date, gdas_base_path, file_type='sfc')
    gfs_atm_data = load_gfs_data(atm_full_path, gfs_variables)
    gfs_sfc_data = load_gfs_data(sfc_full_path, gfs_variables)
    gfs_data = combine_data(gfs_atm_data, gfs_sfc_data)
    regridded_gfs = regrid(gfs_data, output_grid)
    interpolated_gfs = interpolate_to_model_level(regridded_gfs, output_grid, model_level_indices, variables)
    final_data = format_data(interpolated_gfs, regridded_gfs, model_level_indices)

    return final_data


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
    if 'delz' in available_vars and 'delz' in variables:
        ds['delz'].values = np.abs(ds['delz'].cumsum(dim='pfull').values) * 9.81
        ds = ds.rename({'delz': 'Z'})
    return ds


def combine_data(atm_data, sfc_data):

    for var in sfc_data.data_vars:
        atm_data[var] = (sfc_data[var].dims, sfc_data[var].values)

    for var in atm_data.data_vars:
        if var in gfs_map.keys():
            atm_data = atm_data.rename({var: gfs_map[var]})

    return atm_data


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

    xp = regridded_nwp_data['pfull'].values
    fp = regridded_nwp_data
    output_pressure = (output_grid['a_half'] + output_grid['b_half'] * regridded_nwp_data['SP']) / 100
    sampled_output_pressure = output_pressure[model_level_indices].values
    ny, nx = fp['T'].shape[1], fp['T'].shape[2]
    interpolated_data = {}
    for var in upper_vars:
        fp_data = fp[var].values
        interpolated_data[var] = {'dims': ['latitude', 'longitude', 'level'],
                                  'data': np.array([np.interp(sampled_output_pressure[:, j, i], xp, fp_data[:, j, i])
                                                    for j in range(ny) for i in range(nx)]).reshape(ny, nx,
                                                                                                    len(model_level_indices))}
    for var in vars_500:
        fp_data = fp[level_map[var]].values
        interpolated_data[var] = {'dims': ['latitude', 'longitude'],
                                  'data': np.array([np.interp([500], xp, fp_data[:, j, i])
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




