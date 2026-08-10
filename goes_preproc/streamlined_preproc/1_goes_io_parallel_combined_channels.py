#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# environment: /glade/work/dkimpara/conda-envs/xesmf


# %%


import sys
import traceback

import s3fs as s3
import xarray as xr

import os
import glob
import pickle
import argparse

import numpy as np
import datetime


from joblib import Parallel, delayed
import joblib


from functools import partial
from credit.pbs import get_num_cpus

import warnings
warnings.filterwarnings("error")


# %%


def convert_to_brightness_temp(ds):
    ds["BT_or_R"] = (ds["planck_fk2"] /
            np.log(ds["planck_fk1"] /  np.clip(ds["Rad"], 1e-8, None) + 1) -
            ds["planck_bc1"]
               ) / ds["planck_bc2"]

    return ds

def convert_to_reflectance(ds):
    if any(ds["kappa0"] <= -999.):
        raise ValueError("kappa0 does not exist")

    ds["BT_or_R"] = ds["kappa0"] * np.clip(ds["Rad"], 1e-8, None)

    return ds

# regridding
def coarsen_ds(regrid_dict,
               ds,
               ds_outgrid,
               variables=["BT_or_R"],
               keep_vars=["yaw_flip_flag", "kappa0",
                         "planck_bc1", "planck_bc2",
                         "planck_fk1", "planck_fk2"]
              ):

    num_lat, num_lon = len(ds_outgrid.lat), len(ds_outgrid.lon)
    num_t = len(ds.t)

    ds_outgrid = ds_outgrid.copy(deep=False) # just need refs to latlon components, will not modify. But need to copy otherwise will modify underlying ds

    # with warnings.catch_warnings(record=True) as w:
    #     warnings.simplefilter('always')

    for var in variables:
        coarsened_var = coarsen(regrid_dict, ds[var].values)
        ds_outgrid[var] = ("t", "lat", "lon",), np.moveaxis(coarsened_var.reshape(num_lat, num_lon, num_t), -1, 0)

    ds_outgrid[keep_vars] = ds[keep_vars]
        
        # if w: #if a warning is thrown
        #     print(f"nanmean on {ds.t.isel(t=0).values}, channel {ds.band_id.values}")

    return ds_outgrid

def coarsen(regrid_dict, array):

    neighbors = regrid_dict["neighbors"]
    not_nan_indices = regrid_dict["not_nan_indices"]

    # array_flat = array.ravel(order="C")[not_nan_indices]
    num_t, grid_size = array.shape[0], array.shape[1] * array.shape[2]
    array_flat = np.moveaxis(array, 0, -1).reshape(grid_size, num_t, order="C")
    array_flat = array_flat[not_nan_indices]

    means = np.empty((len(neighbors), num_t))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, indices in enumerate(neighbors):
            avg = np.nanmean(array_flat[indices], axis=0)
            means[i] = avg


    return means # flat version of output


def process_L1b_by_HOURS(channels, satellite, ins_sec, loc_outgrid, loc_regrid_dict,
                        hour_datetimes):
    with open(loc_regrid_dict, 'rb') as f:
        regrid_dict = pickle.load(f)

    outgrid = xr.open_dataset(loc_outgrid, engine="h5netcdf")

    errors = []

    for dt in hour_datetimes:
        errors.append(process_L1b_by_hour(channels, satellite, ins_sec, outgrid, regrid_dict,
                           dt))
    return errors

### parallel funcs
def process_L1b_by_hour(channels, satellite, ins_sec, outgrid, regrid_dict,
                        hour_datetime):

    year, day, hour = hour_datetime.year, hour_datetime.timetuple().tm_yday, hour_datetime.hour

    dir = f"s3://noaa-{satellite}/{ins_sec}/{year}/{day:03}/{hour:02}"

    fs = s3.S3FileSystem(anon=True)

    try:
        files = fs.ls(dir)

        for channel in channels:
            channel_files = [file for file in files if f"C{channel:02}" in file]
            assert len(channel_files) > 0

    except:
        error_message = dir + " is missing or not all channels exist"
        print(error_message)
        return dir

    if files:
        for channel in channels:
            # process_L1b(fs, channel, files, outgrid, regrid_dict)

            try:
                process_L1b(hour_datetime, fs, channel, files, outgrid, regrid_dict)
            except:
                print(f"error while processing {str(dir)}")
                print(traceback.format_exc()) # This line is for getting traceback.
                print(sys.exc_info()[2]) # This line is getting for the error type.
                print(f"error in channel {channel} {str(dir)}")
                pass
    else:
        print(dir + " is missing files")
        return f"{dir} is missing files"
    
    return True

# per channel computations
# about 330 CPU-hours per year
# 5gb per CPU

def process_L1b(hour_datetime, fs, channel, files, ds_outgrid, regrid_dict):
    # for channel in channels:
    channel_files = [file for file in files if f"C{channel:02}" in file]

    datasets = []
    for file in channel_files:
        with fs.open(file) as f:
            dataset = xr.open_dataset(f).expand_dims("t")
            drop_vars = set(dataset.variables) & set(["time_bounds_swaths", "time_bounds_rows", "a_h_NRTH", "b_h_NRTH"])
            dataset = dataset.drop_vars(drop_vars)

            datasets.append(dataset.load())

    ds = xr.concat(datasets, dim="t").sortby("t")
    # subset to patch to convolve over
    x_bounds, y_bounds = regrid_dict["x_bounds"], regrid_dict["y_bounds"]

    ds = ds.isel(x=slice(*x_bounds), y=slice(*y_bounds))

    # convert to BT or reflectance
    if int(channel) <= 6:
        ds = convert_to_reflectance(ds)
    else:
        ds = convert_to_brightness_temp(ds)

    ds = ds.drop_vars(["Rad"])

    ############ coarsen
    ds = coarsen_ds(regrid_dict, ds, ds_outgrid)

    means = ds.mean(dim=["lat", "lon"])
    ds["BT_or_R_mean"] = means["BT_or_R"]

    ds = ds.assign_coords({"channel": channel}).expand_dims("channel")

    ############ write out files
    #get timestamp
    # dt64 = ds.t[0].values
    # dt = dt64.astype('datetime64[s]').astype(datetime.datetime)
    
    # get year month day hour from directory we are processing
    year, month, day, hour = hour_datetime.year, hour_datetime.month, hour_datetime.day, hour_datetime.hour

    # Extract and format
    formatted = f"{year}-{month:02}-{day:02}_{hour:02d}Z"
    filename = f"{formatted}_C{channel:02}.nc"

    os.makedirs(os.path.join(save_dir, str(year)), exist_ok=True)

    ds.to_netcdf(os.path.join(save_dir, str(year), filename), mode="w", engine="h5netcdf")

    return True

def chunk_list(a, num_chunks):
    subsets = np.array_split(a, num_chunks)
    return subsets


# # process

# %%

if __name__ == "__main__":
    """
    downloads goes files from specified channels to $save_dir/year
    """
    np.seterr(all='warn')


    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('-c','--config', help='Config File', required=True)
    parser.add_argument('-s','--sat', help='satellite goes##', required=False, type=str)
    parser.add_argument('-y','--year', help='Year', required=True, type=int)
    parser.add_argument('-r','--reprocessed', help='reprocessed data', required=False, type=int)
    parser.add_argument('-n','--num_cpus', help='num_cpus', required=False, type=int)
    parser.add_argument('-t','--test', help='Testing', required=False)
    args = vars(parser.parse_args())
        
    # config_loc = args["config"]
    # with open(config_loc, "r") as f:
    #     config_dict = yaml.safe_load(f)

    config_dict = args
    print(config_dict)
    test_mode = str(args["test"])
    
    # default parameters
    defaults = {
                "save_dir": "/glade/derecho/scratch/dkimpara/goes-cloud-dataset",
                "num_cpus": 62,
                "satellite": "goes16",
                "instrument": "ABI-L1b",
                "sector": "RadF",
                "reprocessed_data": True,
                "channels": list(range(3, 17)),
                "year": None,
                "loc_outgrid": "/glade/work/dkimpara/goes-cloud-data/0_1deg_grid.nc",
                "loc_regrid_dict": "/glade/work/dkimpara/goes-cloud-data/6km_regrid_data.pkl",
                "10m_scans_only": True,
            }


    config_dict = defaults | config_dict # config dict overwrites defaults
    print(config_dict)
    # Assign each key to a variable
    save_dir = config_dict["save_dir"]
    num_cpus = config_dict["num_cpus"]
    instrument = config_dict["instrument"]
    sector = config_dict["sector"]
    reprocessed_data = config_dict["reprocessed_data"]
    channels = config_dict["channels"]
    year = int(config_dict["year"])
    loc_outgrid = config_dict["loc_outgrid"]
    loc_regrid_dict = config_dict["loc_regrid_dict"]
    satellite = config_dict["satellite"]

    print(f"processing {year}")

    ############## setup parallelism ###############
    if not num_cpus:
        num_cpus = get_num_cpus()

    if num_cpus == 1:
        num_cpus = 2 # so joblib doesn't error out
    print(f"using {num_cpus} cpus")
    ins_sec = f"{instrument}-{sector}{'-Reproc' if reprocessed_data else ''}"
    

    f = partial(process_L1b_by_HOURS, channels, satellite, ins_sec, loc_outgrid, loc_regrid_dict)

    if config_dict["10m_scans_only"] and (year == 2018 or year == "2018"):
        print("subsetting 2018 for 10min scans only")
        start_datetime = datetime.datetime(year, 4, 3, 0, 0)
    elif year == 2017 or year == "2017":
        start_datetime = datetime.datetime(year, 12, 17, 0, 0)
    else:
        start_datetime = datetime.datetime(year, 1, 1, 0, 0)

    end_datetime = datetime.datetime(year, 12, 31, 23, 0)
    
    # Generate list of all hours
    all_hours = [start_datetime + datetime.timedelta(hours=i) 
                     for i in range(int((end_datetime - start_datetime).total_seconds() // 3600) + 1)]
    
    all_hours = chunk_list(all_hours, 10 * num_cpus) # we want a big chunk for each cpu so each process doesnt have to load in metadata too often

    if test_mode == "1":
        all_hours = [all_hours[0][:2]]
        print(f"testing with {all_hours}")
   
    ############ compute ##################
    errors = Parallel(n_jobs = num_cpus - 1)(delayed(f)(hour_datetimes)
                                for hour_datetimes in all_hours)
    
    
    err_file = os.path.join(save_dir, f"{year}_errors.txt")
    with open(err_file, "w") as file:
        for item in errors:
            for result in item:
                if result != True:
                    file.write(str(result) + "\n")


