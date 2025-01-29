"""
This script gathers ERA5 data using Dask and converts it to Zarr format.

Dependencies:
- xarray (as xr)
- glob
- os
- numpy as np
- pandas as pd
- ProgressBar from dask.diagnostics
- time
- delayed from dask
- dask.array as da
- xarray (imported again as xr)
- delayed from dask (imported again)
- ProgressBar from dask.diagnostics (imported again)
- re
- argparse
- os.environ['ESMFMKFILE'] set to '/glade/work/wchapman/miniconda3.1/envs/MLWPS/lib/esmf.mk' #WEC
- xesmf as xe #WEC
- datetime, timedelta from datetime

Example use:
python GatherERA5_DistributedDask_toZarr.py --start_date 2011-01-01 --end_date 2012-01-01

Model levels to pressure levels reference:
https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height?preview=/158636068/226495690/levels_137.png

Settings (Modify this block):
- start_date: '2011-01-01'
- end_date: '2012-01-01' (Make sure this date is after the start date)
- interval_hours: 1 (Hour interval for data retrieval)
- FPout: '/glade/derecho/scratch/wchapman/STAGING2/' (Output directory for files)
- prefix_out: 'ERA5_e5.oper.ml.v3' (File prefix)
- project_num: 'NAML0001' (Project key)
- remove_dask_worker_scripts: True

Single level variables (Modify this block):
- varin_l: ['VAR_2T', 'T', 'V', 'U', 'Q', 'Z'] (ERA variable names)
- lev_list: [None, 500, 500, 500, 500, 500] (Levels for variables, use None for surface vars)
- varout_l: ['t2m', 'T500', 'V500', 'U500', 'Q500', 'Z500'] (Output variable names)
"""

import xarray as xr
import glob
import os
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
import time
from dask import delayed
import dask
import re
import argparse

# os.environ['ESMFMKFILE'] = '/glade/work/wchapman/miniconda3.1/envs/MLWPS/lib/esmf.mk' #WEC
import xesmf as xe  # WEC
from datetime import datetime, timedelta

# ###
# example use:
# python GatherERA5_DistributedDask_toZarr.py --start_date 2011-01-01 --end_date 2012-01-01
# ####


# # model levels to pressure levels:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height?preview=/158636068/226495690/levels_137.png

#### settings !!! MODIFY THIS BLOCK !!!
# start_date = '2011-01-01' #these arguements are now passed to the program.
# end_date = '2012-01-01' #make sure this date is after the start date...
interval_hours = 1  # what hour interval would you like to get? [i.e: 1 = 24 files/day, 6 = 4 files/day]
FPout = (
    "/glade/derecho/scratch/wchapman/STAGING2/"  # where do you want the files stored?
)
prefix_out = "ERA5_e5.oper.ml.v3"  # what prefix do you want the files stored with?
project_num = "NAML0001"  # what project key dfo you have?
remove_dask_worker_scripts = True


# #single level variables {supports: T2m, T[lev],V[lev],U[lev],Q[lev],Z[lev]}:
# note! make sure that level exists! We could do a "method=nearest" but I think this
# leads to less clarity on what you are actually predicting (PS comes standard...).

# varin_l = ['VAR_2T','T','V','U','Q','Z'] #what is the ERA variable called?
# lev_list = [None,500,500,500,500,500] #what level do you want (NONE for surface vars).
# varout_l = ['t2m','T500','V500','U500','Q500','Z500'] #what should the variable be named?


levels = [137]
varin_l = []  # what is the ERA variable called?
lev_list = []  # what level do you want (NONE for surface vars).
varout_l = []  # what should the variable be named?


# ### settings !!! MODIFY THIS BLOCK !!!

# ### ++++++ dask NCAR client:
# ### ++++++ dask NCAR client:
print("...setting up dask client...")
if "client" in locals():
    client = locals()["client"]
    client.shutdown()
    print("...shutdown client...")
else:
    print("client does not exist yet")
from distributed import Client
from dask_jobqueue import PBSCluster

cluster = PBSCluster(
    account=project_num,
    walltime="12:00:00",
    cores=1,
    memory="70GB",
    shared_temp_directory="/glade/derecho/scratch/wchapman/tmp",
    queue="main",
)
cluster.scale(jobs=40)
client = Client(cluster)
#### ----- dask NCAR client:
client
#### ----- dask NCAR client:


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument(
        "--start_date", required=True, help="Start date in the format YYYY-MM-DD"
    )
    parser.add_argument(
        "--end_date", required=True, help="End date in the format YYYY-MM-DD"
    )
    # Add other arguments as needed
    args = parser.parse_args()
    return args


# ##dask NCAR client:
# assert that dates wanted > 0


def subtract_one_day(date_str):
    # Convert the input string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # Subtract one day
    new_date_obj = date_obj - timedelta(days=1)

    # Format the result as a string in the original format
    new_date_str = new_date_obj.strftime("%Y-%m-%d")

    return new_date_str


def find_strings_with_substring(string_list, substring):
    """
    Find strings in a list that contain a specified substring.

    Parameters:
    - string_list (list): List of strings to search through.
    - substring (str): Substring to search for within the strings.

    Returns:
    - list: List of strings from string_list that contain the specified substring.
    """
    # Initialize an empty list to store matching strings
    matching_strings = []

    # Iterate through the list
    for string in string_list:
        # Check if the specified substring is present in the current string
        if substring in string:
            matching_strings.append(string)

    # Return the list of matching strings
    return matching_strings


def flatten_list(input_list):
    flattened_list = []
    for item in input_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


##function get file paths ...
def fp_dates_wanted(Dateswanted):
    years_wanted = Dateswanted[:].year
    months_wanted = Dateswanted[:].month
    day_wanted = Dateswanted[:].day

    list_yrm = []
    for ywmw in zip(years_wanted, months_wanted):
        list_yrm.append(str(ywmw[0]) + f"{ywmw[1]:02}")

    fp_t = []
    fp_u = []
    fp_v = []
    fp_q = []
    fp_ps = []

    # other grid:
    fp_z = []
    fp_ub = []
    fp_vb = []
    fp_tb = []
    fp_qb = []

    # surface / single level:
    fp_t2m = []

    lastday = str(Dateswanted[-1])[:10]

    for yrm_fp in np.unique(list_yrm):
        for dayday in np.unique(day_wanted):
            fp_u.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.6/e5.oper.an.ml/"
                        + yrm_fp
                        + "/"
                        + "*_u*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )
            fp_v.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.6/e5.oper.an.ml/"
                        + yrm_fp
                        + "/"
                        + "*_v*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )
            fp_t.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.6/e5.oper.an.ml/"
                        + yrm_fp
                        + "/"
                        + "*_t*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )
            fp_q.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.6/e5.oper.an.ml/"
                        + yrm_fp
                        + "/"
                        + "*_q*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )
            fp_ps.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.6/e5.oper.an.ml/"
                        + yrm_fp
                        + "/"
                        + "*_sp*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )
            fp_z.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
                        + yrm_fp
                        + "/"
                        + "*_z*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )  # WEC
            fp_ub.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
                        + yrm_fp
                        + "/"
                        + "*_u.*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )  # WEC
            fp_vb.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
                        + yrm_fp
                        + "/"
                        + "*_v.*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )  # WEC
            fp_tb.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
                        + yrm_fp
                        + "/"
                        + "*_t.*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )  # WEC
            fp_qb.append(
                sorted(
                    glob.glob(
                        "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
                        + yrm_fp
                        + "/"
                        + "*_q.*"
                        + yrm_fp
                        + f"{dayday:02}"
                        + "*.nc"
                    )
                )
            )  # WEC
            if yrm_fp[:4] + "-" + yrm_fp[4:] + "-" + f"{dayday:02}" == lastday:
                break
        fp_t2m.append(
            sorted(
                glob.glob(
                    "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc/"
                    + yrm_fp
                    + "/"
                    + "*_2t*"
                    + yrm_fp
                    + "*.nc"
                )
            )
        )  # WEC

    fp_u = flatten_list(fp_u)
    fp_v = flatten_list(fp_v)
    fp_t = flatten_list(fp_t)
    fp_q = flatten_list(fp_q)
    fp_ps = flatten_list(fp_ps)
    fp_t2m = flatten_list(fp_t2m)
    fp_z = flatten_list(fp_z)
    fp_ub = flatten_list(fp_ub)
    fp_vb = flatten_list(fp_vb)
    fp_tb = flatten_list(fp_tb)
    fp_qb = flatten_list(fp_qb)

    files_dict = {
        "u": np.unique(fp_u),
        "v": np.unique(fp_v),
        "t": np.unique(fp_t),
        "q": np.unique(fp_q),
        "ps": np.unique(fp_ps),
        "z": np.unique(fp_z),
        "ub": np.unique(fp_ub),
        "vb": np.unique(fp_vb),
        "tb": np.unique(fp_tb),
        "qb": np.unique(fp_qb),
        "t2m": np.unique(fp_t2m),
    }

    return files_dict


def make_nc_files(files_dict, Dateswanted, Dayswanted):
    for dw in Dayswanted:
        print(str(dw)[:10])
        substring_match = str(dw)[:4] + str(dw)[5:7] + str(dw)[8:10]
        smatch_u = find_strings_with_substring(files_dict["u"], substring_match)
        smatch_v = find_strings_with_substring(files_dict["v"], substring_match)
        smatch_t = find_strings_with_substring(files_dict["t"], substring_match)
        smatch_q = find_strings_with_substring(files_dict["q"], substring_match)
        smatch_ps = find_strings_with_substring(files_dict["ps"], substring_match)
        DS_u = xr.open_mfdataset(smatch_u)
        sel_times = Dateswanted.intersection(DS_u["time"])
        DS_v = xr.open_mfdataset(smatch_v).sel(time=sel_times)
        DS_t = xr.open_mfdataset(smatch_t).sel(time=sel_times)
        DS_q = xr.open_mfdataset(smatch_q).sel(time=sel_times)
        DS_ps = xr.open_mfdataset(smatch_ps).sel(time=sel_times)
        print("loading")
        DS = xr.merge([DS_u.sel(time=sel_times), DS_v, DS_t, DS_q]).load()
        print("loaded")

        for ee, tt in enumerate(DS["time"]):
            hourdo = DS["time.hour"][ee]

            datstr = str(dw)[:4] + str(dw)[5:7] + str(dw)[8:10] + f"{hourdo:02}"
            # DS.sel(time=tt).squeeze().to_netcdf()
            out_file = FPout + "/" + prefix_out + ".uvtq." + datstr + ".nc"
            write_job = DS.sel(time=tt).squeeze().to_netcdf(out_file, compute=False)
            with ProgressBar():
                print(f"Writing to {out_file}")
                write_job.compute()
            print(out_file)
            out_file = FPout + "/" + prefix_out + ".ps." + datstr + ".nc"
            DS_ps["Z_GDS4_SFC"] = xr.zeros_like(DS_ps["SP"])
            DS_ps["Z_GDS4_SFC"][:, :] = Static_zheight["Z_GDS4_SFC"].values
            write_job = DS_ps.sel(time=tt).squeeze().to_netcdf(out_file, compute=False)
            with ProgressBar():
                print(f"Writing to {out_file}")
                write_job.compute()
            print(out_file)

    return DS, DS_ps


def add_staggered_grid(FPout, prefix_out):
    prefix_out = "test_out_"
    all_files = sorted(glob.glob(FPout + "/" + prefix_out + "??????????.nc"))

    for fdfd in all_files:
        print(fdfd)
        BB = xr.open_dataset(fdfd)
        bbus = xr.zeros_like(BB["U"]).to_dataset(name="US")
        bbus["US"][:, :] = BB["U"]
        bbvs = xr.zeros_like(BB["V"]).to_dataset(name="VS")
        bbvs["VS"][:, :] = BB["V"]
        bball = xr.merge([BB, bbus, bbvs]).chunk()
        bball.to_netcdf(fdfd[:-13] + ".s." + fdfd[-13:])
        os.remove(fdfd)
    return all_files, BB


# WEC
def regrid_(file_list, varin, varout, method="bilinear", level=None):
    """
    Regrid the input variable from a list of files to a predefined grid.

    Parameters:
    - file_list (list): List of input files.
    - varin (str): Input variable name.
    - varout (str): Output variable name.
    - method (str): Resampling method (default is 'bilinear').
    - level (str, optional): Vertical level to select (default is None).

    Returns:
    - xr.Dataset: Regridded dataset.
    """
    print("!!!!!!! regridding:", varout, "!!!!!!!")
    if level is None:
        DSog = xr.open_mfdataset(file_list, parallel=True)
    else:
        DSog = (
            xr.open_mfdataset(file_list, parallel=True)
            .sel(level=level)
            .squeeze()
            .drop("level")
        )

    ds_out = xr.open_dataset("/glade/u/home/wchapman/MLWPS/Stage_Data/ML_grid.nc")
    fn = xr.open_dataset(
        "/glade/u/home/wchapman/MLWPS/Stage_Data/bilinear_721x1440_640x1280.nc"
    )
    regridder = xe.Regridder(DSog, ds_out, "bilinear", weights=fn)
    drout = regridder(DSog[varin])
    return drout.to_dataset(name=varout)


# WEC
def regrid_save_zarr(regrid_sl, varin_l, varout_l, lev_list, start_time, end_time):
    """
    Regrid and save variables to Zarr format.

    Parameters:
    - regrid_sl (dict): Dictionary containing regridded datasets.
    - varin_l (list): List of input variable names.
    - varout_l (list): List of output variable names.
    - lev_list (list): List of vertical levels.
    - start_time (str): Start date for naming output Zarr files.
    - end_time (str): End date for naming output Zarr files.

    Returns:
    - list: List of paths to saved Zarr files.
    """
    zarr_out = []
    for ii in range(len(varin_l)):
        nameoutz = f"{FPout}/{varout_l[ii]}_{start_date}_{end_date}_staged.zarr"
        print("trying: ", nameoutz)
        if os.path.exists(nameoutz):
            print("skipping zarr as it already exists")
        else:
            DSout = regrid_(
                regrid_sl[varin_l[ii]],
                varin=varin_l[ii],
                varout=varout_l[ii],
                level=lev_list[ii],
                method="bilinear",
            )  # WEC
            DSout = DSout.chunk({"time": 10})
            DSout.to_zarr(nameoutz)
        zarr_out.append(nameoutz)
    return zarr_out


def preprocess_levs(ds, levels):
    return ds.sel(level=levels)


def make_nc_files_optimized(
    files_dict, Dateswanted, Dayswanted, FPout, prefix_out, levels
):
    """
    Optimized function to perform a specific task using Dask with specified resources.

    Parameters:
    - files_dict: A dictionary of files.
    - Dateswanted: List of dates.
    - Dayswanted: List of days.
    - FPout: Output file path.
    - prefix_out: Output file prefix.

    Returns:
    - delayed_writes: List of delayed write operations.
    """
    Static_zheight = xr.open_dataset(
        "/glade/u/home/wchapman/RegriddERA5_CAMFV/static_operation_ERA5_zhght.nc"
    )
    log_files = []
    delayed_writes = []
    for dw in Dayswanted:
        print(str(dw)[:10])
        substring_match = str(dw)[:4] + str(dw)[5:7] + str(dw)[8:10]
        smatch_u = find_strings_with_substring(files_dict["u"], substring_match)
        smatch_v = find_strings_with_substring(files_dict["v"], substring_match)
        smatch_t = find_strings_with_substring(files_dict["t"], substring_match)
        smatch_q = find_strings_with_substring(files_dict["q"], substring_match)
        smatch_ps = find_strings_with_substring(files_dict["ps"], substring_match)

        DS_u = xr.open_mfdataset(
            smatch_u, parallel=True, preprocess=lambda ds: preprocess_levs(ds, levels)
        )
        sel_times = Dateswanted.intersection(DS_u["time"])
        DS_v = xr.open_mfdataset(
            smatch_v, parallel=True, preprocess=lambda ds: preprocess_levs(ds, levels)
        ).sel(time=sel_times)
        DS_t = xr.open_mfdataset(
            smatch_t, parallel=True, preprocess=lambda ds: preprocess_levs(ds, levels)
        ).sel(time=sel_times)
        DS_q = xr.open_mfdataset(
            smatch_q, parallel=True, preprocess=lambda ds: preprocess_levs(ds, levels)
        ).sel(time=sel_times)
        DS_ps = xr.open_mfdataset(smatch_ps, parallel=True).sel(time=sel_times)

        print("loading")
        DS = xr.merge([DS_u.sel(time=sel_times), DS_v, DS_t, DS_q])
        print("done loading")

        DS = DS.drop_vars(
            ["weight", "utc_date", "a_half", "zero", "a_model", "b_model", "b_half"]
        )
        DS_ps = DS_ps.drop_vars(["weight", "utc_date", "zero"])
        DS = xr.merge([DS, DS_ps])
        print("merged with this many time slots: ", DS["time"].shape)

        datstr = str(dw)[:4] + str(dw)[5:7] + str(dw)[8:10]
        out_file_uvtq = FPout + "/" + prefix_out + ".uvtq." + datstr + ".nc"
        # delayed_write_uvtq = delayed(DS.squeeze().to_netcdf)(out_file_uvtq)
        delayed_write_uvtq = delayed(write_to_netcdf)(DS.squeeze(), out_file_uvtq)
        log_files.append(out_file_uvtq)
        delayed_writes.append(delayed_write_uvtq)

    # Compute the delayed write operations concurrently
    print("writing")
    with ProgressBar():
        delayed_writes = list(dask.compute(*delayed_writes))

    return delayed_writes, log_files


def write_to_netcdf(ds, filename):
    print(filename)
    ds.to_netcdf(filename, engine="netcdf4")


def check_file_size(file_path, size_limit):
    if os.path.exists(file_path):
        file_size = get_folder_size(file_path)

        if file_size > size_limit:
            print(f"The file {file_path} exists and is larger than {size_limit} bytes.")

            return True
        else:
            print(
                f"The file {file_path} exists but is not larger than {size_limit} bytes."
            )
            print(f"The file size is {file_size} bytes.")
            return False
    else:
        print(f"The file {file_path} does not exist.")
        return False


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


def find_strings_by_pattern(string_list, pattern):
    """
    Find strings in a list that match a specified pattern.

    Parameters:
    - string_list (list): List of strings to search through.
    - pattern (str): Regular expression pattern to match within the strings.

    Returns:
    - list: List of strings from string_list that match the specified pattern.
    """

    # Initialize an empty list to store matching strings
    matching_strings = []

    # Compile the regular expression pattern
    compiled_pattern = re.compile(pattern)

    # Iterate through the list
    for string in string_list:
        # Check if the pattern matches the current string
        if compiled_pattern.search(string):
            matching_strings.append(string)

    # Return the list of matching strings
    return matching_strings


def divide_datetime_index(date_index, max_items_per_division=30):
    """
    Divide a DatetimeIndex into sublists with a maximum number of items per division.
    This prevents the dask machines from trying to write all the files at once and
    restricts it to the max_items_per_division

    Parameters:
    - date_index: DatetimeIndex to be divided.
    - max_items_per_division: Maximum number of items per division (default is 30).

    Returns:
    - divided_lists: List of sublists.
    """
    # Initialize an empty list to store the divided lists
    divided_lists = []

    # Initialize a sublist with the first date
    sublist = [date_index[0]]

    # Iterate through the remaining dates
    for date in date_index[1:]:
        # Add the current date to the sublist
        sublist.append(date)

        # Check if the sublist has reached the maximum allowed size
        if len(sublist) == max_items_per_division:
            # If it has, add the sublist to the divided_lists and reset the sublist
            divided_lists.append(sublist)
            sublist = []

    # If there are remaining items in the sublist, add it to the divided_lists
    if sublist:
        divided_lists.append(sublist)

    # Ensure that every division has at least two items by merging the last two divisions if necessary
    if len(divided_lists[-1]) < 2 and len(divided_lists) > 1:
        last_two_lists = divided_lists[-2:]  # Get the last two divisions
        combined_list = sum(last_two_lists, [])  # Combine them
        divided_lists = divided_lists[:-2]  # Remove the last two divisions
        divided_lists.append(combined_list)  # Add the combined list back

    return divided_lists


def increment_date_by_one_day(date_str):
    """
    Increment a date by one day and return it as a string.

    Parameters:
    - date_str: Input date string in the format 'YYYY-MM-DD'.

    Returns:
    - incremented_date_str: Date string incremented by one day.
    """
    # Convert the input date string to a pandas Timestamp
    date = pd.Timestamp(date_str)

    # Increment the date by one day
    incremented_date = date + pd.DateOffset(days=1)

    # Convert the incremented date back to a string in the same format
    incremented_date_str = incremented_date.strftime("%Y-%m-%d")

    return incremented_date_str


def find_staged_files(start_date, end_date):
    """
    Generates a list of file paths based on a date range.

    Parameters:
    - start_date (str): The start date of the range in the format 'YYYY-MM-DD'.
    - end_date (str): The end date of the range in the format 'YYYY-MM-DD'.

    Returns:
    - files_ (list): List of file paths corresponding to each date in the range.
    """

    # Generate a daily date range between start_date and end_date
    date_range_daily = pd.date_range(start_date, end_date)
    # List to store file paths
    files_ = []

    # Iterate through each date in the range
    for dtdt in date_range_daily:
        d_file = FPout + prefix_out + ".uvtq." + str(dtdt)[:10].replace("-", "") + ".nc"
        files_.append(d_file)

        # Check if the file exists; raise an error if not
        if not os.path.exists(d_file):
            raise FileNotFoundError(f"File not found: {d_file}")
    return files_


# Function to load and add a new dimension
def load_and_add_dimension(file_path):
    """
    Load data from a given file and add a new dimension.

    Parameters:
    - file_path (str): The path to the file containing the data.

    Returns:
    - data (numpy.ndarray): The loaded data with an added dimension at the beginning.
    """
    data = np.load(file_path)
    return data[np.newaxis, ...]  # Add a new dimension at the beginning


def flatten_list(list_of_lists):
    """
    Flatten a list of lists.

    Parameters:
    - list_of_lists (list): A list containing sublists.

    Returns:
    - flattened_list (list): A flattened list containing all elements from sublists.
    """
    return [item for sublist in list_of_lists for item in sublist]


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Extract start_date and end_date from the arguments
    start_date = args.start_date
    end_date = args.end_date
    end_date_minus1 = subtract_one_day(end_date)

    # ... (rest of your script)
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    print("here we go")
    ##look at all the dates:
    Dayswantedtot = pd.date_range(
        start=start_date, end=end_date, freq=str(interval_hours) + "D"
    )
    ##look at all the dates:
    # log for the files that are created.
    all_files = []
    AFout = f"{FPout}/AllFiles_{start_date}_{end_date_minus1}_staged.zarr"
    size_limit = 800 * 1024 * 1024 * 1024  # 800GB in bytes
    print("checking folder size")
    if not check_file_size(AFout, size_limit):
        print(len(Dayswantedtot))
        if len(Dayswantedtot) < 4:
            start_time = time.time()  # Record the start time
            Dayswanted = pd.date_range(
                start=start_date, end=end_date, freq=str(interval_hours) + "D"
            )
            Dateswanted = pd.date_range(
                start=start_date, end=end_date, freq=str(interval_hours) + "H"
            )
            Static_zheight = xr.open_dataset(
                "/glade/u/home/wchapman/RegriddERA5_CAMFV/static_operation_ERA5_zhght.nc"
            )
            files_dict = fp_dates_wanted(Dateswanted)
            # make the files:
            print("...starting processing...")
            delayed_writes = make_nc_files_optimized(
                files_dict, Dateswanted, Dayswanted, FPout, prefix_out
            )
            elapsed_time = time.time() - start_time
            print(f" executed in {elapsed_time} seconds")
        else:
            print("in here!!")
            divided_lists = divide_datetime_index(Dayswantedtot)
            print("divided lists")

            t2m_list = []  # WEC
            z_list = []  # WEC
            ub_list = []  # WEC
            vb_list = []  # WEC
            tb_list = []  # WEC
            qb_list = []  # WEC

            for dd in divided_lists:
                print("here we go")
                strtd = str(dd[0])[:10]
                endd = str(dd[-1])[:10]
                endd = increment_date_by_one_day(endd)
                print("doing files:", strtd, endd)
                start_time = time.time()  # Record the start time
                Dayswanted = pd.date_range(
                    start=strtd, end=endd, freq=str(interval_hours) + "D"
                )
                Dateswanted = pd.date_range(
                    start=strtd, end=endd, freq=str(interval_hours) + "h"
                )
                Static_zheight = xr.open_dataset(
                    "/glade/u/home/wchapman/RegriddERA5_CAMFV/static_operation_ERA5_zhght.nc"
                )
                files_dict = fp_dates_wanted(Dateswanted)
                # make the files:
                print("...starting processing...")
                delayed_writes, created_files = make_nc_files_optimized(
                    files_dict, Dateswanted, Dayswanted, FPout, prefix_out, levels
                )
                all_files.append(created_files)
                elapsed_time = time.time() - start_time
                print(f" phase executed in {elapsed_time} seconds")
                t2m_list.append(files_dict["t2m"])  # WEC
                z_list.append(files_dict["z"])  # WEC
                ub_list.append(files_dict["ub"])  # WEC
                vb_list.append(files_dict["vb"])  # WEC
                tb_list.append(files_dict["tb"])  # WEC
                qb_list.append(files_dict["qb"])  # WEC

        all_files = flatten_list(all_files)
        all_files.pop()
        print("...creating monthly files...")
        print("these are all the files we created together: ", all_files)

        delayed_writes = []
        for yryr in np.arange(1979, 2100):
            yryrstr = str(np.char.zfill(str(yryr), 4))
            for momo in np.arange(1, 13):
                start_time = time.time()  # Record the start time
                momostr = str(np.char.zfill(str(momo), 2))
                # Get a list of file paths
                pattern = yryrstr + momostr
                matching_strings = find_strings_by_pattern(all_files, pattern)

                if len(matching_strings) == 0:
                    continue
                else:
                    print("matched on:", pattern)
                    print(matching_strings)

                outtot = FPout + "ERA5_compiled." + yryrstr + momostr + ".nc"

                if os.path.exists(outtot):
                    sizefile = os.path.getsize(outtot)
                    if sizefile > 0:  #!!!! set some threshold here!!!!
                        print("file ", outtot, " already in memory")
                        continue

                DSall = xr.open_mfdataset(matching_strings, parallel=True)
                print("loaded")

                # delayed_write_uvtq = delayed(DSall.squeeze().to_netcdf)(outtot)
                delayed_write_uvtq = delayed(write_to_netcdf)(DSall.squeeze(), outtot)
                # delayed_writes.append(delayed_write_uvtq)
                elapsed_time = time.time() - start_time
                print(f" phase executed in {elapsed_time} seconds")

        print("...writing monthly files...")
        with ProgressBar():
            delayed_writes = list(dask.compute(*delayed_writes))

        print("zarr-ifying the files:")
        files_ = find_staged_files(start_date, end_date)
        DS = xr.open_mfdataset(files_, parallel=True)

        ##Add Surface and UL variables....
        # in this block.
        # /glade/campaign/collections/rda/data/ds633.0/
        ##
        print("opened")
        DS = DS.chunk({"time": 10})
        print("chunked")
        print("send to zarr")
        yrz = start_date[:4]
        DS.sel(time=slice(start_date, end_date_minus1)).to_zarr(AFout)
        print("...finished...")

    if "client" in locals():
        client.shutdown()
        print("...shutdown client...")
    else:
        print("client does not exist yet")

    if remove_dask_worker_scripts:
        print("...removing dask workers...")
        fns_rm = sorted(glob.glob("./dask-worker*"))
        print(len(fns_rm))
        for fn in fns_rm:
            os.remove(fn)
