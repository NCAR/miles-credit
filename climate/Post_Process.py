import os
import shutil
import xarray as xr
import glob
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import yaml
import pandas as pd

from multiprocessing import Pool, cpu_count
from functools import partial

import xarray as xr
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os


# example:
# python Post_Process.py /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191/model_multi_WxFormer.yml 1D --variables U V T Qtot PS PRECT TREFHT --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHT

def extract_time_single(filename):
    """Extract time from a single file."""
    with xr.open_dataset(filename, decode_times=True, chunks={'time': 1}) as ds:
        return ds['time'].values[0]

def method1_parallel_process(file_list):
    """Use ProcessPoolExecutor for parallel processing."""
    n_workers = os.cpu_count() // 2  # Use half of available CPUs
    t_array = np.empty(len(file_list), dtype='datetime64[ns]')
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(extract_time_single, file_list))
    
    return np.array(results)

def rescale_file(fn, mean_, std_):
    """Helper function to rescale a single file"""
    try:
        DSfirst = xr.open_dataset(fn)
        
        # Check if already rescaled
        if DSfirst.attrs.get('scaling') == 'rescaled':
            print(f"File {fn} already rescaled. Skipping...")
            DSfirst.close()
            return
            
        print(f"Rescaling {fn}...")
        
        for var in DSfirst.data_vars:
            if var in mean_ and var in std_:
                print(f"Rescaling variable {var}...")
                
                if 'level' in DSfirst[var].dims:
                    mean_broadcast = mean_[var].expand_dims({
                        'time': 1, 
                        'latitude': 192, 
                        'longitude': 288
                    }).transpose('time', 'level', 'latitude', 'longitude')
                    
                    std_broadcast = std_[var].expand_dims({
                        'time': 1, 
                        'latitude': 192, 
                        'longitude': 288
                    }).transpose('time', 'level', 'latitude', 'longitude')
                    
                    DSfirst[var][:] = ((DSfirst[var].values * std_broadcast.values) + 
                                     mean_broadcast.values)
                else:
                    DSfirst[var][:] = ((DSfirst[var] * std_[var]) + mean_[var]).values
            else:
                print(f"Variable {var} not found in scaling datasets. Skipping...")
        
        DSfirst.attrs['scaling'] = 'rescaled'
        
        # Save to temporary file and replace original
        temp_fn = fn + ".tmp.nc"
        DSfirst.to_netcdf(temp_fn)
        DSfirst.close()
        os.sync()
        os.replace(temp_fn, fn)
        
        print(f"File {fn} successfully rescaled and saved.\n")
        
    except Exception as e:
        print(f"Error processing file {fn}: {str(e)}")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_hours_noleap(start, hours):
    # Define the no-leap year structure
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Break down start date
    year, month, day, hour = start.year, start.month, start.day, start.hour
    
    # Calculate total days and hours
    total_hours = hour + hours
    days = total_hours // 24
    remaining_hours = total_hours % 24
    
    while days > 0:
        # Check if the day increment will overflow the month
        if day + 1 > days_in_month[month - 1]:
            day = 1  # Reset day
            month += 1  # Increment month
            
            if month > 12:
                month = 1
                year += 1
        else:
            day += 1
        
        days -= 1

    return datetime(year, month, day, remaining_hours)

def post_process(conf, avg_window, variables, name_string='',reset_times=True, 
                 dask_do=False, monthly=False, rescale_it=False, n_processes=None):
    """
    Post-process NetCDF files by padding numerical suffixes, and optionally average variables over a specified time window.

    Parameters:
    source_dir (str): Directory containing .nc files to process.
    avg_window (str): Averaging window (e.g., '1D' for daily). Use 'None' to skip averaging.
    variables (list): List of variable names to average. If empty, all variables will be used.
    """
    # Open and load the YAML file
    with open(conf, 'r') as file:
        config = yaml.safe_load(file)
    
    # Print the contents
    conf_pred = config['predict']
    conf_pred_for = conf_pred['forecasts']
    conf_data = config['data']

    # Extract values from the dictionary
    year_ = conf_pred_for.get('start_year', 1970)  # Default to 1970 if missing
    month_ = conf_pred_for.get('start_month', 1)   # Default to January
    day_ = conf_pred_for.get('start_day', 1)       # Default to 1st
    hours_ = conf_pred_for.get('start_hours', [0]) # Default to [0]
    
    # Ensure hours is a list and take the first value
    hour_ = hours_[0] if isinstance(hours_, list) and hours_ else 0
    
    # Create datetime object
    init_date_obj = datetime(year_, month_, day_, hour_)
    
    # Convert to string
    init_date_str = init_date_obj.strftime('%Y-%m-%d %H:%M:%S')
    print('init time is: ', init_date_str)
    init_date_strz = init_date_obj.strftime('%Y-%m-%dT%HZ')
    source_dir = f'{conf_pred["save_forecast"]}/{init_date_strz}/'
    print('source dir: ', source_dir)

    # Get all .nc files in the directory
    FNS = sorted(glob.glob(f'{source_dir}/pred*.nc'))
            
    for fn in FNS:
        # Extract the number at the end of the filename using regex
        match = re.search(r'_(\d+)\.nc$', fn)
        if match:
            num_part = match.group(1)
            # Pad the number to 8 digits with leading zeros
            padded_num = num_part.zfill(8)
            # Create new filename by replacing the old number with the padded one
            fn_new = re.sub(r'_(\d+)\.nc$', f'_{padded_num}.nc', fn)
            # Rename the file if the new name differs from the old one
            if fn != fn_new:
                shutil.move(fn, fn_new)
                print(f'Renamed {fn} to {fn_new}')
    print('... done renaming files ...')
    # Perform averaging if avg_window is specified


    mean_ = xr.open_dataset(conf_data['mean_path'])
    std_ = xr.open_dataset(conf_data['std_path'])
    FNS = sorted(glob.glob(f'{source_dir}/pred*.nc'))
    if not FNS:
        raise FileNotFoundError("No prediction files found in the source directory.")

    if rescale_it:
        # Load scaling datasets
        mean_ = xr.open_dataset(conf_data['mean_path'])
        std_ = xr.open_dataset(conf_data['std_path'])
        FNS = sorted(glob.glob(f'{source_dir}/pred*.nc'))
        
        if not FNS:
            raise FileNotFoundError("No prediction files found in the source directory.")
            
        # Determine number of processes
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)  # Leave one CPU free
        
        print(f"Starting parallel processing with {n_processes} processes...")
        
        # Create partial function with fixed arguments
        rescale_file_partial = partial(rescale_file, mean_=mean_, std_=std_)
        
        # Create process pool and map the work
        with Pool(processes=n_processes) as pool:
            pool.map(rescale_file_partial, FNS)
            
        print("Parallel rescaling completed.")
    else:
        print('...Not rescaling...')

    if reset_times:
        print('... resetting times ...')
        last_part = os.path.basename(os.path.normpath(source_dir))
        start_date = datetime.strptime(last_part, '%Y-%m-%dT%HZ')
        FNS = sorted(glob.glob(f'{source_dir}/pred*.nc'))
        all_dates = []
        indices = []
        for ee, fn in enumerate(FNS):
            if ((ee+1)%100==0):
                print(f'resetting time {fn}')
            hrs_ = int(fn.split('.nc')[0].split('_')[-1])
            new_date = add_hours_noleap(start_date,hrs_)
            all_dates.append(new_date)
            new_date = xr.DataArray([new_date], dims='time')
            DS = xr.open_dataset(fn)
            DS = DS.assign_coords(time=new_date)
            indices.append(ee)
            temp_fn = fn + ".tmp.nc"
            DS.to_netcdf(temp_fn)
            DS.close()
            os.sync()  # Force sync to disk
            os.replace(temp_fn, fn)  # Atomically replaces the original file
        
        # Create DataArrays
        time_array = xr.DataArray(all_dates, dims='time', name='time')
        index_array = xr.DataArray(indices, dims='time', name='index')
        
        # Combine into a Dataset
        DS_index = xr.Dataset(
            {
                "index": index_array,  # Index as a data variable
            },
            coords={
                "time": time_array,  # Time as a coordinate
            }
        )
        
        # Print the resulting Dataset
        DS_index.to_netcdf(f'{source_dir}/datetimes_in_files.nc')
    print('... started averaging window ...')

    collect_times = True
    if collect_times:
        if os.path.exists(f'{source_dir}/datetimes_in_files.nc'):
            print('...datetime file already exists...')
        else:
            FNS = sorted(glob.glob(f'{source_dir}/pred*.nc'))

            #look in Future_SST.ipynb file...
            print('... creating datetime file ...')
            times_array = method1_parallel_process(FNS)
            indices = np.arange(len(FNS))
            index_array = xr.DataArray(indices, dims='time', name='index')
                    
            # Combine into a Dataset
            DS_index = xr.Dataset(
                {
                    "index": index_array,  # Index as a data variable
                },
                coords={
                    "time": times_array,  # Time as a coordinate
                }
            )
            
            # Print the resulting Dataset
            DS_index.to_netcdf(f'{source_dir}/datetimes_in_files.nc')
    
    if avg_window:
        avg_dir = os.path.join(source_dir, avg_window)
        os.makedirs(avg_dir, exist_ok=True)
        FNS = sorted(glob.glob(f'{source_dir}/pred*.nc'))

        DSfirst = xr.open_dataset(FNS[0])
        DSlast = xr.open_dataset(FNS[-1])
        
        # Extract start and end year/month
        start_year = DSfirst['time.year'].values[0]
        start_month = DSfirst['time.month'].values[0]
        end_year = DSlast['time.year'].values[0]
        end_month = DSlast['time.month'].values[0]
        
        # Create a list of months between start and end dates
        date_range = pd.date_range(start=f'{start_year}-{start_month:02d}', 
                                   end=f'{end_year}-{end_month:02d}', 
                                   freq='MS')  # 'MS' is Month Start
        
        # Convert to list of year-month strings
        year_month_list = date_range.strftime('%Y-%m').tolist()

        DS_index = xr.open_dataset(f'{source_dir}/datetimes_in_files.nc')


        for yml in year_month_list:
            avg_file = os.path.join(avg_dir, f'averaged_{name_string}_{yml}_{avg_window}.nc')
            
            if os.path.exists(avg_file):
                print(f'{avg_file} already exists... skipping')
                continue
            else:
                print('creating avg file!')
                
            Fil_idx_vals = DS_index.sel(time=yml)['index'].values
            selected_files = [FNS[i] for i in Fil_idx_vals]
            # DS_months = xr.open_mfdataset(selected_files, parallel=True, chunks={'time': 'auto'}, engine='netcdf4')
            
            datasets = []
            for ee,fn in enumerate(selected_files ):
                try:
                    ds = xr.open_dataset(fn, chunks={'time': 'auto'}, engine='netcdf4')
                    # Select variables during file opening if specified
                    if variables:
                        ds = ds[variables]
                    datasets.append(ds)
                except Exception as e:
                    print(f'Error opening {fn}: {e}')

            print('... concatenating files...')
            ds = xr.concat(datasets, dim='time')
            print('... files concatenated....')
            print(ds)
            #Use all variables if none are specified during concatenation
            if not variables:
                variables = list(ds.data_vars)
                print(f'No variables specified, using all available: {variables}')
            print('... resampling ....')
            # Resample the dataset to average over the specified time window
            ds_avg = ds.resample(time=avg_window).mean()
            print('... resampling ....')

            # Save the averaged dataset to a new NetCDF file in the averaging directory
            ds_avg.to_netcdf(avg_file)
            print(f'Saved averaged data to {avg_file}')
    return ds



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Post-process NetCDF files and perform averaging.")
    
    parser.add_argument("conf", type=str, help="Path to YAML configuration file.")
    parser.add_argument("avg_window", type=str, help="Averaging window (e.g., '1D'). Use 'None' to skip.")
    parser.add_argument("--name_string", type=str, default='', help="Name identifier for output files.")
    parser.add_argument("--variables", nargs='+', help="Variables to process (e.g., TREFHT PRECT). Default is all.")
    parser.add_argument("--reset_times", type=str2bool, nargs='?', const=True, default=False,
                    help="Reset time coordinates. Use 'True' or 'False'. Default is False.")
    parser.add_argument("--dask_do", type=str2bool, nargs='?', const=True, default=False,
                    help="Enable dask parallel processing. Use 'True' or 'False'. Default is False.")
    parser.add_argument("--rescale_it", type=str2bool, nargs='?', const=True, default=False,
                    help="Enable dask parallel processing. Use 'True' or 'False'. Default is False.")
    parser.add_argument("--n_processes", type=int, default=1, help="number of parallel processes")
    
    args = parser.parse_args()

    try:
        post_process(
            args.conf,
            args.avg_window,
            variables=args.variables,
            name_string=args.name_string,
            reset_times=args.reset_times,
            dask_do=args.dask_do,
            rescale_it=args.rescale_it,
            n_processes=args.n_processes
        )
        print("Processing complete.")
    except Exception as e:
        print(f"Error: {e}")
