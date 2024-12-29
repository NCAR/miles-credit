'''
A collection of functions used for the verification of CREDIT project
-----------------------------------------------------------------------
Content:
    - create_dir
    - get_nc_files
    - ds_subset_everything
    - process_file_group
    - process_file_group_safe
        
    
Yingkai Sha
ksha@ucar.edu
'''

import os
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

def create_dir(path):
    """
    Create dir if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def lead_to_index(leads_exist, leads_verif):
    '''
    Check if a list of verified lead time exists in a list of all available lead times
    return the index of the verified lead time.
    '''
    ind_lead = []
    for lead_verif in leads_verif:
        if lead_verif in leads_exist:
            ind_lead.append(leads_exist.index(lead_verif))
        else:
            print('lead time {}h is not covered'.format(lead_verif))
            raise
    return ind_lead

def get_forward_data_netCDF4(filename) -> xr.DataArray:
    """Lazily opens netCDF4 files.
    """
    dataset = xr.open_dataset(filename)
    return dataset

def get_forward_data(filename) -> xr.DataArray:
    '''
    Lazily opens the Zarr store on gladefilesystem.
    '''
    dataset = xr.open_zarr(filename, consolidated=True)
    return dataset
    

def get_nc_files(base_dir, folder_prefix='%Y-%m-%dT%HZ'):
    """
    Get a list of lists containing paths to NetCDF files in each subdirectory of the base directory,
    sorted by date.

    output = [
        [lead_time1, lead_time2, ...], # initialization time 1
        [lead_time1, lead_time2, ...], # initialization time 2
        [lead_time1, lead_time2, ...], # initialization time 3
        ]
    
    args:
        base_dir: the storage place of individual forecasts
        folder_prefix: the prefix of sub-folders in terms of their datetime info
    
    """
    
    all_files_list = []
    
    # Collect directories and sort them by date
    for parent_dir, sub_dirs, files in os.walk(base_dir):
        
        # Sort directories by date extracted from their names
        sorted_sub_dirs = sorted(sub_dirs, key=lambda x: datetime.strptime(x, folder_prefix))

        # loop through initialization times
        for dir_ini_time in sorted_sub_dirs:

            # get nc file path and glob
            dir_path = os.path.join(parent_dir, dir_ini_time)
            nc_files = sorted(glob(os.path.join(dir_path, '*.nc')))

            # send glob results to a list
            if nc_files:
                all_files_list.append(nc_files)
            else:
                print('folder {} does not have nc files'.format(dir_path))
                raise
    
    return all_files_list

def ds_subset_everything(ds, variables_levels, time_intervals=None):
    """
    Subset a given xarray.Dataset, preserve specific variable/level/time

    args:
        ds: xarray.Dataset
        variables_levels: a dictionary that looks like this
            variables_levels = {
                                'V500': None,  # Keep all levels
                                'SP': None,
                                't2m': None,
                                'U': [14, 10, 5],  
                                'V': [14, 10, 5], 
                            }
            Leave level as None if (1) keeping all levels or (2) the variable does not have level dim
        time_intervals: a time slice that applies to each variable (optional) 
    """
    # allocate the output xarray.Dataset
    ds_selected = xr.Dataset()

    # loop through the subset info
    for var, levels in variables_levels.items():
        if var in ds:
            if levels is None:
                # keep all level
                ds_selected[var] = ds[var]
            else:
                # subset levels
                ds_selected[var] = ds[var].sel(level=levels)
        else:
            print('variable {} does not exist in the given xarray.Dataset'.format(var))

    # optional time subset
    if time_intervals is not None:
        ds_selected = ds_selected.isel(time=time_intervals)
        
    return ds_selected

def process_file_group(file_list, output_dir, 
                       variables_levels, 
                       time_intervals=None,
                       check_fcst_hour=True,
                       size_thres=4000000000):
    '''
    Process a group of netCDF4 files, combining them into a single netCDF4 file.

    Args:
        file_list: List of NetCDF filenames.
        output_dir: Directory to save the combined NetCDF4 file.
        variables_levels, time_intervals: Parameters for subsetting the dataset.
    '''
    subdir_name = os.path.basename(os.path.dirname(file_list[0]))
    print(f"Processing subdirectory: {subdir_name}")
    
    # Use folder name as output file name
    output_file = os.path.join(output_dir, f'{subdir_name}.nc')
    print(f'Output name: {output_file}')

    # ==================================================================================================== #
    # Check if the output file already exists and is valid
    if os.path.exists(output_file):
        try:
            # Attempt to open the existing file to check for corruption
            test_data = xr.open_dataset(output_file)
            print(f"File {output_file} is valid ... move to file size checks.")
            # get fcst_hour
            if check_fcst_hour:
                fcst_hour = test_data.forecast_hour.values
            test_data.close()

            # Now check the file size against the threshold
            if os.path.getsize(output_file) > size_thres:
                print(f"{output_file} matches the size threshold.")

                if check_fcst_hour:
                    # Now move to forecast hour check
                    spacing = np.diff(fcst_hour)
                    is_equally_spaced = np.allclose(spacing, spacing[0])
                    is_increasing = np.all(spacing > 0)
                    if is_equally_spaced and is_increasing:
                        # True: no need to re-do preprocess
                        print(f"{output_file} have the correct forecast hours, Skip")
                        return True
                    else:
                        print(f"{output_file} have wrong forecast hours, It will be removed")
                        os.remove(output_file)
                else:
                    # True: no need to re-do preprocess
                    print(f"{output_file} have the correct size, Skip")
                    return True
                
        except Exception as e:
            # If the file is corrupted, remove it
            print(f"Corrupted file: {output_file} detected. Error: {e}. It will be removed.")
            os.remove(output_file)

    # ==================================================================================================== #
    # Open multiple NetCDF files as a single dataset and subset to specified variables/levels/time
    
    print('A new file will be created ...')
    
    try:
        if variables_levels is not None:
            ds = xr.open_mfdataset(
                file_list,
                combine='by_coords',
                preprocess=lambda ds: ds_subset_everything(ds, variables_levels, time_intervals),
                parallel=True,
                lock=False
            )
        else:
            ds = xr.open_mfdataset(
                file_list,
                combine='by_coords',
                parallel=True,
                lock=False
            )
        
        # Ensure coordinate names are correct
        if 'datetime' in ds.coords:
            ds = ds.rename({'datetime': 'time'})
        if 'lat' in ds.coords:
            ds = ds.rename({'lat': 'latitude'})
        if 'lon' in ds.coords:
            ds = ds.rename({'lon': 'longitude'})
            
        # Save the dataset using NetCDF4 format
        # encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        ds.to_netcdf(output_file, format='NETCDF4', mode='w')
        ds.close()
        print(f"Successfully saved combined dataset to {output_file}")
    
    except Exception as e:
        # Catch any errors in file creation or dataset saving
        print(f"File creation error: {e}. Incomplete file will be removed.")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Incomplete file {output_file} has been removed.")
            except Exception as remove_error:
                print(f"Error removing incomplete file {output_file}: {remove_error}")
    finally:
        # Ensure dataset is closed to avoid memory leaks
        try:
            ds.close()
        except NameError:
            # ds was never created because of an error
            print(f"Dataset object not created. Skipping close.")
        except Exception as e:
            print(f"Error closing dataset: {e}")
    return False


def process_file_group_safe(file_list, output_dir, 
                            variables_levels, 
                            time_intervals=None,
                            check_fcst_hour=True,
                            size_thres=4000000000):
    '''
    Process a group of netCDF4 files, combining them into a single netCDF4 file.
    This function mutli-thread safe, It uses xr.open_dataset on individual files

    Args:
        file_list: List of NetCDF filenames.
        output_dir: Directory to save the combined NetCDF4 file.
        variables_levels: Parameters for subsetting the dataset.
        time_intervals: Time intervals for subsetting.
        check_fcst_hour: Whether to check forecast hours for correctness.
        size_thres: File size threshold for skipping processing.
    '''
    
    subdir_name = os.path.basename(os.path.dirname(file_list[0]))
    print(f"Processing subdirectory: {subdir_name}")
    
    # Use folder name as output file name
    output_file = os.path.join(output_dir, f'{subdir_name}.nc')
    print(f'Output name: {output_file}')

    # ==================================================================================================== #
    # Check if the output file already exists and is valid
    if os.path.exists(output_file):
        try:
            # Attempt to open the existing file to check for corruption
            test_data = xr.open_dataset(output_file)
            print(f"File {output_file} is valid ... moving to file size checks.")
            # Get forecast_hour values
            if check_fcst_hour:
                fcst_hour = test_data.forecast_hour.values
            test_data.close()

            # Now check the file size against the threshold
            if os.path.getsize(output_file) > size_thres:
                print(f"{output_file} matches the size threshold.")

                if check_fcst_hour:
                    # Now move to forecast hour check
                    spacing = np.diff(fcst_hour)
                    is_equally_spaced = np.allclose(spacing, spacing[0])
                    is_increasing = np.all(spacing > 0)
                    if is_equally_spaced and is_increasing:
                        # No need to re-do preprocess
                        print(f"{output_file} has the correct forecast hours. Skipping.")
                        return True
                    else:
                        print(f"{output_file} has incorrect forecast hours. It will be removed.")
                        os.remove(output_file)
                else:
                    # No need to re-do preprocess
                    print(f"{output_file} has the correct size. Skipping.")
                    return True
                
        except Exception as e:
            # If the file is corrupted, remove it
            print(f"Corrupted file: {output_file} detected. Error: {e}. It will be removed.")
            os.remove(output_file)

    # ==================================================================================================== #
    # Open multiple NetCDF files individually and concatenate them along 'time' dimension
    
    print('A new file will be created ...')
    
    try:
        ds_list = []
        for file in file_list:
            # Open each dataset individually
            if variables_levels is not None:
                ds = xr.open_dataset(file)
                # Apply preprocessing if needed
                ds = ds_subset_everything(ds, variables_levels, time_intervals)
            else:
                ds = xr.open_dataset(file)
            
            # Append to the list
            ds_list.append(ds)
        
        # Concatenate datasets along 'time' dimension
        ds = xr.concat(ds_list, dim='time')
        
        # Ensure coordinate names are correct
        coord_mapping = {'datetime': 'time', 'lat': 'latitude', 'lon': 'longitude'}
        for old_name, new_name in coord_mapping.items():
            if old_name in ds.coords:
                ds = ds.rename({old_name: new_name})
                
        # Save the dataset using NetCDF4 format
        # encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        ds.to_netcdf(output_file, compute=True, format='NETCDF4', mode='w')  # , encoding=encoding
        ds.close()
        print(f"Successfully saved combined dataset to {output_file}")
        
        # Close individual datasets
        for ds_single in ds_list:
            ds_single.close()
    
    except Exception as e:
        # Catch any errors in file creation or dataset saving
        print(f"File creation error: {e}. Incomplete file will be removed.")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Incomplete file {output_file} has been removed.")
            except Exception as remove_error:
                print(f"Error removing incomplete file {output_file}: {remove_error}")
    finally:
        # Ensure dataset is closed to avoid memory leaks
        try:
            ds.close()
        except NameError:
            # ds was never created because of an error
            print(f"Dataset object not created. Skipping close.")
        except Exception as e:
            print(f"Error closing dataset: {e}")
    return False
