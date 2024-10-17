'''
data.py 
-------------------------------------------------------
Content:
    - generate_datetime(start_time, end_time, interval_hr)
    - hour_to_nanoseconds(input_hr)
    - nanoseconds_to_year(nanoseconds_value)
    - extract_month_day_hour(dates)
    - find_common_indices(list1, list2)
    - concat_and_reshape(x1, x2)
    - reshape_only(x1)
    - get_forward_data(filename)
    - drop_var_from_dataset()
    - ERA5_and_Forcing_Dataset(torch.utils.data.Dataset)
    - Predict_Dataset(torch.utils.data.IterableDataset)
'''

# system tools
import os
from typing import Optional, Callable, TypedDict, Union

# data utils
import datetime
import numpy as np
import pandas as pd
import xarray as xr

# Pytorch utils
import torch
import torch.utils.data
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

#
Array = Union[np.ndarray, xr.DataArray]
IMAGE_ATTR_NAMES = ('historical_ERA5_images', 'target_ERA5_images')

def generate_datetime(start_time, end_time, interval_hr):
    '''
    Generate a list of datetime.datetime based on stat, end times, and hour interval. 
    '''
    # Define the time interval (e.g., every hour)
    interval = datetime.timedelta(hours=interval_hr)
    
    # Generate the list of datetime objects
    datetime_list = []
    current_time = start_time
    while current_time <= end_time:
        datetime_list.append(current_time)
        current_time += interval
    return datetime_list

def hour_to_nanoseconds(input_hr):
    '''
    Convert hour to nanoseconds
    '''
    # hr * min_per_hr * sec_per_min * nanosec_per_sec
    return input_hr*60 * 60 * 1000000000

def nanoseconds_to_year(nanoseconds_value):
    '''
    Given datetime info as nanoseconds, compute which year it belongs to.
    '''
    return np.datetime64(nanoseconds_value, 'ns').astype('datetime64[Y]').astype(int) + 1970

def extract_month_day_hour(dates):
    '''
    Given an 1-d array of np.datatime64[ns], extract their mon, day, hr into a zipped list
    '''
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1
    days = (dates - dates.astype('datetime64[M]') + 1).astype('timedelta64[D]').astype(int)
    hours = dates.astype('datetime64[h]').astype(int) % 24
    return list(zip(months, days, hours))

def find_common_indices(list1, list2):
    '''
    find indices of common elements between two lists 
    '''
    # Find common elements
    common_elements = set(list1).intersection(set(list2))
    
    # Find indices of common elements in both lists
    indices_list1 = [i for i, x in enumerate(list1) if x in common_elements]
    indices_list2 = [i for i, x in enumerate(list2) if x in common_elements]
    
    return indices_list1, indices_list2
    
def concat_and_reshape(x1, x2):
    '''
    Flattening the "level" coordinate of upper-air variables and concatenate it will surface variables. 
    '''
    x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5])
    x_concat = torch.cat((x1, x2), dim=2)
    return x_concat.permute(0, 2, 1, 3, 4)

def reshape_only(x1):
    '''
    Flattening the "level" coordinate of upper-air variables.
    As in "concat_and_reshape", but no concat
    '''
    x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5])
    return x1.permute(0, 2, 1, 3, 4)

def get_forward_data(filename) -> xr.DataArray:
    '''
    Check nc vs. zarr files
    open file as xr.Dataset
    '''
    if filename[-3:] == '.nc' or filename[-4:] == '.nc4':
        dataset = xr.open_dataset(filename)
    else:
        dataset = xr.open_zarr(filename, consolidated=True)
    return dataset


class Sample(TypedDict):
    """Simple class for structuring data for the ML model.

    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple,
    which would provide runtime checks, but the deal-breaker with Tuples is that they're immutable
    so we cannot change the values in the transforms.
    """
    # IMAGES
    # Shape: batch_size, seq_length, lat, lon, lev
    historical_ERA5_images: Array
    target_ERA5_images: Array

    # METADATA
    datetime_index: Array


# @dataclass
# class Reshape_Data():
#     size: int = 128  #: Size of the cropped image.

#     def __call__(self, sample: Sample) -> Sample:
#         for attr_name in IMAGE_ATTR_NAMES:
#             image = sample[attr_name]
#             # TODO: Random crop!
#             cropped_image = image[..., :self.size, :self.size]
#             sample[attr_name] = cropped_image
#         return sample


# class CheckForBadData():
#     def __call__(self, sample: Sample) -> Sample:
#         for attr_name in IMAGE_ATTR_NAMES:
#             image = sample[attr_name]
#             if np.any(image < 0):
#                 raise ValueError(f'\n{attr_name} has negative values at {image.time.values}')
#         return sample

# class Segment(NamedTuple):
#     """Represents the start and end indicies of a segment of contiguous samples."""
#     start: int
#     end: int


# def get_contiguous_segments(dt_index: pd.DatetimeIndex, min_timesteps: int, max_gap: pd.Timedelta) -> Iterable[Segment]:
#     """Chunk datetime index into contiguous segments, each at least min_timesteps long.

#     max_gap defines the threshold for what constitutes a 'gap' between contiguous segments.

#     Throw away any timesteps in a sequence shorter than min_timesteps long.
#     """
#     gap_mask = np.diff(dt_index) > max_gap
#     gap_indices = np.argwhere(gap_mask)[:, 0]

#     # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
#     # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
#     segment_boundaries = gap_indices + 1

#     # Capture the last segment of dt_index.
#     segment_boundaries = np.concatenate((segment_boundaries, [len(dt_index)]))

#     segments = []
#     start_i = 0
#     for end_i in segment_boundaries:
#         n_timesteps = end_i - start_i
#         if n_timesteps >= min_timesteps:
#             segment = Segment(start=start_i, end=end_i)
#             segments.append(segment)
#         start_i = end_i

#     return segments

def flatten_list(list_of_lists):
    """
    Flatten a list of lists.

    Parameters:
    - list_of_lists (list): A list containing sublists.

    Returns:
    - flattened_list (list): A flattened list containing all elements from sublists.
    """
    return [item for sublist in list_of_lists for item in sublist]


def generate_integer_list_around(number, spacing=10):
    """
    Generate a list of integers on either side of a given number with a specified spacing.

    Parameters:
    - number (int): The central number around which the list is generated.
    - spacing (int): The spacing between consecutive integers in the list. Default is 10.

    Returns:
    - integer_list (list): List of integers on either side of the given number.
    """
    lower_limit = number - spacing
    upper_limit = number + spacing + 1  # Adding 1 to include the upper limit
    integer_list = list(range(lower_limit, upper_limit))

    return integer_list


def find_key_for_number(input_number, data_dict):
    """
    Find the key in the dictionary based on the given number.

    Parameters:
    - input_number (int): The number to search for in the dictionary.
    - data_dict (dict): The dictionary with keys and corresponding value lists.

    Returns:
    - key_found (str): The key in the dictionary where the input number falls within the specified range.
    """
    for key, value_list in data_dict.items():
        if value_list[1] <= input_number <= value_list[2]:
            return key

    # Return None if the number is not within any range
    return None

def drop_var_from_dataset(xarray_dataset, varname_keep):
    '''
    Preserve a given set of variables from an xarray.Dataset, and drop the rest.
    It will raise error if `varname_key` is missing from `xarray_dataset`
    '''
    varname_all = list(xarray_dataset.keys())

    for varname in varname_all:
        if varname not in varname_keep:
            xarray_dataset = xarray_dataset.drop_vars(varname)

    varname_clean = list(xarray_dataset.keys())
    
    varname_diff = list(set(varname_keep) - set(varname_clean))
    assert len(varname_diff)==0, 'Variable name: {} missing'.format(varname_diff) 
    
    return xarray_dataset
    

class ERA5_and_Forcing_Dataset(torch.utils.data.Dataset):
    '''
    A Pytorch Dataset class that works on:
        - upper-air variables (time, level, lat, lon)
        - surface variables (time, lat, lon)
        - dynamic forcing variables (time, lat, lon)
        - foring variables (time, lat, lon)
        - diagnostic variables (time, lat, lon)
        - static variables (lat, lon)
    '''

    def __init__(
        self,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames,
        filename_surface=None,
        filename_dyn_forcing=None,
        filename_forcing=None,
        filename_static=None,
        filename_diagnostic=None,
        history_len=2,
        forecast_len=0,
        transform=None,
        seed=42,
        skip_periods=None,
        one_shot=None,
        max_forecast_len=None,
        sst_forcing=None,
    ):

        '''
        Initialize the ERA5_and_Forcing_Dataset

        Parameters:
        - varname_upper_air (list): List of upper air variable names.
        - varname_surface (list): List of surface variable names.
        - varname_dyn_forcing (list): List of dynamic forcing variable names.
        - varname_forcing (list): List of forcing variable names.
        - varname_static (list): List of static variable names.
        - varname_diagnostic (list): List of diagnostic variable names.
        - filenames (list): List of filenames for upper air data.
        - filename_surface (list, optional): List of filenames for surface data.
        - filename_dyn_forcing (list, optional): List of filenames for dynamic forcing data.
        - filename_forcing (str, optional): Filename for forcing data.
        - filename_static (str, optional): Filename for static data.
        - filename_diagnostic (list, optional): List of filenames for diagnostic data.
        - history_len (int, optional): Length of the history sequence. Default is 2.
        - forecast_len (int, optional): Length of the forecast sequence. Default is 0.
        - transform (callable, optional): Transformation function to apply to the data.
        - seed (int, optional): Random seed for reproducibility. Default is 42.
        - skip_periods (int, optional): Number of periods to skip between samples.
        - one_shot(bool, optional): Whether to return all states or just
                                    the final state of the training target. Default is None
        - max_forecast_len (int, optional): Maximum length of the forecast sequence.
        - shuffle (bool, optional): Whether to shuffle the data. Default is True.

        Returns:
        - sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.
        '''

        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform

        # skip periods
        self.skip_periods = skip_periods
        if self.skip_periods is None:
            self.skip_periods = 1

        # one shot option
        self.one_shot = one_shot

        # total number of needed forecast lead times 
        self.total_seq_len = self.history_len + self.forecast_len

        # set random seed
        self.rng = np.random.default_rng(seed=seed)

        # max possible forecast len
        self.max_forecast_len = max_forecast_len

        # sst forcing
        self.sst_forcing = sst_forcing
        
        # =================================================================== #
        # flags to determin if any of the [surface, dyn_forcing, diagnostics]
        # variable groups share the same file as upper air variables
        flag_share_surf = False
        flag_share_dyn = False
        flag_share_diag = False
        
        all_files = []
        filenames = sorted(filenames)

        # ------------------------------------------------------------------ #
        # blocks that can handle no-sharing (each group has it own file)
        ## surface
        if filename_surface is not None:
            surface_files = []
            filename_surface = sorted(filename_surface)
            
            if filenames == filename_surface:
                flag_share_surf = True
            else:
                for fn in filename_surface:
    
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_surf = drop_var_from_dataset(ds, varname_surface)
                    surface_files.append(ds_surf)
                    
                self.surface_files = surface_files
        else:
            self.surface_files = False
            
        ## dynamic forcing
        if filename_dyn_forcing is not None:
            dyn_forcing_files = []
            filename_dyn_forcing = sorted(filename_dyn_forcing)
            
            if filenames == filename_dyn_forcing:
                flag_share_dyn = True
            else:
                for fn in filename_dyn_forcing:
    
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_dyn = drop_var_from_dataset(ds, varname_dyn_forcing)
                    dyn_forcing_files.append(ds_dyn)
    
                self.dyn_forcing_files = dyn_forcing_files
        else:
            self.dyn_forcing_files = False

        ## diagnostics
        if filename_diagnostic is not None:
            diagnostic_files = []
            filename_diagnostic = sorted(filename_diagnostic)

            if filenames == filename_diagnostic:
                flag_share_diag = True
            else:
                for fn in filename_diagnostic:
    
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_diag = drop_var_from_dataset(ds, varname_diagnostic)
                    diagnostic_files.append(ds_diag)
                    
                self.diagnostic_files = diagnostic_files
        else:
            self.diagnostic_files = False

        # ------------------------------------------------------------------ #
        # blocks that can handle file sharing (share with upper air file)
        for fn in filenames:
            # drop variables if they are not in the config
            ds = get_forward_data(filename=fn)
            ds_upper = drop_var_from_dataset(ds, varname_upper_air)
            
            if flag_share_surf:
                ds_surf = drop_var_from_dataset(ds, varname_surface)
                surface_files.append(ds_surf)

            if flag_share_dyn:
                ds_dyn = drop_var_from_dataset(ds, varname_dyn_forcing)
                dyn_forcing_files.append(ds_dyn)

            if flag_share_diag:
                ds_diag = drop_var_from_dataset(ds, varname_diagnostic)
                diagnostic_files.append(ds_diag)
                
            all_files.append(ds_upper)
            
        self.all_files = all_files
        
        if flag_share_surf:
            self.surface_files = surface_files
        if flag_share_dyn:
            self.dyn_forcing_files = dyn_forcing_files
        if flag_share_diag:
            self.diagnostic_files = diagnostic_files

        # -------------------------------------------------------------------------- #
        # get sample indices from ERA5 upper-air files:
        ind_start = 0
        self.ERA5_indices = {} # <------ change
        for ind_file, ERA5_xarray in enumerate(self.all_files):
            # [number of samples, ind_start, ind_end]
            self.ERA5_indices[str(ind_file)] = [len(ERA5_xarray['time']),
                                                ind_start,
                                                ind_start + len(ERA5_xarray['time'])]
            ind_start += len(ERA5_xarray['time']) + 1
            
        # ======================================================== #
        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            ds = get_forward_data(filename_forcing)
            ds_forcing = drop_var_from_dataset(ds, varname_forcing).load() # <---- load in static

            self.xarray_forcing = ds_forcing
        else:
            self.xarray_forcing = False

        # ======================================================== #
        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            # drop variables if they are not in the config
            ds = get_forward_data(filename_static)
            ds_static = drop_var_from_dataset(ds, varname_static).load() # <---- load in static
            
            self.xarray_static = ds_static
        else:
            self.xarray_static = False

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += len(ERA5_xarray['time']) - self.total_seq_len + 1
        return total_len

    def __getitem__(self, index):
        # ========================================================================== #
        # cross-year indices --> the index of the year + indices within that year

        # select the ind_file based on the iter index 
        ind_file = find_key_for_number(index, self.ERA5_indices)

        # get the ind within the current file
        ind_start = self.ERA5_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        # handle out-of-bounds
        ind_largest = len(self.all_files[int(ind_file)]['time'])-(self.history_len+self.forecast_len+1)
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # ========================================================================== #
        # subset xarray on time dimension
        
        ind_end_in_file = ind_start_in_file+self.history_len+self.forecast_len
        
        ## ERA5_subset: a xarray dataset that contains training input and target (for the current batch)
        ERA5_subset = self.all_files[int(ind_file)].isel(
            time=slice(ind_start_in_file, ind_end_in_file+1)) #.load() NOT load into memory

        # ========================================================================== #
        # merge surface into the dataset

        if self.surface_files:
            ## subset surface variables
            surface_subset = self.surface_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file+1)) #.load() NOT load into memory
            
            ## merge upper-air and surface here:
            ERA5_subset = ERA5_subset.merge(surface_subset) # <-- lazy merge, ERA5 and surface both not loaded

        # ==================================================== #
        # split ERA5_subset into training inputs and targets
        #   + merge with dynamic forcing, forcing, and static

        # the ind_end of the ERA5_subset
        ind_end_time = len(ERA5_subset['time'])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = ERA5_subset.time.values.astype('datetime64[s]').astype(int)

        # ==================================================== #
        # xarray dataset as input
        ## historical_ERA5_images: the final input

        historical_ERA5_images = ERA5_subset.isel(
            time=slice(0, self.history_len, self.skip_periods)).load() # <-- load into memory

        # ========================================================================== #
        # merge dynamic forcing inputs
        if self.dyn_forcing_files:
            dyn_forcing_subset = self.dyn_forcing_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file+1))
            dyn_forcing_subset = dyn_forcing_subset.isel(
                time=slice(0, self.history_len, self.skip_periods)).load() # <-- load into memory

            historical_ERA5_images = historical_ERA5_images.merge(dyn_forcing_subset)

        # ========================================================================== #
        # merge forcing inputs
        if self.xarray_forcing:
            # ------------------------------------------------------------------------------- #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(np.array(self.xarray_forcing['time']))
            month_day_inputs = extract_month_day_hour(np.array(historical_ERA5_images['time'])) # <-- upper air
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = self.xarray_forcing.isel(time=ind_forcing) #.load() # <-- loadded in init
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input['time'] = historical_ERA5_images['time']
            # ------------------------------------------------------------------------------- #

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(forcing_subset_input)

        # ========================================================================== #
        # merge static inputs
        if self.xarray_static:
            # expand static var on time dim
            N_time_dims = len(ERA5_subset['time'])
            static_subset_input = self.xarray_static.expand_dims(dim={"time": N_time_dims})
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords({'time': ERA5_subset['time']})

            # slice + load to the GPU
            static_subset_input = static_subset_input.isel(
                time=slice(0, self.history_len, self.skip_periods)) #.load() # <-- loaded in init

            # update 
            static_subset_input['time'] = historical_ERA5_images['time']

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(static_subset_input)
        
        # ==================================================== #
        # xarray dataset as target
        ## target_ERA5_images: the final target

        if self.one_shot is not None:
            # one_shot is True (on), go straight to the last element
            target_ERA5_images = ERA5_subset.isel(time=slice(-1, None)).load() # <-- load into memory
            
            ## merge diagnoisc input here:
            if self.diagnostic_files:
                diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file+1))
                
                diagnostic_subset = diagnostic_subset.isel(
                    time=slice(-1, None)).load() # <-- load into memory
                
                target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)
                
        else:
            # one_shot is None (off), get the full target length based on forecast_len
            target_ERA5_images = ERA5_subset.isel(
                time=slice(self.history_len, ind_end_time, self.skip_periods)).load() # <-- load into memory
    
            ## merge diagnoisc input here:
            if self.diagnostic_files:
                
                # subset diagnostic variables
                diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file+1))
                
                diagnostic_subset = diagnostic_subset.isel(
                    time=slice(self.history_len, ind_end_time, self.skip_periods)).load() # <-- load into memory
                
                # merge into the target dataset
                target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

        # ------------------------------------------------------------------ #
        # sst forcing operations
        if self.sst_forcing is not None:
            # get xr.dataset keys
            varname_skt = self.sst_forcing['varname_skt']
            varname_ocean_mask = self.sst_forcing['varname_ocean_mask']
            
            # get xr.dataarray from the dataset
            ocean_mask = historical_ERA5_images[varname_ocean_mask]
            input_skt = historical_ERA5_images[varname_skt]
            target_skt = target_ERA5_images[varname_skt]
        
            # for multi-input cases, use time=-1 ocean mask for all times
            if self.history_len > 1:
                ocean_mask[:self.history_len-1] = ocean_mask.isel(time=-1)
            
            # get ocean mask
            ocean_mask_bool = ocean_mask.isel(time=-1) == 0
        
            # for multi-input cases, use time=-1 ocean SKT for all times
            if self.history_len > 1:
                input_skt[:self.history_len-1] = input_skt[:self.history_len-1].where(~ocean_mask_bool, input_skt.isel(time=-1))
        
            # for target skt, replace ocean values using time=-1 input SKT
            target_skt = target_skt.where(~ocean_mask_bool, input_skt.isel(time=-1))
        
            # Update the target_ERA5_images dataset with the modified target_skt
            historical_ERA5_images[varname_ocean_mask] = ocean_mask
            historical_ERA5_images[varname_skt] = input_skt
            target_ERA5_images[varname_skt] = target_skt
        
        # pipe xarray datasets to the sampler
        sample = Sample(
            historical_ERA5_images=historical_ERA5_images,
            target_ERA5_images=target_ERA5_images,
            datetime_index=datetime_as_number
        )

        # ==================================== #
        # data normalization
        if self.transform:
            sample = self.transform(sample)

        # assign sample index
        sample["index"] = index

        return sample

class Predict_Dataset(torch.utils.data.IterableDataset):
    '''
    Same as ERA5_and_Forcing_Dataset() but work with rollout_to_netcdf_new.py

    *ksha: dynamic forcing has been added to the rollout-only Dataset, but it has
    not been tested. Once the new tsi is ready, this dataset class will be tested
    '''
    def __init__(self,
                 conf, 
                 varname_upper_air,
                 varname_surface,
                 varname_dyn_forcing,
                 varname_forcing,
                 varname_static,
                 filenames,
                 filename_surface,
                 filename_dyn_forcing,
                 filename_forcing,
                 filename_static,
                 fcst_datetime,
                 history_len,
                 rank,
                 world_size,
                 transform=None,
                 rollout_p=0.0,
                 which_forecast=None):
        
        # ------------------------------------------------------------------------------ #
        
        ## no diagnostics because they are output only
        # varname_diagnostic = None
        
        self.rank = rank
        self.world_size = world_size
        self.transform = transform
        self.history_len = history_len
        self.init_datetime = fcst_datetime

        print(self.init_datetime)
        
        self.which_forecast = which_forecast # <-- got from the old roll-out script. Dont know 
        
        # -------------------------------------- #
        # file names
        self.filenames = filenames # <------------------------ a list of files
        self.filename_surface = filename_surface # <---------- a list of files
        self.filename_dyn_forcing = filename_dyn_forcing # <-- a list of files
        self.filename_forcing = filename_forcing # <-- single file
        self.filename_static = filename_static # <---- single file
        
        # -------------------------------------- #
        # var names
        self.varname_upper_air = varname_upper_air
        self.varname_surface = varname_surface
        self.varname_dyn_forcing = varname_dyn_forcing
        self.varname_forcing = varname_forcing
        self.varname_static = varname_static

        # ====================================== #
        # import all upper air zarr files
        all_files = []
        for fn in self.filenames:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename=fn)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, self.varname_upper_air)
            # collect yearly datasets within a list
            all_files.append(xarray_dataset)
        self.all_files = all_files
        # ====================================== #

        # -------------------------------------- #
        # other settings
        self.current_epoch = 0
        self.rollout_p = rollout_p
        
        self.lead_time_periods = conf['data']['lead_time_periods']
        self.skip_periods = conf['data']['skip_periods']

    def ds_read_and_subset(self, filename, time_start, time_end, varnames):
        sliced_x = get_forward_data(filename)
        sliced_x = sliced_x.isel(time=slice(time_start, time_end))
        sliced_x = drop_var_from_dataset(sliced_x, varnames)
        return sliced_x

    def load_zarr_as_input(self, i_file, i_init_start, i_init_end):
        # get the needed file from a list of zarr files
        # open the zarr file as xr.dataset and subset based on the needed time
        
        # sliced_x: the final output, starts with an upper air xr.dataset
        sliced_x = self.ds_read_and_subset(self.filenames[i_file], 
                                           i_init_start,
                                           i_init_end+1,
                                           self.varname_upper_air)
        # surface variables
        if self.filename_surface is not None:
            sliced_surface = self.ds_read_and_subset(self.filename_surface[i_file], 
                                                     i_init_start,
                                                     i_init_end+1,
                                                     self.varname_surface)
            # merge surface to sliced_x
            sliced_surface['time'] = sliced_x['time']
            sliced_x = sliced_x.merge(sliced_surface)


        # dynamic forcing variables
        if self.filename_dyn_forcing is not None:
            sliced_dyn_forcing = self.ds_read_and_subset(self.filename_dyn_forcing[i_file], 
                                                         i_init_start,
                                                         i_init_end+1,
                                                         self.varname_dyn_forcing)
            # merge surface to sliced_x
            sliced_dyn_forcing['time'] = sliced_x['time']
            sliced_x = sliced_x.merge(sliced_dyn_forcing)

        # forcing / static
        if self.filename_forcing is not None:
            sliced_forcing = xr.open_dataset(self.filename_forcing)
            sliced_forcing = drop_var_from_dataset(sliced_forcing, self.varname_forcing)

            # See also `ERA5_and_Forcing_Dataset`
            # =============================================================================== #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(np.array(sliced_forcing['time']))
            month_day_inputs = extract_month_day_hour(np.array(sliced_x['time']))
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            sliced_forcing = sliced_forcing.isel(time=ind_forcing)
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            sliced_forcing['time'] = sliced_x['time']
            # =============================================================================== #
            
            # merge forcing to sliced_x
            sliced_x = sliced_x.merge(sliced_forcing)
            
        if self.filename_static is not None:
            sliced_static = xr.open_dataset(self.filename_static)
            sliced_static = drop_var_from_dataset(sliced_static, self.varname_static)
            sliced_static = sliced_static.expand_dims(dim={"time": len(sliced_x['time'])})
            sliced_static['time'] = sliced_x['time']
            # merge static to sliced_x
            sliced_x = sliced_x.merge(sliced_static)
        return sliced_x
        
    def find_start_stop_indices(self, index):
        # ============================================================================ #
        # shift hours for history_len > 1, becuase more than one init times are needed
        # <--- !! it MAY NOT work when self.skip_period != 1
        shifted_hours = self.lead_time_periods * self.skip_periods * (self.history_len-1)
        # ============================================================================ #
        # subtrack shifted_hour form the 1st & last init times
        # convert to datetime object
        self.init_datetime[index][0] = datetime.datetime.strptime(
            self.init_datetime[index][0], '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours=shifted_hours)
        self.init_datetime[index][1] = datetime.datetime.strptime(
            self.init_datetime[index][1], '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours=shifted_hours)
        
        # convert the 1st & last init times to a list of init times
        self.init_datetime[index] = generate_datetime(self.init_datetime[index][0], self.init_datetime[index][1], self.lead_time_periods)
        # convert datetime obj to nanosecondes
        init_time_list_dt = [np.datetime64(date.strftime('%Y-%m-%d %H:%M:%S')) for date in self.init_datetime[index]]
        self.init_time_list_np = [np.datetime64(str(dt_obj) + '.000000000').astype(datetime.datetime) for dt_obj in init_time_list_dt]

        for i_file, ds in enumerate(self.all_files):
            # get the year of the current file
            ds_year = int(np.datetime_as_string(ds['time'][0].values, unit='Y'))
        
            # get the first and last years of init times
            init_year0 = nanoseconds_to_year(self.init_time_list_np[0])
            
            # found the right yearly file
            if init_year0 == ds_year:
                
                # convert ds['time'] to a list of nanosecondes
                ds_time_list = [np.datetime64(ds_time.values).astype(datetime.datetime) for ds_time in ds['time']]
                ds_start_time = ds_time_list[0]
                ds_end_time = ds_time_list[-1]
                
                init_time_start = self.init_time_list_np[0]
                # if initalization time is within this (yearly) xr.Dataset
                if ds_start_time <= init_time_start <= ds_end_time:
        
                    # try getting the index of the first initalization time 
                    i_init_start = ds_time_list.index(init_time_start)
                    
                    # for multiple init time inputs (history_len > 1), init_end is different for init_start
                    init_time_end = init_time_start + hour_to_nanoseconds(shifted_hours)
                    
                    # see if init_time_end is alos in this file
                    if ds_start_time <= init_time_end <= ds_end_time:
                        
                        # try getting the index
                        i_init_end = ds_time_list.index(init_time_end)
                    else:
                        # this set of initalizations have crossed years
                        # get the last element of the current file
                        # we have anthoer section that checks additional input data
                        i_init_end = len(ds_time_list) - 1
                        
                    info = [i_file, i_init_start, i_init_end]
                    return info

    def __len__(self):
        return len(self.init_datetime)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, 
                                     num_replicas=num_workers*self.world_size, 
                                     rank=self.rank*num_workers+worker_id, 
                                     shuffle=False)
        for index in sampler:
            # get the init time info for the current sample
            data_lookup = self.find_start_stop_indices(index)
            
            for k, _ in enumerate(self.init_time_list_np):
                
                # the first initialization time: get initalization from data
                if k == 0:
                    i_file, i_init_start, i_init_end = data_lookup
                    
                    # allocate output dict
                    output_dict = {}

                    # get all inputs in one xr.Dataset
                    sliced_x = self.load_zarr_as_input(i_file, i_init_start, i_init_end)
                    
                    # Check if additional data from the next file is needed
                    if len(sliced_x['time']) < self.history_len:
                        
                        # Load excess data from the next file
                        next_file_idx = self.filenames.index(self.filenames[i_file]) + 1
                        
                        if next_file_idx >= len(self.filenames):
                            # not enough input data to support this forecast
                            raise OSError("You have reached the end of the available data. Exiting.")
                            
                        else:
                            # i_init_start = 0 because we need the beginning of the next file only
                            sliced_x_next = self.load_zarr_as_input(next_file_idx, 0, self.history_len)
                            
                            # Concatenate excess data from the next file with the current data
                            sliced_x = xr.concat([sliced_x, sliced_x_next], dim='time')
                            sliced_x = sliced_x.isel(time=slice(0, self.history_len))
                                                     
                    # key 'historical_ERA5_images' is recongnized as input in credit.transform

                    print(
                        np.array(sliced_x['time'])
                    )
                    
                    sample_x = {'historical_ERA5_images': sliced_x}
                    
                    if self.transform:
                        sample_x = self.transform(sample_x)
                        
                    for key in sample_x.keys():
                        output_dict[key] = sample_x[key]
            
                    # <--- !! 'forecast_hour' is actually "forecast_step" but named by assuming hourly
                    output_dict['forecast_hour'] = k + 1 
                    # Adjust stopping condition
                    output_dict['stop_forecast'] = k == (len(self.init_time_list_np) - 1)
                    output_dict['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]
                    
                # other later initialization time: the same initalization as in k=0, but add more forecast steps
                else:
                    output_dict['forecast_hour'] = k + 1
                     # Adjust stopping condition
                    output_dict['stop_forecast'] = k == (len(self.init_time_list_np) - 1)
                    
                # return output_dict
                yield output_dict
                
                if output_dict['stop_forecast']:
                    break

# =============================================== #
# This dataset works for hourly model only
# it does not support forcing & static here
# but it pairs to the ToTensor that adds
# TOA (which has problem) and other static fields
# =============================================== #
class ERA5Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            filenames: list = ['/glade/derecho/scratch/wchapman/STAGING/TOTAL_2012-01-01_2012-12-31_staged.zarr',
                               '/glade/derecho/scratch/wchapman/STAGING/TOTAL_2013-01-01_2013-12-31_staged.zarr'],
            history_len: int = 1,
            forecast_len: int = 2,
            transform: Optional[Callable] = None,
            seed=42,
            skip_periods=None,
            one_shot=None,
            max_forecast_len=None
    ):
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.skip_periods = skip_periods
        self.one_shot = one_shot
        self.total_seq_len = self.history_len + self.forecast_len
        all_fils = []
        filenames = sorted(filenames)
        for fn in filenames:
            all_fils.append(get_forward_data(filename=fn))
        self.all_fils = all_fils
        self.data_array = all_fils[0]
        self.rng = np.random.default_rng(seed=seed)
        self.max_forecast_len = max_forecast_len

        # set data places:
        indo = 0
        self.meta_data_dict = {}
        for ee, bb in enumerate(self.all_fils):
            self.meta_data_dict[str(ee)] = [len(bb['time']), indo, indo + len(bb['time'])]
            indo += len(bb['time']) + 1

        # set out of bounds indexes...
        OOB = []
        for kk in self.meta_data_dict.keys():
            OOB.append(generate_integer_list_around(self.meta_data_dict[kk][2]))
        self.OOB = flatten_list(OOB)

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        tlen = 0
        for bb in self.all_fils:
            tlen += len(bb['time']) - self.total_seq_len + 1
        return tlen

    def __getitem__(self, index):
        # find the result key:
        result_key = find_key_for_number(index, self.meta_data_dict)

        # get the data selection:
        true_ind = index - self.meta_data_dict[result_key][1]

        if true_ind > (len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 1)):
            true_ind = len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 1)

        datasel = self.all_fils[int(result_key)].isel(
            time=slice(true_ind, true_ind + self.history_len + self.forecast_len + 1))

        if (self.skip_periods is not None) and (self.one_shot is None):
            sample = Sample(
                historical_ERA5_images=datasel.isel(time=slice(0, self.history_len, self.skip_periods)),
                target_ERA5_images=datasel.isel(time=slice(self.history_len, len(datasel['time']), self.skip_periods)),
                datetime_index=datasel.time.values.astype('datetime64[s]').astype(int)
            )

        elif (self.skip_periods is not None) and (self.one_shot is not None):
            target_ERA5_images = datasel.isel(time=slice(self.history_len, len(datasel['time']), self.skip_periods))
            target_ERA5_images = target_ERA5_images.isel(time=slice(0, 1))

            sample = Sample(
                historical_ERA5_images=datasel.isel(time=slice(0, self.history_len, self.skip_periods)),
                target_ERA5_images=target_ERA5_images,
                datetime_index=datasel.time.values.astype('datetime64[s]').astype(int)
            )

        elif self.one_shot is not None:
            historical_data = datasel.isel(time=slice(0, self.history_len)).load()
            target_data = datasel.isel(time=slice(-1, None)).load()
            # Create the Sample object with the loaded data
            sample = Sample(
                historical_ERA5_images=historical_data,
                target_ERA5_images=target_data,
                datetime_index=[int(historical_data.time.values[0].astype('datetime64[s]').astype(int)),
                                int(target_data.time.values[0].astype('datetime64[s]').astype(int))]
            )
        else:
            sample = Sample(
                historical_ERA5_images=datasel.isel(time=slice(0, self.history_len)),
                target_ERA5_images=datasel.isel(time=slice(self.history_len, len(datasel['time']))),
                datetime_index=datasel.time.values.astype('datetime64[s]').astype(int)
            )

        if self.transform:
            sample = self.transform(sample)

        sample["index"] = index

        return sample

# ================================= #
# This dataset is old, but not sure
# if anyone still uses it
# ================================= #
class Dataset_BridgeScaler(torch.utils.data.Dataset):
    def __init__(
            self,
            conf,
            conf_dataset,
            transform: Optional[Callable] = None,
    ):
        years_do = list(conf["data"][conf_dataset])
        self.available_dates = pd.date_range(str(years_do[0]), str(years_do[1]), freq='1H')
        self.data_path = str(conf["data"]["bs_data_path"])
        self.history_len = int(conf["data"]["history_len"])
        self.forecast_len = int(conf["data"]["forecast_len"])
        self.forecast_len = 1 if self.forecast_len == 0 else self.forecast_len
        self.file_format = str(conf["data"]["bs_file_format"])
        self.transform = transform
        self.skip_periods = conf["data"]["skip_periods"]
        self.one_shot = conf["data"]["one_shot"]
        self.total_seq_len = self.history_len + self.forecast_len
        self.first_date = self.available_dates[0]
        self.last_date = self.available_dates[-1]

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        tlen = 0
        tlen = len(self.available_dates)
        return tlen

    def evenly_spaced_indlist(self, index, skip_periods, forecast_len, history_len):
        # Initialize the list with the base index
        indlist = [index]

        # Add forecast indices
        for i in range(1, forecast_len + 1):
            indlist.append(index + i * skip_periods)

        # Add history indices
        for i in range(1, history_len + 1):
            indlist.append(index - i * skip_periods)

        # Sort the list to maintain order
        indlist = sorted(indlist)
        return indlist

    def __getitem__(self, index):

        if (self.skip_periods is None) & (self.one_shot is None):
            date_index = self.available_dates[index]

            indlist = sorted(
                [index]
                + [index + (i) + 1 for i in range(self.forecast_len)]
                + [index - i - 1 for i in range(self.history_len)]
            )

            if np.min(indlist) < 0:
                indlist = list(np.array(indlist) + np.abs(np.min(indlist)))
                index += np.abs(np.min(indlist))
            if np.max(indlist) >= self.__len__():
                indlist = list(np.array(indlist) - np.abs(np.max(indlist)) + self.__len__() - 1)
                index -= np.abs(np.max(indlist))
            date_index = self.available_dates[indlist]
            str_tot_find = f'%Y/%m/%d/{self.file_format}'
            fs = [f"{self.data_path}/{bb.strftime(str_tot_find)}" for bb in date_index]
            if len(fs) < 2:
                raise "Must be greater than one day in the list [x and x+1 minimum]"

            fe = [1 if os.path.exists(fn) else 0 for fn in fs]
            if np.sum(fe) == len(fs):
                pass
            else:
                raise "weve left the training dataset, check your dataloader logic"

            DShist = xr.open_mfdataset(fs[1:self.history_len + 1]).load()
            DSfor = xr.open_mfdataset(fs[self.history_len + 1:self.history_len + 1 + self.forecast_len]).load()

            sample = Sample(
                historical_ERA5_images=DShist,
                target_ERA5_images=DSfor,
                datetime_index=date_index
            )

            if self.transform:
                sample = self.transform(sample)
            return sample
        if self.one_shot is not None:
            date_index = self.available_dates[index]

            indlist = sorted(
                [index] +
                [index + (i) + 1 for i in range(self.forecast_len)] +
                [index - i - 1 for i in range(self.history_len)]
            )
            # indlist.append(index+self.one_shot)

            if np.min(indlist) < 0:
                indlist = list(np.array(indlist) + np.abs(np.min(indlist)))
                index += np.abs(np.min(indlist))
            if np.max(indlist) >= self.__len__():
                indlist = list(np.array(indlist) - np.abs(np.max(indlist)) + self.__len__() - 1)
                index -= np.abs(np.max(indlist))

            date_index = self.available_dates[indlist]
            str_tot_find = f'%Y/%m/%d/{self.file_format}'
            fs = [f"{self.data_path}/{bb.strftime(str_tot_find)}" for bb in date_index]

            if len(fs) < 2:
                raise "Must be greater than one day in the list [x and x+1 minimum]"

            fe = [1 if os.path.exists(fn) else 0 for fn in fs]
            if np.sum(fe) == len(fs):
                pass
            else:
                raise "weve left the training dataset, check your dataloader logic"

            DShist = xr.open_mfdataset(fs[:self.history_len]).load()
            DSfor = xr.open_mfdataset(fs[-2]).load()

            sample = Sample(
                historical_ERA5_images=DShist,
                target_ERA5_images=DSfor,
                datetime_index=date_index
            )

            if self.transform:
                sample = self.transform(sample)
            return sample

        if (self.skip_periods is not None) and (self.one_shot is None):
            date_index = self.available_dates[index]
            indlist = self.evenly_spaced_indlist(index, self.skip_periods, self.forecast_len, self.history_len)

            if np.min(indlist) < 0:
                indlist = list(np.array(indlist) + np.abs(np.min(indlist)))
                index += np.abs(np.min(indlist))
            if np.max(indlist) >= self.__len__():
                indlist = list(np.array(indlist) - np.abs(np.max(indlist)) + self.__len__() - 1)
                index -= np.abs(np.max(indlist))

            date_index = self.available_dates[indlist]
            str_tot_find = f'%Y/%m/%d/{self.file_format}'
            fs = [f"{self.data_path}/{bb.strftime(str_tot_find)}" for bb in date_index]

            if len(fs) < 2:
                raise "Must be greater than one day in the list [x and x+1 minimum]"

            fe = [1 if os.path.exists(fn) else 0 for fn in fs]
            if np.sum(fe) == len(fs):
                pass
            else:
                raise "weve left the training dataset, check your dataloader logic"

            DShist = xr.open_mfdataset(fs[:self.history_len]).load()
            DSfor = xr.open_mfdataset(fs[self.history_len:self.history_len + self.forecast_len]).load()

            sample = Sample(
                historical_ERA5_images=DShist,
                target_ERA5_images=DSfor,
                datetime_index=date_index
            )

            if self.transform:
                sample = self.transform(sample)
            return sample

# ================================================================== #
# Note: DistributedSequentialDataset & DistributedSequentialDataset
# are legacy; they wrap ERA5Dataset to send data batches to GPUs for
# (1 class of?) huge sharded models, but otherwise have been
# superseded by ERA5Dataset.
# ================================================================== #
class SequentialDataset(torch.utils.data.Dataset):

    def __init__(self, filenames, history_len=1, forecast_len=2, skip_periods=1, transform=None, random_forecast=True):
        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_fils = self.dataset.all_fils
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.filenames = filenames
        self.transform = transform
        self.skip_periods = skip_periods
        self.random_forecast = random_forecast
        self.iteration_count = 0
        self.current_epoch = 0
        self.adjust_forecast = 0

        self.index_list = []
        for i, x in enumerate(self.all_fils):
            times = x['time'].values
            slices = np.arange(0, times.shape[0] - (self.forecast_len + 1))
            self.index_list += [(i, slice) for slice in slices]

    def __len__(self):
        return len(self.index_list)

    def set_params(self, epoch):
        self.current_epoch = epoch
        self.iteration_count = 0

    def __getitem__(self, index):

        if self.random_forecast and (self.iteration_count % self.forecast_len == 0):
            # Randomly choose a starting point within a valid range
            max_start = len(self.index_list) - (self.forecast_len + 1)
            self.adjust_forecast = np.random.randint(0, max_start + 1)

        index = (index + self.adjust_forecast) % self.__len__()
        file_id, slice_idx = self.index_list[index]

        dataset = xr.open_zarr(self.filenames[file_id], consolidated=True).isel(
            time=slice(slice_idx, slice_idx + self.skip_periods + 1, self.skip_periods))

        sample = {
            'x': dataset.isel(time=slice(0, 1, 1)),
            'y': dataset.isel(time=slice(1, 2, 1)),
        }

        if self.transform:
            sample = self.transform(sample)

        sample['forecast_hour'] = self.iteration_count
        sample['forecast_datetime'] = dataset.time.values.astype('datetime64[s]').astype(int)
        sample['stop_forecast'] = False

        if self.iteration_count == self.forecast_len - 1:
            sample['stop_forecast'] = True

        # Increment the iteration count
        self.iteration_count += 1

        return sample

# =========================================== #
# This class may have the one-step-off problem
# =========================================== #
# class PredictForecast(torch.utils.data.IterableDataset):
#     def __init__(self,
#                  filenames,
#                  forecasts,
#                  history_len,
#                  forecast_len,
#                  skip_periods,
#                  rank,
#                  world_size,
#                  shuffle=False,
#                  transform=None,
#                  rollout_p=0.0,
#                  start_time=None,
#                  stop_time=None,
#                  which_forecast=None):

#         self.dataset = ERA5Dataset(
#             filenames=filenames,
#             history_len=history_len,
#             forecast_len=forecast_len,
#             skip_periods=skip_periods,
#             transform=transform
#         )
#         self.meta_data_dict = self.dataset.meta_data_dict
#         self.all_files = self.dataset.all_fils
#         self.history_len = history_len
#         self.forecast_len = forecast_len
#         self.filenames = filenames
#         self.transform = transform
#         self.rank = rank
#         self.world_size = world_size
#         self.shuffle = shuffle
#         self.skip_periods = skip_periods
#         self.current_epoch = 0
#         self.rollout_p = rollout_p
#         self.forecasts = forecasts
#         self.skip_periods = skip_periods if skip_periods is not None else 1
#         self.which_forecast = which_forecast

#     def find_start_stop_indices(self, index):
#         start_time = self.forecasts[index][0]
#         date_object = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
#         shifted_hours = self.skip_periods * self.history_len
#         date_object = date_object - datetime.timedelta(hours=shifted_hours)
#         self.forecasts[index][0] = date_object.strftime('%Y-%m-%d %H:%M:%S')

#         datetime_objs = [np.datetime64(date) for date in self.forecasts[index]]
#         start_time, stop_time = [str(datetime_obj) + '.000000000' for datetime_obj in datetime_objs]
#         self.start_time = np.datetime64(start_time).astype(datetime.datetime)
#         self.stop_time = np.datetime64(stop_time).astype(datetime.datetime)

#         info = {}

#         for idx, dataset in enumerate(self.all_files):
#             start_time = np.datetime64(dataset['time'].min().values).astype(datetime.datetime)
#             stop_time = np.datetime64(dataset['time'].max().values).astype(datetime.datetime)
#             track_start = False
#             track_stop = False

#             if start_time <= self.start_time <= stop_time:
#                 # Start time is in this file, use start time index
#                 dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
#                 start_idx = np.searchsorted(dataset, self.start_time)
#                 start_idx = max(0, min(start_idx, len(dataset) - 1))
#                 track_start = True

#             elif start_time < self.stop_time and stop_time > self.start_time:
#                 # File overlaps time range, use full file
#                 start_idx = 0
#                 track_start = True

#             if start_time <= self.stop_time <= stop_time:
#                 # Stop time is in this file, use stop time index
#                 if isinstance(dataset, np.ndarray):
#                     pass
#                 else:
#                     dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
#                 stop_idx = np.searchsorted(dataset, self.stop_time)
#                 stop_idx = max(0, min(stop_idx, len(dataset) - 1))
#                 track_stop = True

#             elif start_time < self.stop_time and stop_time > self.start_time:
#                 # File overlaps time range, use full file
#                 stop_idx = len(dataset) - 1
#                 track_stop = True

#             # Only include files that overlap the time range
#             if track_start and track_stop:
#                 info[idx] = ((idx, start_idx), (idx, stop_idx))

#         indices = []
#         for dataset_idx, (start, stop) in info.items():
#             for i in range(start[1], stop[1] + 1):
#                 indices.append((start[0], i))
#         return indices

#     def __len__(self):
#         return len(self.forecasts)

#     def __iter__(self):
#         worker_info = get_worker_info()
#         num_workers = worker_info.num_workers if worker_info is not None else 1
#         worker_id = worker_info.id if worker_info is not None else 0
#         sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
#                                      rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)

#         for index in sampler:

#             data_lookup = self.find_start_stop_indices(index)

#             for k, (file_key, time_key) in enumerate(data_lookup):
#                 concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': []}
#                 sliced_x = xr.open_zarr(self.filenames[file_key], consolidated=True).isel(
#                     time=slice(time_key, time_key + self.history_len + 1))

#                 # Check if additional data from the next file is needed
#                 if len(sliced_x['time']) < self.history_len + 1:
#                     # Load excess data from the next file
#                     next_file_idx = self.filenames.index(self.filenames[file_key]) + 1
#                     if next_file_idx == len(self.filenames):
#                         raise OSError("You have reached the end of the available data. Exiting.")
#                     sliced_x_next = xr.open_zarr(
#                         self.filenames[next_file_idx],
#                         consolidated=True).isel(time=slice(0, self.history_len + 1 - len(sliced_x['time'])))

#                     # Concatenate excess data from the next file with the current data
#                     sliced_x = xr.concat([sliced_x, sliced_x_next], dim='time')

#                 sample_x = {
#                     'x': sliced_x.isel(time=slice(0, self.history_len)),
#                     'y': sliced_x.isel(time=slice(self.history_len, self.history_len + 1))  # Fetch y data for t(i+1)
#                 }

#                 if self.transform:
#                     sample_x = self.transform(sample_x)
#                     # Add static vars, if any, to the return dictionary
#                     if "static" in sample_x:
#                         concatenated_samples["static"] = []
#                     if "TOA" in sample_x:
#                         concatenated_samples["TOA"] = []

#                 for key in concatenated_samples.keys():
#                     concatenated_samples[key] = sample_x[key].squeeze(0) if self.history_len == 1 else sample_x[key]

#                 concatenated_samples['forecast_hour'] = k + 1
#                 concatenated_samples['stop_forecast'] = (
#                             k == (len(data_lookup) - self.history_len - 1))  # Adjust stopping condition
#                 concatenated_samples['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]

#                 yield concatenated_samples

#                 if concatenated_samples['stop_forecast']:
#                     break

# =========================================== #
# This class may have the one-step-off problem
# =========================================== #
# class PredictForecastRollout(torch.utils.data.IterableDataset):
#     def __init__(self,
#                  filenames,
#                  forecasts,
#                  history_len,
#                  forecast_len,
#                  skip_periods,
#                  rank,
#                  world_size,
#                  shuffle=False,
#                  transform=None,
#                  rollout_p=0.0,
#                  start_time=None,
#                  stop_time=None,
#                  which_forecast=None):

#         self.dataset = ERA5Dataset(
#             filenames=filenames,
#             history_len=history_len,
#             forecast_len=forecast_len,
#             skip_periods=skip_periods,
#             transform=transform
#         )
#         self.meta_data_dict = self.dataset.meta_data_dict
#         self.all_files = self.dataset.all_fils
#         self.history_len = history_len
#         self.forecast_len = forecast_len
#         self.filenames = filenames
#         self.transform = transform
#         self.rank = rank
#         self.world_size = world_size
#         self.shuffle = shuffle
#         self.skip_periods = skip_periods
#         self.current_epoch = 0
#         self.rollout_p = rollout_p
#         self.forecasts = forecasts
#         self.skip_periods = skip_periods if skip_periods is not None else 1
#         self.which_forecast = which_forecast

#     def find_start_stop_indices(self, index):
#         start_time = self.forecasts[index][0]
#         date_object = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
#         shifted_hours = self.skip_periods * self.history_len
#         date_object = date_object - datetime.timedelta(hours=shifted_hours)
#         self.forecasts[index][0] = date_object.strftime('%Y-%m-%d %H:%M:%S')

#         datetime_objs = [np.datetime64(date) for date in self.forecasts[index]]
#         start_time, stop_time = [str(datetime_obj) + '.000000000' for datetime_obj in datetime_objs]
#         self.start_time = np.datetime64(start_time).astype(datetime.datetime)
#         self.stop_time = np.datetime64(stop_time).astype(datetime.datetime)

#         info = {}

#         for idx, dataset in enumerate(self.all_files):
#             start_time = np.datetime64(dataset['time'].min().values).astype(datetime.datetime)
#             stop_time = np.datetime64(dataset['time'].max().values).astype(datetime.datetime)
#             track_start = False
#             track_stop = False

#             if start_time <= self.start_time <= stop_time:
#                 # Start time is in this file, use start time index
#                 dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
#                 start_idx = np.searchsorted(dataset, self.start_time)
#                 start_idx = max(0, min(start_idx, len(dataset) - 1))
#                 track_start = True

#             elif start_time < self.stop_time and stop_time > self.start_time:
#                 # File overlaps time range, use full file
#                 start_idx = 0
#                 track_start = True

#             if start_time <= self.stop_time <= stop_time:
#                 # Stop time is in this file, use stop time index
#                 if isinstance(dataset, np.ndarray):
#                     pass
#                 else:
#                     dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
#                 stop_idx = np.searchsorted(dataset, self.stop_time)
#                 stop_idx = max(0, min(stop_idx, len(dataset) - 1))
#                 track_stop = True

#             elif start_time < self.stop_time and stop_time > self.start_time:
#                 # File overlaps time range, use full file
#                 stop_idx = len(dataset) - 1
#                 track_stop = True

#             # Only include files that overlap the time range
#             if track_start and track_stop:
#                 info[idx] = ((idx, start_idx), (idx, stop_idx))

#         indices = []
#         for dataset_idx, (start, stop) in info.items():
#             for i in range(start[1], stop[1] + 1):
#                 indices.append((start[0], i))
#         return indices

#     def __len__(self):
#         return len(self.forecasts)

#     def __iter__(self):
#         worker_info = get_worker_info()
#         num_workers = worker_info.num_workers if worker_info is not None else 1
#         worker_id = worker_info.id if worker_info is not None else 0
#         sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
#                                      rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)

#         for index in sampler:

#             data_lookup = self.find_start_stop_indices(index)

#             for k, (file_key, time_key) in enumerate(data_lookup):
#                 concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': []}
#                 sliced_x = xr.open_zarr(self.filenames[file_key], consolidated=True).isel(
#                     time=slice(time_key, time_key + self.history_len + 1))

#                 # Check if additional data from the next file is needed
#                 if len(sliced_x['time']) < self.history_len + 1:
#                     # Load excess data from the next file
#                     next_file_idx = self.filenames.index(self.filenames[file_key]) + 1
#                     if next_file_idx == len(self.filenames):
#                         raise OSError("You have reached the end of the available data. Exiting.")
#                     sliced_x_next = xr.open_zarr(
#                         self.filenames[next_file_idx],
#                         consolidated=True).isel(time=slice(0, self.history_len + 1 - len(sliced_x['time'])))

#                     # Concatenate excess data from the next file with the current data
#                     sliced_x = xr.concat([sliced_x, sliced_x_next], dim='time')

#                 sample_x = {
#                     'x': sliced_x.isel(time=slice(0, self.history_len)),
#                     'y': sliced_x.isel(time=slice(self.history_len, self.history_len + 1))  # Fetch y data for t(i+1)
#                 }

#                 if self.transform:
#                     sample_x = self.transform(sample_x)
#                     # Add static vars, if any, to the return dictionary
#                     if "static" in sample_x:
#                         concatenated_samples["static"] = []
#                     if "TOA" in sample_x:
#                         concatenated_samples["TOA"] = []

#                 for key in concatenated_samples.keys():
#                     concatenated_samples[key] = sample_x[key].squeeze(0) if self.history_len == 1 else sample_x[key]

#                 concatenated_samples['forecast_hour'] = k + 1
#                 concatenated_samples['stop_forecast'] = (
#                             k == (len(data_lookup) - self.history_len - 1))  # Adjust stopping condition
#                 concatenated_samples['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]

#                 yield concatenated_samples

#                 break

# =========================================== #
# This class may have the one-step-off problem
# =========================================== #
# class PredictForecastQuantile(PredictForecast):

#     def __init__(self,
#                  conf,
#                  filenames,
#                  forecasts,
#                  history_len,
#                  forecast_len,
#                  skip_periods,
#                  rank,
#                  world_size,
#                  shuffle=False,
#                  transform=None,
#                  rollout_p=0.0,
#                  start_time=None,
#                  stop_time=None):

#         from credit.transforms import load_transforms

#         transform = load_transforms(conf)

#         self.dataset = Dataset_BridgeScaler(
#             conf,
#             conf_dataset='bs_years_test',
#             transform=transform
#         )

#         # Need information on the saved files
#         self.all_files = [get_forward_data(filename=fn) for fn in sorted(filenames)]
#         # Set data places:
#         indo = 0
#         self.meta_data_dict = {}
#         for ee, bb in enumerate(self.all_files):
#             self.meta_data_dict[str(ee)] = [len(bb['time']), indo, indo + len(bb['time'])]
#             indo += len(bb['time']) + 1

#         self.history_len = history_len
#         self.forecast_len = forecast_len
#         self.filenames = filenames
#         self.transform = transform
#         self.rank = rank
#         self.world_size = world_size
#         self.shuffle = shuffle
#         self.skip_periods = skip_periods
#         self.current_epoch = 0
#         self.rollout_p = rollout_p
#         self.forecasts = forecasts
#         self.skip_periods = skip_periods if skip_periods is not None else 1

#     def find_start_stop_indices(self, index):
#         start_time = self.forecasts[index][0]
#         date_object = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
#         shifted_hours = self.skip_periods * self.history_len
#         date_object = date_object - datetime.timedelta(hours=shifted_hours)
#         self.forecasts[index][0] = date_object.strftime('%Y-%m-%d %H:%M:%S')

#         datetime_objs = [np.datetime64(date) for date in self.forecasts[index]]
#         start_time, stop_time = [str(datetime_obj) + '.000000000' for datetime_obj in datetime_objs]
#         self.start_time = np.datetime64(start_time).astype(datetime.datetime)
#         self.stop_time = np.datetime64(stop_time).astype(datetime.datetime)

#         info = {}

#         for idx, dataset in enumerate(self.all_files):
#             start_time = np.datetime64(dataset['time'].min().values).astype(datetime.datetime)
#             stop_time = np.datetime64(dataset['time'].max().values).astype(datetime.datetime)
#             track_start = False
#             track_stop = False

#             if start_time <= self.start_time <= stop_time:
#                 # Start time is in this file, use start time index
#                 dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
#                 start_idx = np.searchsorted(dataset, self.start_time)
#                 start_idx = max(0, min(start_idx, len(dataset) - 1))
#                 track_start = True

#             elif start_time < self.stop_time and stop_time > self.start_time:
#                 # File overlaps time range, use full file
#                 start_idx = 0
#                 track_start = True

#             if start_time <= self.stop_time <= stop_time:
#                 # Stop time is in this file, use stop time index
#                 if isinstance(dataset, np.ndarray):
#                     pass
#                 else:
#                     dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
#                 stop_idx = np.searchsorted(dataset, self.stop_time)
#                 stop_idx = max(0, min(stop_idx, len(dataset) - 1))
#                 track_stop = True

#             elif start_time < self.stop_time and stop_time > self.start_time:
#                 # File overlaps time range, use full file
#                 stop_idx = len(dataset) - 1
#                 track_stop = True

#             # Only include files that overlap the time range
#             if track_start and track_stop:
#                 info[idx] = ((idx, start_idx), (idx, stop_idx))

#         indices = []
#         for dataset_idx, (start, stop) in info.items():
#             for i in range(start[1], stop[1] + 1):
#                 indices.append((start[0], i))
#         return indices

#     def __len__(self):
#         return len(self.forecasts)

#     def __iter__(self):
#         worker_info = get_worker_info()
#         num_workers = worker_info.num_workers if worker_info is not None else 1
#         worker_id = worker_info.id if worker_info is not None else 0
#         sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
#                                      rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)

#         for index in sampler:

#             data_lookup = self.find_start_stop_indices(index)

#             for k, (file_key, time_key) in enumerate(data_lookup):
#                 concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': []}
#                 sliced_x = xr.open_zarr(self.filenames[file_key], consolidated=True).isel(
#                     time=slice(time_key, time_key + self.history_len + 1))

#                 # Check if additional data from the next file is needed
#                 if len(sliced_x['time']) < self.history_len + 1:
#                     # Load excess data from the next file
#                     next_file_idx = self.filenames.index(self.filenames[file_key]) + 1
#                     if next_file_idx == len(self.filenames):
#                         raise OSError("You have reached the end of the available data. Exiting.")
#                     sliced_x_next = xr.open_zarr(
#                         self.filenames[next_file_idx],
#                         consolidated=True).isel(time=slice(0, self.history_len + 1 - len(sliced_x['time'])))

#                     # Concatenate excess data from the next file with the current data
#                     sliced_x = xr.concat([sliced_x, sliced_x_next], dim='time')

#                 sample_x = {
#                     'x': sliced_x.isel(time=slice(0, self.history_len)),
#                     'y': sliced_x.isel(time=slice(self.history_len, self.history_len + 1))  # Fetch y data for t(i+1)
#                 }

#                 if self.transform:
#                     sample_x = self.transform(sample_x)
#                     # Add static vars, if any, to the return dictionary
#                     if "static" in sample_x:
#                         concatenated_samples["static"] = []
#                     if "TOA" in sample_x:
#                         concatenated_samples["TOA"] = []

#                 for key in concatenated_samples.keys():
#                     concatenated_samples[key] = sample_x[key].squeeze(0) if self.history_len == 1 else sample_x[key]

#                 concatenated_samples['forecast_hour'] = k + 1
#                 concatenated_samples['stop_forecast'] = (
#                             k == (len(data_lookup) - self.history_len - 1))  # Adjust stopping condition
#                 concatenated_samples['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]

#                 yield concatenated_samples

#                 if concatenated_samples['stop_forecast']:
#                     break
