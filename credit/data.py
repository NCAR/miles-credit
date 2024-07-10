'''
data.py 
-------------------------------------------------------
Content:
    - get_forward_data(filename) -> xr.DataArray
    - get_forward_data_netCDF4(filename) -> xr.DataArray
    - ERA5_Static_Dataset(torch.utils.data.Dataset)

Yingkai Sha
ksha@ucar.edu
'''

# system tools
import os
from glob import glob
from timeit import timeit
from functools import reduce
from itertools import repeat
from dataclasses import dataclass, field
from typing import Optional, Callable, TypedDict, Union, Iterable, NamedTuple, List

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


#

def get_forward_data(filename) -> xr.DataArray:
    """Lazily opens the Zarr store on gladefilesystem.
    """
    dataset = xr.open_zarr(filename, consolidated=True)
    return dataset


def get_forward_data_netCDF4(filename) -> xr.DataArray:
    """Lazily opens netCDF4 files.
    """
    dataset = xr.open_dataset(filename)
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


@dataclass
class Reshape_Data():
    size: int = 128  #: Size of the cropped image.

    def __call__(self, sample: Sample) -> Sample:
        for attr_name in IMAGE_ATTR_NAMES:
            image = sample[attr_name]
            # TODO: Random crop!
            cropped_image = image[..., :self.size, :self.size]
            sample[attr_name] = cropped_image
        return sample


class CheckForBadData():
    def __call__(self, sample: Sample) -> Sample:
        for attr_name in IMAGE_ATTR_NAMES:
            image = sample[attr_name]
            if np.any(image < 0):
                raise ValueError(f'\n{attr_name} has negative values at {image.time.values}')
        return sample


# class NormalizeState():
#     def __init__(self, mean_file, std_file):
#         self.mean_ds = xr.open_dataset(mean_file)
#         self.std_ds = xr.open_dataset(std_file)

#     def __call__(self, sample: Sample) -> Sample:
#         normalized_sample = {}
#         for key, value in sample.items():
#             if isinstance(value, xr.Dataset):
#                 #key_change = key
#                 #value_change = (value - self.mean_ds)/self.std_ds
#                 #sample[key]=value_change
#                 normalized_sample[key] = (value - self.mean_ds) / self.std_ds
#         return normalized_sample


class Segment(NamedTuple):
    """Represents the start and end indicies of a segment of contiguous samples."""
    start: int
    end: int


def get_contiguous_segments(dt_index: pd.DatetimeIndex, min_timesteps: int, max_gap: pd.Timedelta) -> Iterable[Segment]:
    """Chunk datetime index into contiguous segments, each at least min_timesteps long.

    max_gap defines the threshold for what constitutes a 'gap' between contiguous segments.

    Throw away any timesteps in a sequence shorter than min_timesteps long.
    """
    gap_mask = np.diff(dt_index) > max_gap
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(dt_index)]))

    segments = []
    start_i = 0
    for end_i in segment_boundaries:
        n_timesteps = end_i - start_i
        if n_timesteps >= min_timesteps:
            segment = Segment(start=start_i, end=end_i)
            segments.append(segment)
        start_i = end_i

    return segments


def get_zarr_chunk_sequences(
        n_chunks_per_disk_load: int,
        zarr_chunk_boundaries: Iterable[int],
        contiguous_segments: Iterable[Segment]
) -> Iterable[Segment]:
    """

    Args:
      n_chunks_per_disk_load: Maximum number of Zarr chunks to load from disk in one go.
      zarr_chunk_boundaries: The indicies into the Zarr store's time dimension which define the Zarr chunk boundaries.
        Must be sorted.
      contiguous_segments: Indicies into the Zarr store's time dimension that define contiguous timeseries.
        That is, timeseries with no gaps.

    Returns zarr_chunk_sequences: a list of Segments representing the start and end indicies of contiguous sequences of multiple Zarr chunks,
    all exactly n_chunks_per_disk_load long (for contiguous segments at least as long as n_chunks_per_disk_load zarr chunks),
    and at least one side of the boundary will lie on a 'natural' Zarr chunk boundary.

    For example, say that n_chunks_per_disk_load = 3, and the Zarr chunks sizes are all 5:


                  0    5   10   15   20   25   30   35
                  |....|....|....|....|....|....|....|

    INPUTS:
                     |------CONTIGUOUS SEGMENT----|

    zarr_chunk_boundaries:
                  |----|----|----|----|----|----|----|

    OUTPUT:
    zarr_chunk_sequences:
           3 to 15:  |-|----|----|
           5 to 20:    |----|----|----|
          10 to 25:         |----|----|----|
          15 to 30:              |----|----|----|
          20 to 32:                   |----|----|-|

    """
    assert n_chunks_per_disk_load > 0

    zarr_chunk_sequences = []

    for contig_segment in contiguous_segments:
        # searchsorted() returns the index into zarr_chunk_boundaries at which contig_segment.start
        # should be inserted into zarr_chunk_boundaries to maintain a sorted list.
        # i_of_first_zarr_chunk is the index to the element in zarr_chunk_boundaries which defines
        # the start of the current contig chunk.
        i_of_first_zarr_chunk = np.searchsorted(zarr_chunk_boundaries, contig_segment.start)

        # i_of_first_zarr_chunk will be too large by 1 unless contig_segment.start lies
        # exactly on a Zarr chunk boundary.  Hence we must subtract 1, or else we'll
        # end up with the first contig_chunk being 1 + n_chunks_per_disk_load chunks long.
        if zarr_chunk_boundaries[i_of_first_zarr_chunk] > contig_segment.start:
            i_of_first_zarr_chunk -= 1

        # Prepare for looping to create multiple Zarr chunk sequences for the current contig_segment.
        zarr_chunk_seq_start_i = contig_segment.start
        zarr_chunk_seq_end_i = None  # Just a convenience to allow us to break the while loop by checking if zarr_chunk_seq_end_i != contig_segment.end.
        while zarr_chunk_seq_end_i != contig_segment.end:
            zarr_chunk_seq_end_i = zarr_chunk_boundaries[i_of_first_zarr_chunk + n_chunks_per_disk_load]
            zarr_chunk_seq_end_i = min(zarr_chunk_seq_end_i, contig_segment.end)
            zarr_chunk_sequences.append(Segment(start=zarr_chunk_seq_start_i, end=zarr_chunk_seq_end_i))
            i_of_first_zarr_chunk += 1
            zarr_chunk_seq_start_i = zarr_chunk_boundaries[i_of_first_zarr_chunk]

    return zarr_chunk_sequences


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
    It will raise if xarray.Dataset.keys() != varname_keep.
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
        - foring variables (time, lat, lon)
        - diagnostic variables (time, lat, lon)
        - static variables (lat, lon)
        
    Parameters:
    - filenames: ERA5 file path as *.zarr with re (e.g., /user/ERA5/*.zarr)
    - filename_forcing: None /or a netCDF4 file that contains all the forcing variables.
    - filename_static: None /or a netCDF4 file that contains all the static variables.
    
    '''

    def __init__(
        self,
        varname_upper_air,
        varname_surface,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames,
        filename_surface=None,
        filename_forcing=None,
        filename_static=None,
        filename_diagnostic=None,
        history_len=2,
        forecast_len=0,
        transform=None,
        seed=42,
        skip_periods=None,
        one_shot=None,
        max_forecast_len=None
    ):
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

        # ======================================================== #
        # ERA5 operations
        all_files = []
        filenames = sorted(filenames)
        
        for fn in filenames:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename=fn)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_upper_air)

            # collect yearly datasets within a list
            all_files.append(xarray_dataset)
            
        self.all_files = all_files
        
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
            assert os.path.isfile(filename_forcing), 'Cannot find forcing file [{}]'.format(filename_forcing)

            # drop variables if they are not in the config
            xarray_dataset = get_forward_data_netCDF4(filename_forcing)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_forcing)
            
            self.xarray_forcing = xarray_dataset
        else:
            self.xarray_forcing = False

        # ======================================================== #
        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            assert os.path.isfile(filename_static), 'Cannot find static file [{}]'.format(filename_static)

            # drop variables if they are not in the config
            xarray_dataset = get_forward_data_netCDF4(filename_static)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_static)
            
            self.xarray_static = xarray_dataset
        else:
            self.xarray_static = False

        # ======================================================== #
        # diagnostic file
        self.filename_diagnostic = filename_diagnostic
        
        if self.filename_diagnostic is not None:

            diagnostic_files = []
            filename_diagnostic = sorted(filename_diagnostic)
            
            for fn in filename_diagnostic:

                # drop variables if they are not in the config
                xarray_dataset = get_forward_data(filename=fn)
                xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_diagnostic)
                
                diagnostic_files.append(xarray_dataset)
                
            self.diagnostic_files = diagnostic_files
            
            assert len(self.diagnostic_files)==len(self.all_files), \
                'Mismatch between the total number of diagnostic files and upper-air files'
        else:
            self.diagnostic_files = False
            
        # ======================================================== #
        # surface files
        if filename_surface is not None:
        
            surface_files = []
            filename_surface = sorted(filename_surface)
        
            for fn in filename_surface:

                # drop variables if they are not in the config
                xarray_dataset = get_forward_data(filename=fn)
                xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_surface)
                
                surface_files.append(xarray_dataset)
                
            self.surface_files = surface_files
            
            assert len(self.surface_files)==len(self.all_files), \
                'Mismatch between the total number of surface files and upper-air files'
        else:
            self.surface_files = False
            
    
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
        # subset xarray on time dimension & load it to the memory
        
        ind_end_in_file = ind_start_in_file+self.history_len+self.forecast_len
        
        ## ERA5_subset: a xarray dataset that contains training input and target (for the current index)
        ERA5_subset = self.all_files[int(ind_file)].isel(
            time=slice(ind_start_in_file, ind_end_in_file+1)).load()
        
        if self.surface_files:
            ## subset surface variables
            surface_subset = self.surface_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file+1)).load()
    
            ## merge upper-air and surface here:
            ERA5_subset = ERA5_subset.merge(surface_subset)

        # ==================================================== #
        # split ERA5_subset into training inputs and targets + merge with forcing and static

        # the ind_end of the ERA5_subset
        ind_end_time = len(ERA5_subset['time'])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = ERA5_subset.time.values.astype('datetime64[s]').astype(int)

        # ==================================================== #
        # xarray dataset as input
        ## historical_ERA5_images: the final input

        historical_ERA5_images = ERA5_subset.isel(time=slice(0, self.history_len, self.skip_periods))

        # merge forcing inputs
        if self.xarray_forcing:
            # slice + load to the GPU
            forcing_subset_input = self.xarray_forcing.isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1))
            forcing_subset_input = forcing_subset_input.isel(time=slice(0, self.history_len, self.skip_periods)).load()

            # update
            
            forcing_subset_input['time'] = historical_ERA5_images['time']

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(forcing_subset_input)

        # merge static inputs
        if self.xarray_static:
            # expand static var on time dim
            N_time_dims = len(ERA5_subset['time'])
            static_subset_input = self.xarray_static.expand_dims(dim={"time": N_time_dims})
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords({'time': ERA5_subset['time']})

            # slice + load to the GPU
            static_subset_input = static_subset_input.isel(time=slice(0, self.history_len, self.skip_periods)).load()

            # update 
            static_subset_input['time'] = historical_ERA5_images['time']

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(static_subset_input)
        
        # ==================================================== #
        # xarray dataset as target
        ## target_ERA5_images: the final target
        
        target_ERA5_images = ERA5_subset.isel(time=slice(self.history_len, ind_end_time, self.skip_periods))

        ## merge diagnoisc input here:
        if self.diagnostic_files:
            
            # subset diagnostic variables
            diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file+1)).load()
            
            # merge into the target dataset
            target_diagnostic = diagnostic_subset.isel(time=slice(self.history_len, ind_end_time, self.skip_periods))
            target_ERA5_images = target_ERA5_images.merge(target_diagnostic)
            
        if self.one_shot is not None:
            # get the final state of the target as one-shot
            target_ERA5_images = target_ERA5_images.isel(time=slice(0, 1))

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


class ERA5(torch.utils.data.Dataset):

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

    def update_forecast_len(self, new_forecast_len):
        """Update the forecast length and recompute dependent attributes."""
        self.forecast_len = new_forecast_len
        self.total_seq_len = self.history_len + self.forecast_len

    def __getitem__(self, index):

        # Update forecast_len if needed
        if isinstance(self.max_forecast_len, int):
            self._forecast_len = self.forecast_len
            std_dev = 1.0
            new_len = int(np.random.normal(loc=self._forecast_len, scale=std_dev, size=1))
            new_len = np.clip(new_len, 1, 120)
            self.update_forecast_len(new_len)

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

        if isinstance(self.max_forecast_len, int):
            sample["forecast_hour"] = self.forecast_len
            self.forecast_len = self._forecast_len

        return sample

# Note: DistributedSequentialDataset & DistributedSequentialDataset
# are legacy; they wrap ERA5Dataset to send data batches to GPUs for
# (1 class of?) huge sharded models, but otherwise have been
# superseded by ERA5Dataset.


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


class DistributedSequentialDataset(torch.utils.data.IterableDataset):
    # https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-?usp=sharing#scrollTo=CxSCQPOMHgwo

    def __init__(self, filenames, history_len, forecast_len, skip_periods, rank, world_size, shuffle=False,
                 transform=None, rollout_p=0.0):

        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            skip_periods=skip_periods,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_fils = self.dataset.all_fils
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.filenames = filenames
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.skip_periods = skip_periods
        self.current_epoch = 0
        self.rollout_p = rollout_p

    def __len__(self):
        tlen = 0
        for bb in self.all_fils:
            tlen += (len(bb['time']) - self.forecast_len)
        return tlen

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_rollout_prob(self, p):
        self.rollout_p = p

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
                                     rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)
        sampler.set_epoch(self.current_epoch)

        for index in iter(sampler):
            result_key = find_key_for_number(index, self.meta_data_dict)
            true_ind = index - self.meta_data_dict[result_key][1]

            if true_ind > (len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 1)):
                true_ind = len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 3)

            indices = list(range(true_ind, true_ind + self.history_len + self.forecast_len))
            stop_forecast = False

            for k, ind in enumerate(indices):

                concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': [], "static": [], "TOA": []}
                sliced = xr.open_zarr(self.filenames[int(result_key)], consolidated=True).isel(
                    time=slice(ind, ind + self.history_len + self.forecast_len + 1, self.skip_periods))
                sample = {
                    'x': sliced.isel(time=slice(k, k + self.history_len, 1)),
                    'y': sliced.isel(time=slice(k + self.history_len, k + self.history_len + 1, 1)),
                    't': sliced.time.values.astype('datetime64[s]').astype(int),
                }

                if self.transform:
                    sample = self.transform(sample)

                for key in concatenated_samples.keys():
                    concatenated_samples[key] = sample[key].squeeze()

                stop_forecast = (torch.rand(1).item() < self.rollout_p)
                stop_forecast = stop_forecast or (k == self.forecast_len)

                concatenated_samples['forecast_hour'] = k
                concatenated_samples['index'] = index
                concatenated_samples['stop_forecast'] = stop_forecast

                yield concatenated_samples

                if stop_forecast:
                    break

                if (k == self.forecast_len):
                    break


class PredictForecast(torch.utils.data.IterableDataset):
    def __init__(self,
                 filenames,
                 forecasts,
                 history_len,
                 forecast_len,
                 skip_periods,
                 rank,
                 world_size,
                 shuffle=False,
                 transform=None,
                 rollout_p=0.0,
                 start_time=None,
                 stop_time=None,
                 which_forecast=None):

        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            skip_periods=skip_periods,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_files = self.dataset.all_fils
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.filenames = filenames
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.skip_periods = skip_periods
        self.current_epoch = 0
        self.rollout_p = rollout_p
        self.forecasts = forecasts
        self.skip_periods = skip_periods if skip_periods is not None else 1
        self.which_forecast = which_forecast

    def find_start_stop_indices(self, index):
        start_time = self.forecasts[index][0]
        date_object = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        shifted_hours = self.skip_periods * self.history_len
        date_object = date_object - datetime.timedelta(hours=shifted_hours)
        self.forecasts[index][0] = date_object.strftime('%Y-%m-%d %H:%M:%S')

        datetime_objs = [np.datetime64(date) for date in self.forecasts[index]]
        start_time, stop_time = [str(datetime_obj) + '.000000000' for datetime_obj in datetime_objs]
        self.start_time = np.datetime64(start_time).astype(datetime.datetime)
        self.stop_time = np.datetime64(stop_time).astype(datetime.datetime)

        info = {}

        for idx, dataset in enumerate(self.all_files):
            start_time = np.datetime64(dataset['time'].min().values).astype(datetime.datetime)
            stop_time = np.datetime64(dataset['time'].max().values).astype(datetime.datetime)
            track_start = False
            track_stop = False

            if start_time <= self.start_time <= stop_time:
                # Start time is in this file, use start time index
                dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                start_idx = np.searchsorted(dataset, self.start_time)
                start_idx = max(0, min(start_idx, len(dataset) - 1))
                track_start = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                start_idx = 0
                track_start = True

            if start_time <= self.stop_time <= stop_time:
                # Stop time is in this file, use stop time index
                if isinstance(dataset, np.ndarray):
                    pass
                else:
                    dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                stop_idx = np.searchsorted(dataset, self.stop_time)
                stop_idx = max(0, min(stop_idx, len(dataset) - 1))
                track_stop = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                stop_idx = len(dataset) - 1
                track_stop = True

            # Only include files that overlap the time range
            if track_start and track_stop:
                info[idx] = ((idx, start_idx), (idx, stop_idx))

        indices = []
        for dataset_idx, (start, stop) in info.items():
            for i in range(start[1], stop[1] + 1):
                indices.append((start[0], i))
        return indices

    def __len__(self):
        return len(self.forecasts)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
                                     rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)

        for index in sampler:

            data_lookup = self.find_start_stop_indices(index)

            for k, (file_key, time_key) in enumerate(data_lookup):
                concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': []}
                sliced_x = xr.open_zarr(self.filenames[file_key], consolidated=True).isel(
                    time=slice(time_key, time_key + self.history_len + 1))

                # Check if additional data from the next file is needed
                if len(sliced_x['time']) < self.history_len + 1:
                    # Load excess data from the next file
                    next_file_idx = self.filenames.index(self.filenames[file_key]) + 1
                    if next_file_idx == len(self.filenames):
                        raise OSError("You have reached the end of the available data. Exiting.")
                    sliced_x_next = xr.open_zarr(
                        self.filenames[next_file_idx],
                        consolidated=True).isel(time=slice(0, self.history_len + 1 - len(sliced_x['time'])))

                    # Concatenate excess data from the next file with the current data
                    sliced_x = xr.concat([sliced_x, sliced_x_next], dim='time')

                sample_x = {
                    'x': sliced_x.isel(time=slice(0, self.history_len)),
                    'y': sliced_x.isel(time=slice(self.history_len, self.history_len + 1))  # Fetch y data for t(i+1)
                }

                if self.transform:
                    sample_x = self.transform(sample_x)
                    # Add static vars, if any, to the return dictionary
                    if "static" in sample_x:
                        concatenated_samples["static"] = []
                    if "TOA" in sample_x:
                        concatenated_samples["TOA"] = []

                for key in concatenated_samples.keys():
                    concatenated_samples[key] = sample_x[key].squeeze(0) if self.history_len == 1 else sample_x[key]

                concatenated_samples['forecast_hour'] = k + 1
                concatenated_samples['stop_forecast'] = (
                            k == (len(data_lookup) - self.history_len - 1))  # Adjust stopping condition
                concatenated_samples['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]

                yield concatenated_samples

                if concatenated_samples['stop_forecast']:
                    break


class PredictForecastRollout(torch.utils.data.IterableDataset):
    def __init__(self,
                 filenames,
                 forecasts,
                 history_len,
                 forecast_len,
                 skip_periods,
                 rank,
                 world_size,
                 shuffle=False,
                 transform=None,
                 rollout_p=0.0,
                 start_time=None,
                 stop_time=None,
                 which_forecast=None):

        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            skip_periods=skip_periods,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_files = self.dataset.all_fils
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.filenames = filenames
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.skip_periods = skip_periods
        self.current_epoch = 0
        self.rollout_p = rollout_p
        self.forecasts = forecasts
        self.skip_periods = skip_periods if skip_periods is not None else 1
        self.which_forecast = which_forecast

    def find_start_stop_indices(self, index):
        start_time = self.forecasts[index][0]
        date_object = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        shifted_hours = self.skip_periods * self.history_len
        date_object = date_object - datetime.timedelta(hours=shifted_hours)
        self.forecasts[index][0] = date_object.strftime('%Y-%m-%d %H:%M:%S')

        datetime_objs = [np.datetime64(date) for date in self.forecasts[index]]
        start_time, stop_time = [str(datetime_obj) + '.000000000' for datetime_obj in datetime_objs]
        self.start_time = np.datetime64(start_time).astype(datetime.datetime)
        self.stop_time = np.datetime64(stop_time).astype(datetime.datetime)

        info = {}

        for idx, dataset in enumerate(self.all_files):
            start_time = np.datetime64(dataset['time'].min().values).astype(datetime.datetime)
            stop_time = np.datetime64(dataset['time'].max().values).astype(datetime.datetime)
            track_start = False
            track_stop = False

            if start_time <= self.start_time <= stop_time:
                # Start time is in this file, use start time index
                dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                start_idx = np.searchsorted(dataset, self.start_time)
                start_idx = max(0, min(start_idx, len(dataset) - 1))
                track_start = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                start_idx = 0
                track_start = True

            if start_time <= self.stop_time <= stop_time:
                # Stop time is in this file, use stop time index
                if isinstance(dataset, np.ndarray):
                    pass
                else:
                    dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                stop_idx = np.searchsorted(dataset, self.stop_time)
                stop_idx = max(0, min(stop_idx, len(dataset) - 1))
                track_stop = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                stop_idx = len(dataset) - 1
                track_stop = True

            # Only include files that overlap the time range
            if track_start and track_stop:
                info[idx] = ((idx, start_idx), (idx, stop_idx))

        indices = []
        for dataset_idx, (start, stop) in info.items():
            for i in range(start[1], stop[1] + 1):
                indices.append((start[0], i))
        return indices

    def __len__(self):
        return len(self.forecasts)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
                                     rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)

        for index in sampler:

            data_lookup = self.find_start_stop_indices(index)

            for k, (file_key, time_key) in enumerate(data_lookup):
                concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': []}
                sliced_x = xr.open_zarr(self.filenames[file_key], consolidated=True).isel(
                    time=slice(time_key, time_key + self.history_len + 1))

                # Check if additional data from the next file is needed
                if len(sliced_x['time']) < self.history_len + 1:
                    # Load excess data from the next file
                    next_file_idx = self.filenames.index(self.filenames[file_key]) + 1
                    if next_file_idx == len(self.filenames):
                        raise OSError("You have reached the end of the available data. Exiting.")
                    sliced_x_next = xr.open_zarr(
                        self.filenames[next_file_idx],
                        consolidated=True).isel(time=slice(0, self.history_len + 1 - len(sliced_x['time'])))

                    # Concatenate excess data from the next file with the current data
                    sliced_x = xr.concat([sliced_x, sliced_x_next], dim='time')

                sample_x = {
                    'x': sliced_x.isel(time=slice(0, self.history_len)),
                    'y': sliced_x.isel(time=slice(self.history_len, self.history_len + 1))  # Fetch y data for t(i+1)
                }

                if self.transform:
                    sample_x = self.transform(sample_x)
                    # Add static vars, if any, to the return dictionary
                    if "static" in sample_x:
                        concatenated_samples["static"] = []
                    if "TOA" in sample_x:
                        concatenated_samples["TOA"] = []

                for key in concatenated_samples.keys():
                    concatenated_samples[key] = sample_x[key].squeeze(0) if self.history_len == 1 else sample_x[key]

                concatenated_samples['forecast_hour'] = k + 1
                concatenated_samples['stop_forecast'] = (
                            k == (len(data_lookup) - self.history_len - 1))  # Adjust stopping condition
                concatenated_samples['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]

                yield concatenated_samples

                break


class PredictForecastQuantile(PredictForecast):

    def __init__(self,
                 conf,
                 filenames,
                 forecasts,
                 history_len,
                 forecast_len,
                 skip_periods,
                 rank,
                 world_size,
                 shuffle=False,
                 transform=None,
                 rollout_p=0.0,
                 start_time=None,
                 stop_time=None):

        from credit.transforms import load_transforms

        transform = load_transforms(conf)

        self.dataset = Dataset_BridgeScaler(
            conf,
            conf_dataset='bs_years_test',
            transform=transform
        )

        # Need information on the saved files
        self.all_files = [get_forward_data(filename=fn) for fn in sorted(filenames)]
        # Set data places:
        indo = 0
        self.meta_data_dict = {}
        for ee, bb in enumerate(self.all_files):
            self.meta_data_dict[str(ee)] = [len(bb['time']), indo, indo + len(bb['time'])]
            indo += len(bb['time']) + 1

        self.history_len = history_len
        self.forecast_len = forecast_len
        self.filenames = filenames
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.skip_periods = skip_periods
        self.current_epoch = 0
        self.rollout_p = rollout_p
        self.forecasts = forecasts
        self.skip_periods = skip_periods if skip_periods is not None else 1

    def find_start_stop_indices(self, index):
        start_time = self.forecasts[index][0]
        date_object = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        shifted_hours = self.skip_periods * self.history_len
        date_object = date_object - datetime.timedelta(hours=shifted_hours)
        self.forecasts[index][0] = date_object.strftime('%Y-%m-%d %H:%M:%S')

        datetime_objs = [np.datetime64(date) for date in self.forecasts[index]]
        start_time, stop_time = [str(datetime_obj) + '.000000000' for datetime_obj in datetime_objs]
        self.start_time = np.datetime64(start_time).astype(datetime.datetime)
        self.stop_time = np.datetime64(stop_time).astype(datetime.datetime)

        info = {}

        for idx, dataset in enumerate(self.all_files):
            start_time = np.datetime64(dataset['time'].min().values).astype(datetime.datetime)
            stop_time = np.datetime64(dataset['time'].max().values).astype(datetime.datetime)
            track_start = False
            track_stop = False

            if start_time <= self.start_time <= stop_time:
                # Start time is in this file, use start time index
                dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                start_idx = np.searchsorted(dataset, self.start_time)
                start_idx = max(0, min(start_idx, len(dataset) - 1))
                track_start = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                start_idx = 0
                track_start = True

            if start_time <= self.stop_time <= stop_time:
                # Stop time is in this file, use stop time index
                if isinstance(dataset, np.ndarray):
                    pass
                else:
                    dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                stop_idx = np.searchsorted(dataset, self.stop_time)
                stop_idx = max(0, min(stop_idx, len(dataset) - 1))
                track_stop = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                stop_idx = len(dataset) - 1
                track_stop = True

            # Only include files that overlap the time range
            if track_start and track_stop:
                info[idx] = ((idx, start_idx), (idx, stop_idx))

        indices = []
        for dataset_idx, (start, stop) in info.items():
            for i in range(start[1], stop[1] + 1):
                indices.append((start[0], i))
        return indices

    def __len__(self):
        return len(self.forecasts)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
                                     rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)

        for index in sampler:

            data_lookup = self.find_start_stop_indices(index)

            for k, (file_key, time_key) in enumerate(data_lookup):
                concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': []}
                sliced_x = xr.open_zarr(self.filenames[file_key], consolidated=True).isel(
                    time=slice(time_key, time_key + self.history_len + 1))

                # Check if additional data from the next file is needed
                if len(sliced_x['time']) < self.history_len + 1:
                    # Load excess data from the next file
                    next_file_idx = self.filenames.index(self.filenames[file_key]) + 1
                    if next_file_idx == len(self.filenames):
                        raise OSError("You have reached the end of the available data. Exiting.")
                    sliced_x_next = xr.open_zarr(
                        self.filenames[next_file_idx],
                        consolidated=True).isel(time=slice(0, self.history_len + 1 - len(sliced_x['time'])))

                    # Concatenate excess data from the next file with the current data
                    sliced_x = xr.concat([sliced_x, sliced_x_next], dim='time')

                sample_x = {
                    'x': sliced_x.isel(time=slice(0, self.history_len)),
                    'y': sliced_x.isel(time=slice(self.history_len, self.history_len + 1))  # Fetch y data for t(i+1)
                }

                if self.transform:
                    sample_x = self.transform(sample_x)
                    # Add static vars, if any, to the return dictionary
                    if "static" in sample_x:
                        concatenated_samples["static"] = []
                    if "TOA" in sample_x:
                        concatenated_samples["TOA"] = []

                for key in concatenated_samples.keys():
                    concatenated_samples[key] = sample_x[key].squeeze(0) if self.history_len == 1 else sample_x[key]

                concatenated_samples['forecast_hour'] = k + 1
                concatenated_samples['stop_forecast'] = (
                            k == (len(data_lookup) - self.history_len - 1))  # Adjust stopping condition
                concatenated_samples['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]

                yield concatenated_samples

                if concatenated_samples['stop_forecast']:
                    break
