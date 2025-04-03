"""Data.py contains modules for processing training data.

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
"""

# system tools
from typing import TypedDict, Union, List

# data utils
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import cftime

# Pytorch utils
import torch
import torch.utils.data
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

#
Array = Union[np.ndarray, xr.DataArray]
IMAGE_ATTR_NAMES = ("historical_ERA5_images", "target_ERA5_images")


def ensure_numpy_datetime(value):
    """
    Converts an input value (or array) to numpy.datetime64.
    Handles numpy arrays, pandas timestamps, cftime objects, and strings.
    """
    # If the value is an array, extract the first element
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()  # Extract scalar value
        else:
            raise TypeError(f"Cannot convert array with multiple elements: {value}")

    if isinstance(value, np.datetime64):
        return value  # Already correct
    elif isinstance(value, pd.Timestamp):
        return np.datetime64(value)  # Convert from pandas Timestamp
    elif isinstance(value, str):
        try:
            return np.datetime64(value)  # Convert from string
        except ValueError:
            pass  # If it fails, let it fall through
    elif isinstance(value, cftime.datetime):
        return np.datetime64(value.strftime('%Y-%m-%dT%H:%M:%S'))  # Convert from cftime
    elif isinstance(value, object):  # Catch-all for potential unexpected object types
        try:
            return np.datetime64(pd.to_datetime(value))
        except Exception:
            raise TypeError(f"Cannot convert type {type(value)} to numpy.datetime64")
    else:
        raise TypeError(f"Unsupported type {type(value)} for datetime conversion")


def generate_datetime(start_time, end_time, interval_hr):
    """Generate a list of datetime.datetime based on stat, end times, and hour interval.

    Args:
        start_time (datetime.datetime): start time
        end_time (datetime.datetime): end time
        interval_hr (int): hour interval

    """
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
    """Convert hour to nanoseconds."""
    # hr * min_per_hr * sec_per_min * nanosec_per_sec
    return input_hr * 60 * 60 * 1000000000


def nanoseconds_to_year(nanoseconds_value):
    """Given datetime info as nanoseconds, compute which year it belongs to."""
    return (
        np.datetime64(nanoseconds_value, "ns").astype("datetime64[Y]").astype(int)
        + 1970
    )


def extract_month_day_hour(dates):
    """Given an 1-d array of np.datatime64[ns], extract their mon, day, hr into a zipped list."""
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    days = (
        (dates - dates.astype("datetime64[M]") + 1).astype("timedelta64[D]").astype(int)
    )
    hours = dates.astype("datetime64[h]").astype(int) % 24
    return list(zip(months, days, hours))


def find_common_indices(list1, list2):
    """Find indices of common elements between two lists."""
    # Find common elements
    common_elements = set(list1).intersection(set(list2))

    # Find indices of common elements in both lists
    indices_list1 = [i for i, x in enumerate(list1) if x in common_elements]
    indices_list2 = [i for i, x in enumerate(list2) if x in common_elements]

    return indices_list1, indices_list2


def concat_and_reshape(x1, x2):
    """Flattening the "level" coordinate of upper-air variables and concatenate it will surface variables."""
    x1 = x1.view(
        x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5]
    )
    x_concat = torch.cat((x1, x2), dim=2)
    return x_concat.permute(0, 2, 1, 3, 4)


def reshape_only(x1):
    """Flattening the "level" coordinate of upper-air variables.

    As in "concat_and_reshape", but no concat.
    """
    x1 = x1.view(
        x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5]
    )
    return x1.permute(0, 2, 1, 3, 4)


def get_forward_data(filename) -> xr.Dataset:
    """Check nc vs. zarr files and open file as xr.Dataset."""
    if filename[-3:] == ".nc" or filename[-4:] == ".nc4":
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


def flatten_list(list_of_lists):
    """Flatten a list of lists.

    Parameters
    ----------
    - list_of_lists (list): A list containing sublists.

    Returns
    -------
    - flattened_list (list): A flattened list containing all elements from sublists.

    """
    return [item for sublist in list_of_lists for item in sublist]


def generate_integer_list_around(number, spacing=10):
    """Generate a list of integers on either side of a given number with a specified spacing.

    Parameters
    ----------
    - number (int): The central number around which the list is generated.
    - spacing (int): The spacing between consecutive integers in the list. Default is 10.

    Returns
    -------
    - integer_list (list): List of integers on either side of the given number.

    """
    lower_limit = number - spacing
    upper_limit = number + spacing + 1  # Adding 1 to include the upper limit
    integer_list = list(range(lower_limit, upper_limit))

    return integer_list


def find_key_for_number(input_number, data_dict):
    """Find the key in the dictionary based on the given number.

    Parameters
    ----------
    - input_number (int): The number to search for in the dictionary.
    - data_dict (dict): The dictionary with keys and corresponding value lists.

    Returns
    -------
    - key_found (str): The key in the dictionary where the input number falls within the specified range.

    """
    for key, value_list in data_dict.items():
        if value_list[1] <= input_number <= value_list[2]:
            return key

    # Return None if the number is not within any range
    return None


def drop_var_from_dataset(xarray_dataset, varname_keep):
    """Preserve a given set of variables from an xarray.Dataset, and drop the rest.

    It will raise error if `varname_key` is missing from `xarray_dataset`.
    """
    varname_all = list(xarray_dataset.keys())

    for varname in varname_all:
        if varname not in varname_keep:
            xarray_dataset = xarray_dataset.drop_vars(varname)

    varname_clean = list(xarray_dataset.keys())

    varname_diff = list(set(varname_keep) - set(varname_clean))
    assert len(varname_diff) == 0, "Variable name: {} missing".format(varname_diff)

    return xarray_dataset


def keep_dataset_vars(xarray_dataset: xr.Dataset, varnames_keep: List[str]):
    """Return a version of an xarray dataset with only a selected subset of variables.

    Args:
        xarray_dataset (xr.Dataset): The xarray dataset.
        varnames_keep (List[str]): a list of variable names to be kept.

    Returns:
        xr.Dataset with only the variables in varnames_keep included.

    """
    return xarray_dataset[varnames_keep]


class ERA5_and_Forcing_Dataset(torch.utils.data.Dataset):
    """A Pytorch Dataset class that works on the following kinds of variables.

    * upper-air variables (time, level, lat, lon)
    * surface variables (time, lat, lon)
    * dynamic forcing variables (time, lat, lon)
    * forcing variables (time, lat, lon)
    * diagnostic variables (time, lat, lon)
    * static variables (lat, lon).
    """

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
        """Initialize the ERA5_and_Forcing_Dataset.

        Args:
            varname_upper_air (list): List of upper air variable names.
            varname_surface (list): List of surface variable names.
            varname_dyn_forcing (list): List of dynamic forcing variable names.
            varname_forcing (list): List of forcing variable names.
            varname_static (list): List of static variable names.
            varname_diagnostic (list): List of diagnostic variable names.
            filenames (list): List of filenames for upper air data.
            filename_surface (list, optional): List of filenames for surface data.
            filename_dyn_forcing (list, optional): List of filenames for dynamic forcing data.
            filename_forcing (str, optional): Filename for forcing data.
            filename_static (str, optional): Filename for static data.
            filename_diagnostic (list, optional): List of filenames for diagnostic data.
            history_len (int, optional): Length of the history sequence. Default is 2.
            forecast_len (int, optional): Length of the forecast sequence. Default is 0.
            transform (callable, optional): Transformation function to apply to the data.
            seed (int, optional): Random seed for reproducibility. Default is 42.
            skip_periods (int, optional): Number of periods to skip between samples.
            one_shot(bool, optional): Whether to return all states or just
                                    the final state of the training target. Default is None
            max_forecast_len (int, optional): Maximum length of the forecast sequence.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.

        Returns:
            sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.

        """
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
        self.ERA5_indices = {}  # <------ change
        for ind_file, ERA5_xarray in enumerate(self.all_files):
            # [number of samples, ind_start, ind_end]
            self.ERA5_indices[str(ind_file)] = [
                len(ERA5_xarray["time"]),
                ind_start,
                ind_start + len(ERA5_xarray["time"]),
            ]
            ind_start += len(ERA5_xarray["time"]) + 1

        # ======================================================== #
        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            ds = get_forward_data(filename_forcing)
            ds_forcing = drop_var_from_dataset(
                ds, varname_forcing
            ).load()  # <---- load in static

            self.xarray_forcing = ds_forcing
        else:
            self.xarray_forcing = False

        # ======================================================== #
        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            # drop variables if they are not in the config
            ds = get_forward_data(filename_static)
            ds_static = drop_var_from_dataset(
                ds, varname_static
            ).load()  # <---- load in static

            self.xarray_static = ds_static
        else:
            self.xarray_static = False

    def __post_init__(self):
        """Calculate total sequence length after init."""
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        """Length of Dataset."""
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += len(ERA5_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def __getitem__(self, index):
        """Get single item from the dataset."""
        # ========================================================================== #
        # cross-year indices --> the index of the year + indices within that year

        # select the ind_file based on the iter index
        ind_file = find_key_for_number(index, self.ERA5_indices)

        # get the ind within the current file
        ind_start = self.ERA5_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        # handle out-of-bounds
        ind_largest = len(self.all_files[int(ind_file)]["time"]) - (
            self.history_len + self.forecast_len + 1
        )
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # ========================================================================== #
        # subset xarray on time dimension

        ind_end_in_file = ind_start_in_file + self.history_len + self.forecast_len

        ## ERA5_subset: a xarray dataset that contains training input and target (for the current batch)
        ERA5_subset = self.all_files[int(ind_file)].isel(
            time=slice(ind_start_in_file, ind_end_in_file + 1)
        )  # .load() NOT load into memory

        # ========================================================================== #
        # merge surface into the dataset

        if self.surface_files:
            ## subset surface variables
            surface_subset = self.surface_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )  # .load() NOT load into memory

            ## merge upper-air and surface here:
            ERA5_subset = ERA5_subset.merge(
                surface_subset
            )  # <-- lazy merge, ERA5 and surface both not loaded

        # ==================================================== #
        # split ERA5_subset into training inputs and targets
        #   + merge with dynamic forcing, forcing, and static

        # the ind_end of the ERA5_subset
        ind_end_time = len(ERA5_subset["time"])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = ERA5_subset.time.values.astype("datetime64[s]").astype(int)

        # ==================================================== #
        # xarray dataset as input
        ## historical_ERA5_images: the final input

        historical_ERA5_images = ERA5_subset.isel(
            time=slice(0, self.history_len, self.skip_periods)
        ).load()  # <-- load into memory

        # ========================================================================== #
        # merge dynamic forcing inputs
        if self.dyn_forcing_files:
            dyn_forcing_subset = self.dyn_forcing_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(
                time=slice(0, self.history_len, self.skip_periods)
            ).load()  # <-- load into memory

            historical_ERA5_images = historical_ERA5_images.merge(dyn_forcing_subset)

        # ========================================================================== #
        # merge forcing inputs
        if self.xarray_forcing:
            # ------------------------------------------------------------------------------- #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(
                np.array(self.xarray_forcing["time"])
            )
            month_day_inputs = extract_month_day_hour(
                np.array(historical_ERA5_images["time"])
            )  # <-- upper air
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = self.xarray_forcing.isel(
                time=ind_forcing
            )  # .load() # <-- loadded in init
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input["time"] = historical_ERA5_images["time"]
            # ------------------------------------------------------------------------------- #

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(forcing_subset_input)

        # ========================================================================== #
        # merge static inputs
        if self.xarray_static:
            # expand static var on time dim
            N_time_dims = len(ERA5_subset["time"])
            static_subset_input = self.xarray_static.expand_dims(
                dim={"time": N_time_dims}
            )
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords(
                {"time": ERA5_subset["time"]}
            )

            # slice + load to the GPU
            static_subset_input = static_subset_input.isel(
                time=slice(0, self.history_len, self.skip_periods)
            )  # .load() # <-- loaded in init

            # update
            static_subset_input["time"] = historical_ERA5_images["time"]

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(static_subset_input)

        # ==================================================== #
        # xarray dataset as target
        ## target_ERA5_images: the final target

        if self.one_shot is not None:
            # one_shot is True (on), go straight to the last element
            target_ERA5_images = ERA5_subset.isel(
                time=slice(-1, None)
            ).load()  # <-- load into memory

            ## merge diagnoisc input here:
            if self.diagnostic_files:
                diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file + 1)
                )

                diagnostic_subset = diagnostic_subset.isel(
                    time=slice(-1, None)
                ).load()  # <-- load into memory

                target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

        else:
            # one_shot is None (off), get the full target length based on forecast_len
            target_ERA5_images = ERA5_subset.isel(
                time=slice(self.history_len, ind_end_time, self.skip_periods)
            ).load()  # <-- load into memory

            ## merge diagnoisc input here:
            if self.diagnostic_files:
                # subset diagnostic variables
                diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file + 1)
                )

                diagnostic_subset = diagnostic_subset.isel(
                    time=slice(self.history_len, ind_end_time, self.skip_periods)
                ).load()  # <-- load into memory

                # merge into the target dataset
                target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

        # ------------------------------------------------------------------ #
        # sst forcing operations
        if self.sst_forcing is not None:
            # get xr.dataset keys
            varname_skt = self.sst_forcing["varname_skt"]
            varname_ocean_mask = self.sst_forcing["varname_ocean_mask"]

            # get xr.dataarray from the dataset
            ocean_mask = historical_ERA5_images[varname_ocean_mask]
            input_skt = historical_ERA5_images[varname_skt]
            target_skt = target_ERA5_images[varname_skt]

            # for multi-input cases, use time=-1 ocean mask for all times
            if self.history_len > 1:
                ocean_mask[: self.history_len - 1] = ocean_mask.isel(time=-1)

            # get ocean mask
            ocean_mask_bool = ocean_mask.isel(time=-1) == 0

            # for multi-input cases, use time=-1 ocean SKT for all times
            if self.history_len > 1:
                input_skt[: self.history_len - 1] = input_skt[
                    : self.history_len - 1
                ].where(~ocean_mask_bool, input_skt.isel(time=-1))

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
            datetime_index=datetime_as_number,
        )

        # ==================================== #
        # data normalization
        if self.transform:
            sample = self.transform(sample)

        # assign sample index
        sample["index"] = index

        return sample


class ERA5_Dataset_Distributed(torch.utils.data.Dataset):
    """ERA5 Dataset for Distributed training (legacy)."""

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
        """Initialize the ERA5_and_Forcing_Dataset.

        Args:
            varname_upper_air (list): List of upper air variable names.
            varname_surface (list): List of surface variable names.
            varname_dyn_forcing (list): List of dynamic forcing variable names.
            varname_forcing (list): List of forcing variable names.
            varname_static (list): List of static variable names.
            varname_diagnostic (list): List of diagnostic variable names.
            filenames (list): List of filenames for upper air data.
            filename_surface (list, optional): List of filenames for surface data.
            filename_dyn_forcing (list, optional): List of filenames for dynamic forcing data.
            filename_forcing (str, optional): Filename for forcing data.
            filename_static (str, optional): Filename for static data.
            filename_diagnostic (list, optional): List of filenames for diagnostic data.
            history_len (int, optional): Length of the history sequence. Default is 2.
            forecast_len (int, optional): Length of the forecast sequence. Default is 0.
            transform (callable, optional): Transformation function to apply to the data.
            seed (int, optional): Random seed for reproducibility. Default is 42.
            skip_periods (int, optional): Number of periods to skip between samples.
            one_shot(bool, optional): Whether to return all states or just
                the final state of the training target. Default is None
            max_forecast_len (int, optional): Maximum length of the forecast sequence.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.

        Returns:
            sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.

        """
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.varname_upper_air = varname_upper_air
        self.varname_surface = varname_surface
        self.varname_dyn_forcing = varname_dyn_forcing
        self.varname_forcing = varname_forcing
        self.varname_static = varname_static

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
        self.filenames = sorted(filenames)

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
        self.ERA5_indices = {}  # <------ change
        for ind_file, ERA5_xarray in enumerate(self.all_files):
            # [number of samples, ind_start, ind_end]
            self.ERA5_indices[str(ind_file)] = [
                len(ERA5_xarray["time"]),
                ind_start,
                ind_start + len(ERA5_xarray["time"]),
            ]
            ind_start += len(ERA5_xarray["time"]) + 1

        # ======================================================== #
        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            ds = get_forward_data(filename_forcing)
            ds_forcing = drop_var_from_dataset(
                ds, varname_forcing
            ).load()  # <---- load in static

            self.xarray_forcing = ds_forcing
        else:
            self.xarray_forcing = False

        # ======================================================== #
        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            # drop variables if they are not in the config
            ds = get_forward_data(filename_static)
            ds_static = drop_var_from_dataset(
                ds, varname_static
            ).load()  # <---- load in static

            self.xarray_static = ds_static
        else:
            self.xarray_static = False

    def __post_init__(self):
        """Calculate total sequence length."""
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        """Length of dataset."""
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += len(ERA5_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def __getitem__(self, index):
        """Get item.

        Args:
            index: index of timestep

        Returns:
            pytorch Tensor containing a full state.

        """
        # ========================================================================== #
        # cross-year indices --> the index of the year + indices within that year

        # select the ind_file based on the iter index
        ind_file = find_key_for_number(index, self.ERA5_indices)

        # get the ind within the current file
        ind_start = self.ERA5_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        # handle out-of-bounds
        ind_largest = len(self.all_files[int(ind_file)]["time"]) - (
            self.history_len + self.forecast_len + 1
        )
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # ========================================================================== #
        # subset xarray on time dimension

        ind_end_in_file = ind_start_in_file + self.history_len + self.forecast_len

        ## ERA5_subset: a xarray dataset that contains training input and target (for the current batch)
        ERA5_subset = self.all_files[int(ind_file)].isel(
            time=slice(ind_start_in_file, ind_end_in_file + 1)
        )  # .load() NOT load into memory

        # ========================================================================== #
        # merge surface into the dataset

        if self.surface_files:
            ## subset surface variables
            surface_subset = self.surface_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )  # .load() NOT load into memory

            ## merge upper-air and surface here:
            ERA5_subset = ERA5_subset.merge(
                surface_subset
            )  # <-- lazy merge, ERA5 and surface both not loaded

        # ==================================================== #
        # split ERA5_subset into training inputs and targets
        #   + merge with dynamic forcing, forcing, and static

        # the ind_end of the ERA5_subset
        ind_end_time = len(ERA5_subset["time"])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = ERA5_subset.time.values.astype("datetime64[s]").astype(int)

        # ==================================================== #
        # xarray dataset as input
        ## historical_ERA5_images: the final input

        historical_ERA5_images = ERA5_subset.isel(
            time=slice(0, self.history_len, self.skip_periods)
        ).load()  # <-- load into memory

        # ========================================================================== #
        # merge dynamic forcing inputs
        if self.dyn_forcing_files:
            dyn_forcing_subset = self.dyn_forcing_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(
                time=slice(0, self.history_len, self.skip_periods)
            ).load()  # <-- load into memory

            historical_ERA5_images = historical_ERA5_images.merge(dyn_forcing_subset)

        # ========================================================================== #
        # merge forcing inputs
        if self.xarray_forcing:
            # ------------------------------------------------------------------------------- #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(
                np.array(self.xarray_forcing["time"])
            )
            month_day_inputs = extract_month_day_hour(
                np.array(historical_ERA5_images["time"])
            )  # <-- upper air
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = self.xarray_forcing.isel(
                time=ind_forcing
            )  # .load() # <-- loadded in init
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input["time"] = historical_ERA5_images["time"]
            # ------------------------------------------------------------------------------- #

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(forcing_subset_input)

        # ========================================================================== #
        # merge static inputs
        if self.xarray_static:
            # expand static var on time dim
            N_time_dims = len(ERA5_subset["time"])
            static_subset_input = self.xarray_static.expand_dims(
                dim={"time": N_time_dims}
            )
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords(
                {"time": ERA5_subset["time"]}
            )

            # slice + load to the GPU
            static_subset_input = static_subset_input.isel(
                time=slice(0, self.history_len, self.skip_periods)
            )  # .load() # <-- loaded in init

            # update
            static_subset_input["time"] = historical_ERA5_images["time"]

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(static_subset_input)

        # ==================================================== #
        # xarray dataset as target
        ## target_ERA5_images: the final target

        if self.one_shot is not None:
            # one_shot is True (on), go straight to the last element
            target_ERA5_images = ERA5_subset.isel(
                time=slice(-1, None)
            ).load()  # <-- load into memory

            ## merge diagnoisc input here:
            if self.diagnostic_files:
                diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file + 1)
                )

                diagnostic_subset = diagnostic_subset.isel(
                    time=slice(-1, None)
                ).load()  # <-- load into memory

                target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

        else:
            # one_shot is None (off), get the full target length based on forecast_len
            target_ERA5_images = ERA5_subset.isel(
                time=slice(self.history_len, ind_end_time, self.skip_periods)
            ).load()  # <-- load into memory

            ## merge diagnoisc input here:
            if self.diagnostic_files:
                # subset diagnostic variables
                diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file + 1)
                )

                diagnostic_subset = diagnostic_subset.isel(
                    time=slice(self.history_len, ind_end_time, self.skip_periods)
                ).load()  # <-- load into memory

                # merge into the target dataset
                target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

        # ------------------------------------------------------------------ #
        # sst forcing operations
        if self.sst_forcing is not None:
            # get xr.dataset keys
            varname_skt = self.sst_forcing["varname_skt"]
            varname_ocean_mask = self.sst_forcing["varname_ocean_mask"]

            # get xr.dataarray from the dataset
            ocean_mask = historical_ERA5_images[varname_ocean_mask]
            input_skt = historical_ERA5_images[varname_skt]
            target_skt = target_ERA5_images[varname_skt]

            # for multi-input cases, use time=-1 ocean mask for all times
            if self.history_len > 1:
                ocean_mask[: self.history_len - 1] = ocean_mask.isel(time=-1)

            # get ocean mask
            ocean_mask_bool = ocean_mask.isel(time=-1) == 0

            # for multi-input cases, use time=-1 ocean SKT for all times
            if self.history_len > 1:
                input_skt[: self.history_len - 1] = input_skt[
                    : self.history_len - 1
                ].where(~ocean_mask_bool, input_skt.isel(time=-1))

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
            datetime_index=datetime_as_number,
        )

        # ==================================== #
        # data normalization
        if self.transform:
            sample = self.transform(sample)

        # assign sample index
        sample["index"] = index

        return sample


class Predict_Dataset(torch.utils.data.IterableDataset):
    """Same as ERA5_and_Forcing_Dataset() but work with old rollout_to_netcdf.py."""

    def __init__(
        self,
        conf,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames,
        filename_surface,
        filename_dyn_forcing,
        filename_forcing,
        filename_static,
        filename_diagnostic,
        fcst_datetime,
        history_len,
        rank,
        world_size,
        transform=None,
        rollout_p=0.0,
        which_forecast=None,
    ):
        """Normalize via quantile bridgescaler.

        Normalize via provided scaler file/s.

        Args:
            conf (str): path to config file.
            varname_upper_air (list): list of upper air variables.
            varname_surface (list): list of surface variables.
            varname_dyn_forcing (list): list of dynamic forcing variables.
            varname_forcing (list): list of forcing variables.
            varname_static (list): list of static variables.
            varname_diagnostic (list): list of diagnostic variables.
            filenames (str): path to upper air variables file.
            filename_surface (str): path surface variables file.
            filename_dyn_forcing (str): path to dynamic forcing variables file.
            filename_forcing (str): path to forcing variables file.
            filename_static (str): path to static variables file.
            filename_diagnostic (str): path to diagnostic variables file.
            fcst_datetime (str): initial dates for forecasts.
            hist_len (bool): state-in-state-out.
            rank (int): Which MPI worker is being used.
            world_size (int): Number of MPI ranks (processes).
            transform (function): Function used to normalize and convert to tensors.
            rollout_p (): Probability of a rollout.
            which_forecast ():

        Attributes:
            current_epoch (int): current epoch.
            lead_time_periods (): .
            skip_periods (): .

        """
        ## no diagnostics because they are output only
        # varname_diagnostic = None

        self.rank = rank
        self.world_size = world_size
        self.transform = transform
        self.history_len = history_len
        self.init_datetime = fcst_datetime

        self.which_forecast = (
            which_forecast  # <-- got from the old roll-out script. Dont know
        )

        # -------------------------------------- #
        # file names
        self.filenames = filenames  # <------------------------ a list of files
        self.filename_surface = filename_surface  # <---------- a list of files
        self.filename_dyn_forcing = filename_dyn_forcing  # <-- a list of files
        self.filename_forcing = filename_forcing  # <-- single file
        self.filename_static = filename_static  # <---- single file
        self.filename_diagnostic = filename_diagnostic  # <---- single file

        # -------------------------------------- #
        # var names
        self.varname_upper_air = varname_upper_air
        self.varname_surface = varname_surface
        self.varname_dyn_forcing = varname_dyn_forcing
        self.varname_forcing = varname_forcing
        self.varname_static = varname_static
        self.varname_diagnostic = varname_diagnostic

        # ====================================== #
        # import all upper air zarr files
        all_files = []
        for fn in self.filenames:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename=fn)
            xarray_dataset = drop_var_from_dataset(
                xarray_dataset, self.varname_upper_air
            )
            # collect yearly datasets within a list
            all_files.append(xarray_dataset)
        self.all_files = all_files
        # ====================================== #

        # -------------------------------------- #
        # other settings
        self.current_epoch = 0
        self.rollout_p = rollout_p

        self.lead_time_periods = conf["data"]["lead_time_periods"]
        self.skip_periods = conf["data"]["skip_periods"]

    def ds_read_and_subset(self, filename, time_start, time_end, varnames):
        """Read and subset specified dataset.

        Args:
            filename (str): path to specified dataset file.
            time_start (int): start time index.
            time_end (int): end time index.
            varnames (list): List of variables to be read.

        """
        sliced_x = get_forward_data(filename)
        sliced_x = sliced_x.isel(time=slice(time_start, time_end))
        sliced_x = drop_var_from_dataset(sliced_x, varnames)
        return sliced_x

    def load_zarr_as_input(self, i_file, i_init_start, i_init_end, mode="input"):
        """Load input data from zarr files.

        Args:
            i_file: index of the file
            i_init_start: start index of the data being loaded
            i_init_end: end index of the data being loaded.
            mode: "input" or "target"

        Returns:
            xr.Dataset containing all the variables.

        """
        # get the needed file from a list of zarr files
        # open the zarr file as xr.dataset and subset based on the needed time

        # sliced_x: the final output, starts with an upper air xr.dataset
        sliced_x = self.ds_read_and_subset(
            self.filenames[i_file], i_init_start, i_init_end + 1, self.varname_upper_air
        )
        # surface variables
        if self.filename_surface is not None:
            sliced_surface = self.ds_read_and_subset(
                self.filename_surface[i_file],
                i_init_start,
                i_init_end + 1,
                self.varname_surface,
            )
            # merge surface to sliced_x
            sliced_surface["time"] = sliced_x["time"]
            sliced_x = sliced_x.merge(sliced_surface)

        if mode == "input":
            # dynamic forcing variables
            if self.filename_dyn_forcing is not None:
                sliced_dyn_forcing = self.ds_read_and_subset(
                    self.filename_dyn_forcing[i_file],
                    i_init_start,
                    i_init_end + 1,
                    self.varname_dyn_forcing,
                )
                # merge surface to sliced_x
                sliced_dyn_forcing["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_dyn_forcing)

            # forcing / static
            if self.filename_forcing is not None:
                sliced_forcing = get_forward_data(self.filename_forcing)
                sliced_forcing = drop_var_from_dataset(
                    sliced_forcing, self.varname_forcing
                )

                # See also `ERA5_and_Forcing_Dataset`
                # =============================================================================== #
                # matching month, day, hour between forcing and upper air [time]
                # this approach handles leap year forcing file and non-leap-year upper air file
                month_day_forcing = extract_month_day_hour(
                    np.array(sliced_forcing["time"])
                )
                month_day_inputs = extract_month_day_hour(np.array(sliced_x["time"]))
                # indices to subset
                ind_forcing, _ = find_common_indices(
                    month_day_forcing, month_day_inputs
                )
                sliced_forcing = sliced_forcing.isel(time=ind_forcing)
                # forcing and upper air have different years but the same mon/day/hour
                # safely replace forcing time with upper air time
                sliced_forcing["time"] = sliced_x["time"]
                # =============================================================================== #

                # merge forcing to sliced_x
                sliced_x = sliced_x.merge(sliced_forcing)

            if self.filename_static is not None:
                sliced_static = get_forward_data(self.filename_static)
                sliced_static = drop_var_from_dataset(
                    sliced_static, self.varname_static
                )
                sliced_static = sliced_static.expand_dims(
                    dim={"time": len(sliced_x["time"])}
                )
                sliced_static["time"] = sliced_x["time"]
                # merge static to sliced_x
                sliced_x = sliced_x.merge(sliced_static)

        elif mode == "target":
            # diagnostic
            if self.filename_diagnostic is not None:
                sliced_diagnostic = self.ds_read_and_subset(
                    self.filename_diagnostic[i_file],
                    i_init_start,
                    i_init_end + 1,
                    self.varname_diagnostic,
                )
                # merge diagnostics to sliced_x
                sliced_diagnostic["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_diagnostic)

        return sliced_x

    def find_start_stop_indices(self, index):
        """Find start and stop indices for a given yearly data zarr file.

        Args:
            index: indices of zarr file.

        """
        # ============================================================================ #
        # shift hours for history_len > 1, becuase more than one init times are needed
        # <--- !! it MAY NOT work when self.skip_period != 1
        shifted_hours = (
            self.lead_time_periods * self.skip_periods * (self.history_len - 1)
        )
        # ============================================================================ #
        # subtrack shifted_hour form the 1st & last init times
        # convert to datetime object
        self.init_datetime[index][0] = datetime.datetime.strptime(
            self.init_datetime[index][0], "%Y-%m-%d %H:%M:%S"
        ) - datetime.timedelta(hours=shifted_hours)
        self.init_datetime[index][1] = datetime.datetime.strptime(
            self.init_datetime[index][1], "%Y-%m-%d %H:%M:%S"
        ) - datetime.timedelta(hours=shifted_hours)

        # convert the 1st & last init times to a list of init times
        self.init_datetime[index] = generate_datetime(
            self.init_datetime[index][0],
            self.init_datetime[index][1],
            self.lead_time_periods,
        )
        # convert datetime obj to nanosecondes
        init_time_list_dt = [
            np.datetime64(date.strftime("%Y-%m-%d %H:%M:%S"))
            for date in self.init_datetime[index]
        ]

        # init_time_list_np: a list of python datetime objects, each is a forecast step
        # init_time_list_np[0]: the first initialization time
        # init_time_list_np[t]: the forcasted time of the (t-1)th step; the initialization time of the t-th step
        self.init_time_list_np = [
            np.datetime64(str(dt_obj) + ".000000000").astype(datetime.datetime)
            for dt_obj in init_time_list_dt
        ]

        info = []
        for init_time in self.init_time_list_np:
            for i_file, ds in enumerate(self.all_files):
                # get the year of the current file
                #print('Check time values, data.py:' ,ds["time"][0].values)
                #print('Check time values, data.py:' ,ds["time"][0].values.dtype)

                time_value = ensure_numpy_datetime(ds["time"][0].values)
                
                ds_year = int(np.datetime_as_string(time_value, unit="Y"))

                # get the first and last years of init times
                init_year0 = nanoseconds_to_year(init_time)

                # found the right yearly file
                if init_year0 == ds_year:
                    N_times = len(ds["time"])

                    # convert ds['time'] to a list of nanoseconds
                    ds_time_list = [
                        np.datetime64(ensure_numpy_datetime(ds_time.values)).astype('datetime64[ns]').astype(int)
                        for ds_time in ds["time"]
                    ]

                    ds_start_time = ds_time_list[0]
                    ds_end_time = ds_time_list[-1]

                    init_time_start = init_time
                    # if initalization time is within this (yearly) xr.Dataset
                    #print('Check time values, data.py: start time',ds_start_time)
                    #print('Check time values, data.py: end time' ,ds_end_time)
                    #print('Check time values, data.py: init time' , init_time_start)
                    
                    if ds_start_time <= init_time_start <= ds_end_time:
                        # try getting the index of the first initalization time
                        i_init_start = ds_time_list.index(init_time_start)

                        # for multiple init time inputs (history_len > 1), init_end is different for init_start
                        init_time_end = init_time_start + hour_to_nanoseconds(
                            shifted_hours
                        )

                        # see if init_time_end is alos in this file
                        if ds_start_time <= init_time_end <= ds_end_time:
                            # try getting the index
                            i_init_end = ds_time_list.index(init_time_end)
                        else:
                            # this set of initalizations have crossed years
                            # get the last element of the current file
                            # we have anthoer section that checks additional input data
                            i_init_end = len(ds_time_list) - 1

                        info.append([i_file, i_init_start, i_init_end, N_times])
        return info

    def __len__(self):
        """Length of dataset."""
        return len(self.init_datetime)

    def __iter__(self):
        """Iterate through batch."""
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(
            self,
            num_replicas=num_workers * self.world_size,
            rank=self.rank * num_workers + worker_id,
            shuffle=False,
        )
        for index in sampler:
            # get the init time info for the current sample
            data_lookup = self.find_start_stop_indices(index)

            for k, _ in enumerate(self.init_time_list_np):
                # the first initialization time: get initalization from data
                i_file, i_init_start, i_init_end, N_times = data_lookup[k]

                # allocate output dict
                output_dict = {}

                # get all inputs in one xr.Dataset
                sliced_x = self.load_zarr_as_input(
                    i_file, i_init_start, i_init_end, mode="input"
                )

                # Check if additional data from the next file is needed
                if (len(sliced_x["time"]) < self.history_len) or (
                    i_init_end + 1 >= N_times
                ):
                    # Load excess data from the next file
                    next_file_idx = self.filenames.index(self.filenames[i_file]) + 1

                    if next_file_idx >= len(self.filenames):
                        # not enough input data to support this forecast
                        raise OSError(
                            "You have reached the end of the available data. Exiting."
                        )

                    else:
                        sliced_y = self.load_zarr_as_input(
                            i_file, i_init_end, i_init_end, mode="target"
                        )

                        # i_init_start = 0 because we need the beginning of the next file only
                        sliced_x_next = self.load_zarr_as_input(
                            next_file_idx, 0, self.history_len, mode="input"
                        )
                        sliced_y_next = self.load_zarr_as_input(
                            next_file_idx, 0, 1, mode="target"
                        )
                        # 1 becuase taregt is one step a time

                        # Concatenate excess data from the next file with the current data
                        sliced_x_combine = xr.concat(
                            [sliced_x, sliced_x_next], dim="time"
                        )
                        sliced_y_combine = xr.concat(
                            [sliced_y, sliced_y_next], dim="time"
                        )

                        sliced_x = sliced_x_combine.isel(
                            time=slice(0, self.history_len)
                        )
                        sliced_y = sliced_y_combine.isel(
                            time=slice(self.history_len, self.history_len + 1)
                        )
                else:
                    sliced_y = self.load_zarr_as_input(
                        i_file, i_init_end + 1, i_init_end + 1, mode="target"
                    )

                sample_x = {
                    "historical_ERA5_images": sliced_x,
                    "target_ERA5_images": sliced_y,
                }

                if self.transform:
                    sample_x = self.transform(sample_x)

                for key in sample_x.keys():
                    output_dict[key] = sample_x[key]

                # <--- !! 'forecast_hour' is actually "forecast_step" but named by assuming hourly
                output_dict["forecast_hour"] = k + 1
                # Adjust stopping condition
                output_dict["stop_forecast"] = k == (len(self.init_time_list_np) - 1)
                output_dict["datetime"] = sliced_x.time.values.astype(
                    "datetime64[s]"
                ).astype(int)[-1]

                # return output_dict
                yield output_dict

                if output_dict["stop_forecast"]:
                    break
