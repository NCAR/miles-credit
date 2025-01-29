import os
import logging
import warnings

# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import math

# ---------- #
import torch
from torch.utils.data.distributed import DistributedSampler

from credit.data import (
    drop_var_from_dataset,
    generate_datetime,
    nanoseconds_to_year,
    hour_to_nanoseconds,
    get_forward_data,
    extract_month_day_hour,
    find_common_indices,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class BatchForecastLenDataLoader:
    """
    A custom DataLoader that supports datasets with a non-trivial forecast length.

    This DataLoader is designed to iterate over datasets that provide a
     `forecast_len` attribute, optionally incorporating batch-specific
     properties like `batches_per_epoch()` if available.

    Attributes:
        dataset: The dataset object, which must have a `forecast_len` attribute
                  and may optionally have a `batches_per_epoch()` method.
        forecast_len: The forecast length incremented by 1.
    """

    def __init__(self, dataset):
        """
        Initializes the BatchForecastLenDataLoader.

        Args:
            dataset: A dataset object with a `forecast_len` attribute and
                      optionally a `batches_per_epoch()` method.
        """
        self.dataset = dataset
        self.forecast_len = dataset.forecast_period

    def __iter__(self):
        """
        Iterates over the dataset.

        This method directly yields samples from the dataset. The forecast
         length is not explicitly handled here; it is assumed to be accounted
         for in the dataset's structure or sampling.

        Yields:
            sample: A single sample from the dataset.
        """
        dataset_iter = iter(self.dataset)
        for _ in range(len(self)):
            yield next(dataset_iter)

    def __len__(self):
        """
        Returns the length of the DataLoader.

        The length is determined by the forecast length and either the
         dataset's `batches_per_epoch()` method (if available) or the dataset's
         overall length.

        Returns:
            int: The total number of samples or iterations.
        """
        return self.forecast_len * math.ceil(len(self.dataset) / self.dataset.batch_size)


class Predict_Dataset_Batcher(torch.utils.data.Dataset):
    """
    A Pytorch Dataset class that works on:
        - upper-air variables (time, level, lat, lon)
        - surface variables (time, lat, lon)
        - dynamic forcing variables (time, lat, lon)
        - foring variables (time, lat, lon)
        - diagnostic variables (time, lat, lon)
        - static variables (lat, lon)
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
        sst_forcing=None,
        fcst_datetime=None,
        lead_time_periods=6,
        history_len=1,
        transform=None,
        seed=42,
        rank=0,
        world_size=1,
        skip_periods=None,
        batch_size=1,
        skip_target=False
    ):
        """
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
        - transform (callable, optional): Transformation function to apply to the data.
        - seed (int, optional): Random seed for reproducibility. Default is 42.
        - skip_periods (int, optional): Number of periods to skip between samples.
        - max_forecast_len (int, optional): Maximum length of the forecast sequence.
        - sst_forcing (optional):
        - skip_target: Do not return y truth data
        Returns:
        - sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.
        """

        self.history_len = history_len
        self.transform = transform
        self.init_datetime = fcst_datetime
        self.lead_time_periods = lead_time_periods
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.skip_target = skip_target

        # skip periods
        self.skip_periods = skip_periods
        if self.skip_periods is None:
            self.skip_periods = 1

        # set random seed
        self.rng = np.random.default_rng(seed=seed)

        # sst forcing
        self.sst_forcing = sst_forcing

        # flags to determine if any of the [surface, dyn_forcing, diagnostics]
        # variable groups share the same file as upper air variables
        flag_share_surf = False
        flag_share_dyn = False
        flag_share_diag = False

        all_files = []
        filenames = sorted(filenames)

        # blocks that can handle no-sharing (each group has it own file)
        # surface
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

        # dynamic forcing
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

        # diagnostics
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

        # file names
        self.all_files = all_files
        self.filenames = filenames
        self.filename_surface = filename_surface
        self.filename_dyn_forcing = filename_dyn_forcing
        self.filename_forcing = filename_forcing
        self.filename_static = filename_static
        self.filename_diagnostic = filename_diagnostic

        # var names
        self.varname_upper_air = varname_upper_air
        self.varname_surface = varname_surface
        self.varname_dyn_forcing = varname_dyn_forcing
        self.varname_forcing = varname_forcing
        self.varname_static = varname_static
        self.varname_diagnostic = varname_diagnostic

        if flag_share_surf:
            self.surface_files = surface_files
        if flag_share_dyn:
            self.dyn_forcing_files = dyn_forcing_files
        if flag_share_diag:
            self.diagnostic_files = diagnostic_files

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

        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_forcing)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_forcing)

            self.xarray_forcing = xarray_dataset
        else:
            self.xarray_forcing = False

        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_static)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_static)

            self.xarray_static = xarray_dataset
        else:
            self.xarray_static = False

        # Initialize the first forecast so we can get the forecast_len
        # which up to here is not defined. Needed in __len__ so DataLoader knows when to stop
        shifted_hours = (
            self.lead_time_periods * self.skip_periods * (self.history_len - 1)
        )
        # subtrack shifted_hour form the 1st & last init times
        # convert to datetime object
        fcst_datetime_0 = datetime.strptime(
            self.init_datetime[0][0], "%Y-%m-%d %H:%M:%S"
        ) - timedelta(hours=shifted_hours)
        fcst_datetime_1 = datetime.strptime(
            self.init_datetime[0][1], "%Y-%m-%d %H:%M:%S"
        ) - timedelta(hours=shifted_hours)
        # convert the 1st & last init times to a list of init times
        self.forecast_period = len(generate_datetime(
            fcst_datetime_0,
            fcst_datetime_1,
            self.lead_time_periods,
        ))
        if self.forecast_period < self.batch_size:
            self.batch_size = self.forecast_period

        # Use DistributedSampler for index management
        self.sampler = DistributedSampler(
            self,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False
        )

        # Initialze the batch indices by faking the epoch number here and resetting to None
        # this is mainly a feature for working with smaller datasets / testing purposes
        self.set_epoch(0)

    def __len__(self):
        total_forecasts = len(self.init_datetime)
        return total_forecasts

    def ds_read_and_subset(self, filename, time_start, time_end, varnames):
        sliced_x = get_forward_data(filename)
        sliced_x = sliced_x.isel(time=slice(time_start, time_end))
        sliced_x = drop_var_from_dataset(sliced_x, varnames)
        return sliced_x

    def load_zarr_as_input(self, i_file, i_init_start, i_init_end, mode="input"):
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
        # shift hours for history_len > 1, becuase more than one init times are needed
        # <--- !! it MAY NOT work when self.skip_period != 1
        shifted_hours = (
            self.lead_time_periods * self.skip_periods * (self.history_len - 1)
        )

        # subtrack shifted_hour form the 1st & last init times
        # convert to datetime object
        self.init_datetime[index][0] = datetime.strptime(
            self.init_datetime[index][0], "%Y-%m-%d %H:%M:%S"
        ) - timedelta(hours=shifted_hours)
        self.init_datetime[index][1] = datetime.strptime(
            self.init_datetime[index][1], "%Y-%m-%d %H:%M:%S"
        ) - timedelta(hours=shifted_hours)

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
            np.datetime64(str(dt_obj) + ".000000000").astype(datetime)
            for dt_obj in init_time_list_dt
        ]

        info = []
        for init_time in self.init_time_list_np:
            for i_file, ds in enumerate(self.all_files):
                # get the year of the current file
                ds_year = int(np.datetime_as_string(ds["time"][0].values, unit="Y"))

                # get the first and last years of init times
                init_year0 = nanoseconds_to_year(init_time)

                # found the right yearly file
                if init_year0 == ds_year:
                    N_times = len(ds["time"])
                    # convert ds['time'] to a list of nanosecondes
                    ds_time_list = [
                        np.datetime64(ds_time.values).astype(datetime)
                        for ds_time in ds["time"]
                    ]
                    ds_start_time = ds_time_list[0]
                    ds_end_time = ds_time_list[-1]

                    init_time_start = init_time
                    # if initalization time is within this (yearly) xr.Dataset
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

    def initialize_batch(self):
        """
        Initializes batch indices using DistributedSampler's indices.
        Ensures proper cycling when shuffle=False.
        """
        # Initialize the call count if not already present
        if not hasattr(self, "batch_call_count"):
            self.batch_call_count = 0

        # Set epoch for DistributedSampler to ensure consistent shuffling across devices
        if self.current_epoch is None:
            logging.warning("You must first set the epoch number using set_epoch method.")

        # Retrieve indices for this GPU
        total_indices = len(self.batch_indices)

        # Select batch indices based on call count (deterministic cycling)
        start = self.batch_call_count * self.batch_size
        end = start + self.batch_size

        if end > total_indices:
            # Simple wraparound by incrementing start index
            start = start % total_indices
            end = min(start + self.batch_size, total_indices)
        indices = self.batch_indices[start:end]

        # Increment batch_call_count, reset when all indices are cycled
        self.batch_call_count += 1
        if start + self.batch_size >= total_indices:
            self.batch_call_count = 0  # Reset for next cycle

        # Assign batch indices
        self.current_batch_indices = list(indices)  # this will be the local indices used in getitem
        self.time_steps = [0] * len(self.current_batch_indices)
        self.forecast_step_counts = [0] * len(self.current_batch_indices)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.sampler.set_epoch(epoch)
        self.batch_indices = list(self.sampler)
        self.batch_call_count = 0
        self.data_lookup = None
        self.initialize_batch()

    def __getitem__(self, _):
        batch = {}

        if self.forecast_step_counts[0] == self.forecast_period:
            self.initialize_batch()

        if self.forecast_step_counts[0] == 0:
            self.data_lookup = [self.find_start_stop_indices(idx) for idx in self.current_batch_indices]

        for k, idx in enumerate(self.current_batch_indices):
            # Get data for current timestep
            current_t = self.time_steps[k]
            i_file, i_init_start, i_init_end, N_times = self.data_lookup[k][current_t]

            # Load input data
            sliced_x = self.load_zarr_as_input(i_file, i_init_start, i_init_end, mode="input")

            # Handle cross-file data if needed
            if (len(sliced_x["time"]) < self.history_len) or (i_init_end + 1 >= N_times):
                next_file_idx = self.filenames.index(self.filenames[i_file]) + 1
                if next_file_idx >= len(self.filenames):
                    raise OSError("End of available data reached.")
                # Input data
                sliced_x_next = self.load_zarr_as_input(next_file_idx, 0, self.history_len, mode="input")
                sliced_x = xr.concat([sliced_x, sliced_x_next], dim="time").isel(time=slice(0, self.history_len))
                # Truth data
                if not self.skip_target:
                    sliced_y = self.load_zarr_as_input(i_file, i_init_end, i_init_end, mode="target")
                    sliced_y_next = self.load_zarr_as_input(next_file_idx, 0, 1, mode="target")
                    sliced_y = xr.concat([sliced_y, sliced_y_next], dim="time").isel(
                        time=slice(self.history_len, self.history_len + 1))
            elif not self.skip_target:
                sliced_y = self.load_zarr_as_input(i_file, i_init_end + 1, i_init_end + 1, mode="target")

            # Transform data
            sample = {"historical_ERA5_images": sliced_x}
            if not self.skip_target:
                sample["target_ERA5_images"] = sliced_y
            if self.transform:
                sample = self.transform(sample)

            # Add metadata
            sample["index"] = idx + current_t
            sample["datetime"] = sliced_x.time.values.astype("datetime64[s]").astype(int)[-1]

            # Convert and add to batch
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    value = torch.tensor(value)
                elif isinstance(value, np.int64):
                    value = torch.tensor(value, dtype=torch.int64)
                elif isinstance(value, (int, float)):
                    value = torch.tensor(value, dtype=torch.float32)
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)

                if value.ndimension() == 0:
                    value = value.unsqueeze(0)

                if value.ndim in (4, 5):
                    value = value.unsqueeze(0)

                if key not in batch:
                    batch[key] = value
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)

            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = torch.tensor([self.forecast_step_counts[0]])
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_period

        return batch
