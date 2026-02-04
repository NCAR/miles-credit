import torch
import logging
import numpy as np
import xarray as xr
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from credit.data import (
    Sample_WRF,
    extract_month_day_hour,
    find_common_indices,
    get_forward_data,
    drop_var_from_dataset,
    find_key_for_number,
    next_n_hour,
    previous_hourly_steps,
    encode_datetime64,
    filter_ds,
)

logger = logging.getLogger(__name__)


def worker(
    tuple_index: Tuple[int, int],
    WRF_file_indices: Dict[str, List[int]],
    list_upper_ds: List[Any],
    list_surf_ds: Optional[List[Any]],
    list_dyn_forcing_ds: Optional[List[Any]],
    list_diag_ds: Optional[List[Any]],
    xarray_forcing: Optional[Any],
    xarray_static: Optional[Any],
    history_len: int,
    forecast_len: int,
    list_upper_ds_outside: Optional[List[Any]],
    list_surf_ds_outside: Optional[List[Any]],
    outside_file_year_range: Optional[List[Any]],
    outside_file_indices: Optional[List[Any]],
    history_len_outside: int,
    transform: Optional[Callable],
) -> Dict[str, Any]:
    index, ind_start_current_step = tuple_index

    try:
        # ========================================================================== #
        # cross-year indices --> the index of the year + indices within that year

        # select the ind_file based on the iter index
        ind_file = find_key_for_number(ind_start_current_step, WRF_file_indices)

        # get the ind within the current file
        ind_start = WRF_file_indices[ind_file][1]
        ind_start_in_file = ind_start_current_step - ind_start

        # handle out-of-bounds
        ind_largest = len(list_upper_ds[int(ind_file)]["time"]) - (history_len + forecast_len + 1)

        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # ========================================================================== #
        # subset xarray on time dimension

        ind_end_in_file = ind_start_in_file + history_len + forecast_len

        ## WRF_file_subset: a xarray dataset that contains training input and target (for the current batch)
        WRF_subset = list_upper_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

        # ========================================================================== #
        # merge surface into the dataset

        if list_surf_ds:
            ## subset surface variables
            surface_subset = list_surf_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

            ## merge upper-air and surface here:
            WRF_subset = WRF_subset.merge(surface_subset)  # <-- lazy merge, upper and surface both not loaded

        # ==================================================== #
        # split WRF_subset into training inputs and targets
        #   + merge with dynamic forcing, forcing, and static

        # the ind_end of the WRF_subset
        # ind_end_time = len(WRF_subset["time"])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = WRF_subset.time.values.astype("datetime64[s]").astype(int)

        # ==================================================== #
        # xarray dataset as input
        ## WRF_input: the final input

        WRF_input = WRF_subset.isel(time=slice(0, history_len, 1)).load()

        # ========================================================================== #
        # merge dynamic forcing inputs
        if list_dyn_forcing_ds:
            dyn_forcing_subset = list_dyn_forcing_ds[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(time=slice(0, history_len, 1)).load()

            WRF_input = WRF_input.merge(dyn_forcing_subset)

        # ========================================================================== #
        # merge forcing inputs
        if xarray_forcing:
            # ------------------------------------------------------------------------------- #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(np.array(xarray_forcing["time"]))
            month_day_inputs = extract_month_day_hour(np.array(WRF_input["time"]))
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = xarray_forcing.isel(time=ind_forcing)
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input["time"] = WRF_input["time"]
            # ------------------------------------------------------------------------------- #

            # merge
            WRF_input = WRF_input.merge(forcing_subset_input)

        # ========================================================================== #
        # merge static inputs
        if xarray_static:
            # expand static var on time dim
            N_time_dims = len(WRF_subset["time"])
            static_subset_input = xarray_static.expand_dims(dim={"time": N_time_dims})
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords({"time": WRF_subset["time"]})
            # slice, update time and merge
            static_subset_input = static_subset_input.isel(time=slice(0, history_len, 1))
            static_subset_input["time"] = WRF_input["time"]
            WRF_input = WRF_input.merge(static_subset_input)

        # ==================================================== #
        # xarray dataset as target
        ## WRF_target: the final target

        WRF_target = WRF_subset.isel(time=slice(history_len, history_len + 1, 1)).load()

        ## merge diagnoisc input here:
        if list_diag_ds:
            # subset diagnostic variables
            diagnostic_subset = list_diag_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

            diagnostic_subset = diagnostic_subset.isel(time=slice(history_len, history_len + 1, 1)).load()

            # merge into the target dataset
            WRF_target = WRF_target.merge(diagnostic_subset)

        # ==================================================== #
        # handle boundary files
        # ==================================================== #
        time_boundary = WRF_target["time"].values[0]  # <--- assuming single time value here
        time_round = next_n_hour(time_boundary, 3)

        if history_len_outside == 1:
            time_year = int(np.datetime_as_string(time_round, unit="Y"))
            ind_year = time_year - outside_file_year_range[0]
            ind_date = np.searchsorted(outside_file_indices[str(ind_year)], time_round)
            ds_upper_outside = list_upper_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
            ds_surf_outside = list_surf_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
            ds_outside = xr.merge([ds_upper_outside, ds_surf_outside])

        else:
            list_ds_upper_outside_slice = []
            list_ds_surf_outside_slice = []

            for i_time_backward in range(history_len_outside):
                time_round_loop = previous_hourly_steps(time_round, 3, i_time_backward)
                time_year = int(np.datetime_as_string(time_round_loop, unit="Y"))
                ind_year = time_year - outside_file_year_range[0]
                ind_date = np.searchsorted(outside_file_indices[str(ind_year)], time_round_loop)
                list_ds_upper_outside_slice.append(
                    list_upper_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
                )
                list_ds_surf_outside_slice.append(
                    list_surf_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
                )

            ds_upper_outside = xr.concat(
                list_ds_upper_outside_slice[::-1], dim="time"
            )  # ::-1 so the latest time is the last
            ds_surf_outside = xr.concat(list_ds_surf_outside_slice[::-1], dim="time")
            ds_outside = xr.merge([ds_upper_outside, ds_surf_outside])

        # ==================================================== #
        # encode datetime input
        # ==================================================== #
        t0 = WRF_input["time"].values
        t1 = WRF_target["time"].values
        t2 = ds_outside["time"].values
        time_encode = encode_datetime64(np.concatenate([t0, t1, t2]))

        sample = Sample_WRF(
            WRF_input=WRF_input,
            WRF_target=WRF_target,
            boundary_input=ds_outside,
            time_encode=time_encode,
            datetime_index=datetime_as_number,
        )

        # ==================================== #
        # data normalization
        if transform:
            sample = transform(sample)

        sample["index"] = index
        sample["datetime"] = [
            int(WRF_input.time.values[0].astype("datetime64[s]").astype(int)),
            int(WRF_target.time.values[0].astype("datetime64[s]").astype(int)),
        ]

    except Exception as e:
        logger.error(f"Error processing index {tuple_index}: {e}")
        raise

    return sample


class RepeatingIndexSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        forecast_len,
        shuffle=True,
        seed=42,
        rank=0,
        num_replicas=1,
    ):
        self.dataset = dataset
        self.forecast_len = forecast_len + 1  # Total steps in the forecast sequence
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.num_replicas = num_replicas

        # Compute valid starting indices ensuring full sequences fit
        all_start_indices = list(range(0, len(self.dataset), 1))

        num_indices = len(all_start_indices)  # Trim the number of indices to ensure it's divisible by world_size
        num_indices_per_rank = num_indices // self.num_replicas
        all_start_indices = all_start_indices[: num_indices_per_rank * self.num_replicas]
        self.all_start_indices = all_start_indices
        self.num_indices_per_rank = num_indices_per_rank

        if self.shuffle:
            self.rng = np.random.default_rng(seed)
            # rng.shuffle(self.start_indices)

    def __len__(self):
        """Returns the total number of indices for this rank."""
        # return len(self.start_indices) * self.forecast_len
        return self.num_indices_per_rank * self.forecast_len

    def __iter__(self):
        """
        Yields each start index repeated (forecast_len + 1) times.
        """
        all_indices = self.all_start_indices
        if self.shuffle:
            all_indices = self.rng.permutation(all_indices)

        self.start_indices = all_indices[self.rank :: self.num_replicas]
        assert len(self.start_indices) == self.num_indices_per_rank
        for idx in self.start_indices:
            for _ in range(self.forecast_len):
                yield idx

    def batches_per_epoch(self):
        """
        Computes the number of batches per epoch for a given batch size.

        Returns:
        - int: Number of batches per epoch.
        """
        return self.num_indices_per_rank


class WRFMultiStep(torch.utils.data.Dataset):
    def __init__(
        self,
        param_interior,
        param_outside,
        transform=None,
        seed=42,
        rank=0,
        world_size=1,
        max_forecast_len=None,
    ):
        # ========================================================== #
        # WRF domain variable and filename info
        varname_upper_air = param_interior["varname_upper_air"]
        varname_surface = param_interior["varname_surface"]
        varname_dyn_forcing = param_interior["varname_dyn_forcing"]
        varname_forcing = param_interior["varname_forcing"]
        varname_static = param_interior["varname_static"]
        varname_diagnostic = param_interior["varname_diagnostic"]
        filenames = param_interior["filenames"]
        filename_surface = param_interior["filename_surface"]
        filename_dyn_forcing = param_interior["filename_dyn_forcing"]
        filename_forcing = param_interior["filename_forcing"]
        filename_static = param_interior["filename_static"]
        filename_diagnostic = param_interior["filename_diagnostic"]
        # ----------------------------------------------------------- #

        list_upper_ds = []
        list_surf_ds = []
        list_dyn_forcing_ds = []
        list_diag_ds = []
        filenames = sorted(filenames)

        all_ds = [get_forward_data(fn) for fn in filenames]

        # 1. Upper‐air
        list_upper_ds = [filter_ds(ds, varname_upper_air) for ds in all_ds]

        # 2. Surface
        if filename_surface:
            list_surf_ds = [filter_ds(ds, varname_surface) for ds in all_ds]
        else:
            list_surf_ds = False

        # 3. Dynamic forcing
        if filename_dyn_forcing:
            list_dyn_forcing_ds = [filter_ds(ds, varname_dyn_forcing) for ds in all_ds]
        else:
            list_dyn_forcing_ds = False

        # 4. Diagnostics
        if filename_diagnostic:
            list_diag_ds = [filter_ds(ds, varname_diagnostic) for ds in all_ds]
        else:
            list_diag_ds = False

        self.list_upper_ds = list_upper_ds
        self.list_surf_ds = list_surf_ds
        self.list_dyn_forcing_ds = list_dyn_forcing_ds
        self.list_diag_ds = list_diag_ds

        self.history_len = param_interior["history_len"]
        self.forecast_len = param_interior["forecast_len"]
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.total_seq_len = self.history_len + self.forecast_len

        # -------------------------------------------------------------------------- #
        # get sample indices from WRF upper-air files:
        ind_start = 0
        self.WRF_file_indices = {}  # <------ change
        for ind_file, WRF_file_xarray in enumerate(self.list_upper_ds):
            # [number of samples, ind_start, ind_end]
            self.WRF_file_indices[str(ind_file)] = [
                len(WRF_file_xarray["time"]),
                ind_start,
                ind_start + len(WRF_file_xarray["time"]),
            ]
            ind_start += len(WRF_file_xarray["time"]) + 1

        # -------------------------------------------------------------------------- #
        # forcing file
        self.filename_forcing = filename_forcing
        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            ds = get_forward_data(filename_forcing)
            ds_forcing = drop_var_from_dataset(ds, varname_forcing).load()
            self.xarray_forcing = ds_forcing
        else:
            self.xarray_forcing = False

        # -------------------------------------------------------------------------- #
        # static file
        self.filename_static = filename_static
        if self.filename_static is not None:
            # drop variables if they are not in the config
            ds = get_forward_data(filename_static)
            ds_static = drop_var_from_dataset(ds, varname_static).load()
            self.xarray_static = ds_static
        else:
            self.xarray_static = False

        # ========================================================== #
        # boundary variable and filename info
        varname_upper_air_outside = param_outside["varname_upper_air"]
        varname_surface_outside = param_outside["varname_surface"]
        filenames_outside = param_outside["filenames"]
        filename_surface_outside = param_outside["filename_surface"]
        # ----------------------------------------------------------- #
        # collecting xr.datasets
        list_upper_ds_outside = []
        list_surf_ds_outside = []
        filenames_outside = sorted(filenames_outside)

        for fn_outside in filenames_outside:
            # drop variables if they are not in the config
            ds_outside = get_forward_data(filename=fn_outside)
            ds_upper_outside = drop_var_from_dataset(ds_outside, varname_upper_air_outside)

            if filename_surface_outside is not None:
                ds_surf_outside = drop_var_from_dataset(ds_outside, varname_surface_outside)
                list_surf_ds_outside.append(ds_surf_outside)
            else:
                self.list_surf_ds_outside = False

            list_upper_ds_outside.append(ds_upper_outside)

        self.list_upper_ds_outside = list_upper_ds_outside
        self.list_surf_ds_outside = list_surf_ds_outside
        self.history_len_outside = param_outside["history_len"]
        self.forecast_len_outside = param_outside["forecast_len"]
        self.total_seq_len = self.history_len_outside + self.forecast_len_outside
        # -------------------------------------------------------------------------- #
        # get sample indices from boundary upper-air files:
        self.outside_file_year_range = [
            int(np.datetime_as_string(self.list_upper_ds_outside[0]["time"][0].values, unit="Y")),
            int(np.datetime_as_string(self.list_upper_ds_outside[-1]["time"][0].values, unit="Y")),
        ]

        self.outside_file_indices = {}  # <------ change
        for ind_file, outside_file_xarray in enumerate(self.list_upper_ds_outside):
            self.outside_file_indices[str(ind_file)] = outside_file_xarray["time"].values

        # ========================================================== #

        self.transform = transform
        self.rng = np.random.default_rng(seed=seed)
        self.max_forecast_len = max_forecast_len

        self.worker = partial(
            worker,
            WRF_file_indices=self.WRF_file_indices,
            list_upper_ds=self.list_upper_ds,
            list_surf_ds=self.list_surf_ds,
            list_dyn_forcing_ds=self.list_dyn_forcing_ds,
            list_diag_ds=self.list_diag_ds,
            xarray_forcing=self.xarray_forcing,
            xarray_static=self.xarray_static,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
            list_upper_ds_outside=self.list_upper_ds_outside,
            list_surf_ds_outside=self.list_surf_ds_outside,
            outside_file_year_range=self.outside_file_year_range,
            outside_file_indices=self.outside_file_indices,
            history_len_outside=self.history_len_outside,
            transform=self.transform,
        )

        self.total_length = len(self.WRF_file_indices)
        self.current_epoch = None
        self.forecast_step_count = 0
        self.current_index = None
        self.initial_index = None

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for WRF_xarray in self.list_upper_ds:
            total_len += len(WRF_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.forecast_step_count = 0
        self.current_index = None
        self.initial_index = None

    def __getitem__(self, index):
        if (self.forecast_step_count == self.forecast_len + 1) or (self.current_index is None):
            # We've completed the last forecast or we're starting for the first time
            # Start a new forecast using the sampler index
            self.current_index = index  # self._get_random_start_index()
            self.forecast_step_count = 0
            index = self.current_index
            self.initial_index = self.current_index
        else:
            # Ignore the sampler index and continue the forecast
            self.current_index += 1
            index = self.current_index

        # Worker process
        sample = self.worker((self.initial_index, index))

        # assign sample index
        sample["forecast_step"] = self.forecast_step_count + 1
        sample["index"] = index
        sample["stop_forecast"] = self.forecast_step_count == self.forecast_len

        # update the step count
        self.forecast_step_count += 1

        return sample
