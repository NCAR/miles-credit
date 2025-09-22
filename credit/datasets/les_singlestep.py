"""
les_dataset.py
-------------------------------------------------------
Content:
    - LES_Dataset
    - LES_Predict

"""

import datetime
import numpy as np
import xarray as xr
from typing import TypedDict, Union, Sequence

import torch
import random
import torch.utils.data
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

from credit.data import Sample_LES

from credit.data import (
    drop_var_from_dataset,
    extract_month_day_hour,
    find_common_indices,
    get_forward_data,
    keep_dataset_vars,
    generate_datetime,
    hour_to_nanoseconds,
    nanoseconds_to_year,
    find_key_for_number,
    subset_patch,
    filter_ds,
)


class LES_Dataset(torch.utils.data.Dataset):
    """
    LES model Pytorch Dataset class
    """

    def __init__(
        self,
        param_interior,
        transform=None,
        seed=42,
    ):
        # ========================================================== #
        # LES domain variable and filename info
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
        # collecting xr.datasets
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
        self.total_seq_len = self.history_len + self.forecast_len
        # -------------------------------------------------------------------------- #
        # get sample indices from LES upper-air files:
        ind_start = 0
        self.LES_file_indices = {}  # <------ change
        for ind_file, LES_file_xarray in enumerate(self.list_upper_ds):
            # [number of samples, ind_start, ind_end]
            self.LES_file_indices[str(ind_file)] = [
                len(LES_file_xarray["time"]),
                ind_start,
                ind_start + len(LES_file_xarray["time"]),
            ]
            ind_start += len(LES_file_xarray["time"]) + 1

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

        self.transform = transform
        self.size_list = param_interior["size_list"]
        self.size_full = param_interior["size_full"]
        self.rng = np.random.default_rng(seed=seed)

        self.total_len = 0
        for LES_file_xarray in self.list_upper_ds:
            self.total_len += len(LES_file_xarray["time"]) - self.total_seq_len + 1

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for LES_file_xarray in self.list_upper_ds:
            total_len += len(LES_file_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def __getitem__(self, index):
        # ========================================================================== #
        # cross-year indices --> the index of the year + indices within that year
        if index > self.total_len:
            n_fold = index // self.total_len
            index = index - n_fold * self.total_len

        # select the ind_file based on the iter index
        ind_file = find_key_for_number(index, self.LES_file_indices)

        # get the ind within the current file
        ind_start = self.LES_file_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        # handle out-of-bounds
        ind_largest = len(self.list_upper_ds[int(ind_file)]["time"]) - (self.history_len + self.forecast_len + 1)

        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # ========================================================================== #
        # subset xarray on time dimension
        ind_end_in_file = ind_start_in_file + self.history_len + self.forecast_len

        ## LES_file_subset: a xarray dataset that contains training input and target (for the current batch)
        LES_subset = self.list_upper_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

        # ========================================================================== #
        # merge surface into the dataset

        if self.list_surf_ds:
            ## subset surface variables
            surface_subset = self.list_surf_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

            ## merge upper-air and surface here:
            LES_subset = LES_subset.merge(surface_subset)

        # ==================================================== #
        # split LES_subset into training inputs and targets
        #   + merge with dynamic forcing, forcing, and static

        # the ind_end of the LES_subset
        ind_end_time = len(LES_subset["time"])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = LES_subset.time.values.astype("datetime64[s]").astype(int)

        # ==================================================== #
        # xarray dataset as input
        ## LES_input: the final input

        LES_input = LES_subset.isel(time=slice(0, self.history_len, 1)).load()

        # ========================================================================== #
        # merge dynamic forcing inputs
        if self.list_dyn_forcing_ds:
            dyn_forcing_subset = self.list_dyn_forcing_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

            dyn_forcing_subset = dyn_forcing_subset.isel(time=slice(0, self.history_len, 1)).load()

            LES_input = LES_input.merge(dyn_forcing_subset)

        # ========================================================================== #
        # merge forcing inputs
        if self.xarray_forcing:
            # ------------------------------------------------------------------------------- #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(np.array(self.xarray_forcing["time"]))
            month_day_inputs = extract_month_day_hour(np.array(LES_input["time"]))
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = self.xarray_forcing.isel(time=ind_forcing)
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input["time"] = LES_input["time"]
            # ------------------------------------------------------------------------------- #

            # merge
            LES_input = LES_input.merge(forcing_subset_input)

        # ========================================================================== #
        # merge static inputs
        if self.xarray_static:
            # expand static var on time dim
            N_time_dims = len(LES_subset["time"])
            static_subset_input = self.xarray_static.expand_dims(dim={"time": N_time_dims})
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords({"time": LES_subset["time"]})
            # slice, update time and merge
            static_subset_input = static_subset_input.isel(time=slice(0, self.history_len, 1))
            static_subset_input["time"] = LES_input["time"]
            LES_input = LES_input.merge(static_subset_input)

        # ==================================================== #
        # xarray dataset as target
        ## LES_target: the final target

        LES_target = LES_subset.isel(time=slice(self.history_len, ind_end_time, 1)).load()

        ## merge diagnoisc input here:
        if self.list_diag_ds:
            # subset diagnostic variables
            diagnostic_subset = self.list_diag_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

            diagnostic_subset = diagnostic_subset.isel(time=slice(self.history_len, ind_end_time, 1)).load()

            # merge into the target dataset
            LES_target = LES_target.merge(diagnostic_subset)

        # ============================================== #
        # random subsetting
        subset_size = random.choice(self.size_list)
        dy, dx = subset_size

        # Compute the max allowable start indices
        max_iy_start = self.size_full[0] - dy
        max_ix_start = self.size_full[1] - dx

        # Sample random starting indices
        iy_start = random.randint(0, max_iy_start)
        ix_start = random.randint(0, max_ix_start)

        # Compute end indices
        iy_end = iy_start + dy
        ix_end = ix_start + dx

        LES_input = LES_input.isel(yIndex=slice(iy_start, iy_end), xIndex=slice(ix_start, ix_end))

        LES_target = LES_target.isel(yIndex=slice(iy_start, iy_end), xIndex=slice(ix_start, ix_end))

        # pipe xarray datasets to the sampler
        sample = Sample_LES(
            LES_input=LES_input,
            LES_target=LES_target,
            datetime_index=datetime_as_number,
        )

        # ==================================== #
        # data normalization
        if self.transform:
            sample = self.transform(sample)

        # assign sample index
        sample["index"] = index

        return sample


class LES_Predict(torch.utils.data.IterableDataset):
    def __init__(
        self,
        param_interior,
        data_lookup,
        rank,
        world_size,
        transform=None,
    ):
        ## no diagnostics because they are output only
        # varname_diagnostic = None
        # ========================================================== #
        # LES domain variable and filename info
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
        # collecting xr.datasets
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

        # -------------------------------------- #
        # file names
        self.filenames = filenames  # <------------------------ a list of files
        self.filename_surface = filename_surface  # <---------- a list of files
        self.filename_dyn_forcing = filename_dyn_forcing  # <-- a list of files
        self.filename_forcing = filename_forcing  # <-- single file
        self.filename_static = filename_static  # <---- single file
        self.filename_diagnostic = filename_diagnostic  # <---- single file

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

        # -------------------------------------- #
        # other settings
        self.rank = rank
        self.world_size = world_size
        self.transform = transform
        self.history_len = param_interior["history_len"]
        self.data_lookup = data_lookup
        # self.lead_time_periods = lead_time_periods

    def load_zarr_as_input(self, i_file, i_init_start, i_init_end, mode="input"):
        # sliced_x: the final output, starts with an upper air xr.dataset
        sliced_x = self.list_upper_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))

        # surface variables
        if self.filename_surface is not None:
            sliced_surface = self.list_surf_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))
            # sliced_surface["time"] = sliced_x["time"]
            sliced_x = sliced_x.merge(sliced_surface)

        if mode == "input":
            # dynamic forcing variables
            if self.filename_dyn_forcing is not None:
                sliced_dyn_forcing = self.list_dyn_forcing_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))
                # sliced_dyn_forcing["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_dyn_forcing)

            if self.filename_forcing is not None:
                sliced_forcing = self.xarray_forcing.copy()  # <-- shallow copy
                month_day_forcing = extract_month_day_hour(np.array(sliced_forcing["time"]))
                month_day_inputs = extract_month_day_hour(np.array(sliced_x["time"]))
                # indices to subset
                ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
                sliced_forcing = sliced_forcing.isel(time=ind_forcing)
                sliced_forcing["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_forcing)

            if self.filename_static is not None:
                sliced_static = self.xarray_static.copy()  # <-- shallow copy
                sliced_static = sliced_static.expand_dims(dim={"time": len(sliced_x["time"])})
                sliced_static["time"] = sliced_x["time"]
                # merge static to sliced_x
                sliced_x = sliced_x.merge(sliced_static)

        elif mode == "target":
            # diagnostic
            if self.filename_diagnostic is not None:
                sliced_diagnostic = self.list_diag_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))
                # sliced_diagnostic["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_diagnostic)

        return sliced_x

    def __len__(self):
        return len(self.data_lookup)

    def __iter__(self):
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
            data_lookup = self.data_lookup

            i_file, i_init_start, i_init_end, N_times = data_lookup[index]

            while i_init_end <= N_times:
                # for k, _ in enumerate(data_lookup):
                # the first initialization time: get initalization from data
                # i_file, i_init_start, i_init_end, N_times = data_lookup

                # allocate output dict
                output_dict = {}

                # get all inputs in one xr.Dataset
                sliced_x = self.load_zarr_as_input(i_file, i_init_start, i_init_end, mode="input")
                sliced_y = self.load_zarr_as_input(i_file, i_init_end + 1, i_init_end + 1, mode="target")

                sliced_x = subset_patch(sliced_x, input_size=(256, 256), start=None)
                sliced_y = subset_patch(sliced_y, input_size=(256, 256), start=None)

                # print((i_init_start, i_init_end))

                i_init_start += 1
                i_init_end += 1

                sample_x = {
                    "LES_input": sliced_x,
                    "LES_target": sliced_y,
                }

                if self.transform:
                    sample_x = self.transform(sample_x)

                for key in sample_x.keys():
                    output_dict[key] = sample_x[key]

                # <--- !! 'forecast_hour' is actually "forecast_step" but named by assuming hourly
                output_dict["forecast_hour"] = i_init_end

                # Adjust stopping condition
                output_dict["stop_forecast"] = i_init_end == N_times
                output_dict["datetime"] = sliced_x.time.values.astype("datetime64[s]").astype(int)[-1]

                # return output_dict
                yield output_dict

                if output_dict["stop_forecast"]:
                    break
