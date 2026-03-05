import datetime
import numpy as np
import xarray as xr

import torch
import torch.utils.data
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

from credit.data import Sample_WRF

from credit.data import (
    ensure_numpy_datetime,
    drop_var_from_dataset,
    extract_month_day_hour,
    find_common_indices,
    get_forward_data,
    generate_datetime,
    hour_to_nanoseconds,
    nanoseconds_to_year,
    find_key_for_number,
    next_n_hour,
    previous_hourly_steps,
    encode_datetime64,
    filter_ds,
)


class WRFDataset(torch.utils.data.Dataset):
    """
    WRF/regional model Pytorch Dataset class
    """

    def __init__(
        self,
        param_interior,
        param_outside,
        transform=None,
        seed=42,
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
        # shared by the two domains
        self.transform = transform
        self.rng = np.random.default_rng(seed=seed)

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for WRF_file_xarray in self.list_upper_ds:
            total_len += len(WRF_file_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def __getitem__(self, index):
        # ========================================================================== #
        # cross-year indices --> the index of the year + indices within that year

        # select the ind_file based on the iter index
        ind_file = find_key_for_number(index, self.WRF_file_indices)

        # get the ind within the current file
        ind_start = self.WRF_file_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        # handle out-of-bounds
        ind_largest = len(self.list_upper_ds[int(ind_file)]["time"]) - (self.history_len + self.forecast_len + 1)
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # ========================================================================== #
        # subset xarray on time dimension

        ind_end_in_file = ind_start_in_file + self.history_len + self.forecast_len

        ## WRF_file_subset: a xarray dataset that contains training input and target (for the current batch)
        WRF_subset = self.list_upper_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

        # ========================================================================== #
        # merge surface into the dataset

        if self.list_surf_ds:
            ## subset surface variables
            surface_subset = self.list_surf_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

            ## merge upper-air and surface here:
            WRF_subset = WRF_subset.merge(surface_subset)  # <-- lazy merge, upper and surface both not loaded

        # ==================================================== #
        # split WRF_subset into training inputs and targets
        #   + merge with dynamic forcing, forcing, and static

        # the ind_end of the WRF_subset
        ind_end_time = len(WRF_subset["time"])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = WRF_subset.time.values.astype("datetime64[s]").astype(int)

        # ==================================================== #
        # xarray dataset as input
        ## WRF_input: the final input

        WRF_input = WRF_subset.isel(time=slice(0, self.history_len, 1)).load()

        # ========================================================================== #
        # merge dynamic forcing inputs
        if self.list_dyn_forcing_ds:
            dyn_forcing_subset = self.list_dyn_forcing_ds[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(time=slice(0, self.history_len, 1)).load()

            WRF_input = WRF_input.merge(dyn_forcing_subset)

        # ========================================================================== #
        # merge forcing inputs
        if self.xarray_forcing:
            # ------------------------------------------------------------------------------- #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(np.array(self.xarray_forcing["time"]))
            month_day_inputs = extract_month_day_hour(np.array(WRF_input["time"]))
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = self.xarray_forcing.isel(time=ind_forcing)
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input["time"] = WRF_input["time"]
            # ------------------------------------------------------------------------------- #

            # merge
            WRF_input = WRF_input.merge(forcing_subset_input)

        # ========================================================================== #
        # merge static inputs
        if self.xarray_static:
            # expand static var on time dim
            N_time_dims = len(WRF_subset["time"])
            static_subset_input = self.xarray_static.expand_dims(dim={"time": N_time_dims})
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords({"time": WRF_subset["time"]})
            # slice, update time and merge
            static_subset_input = static_subset_input.isel(time=slice(0, self.history_len, 1))
            static_subset_input["time"] = WRF_input["time"]
            WRF_input = WRF_input.merge(static_subset_input)

        # ==================================================== #
        # xarray dataset as target
        ## WRF_target: the final target

        WRF_target = WRF_subset.isel(time=slice(self.history_len, ind_end_time, 1)).load()

        ## merge diagnoisc input here:
        if self.list_diag_ds:
            # subset diagnostic variables
            diagnostic_subset = self.list_diag_ds[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )

            diagnostic_subset = diagnostic_subset.isel(time=slice(self.history_len, ind_end_time, 1)).load()

            # merge into the target dataset
            WRF_target = WRF_target.merge(diagnostic_subset)

        # ==================================================== #
        # handle boundary files
        # ==================================================== #
        time_boundary = WRF_target["time"].values[0]  # <--- assuming single time value here
        time_round = next_n_hour(time_boundary, 3)

        if self.history_len_outside == 1:
            time_year = int(np.datetime_as_string(time_round, unit="Y"))
            ind_year = time_year - self.outside_file_year_range[0]
            ind_date = np.searchsorted(self.outside_file_indices[str(ind_year)], time_round)
            ds_upper_outside = self.list_upper_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
            ds_surf_outside = self.list_surf_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
            ds_outside = xr.merge([ds_upper_outside, ds_surf_outside])

        else:
            list_ds_upper_outside_slice = []
            list_ds_surf_outside_slice = []

            for i_time_backward in range(self.history_len_outside):
                time_round_loop = previous_hourly_steps(time_round, 3, i_time_backward)
                time_year = int(np.datetime_as_string(time_round_loop, unit="Y"))
                ind_year = time_year - self.outside_file_year_range[0]
                ind_date = np.searchsorted(self.outside_file_indices[str(ind_year)], time_round_loop)
                list_ds_upper_outside_slice.append(
                    self.list_upper_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
                )
                list_ds_surf_outside_slice.append(
                    self.list_surf_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
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

        # pipe xarray datasets to the sampler
        sample = Sample_WRF(
            WRF_input=WRF_input,
            WRF_target=WRF_target,
            boundary_input=ds_outside,
            time_encode=time_encode,
            datetime_index=datetime_as_number,
        )

        # ==================================== #
        # data normalization
        if self.transform:
            sample = self.transform(sample)

        # assign sample index
        sample["index"] = index

        return sample


class WRFPredict(torch.utils.data.IterableDataset):
    def __init__(
        self,
        param_interior,
        param_outside,
        fcst_datetime,
        lead_time_periods,
        rank,
        world_size,
        transform=None,
    ):
        ## no diagnostics because they are output only
        # varname_diagnostic = None
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

        # # -------------------------------------- #
        # # var names
        # self.varname_upper_air = varname_upper_air
        # self.varname_surface = varname_surface
        # self.varname_dyn_forcing = varname_dyn_forcing
        # self.varname_forcing = varname_forcing
        # self.varname_static = varname_static
        # self.varname_diagnostic = varname_diagnostic

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
        # -------------------------------------------------------------------------- #
        # get sample indices from boundary upper-air files:
        self.outside_file_year_range = [
            int(np.datetime_as_string(self.list_upper_ds_outside[0]["time"][0].values, unit="Y")),
            int(np.datetime_as_string(self.list_upper_ds_outside[-1]["time"][0].values, unit="Y")),
        ]

        self.outside_file_indices = {}  # <------ change
        for ind_file, outside_file_xarray in enumerate(self.list_upper_ds_outside):
            self.outside_file_indices[str(ind_file)] = outside_file_xarray["time"].values

        # -------------------------------------- #
        # other settings
        self.rank = rank
        self.world_size = world_size
        self.transform = transform
        self.history_len = param_interior["history_len"]
        self.init_datetime = fcst_datetime
        self.lead_time_periods = lead_time_periods

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

    def find_start_stop_indices(self, index):
        # ============================================================================ #
        # shift hours for history_len > 1, becuase more than one init times are needed
        shifted_hours = self.lead_time_periods * (self.history_len - 1)
        # ============================================================================ #
        # subtrack shifted_hour form the 1st & last init times
        # convert to datetime object

        # init_datetime = copy.deepcopy(self.init_datetime)
        init_datetime = self.init_datetime

        init_datetime[index][0] = datetime.datetime.strptime(
            init_datetime[index][0], "%Y-%m-%d %H:%M:%S"
        ) - datetime.timedelta(hours=shifted_hours)
        init_datetime[index][1] = datetime.datetime.strptime(
            init_datetime[index][1], "%Y-%m-%d %H:%M:%S"
        ) - datetime.timedelta(hours=shifted_hours)

        # convert the 1st & last init times to a list of init times
        init_datetime[index] = generate_datetime(
            init_datetime[index][0],
            init_datetime[index][1],
            self.lead_time_periods,
        )

        # convert datetime obj to nanosecondes
        init_time_list_dt = [np.datetime64(date.strftime("%Y-%m-%d %H:%M:%S")) for date in init_datetime[index]]

        # init_time_list_np: a list of python datetime objects, each is a forecast step
        # init_time_list_np[0]: the first initialization time
        # init_time_list_np[t]: the forcasted time of the (t-1)th step; the initialization time of the t-th step
        self.init_time_list_np = [
            np.datetime64(str(dt_obj) + ".000000000").astype(datetime.datetime) for dt_obj in init_time_list_dt
        ]

        info = []
        for init_time in self.init_time_list_np:
            for i_file, ds in enumerate(self.list_upper_ds):
                # get the year of the current file

                ds_values = ds["time"].values

                time_value = ensure_numpy_datetime(ds_values[0])
                ds_year = int(np.datetime_as_string(time_value, unit="Y"))

                # get the first and last years of init times
                init_year0 = nanoseconds_to_year(init_time)

                # found the right yearly file
                if init_year0 == ds_year:
                    N_times = len(ds_values)

                    # convert ds['time'] to a list of nanoseconds
                    ds_time_list = [
                        np.datetime64(ensure_numpy_datetime(ds_time)).astype("datetime64[ns]").astype(int)
                        for ds_time in ds_values
                    ]

                    ds_start_time = ds_time_list[0]
                    ds_end_time = ds_time_list[-1]

                    init_time_start = init_time

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

                        info.append([i_file, i_init_start, i_init_end, N_times])

        return info

    def __len__(self):
        return len(self.init_datetime)

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
            data_lookup = self.find_start_stop_indices(index)

            for k, _ in enumerate(self.init_time_list_np):
                # the first initialization time: get initalization from data
                i_file, i_init_start, i_init_end, N_times = data_lookup[k]

                # allocate output dict
                output_dict = {}

                # get all inputs in one xr.Dataset
                sliced_x = self.load_zarr_as_input(i_file, i_init_start, i_init_end, mode="input")

                # Check if additional data from the next file is needed
                if (len(sliced_x["time"]) < self.history_len) or (i_init_end + 1 >= N_times):
                    # Load excess data from the next file
                    next_file_idx = self.filenames.index(self.filenames[i_file]) + 1

                    if next_file_idx >= len(self.filenames):
                        # not enough input data to support this forecast
                        raise OSError("You have reached the end of the available data. Exiting.")

                    else:
                        sliced_y = self.load_zarr_as_input(i_file, i_init_end, i_init_end, mode="target")

                        # i_init_start = 0 because we need the beginning of the next file only
                        sliced_x_next = self.load_zarr_as_input(next_file_idx, 0, self.history_len, mode="input")
                        sliced_y_next = self.load_zarr_as_input(next_file_idx, 0, 1, mode="target")
                        # 1 becuase taregt is one step a time

                        # Concatenate excess data from the next file with the current data
                        sliced_x_combine = xr.concat([sliced_x, sliced_x_next], dim="time")
                        sliced_y_combine = xr.concat([sliced_y, sliced_y_next], dim="time")

                        sliced_x = sliced_x_combine.isel(time=slice(0, self.history_len))
                        sliced_y = sliced_y_combine.isel(time=slice(self.history_len, self.history_len + 1))
                else:
                    sliced_y = self.load_zarr_as_input(i_file, i_init_end + 1, i_init_end + 1, mode="target")

                # ========================== #
                # boundary conditions
                # sliced_y['time']
                time_boundary = sliced_y["time"].values[0]  # <--- assuming single time value here
                time_round = next_n_hour(time_boundary, 3)

                if self.history_len_outside == 1:
                    time_year = int(np.datetime_as_string(time_round, unit="Y"))
                    ind_year = time_year - self.outside_file_year_range[0]
                    ind_date = np.searchsorted(self.outside_file_indices[str(ind_year)], time_round)
                    ds_upper_outside = self.list_upper_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
                    ds_surf_outside = self.list_surf_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
                    ds_outside = xr.merge([ds_upper_outside, ds_surf_outside])

                else:
                    list_ds_upper_outside_slice = []
                    list_ds_surf_outside_slice = []

                    for i_time_backward in range(self.history_len_outside):
                        time_round_loop = previous_hourly_steps(time_round, 3, i_time_backward)
                        time_year = int(np.datetime_as_string(time_round_loop, unit="Y"))
                        ind_year = time_year - self.outside_file_year_range[0]
                        ind_date = np.searchsorted(self.outside_file_indices[str(ind_year)], time_round_loop)
                        list_ds_upper_outside_slice.append(
                            self.list_upper_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
                        )
                        list_ds_surf_outside_slice.append(
                            self.list_surf_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
                        )

                    ds_upper_outside = xr.concat(
                        list_ds_upper_outside_slice[::-1], dim="time"
                    )  # ::-1 so the latest time is the last
                    ds_surf_outside = xr.concat(list_ds_surf_outside_slice[::-1], dim="time")
                    ds_outside = xr.merge([ds_upper_outside, ds_surf_outside])

                # ==================================================== #
                # encode datetime input
                # ==================================================== #
                t0 = sliced_x["time"].values
                t1 = sliced_y["time"].values
                t2 = ds_outside["time"].values
                time_encode = encode_datetime64(np.concatenate([t0, t1, t2]))

                sample_x = {
                    "WRF_input": sliced_x,
                    "WRF_target": sliced_y,
                    "boundary_input": ds_outside,
                    "time_encode": time_encode,
                }

                if self.transform:
                    sample_x = self.transform(sample_x)

                for key in sample_x.keys():
                    output_dict[key] = sample_x[key]

                # <--- !! 'forecast_hour' is actually "forecast_step" but named by assuming hourly
                output_dict["forecast_hour"] = k + 1

                # Adjust stopping condition
                output_dict["stop_forecast"] = k == (len(self.init_time_list_np) - 1)
                output_dict["datetime"] = sliced_x.time.values.astype("datetime64[s]").astype(int)[-1]

                # return output_dict
                yield output_dict

                if output_dict["stop_forecast"]:
                    break
