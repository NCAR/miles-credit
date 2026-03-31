import datetime
import numpy as np
import xarray as xr

import torch
import torch.utils.data
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

from credit.data import Sample_dscale

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
    encode_datetime64,
    filter_ds,
)


class Dscale_Dataset(torch.utils.data.Dataset):
    """
    Initialization model Pytorch Dataset class
    """

    def __init__(
        self,
        param_HR,
        param_LR,
        transform=None,
        seed=42,
    ):
        # ========================================================== #
        # HR domain variable and filename info
        varname_upper_air = param_HR["varname_upper_air"]
        varname_surface = param_HR["varname_surface"]
        varname_dyn_forcing = param_HR["varname_dyn_forcing"]
        varname_forcing = param_HR["varname_forcing"]
        varname_static = param_HR["varname_static"]
        varname_diagnostic = param_HR["varname_diagnostic"]
        filenames = param_HR["filenames"]
        filename_surface = param_HR["filename_surface"]
        filename_dyn_forcing = param_HR["filename_dyn_forcing"]
        filename_forcing = param_HR["filename_forcing"]
        filename_static = param_HR["filename_static"]
        filename_diagnostic = param_HR["filename_diagnostic"]
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
        self.history_len = param_HR["history_len"]
        self.forecast_len = param_HR["forecast_len"]
        self.total_seq_len = self.history_len + self.forecast_len
        # -------------------------------------------------------------------------- #
        # get sample indices from HR upper-air files:
        ind_start = 0
        self.HR_file_indices = {}  # <------ change
        for ind_file, HR_file_xarray in enumerate(self.list_upper_ds):
            # [number of samples, ind_start, ind_end]
            self.HR_file_indices[str(ind_file)] = [
                len(HR_file_xarray["time"]),
                ind_start,
                ind_start + len(HR_file_xarray["time"]),
            ]
            ind_start += len(HR_file_xarray["time"]) + 1

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

        if filename_dyn_forcing or filename_forcing or filename_static:
            self.flag_HR_input = True
        else:
            self.flag_HR_input = False

        # ========================================================== #
        # LR variable and filename info
        varname_upper_air_LR = param_LR["varname_upper_air"]
        varname_surface_LR = param_LR["varname_surface"]
        filenames_LR = param_LR["filenames"]
        filename_surface_LR = param_LR["filename_surface"]
        # ----------------------------------------------------------- #
        # collecting xr.datasets
        list_upper_ds_LR = []
        list_surf_ds_LR = []
        filenames_LR = sorted(filenames_LR)

        for fn_LR in filenames_LR:
            # drop variables if they are not in the config
            ds_LR = get_forward_data(filename=fn_LR)
            ds_upper_LR = drop_var_from_dataset(ds_LR, varname_upper_air_LR)

            if filename_surface_LR is not None:
                ds_surf_LR = drop_var_from_dataset(ds_LR, varname_surface_LR)
                list_surf_ds_LR.append(ds_surf_LR)
            else:
                self.list_surf_ds_LR = False

            list_upper_ds_LR.append(ds_upper_LR)

        self.list_upper_ds_LR = list_upper_ds_LR
        self.list_surf_ds_LR = list_surf_ds_LR
        self.history_len_LR = param_LR["history_len"]
        self.forecast_len_LR = param_LR["forecast_len"]
        self.total_seq_len = self.history_len_LR + self.forecast_len_LR
        # -------------------------------------------------------------------------- #
        # get sample indices from LR upper-air files:
        self.LR_file_year_range = [
            int(np.datetime_as_string(self.list_upper_ds_LR[0]["time"][0].values, unit="Y")),
            int(np.datetime_as_string(self.list_upper_ds_LR[-1]["time"][0].values, unit="Y")),
        ]

        self.LR_file_indices = {}  # <------ change
        for ind_file, LR_file_xarray in enumerate(self.list_upper_ds_LR):
            self.LR_file_indices[str(ind_file)] = LR_file_xarray["time"].values

        # ========================================================== #
        # shared by the two domains
        self.transform = transform
        self.rng = np.random.default_rng(seed=seed)

        self.levels_LR = param_LR["levels"]
        self.level_pick_LR = param_LR["level_pick"]

        self.levels_HR = param_HR["levels"]
        self.level_pick_HR = param_HR["level_pick"]

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len  # + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for HR_file_xarray in self.list_upper_ds:
            total_len += len(HR_file_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def __getitem__(self, index):
        # ========================================================================== #
        # cross-year indices --> the index of the year + indices within that year

        # select the ind_file based on the iter index
        ind_file = find_key_for_number(index, self.HR_file_indices)

        # get the ind within the current file
        ind_start = self.HR_file_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        # handle out-of-bounds
        ind_largest = len(self.list_upper_ds[int(ind_file)]["time"]) - (self.history_len + 1)

        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # ========================================================================== #
        # subset xarray on time dimension

        ind_end_in_file = ind_start_in_file + self.history_len  # + self.forecast_len

        ## HR_file_subset: a xarray dataset that contains training input and target (for the current batch)
        HR_subset = self.list_upper_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file))

        # ========================================================================== #
        # merge surface into the dataset

        if self.list_surf_ds:
            ## subset surface variables
            surface_subset = self.list_surf_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file))

            ## merge upper-air and surface here:
            HR_subset = HR_subset.merge(surface_subset)

        ## merge diagnoisc input here:
        if self.list_diag_ds:
            # subset diagnostic variables
            diagnostic_subset = self.list_diag_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file))

            # merge into the target dataset
            HR_subset = HR_subset.merge(diagnostic_subset)

        # the ind_end of the HR_subset
        ind_end_time = len(HR_subset["time"])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = HR_subset.time.values.astype("datetime64[s]").astype(int)

        # ==================================================== #\
        # load HR_target
        HR_target = HR_subset.load()

        # ========================================================================== #
        # merge dynamic forcing inputs
        if self.flag_HR_input:
            # creats HR_input from HR_target
            HR_input = xr.Dataset(coords={k: v.copy(deep=False) for k, v in HR_target.coords.items()})
        else:
            HR_input = None

        if self.list_dyn_forcing_ds:
            dyn_forcing_subset = self.list_dyn_forcing_ds[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(time=slice(0, self.history_len, 1)).load()

            HR_input = HR_input.merge(dyn_forcing_subset)

        # ========================================================================== #
        # merge forcing inputs
        if self.xarray_forcing:
            # ------------------------------------------------------------------------------- #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(np.array(self.xarray_forcing["time"]))
            month_day_inputs = extract_month_day_hour(np.array(HR_input["time"]))
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = self.xarray_forcing.isel(time=ind_forcing)
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input["time"] = HR_input["time"]
            # ------------------------------------------------------------------------------- #
            # merge
            HR_input = HR_input.merge(forcing_subset_input)

        # ========================================================================== #
        # merge static inputs
        if self.xarray_static:
            # expand static var on time dim
            N_time_dims = len(HR_subset["time"])
            static_subset_input = self.xarray_static.expand_dims(dim={"time": N_time_dims})
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords({"time": HR_input["time"]})
            # slice, update time and merge
            static_subset_input = static_subset_input.isel(time=slice(0, self.history_len, 1))
            static_subset_input["time"] = HR_input["time"]
            HR_input = HR_input.merge(static_subset_input)

        # ==================================================== #
        # handle LR files
        # ==================================================== #
        time_HR = HR_target["time"].values[0]

        time_year = int(np.datetime_as_string(time_HR, unit="Y"))
        ind_year = time_year - self.LR_file_year_range[0]
        ind_date = np.searchsorted(self.LR_file_indices[str(ind_year)], time_HR)
        ds_upper_LR = self.list_upper_ds_LR[ind_year].isel(time=slice(ind_date, ind_date + self.history_len_LR))
        ds_surf_LR = self.list_surf_ds_LR[ind_year].isel(time=slice(ind_date, ind_date + self.history_len_LR))
        ds_LR = xr.merge([ds_upper_LR, ds_surf_LR])

        # ==================================================== #
        # encode datetime input
        # ==================================================== #
        t_LR = ds_LR["time"].values
        time_encode = encode_datetime64(t_LR)

        if (self.level_pick_LR is not None) and (self.level_pick_HR is not None):
            ds_LR = ds_LR.isel(level=slice(0, self.level_pick_LR))
            HR_input = HR_input.isel(bottom_top=slice(0, self.level_pick_HR))
            HR_target = HR_target.isel(bottom_top=slice(0, self.level_pick_HR))

        # self.levels_LR = param_LR["levels"]
        # self.level_pick_LR = param_LR["level_pick"]
        # self.levels_HR = param_HR["levels"]
        # self.level_pick_HR = param_HR["level_pick"]

        # pipe xarray datasets to the sampler
        sample = Sample_dscale(
            HR_input=HR_input,
            HR_target=HR_target,
            LR_input=ds_LR,
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


class Dscale_Predict(torch.utils.data.IterableDataset):
    def __init__(
        self,
        param_HR,
        param_LR,
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
        varname_upper_air = param_HR["varname_upper_air"]
        varname_surface = param_HR["varname_surface"]
        varname_dyn_forcing = param_HR["varname_dyn_forcing"]
        varname_forcing = param_HR["varname_forcing"]
        varname_static = param_HR["varname_static"]
        varname_diagnostic = param_HR["varname_diagnostic"]
        filenames = param_HR["filenames"]
        filename_surface = param_HR["filename_surface"]
        filename_dyn_forcing = param_HR["filename_dyn_forcing"]
        filename_forcing = param_HR["filename_forcing"]
        filename_static = param_HR["filename_static"]
        filename_diagnostic = param_HR["filename_diagnostic"]

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

        if filename_dyn_forcing or filename_forcing or filename_static:
            self.flag_HR_input = True
        else:
            self.flag_HR_input = False

        # ========================================================== #
        # boundary variable and filename info
        varname_upper_air_LR = param_LR["varname_upper_air"]
        varname_surface_LR = param_LR["varname_surface"]
        filenames_LR = param_LR["filenames"]
        filename_surface_LR = param_LR["filename_surface"]
        # ----------------------------------------------------------- #
        # collecting xr.datasets
        list_upper_ds_LR = []
        list_surf_ds_LR = []
        filenames_LR = sorted(filenames_LR)

        for fn_LR in filenames_LR:
            # drop variables if they are not in the config
            ds_LR = get_forward_data(filename=fn_LR)
            ds_upper_LR = drop_var_from_dataset(ds_LR, varname_upper_air_LR)

            if filename_surface_LR is not None:
                ds_surf_LR = drop_var_from_dataset(ds_LR, varname_surface_LR)
                list_surf_ds_LR.append(ds_surf_LR)
            else:
                self.list_surf_ds_LR = False

            list_upper_ds_LR.append(ds_upper_LR)

        self.list_upper_ds_LR = list_upper_ds_LR
        self.list_surf_ds_LR = list_surf_ds_LR
        self.history_len_LR = param_LR["history_len"]
        self.forecast_len_LR = param_LR["forecast_len"]
        # -------------------------------------------------------------------------- #
        # get sample indices from boundary upper-air files:
        self.LR_file_year_range = [
            int(np.datetime_as_string(self.list_upper_ds_LR[0]["time"][0].values, unit="Y")),
            int(np.datetime_as_string(self.list_upper_ds_LR[-1]["time"][0].values, unit="Y")),
        ]

        self.LR_file_indices = {}  # <------ change
        for ind_file, LR_file_xarray in enumerate(self.list_upper_ds_LR):
            self.LR_file_indices[str(ind_file)] = LR_file_xarray["time"].values

        # -------------------------------------- #
        # other settings
        self.rank = rank
        self.world_size = world_size
        self.transform = transform
        self.history_len = param_HR["history_len"]
        self.init_datetime = fcst_datetime
        self.lead_time_periods = lead_time_periods

        self.levels_LR = param_LR["levels"]
        self.level_pick_LR = param_LR["level_pick"]

        self.levels_HR = param_HR["levels"]
        self.level_pick_HR = param_HR["level_pick"]

    def load_zarr_as_input(self, i_file, i_init_start, i_init_end, mode="input"):
        if mode == "target":
            # sliced_x: the final output, starts with an upper air xr.dataset
            sliced_x = self.list_upper_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))

            # surface variables
            if self.filename_surface is not None:
                sliced_surface = self.list_surf_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))
                sliced_x = sliced_x.merge(sliced_surface)

            # diagnostic
            if self.filename_diagnostic is not None:
                sliced_diagnostic = self.list_diag_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))
                sliced_x = sliced_x.merge(sliced_diagnostic)

            return sliced_x

        if mode == "input":
            sliced_HR = self.list_upper_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))
            sliced_x = xr.Dataset(coords={k: v.copy(deep=False) for k, v in sliced_HR.coords.items()})

            # dynamic forcing variables
            if self.filename_dyn_forcing is not None:
                sliced_dyn_forcing = self.list_dyn_forcing_ds[i_file].isel(time=slice(i_init_start, i_init_end + 1))
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
                        sliced_y_next = self.load_zarr_as_input(next_file_idx, 0, self.history_len, mode="target")
                        # 1 becuase taregt is one step a time

                        # Concatenate excess data from the next file with the current data
                        sliced_x_combine = xr.concat([sliced_x, sliced_x_next], dim="time")
                        sliced_y_combine = xr.concat([sliced_y, sliced_y_next], dim="time")

                        sliced_x = sliced_x_combine.isel(time=slice(0, self.history_len))
                        sliced_y = sliced_y_combine.isel(time=slice(0, self.history_len))
                else:
                    sliced_y = self.load_zarr_as_input(i_file, i_init_start, i_init_end, mode="target")

                # ========================== #
                # LR inputs
                # sliced_y['time']
                time_HR = sliced_x["time"].values[0]

                time_year = int(np.datetime_as_string(time_HR, unit="Y"))
                ind_year = time_year - self.LR_file_year_range[0]
                ind_date = np.searchsorted(self.LR_file_indices[str(ind_year)], time_HR)
                ds_upper_LR = self.list_upper_ds_LR[ind_year].isel(time=slice(ind_date, ind_date + self.history_len))
                ds_surf_LR = self.list_surf_ds_LR[ind_year].isel(time=slice(ind_date, ind_date + self.history_len))
                ds_LR = xr.merge([ds_upper_LR, ds_surf_LR])

                # ==================================================== #
                # encode datetime input
                # ==================================================== #
                t_LR = ds_LR["time"].values
                time_encode = encode_datetime64(t_LR)

                if (self.level_pick_LR is not None) and (self.level_pick_HR is not None):
                    ds_LR = ds_LR.isel(level=slice(0, self.level_pick_LR))
                    sliced_x = sliced_x.isel(bottom_top=slice(0, self.level_pick_HR))
                    sliced_y = sliced_y.isel(bottom_top=slice(0, self.level_pick_HR))

                sample_x = {
                    "HR_input": sliced_x,
                    "HR_target": sliced_y,
                    "LR_input": ds_LR,
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
