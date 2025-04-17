import torch
import xarray as xr
from credit.data import keep_dataset_vars, get_forward_data
import pandas as pd
import numpy as np


class RealtimePredictDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        forecast_start_time,
        forecast_end_time,
        forecast_timestep,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_static,
        varname_diagnostic,
        filenames=None,
        filename_surface=None,
        filename_dyn_forcing=None,
        filename_forcing=None,
        filename_static=None,
        sst_forcing=None,
        history_len=1,
        transform=None,
        seed=42,
        rank=0,
        world_size=1,
    ):
        # var names
        self.varname_upper_air = varname_upper_air
        self.varname_surface = varname_surface
        self.varname_dyn_forcing = varname_dyn_forcing
        self.varname_static = varname_static
        self.varname_diagnostic = varname_diagnostic

        self.filenames = sorted(filenames)
        self.filename_surface = filename_surface
        self.filename_dyn_forcing = filename_dyn_forcing
        self.filename_forcing = filename_forcing
        self.filename_static = filename_static

        self.forecast_start_time = pd.Timestamp(forecast_start_time)
        self.forecast_end_time = pd.Timestamp(forecast_end_time)
        self.forecast_timestep = forecast_timestep
        self.forecast_times = pd.date_range(
            start=self.forecast_start_time,
            end=self.forecast_end_time,
            freq=self.forecast_timestep,
        )
        self.forecast_len = 0
        self.history_len = history_len
        self.transform = transform
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # set random seed
        self.rng = np.random.default_rng(seed=seed)

        # sst forcing
        self.sst_forcing = sst_forcing

        # flags to determine if any of the [surface, dyn_forcing, diagnostics]
        # variable groups share the same file as upper air variables
        flag_share_surf = False
        flag_share_dyn = False

        # blocks that can handle no-sharing (each group has it own file)
        # surface
        self.surface_files = None
        if filename_surface is not None:
            surface_files = {}
            filename_surface = sorted(filename_surface)

            if filenames == filename_surface:
                flag_share_surf = True
            else:
                for fn in filename_surface:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_surf = keep_dataset_vars(ds, varname_surface)
                    surf_init_time = pd.Timestamp(ds_surf["time"][0].values)
                    surface_files[surf_init_time] = ds_surf

                self.surface_files = surface_files
        else:
            self.surface_files = None

        # dynamic forcing
        dyn_forcing_files = None
        if filename_dyn_forcing is not None:
            dyn_forcing_files = {}
            filename_dyn_forcing = sorted(filename_dyn_forcing)
            if filenames == filename_dyn_forcing:
                flag_share_dyn = True
            else:
                for fn in filename_dyn_forcing:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_dyn = keep_dataset_vars(ds, varname_dyn_forcing)
                    dyn_year = pd.Timestamp(ds_dyn["time"][0].values).year
                    dyn_forcing_files[dyn_year] = ds_dyn

                self.dyn_forcing_files = dyn_forcing_files
        else:
            self.dyn_forcing_files = None

        all_files = {}
        # blocks that can handle file sharing (share with upper air file)
        if self.surface_files is None and len(self.varname_surface) > 0:
            self.surface_files = {}
        if self.dyn_forcing_files is None and len(self.varname_dyn_forcing) > 0:
            self.dyn_forcing_files = {}
        for fn in self.filenames:
            # drop variables if they are not in the config
            ds = get_forward_data(filename=fn)
            ds_upper = keep_dataset_vars(ds, varname_upper_air)
            upper_init_time = pd.Timestamp(ds_upper["time"][0].values)
            if flag_share_surf:
                ds_surf = keep_dataset_vars(ds, varname_surface)
                self.surface_files[upper_init_time] = ds_surf

            if flag_share_dyn:
                ds_dyn = keep_dataset_vars(ds, varname_dyn_forcing)
                dyn_forcing_files[upper_init_time.year] = ds_dyn

            all_files[upper_init_time] = ds_upper

        # file names
        self.all_files = all_files

        if self.filename_static is not None:
            # drop variables if they are not in the config
            static_dataset = get_forward_data(filename_static)
            static_dataset = keep_dataset_vars(static_dataset, varname_static)

            self.xarray_static = static_dataset.load()
        else:
            self.xarray_static = None
        return

    def __len__(self):
        return self.forecast_times.size - 1

    def __getitem__(self, idx):
        valid_date = self.forecast_times[idx]
        valid_year = valid_date.year
        batch = {}
        if idx == 0:
            if self.history_len == 1:
                x_list = []
                upper_x = (
                    self.all_files[valid_date]
                    .sel(time=valid_date)
                    .expand_dims("time", 0)
                    .load()
                )
                surface_x = (
                    self.surface_files[valid_date]
                    .sel(time=valid_date)
                    .expand_dims("time", 0)
                    .load()
                )
                x_list.extend([upper_x, surface_x])
                if self.filename_dyn_forcing is not None:
                    dyn_forcing_x = (
                        self.dyn_forcing_files[valid_year]
                        .sel(time=valid_date)
                        .expand_dims("time", 0)
                        .load()
                    )
                    x_list.append(dyn_forcing_x)
                if self.filename_static is not None:
                    static_x = self.xarray_static
                    static_x["time"] = dyn_forcing_x["time"]
                    for var in static_x.data_vars:
                        static_x[var] = static_x[var].expand_dims(time=static_x.time)
                    x_list.append(static_x)
                sliced_x = xr.merge(x_list)
            else:
                raise NotImplementedError
        else:
            if self.history_len == 1:
                x_list = []
                if self.filename_dyn_forcing is not None:
                    dyn_forcing_x = (
                        self.dyn_forcing_files[valid_year]
                        .sel(time=valid_date)
                        .expand_dims("time", 0)
                        .load()
                    )
                    x_list.append(dyn_forcing_x)
                if self.filename_static is not None:
                    static_x = self.xarray_static
                    static_x["time"] = dyn_forcing_x["time"]
                    x_list.append(static_x)
                sliced_x = xr.merge(x_list)
            else:
                raise NotImplementedError
        sample = {"historical_ERA5_images": sliced_x}
        if self.transform:
            sample = self.transform(sample)
        sample["index"] = idx
        sample["datetime"] = valid_date
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                value = torch.tensor(value)
            elif isinstance(value, np.int64):
                value = torch.tensor(value, dtype=torch.int64)
            elif isinstance(value, (int, float)):
                value = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, pd.Timestamp):
                value = torch.tensor(value.timestamp(), dtype=torch.float64)
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
        batch["forecast_step"] = torch.tensor(idx + 1)
        batch["stop_forecast"] = idx == (self.forecast_times.size - 1)
        return batch
