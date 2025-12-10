import logging
import pandas as pd
import numpy as np
import xarray as xr
import torch
import itertools

logger = logging.getLogger(__name__)


class ERA5Dataset(Dataset):
    """ Pytorch Dataset for processed ERA5 data. Relies on a configuration dictionary to define:
            1) 2D / 3D variables
            2) Start, End and Frequency of Datetimes
            3) path to glob for the data
            4) Example YAML Format:

                data:
                  source:
                    ERA5:
                      vars_3D: ['T', 'U', 'V', 'Q']
                      vars_2D: ['T500', 'U500', 'V500', 'Q500' ,'Z500', 'tsi', 't2m','SP']
                      vars_persist: None
                      path: "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/all_in_one/"

                  start_datetime: "2017-01-01"
                  end_datetime: "2019-12-31"
                  timestep: "6h"

        Assumptions:
            1) The data must be stored in yearly zarr files with a unique 4-digit year (YYYY) in the file name
            2) "time" dimension / coordinate is present with the datetime64[ns] datatype
            3) "level" dimension name representing the vertical level
            4) Dimention order of ('time', level', 'latitude', 'longitude') for 3D vars (remove level for 2D)
            5) Stored Zarr data should be chunked efficiently for a fast read (recommend small chunks across time dimension).

            """

    def __init__(self, config, source="ERA5", transform=None):

        self.source_name = source

        # valid sampling modes
        self.valid_sampling_modes = ["init", "forcing", "y", "stop"]

        # time config
        self.timestep = pd.Timedelta(config['timestep'])
        self.num_forecast_steps = config["forecast_len"] + 1
        self.start_datetime = pd.Timestamp(config['start_datetime'])
        self.end_datetime = pd.Timestamp(config['end_datetime'])

        self.init_times = self._timestamps()
        self.years = [str(y) for y in self.init_times.year]

        self.file_dict = {}
        self.var_dict = {}

        for field_type, d in config['source'][self.source_name].items():
            if isinstance(d, dict):
                files = sorted(glob(d.get("path", "")))
                # stores a dict to lookup files from that field
                self.file_dict[field_type] = self._map_files(files) if files else None

                self.var_dict[field_type] = {
                    "vars_3D": d.get("vars_3D", []),
                    "vars_2D": d.get("vars_2D", []),
                    "levels": d.get("levels", []),  # will return all levels by default
                }
                if self.var_dict[field_type]["levels"]:
                    logger.info(f"subsetting {field_type} to levels {self.var_dict[field_type]['levels']}")
            else:
                self.file_dict[field_type] = None

        # handle transform: (not yet implimented)
        if not transform:
            self.transform_xarray = lambda x: x
        else:
            self.transform_xarray = transform.transform_xarray
        # print("TIMES", self.init_times)

    def _timestamps(self):

        return pd.date_range(self.start_datetime, self.end_datetime - self.num_forecast_steps * self.timestep,
                             freq=self.timestep)

    def __len__(self):

        return len(self.init_times)

    def _map_files(self, file_list):

        """ Create a dictionary to lookup the file for a timestep """

        if len(file_list) > 1:
            file_map = {int(y): f for f in file_list for y in self.years if y in f}
        else:
            file_map = {int(y): file_list[0] for y in self.years}

        return file_map

    def __getitem__(self, args):
        """
            can return either a dictionary of tensors or an xarray object
        """
        ts, mode = args
        ts = pd.Timestamp(ts)
        extract_fn = self._open_ds_extract_fields

        return_data = {"mode": mode,
                       "stop_forecast": mode == "stop",
                       "time": int(ts.value)}

        return_data = extract_fn("dynamic_forcing", ts, return_data)
        if mode == "forcing":
            return return_data

        # load prognostic for the remaining modes
        return_data = extract_fn("prognostic", ts, return_data)
        if mode == "init":
            # load static
            return_data = extract_fn("static", ts, return_data)
            return return_data

        if mode == "y" or mode == "stop":
            # load diagnostic
            return_data = extract_fn("diagnostic", ts, return_data)  # not yet implimented
            return return_data

        raise ValueError(f"{mode} is not a valid sampling mode in {self.valid_sampling_modes}")

    def _open_ds_extract_fields(self, field_type, ts, return_data):
        """
        opens the dataset, reshapes and concats the variables into an np array, packs it into the return dict if the data exists
        assumes both vars_3D and vars_2D are in the same file.
        """
        if self.file_dict[field_type]:
            ds = xr.open_dataset(self.file_dict[field_type][ts.year])
            if field_type != "static":
                ds = ds.sel(time=ts)

            ds = ds[self.var_dict[field_type]["vars_3D"] + self.var_dict[field_type]["vars_2D"]]
            if field_type in ["prognostic", "dynamic_forcing"]:  # not yet implimented
                ds = self.transform_xarray(ds)

            ds_3D = ds[self.var_dict[field_type]["vars_3D"]]
            ds_2D = ds[self.var_dict[field_type]["vars_2D"]]

            levels = self.var_dict[field_type]["levels"]
            if levels:
                ds_3D = ds_3D.sel(level=levels)

            data_np = self._reshape_and_concat(ds_3D, ds_2D)

            if data_np.size > 0:
                return_data[field_type] = torch.tensor(data_np).float()

        return return_data

    def _reshape_and_concat(self, ds_3D, ds_2D):

        """ Stack 3D variables along level and variable, concatenate with 2D variables, and reorder dimesions. """

        # for 3D, order by variables (according to the order in config file) then levels
        data_list = []
        if ds_3D:
            data_3D = ds_3D.to_array().stack({'level_var': ['variable', 'level']}).values
            data_3D = np.expand_dims(data_3D.transpose(2, 0, 1), axis=1)
            data_list.append(data_3D)
        if ds_2D:
            data_2D = np.expand_dims(ds_2D.to_array().values, axis=1)
            data_list.append(data_2D)

        combined_data = np.concatenate(data_list, axis=0)

        return combined_data


    if __name__ == "__main__":
        import yaml
        from credit.samplers import DistributedMultiStepBatchSampler
        path = "../config/era5_new_data_config.yaml"
        with open(path) as cnfg:
            config = yaml.safe_load(cnfg)

        data_config = config["data"]

        dataset = ERA5Dataset(config['data'])
        sampler = DistributedMultiStepBatchSampler(dataset=dataset, batch_size=4, sampling_modes=['init'] + ['y'] * 1,
                                                   rank=0, num_replicas=1, shuffle=True)
        loader = iter(DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=8, pin_memory=True))

        for _ in range(10):
            batch = next(loader)
            print(batch.keys())

