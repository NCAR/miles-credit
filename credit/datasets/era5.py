import logging
import pandas as pd
import numpy as np
import xarray as xr
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
from glob import glob

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
                      prognostic:
                        vars_3D: ['T', 'U', 'V', 'Q']
                        vars_2D: ['T500', 'U500', 'V500', 'Q500' ,'Z500', 'tsi', 't2m','SP']
                        path: "<path to prognostic>"
                      diagnostic:
                        vars_3D: ['T', 'U', 'V', 'Q']
                        vars_2D: ['T500', 'U500', 'V500', 'Q500' ,'Z500', 'tsi', 't2m','SP']
                        path: "<path to diagnostic>"
                      static:
                        vars_3D: ['T', 'U', 'V', 'Q']
                        vars_2D: ['T500', 'U500', 'V500', 'Q500' ,'Z500', 'tsi', 't2m','SP']
                        path: "<path to static>"
                      dynamic_forcing:
                        vars_3D: ['T', 'U', 'V', 'Q']
                        vars_2D: ['T500', 'U500', 'V500', 'Q500' ,'Z500', 'tsi', 't2m','SP']
                        path: "<path to dynamic forcing>"

                start_datetime: "2017-01-01"
                end_datetime: "2019-12-31"
                timestep: "6h"

        Assumptions:
            1) The data MUST be stored in yearly zarr or netCDF files with a unique 4-digit year (YYYY) in the file name
            2) "time" dimension / coordinate is present
            3) "level" dimension name representing the vertical level
            4) Dimention order of ('time', level', 'latitude', 'longitude') for 3D vars (remove level for 2D)
            5) Data should be chunked efficiently for a fast read (recommend small chunks across time dimension).
            """

    def __init__(self, config, return_target=False, transform=None):

        self.source_name = "ERA5"
        self.return_target = return_target
        self.dt = pd.Timedelta(config['timestep'])
        self.num_forecast_steps = config["forecast_len"] + 1
        self.start_datetime = pd.Timestamp(config['start_datetime'])
        self.end_datetime = pd.Timestamp(config['end_datetime'])
        self.datetimes = self._timestamps()
        self.years = [str(y) for y in self.datetimes.year]
        self.file_dict = {}
        self.var_dict = {}

        for field_type, d in config['source'][self.source_name].items():
            if isinstance(d, dict):
                files = sorted(glob(d.get("path", "")))

                # stores a dict to lookup files from that field
                self.file_dict[field_type] = self._map_files(files) if files else None

                self.var_dict[field_type] = {"vars_3D": d.get("vars_3D", []),
                                             "vars_2D": d.get("vars_2D", [])}
                # "levels": d.get("levels", [])} # not implimented yet

                # if self.var_dict[field_type]["levels"]:

                #     logger.info(f"subsetting {field_type} to levels {self.var_dict[field_type]['levels']}")
            else:

                self.file_dict[field_type] = None

        # handle transform: (not yet implimented)
        if not transform:
            self.transform_xarray = lambda x: x

        else:
            self.transform = transform.transform

    def _timestamps(self):

        """ return total time steps """

        return pd.date_range(self.start_datetime, self.end_datetime - self.num_forecast_steps * self.dt, freq=self.dt)

    def __len__(self):

        return len(self.datetimes)

    def _map_files(self, file_list):

        """ Create a dictionary to lookup the file for a timestep """

        if len(file_list) > 1:
            file_map = {int(y): f for f in file_list for y in self.years if y in f}
        else:
            file_map = {int(y): file_list[0] for y in self.years}

        return file_map

    def __getitem__(self, args):
        """
        Returns a sample of data.

        args (tuple): Input_time step from sampler, step index from sampler
        """

        return_data = {"metadata": {}}
        t, i = args
        t = pd.Timestamp(t)
        t_target = t + self.dt
        self._open_ds_extract_fields("dynamic_forcing", t, return_data)

        # load prognostic and static if first time step
        if i == 0:
            self._open_ds_extract_fields("static", t, return_data)
            self._open_ds_extract_fields("prognostic", t, return_data)

        # load t+1 if training
        if self.return_target:
            self._open_ds_extract_fields("prognostic", t_target, return_data, is_target=True)

        self._add_metadata(return_data, t, t_target)
        # print(return_data['metadata'].items())

        return return_data

    def _open_ds_extract_fields(self, field_type, t, return_data, is_target=False):

        """ opens the dataset, reshapes and concats the variables into an np array, packs it into the return dict if the data exists. """

        if self.file_dict[field_type]:
            ds = xr.open_dataset(self.file_dict[field_type][t.year])
            if isinstance(z.time[0].item(), cftime.datetime):
                t = self._convert_cf_time(t)

            if field_type != "static":
                ds = ds.sel(time=t)

            ds = ds[self.var_dict[field_type]["vars_3D"] + self.var_dict[field_type]["vars_2D"]]

            ds_3D = ds[self.var_dict[field_type]["vars_3D"]]
            ds_2D = ds[self.var_dict[field_type]["vars_2D"]]
            data_np, meta = self._reshape_and_concat(ds_3D, ds_2D)

            return_data["metadata"][f"{field_type}_var_order"] = meta
            if is_target:
                return_data["target"] = torch.tensor(data_np).float()
            else:
                return_data[field_type] = torch.tensor(data_np).float()

    def _reshape_and_concat(self, ds_3D, ds_2D):

        """ Stack 3D variables along level and variable, concatenate with 2D variables, and reorder dimesions. """

        data_list = []
        meta_3D, meta_2D = [], []

        if ds_3D:
            data_3D = ds_3D.to_array().stack({'level_var': ['variable', 'level']})
            meta_3D = data_3D.level_var.values.tolist()
            data_3D = np.expand_dims(data_3D.values.transpose(2, 0, 1), axis=1)
            data_list.append(data_3D)

        if ds_2D:
            data_2D = ds_2D.to_array()
            meta_2D = data_2D["variable"].values.tolist()
            data_2D = np.expand_dims(data_2D, axis=1)
            data_list.append(data_2D)

        combined_data = np.concatenate(data_list, axis=0)
        meta = meta_3D + meta_2D

        return combined_data, meta

    def _add_metadata(self, return_data, t, t_target=None):

        """ Update metadata dictionary """

        return_data["metadata"]["input_datetime"] = int(t.value)
        return_data["metadata"]["dimension_order"] = ["batch", "variable", "time", "latitude", "longitude"]

        if self.return_target:
            return_data["metadata"]['target_datetime'] = int(t_target.value)

    def _convert_cf_time(self, ts):

        """ Convert pandas timestamp to cftime """

        cf_t = cftime.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, calendar='noleap')

        return cf_t


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

