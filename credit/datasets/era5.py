import xarray as xr
import pandas as pd
import numpy as np
import cftime
import torch
from torch.utils.data import Dataset
import logging
from glob import glob

logger = logging.getLogger(__name__)

VALID_FIELD_TYPES = {"prognostic", "dynamic_forcing", "static", "diagnostic"}


class ERA5Dataset(Dataset):
    """
    Pytorch Dataset for processed ERA5 data. Relies on a configuration dictionary to define:
        1) 2D / 3D variables
        2) Start, End and Frequency of Datetimes
        3) path to glob for the data

    Example YAML configuration
    --------------------------
    .. code-block:: yaml

        data:
          source:
            ERA5:
              level_coord: "level"
              levels: [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]
              variables:
                prognostic:
                  vars_3D: ['T', 'U', 'V', 'Q']
                  vars_2D: ['T500', 'U500', 'V500', 'Q500' ,'Z500', 'tsi', 't2m','SP']
                  path: "<path to prognostic>"
                diagnostic: null
                static:
                  vars_2D: ['Z_GDS4_SFC','LSM']
                  path: "<path to static>"
                dynamic_forcing:
                  vars_2D: ['tsi']
                  path: "<path to dynamic forcing>"

        start_datetime: "2017-01-01"
        end_datetime: "2019-12-31"
        timestep: "6h"

    Assumptions:
        1) The data MUST be stored in yearly zarr or netCDF files with a unique 4-digit year (YYYY) in the file name
        2) "time" dimension / coordinate is present
        3) "level" or "pressure" coordinate name representing the vertical level
        4) Dimension order of ('time', level/pressure', 'latitude', 'longitude') for 3D vars (remove level/pressure for 2D)
        5) Data should be chunked efficiently for a fast read (recommend small chunks across time dimension).
    """

    def __init__(self, config, return_target=False):
        self.source_name = "ERA5"
        self.level_coord = config["source"]["ERA5"]["level_coord"]
        self.levels = config["source"]["ERA5"]["levels"]
        self.return_target = return_target
        self.dt = pd.Timedelta(config["timestep"])
        self.num_forecast_steps = config["forecast_len"] + 1
        self.start_datetime = pd.Timestamp(config["start_datetime"])
        self.end_datetime = pd.Timestamp(config["end_datetime"])
        self.datetimes = self._timestamps()
        self.years = [str(y) for y in self.datetimes.year]
        self.file_dict = {}
        self.var_dict = {}
        self.variable_meta = self._build_var_metadata(config)

        for field_type, d in config["source"][self.source_name]['variables'].items():
            if field_type not in VALID_FIELD_TYPES:
                raise KeyError(
                    f"Unknown field_type '{field_type}' in config['source']['{self.source_name}']. "
                    f"Valid options are: {sorted(VALID_FIELD_TYPES)}"
                )

            if isinstance(d, dict):
                if not d.get("vars_3D") and not d.get("vars_2D"):
                    raise ValueError(f"Field '{field_type}' must define at least one of vars_3D or vars_2D")

                files = sorted(glob(d.get("path", "")))
                self.file_dict[field_type] = self._map_files(files) if files else None
                self.var_dict[field_type] = {
                    "vars_3D": d.get("vars_3D", []),
                    "vars_2D": d.get("vars_2D", []),
                }
            else:
                self.file_dict[field_type] = None

    def _timestamps(self):
        """
        return total time steps
        """
        return pd.date_range(
            self.start_datetime,
            self.end_datetime - self.num_forecast_steps * self.dt,
            freq=self.dt,
        )

    def __len__(self):
        return len(self.datetimes)

    def _map_files(self, file_list):
        """
        Create a dictionary to lookup the file for a timestep

        Args:
             file_list (list): List of file paths
        """
        if len(file_list) > 1:
            file_map = {int(y): f for f in file_list for y in self.years if y in f}
        else:
            file_map = {int(y): file_list[0] for y in self.years}

        return file_map

    def __getitem__(self, args):
        """
        Returns a sample of data.

        Args:
            args (tuple): Input_time step from sampler, step index from sampler
        """
        return_data = {"metadata": {}}
        t, i = args
        t = pd.Timestamp(t)
        t_target = t + self.dt
        
        
        # always load dynamic forcing
        self._open_ds_extract_fields("dynamic_forcing", t, return_data)

        # load prognostic and static if first time step
        if i == 0:
            self._open_ds_extract_fields("static", t, return_data)
            self._open_ds_extract_fields("prognostic", t, return_data)
        
        # load t+1 if training
        if self.return_target:
            for key in ("prognostic", "diagnostic"):
                if key in self.file_dict.keys():
                    self._open_ds_extract_fields(key, t_target, return_data, is_target=True)
            self._pop_and_merge_targets(return_data)
            return_data["metadata"]["target_datetime"] = int(t_target.value)
            
        return_data["metadata"]["input_datetime"] = int(t.value)
        
        return return_data

    def _open_ds_extract_fields(self, field_type, t, return_data, is_target=False):
        """
        opens the dataset, reshapes and concats the variables into n np array,
        packs it into the return dict if the data exists.

        Args:
             field_type (str): Field type ("prognostic", "diagnostic", etc)
             t (pd.Timestamp): Current timestamp
             return_data (dict): Dictionary of data to return
             is_target (bool): Flag for if data is x or y data
        """
        if self.file_dict[field_type]:
            with xr.open_dataset(self.file_dict[field_type][t.year]) as dataset:
                if "time" in dataset.dims:
                    if isinstance(dataset.time.values[0], cftime.datetime):
                        t = self._convert_cf_time(t)
                    ds = dataset.sel(time=t)
                else:
                    ds = dataset
                ds_all_vars = ds[self.var_dict[field_type]["vars_3D"] + self.var_dict[field_type]["vars_2D"]]

                ds_3D = ds_all_vars[self.var_dict[field_type]["vars_3D"]]
                ds_2D = ds_all_vars[self.var_dict[field_type]["vars_2D"]]
                data_np = self._reshape_and_concat(ds_3D, ds_2D)

                if is_target:
                    if field_type == "prognostic":
                        return_data["target_prognostic"] = torch.tensor(data_np).float()
                    elif field_type == "diagnostic":
                        return_data["target_diagnostic"] = torch.tensor(data_np).float()
                    
                else:
                    return_data[field_type] = torch.tensor(data_np).float()                

    def _reshape_and_concat(self, ds_3D, ds_2D):
        """
        Stack 3D variables along level and variable, concatenate with 2D variables, and reorder dimensions.

        Args:
            ds_3D (xr.Dataset): Xarray dataset with 3D spatial variables
            ds_2D (xr.Dataset): Xarray dataset with 2D spatial variables
        """
        data_list = []

        if ds_3D:
            data_3D = ds_3D.sel({self.level_coord: self.levels}).to_array().stack({"level_var": ["variable", self.level_coord]})
            data_3D = np.expand_dims(data_3D.values.transpose(2, 0, 1), axis=1)
            data_list.append(data_3D)

        if ds_2D:
            data_2D = ds_2D.to_array()
            data_2D = np.expand_dims(data_2D, axis=1)
            data_list.append(data_2D)

        combined_data = np.concatenate(data_list, axis=0)
        
        return combined_data

    def _build_var_metadata(self, config):
        """ Build variable order metadata """
        
        var_meta = {}
        source_cfg = config['source'][self.source_name]
        levels = source_cfg.get("levels", [])
        variables = source_cfg.get("variables", {}) or {}
    
        for field_type, spec in variables.items():
            if spec is None:
                continue
    
            var_meta[field_type] = []
    
            # Expand 3D variables over levels
            for v in (spec.get("vars_3D") or []):
                for lev in levels:
                    var_meta[field_type].append(f"{self.source_name}_{v}_{lev}")
    
            # Add 2D variables directly
            for v in (spec.get("vars_2D") or []):
                var_meta[field_type].append(f"{self.source_name}_{v}")
    
        return var_meta

    def _convert_cf_time(self, ts):
        """
        Convert pandas timestamp to cftime

        Args:
            ts: pandas timestamp
        """
        cf_t = cftime.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, calendar="noleap")

        return cf_t

    def _pop_and_merge_targets(self, return_data, dim=0):
        """
        Look for target diagnostic and prognostic variables. If both exist, concatenate them along specified dimension.

        Args:
            return_data: Dictionary of current data to return
            dim: Concat dimension
        """
        target_tensors = []
        for key in ("target_prognostic", "target_diagnostic"):
            if key in return_data:
                target_tensors.append(return_data.pop(key))

        if not target_tensors:
            return

        return_data["target"] = target_tensors[0] if len(target_tensors) == 1 else torch.cat(target_tensors, dim=dim)
