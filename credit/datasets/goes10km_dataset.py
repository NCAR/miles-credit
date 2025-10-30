from typing import  Dict

import torch
from torch.utils.data import Dataset


from os.path import join
import os
import numpy as np
import pandas as pd
import xarray as xr
import logging
logger = logging.getLogger(__name__)


class GOES10kmDataset(Dataset):
    def __init__(self,
                 ds: xr.Dataset,
                 data_conf: Dict,
                 time_config: Dict = None,
                 valid_init_dir = "/glade/derecho/scratch/dkimpara/goes-cloud-dataset/valid_init_times",
                 scaler_ds_path = "/glade/derecho/scratch/dkimpara/goes-cloud-dataset/data_stats.nc",):
        """
        taking advantage of DistributedSampler class code with this dataset
        
        Args:
            ds: xr dataset with a time attribute

            example time config:
            {"timestep": pd.Timedelta(1, "h"),
                "num_forecast_steps": 1
                 },
        """
        self.ds = ds.drop_duplicates(dim="t").sortby("t")
        self.num_items = len(ds)
        
        self.timestep = time_config["timestep"]
        self.num_forecast_steps = time_config["num_forecast_steps"]
        self.years = time_config["years"]
        self.valid_sampling_modes = ["init", "forcing", "y", "stop"]
        
        self.valid_init_dir = valid_init_dir
        self.init_times = self._timestamps() # will generate valid init times if needed

        # setup scaler

        self.log_normal_scaling = data_conf.get("log_normal_scaling", False)
        if self.log_normal_scaling:
            scaler_ds_path = "/glade/derecho/scratch/dkimpara/goes-cloud-dataset/data_stats_logC04.nc"
            logger.info("log normalizing visible channels")
        
        self.scaler_ds = xr.open_dataset(scaler_ds_path)

        # setup rollout mode if configured
        if "rollout_init_times" in time_config.keys():
            logger.info("setting up GOES 10km dataset for rollout mode by subsetting times")
            self._rollout_mode(time_config["rollout_init_times"],
                               time_config.get("time_tol", (1, "D")),
                               )


    def _rollout_mode(self, rollout_init_times, time_tol):
        
        logger.info(f"selecting rollout times with time tolerance {time_tol}")
        self.init_times = self.init_times.sel(t=rollout_init_times, method="nearest",
                                              tolerance=pd.Timedelta(time_tol[0], time_tol[1])
                                              )

    def _generate_valid_init_times(self, valid_init_filepath):
        # due to missing data, need to have a different list of valid init times
        def check_valid_forecast_times(t_init, timestep, num_forecast_steps):
            time_tolerance = pd.Timedelta(11, "m")
            target_times = [t_init.values + timestep * step for step in range(1, num_forecast_steps + 1)]
            zarr_times = self.ds.t.sel(t=target_times, method="nearest")
            within_tol = (zarr_times.values - np.array(target_times).astype(zarr_times.dtype)) < time_tolerance

            return all(within_tol)

        logger.info(f"generating valid init times and saving to {valid_init_filepath}. will take around 5 min")

        valid_times = []
        for t in self.ds.t:
            if check_valid_forecast_times(t, self.timestep, self.num_forecast_steps):
                valid_times.append(t.values)

        valid_times_da = xr.DataArray(valid_times, coords={"t": valid_times})

        times_to_drop = xr.open_dataarray(join(self.valid_init_dir, "nan_times.nc"))
        valid_times_da = valid_times_da[ ~ valid_times_da.t.isin(times_to_drop)]

        valid_times_da.to_netcdf(valid_init_filepath)
        logger.info(f"wrote valid init times to {valid_init_filepath}")

        return valid_times_da

    def _timestamps(self):
        # grab or compute valid init times across whole dataset, due to missing data
        # this removes need to mess with the samplers
        filename = f"{self.timestep.seconds // 3600:02d}h_{self.num_forecast_steps:02}step.nc"
        valid_init_filepath = join(self.valid_init_dir, filename)
        
        if os.path.exists(valid_init_filepath):
            timestamps = xr.open_dataarray(valid_init_filepath)
        else:
            timestamps = self._generate_valid_init_times(valid_init_filepath)
        
        logger.debug(f"loading years {self.years}")
        timestamps = timestamps.sel(t=(timestamps.t.dt.year >= self.years[0]) 
                                    & (timestamps.t.dt.year < self.years[1]))

        return timestamps
    
    def __len__(self):
        # total number of valid start times
        return len(self.init_times)
    
    def inverse_transform_ABI(self, da):
        # da must be in the same order as the source data
        # channels must be the first dimension
        unscaled = da * self.scaler_ds["std"] + self.scaler_ds["mean"]

        if self.log_normal_scaling:
            da[0] = np.exp(unscaled[0])

        return da
        
    def _normalize_ABI(self, da):
        return (da - self.scaler_ds["mean"]) / self.scaler_ds["std"]

    def _nanfill_ABI(self, da):
        return da.fillna(0.0)

    def __getitem__(self, args):
        # default: load target state
        ts, mode = args

        ds = self.ds.sel(t=ts, method="nearest")
        # no need to check time tolerance, should be taken care of by init time generation
        
        time_str = pd.Timestamp(ds.t.values).strftime("%Y-%m-%dT%H:%M:%S")

        if mode == "forcing":
            return {"mode": mode,
                    "stop_forecast": False,
                    "datetime": time_str,}
        
        da = ds["BT_or_R"].copy()

        if self.log_normal_scaling: #channels is the first axis
            da[0] = np.log(da[0])
        
        da = self._normalize_ABI(da)
        da = self._nanfill_ABI(da)
        
        # da.shape = c, 1003, 923
        data = torch.tensor(da.values).unsqueeze(1)
        # data = torch.nn.functional.pad(data, (0,0,11,10), "replicate")
        data = torch.nn.functional.pad(data, (19,18,11,10), "constant", 0.0)
        # coerce to c, t, 1024, 960 to work with wxformer

        if mode == "init":
            return {"x": data,
                    "mode": mode,
                    "stop_forecast": False,
                    "datetime": time_str,}
        elif mode == "y":
            return {"y": data,
                    "mode": mode,
                    "stop_forecast": False,
                    "datetime": time_str,}      
        elif mode == "stop":
            return {"y": data,
                    "mode": mode,
                    "stop_forecast": True,
                    "datetime": time_str,}
        else:
            raise ValueError(f"{mode} is not a valid sampling mode")
