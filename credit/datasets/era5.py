from glob import glob
import logging

import pandas as pd
import xarray as xr
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ERA5Dataset(Dataset):
    """Pytorch Dataset for processed ERA5 data. Relies on a configuration dictionary to define:
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
        1) The data must be stored in yearly zarr/netcdf files with a unique 4-digit year (YYYY) in the file name
        2) "time" dimension / coordinate is present with the datetime64[ns] datatype
        3) "level" dimension name representing the vertical level
        4) Dimention order of ('time', level', 'latitude', 'longitude') for 3D vars (remove level for 2D)
        5) Stored Zarr data should be chunked efficiently for a fast read (recommend small chunks across time dimension).

    """

    def __init__(self, config, time_config, source, transforms=None):
        self.source_name = source

        # valid sampling modes
        self.valid_sampling_modes = ["init", "forcing", "y", "stop"]

        # time config
        self.timestep = time_config["timestep"]
        self.num_forecast_steps = time_config["num_forecast_steps"]
        self.start_datetime = pd.Timestamp(time_config["start_datetime"])
        self.end_datetime = pd.Timestamp(time_config["end_datetime"])

        self.init_times = self._timestamps()
        self.years = [str(y) for y in self.init_times.year]  # only unique years

        self.file_dict = {}
        self.var_dict = {}

        # handle variables and their files
        for field_type, d in config["data"]["source"][self.source_name].items():  # prognostic, diagnostic, dynamic forcing, static, temporal_forcing_10m

            # FIX: Force ERA5 dataloader to completely ignore the 10-minute temporal forcing.
            # The GOES10kmDataset handles it natively to prevent hourly rounding!
            if field_type == "temporal_forcing_10m":
                continue

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
                    logger.info(
                        f"subsetting {field_type} to levels {self.var_dict[field_type]['levels']}"
                    )
            else:
                self.file_dict[field_type] = None

        # ------------------------------------------------------------------ #
        # LAZY DATASET CACHE
        #
        # Previously, all yearly datasets were opened here in __init__ using a
        # loop over self.file_dict, pre-populating self._ds_cache with
        # xr.open_zarr() handles. While convenient, this caused the same
        # "PyTorch + Dask Multiprocessing Collision" problem as GOES10kmDataset:
        #
        #   - xr.open_zarr() wraps data in dask.array by default
        #   - PyTorch forks the main process to create DataLoader workers
        #   - Every worker inherits a clone of the Dask graph and its internal
        #     thread pool state
        #   - On the first __getitem__ call, all workers simultaneously try to
        #     compute Dask graphs, spawning thousands of threads and saturating
        #     Lustre metadata bandwidth
        #
        # Fix: _ds_cache is left empty here. Each worker populates it lazily
        # on first access via _get_ds_for_year(), which opens files with
        # chunks=None to bypass Dask entirely. Since each DataLoader worker is
        # a separate process, the per-worker cache is never shared and there is
        # no race condition on the dict itself.
        #
        # With persistent_workers=True in the DataLoader, each worker's cache
        # is populated once and reused for the entire training run, so the
        # per-open overhead is fully amortized.
        # ------------------------------------------------------------------ #
        self._ds_cache: dict = {}

        # per-field transforms: dict of field_type -> ERA5FieldTransform
        self.transforms = transforms or {}

    def _timestamps(self):
        return pd.date_range(
            self.start_datetime,
            self.end_datetime - self.num_forecast_steps * self.timestep,
            freq=self.timestep,
        )

    def __len__(self):
        return len(self.init_times)

    def _map_files(self, file_list):
        """Create a dictionary to lookup the file for a timestep"""
        if len(file_list) > 1:
            file_map = {int(y): f for f in file_list for y in self.years if y in f}
        else:
            file_map = {int(y): file_list[0] for y in self.years}

        return file_map

    def _get_ds_for_year(self, field_type, year):
        """
        Return the cached dataset for (field_type, year), opening it on first
        access in this worker process.

        This is the core of the lazy-open strategy. Rather than opening all
        yearly files in __init__ (which would fork Dask-backed handles into
        every DataLoader worker), we open each file the first time it is
        actually needed inside the worker. Key properties:

          - chunks=None: opens Zarr stores with synchronous NumPy-backed I/O.
            No Dask thread pool is created, so there is zero risk of thread
            contention on Lustre when multiple workers read simultaneously.
          - Per-process cache: self._ds_cache is an instance attribute. Because
            DataLoader workers are separate processes (not threads), each worker
            has its own independent copy of _ds_cache. There is no shared state
            and no locking needed.
          - Bounded cache size: ERA5 data is stored in yearly files, so the
            cache for a single worker holds at most one open handle per
            (field_type, year) pair — typically a handful of entries total.
          - NetCDF fallback: non-Zarr files are opened with xr.open_dataset(),
            which is also synchronous and Dask-free.

        Args:
            field_type: one of "prognostic", "diagnostic", "dynamic_forcing", "static"
            year: integer calendar year (e.g. 2018)

        Returns:
            xr.Dataset opened with chunks=None (no Dask)
        """
        # Initialise the inner dict for this field_type on first access
        if field_type not in self._ds_cache:
            self._ds_cache[field_type] = {}

        if year not in self._ds_cache[field_type]:
            filepath = self.file_dict[field_type][year]
            logger.info(f"Lazy-opening {field_type} year={year}: {filepath}")
            # chunks=None bypasses Dask — all reads are synchronous numpy I/O.
            # This is safe inside a DataLoader worker because there is no Dask
            # thread pool to collide with PyTorch's own process management.
            if filepath.endswith('.zarr'):
                self._ds_cache[field_type][year] = xr.open_zarr(filepath, chunks=None, consolidated=True)
            else:
                self._ds_cache[field_type][year] = xr.open_dataset(filepath)

        return self._ds_cache[field_type][year]

    def __getitem__(self, args):
        """
        can return either a dictionary of tensors or an xarray object
        """
        ts, mode = args

        if "xarray" in mode:
            extract_fn = self._open_ds_extract_xarray
            mode = mode.split("_")[0]
        else:
            extract_fn = self._open_ds_extract_fields

        return_data = {"mode": mode, "stop_forecast": mode == "stop"}

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
            return_data = extract_fn("diagnostic", ts, return_data)
            return return_data

        raise ValueError(
            f"{mode} is not a valid sampling mode in {self.valid_sampling_modes}"
        )

    # Helper function to handle both .nc and .zarr files safely
    # NOTE: this method is retained for _open_ds_extract_xarray, which opens
    # fresh datasets on each call (xarray mode is not performance-critical).
    # _open_ds_extract_fields now uses _get_ds_for_year() instead so that the
    # hot path (tensor extraction) always goes through the lazy worker cache.
    def _open_file_safely(self, filepath):
        if filepath.endswith('.zarr'):
            return xr.open_zarr(filepath, chunks=None, consolidated=True)
        else:
            return xr.open_dataset(filepath)

    def _open_ds_extract_fields(self, field_type, ts, return_data):
        """
        opens the dataset, reshapes and concats the variables into an np array, packs it into the return dict if the data exists
        assumes both vars_3D and vars_2D are in the same file
        """
        if self.file_dict[field_type] and (self.var_dict[field_type]["vars_3D"] + self.var_dict[field_type]["vars_2D"]):
            # if the file map is not None, and there are variables do the op

            # --- CHANGED: use lazy per-worker cache instead of pre-opened handle ---
            # _get_ds_for_year opens the file with chunks=None on first access
            # in this worker, then caches it for all subsequent calls.
            # Previously this read from self._ds_cache[field_type][ts.year], which
            # was pre-populated in __init__ with Dask-backed datasets that caused
            # thread contention when forked into DataLoader workers.
            ds = self._get_ds_for_year(field_type, ts.year)

            if field_type != "static":
                # Fallback safety: If by some chance a file uses 't' instead of 'time', catch it gracefully
                time_coord = "time" if "time" in ds.dims else "t"
                ds = ds.sel({time_coord: ts}, method="nearest")

            ds = ds[
                self.var_dict[field_type]["vars_3D"]
                + self.var_dict[field_type]["vars_2D"]
            ]

            # Use dictionary-based transforms safely
            field_transform = self.transforms.get(field_type)
            if field_transform:
                ds = field_transform.transform_xarray(ds)

            ds_3D = ds[self.var_dict[field_type]["vars_3D"]]
            ds_2D = ds[self.var_dict[field_type]["vars_2D"]]

            levels = self.var_dict[field_type]["levels"]
            if levels:
                ds_3D = ds_3D.sel(level=levels)

            data_np = self._reshape_and_concat(ds_3D, ds_2D)

            if data_np.size > 0:
                return_data[field_type] = torch.tensor(data_np).float()
        else:
            # Fallback to prevent key errors downstream if a block is empty
            return_data[field_type] = torch.tensor([])

        return return_data

    def _open_ds_extract_xarray(self, field_type, ts, return_data):
        """
        warning: this does not transform the data
        """
        if self.file_dict[field_type]:  # if the file map is not None, do the op
            # Use _get_ds_for_year so that xarray-mode reads also go through
            # the lazy worker cache with chunks=None. Previously this called
            # _open_file_safely(filepath) on every __getitem__ invocation,
            # opening a fresh file handle each time. Using the cache avoids
            # redundant opens and ensures consistent chunks=None behaviour.
            ds = self._get_ds_for_year(field_type, ts.year)

            if field_type != "static":
                time_coord = "time" if "time" in ds.dims else "t"
                ds = ds.sel({time_coord: ts}, method="nearest")

            ds = ds[
                self.var_dict[field_type]["vars_3D"]
                + self.var_dict[field_type]["vars_2D"]
            ]

            if ds:
                return_data[field_type] = ds

        return return_data

    def _reshape_and_concat(self, ds_3D, ds_2D):
        """Stack 3D variables along level and variable, concatenate with 2D variables, and reorder dimesions."""

        # for 3D, order by variables (according to the order in config file) then levels
        data_list = []
        if ds_3D:
            data_3D = (
                ds_3D.to_array().stack({"level_var": ["variable", "level"]}).values
            )
            data_3D = np.expand_dims(data_3D.transpose(2, 0, 1), axis=1)
            data_list.append(data_3D)
        if ds_2D:
            data_2D = np.expand_dims(ds_2D.to_array().values, axis=1)
            data_list.append(data_2D)

        if data_list:
            combined_data = np.concatenate(data_list, axis=0)
            return combined_data
        else:
            return np.array([])


if __name__ == "__main__":
    import yaml

    path = "/glade/u/home/dkimpara/miles-credit/config/era5_new_data_config.yaml"
    with open(path) as cnfg:
        config = yaml.safe_load(cnfg)

    data_config = config["data"]

    time_config = {
        "timestep": pd.Timedelta(data_config["timestep"]),
        "num_forecast_steps": data_config["forecast_len"] + 1,
        "start_datetime": data_config["start_datetime"],
        "end_datetime": data_config["end_datetime"],
    }
    source = "ERA5"
    dataset = ERA5Dataset(config, time_config, source)

    ts = dataset.init_times[0]

    for mode in dataset.valid_sampling_modes:
        print(dataset[(ts, mode)].keys())