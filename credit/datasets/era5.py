"""
era5.py
-------
ARCOERA5Dataset: PyTorch Dataset for streaming ERA5 data from the Google Cloud
ARCO ERA5 public Zarr store.

Sample structure returned by __getitem__::

    {
        "input": {
            "{source_name}/prognostic/3d/temperature":              tensor,  # (n_levels, 1, lat, lon)
            "{source_name}/prognostic/2d/surface_pressure":         tensor,  # (1,        1, lat, lon)
            "{source_name}/dynamic_forcing/2d/toa_incident_solar_radiation": tensor,
            "{source_name}/static/2d/land_sea_mask":                tensor,
            ...
        },
        "target": {                                  # only when return_target=True
            "{source_name}/prognostic/3d/temperature":      tensor,
            "{source_name}/prognostic/2d/surface_pressure": tensor,
            ...
        },
        "metadata": {
            "input_datetime":  int,                  # nanoseconds since epoch
            "target_datetime": int,                  # only when return_target=True
        },
    }

Output key format (flat, slash-delimited):
    "{source_name}/{field_type}/{dim}/{varname}"

    field_type: "prognostic" | "dynamic_forcing" | "static" | "diagnostic"
    dim       : "2d"  (surface / single-level)
                "3d"  (multi-level; level_coord = "level" or "hybrid")
    varname   : variable name as in the ARCO ERA5 Zarr store
"""

from __future__ import annotations

import cftime
from typing import Any

from gcsfs import GCSFileSystem
import pandas as pd
import torch
import xarray as xr
import zarr

from credit.datasets._utils import _to_cftime  # pyright: ignore[reportPrivateUsage]
from credit.datasets.base_dataset import BaseDataset


class ARCOERA5Dataset(BaseDataset):
    """PyTorch Dataset for Google Cloud ARCO ERA5 data with nested input/target structure.

    See module docstring for full description of output format and file naming.

    Example YAML configuration::

        data:
          source:
            Example_ARCOERA5:  # User-provided name (arbitrary key)
              dataset_type: "arco_era5"
              level_coord: "hybrid"
              levels: [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]
              variables:
                prognostic:
                  vars_3D: ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"]
                  vars_2D: ["surface_pressure"]
                dynamic_forcing:
                  vars_2D: ["toa_incident_solar_radiation"]
                static:
                  vars_2D: ["land_sea_mask"]
                diagnostic:
                  vars_2D: ["total_precipitation"]

          start_datetime: "2017-01-01"
          end_datetime: "2019-12-31"
          timestep: "6h"
          forecast_len: 1

    Assumptions:
        1. A "time" dimension / coordinate is present for non-static fields.
        2. A level coordinate (name given by ``level_coord``) represents the
           vertical axis of 3D variables.
        3. Dimension order: (time, level, latitude, longitude) for 3D;
           (time, latitude, longitude) for 2D; (latitude, longitude) for static.
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """Initialize ARCOERA5Dataset with config parsing, timestamp generation, file mapping from BaseDataset,
        then set ARCOERA5-specific attributes.

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return target variables. Defaults to False.
        """
        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "arco_era5", (
            f"Expected dataset_type 'arco_era5' in config for ARCOERA5Dataset, got '{self.curr_source_cfg['dataset_type']}'"
        )

        # Set ARCOERA5-specific attributes
        self.dataset_type = "arco_era5"
        self.pressure_lev_era5_path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        self.model_lev_era5_path = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
        self.model_lev_vars = [
            "divergence",
            "fraction_of_cloud_cover",
            "geopotential",
            "ozone_mass_mixing_ratio",
            "specific_cloud_ice_water_content",
            "specific_cloud_liquid_water_content",
            "specific_humidity",
            "specific_rain_water_content",
            "specific_snow_water_content",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "vorticity",
        ]
        self.level_coord: str = self.curr_source_cfg[
            "level_coord"
        ]  # hybrid for model levels and level for pressure levels
        if "levels" not in self.curr_source_cfg:
            # Assume all levels are being requested
            if self.level_coord == "hybrid":
                self.levels: list[int] = list(range(1, 138))
            else:
                self.levels: list[int] = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70] + list(range(100, 1025, 25))
        else:
            self.levels: list[int] = self.curr_source_cfg["levels"]
        self.mod_level_store = None
        self.pres_level_store = None
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }
        self.mode = "remote"

        # Initialize the field registration based on the provided config and populate
        #   dictionary of variables and file paths for each field type
        self.init_register_all_fields()

        # Initialize the s3fs on the first call to _extract_field within __getitem__
        self._fs = None

    def _init_fs(self):
        """Initialize the GCSFileSystem and zarr stores for pressure-level and model-level ERA5 data."""
        fs_config: dict[str, Any] = {
            "cache_timeout": -1,
            "token": "anon",  # noqa: S106 # nosec B106
            "access": "read_only",
            "block_size": 8**20,
            "asynchronous": True,
            "skip_instance_cache": True,
        }
        self._fs = GCSFileSystem(**fs_config)
        self.pres_level_store = zarr.storage.FsspecStore(fs=self._fs, path=self.pressure_lev_era5_path)
        self.mod_level_store = zarr.storage.FsspecStore(fs=self._fs, path=self.model_lev_era5_path)

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """
        Open the dataset for *field_type* at time *t* and populate *sample*.

        Keys written are ``"{source_name}/{field_type}/3d/{varname}"`` for 3D variables
        and ``"{source_name}/{field_type}/2d/{varname}"`` for 2D variables.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t: Timestamp to select.
            sample: Dict to write variable tensors into (modified in place).
                Tensor shapes (no batch dimension):

                - 3D variable: ``(n_levels, 1, lat, lon)``
                - 2D variable: ``(1, 1, lat, lon)``
        """
        if self._fs is None:
            self._init_fs()

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]
        if self.level_coord == "level":
            with xr.open_zarr(self.pres_level_store, chunks=None) as ds:
                # Select the time step; static fields have no time dim
                if "time" in ds.dims:
                    if isinstance(ds.time.values[0], cftime.datetime):
                        calendar = ds.time.values[0].calendar
                        t_sel = _to_cftime(t, calendar)
                    else:
                        t_sel = t
                    ds_t = ds.sel(time=t_sel)
                else:
                    ds_t = ds

                # 3D variables: (n_levels, lat, lon) → (n_levels, 1, lat, lon)
                for vname in vars_3D:
                    arr = ds_t[vname].sel({self.level_coord: self.levels}).values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                    key = self._get_field_name(field_type, "3d", vname)
                    sample[key] = tensor

                # 2D variables: (lat, lon) → (1, 1, lat, lon)
                for vname in vars_2D:
                    arr = ds_t[vname].values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    key = self._get_field_name(field_type, "2d", vname)
                    sample[key] = tensor
        else:
            with xr.open_zarr(self.mod_level_store, chunks=None) as ds:
                # Select the time step; static fields have no time dim
                if "time" in ds.dims:
                    if isinstance(ds.time.values[0], cftime.datetime):
                        calendar = ds.time.values[0].calendar
                        t_sel = _to_cftime(t, calendar)
                    else:
                        t_sel = t
                    ds_t = ds.sel(time=t_sel)
                else:
                    ds_t = ds

                # 3D variables: (n_levels, lat, lon) → (n_levels, 1, lat, lon)
                for vname in vars_3D:
                    arr = ds_t[vname].sel({self.level_coord: self.levels}).values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                    key = self._get_field_name(field_type, "3d", vname)
                    sample[key] = tensor

            with xr.open_zarr(self.pres_level_store, chunks=None) as ds:
                # Select the time step; static fields have no time dim
                if "time" in ds.dims:
                    if isinstance(ds.time.values[0], cftime.datetime):
                        calendar = ds.time.values[0].calendar
                        t_sel = _to_cftime(t, calendar)
                    else:
                        t_sel = t
                    ds_t = ds.sel(time=t_sel)
                else:
                    ds_t = ds
                # 2D variables: (lat, lon) → (1, 1, lat, lon)
                for vname in vars_2D:
                    arr = ds_t[vname].values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    key = self._get_field_name(field_type, "2d", vname)
                    sample[key] = tensor
