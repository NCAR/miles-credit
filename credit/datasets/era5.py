"""
era5.py
-------------------------------------------------------
Refactored ERA5Dataset and ARCOERA5Dataset with nested input/target structure.

Sample structure returned by __getitem__:

.. code-block:: python

    {
        "input": {
            "Example_ERA5/era5/prognostic/3d/T":        tensor,  # (n_levels, 1, lat, lon)
            "Example_ERA5/era5/prognostic/2d/SP":       tensor,  # (1,        1, lat, lon)
            "Example_ERA5/era5/dynamic_forcing/2d/tsi": tensor,
            "Example_ERA5/era5/static/2d/LSM":          tensor,
            ...
        },
        "target": {                                  # only when return_target=True
            "Example_ERA5/era5/prognostic/3d/T":        tensor,
            "Example_ERA5/era5/prognostic/2d/SP":       tensor,
            ...
        },
        "metadata": {
            "input_datetime":  int,                  # nanoseconds since epoch
            "target_datetime": int,                  # only when return_target=True
        },
    }

Output key format (flat, slash-delimited):
    "Example_ERA5/{dataset_type}/{field_type}/{dim}/{varname}"

    dataset_type: "era5"
    field_type: "prognostic" | "dynamic_forcing" | "static" | "diagnostic"
    dim       : "2d"  (surface / single-level)
                "3d"  (multi-level upper-air)
    varname   : variable name as given in config (e.g. "T", "SP", "tsi")

Tensor shapes (no batch dimension):
    3D variable : (n_levels, 1, lat, lon)   — n_levels = len(config levels)
    2D variable : (1,        1, lat, lon)   — singleton level dim

After DataLoader collation the batch dimension is prepended:
    (batch, n_levels, 1, lat, lon)

File naming:
    Each field type supports an optional ``filename_time_format`` config key
    that specifies a strftime format string describing how the datetime appears
    in the file name.  Defaults to ``"%Y"`` (annual files).

    Examples::

        filename_time_format: "%Y"       # era5_2021.zarr
        filename_time_format: "%Y_%m"    # era5_2021_06.nc
        filename_time_format: "%Y%m%d"   # era5_20210601.nc

    If only a single file matches the glob pattern, ``filename_time_format`` is
    ignored and that file is used for all timestamps.
"""

from __future__ import annotations

import cftime
from typing import Any

from gcsfs import GCSFileSystem
import pandas as pd
import torch
import xarray as xr
import zarr

from credit.datasets._utils import _find_file, _to_cftime  # pyright: ignore[reportPrivateUsage]
from credit.datasets.base_dataset import BaseDataset


class ERA5Dataset(BaseDataset):
    """PyTorch Dataset for processed ERA5 data with nested input/target structure.

    See module docstring for full description of output format and file naming.

    Example YAML configuration::

        data:
          source:
            Example_ERA5:  # User-provided name (arbitrary key)
              dataset_type: "era5"
              level_coord: "level"
              levels: [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]
              variables:
                prognostic:
                  vars_3D: ['T', 'U', 'V', 'Q']
                  vars_2D: ['SP', 't2m']
                  path: "/data/era5_*.zarr"
                  filename_time_format: "%Y"        # annual (default)
                dynamic_forcing:
                  vars_2D: ['tsi']
                  path: "/data/solar_*.nc"
                  filename_time_format: "%Y_%m"     # monthly
                static:
                  vars_2D: ['Z_GDS4_SFC', 'LSM']
                  path: "/data/lsm.nc"
                  # single file — filename_time_format not needed
                diagnostic: null

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
        """Initialize ERA5Dataset with config parsing, timestamp generation, file mapping from BaseDataset,
        then set ERA5-specific attributes.

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return target variables. Defaults to False.
        """
        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "era5", (
            f"Expected dataset_type 'era5' in config for ERA5Dataset, got '{self.curr_source_cfg['dataset_type']}'"
        )

        # Set ERA5-specific attributes
        self.dataset_type = "era5"
        self.level_coord: str = self.curr_source_cfg["level_coord"]
        self.levels: list[int] = self.curr_source_cfg["levels"]
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }
        self.mode = "local"

        # Initialize the field registration based on the provided config and populate
        #   dictionary of variables and file paths for each field type
        self.init_register_all_fields()

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """Open the dataset for *field_type* at time *t* and populate *sample*.

        Keys written are ``"Example_ERA5/{dataset_type}/{field_type}/3d/{varname}"`` for 3D variables
        and ``"Example_ERA5/{dataset_type}/{field_type}/2d/{varname}"`` for 2D variables.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t: Timestamp to select.
            sample: Dict to write variable tensors into (modified in place).
                Tensor shapes (no batch dimension):

                - 3D variable: ``(n_levels, 1, lat, lon)``
                - 2D variable: ``(1, 1, lat, lon)``
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals or field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        with xr.open_dataset(_find_file(file_intervals, t)) as ds:
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
        self.fs = None

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
        self.fs = GCSFileSystem(**fs_config)
        self.pres_level_store = zarr.storage.FsspecStore(fs=self.fs, path=self.pressure_lev_era5_path)
        self.mod_level_store = zarr.storage.FsspecStore(fs=self.fs, path=self.model_lev_era5_path)

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """
        Open the dataset for *field_type* at time *t* and populate *sample*.

        Keys written are ``"Example_ARCOERA5/{dataset_type}/{field_type}/3d/{varname}"`` for 3D variables
        and ``"Example_ARCOERA5/{dataset_type}/{field_type}/2d/{varname}"`` for 2D variables.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t: Timestamp to select.
            sample: Dict to write variable tensors into (modified in place).
                Tensor shapes (no batch dimension):

                - 3D variable: ``(n_levels, 1, lat, lon)``
                - 2D variable: ``(1, 1, lat, lon)``
        """
        if self.fs is None:
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
