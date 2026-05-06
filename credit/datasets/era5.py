"""
era5.py
-------------------------------------------------------
Refactored ERA5Dataset with nested input/target structure.

Sample structure returned by __getitem__:

.. code-block:: python

    {
        "input": {
            "era5/prognostic/3d/T":        tensor,  # (n_levels, 1, lat, lon)
            "era5/prognostic/2d/SP":       tensor,  # (1,        1, lat, lon)
            "era5/dynamic_forcing/2d/tsi": tensor,
            "era5/static/2d/LSM":          tensor,
            ...
        },
        "target": {                                  # only when return_target=True
            "era5/prognostic/3d/T":        tensor,
            "era5/prognostic/2d/SP":       tensor,
            ...
        },
        "metadata": {
            "input_datetime":  int,                  # nanoseconds since epoch
            "target_datetime": int,                  # only when return_target=True
        },
    }

Output key format (flat, slash-delimited):
    "{source}/{field_type}/{dim}/{varname}"

    source    : "era5"
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

from typing import Any

import cftime

import pandas as pd
import torch
import xarray as xr
from gcsfs import GCSFileSystem
import zarr

from credit.datasets._utils import _find_file
from credit.datasets.base_dataset import BaseDataset


class ERA5Dataset(BaseDataset):
    """PyTorch Dataset for processed ERA5 data with nested input/target structure.

    See module docstring for full description of output format and file naming.

    Example YAML configuration::

        data:
          source:
            ERA5:
              dataset_name: "era5"
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
        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)

        assert self.curr_source_cfg["dataset_name"] == "era5", (
            f"Expected dataset_name 'era5' in config for ERA5Dataset, got '{self.curr_source_cfg['dataset_name']}'"
        )
        self.dataset_name = "era5"

        self.source_name: str = "era5"
        self.level_coord: str = self.curr_source_cfg["level_coord"]
        self.levels: list[int] = self.curr_source_cfg["levels"]
        self.return_target: bool = return_target
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }

        # self.dt = pd.Timedelta(data_config["timestep"])
        # self.num_forecast_steps: int = data_config["forecast_len"]

        # self.start_datetime = pd.Timestamp(data_config["start_datetime"])
        # self.end_datetime = pd.Timestamp(data_config["end_datetime"])
        # self.datetimes: pd.DatetimeIndex = self._build_timestamps()

        # # file_dict maps field_type → sorted list of (start, end, path) intervals
        # self.file_dict: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]] | None] = {}
        # self.var_dict: dict[str, dict[str, list[str]]] = {}

        # for field_type, d in source_cfg["variables"].items():
        #     self._register_field(field_type, d)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    # inheriting from BaseDataset

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """Open the dataset for *field_type* at time *t* and populate *sample*.

        Keys written are ``"era5/{field_type}/3d/{varname}"`` for 3D variables
        and ``"era5/{field_type}/2d/{varname}"`` for 2D variables.

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
                    t_sel = self._to_cftime(t, calendar)
                else:
                    t_sel = t
                ds_t = ds.sel(time=t_sel)
            else:
                ds_t = ds

            # 3D variables: (n_levels, lat, lon) → (n_levels, 1, lat, lon)
            for vname in vars_3D:
                arr = ds_t[vname].sel({self.level_coord: self.levels}).values
                tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                sample[f"{self.dataset_name}/{field_type}/3d/{vname}"] = tensor

            # 2D variables: (lat, lon) → (1, 1, lat, lon)
            for vname in vars_2D:
                arr = ds_t[vname].values
                tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                sample[f"{self.dataset_name}/{field_type}/2d/{vname}"] = tensor


class ARCOERA5Dataset(BaseDataset):
    """PyTorch Dataset for Google Cloud ARCO ERA5 data with nested input/target structure.

    See module docstring for full description of output format and file naming.

    Example YAML configuration::

        data:
          source:
            ARCO_ERA5:
              dataset_name: "arco_era5"
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
        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)

        assert self.curr_source_cfg["dataset_name"] == "arco_era5", (
            f"Expected dataset_name 'arco_era5' in config for ARCOERA5Dataset, got '{self.curr_source_cfg['dataset_name']}'"
        )
        self.dataset_name = "arco_era5"

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
        # self.source_name: str = "arco_era5"
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
        self.return_target: bool = return_target
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }

        # self.dt = pd.Timedelta(data_config["timestep"])
        # self.num_forecast_steps: int = data_config["forecast_len"]

        # self.start_datetime = pd.Timestamp(data_config["start_datetime"])
        # self.end_datetime = pd.Timestamp(data_config["end_datetime"])
        # self.datetimes: pd.DatetimeIndex = self._build_timestamps()

        # # file_dict maps field_type → sorted list of (start, end, path) intervals
        # self.file_dict: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
        # self.var_dict: dict[str, dict[str, list[str]]] = {}

        self.mode = "remote"

        for field_type, d in self.curr_source_cfg["variables"].items():
            self._register_field(field_type, d)

        # Initialize on first call to __getitem__
        self.fs = None
        self.mod_level_store = None
        self.pres_level_store = None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    # inheriting from BaseDataset

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _init_fs(self):
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

    # def _register_field(self, field_type: str, d: dict | None) -> None:
    #     """
    #     Validate and register one field type from the config variables block.

    #     Populates ``self.file_dict`` and ``self.var_dict`` for *field_type*.

    #     Args:
    #         field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
    #             ``"static"``, ``"diagnostic"``.
    #         d: Field-type config dict, or ``None`` / null to disable the field.

    #     Raises:
    #         KeyError: If *field_type* is not a recognised field type.
    #         ValueError: If *d* defines neither ``vars_3D`` nor ``vars_2D``.
    #     """
    #     if not isinstance(d, dict):
    #         return

    #     if field_type not in VALID_FIELD_TYPES:
    #         raise KeyError(
    #             f"Unknown field_type '{field_type}' in config['source']['ERA5']. "
    #             f"Valid options are: {sorted(VALID_FIELD_TYPES)}"
    #         )

    #     if not d.get("vars_3D") and not d.get("vars_2D"):
    #         raise ValueError(f"Field '{field_type}' must define at least one of vars_3D or vars_2D")

    #     self.file_dict[field_type] = True
    #     self.var_dict[field_type] = {
    #         "vars_3D": d.get("vars_3D") or [],
    #         "vars_2D": d.get("vars_2D") or [],
    #     }

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """
        Open the dataset for *field_type* at time *t* and populate *sample*.

        Keys written are ``"era5/{field_type}/3d/{varname}"`` for 3D variables
        and ``"era5/{field_type}/2d/{varname}"`` for 2D variables.

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
                        t_sel = self._to_cftime(t, calendar)
                    else:
                        t_sel = t
                    ds_t = ds.sel(time=t_sel)
                else:
                    ds_t = ds

                # 3D variables: (n_levels, lat, lon) → (n_levels, 1, lat, lon)
                for vname in vars_3D:
                    arr = ds_t[vname].sel({self.level_coord: self.levels}).values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                    sample[f"{self.dataset_name}/{field_type}/3d/{vname}"] = tensor

                # 2D variables: (lat, lon) → (1, 1, lat, lon)
                for vname in vars_2D:
                    arr = ds_t[vname].values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    sample[f"{self.dataset_name}/{field_type}/2d/{vname}"] = tensor
        else:
            with xr.open_zarr(self.mod_level_store, chunks=None) as ds:
                # Select the time step; static fields have no time dim
                if "time" in ds.dims:
                    if isinstance(ds.time.values[0], cftime.datetime):
                        calendar = ds.time.values[0].calendar
                        t_sel = self._to_cftime(t, calendar)
                    else:
                        t_sel = t
                    ds_t = ds.sel(time=t_sel)
                else:
                    ds_t = ds

                # 3D variables: (n_levels, lat, lon) → (n_levels, 1, lat, lon)
                for vname in vars_3D:
                    arr = ds_t[vname].sel({self.level_coord: self.levels}).values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                    sample[f"{self.dataset_name}/{field_type}/3d/{vname}"] = tensor

            with xr.open_zarr(self.pres_level_store, chunks=None) as ds:
                # Select the time step; static fields have no time dim
                if "time" in ds.dims:
                    if isinstance(ds.time.values[0], cftime.datetime):
                        calendar = ds.time.values[0].calendar
                        t_sel = self._to_cftime(t, calendar)
                    else:
                        t_sel = t
                    ds_t = ds.sel(time=t_sel)
                else:
                    ds_t = ds
                # 2D variables: (lat, lon) → (1, 1, lat, lon)
                for vname in vars_2D:
                    arr = ds_t[vname].values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    sample[f"{self.dataset_name}/{field_type}/2d/{vname}"] = tensor
