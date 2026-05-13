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
from credit.datasets.base_dataset import BaseDataset, VALID_FIELD_TYPES


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
        field_type: VALID_FIELD_TYPES,
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

    See the module docstring for a full description of the output format and file naming.

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
        field_type: VALID_FIELD_TYPES,
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


_WB2_ERA5_BASE = "gs://weatherbench2/datasets/era5"

_WB2_ERA5_STORE_PATHS: dict[str, str] = {
    "1440x721": f"{_WB2_ERA5_BASE}/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    "240x121": f"{_WB2_ERA5_BASE}/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
    "64x32": f"{_WB2_ERA5_BASE}/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
    "full": f"{_WB2_ERA5_BASE}/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
}

# Default pressure levels (hPa) available in each store.
# "1440x721", "240x121", and "64x32" carry the 13 WeatherBench2 pressure levels.
# "full" carries the standard ERA5 37 pressure levels.
_WB2_ERA5_DEFAULT_LEVELS: dict[str, list[int]] = {
    "1440x721": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    "240x121": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    "64x32": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    "full": [
        1,
        2,
        3,
        5,
        7,
        10,
        20,
        30,
        50,
        70,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        775,
        800,
        825,
        850,
        875,
        900,
        925,
        950,
        975,
        1000,
    ],
}


class WeatherBench2ERA5Dataset(BaseDataset):
    """PyTorch Dataset for WeatherBench2 ERA5 data on Google Cloud Storage.

    Provides access to ERA5 reanalysis data prepared for the WeatherBench2
    benchmark at multiple resolutions. All data is read lazily from public
    Google Cloud Storage zarr stores (anonymous access, no credentials required).

    Available resolutions:

    +--------------+-------------------+------------+------------------+
    | ``resolution`` | Grid              | Approx deg | Timestep         |
    +==============+===================+============+==================+
    | ``"1440x721"`` | 1440 × 721 global | 0.25°      | 6-hourly, 13 lev |
    +--------------+-------------------+------------+------------------+
    | ``"240x121"``  | 240 × 121 global  | 1.5°       | 6-hourly, 13 lev |
    +--------------+-------------------+------------+------------------+
    | ``"64x32"``    | 64 × 32 global    | ~5.6°      | 6-hourly, 13 lev |
    +--------------+-------------------+------------+------------------+
    | ``"full"``     | 1440 × 721 global | 0.25°      | hourly, 37 lev   |
    +--------------+-------------------+------------+------------------+

    See ``_WB2_ERA5_DEFAULT_LEVELS`` for default pressure levels per resolution.

    Example YAML configuration::

        data:
          source:
            WeatherBench2_ERA5:
              dataset_type: "weatherbench2_era5"
              resolution: "1440x721"   # optional; overridden by the resolution kwarg
              level_coord: "level"
              levels: [50, 100, 200, 500, 850, 1000]  # optional; defaults to all available
              variables:
                prognostic:
                  vars_3D: ["temperature", "u_component_of_wind", "v_component_of_wind",
                             "specific_humidity"]
                  vars_2D: ["surface_pressure", "2m_temperature"]
                dynamic_forcing:
                  vars_2D: ["total_precipitation_6hr"]
                static:
                  vars_2D: ["geopotential_at_surface"]
                diagnostic: null

          start_datetime: "2017-01-01"
          end_datetime:   "2019-12-31"
          timestep: "6h"
          forecast_len: 1

    Output key format::

        "weatherbench2_era5/{field_type}/{dim}/{varname}"

    Assumptions:
        1. Non-static variables have a "time" dimension in the zarr store.
        2. 3D pressure-level variables have a "level" coordinate (hPa).
        3. Dimension order: (time, level, latitude, longitude) for 3D;
           (time, latitude, longitude) for 2D; (latitude, longitude) for static.
    """

    def __init__(
        self,
        data_config: dict,
        resolution: str = "1440x721",
        return_target: bool = False,
    ) -> None:
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "weatherbench2_era5", (
            f"Expected dataset_type 'weatherbench2_era5' in config for ARCOERA5Dataset, got {self.curr_source_cfg['dataset_type']}"
        )
        self.dataset_type: str = "weatherbench2_era5"
        # Config key takes precedence over the kwarg default.
        self.resolution: str = self.curr_source_cfg.get("resolution", resolution)
        if self.resolution not in _WB2_ERA5_STORE_PATHS:
            raise ValueError(f"Invalid resolution '{self.resolution}'. Valid options: {sorted(_WB2_ERA5_STORE_PATHS)}")

        self.store_path: str = _WB2_ERA5_STORE_PATHS[self.resolution]
        self.level_coord: str = self.curr_source_cfg.get("level_coord", "level")
        self.levels: list[int] = self.curr_source_cfg.get("levels") or _WB2_ERA5_DEFAULT_LEVELS[self.resolution]
        self.static_metadata: dict = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }
        # Initialised lazily on the first __getitem__ call (worker-safe).
        self._fs = None
        self.store = None
        super().init_register_all_fields()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_fs(self) -> None:
        fs_config = {
            "cache_timeout": -1,
            "token": "anon",  # noqa: S106 # nosec B106
            "access": "read_only",
            "block_size": 8**20,
            "asynchronous": True,
            "skip_instance_cache": True,
        }
        self._fs = GCSFileSystem(**fs_config)
        self.store = zarr.storage.FsspecStore(fs=self._fs, path=self.store_path)

    def _build_timestamps(self) -> pd.DatetimeIndex:
        """Return valid initialisation timestamps for the dataset.

        Returns:
            DatetimeIndex from ``start_datetime`` to ``end_datetime`` minus
            the forecast horizon, at the configured timestep frequency.
        """
        return pd.date_range(
            self.start_datetime,
            self.end_datetime - self.num_forecast_steps * self.dt,
            freq=self.dt,
        )

    def _extract_field(
        self,
        field_type: VALID_FIELD_TYPES,
        t: pd.Timestamp,
        sample: dict,
    ) -> None:
        """Open the zarr store and extract variables for *field_type* at time *t*.

        Keys written to *sample*:

        - ``"weatherbench2_era5/{field_type}/3d/{varname}"`` — shape ``(n_levels, 1, lat, lon)``
        - ``"weatherbench2_era5/{field_type}/2d/{varname}"`` — shape ``(1, 1, lat, lon)``

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t: Timestamp to select.
            sample: Dict to write variable tensors into (modified in place).
        """
        if self._fs is None:
            self._init_fs()
        if field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        with xr.open_zarr(self.store, chunks=None) as ds:
            if "time" in ds.dims:
                if isinstance(ds.time.values[0], cftime.datetime):
                    calendar = ds.time.values[0].calendar
                    t_sel = _to_cftime(t, calendar)
                else:
                    t_sel = t
                ds_t = ds.sel(time=t_sel)
            else:
                ds_t = ds

            for vname in vars_3D:
                arr = ds_t[vname].sel({self.level_coord: self.levels}).values
                tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                key = self._get_field_name(field_type, "3d", vname)
                sample[key] = tensor

            for vname in vars_2D:
                arr = ds_t[vname].values
                tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                key = self._get_field_name(field_type, "2d", vname)
                sample[key] = tensor
