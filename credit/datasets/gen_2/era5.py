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

Output key format (flat, slash-delimited)::

    "{source_name}/{field_type}/{dim}/{varname}"

    field_type: "prognostic" | "dynamic_forcing" | "static" | "diagnostic"
    dim       : "2d"  (surface / single-level)
                "3d"  (multi-level; level_coord = "level" or "hybrid")
    varname   : variable name as in the ARCO ERA5 Zarr store
"""

from __future__ import annotations

import cftime
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import obstore.store as obs
import pandas as pd
import torch
import xarray as xr
import zarr

from credit.datasets.gen_2._utils import _to_cftime  # pyright: ignore[reportPrivateUsage]
from credit.datasets.gen_2.base_dataset import BaseDataset, VALID_FIELD_TYPES
from credit.datasets.gen_2.grid_utils import find_coord_pair, infer_grid_type, write_source_grid_schema_if_missing

logger = logging.getLogger(__name__)


class _ObjectStoreZarrMixin:
    """Per-process zarr store caching and concurrent variable reads for obstore-backed sources.

    Remote ERA5 reads are network-bound, and opening a large store re-reads its metadata
    and decodes its time coordinate every time. Mixed into a ``BaseDataset`` subclass,
    this opens each store once per process and fetches a field's variables concurrently
    on a pool of ``io_threads`` threads (source config key, default 16; 1 = sequential).

    Subclasses implement ``_init_fs`` (which sets the attributes named in ``_STORE_ATTRS``)
    and ``_cache_grid``, call ``_init_io_state`` in ``__init__``, and read through
    ``_open_cached`` / ``_select_time`` / ``_read_arrays``.
    """

    _STORE_ATTRS: tuple[str, ...] = ()

    def _init_io_state(self) -> None:
        self.io_threads: int = int(self.curr_source_cfg.get("io_threads", 16))
        if self.io_threads < 1:
            raise ValueError(f"io_threads must be >= 1 for source '{self.curr_source_name}', got {self.io_threads}")
        # Per-process I/O handles, created lazily on the first read (see _ensure_handles).
        self._fs = None
        self._handles_pid: int | None = None
        self._ds_cache: dict[str, xr.Dataset] = {}
        self._executor: ThreadPoolExecutor | None = None
        for attr in self._STORE_ATTRS:
            setattr(self, attr, None)

    def __getstate__(self) -> dict[str, Any]:
        """Drop open stores, datasets, and the thread pool so each unpickled copy (e.g. a spawned DataLoader worker) opens its own."""
        state = self.__dict__.copy()
        state.update(_fs=None, _handles_pid=None, _ds_cache={}, _executor=None)
        state.update(dict.fromkeys(self._STORE_ATTRS))
        return state

    def _ensure_handles(self) -> None:
        """Open the object stores on this process's first read; fail fast in a forked child.

        obstore's async runtime is process-global and doesn't survive ``fork``: once the
        parent has read from any obstore store, every obstore call in a forked child hangs
        forever, even on a freshly created store. Pickled copies (``spawn`` workers, which
        the gen2 trainer uses) reset ``_handles_pid`` via ``__getstate__`` and are fine.
        A copy inherited through ``fork`` after the parent already read is not, so it
        raises instead of deadlocking.
        """
        if self._handles_pid is None:
            self._init_fs()
            self._handles_pid = os.getpid()
        elif self._handles_pid != os.getpid():
            raise RuntimeError(
                f"{type(self).__name__} '{self.curr_source_name}' was read in process {self._handles_pid} and then "
                f"forked into process {os.getpid()}. obstore cannot be used after fork once the parent has "
                "read from it (reads hang). Use DataLoader(multiprocessing_context='spawn'), or avoid "
                "reading from the dataset in the parent before starting workers."
            )

    def _open_cached(self, name: str, store_attr: str) -> xr.Dataset:
        """Return the dataset for the store in attribute *store_attr*, opened once per process under key *name*."""
        self._ensure_handles()
        ds = self._ds_cache.get(name)
        if ds is None:
            ds = xr.open_zarr(getattr(self, store_attr), chunks=None)
            if "grid" not in self.static_metadata:
                self._cache_grid(ds)
            self._ds_cache[name] = ds
        return ds

    @staticmethod
    def _select_time(ds: xr.Dataset, t: pd.Timestamp) -> xr.Dataset:
        """Select time step *t* (lazily); datasets without a time dim are returned unchanged."""
        if "time" not in ds.dims:
            return ds
        if isinstance(ds.time.values[0], cftime.datetime):
            return ds.sel(time=_to_cftime(t, ds.time.values[0].calendar))
        return ds.sel(time=t)

    def _read_arrays(self, requests: list[tuple[str, xr.DataArray]]) -> list[tuple[str, np.ndarray]]:
        """Load each lazy DataArray in *requests*, concurrently when ``io_threads > 1``.

        Results come back in request order, which matters because concat preserves
        insertion order within each (field_type, dim) channel bucket.
        """
        if self.io_threads == 1 or len(requests) <= 1:
            return [(key, da.values) for key, da in requests]
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.io_threads, thread_name_prefix="era5-io")
        futures = [(key, self._executor.submit(lambda d: d.values, da)) for key, da in requests]
        return [(key, fut.result()) for key, fut in futures]


class ARCOERA5Dataset(_ObjectStoreZarrMixin, BaseDataset):
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
              io_threads: 16  # optional: concurrent variable reads per field (1 = sequential)

          start_datetime: "2017-01-01"
          end_datetime: "2019-12-31"
          timestep: "6h"
          forecast_len: 1

    Performance:
        Reads are network-bound (each 3D variable is one ~95 MB chunk holding all 37
        pressure levels), so every variable of a field type at one time step is fetched
        concurrently on a thread pool of ``io_threads`` workers. The zarr stores are opened
        once per process and reused across samples. The open handles and the pool are
        dropped on pickling, so each ``spawn`` DataLoader worker opens its own. ``fork``
        workers are unsupported once the parent has read (obstore hangs after fork), and
        they raise a RuntimeError instead.

    Assumptions:
        1. A "time" dimension / coordinate is present for non-static fields.
        2. A level coordinate (name given by ``level_coord``) represents the
           vertical axis of 3D variables.
        3. Dimension order: (time, level, latitude, longitude) for 3D;
           (time, latitude, longitude) for 2D; (latitude, longitude) for static.
    """

    _STORE_ATTRS = ("pres_level_store", "mod_level_store")

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
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }
        self.mode = "remote"
        self._init_io_state()

        # Initialize the field registration based on the provided config and populate
        #   dictionary of variables and file paths for each field type
        self.init_register_all_fields()

    def _get_ds(self, level_type: str) -> xr.Dataset:
        """Return the pressure-level (``"pres"``) or model-level (``"mod"``) dataset, opened once per process.

        Opening the ARCO store (273 variables, ~1.3M hourly timestamps) costs ~0.35 s of
        metadata reads and time decoding, so it is cached instead of reopened per read.
        """
        return self._open_cached(level_type, "pres_level_store" if level_type == "pres" else "mod_level_store")

    def _init_fs(self):
        """Initialize the obstore GCS stores and zarr stores for pressure-level and model-level ERA5 data."""
        # skip_signature -> anonymous access to the public ARCO ERA5 bucket; without it
        # obstore tries to fetch a token from the GCP metadata server (fails off-GCP, e.g. CI).
        pres_obs = obs.from_url(self.pressure_lev_era5_path, config={"skip_signature": True})
        mod_obs = obs.from_url(self.model_lev_era5_path, config={"skip_signature": True})
        self._fs = True  # marker: stores initialized
        self.pres_level_store = zarr.storage.ObjectStore(pres_obs, read_only=True)
        self.mod_level_store = zarr.storage.ObjectStore(mod_obs, read_only=True)

    def _cache_grid(self, ds: xr.Dataset) -> None:
        """Cache this source's native grid, once — call only when not yet cached.

        Debugging aid (``self.static_metadata["grid"]``); not necessarily the
        grid actually written to output — see
        ``credit.datasets.gen_2.grid_utils.GridSchema``.
        """
        try:
            lon, lat, _, _ = find_coord_pair(ds)
            grid = {"grid_type": infer_grid_type(lat, lon), "lat": lat, "lon": lon}
            self.static_metadata["grid"] = grid
            write_source_grid_schema_if_missing(self.curr_source_name, grid, self.save_loc)
        except Exception as exc:
            logger.warning("%s '%s': could not find grid (%s).", type(self).__name__, self.curr_source_name, exc)
            self.static_metadata["grid"] = None

    def _extract_field(
        self,
        field_type: VALID_FIELD_TYPES,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """
        Read every variable of *field_type* at time *t* (concurrently) and populate *sample*.

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
        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        # Build lazy selections first, then fetch them all concurrently. Model-level runs
        # take 3D variables from the model-level store and 2D ones from the pressure-level store.
        store_3d = "pres" if self.level_coord == "level" else "mod"
        requests: list[tuple[str, xr.DataArray]] = []
        if vars_3D:
            ds_t = self._select_time(self._get_ds(store_3d), t)
            for vname in vars_3D:
                key = self._get_field_name(field_type, "3d", vname)
                requests.append((key, ds_t[vname].sel({self.level_coord: self.levels})))
        n_3d = len(requests)
        if vars_2D:
            ds_t = self._select_time(self._get_ds("pres"), t)
            for vname in vars_2D:
                requests.append((self._get_field_name(field_type, "2d", vname), ds_t[vname]))

        for i, (key, arr) in enumerate(self._read_arrays(requests)):
            tensor = torch.tensor(arr, dtype=torch.float32)
            # 3D: (n_levels, lat, lon) -> (n_levels, 1, lat, lon); 2D: (lat, lon) -> (1, 1, lat, lon)
            sample[key] = tensor.unsqueeze(1) if i < n_3d else tensor.unsqueeze(0).unsqueeze(0)


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


class WeatherBench2ERA5Dataset(_ObjectStoreZarrMixin, BaseDataset):
    """PyTorch Dataset for WeatherBench2 ERA5 data on Google Cloud Storage.

    Provides access to ERA5 reanalysis data prepared for the WeatherBench2
    benchmark at multiple resolutions. All data is read lazily from public
    Google Cloud Storage zarr stores (anonymous access, no credentials required).

    Available resolutions::

        +----------------+-------------------+------------+------------------+
        | ``resolution`` | Grid              | Approx deg | Timestep         |
        +================+===================+============+==================+
        | ``"1440x721"`` | 1440 × 721 global | 0.25°      | 6-hourly, 13 lev |
        +----------------+-------------------+------------+------------------+
        | ``"240x121"``  | 240 × 121 global  | 1.5°       | 6-hourly, 13 lev |
        +----------------+-------------------+------------+------------------+
        | ``"64x32"``    | 64 × 32 global    | ~5.6°      | 6-hourly, 13 lev |
        +----------------+-------------------+------------+------------------+
        | ``"full"``     | 1440 × 721 global | 0.25°      | hourly, 37 lev   |
        +----------------+-------------------+------------+------------------+

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
              io_threads: 16  # optional: concurrent variable reads per field (1 = sequential)

          start_datetime: "2017-01-01"
          end_datetime:   "2019-12-31"
          timestep: "6h"
          forecast_len: 1

    Output key format::

        "weatherbench2_era5/{field_type}/{dim}/{varname}"

    Performance:
        The store is opened once per process and reused across samples, and every variable
        of a field type at one time step is fetched concurrently on ``io_threads`` threads.
        Both matter most for the 0.25° stores. As with ``ARCOERA5Dataset``, use ``spawn``
        DataLoader workers: a ``fork`` after the parent has read raises a RuntimeError.

    Assumptions:
        1. Non-static variables have a "time" dimension in the zarr store.
        2. 3D pressure-level variables have a "level" coordinate (hPa).
        3. Dimension order: (time, level, latitude, longitude) for 3D;
           (time, latitude, longitude) for 2D; (latitude, longitude) for static.
    """

    _STORE_ATTRS = ("store",)

    def __init__(
        self,
        data_config: dict,
        return_target: bool = False,
        resolution: str = "1440x721",
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
        self.mode = "remote"
        # Store handles are opened lazily on the first read, once per process.
        self._init_io_state()
        super().init_register_all_fields()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_fs(self) -> None:
        # skip_signature -> anonymous access to the public WeatherBench2 bucket; without it
        # obstore tries to fetch a token from the GCP metadata server (fails off-GCP, e.g. CI).
        obs_store = obs.from_url(self.store_path, config={"skip_signature": True})
        self._fs = True  # marker: store initialized
        self.store = zarr.storage.ObjectStore(obs_store, read_only=True)

    def _cache_grid(self, ds: xr.Dataset) -> None:
        """Cache this source's native grid, once — call only when not yet cached.

        Debugging aid (``self.static_metadata["grid"]``); not necessarily the
        grid actually written to output — see
        ``credit.datasets.gen_2.grid_utils.GridSchema``.
        """
        try:
            lon, lat, _, _ = find_coord_pair(ds)
            grid = {"grid_type": infer_grid_type(lat, lon), "lat": lat, "lon": lon}
            self.static_metadata["grid"] = grid
            write_source_grid_schema_if_missing(self.curr_source_name, grid, self.save_loc)
        except Exception as exc:
            logger.warning("%s '%s': could not find grid (%s).", type(self).__name__, self.curr_source_name, exc)
            self.static_metadata["grid"] = None

    def _extract_field(
        self,
        field_type: VALID_FIELD_TYPES,
        t: pd.Timestamp,
        sample: dict,
    ) -> None:
        """Read every variable of *field_type* at time *t* (concurrently) from the cached store.

        Keys written to *sample*:

        - ``"weatherbench2_era5/{field_type}/3d/{varname}"`` — shape ``(n_levels, 1, lat, lon)``
        - ``"weatherbench2_era5/{field_type}/2d/{varname}"`` — shape ``(1, 1, lat, lon)``

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t: Timestamp to select.
            sample: Dict to write variable tensors into (modified in place).
        """
        if field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        ds_t = self._select_time(self._open_cached("store", "store"), t)
        # WeatherBench2 stores spatial dims as (longitude, latitude); transpose
        # to (latitude, longitude) to match the CREDIT (lat, lon) convention.
        requests: list[tuple[str, xr.DataArray]] = [
            (
                self._get_field_name(field_type, "3d", vname),
                ds_t[vname].sel({self.level_coord: self.levels}).transpose(..., "latitude", "longitude"),
            )
            for vname in vars_3D
        ]
        n_3d = len(requests)
        requests += [
            (self._get_field_name(field_type, "2d", vname), ds_t[vname].transpose(..., "latitude", "longitude"))
            for vname in vars_2D
        ]

        for i, (key, arr) in enumerate(self._read_arrays(requests)):
            tensor = torch.tensor(arr, dtype=torch.float32)
            # 3D: (n_levels, lat, lon) -> (n_levels, 1, lat, lon); 2D: (lat, lon) -> (1, 1, lat, lon)
            sample[key] = tensor.unsqueeze(1) if i < n_3d else tensor.unsqueeze(0).unsqueeze(0)
