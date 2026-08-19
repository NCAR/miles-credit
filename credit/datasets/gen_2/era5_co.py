"""
era5_co.py
----------
ARCOERA5CODataset: PyTorch Dataset for streaming the *cloud-optimized* (``co/``)
ARCO ERA5 zarr stores — ERA5 in its native IFS representation:

- ``co/model-level-wind.zarr-v2``: temperature, vorticity, divergence,
  vertical velocity on 137 hybrid levels as **T639 spherical-harmonic
  coefficients** (flat ``values`` dim of 410,240 floats = 205,120 complex).
- ``co/single-level-surface.zarr-v2``: log surface pressure and surface
  geopotential as T639 spherical-harmonic coefficients.
- ``co/model-level-moisture.zarr-v2``: specific humidity, cloud species, ozone
  on 137 hybrid levels on the **N320 reduced Gaussian grid** (542,080 points).
- ``co/single-level-reanalysis.zarr-v2``: 2m/10m/100m fields, SST, soil
  variables, column totals on the N320 reduced Gaussian grid.

Because winds only exist as spectral vorticity/divergence, requesting
``u_component_of_wind`` / ``v_component_of_wind`` makes this dataset emit the
spectral ``vorticity`` and ``divergence`` keys instead; the ``spectral_to_grid``
preblock (``credit.preblock.spectral.SpectralToGrid``) then derives u/v and
synthesizes all other spectral variables to grid points, and a ``regrid``
preblock maps the reduced Gaussian points to a lat/lon grid (weight file built
with ``credit.reduced_gaussian``). Tensors are returned flat:

- spectral variable:      ``(n_levels, 1, 410240)`` packed coefficients
- reduced-Gaussian field: ``(n_levels, 1, 542080)`` ring-major N->S points

Variables are requested by their ARCO analysis-ready long names (matching the
``arco_era5`` source, e.g. ``temperature``, ``2m_temperature``) or by their
GRIB short names (``t``, ``t2m``). The full mapping is ``_CO_VARIABLES`` below.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cftime
import obstore.store as obs
import pandas as pd
import torch
import xarray as xr
import zarr

from credit.datasets.gen_2._utils import _to_cftime  # pyright: ignore[reportPrivateUsage]
from credit.datasets.gen_2.base_dataset import BaseDataset, VALID_FIELD_TYPES
from credit.datasets.gen_2.grid_utils import write_source_grid_schema_if_missing

logger = logging.getLogger(__name__)

_CO_BASE = "gs://gcp-public-data-arco-era5/co"

_CO_STORES: dict[str, str] = {
    "wind": f"{_CO_BASE}/model-level-wind.zarr-v2",
    "moisture": f"{_CO_BASE}/model-level-moisture.zarr-v2",
    "surface": f"{_CO_BASE}/single-level-surface.zarr-v2",
    "reanalysis": f"{_CO_BASE}/single-level-reanalysis.zarr-v2",
}

# long_name: (store, grib_short_name, representation)
#   representation: "spectral" (T639 coefficients) | "grid" (N320 reduced Gaussian)
_CO_VARIABLES: dict[str, tuple[str, str, str]] = {
    # --- 3D, hybrid model levels, spectral (wind store) ---
    "temperature": ("wind", "t", "spectral"),
    "vorticity": ("wind", "vo", "spectral"),
    "divergence": ("wind", "d", "spectral"),
    "vertical_velocity": ("wind", "w", "spectral"),
    # --- 3D, hybrid model levels, reduced Gaussian (moisture store) ---
    "specific_humidity": ("moisture", "q", "grid"),
    "ozone_mass_mixing_ratio": ("moisture", "o3", "grid"),
    "specific_cloud_liquid_water_content": ("moisture", "clwc", "grid"),
    "specific_cloud_ice_water_content": ("moisture", "ciwc", "grid"),
    "fraction_of_cloud_cover": ("moisture", "cc", "grid"),
    "specific_rain_water_content": ("moisture", "crwc", "grid"),
    "specific_snow_water_content": ("moisture", "cswc", "grid"),
    # --- 2D, spectral (surface store) ---
    "log_surface_pressure": ("surface", "lnsp", "spectral"),
    # --- 2D, reduced Gaussian (reanalysis store) ---
    "2m_temperature": ("reanalysis", "t2m", "grid"),
    "2m_dewpoint_temperature": ("reanalysis", "d2m", "grid"),
    "surface_pressure": ("reanalysis", "sp", "grid"),
    "mean_sea_level_pressure": ("reanalysis", "msl", "grid"),
    "sea_surface_temperature": ("reanalysis", "sst", "grid"),
    "skin_temperature": ("reanalysis", "skt", "grid"),
    "sea_ice_cover": ("reanalysis", "siconc", "grid"),
    "geopotential_at_surface": ("reanalysis", "z", "grid"),
    "temperature_of_snow_layer": ("reanalysis", "tsn", "grid"),
    "10m_u_component_of_wind": ("reanalysis", "u10", "grid"),
    "10m_v_component_of_wind": ("reanalysis", "v10", "grid"),
    "100m_u_component_of_wind": ("reanalysis", "u100", "grid"),
    "100m_v_component_of_wind": ("reanalysis", "v100", "grid"),
    "total_cloud_cover": ("reanalysis", "tcc", "grid"),
    "low_cloud_cover": ("reanalysis", "lcc", "grid"),
    "medium_cloud_cover": ("reanalysis", "mcc", "grid"),
    "high_cloud_cover": ("reanalysis", "hcc", "grid"),
    "total_column_water": ("reanalysis", "tcw", "grid"),
    "total_column_water_vapour": ("reanalysis", "tcwv", "grid"),
    "total_column_cloud_ice_water": ("reanalysis", "tciw", "grid"),
    "total_column_cloud_liquid_water": ("reanalysis", "tclw", "grid"),
    "total_column_rain_water": ("reanalysis", "tcrw", "grid"),
    "total_column_snow_water": ("reanalysis", "tcsw", "grid"),
    "convective_available_potential_energy": ("reanalysis", "cape", "grid"),
    "soil_temperature_level_1": ("reanalysis", "stl1", "grid"),
    "soil_temperature_level_2": ("reanalysis", "stl2", "grid"),
    "soil_temperature_level_3": ("reanalysis", "stl3", "grid"),
    "soil_temperature_level_4": ("reanalysis", "stl4", "grid"),
    "volumetric_soil_water_layer_1": ("reanalysis", "swvl1", "grid"),
    "volumetric_soil_water_layer_2": ("reanalysis", "swvl2", "grid"),
    "volumetric_soil_water_layer_3": ("reanalysis", "swvl3", "grid"),
    "volumetric_soil_water_layer_4": ("reanalysis", "swvl4", "grid"),
    "ice_temperature_layer_1": ("reanalysis", "istl1", "grid"),
    "ice_temperature_layer_2": ("reanalysis", "istl2", "grid"),
    "ice_temperature_layer_3": ("reanalysis", "istl3", "grid"),
    "ice_temperature_layer_4": ("reanalysis", "istl4", "grid"),
}
# GRIB short names as accepted aliases (e.g. "t2m" for "2m_temperature").
_CO_SHORT_ALIASES: dict[str, str] = {short: long for long, (_, short, _) in _CO_VARIABLES.items()}

# Winds must be derived from spectral vorticity/divergence by the
# spectral_to_grid preblock; the dataset emits those two keys instead.
_DERIVED_FROM_VO_D = ("u_component_of_wind", "v_component_of_wind")

# Filename of the ring-description NetCDF this dataset writes to save_loc —
# input for credit.reduced_gaussian.reduced_gaussian_to_latlon_bilinear_weights
# and for the spectral_to_grid preblock's grid_file argument.
REDUCED_GRID_FILENAME = "{source}_reduced_gaussian_grid.nc"


class ARCOERA5CODataset(BaseDataset):
    """PyTorch Dataset for the cloud-optimized (native-representation) ARCO ERA5 stores.

    See the module docstring for the store layout and the preblock chain this
    source requires (``spectral_to_grid`` + ``regrid``).

    Example YAML configuration::

        data:
          source:
            ERA5:
              dataset_type: "arco_era5_co"
              levels: [10, 30, 50, 70, 90, 100, 110, 120, 130, 137]
              variables:
                prognostic:
                  vars_3D: ["temperature", "u_component_of_wind",
                            "v_component_of_wind", "specific_humidity"]
                  vars_2D: ["surface_pressure", "2m_temperature"]
                static:
                  vars_2D: ["geopotential_at_surface"]
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "arco_era5_co", (
            f"Expected dataset_type 'arco_era5_co' in config for ARCOERA5CODataset, "
            f"got '{self.curr_source_cfg['dataset_type']}'"
        )
        self.dataset_type = "arco_era5_co"
        level_coord = self.curr_source_cfg.get("level_coord", "hybrid")
        if level_coord != "hybrid":
            raise ValueError(
                "arco_era5_co only provides hybrid model levels (the co/ stores are the "
                f"native IFS representation); got level_coord='{level_coord}'. "
                "For pressure-level data use dataset_type 'arco_era5'."
            )
        self.level_coord = "hybrid"
        self.levels: list[int] = self.curr_source_cfg.get("levels", list(range(1, 138)))
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }
        self.mode = "remote"

        self.init_register_all_fields()
        # Validate every requested variable up front and precompute per-field-type
        # fetch plans: list of (emitted_varname, store, short_name, representation).
        self._fetch_plans: dict[str, dict[str, list[tuple[str, str, str, str]]]] = {}
        for field_type, vd in self.var_dict.items():
            self._fetch_plans[field_type] = {
                "3d": self._build_fetch_plan(field_type, vd["vars_3D"], "3D"),
                "2d": self._build_fetch_plan(field_type, vd["vars_2D"], "2D"),
            }

        self._fs = None
        self._stores: dict[str, zarr.storage.ObjectStore] = {}

    # ------------------------------------------------------------------
    # Variable resolution
    # ------------------------------------------------------------------

    def _build_fetch_plan(self, field_type: str, requested: list[str], dim_label: str) -> list:
        plan: list[tuple[str, str, str, str]] = []
        seen: set[str] = set()

        def _add(varname: str) -> None:
            if varname in seen:
                return
            seen.add(varname)
            store, short, repr_ = _CO_VARIABLES[varname]
            plan.append((varname, store, short, repr_))

        for vname in requested:
            resolved = _CO_SHORT_ALIASES.get(vname, vname)
            if resolved in _DERIVED_FROM_VO_D:
                # u/v do not exist in the co/ stores: emit spectral vorticity
                # and divergence (once) for the spectral_to_grid preblock.
                _add("vorticity")
                _add("divergence")
            elif resolved in _CO_VARIABLES:
                _add(resolved)
            else:
                raise KeyError(
                    f"arco_era5_co: unknown variable '{vname}' (field type '{field_type}', {dim_label}). "
                    f"Supported names: {sorted(_CO_VARIABLES) + list(_DERIVED_FROM_VO_D)} "
                    f"or GRIB short names {sorted(_CO_SHORT_ALIASES)}."
                )
        return plan

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def _init_fs(self) -> None:
        """Initialize the obstore GCS stores (anonymous access) on first use."""
        for name, url in _CO_STORES.items():
            # skip_signature -> anonymous access to the public ARCO ERA5 bucket; without it
            # obstore tries to fetch a token from the GCP metadata server (fails off-GCP, e.g. CI).
            store = obs.from_url(url, config={"skip_signature": True})
            self._stores[name] = zarr.storage.ObjectStore(store, read_only=True)
        self._fs = True  # marker: stores initialized

    def _cache_grid(self, ds: xr.Dataset) -> None:
        """Cache the native N320 reduced Gaussian grid (per-point coords) once,
        and persist the ring description used to build regrid weight files."""
        try:
            lat = ds["latitude"].values.astype(float)
            lon = ds["longitude"].values.astype(float)
            # Per-point coords on a reduced Gaussian grid are unstructured as far
            # as GridSchema is concerned (a Regridder preblock defines the output
            # grid); don't let infer_grid_type misread two 1D vectors as rectilinear.
            grid = {"grid_type": "unstructured", "lat": lat, "lon": lon}
            self.static_metadata["grid"] = grid
            write_source_grid_schema_if_missing(self.curr_source_name, grid, self.save_loc)
            self._write_reduced_grid_file(lat, lon)
        except Exception as exc:
            logger.warning("%s '%s': could not find grid (%s).", type(self).__name__, self.curr_source_name, exc)
            self.static_metadata["grid"] = None

    def _write_reduced_grid_file(self, lat, lon) -> None:
        """Best-effort persist the ring description to save_loc (skipped if present)."""
        if not self.save_loc:
            return
        path = os.path.join(
            os.path.expandvars(self.save_loc), REDUCED_GRID_FILENAME.format(source=self.curr_source_name)
        )
        if os.path.isfile(path):
            return
        try:
            from credit.reduced_gaussian import ReducedGaussianGrid

            ReducedGaussianGrid.from_points(lat, lon).save(path)
        except Exception as exc:
            logger.warning("Could not write reduced Gaussian grid file to %s (%s).", path, exc)

    def _extract_field(
        self,
        field_type: VALID_FIELD_TYPES,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """Open the co/ stores needed for *field_type* at time *t* and populate *sample*.

        Tensor shapes (no batch dimension): 3D ``(n_levels, 1, values)``,
        2D ``(1, 1, values)`` — ``values`` is 410,240 (spectral) or 542,080
        (reduced Gaussian).
        """
        if self._fs is None:
            self._init_fs()

        plans = self._fetch_plans[field_type]
        needed_stores = {store for dim in ("3d", "2d") for (_, store, _, _) in plans[dim]}
        for store_name in sorted(needed_stores):
            with xr.open_zarr(self._stores[store_name], chunks=None) as ds:
                if "grid" not in self.static_metadata and store_name in ("moisture", "reanalysis"):
                    self._cache_grid(ds)
                if "time" in ds.dims:
                    if isinstance(ds.time.values[0], cftime.datetime):
                        t_sel = _to_cftime(t, ds.time.values[0].calendar)
                    else:
                        t_sel = t
                    ds_t = ds.sel(time=t_sel)
                else:
                    ds_t = ds

                for varname, store, short, _repr in plans["3d"]:
                    if store != store_name:
                        continue
                    arr = ds_t[short].sel({self.level_coord: self.levels}).values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)  # (n_levels, 1, values)
                    sample[self._get_field_name(field_type, "3d", varname)] = tensor

                for varname, store, short, _repr in plans["2d"]:
                    if store != store_name:
                        continue
                    arr = ds_t[short].values
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, values)
                    sample[self._get_field_name(field_type, "2d", varname)] = tensor

        # The reduced-Gaussian grid description is also needed when only spectral
        # variables were requested (the spectral_to_grid preblock synthesizes onto
        # it): fall back to reading the coords from the reanalysis store once.
        if "grid" not in self.static_metadata:
            with xr.open_zarr(self._stores["reanalysis"], chunks=None) as ds:
                self._cache_grid(ds)
