"""
gfs.py
------
GFSDataset: PyTorch Dataset for GFS GDAS analysis files.

Streams lazily from Google Cloud Storage (archives 2021-present) or NOAA
NOMADS (rolling ~10-day window). One analysis file per 6-hourly cycle; no
forecast lead-time data is used.

No spatial regridding or vertical interpolation is performed here:
  - Spatial regridding (0.25° GFS → model grid): use the Regridder preblock
    (ic_only phase) with a pre-computed ESMF weight file.
  - Vertical interpolation (GFS hybrid-sigma → model levels): use a future
    vertical-interpolation preblock.

GFS GDAS splits upper-air and surface data across two files per cycle:
  - ``gdas.t{HH}z.atmf000.nc``  — 3D model-level fields (tmp, spfh, ugrd, vgrd, …)
  - ``gdas.t{HH}z.sfcf000.nc``  — 2D surface fields (pressfc, tmp2m, ugrd10m, …)

``GFSDataset`` routes vars_3D to the ATM file and vars_2D to the SFC file.
Variable names in the output follow ERA5 naming conventions via a mapping YAML
from ``credit/metadata/gfs_to_{mapping}.yml``.

At step index ``i > 0`` only ``dynamic_forcing`` variables are loaded (standard
BaseDataset behaviour). Since GFS GDAS provides single-timestep analyses,
``dynamic_forcing`` is typically left null and provided by a separate ERA5 source
in a MultiSourceDataset.

Example config::

    data:
      source:
        GFS:
          dataset_type: "gfs"
          gdas_base: "auto"               # "auto" | "gcs" | "nomads" | explicit base URL
          variable_mapping: "arcoera5"    # suffix for gfs_to_{mapping}.yml
          level_coord: "pfull"            # level-dimension name in ATM files
          levels: null                    # null = all levels; list = pfull index subset
          variables:
            prognostic:
              vars_3D: [u_component_of_wind, v_component_of_wind,
                        temperature, specific_total_water]
              vars_2D: [SP, VAR_2T, VAR_10U, VAR_10V]
            static: null
            dynamic_forcing: null
            diagnostic: null

      start_datetime: "2021-01-01"
      end_datetime:   "2024-12-31"
      timestep:       "6h"
      forecast_len:   1

Output key format:  ``"{source_name}/{field_type}/{dim}/{varname}"``
Tensor shapes (no batch dimension):
    3D variable: ``(n_levels, 1, lat, lon)``
    2D variable: ``(1,        1, lat, lon)``
"""

from __future__ import annotations

import logging
import os
from importlib.resources import files as pkg_files
from typing import Any

import pandas as pd
import torch
import xarray as xr
import yaml

from credit.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# Age threshold beyond which GCS archive is preferred over NOMADS
_NOMADS_WINDOW = pd.Timedelta(days=10)
_GCS_BASE = "gs://global-forecast-system/"
_NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"


class GFSDataset(BaseDataset):
    """PyTorch Dataset for GFS GDAS analysis files from GCS or NOMADS.

    See module docstring for full description, output format, and example config.
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        super().__init__(data_config, return_target)

        assert self.curr_source_cfg["dataset_type"].lower() == "gfs", (
            f"Expected dataset_type 'gfs', got '{self.curr_source_cfg['dataset_type']}'"
        )
        self.dataset_type = "gfs"
        self.mode = "remote"

        self._gdas_base: str = self.curr_source_cfg.get("gdas_base", "auto")
        self._level_coord: str = self.curr_source_cfg.get("level_coord", "pfull")
        self._levels: list | None = self.curr_source_cfg.get("levels")

        mapping_key = self.curr_source_cfg.get("variable_mapping", "arcoera5")
        self._era5_to_gfs: dict[str, str] = self._load_variable_mapping(mapping_key)

        self.static_metadata: dict[str, Any] = {
            "levels": self._levels,
            "datetime_fmt": "unix_ns",
        }

        self.init_register_all_fields()

    # ------------------------------------------------------------------
    # BaseDataset overrides
    # ------------------------------------------------------------------

    def _get_file_source(self, field_config: dict[str, Any]) -> bool:
        """GFS data is always remote — return True for any configured field."""
        return True

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """Fetch GFS variables for field_type at timestamp t and write into sample.

        3D variables (vars_3D) are read from the ATM file (upper-air, model levels).
        2D variables (vars_2D) are read from the SFC file (surface).

        Tensor shapes written into sample:
            3D: ``(n_levels, 1, lat, lon)``
            2D: ``(1,        1, lat, lon)``
        """
        if field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd.get("vars_3D", [])
        vars_2D: list[str] = vd.get("vars_2D", [])
        base = self._resolve_gdas_base(t)

        # ── 3D variables from ATM file ──────────────────────────────────────
        if vars_3D:
            pairs_3d = [(v, self._era5_to_gfs[v]) for v in vars_3D if v in self._era5_to_gfs]
            unmapped = [v for v in vars_3D if v not in self._era5_to_gfs]
            if unmapped:
                logger.warning("GFSDataset: no GFS mapping for 3D variables %s — skipping", unmapped)

            if pairs_3d:
                gfs_names = [gfs for _, gfs in pairs_3d]
                atm_url = _build_file_url(t, base, file_type="atm")
                ds = _open_gfs_file(atm_url, gfs_names)

                if self._levels is not None and self._level_coord in ds.dims:
                    ds = ds.sel({self._level_coord: self._levels})

                # Persist discovered level values so static_metadata stays current
                if self._level_coord in ds.coords:
                    self.static_metadata["levels"] = ds[self._level_coord].values.tolist()

                for era5_name, gfs_name in pairs_3d:
                    if gfs_name not in ds:
                        logger.warning("GFSDataset: '%s' not found in ATM file at %s", gfs_name, t)
                        continue
                    arr = ds[gfs_name].values
                    if arr.ndim == 4:  # (time, level, lat, lon)
                        arr = arr[0]
                    # arr shape: (n_levels, lat, lon) → (n_levels, 1, lat, lon)
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                    sample[self._get_field_name(field_type, "3d", era5_name)] = tensor

        # ── 2D variables from SFC file ──────────────────────────────────────
        if vars_2D:
            pairs_2d = [(v, self._era5_to_gfs[v]) for v in vars_2D if v in self._era5_to_gfs]
            unmapped = [v for v in vars_2D if v not in self._era5_to_gfs]
            if unmapped:
                logger.warning("GFSDataset: no GFS mapping for 2D variables %s — skipping", unmapped)

            if pairs_2d:
                gfs_names = [gfs for _, gfs in pairs_2d]
                sfc_url = _build_file_url(t, base, file_type="sfc")
                ds = _open_gfs_file(sfc_url, gfs_names)

                for era5_name, gfs_name in pairs_2d:
                    if gfs_name not in ds:
                        logger.warning("GFSDataset: '%s' not found in SFC file at %s", gfs_name, t)
                        continue
                    arr = ds[gfs_name].values
                    if arr.ndim == 3:  # (time, lat, lon)
                        arr = arr[0]
                    # arr shape: (lat, lon) → (1, 1, lat, lon)
                    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    sample[self._get_field_name(field_type, "2d", era5_name)] = tensor

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_gdas_base(self, t: pd.Timestamp) -> str:
        """Return the GCS or NOMADS base URL for timestamp t.

        "auto"  — use GCS if the timestamp is older than _NOMADS_WINDOW, else NOMADS
        "gcs"   — always use GCS archive
        "nomads"— always use NOMADS
        other   — treat as explicit base URL
        """
        cfg = (self._gdas_base or "auto").lower()
        if cfg == "auto":
            age = pd.Timestamp.utcnow().tz_localize(None) - t
            return _GCS_BASE if age >= _NOMADS_WINDOW else _NOMADS_BASE
        if cfg in ("gcs", "gcloud"):
            return _GCS_BASE
        if cfg == "nomads":
            return _NOMADS_BASE
        return self._gdas_base  # explicit base URL

    @staticmethod
    def _load_variable_mapping(mapping_key: str) -> dict[str, str]:
        """Load and invert GFS→ERA5 variable mapping from credit/metadata.

        Returns era5_name → gfs_native_name dict for use in _extract_field.

        Raises:
            FileNotFoundError: if credit/metadata/gfs_to_{mapping_key}.yml does not exist.
        """
        meta_path = str(pkg_files("credit.metadata"))
        mapping_file = os.path.join(meta_path, f"gfs_to_{mapping_key}.yml")
        if not os.path.isfile(mapping_file):
            raise FileNotFoundError(
                f"GFSDataset: no mapping file at '{mapping_file}'. Available keys: arcoera5, wchapmanera5, cam"
            )
        with open(mapping_file) as f:
            mapping = yaml.safe_load(f)
        gfs_to_era5: dict[str, str] = mapping.get("gfs_map", {})
        return {era5: gfs for gfs, era5 in gfs_to_era5.items()}


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the dataset class)
# ---------------------------------------------------------------------------


def _build_file_url(date: pd.Timestamp, base: str, file_type: str = "atm", step: str = "f000") -> str:
    """Build a GFS GDAS file URL for a given date, base path, and file type.

    Args:
        date:      Analysis timestamp (00Z / 06Z / 12Z / 18Z).
        base:      GCS or NOMADS root URL (with or without trailing slash).
        file_type: ``"atm"`` (upper-air) or ``"sfc"`` (surface).
        step:      Forecast step string, e.g. ``"f000"`` (analysis) or ``"anl"``.

    Returns:
        Full URL readable by xarray with the h5netcdf engine + fsspec.

    Example::

        >>> _build_file_url(pd.Timestamp("2024-01-15 06:00"), "gs://global-forecast-system/")
        'gs://global-forecast-system/gdas.20240115/06/atmos/gdas.t06z.atmf000.nc'
    """
    dir_path = date.strftime("gdas.%Y%m%d/%H/atmos/")
    file_name = date.strftime(f"gdas.t%Hz.{file_type}{step}.nc")
    return f"{base.rstrip('/')}/{dir_path}{file_name}"


def _open_gfs_file(url: str, variables: list[str]) -> xr.Dataset:
    """Open a GFS netCDF file and eagerly load the requested variables into memory.

    Uses anonymous GCS access (``token="anon"``) via h5netcdf + fsspec.
    Renames ``grid_xt`` → ``longitude`` and ``grid_yt`` → ``latitude`` for
    consistency with other CREDIT datasets.

    Args:
        url:       Full GCS or NOMADS URL for the ATM or SFC file.
        variables: GFS-native variable names to load (e.g. ``["tmp", "ugrd"]``).

    Returns:
        In-memory xr.Dataset with requested variables and renamed dimensions.
        Returns an empty Dataset if none of the requested variables are present.
    """
    _rename = {"grid_xt": "longitude", "grid_yt": "latitude"}
    storage_opts = {"token": "anon"}

    try:
        with xr.open_dataset(url, engine="h5netcdf", storage_options=storage_opts) as full_ds:
            available = [v for v in variables if v in full_ds.data_vars]
            if not available:
                logger.warning("_open_gfs_file: none of %s found in %s", variables, url)
                return xr.Dataset()
            missing = set(variables) - set(available)
            if missing:
                logger.warning("_open_gfs_file: variables %s not found in %s", sorted(missing), url)
            ds = full_ds[available].load()
    except Exception as exc:
        logger.error("_open_gfs_file: failed to open %s — %s", url, exc)
        raise

    rename = {k: v for k, v in _rename.items() if k in ds.dims or k in ds.coords}
    if rename:
        ds = ds.rename(rename)
    return ds
