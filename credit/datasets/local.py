"""
local.py
--------
LocalDataset: generic PyTorch Dataset for loading atmospheric data from local
NetCDF/Zarr files. Supports any combination of prognostic, dynamic_forcing,
static, and diagnostic field types with optional 3D (multi-level) and 2D
(surface/single-level) variables.

Sample structure returned by __getitem__::

    {
        "input": {
            "{source_name}/prognostic/3d/T":        tensor,  # (n_levels, 1, lat, lon)
            "{source_name}/prognostic/2d/SP":       tensor,  # (1,        1, lat, lon)
            "{source_name}/dynamic_forcing/2d/tsi": tensor,
            "{source_name}/static/2d/LSM":          tensor,
            ...
        },
        "target": {                                  # only when return_target=True
            "{source_name}/prognostic/3d/T":        tensor,
            "{source_name}/prognostic/2d/SP":       tensor,
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
                "3d"  (multi-level upper-air; requires level_coord in config;
                       if levels is omitted all levels in the file are used)
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

        filename_time_format: "%Y"       # data_2021.zarr
        filename_time_format: "%Y_%m"    # data_2021_06.nc
        filename_time_format: "%Y%m%d"   # data_20210601.nc

    If only a single file matches the glob pattern, ``filename_time_format`` is
    ignored and that file is used for all timestamps.
"""

from __future__ import annotations

import cftime
import logging
from glob import glob
from typing import Any

import pandas as pd
import torch
import xarray as xr

from credit.datasets._utils import (  # pyright: ignore[reportPrivateUsage]
    _find_file,
    _path_template_to_glob,
    _to_cftime,
    is_standard_calendar,
    normalize_calendar,
    to_calendar,
)
from credit.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class LocalDataset(BaseDataset):
    """Generic PyTorch Dataset for local NetCDF/Zarr atmospheric data files.

    See module docstring for full description of output format and file naming.

    Example YAML configuration::

        data:
          source:
            My_Surface_Data:  # User-provided name (arbitrary key)
              dataset_type: "local"
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
        """Initialize LocalDataset with config parsing, timestamp generation, and file mapping.

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return target variables. Defaults to False.
        """
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "local", (
            f"Expected dataset_type 'local' in config for LocalDataset, got '{self.curr_source_cfg['dataset_type']}'"
        )

        self.dataset_type = "local"
        self.level_coord: str | None = self.curr_source_cfg.get("level_coord")
        self.levels: list | None = self.curr_source_cfg.get("levels")
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "calendar": self.calendar,
            "datetime_fmt": "unix_ns" if is_standard_calendar(self.calendar) else f"cf_ns:{self.calendar}",
        }
        self.mode = "local"
        self.time_coord = self.curr_source_cfg.get("time_coord", "time")
        self.init_register_all_fields()

    def _resolve_calendar(self, data_config: dict[str, Any], curr_source_config: dict[str, Any]) -> str | None:
        """Resolve the calendar from config, falling back to sniffing the data.

        Config (source-level then data-level ``calendar:`` key) wins; otherwise
        the first time-bearing data file's time coordinate is inspected once at
        init. Returns None (→ "standard") when neither yields an answer.
        """
        cal = super()._resolve_calendar(data_config, curr_source_config)
        if cal:
            return cal
        return self._sniff_calendar(curr_source_config)

    def _sniff_calendar(self, source_cfg: dict[str, Any]) -> str | None:
        """Read the CF calendar from the first available time-bearing file.

        xarray decodes non-standard-calendar time coordinates to cftime objects
        automatically, so the coordinate's element type is the discriminator.
        Failures are non-fatal: warn and fall back to the standard default.
        """
        time_coord = source_cfg.get("time_coord", "time")
        engine = source_cfg.get("engine")
        variables = source_cfg.get("variables") or {}
        for field_type in ("prognostic", "dynamic_forcing", "diagnostic"):
            field_cfg = variables.get(field_type)
            if not isinstance(field_cfg, dict) or not field_cfg.get("path"):
                continue
            files = sorted(glob(_path_template_to_glob(field_cfg["path"])))
            if not files:
                continue
            try:
                with xr.open_dataset(files[0], engine=engine) as ds:
                    if time_coord not in ds:
                        continue
                    t0 = ds[time_coord].values.ravel()[0]
            except Exception as exc:
                logger.warning(
                    "LocalDataset '%s': could not sniff calendar from %s (%s); assuming 'standard'. "
                    "Set an explicit `calendar:` key in the source config to silence this.",
                    self.curr_source_name,
                    files[0],
                    exc,
                )
                return None
            calendar = normalize_calendar(t0.calendar) if isinstance(t0, cftime.datetime) else "standard"
            if not is_standard_calendar(calendar):
                logger.info(
                    "LocalDataset '%s': sniffed calendar '%s' from %s", self.curr_source_name, calendar, files[0]
                )
            return calendar
        return None

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """Open the dataset for *field_type* at time *t* and populate *sample*.

        Keys written are ``"{source_name}/{field_type}/3d/{varname}"`` for 3D variables
        and ``"{source_name}/{field_type}/2d/{varname}"`` for 2D variables.

        This is the single-step reader (one timestamp). Multi-step history
        (history_len > 1) is handled by :meth:`_extract_field_window`, which
        opens each file at most once across the window.

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
            ds_t = self._select_at_time(ds, t)
            self._write_field_tensors(ds_t, vars_3D, vars_2D, field_type, sample)

    def _extract_field_window(
        self,
        field_type: str,
        t_history: pd.DatetimeIndex,
        sample: dict[str, Any],
    ) -> None:
        """Load *field_type* over the history window and stack along time.

        Overrides the generic per-step reader in
        :meth:`BaseDataset._extract_field_window` to open each underlying file at
        most once. Timestamps in ``t_history`` may span multiple data files (e.g.
        yearly zarrs around a year boundary), so they are grouped by their
        resolved file and each file is opened a single time.

        For fields without a time dimension in the source dataset (typical
        ``"static"``), the single available slice is replicated along the time
        axis ``len(t_history)`` times.

        Produces the same output as the generic reader: 3D →
        ``(n_levels, len(t_history), lat, lon)``, 2D → ``(1, len(t_history), lat, lon)``.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t_history: Chronological timestamps of the history window.
            sample: Dict to write the stacked variable tensors into (modified in place).
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals or field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        # History path: load each timestamp in t_history, stack along time.
        # Group by file so each file is opened at most once.
        n_t = len(t_history)
        per_var_3D: dict[str, list[Any]] = {v: [None] * n_t for v in vars_3D}
        per_var_2D: dict[str, list[Any]] = {v: [None] * n_t for v in vars_2D}

        # Resolve each timestamp to a file path and group indices by file.
        groups: dict[str, list[int]] = {}
        for k, tk in enumerate(t_history):
            # No pd.Timestamp(...) wrap: tk may be cftime.datetime (non-standard
            # calendar), which pd.Timestamp cannot consume; _find_file handles
            # both natively.
            path = _find_file(file_intervals, tk)
            groups.setdefault(path, []).append(k)

        for path, indices in groups.items():
            with xr.open_dataset(path) as ds:
                has_time = self.time_coord in ds.dims
                if not has_time:
                    # Static-style field: load once, replicate at each index.
                    for k in indices:
                        for v in vars_3D:
                            arr = self._read_3d_array(ds, v)
                            per_var_3D[v][k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                        for v in vars_2D:
                            arr = ds[v].values
                            per_var_2D[v][k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                else:
                    # Time-indexed field: select each timestamp in this file.
                    for k in indices:
                        ds_t = self._select_at_time(ds, t_history[k])
                        for v in vars_3D:
                            arr = self._read_3d_array(ds_t, v)
                            per_var_3D[v][k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                        for v in vars_2D:
                            arr = ds_t[v].values
                            per_var_2D[v][k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Concatenate along time dim (dim=1): list of (n_lev, 1, lat, lon)
        # -> (n_lev, n_t, lat, lon); 2D analogously.
        for v in vars_3D:
            key = self._get_field_name(field_type, "3d", v)
            sample[key] = torch.cat(per_var_3D[v], dim=1)
        for v in vars_2D:
            key = self._get_field_name(field_type, "2d", v)
            sample[key] = torch.cat(per_var_2D[v], dim=1)

    # ------------------------------------------------------------------
    # Internal helpers for _extract_field
    # ------------------------------------------------------------------

    def _select_at_time(self, ds: xr.Dataset, t: pd.Timestamp) -> xr.Dataset:
        """Select a single time slice from *ds* at timestamp *t*.

        Handles both numpy datetime64 and cftime calendars, in both directions:
        *t* may be a plain ``pd.Timestamp`` or a ``cftime.datetime`` (e.g. this
        source's own calendar differs from the master clock's, or a mixed-source
        run pairs a noleap source with a standard-calendar one). If the dataset
        has no time dimension (e.g. static fields), returns the dataset
        unchanged.

        Args:
            ds: The open dataset (or a per-time slice of it).
            t: Timestamp to select.

        Returns:
            The dataset selected at *t*, or unchanged if it has no time dim.
        """
        if self.time_coord not in ds.dims:
            return ds
        file_time0 = ds[self.time_coord].values[0]
        if isinstance(file_time0, cftime.datetime):
            t_sel = to_calendar(t, file_time0.calendar)
            if not isinstance(t_sel, cftime.datetime):
                # standard-named calendar stored as cftime objects (e.g. dates
                # outside pandas' nanosecond range): selection still needs a
                # cftime key.
                t_sel = _to_cftime(t_sel, file_time0.calendar)
        else:
            t_sel = to_calendar(t, "standard")
        return ds.sel({self.time_coord: t_sel})

    def _read_3d_array(self, ds_t: xr.Dataset, vname: str):
        """Read a 3D variable from a per-time-slice dataset, applying level
        selection if configured. Lazily caches ``self.levels`` on first use,
        matching the original single-step logic.

        Args:
            ds_t: A single-time-slice dataset.
            vname: 3D variable name.

        Returns:
            numpy array of the 3D variable, level-selected if configured.
        """
        if self.levels is None:
            arr = ds_t[vname].values
            if self.level_coord in ds_t.coords:
                self.levels = ds_t[self.level_coord].values.tolist()
                self.static_metadata["levels"] = self.levels
        else:
            arr = ds_t[vname].sel({self.level_coord: self.levels}, method="nearest").values
        return arr

    def _write_field_tensors(
        self,
        ds_t: xr.Dataset,
        vars_3D: list[str],
        vars_2D: list[str],
        field_type: str,
        sample: dict[str, Any],
    ) -> None:
        """Write single-time-slice tensors for vars_3D/vars_2D into sample.

        Used by the legacy single-step path (history_len == 1).

        Args:
            ds_t: A single-time-slice dataset.
            vars_3D: 3D variable names.
            vars_2D: 2D variable names.
            field_type: The field type being written.
            sample: Dict to write tensors into (modified in place).
        """
        # 3D variables: (n_levels, lat, lon) → (n_levels, 1, lat, lon)
        for vname in vars_3D:
            arr = self._read_3d_array(ds_t, vname)
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
            key = self._get_field_name(field_type, "3d", vname)
            sample[key] = tensor

        # 2D variables: (lat, lon) → (1, 1, lat, lon)
        for vname in vars_2D:
            arr = ds_t[vname].values
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            key = self._get_field_name(field_type, "2d", vname)
            sample[key] = tensor
