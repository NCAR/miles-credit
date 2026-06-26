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
from typing import Any

import pandas as pd
import torch
import xarray as xr

from credit.datasets._utils import _find_file, _to_cftime  # pyright: ignore[reportPrivateUsage]
from credit.datasets.base_dataset import BaseDataset


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
            "datetime_fmt": "unix_ns",
        }
        self.mode = "local"
        self.time_coord = self.curr_source_cfg.get("time_coord", "time")
        self.init_register_all_fields()

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
        *,
        t_history: pd.DatetimeIndex | None = None,
    ) -> None:
        """Open the dataset for *field_type* at time *t* and populate *sample*.

        Keys written are ``"{source_name}/{field_type}/3d/{varname}"`` for 3D variables
        and ``"{source_name}/{field_type}/2d/{varname}"`` for 2D variables.

        When ``t_history`` is provided (history_len > 1 path), the field is
        loaded as a window of ``len(t_history)`` time steps stacked along the
        time dimension. Timestamps in ``t_history`` may span multiple data
        files (e.g. yearly zarrs around a year boundary); the function groups
        timestamps by their resolved file and opens each file once.

        For fields without a time dimension in the source dataset (typical
        ``"static"``), the single available slice is replicated along the
        time axis ``len(t_history)`` times.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t: Timestamp to select (single-step path) or end timestamp of the
                history window (history path; equals ``t_history[-1]``).
            sample: Dict to write variable tensors into (modified in place).
                Tensor shapes (no batch dimension):

                - 3D variable: ``(n_levels, T, lat, lon)``
                - 2D variable: ``(1, T, lat, lon)``

                where ``T == 1`` in the single-step path and
                ``T == len(t_history)`` in the history path.
            t_history: When provided, the time stamps (in chronological order,
                ending at ``t``) to load and stack. When None, only ``t`` is
                loaded.
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals or field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        # Single-step path (legacy history_len == 1).
        if t_history is None:
            with xr.open_dataset(_find_file(file_intervals, t)) as ds:
                ds_t = self._select_at_time(ds, t)
                self._write_field_tensors(ds_t, vars_3D, vars_2D, field_type, sample)
            return

        # History path: load each timestamp in t_history, stack along time.
        # Group by file so each file is opened at most once.
        n_t = len(t_history)
        per_var_3D: dict[str, list[Any]] = {v: [None] * n_t for v in vars_3D}
        per_var_2D: dict[str, list[Any]] = {v: [None] * n_t for v in vars_2D}

        # Resolve each timestamp to a file path and group indices by file.
        groups: dict[str, list[int]] = {}
        for k, tk in enumerate(t_history):
            path = _find_file(file_intervals, pd.Timestamp(tk))
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
                        ds_t = self._select_at_time(ds, pd.Timestamp(t_history[k]))
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

        Handles both numpy datetime64 and cftime calendars. If the dataset
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
        if isinstance(ds[self.time_coord].values[0], cftime.datetime):
            calendar = ds[self.time_coord].values[0].calendar
            t_sel = _to_cftime(t, calendar)
        else:
            t_sel = t
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