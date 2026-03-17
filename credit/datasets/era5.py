"""
era5.py
-------------------------------------------------------
Refactored ERA5Dataset with nested input/target structure.

Sample structure returned by __getitem__:

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

import cftime
import logging
from glob import glob

import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from credit.datasets._file_utils import _find_file, _map_files

logger = logging.getLogger(__name__)

VALID_FIELD_TYPES = {"prognostic", "dynamic_forcing", "static", "diagnostic"}


class ERA5Dataset(Dataset):
    """PyTorch Dataset for processed ERA5 data with nested input/target structure.

    See module docstring for full description of output format and file naming.

    Example YAML configuration::

        data:
          source:
            ERA5:
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
          end_datetime:   "2019-12-31"
          timestep:       "6h"
          forecast_len:   0

    Assumptions:
        1. A "time" dimension / coordinate is present for non-static fields.
        2. A level coordinate (name given by ``level_coord``) represents the
           vertical axis of 3D variables.
        3. Dimension order: (time, level, latitude, longitude) for 3D;
           (time, latitude, longitude) for 2D; (latitude, longitude) for static.
    """

    def __init__(self, config: dict, return_target: bool = False) -> None:
        source_cfg = config["source"]["ERA5"]

        self.source_name: str = "era5"
        self.level_coord: str = source_cfg["level_coord"]
        self.levels: list[int] = source_cfg["levels"]
        self.return_target: bool = return_target
        self.static_metadata: dict = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }

        self.dt = pd.Timedelta(config["timestep"])
        self.num_forecast_steps: int = config["forecast_len"]

        self.start_datetime = pd.Timestamp(config["start_datetime"])
        self.end_datetime = pd.Timestamp(config["end_datetime"])
        self.datetimes: pd.DatetimeIndex = self._build_timestamps()

        # file_dict maps field_type → sorted list of (start, end, path) intervals
        self.file_dict: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]] | None] = {}
        self.var_dict: dict[str, dict[str, list[str]]] = {}

        for field_type, d in source_cfg["variables"].items():
            self._register_field(field_type, d)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.datetimes)

    def __getitem__(self, args: tuple) -> dict:
        """Return a nested input/target sample dict.

        Args:
            args: ``(t, i)`` where *t* is the current timestamp (nanoseconds
                or pd.Timestamp) and *i* is the within-sequence step index
                produced by the sampler. When ``i == 0`` prognostic and static
                fields are loaded in addition to dynamic forcing.

        Returns:
            Dict with keys ``"input"``, ``"metadata"``, and optionally
            ``"target"`` (when ``return_target=True``). Both ``"input"`` and
            ``"target"`` are dicts of per-variable tensors keyed by
            ``"era5/{field_type}/{dim}/{varname}"``.
        """
        t, i = args
        t = pd.Timestamp(t)
        t_target = t + self.dt

        input_data: dict = {}

        # Dynamic forcing is loaded at every step
        self._extract_field("dynamic_forcing", t, input_data)

        # Prognostic + static are only needed at the initial step
        if i == 0:
            self._extract_field("static", t, input_data)
            self._extract_field("prognostic", t, input_data)

        sample: dict = {
            "input": input_data,
            "metadata": {"input_datetime": int(t.value)},
        }

        # Optionally load t+1 as the supervised target
        if self.return_target:
            target_data: dict = {}
            for field_type in ("prognostic", "diagnostic"):
                if self.file_dict.get(field_type) and field_type in self.var_dict:
                    self._extract_field(field_type, t_target, target_data)

            sample["target"] = target_data
            sample["metadata"]["target_datetime"] = int(t_target.value)

        return sample

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register_field(self, field_type: str, d: dict | None) -> None:
        """Validate and register one field type from the config variables block.

        Populates ``self.file_dict`` and ``self.var_dict`` for *field_type*.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            d: Field-type config dict, or ``None`` / null to disable the field.

        Raises:
            KeyError: If *field_type* is not a recognised field type.
            ValueError: If *d* defines neither ``vars_3D`` nor ``vars_2D``.
        """
        if field_type not in VALID_FIELD_TYPES:
            raise KeyError(
                f"Unknown field_type '{field_type}' in config['source']['ERA5']. "
                f"Valid options are: {sorted(VALID_FIELD_TYPES)}"
            )
        if not isinstance(d, dict):
            # null / disabled field
            self.file_dict[field_type] = None
            return

        if not d.get("vars_3D") and not d.get("vars_2D"):
            raise ValueError(f"Field '{field_type}' must define at least one of vars_3D or vars_2D")

        files = sorted(glob(d.get("path", "")))
        time_fmt: str = d.get("filename_time_format", "%Y")
        self.file_dict[field_type] = _map_files(files, time_fmt) if files else None
        self.var_dict[field_type] = {
            "vars_3D": d.get("vars_3D") or [],
            "vars_2D": d.get("vars_2D") or [],
        }

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
        field_type: str,
        t: pd.Timestamp,
        sample: dict,
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
                sample[f"{self.source_name}/{field_type}/3d/{vname}"] = tensor

            # 2D variables: (lat, lon) → (1, 1, lat, lon)
            for vname in vars_2D:
                arr = ds_t[vname].values
                tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                sample[f"{self.source_name}/{field_type}/2d/{vname}"] = tensor

    @staticmethod
    def _to_cftime(ts: pd.Timestamp, calendar: str) -> cftime.datetime:
        """Convert a pandas Timestamp to a cftime.datetime.

        Args:
            ts: Pandas Timestamp to convert.
            calendar: cftime calendar string read from the dataset
                (e.g. ``"noleap"``, ``"gregorian"``, ``"proleptic_gregorian"``).

        Returns:
            cftime.datetime with the specified calendar.
        """
        return cftime.datetime(
            ts.year,
            ts.month,
            ts.day,
            ts.hour,
            ts.minute,
            ts.second,
            calendar=calendar,
        )
