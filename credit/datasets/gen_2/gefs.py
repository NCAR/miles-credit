"""
gefs.py
-------
GEFS ensemble cube-sphere data loading for CREDIT Gen2.

This module provides ``GEFSDataset`` for the raw GEFS initialization files in
the public ``gfs-ensemble-forecast-system`` Google Cloud bucket. Each selected
ensemble member contains six atmospheric and six surface cube-sphere tiles.
The dataset reads only configured variables, stacks members first, flattens
the six tiles into ``tile_lat_lon``, and leaves regridding and vertical
interpolation to downstream CREDIT blocks.

The raw files are organized as::

    gefs.YYYYMMDD/HH/atmos/init/{member}/gfs_ctrl.nc
    gefs.YYYYMMDD/HH/atmos/init/{member}/gfs_data.tile{1..6}.nc
    gefs.YYYYMMDD/HH/atmos/init/{member}/sfc_data.tile{1..6}.nc

Only initialization-time cube-sphere NetCDF files are supported. Forecast
lead products in the bucket are different GRIB2 products and are intentionally
outside this dataset's scope.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from credit.datasets.gen_2.base_dataset import VALID_FIELD_TYPES, BaseDataset
from credit.datasets.gen_2.gfs import _ObstoreFile  # pyright: ignore[reportPrivateUsage]
from credit.datasets.gen_2.grid_utils import write_source_grid_schema_if_missing

logger = logging.getLogger(__name__)

_GEFS_BUCKET = "gfs-ensemble-forecast-system"
_NUM_TILES = 6
_MEMBER_PATTERN = re.compile(r"(?:c00|p\d{2})$")
_ATM_2D_VARIABLES = {"ps"}
_ATM_3D_VARIABLES = {
    "w",
    "zh",
    "t",
    "delp",
    "sphum",
    "liq_wat",
    "o3mr",
    "ice_wat",
    "rainwat",
    "snowwat",
    "graupel",
    "u_w",
    "v_w",
    "u_s",
    "v_s",
    "u_a",
    "v_a",
}
_SURFACE_VARIABLES = {
    "slmsk",
    "tsea",
    "sheleg",
    "tg3",
    "zorl",
    "alvsf",
    "alvwf",
    "alnsf",
    "alnwf",
    "facsf",
    "facwf",
    "vfrac",
    "canopy",
    "f10m",
    "t2m",
    "q2m",
    "vtype",
    "stype",
    "uustar",
    "ffmm",
    "ffhh",
    "hice",
    "fice",
    "tisfc",
    "tprcp",
    "srflag",
    "snwdph",
    "shdmin",
    "shdmax",
    "slope",
    "snoalb",
    "stc",
    "smc",
    "slc",
    "tref",
    "z_c",
    "c_0",
    "c_d",
    "w_0",
    "w_d",
    "xt",
    "xs",
    "xu",
    "xv",
    "xz",
    "zm",
    "xtts",
    "xzts",
    "d_conv",
    "ifd",
    "dt_cool",
    "qrain",
}


def _run_prefix(t: pd.Timestamp) -> str:
    return f"gefs.{t:%Y%m%d}/{t:%H}/atmos/init"


def _member_prefix(t: pd.Timestamp, member: str, base_path: str | None = None) -> str:
    prefix = f"{_run_prefix(t)}/{member}"
    return os.path.join(base_path, prefix) if base_path else prefix


def _member_file_paths(
    t: pd.Timestamp,
    member: str,
    base_path: str | None = None,
) -> tuple[str, list[str], list[str]]:
    prefix = _member_prefix(t, member, base_path)
    control = os.path.join(prefix, "gfs_ctrl.nc")
    atmospheric = [os.path.join(prefix, f"gfs_data.tile{tile}.nc") for tile in range(1, _NUM_TILES + 1)]
    surface = [os.path.join(prefix, f"sfc_data.tile{tile}.nc") for tile in range(1, _NUM_TILES + 1)]
    return control, atmospheric, surface


def _member_sort_key(member: str) -> tuple[int, int]:
    return (0, 0) if member == "c00" else (1, int(member[1:]))


class GEFSDataset(BaseDataset):
    """Read raw GEFS cube-sphere initialization data for selected members.

    The selected members are stacked in the leading tensor dimension. Six
    cube-sphere tiles are flattened into one spatial dimension so a 3D tensor
    has shape ``(members, levels, 1, tile_lat_lon)`` and a 2D tensor has shape
    ``(members, 1, 1, tile_lat_lon)``. This leading member dimension is
    intentionally preserved through the Gen2 data pipeline and is compatible
    with the flattened-spatial handling in the ``Regridder`` preblock.

    Raw staggered wind fields are requested with their native names ``u_s``,
    ``v_s``, ``u_w``, and ``v_w``. Users who want unstaggered winds should
    request the virtual variables ``u_a`` and ``v_a``; these are computed from
    ``u_s`` and ``v_w`` respectively. There is no ``forecast_hour`` setting:
    this class reads only the cube-sphere initialization NetCDF files.

    Input settings:
        dataset_type (str): Must be ``"gefs"`` when routed through
            ``MultiSourceDataset``.
        members (list[str] | None): Members to read, such as
            ``["c00", "p01", "p02"]``. Omitted configuration defaults to
            ``["c00"]``. An explicit empty list discovers and selects every
            member available in the first requested run.
        mode (str): ``"remote"`` reads from Google Cloud Storage using
            obstore. ``"local"`` reads the directory created by
            ``gefs_download.py``. Defaults to ``"remote"``.
        base_path (str): Root directory for local mode. Required when
            ``mode`` is ``"local"``.
        levels (list[int] | None): One-based model-level indices in the raw
            ``lev`` dimension. The GEFS cube-sphere files contain 65 model
            levels. ``None`` selects all 65 levels. ``zh`` is converted from
            66 interfaces to 65 model-level midpoints before this selection.
        variables (dict): Field definitions grouped under ``prognostic``,
            ``dynamic_forcing``, ``static``, and ``diagnostic``. Use native
            GEFS names in ``vars_3D`` and ``vars_2D``.
        return_target (bool): Constructor argument controlling whether the
            sample includes the next initialization time under ``target``.

    Attributes:
        dataset_type (str): The registered dataset type, ``"gefs"``.
        members (list[str]): Selected members in output stacking order.
        mode (str): Active storage mode, ``"remote"`` or ``"local"``.
        base_path (str | None): Expanded local storage root, if configured.
        levels (list[int] | None): Configured one-based model-level selection.
        datetimes (pandas.DatetimeIndex): Available initialization timestamps.
        file_dict (dict): Registered field types and their source markers.
        var_dict (dict): Registered native GEFS variables grouped by field.
        static_metadata (dict): Selected members, vertical-coordinate metadata,
            and the native unstructured cube-sphere grid.

    Example YAML configuration::

        data:
          source:
            GEFS:
              dataset_type: "gefs"
              mode: "remote"
              members: ["c00"]
              levels: [1, 65]
              variables:
                prognostic:
                  vars_3D: [t]
                  vars_2D: [ps]
                dynamic_forcing: null
                static: null
                diagnostic: null
          start_datetime: "2024-01-01T00:00:00"
          end_datetime: "2024-01-01T06:00:00"
          timestep: "6h"
          forecast_len: 1

    An empty member list selects all members discovered in the first run::

        members: []

    Python usage::

        import yaml
        from credit.datasets.gen_2.gefs import GEFSDataset

        with open("config/gefs.yml") as file:
            config = yaml.safe_load(file)
        dataset = GEFSDataset(config["data"], return_target=True)
        sample = dataset[(dataset.datetimes[0], 0)]
        print(sample["input"]["GEFS/prognostic/3d/t"].shape)
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        source_name = next(iter(data_config.get("source", {})), None)
        source_cfg = data_config.get("source", {}).get(source_name, {}) if source_name else {}
        if "forecast_hour" in source_cfg:
            raise ValueError(
                "GEFSDataset supports initialization-time cube-sphere NetCDF files only; remove forecast_hour."
            )

        configured_members = source_cfg.get("members")
        self._select_all_members = configured_members == []
        self.members: list[str] = ["c00"] if configured_members is None else list(configured_members)
        self._validate_members(self.members)
        self.mode = source_cfg.get("mode", "remote")
        self.engine = source_cfg.get("engine")
        self.base_path = (
            os.path.expanduser(os.path.expandvars(source_cfg["base_path"])) if source_cfg.get("base_path") else None
        )
        self.levels: list[int] | None = source_cfg.get("levels")
        self._obstore = None
        self._run_objects: dict[tuple[pd.Timestamp, str], set[str]] = {}
        self._vcoord: np.ndarray | None = None
        self._initial_source_cfg = source_cfg

        super().__init__(data_config, return_target)

        if "mode" not in self.curr_source_cfg:
            self.mode = "remote"
        if self.mode not in ("local", "remote"):
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'local' or 'remote'.")
        if self.mode == "local" and self.base_path is None:
            raise ValueError(f"A base_path is required for local GEFS mode in source '{self.curr_source_name}'.")
        if self.levels is not None:
            self.levels = [int(level) for level in self.levels]

        self.dataset_type = "gefs"
        self.static_metadata = {
            "members": self.members,
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }
        if self._vcoord is not None:
            self._set_vertical_metadata(self._vcoord)
        self.init_register_all_fields()

    @staticmethod
    def _validate_members(members: list[str]) -> None:
        invalid = [
            member for member in members if not isinstance(member, str) or _MEMBER_PATTERN.fullmatch(member) is None
        ]
        if invalid:
            raise ValueError(f"Invalid GEFS members {invalid}; use 'c00' or perturbation members such as 'p01'.")
        if len(set(members)) != len(members):
            raise ValueError(f"GEFS members must be unique, got {members}.")

    def _build_timestamps(self) -> pd.DatetimeIndex:
        timestamps = super()._build_timestamps()
        if not isinstance(timestamps, pd.DatetimeIndex) or not len(timestamps):
            return timestamps

        first_time = pd.Timestamp(timestamps[0])
        if self._select_all_members:
            self.members = self._discover_members(first_time)
            if not self.members:
                raise FileNotFoundError(f"No GEFS members were found for initialization {first_time}.")

        for timestamp in timestamps:
            self._require_run(pd.Timestamp(timestamp))
        self._load_control_metadata(first_time, self.members[0])
        return timestamps

    def _discover_members(self, t: pd.Timestamp) -> list[str]:
        if self.mode == "local":
            run_path = os.path.join(self.base_path or "", _run_prefix(t))
            if not os.path.isdir(run_path):
                raise FileNotFoundError(f"GEFS initialization directory not found: {run_path}")
            members = [name for name in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, name))]
        else:
            import obstore
            from obstore.store import GCSStore

            if self._obstore is None:
                self._obstore = GCSStore(bucket=_GEFS_BUCKET, config={"skip_signature": True})
            result = obstore.list_with_delimiter(self._obstore, prefix=f"{_run_prefix(t)}/")
            members = [prefix.rstrip("/").rsplit("/", 1)[-1] for prefix in result.get("common_prefixes", [])]
            members.extend(
                entry["path"].rstrip("/").rsplit("/", 1)[-1]
                for entry in result.get("objects", [])
                if entry["path"].rstrip("/").count("/") == _run_prefix(t).count("/") + 1
            )
        valid = sorted({member for member in members if _MEMBER_PATTERN.fullmatch(member)}, key=_member_sort_key)
        return valid

    def _require_run(self, t: pd.Timestamp) -> None:
        missing_by_member: dict[str, list[str]] = {}
        for member in self.members:
            missing = self._missing_member_files(t, member)
            if missing:
                missing_by_member[member] = missing
        if missing_by_member:
            details = "; ".join(f"{member}: {', '.join(files)}" for member, files in missing_by_member.items())
            raise FileNotFoundError(f"GEFS initialization {t} is incomplete for selected members ({details}).")

    def _missing_member_files(self, t: pd.Timestamp, member: str) -> list[str]:
        control, atmospheric, surface = _member_file_paths(t, member, self.base_path)
        required = [
            os.path.basename(control),
            *(os.path.basename(path) for path in atmospheric),
            *(os.path.basename(path) for path in surface),
        ]
        if self.mode == "local":
            paths = [control, *atmospheric, *surface]
            return [os.path.basename(path) for path in paths if not os.path.isfile(path)]

        key = (t, member)
        if key not in self._run_objects:
            self._run_objects[key] = self._list_member_objects(t, member)
        return [name for name in required if name not in self._run_objects[key]]

    def _list_member_objects(self, t: pd.Timestamp, member: str) -> set[str]:
        import obstore
        from obstore.store import GCSStore

        if self._obstore is None:
            self._obstore = GCSStore(bucket=_GEFS_BUCKET, config={"skip_signature": True})
        result = obstore.list_with_delimiter(self._obstore, prefix=f"{_member_prefix(t, member)}/")
        return {entry["path"].rsplit("/", 1)[-1] for entry in result.get("objects", [])}

    def _get_file_source(self, field_config: dict[str, Any]) -> bool:
        return True

    def _register_field(self, field_type: VALID_FIELD_TYPES, field_config: dict[str, Any] | None) -> None:
        super()._register_field(field_type, field_config)
        if field_config is None:
            return
        for variable in self.var_dict[field_type]["vars_3D"] + self.var_dict[field_type]["vars_2D"]:
            if variable not in _ATM_3D_VARIABLES | _ATM_2D_VARIABLES | _SURFACE_VARIABLES:
                raise KeyError(f"Unknown GEFS variable '{variable}'.")
        self.file_dict[field_type] = True

    def _open_dataset(self, path: str) -> tuple[xr.Dataset, Any | None]:
        if self.mode == "local":
            return xr.open_dataset(path, engine=self.engine or "netcdf4"), None

        import obstore
        from obstore.store import GCSStore

        if self._obstore is None:
            self._obstore = GCSStore(bucket=_GEFS_BUCKET, config={"skip_signature": True})
        reader = obstore.open_reader(self._obstore, path)
        return xr.open_dataset(_ObstoreFile(reader), engine="h5netcdf"), reader

    def _load_control_metadata(self, t: pd.Timestamp, member: str) -> None:
        base_path = self.base_path if self.mode == "local" else None
        control_path, _, _ = _member_file_paths(t, member, base_path)
        ds, reader = self._open_dataset(control_path)
        try:
            self._vcoord = np.asarray(ds["vcoord"].values)
        finally:
            ds.close()
            if reader is not None:
                reader.close()
        self._set_vertical_metadata(self._vcoord)

    def _set_vertical_metadata(self, vcoord: np.ndarray | None) -> None:
        if vcoord is None or not hasattr(self, "static_metadata"):
            return
        self.static_metadata["vcoord"] = vcoord
        self.static_metadata["model_a"] = vcoord[0]
        self.static_metadata["model_b"] = vcoord[1]
        self.static_metadata["levels"] = self.levels or list(range(1, vcoord.shape[1]))

    def _extract_field(
        self,
        field_type: VALID_FIELD_TYPES,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        vd = self.var_dict.get(field_type)
        if not vd:
            return
        vars_3d = vd["vars_3D"]
        vars_2d = vd["vars_2D"]
        arrays: dict[str, list[np.ndarray]] = {variable: [] for variable in vars_3d + vars_2d}
        grid_lat: list[np.ndarray] = []
        grid_lon: list[np.ndarray] = []

        for member in self.members:
            member_arrays = {variable: [] for variable in vars_3d + vars_2d}
            for tile in range(1, _NUM_TILES + 1):
                base_path = self.base_path if self.mode == "local" else None
                _, atmospheric, surface = _member_file_paths(pd.Timestamp(t), member, base_path)
                atm_path = atmospheric[tile - 1]
                sfc_path = surface[tile - 1]
                atm_vars = [variable for variable in vars_3d if variable in _ATM_3D_VARIABLES]
                atm_vars.extend(variable for variable in vars_2d if variable in _ATM_2D_VARIABLES)
                if "u_a" in vars_3d and "u_s" not in atm_vars:
                    atm_vars.append("u_s")
                if "v_a" in vars_3d and "v_w" not in atm_vars:
                    atm_vars.append("v_w")
                if atm_vars:
                    ds, reader = self._open_dataset(atm_path)
                    try:
                        if len(grid_lat) < tile:
                            grid_lat.append(np.asarray(ds["geolat"].values))
                            grid_lon.append(np.asarray(ds["geolon"].values))
                        for variable in vars_3d:
                            if variable in _ATM_3D_VARIABLES:
                                member_arrays[variable].append(self._read_atmospheric_variable(ds, variable))
                        for variable in vars_2d:
                            if variable in _ATM_2D_VARIABLES:
                                member_arrays[variable].append(np.asarray(ds[variable].values))
                    finally:
                        ds.close()
                        if reader is not None:
                            reader.close()
                sfc_vars = [variable for variable in vars_2d if variable in _SURFACE_VARIABLES]
                if sfc_vars:
                    ds, reader = self._open_dataset(sfc_path)
                    try:
                        ds_t = ds.isel(Time=0, drop=True) if "Time" in ds.dims else ds
                        if len(grid_lat) < tile and "geolat" in ds_t and "geolon" in ds_t:
                            grid_lat.append(np.asarray(ds_t["geolat"].values))
                            grid_lon.append(np.asarray(ds_t["geolon"].values))
                        for variable in sfc_vars:
                            member_arrays[variable].append(np.asarray(ds_t[variable].values))
                    finally:
                        ds.close()
                        if reader is not None:
                            reader.close()

            for variable in vars_3d + vars_2d:
                if not member_arrays[variable]:
                    raise KeyError(f"GEFS variable '{variable}' was not found in the requested files.")
                arrays[variable].append(
                    np.concatenate([array.reshape(array.shape[0], -1) for array in member_arrays[variable]], axis=1)
                    if variable in vars_3d
                    else np.concatenate([array.reshape(-1) for array in member_arrays[variable]])
                )

        if grid_lat and "grid" not in self.static_metadata:
            self._cache_grid(grid_lat, grid_lon)
        for variable in vars_3d:
            values = torch.as_tensor(np.stack(arrays[variable]), dtype=torch.float32).unsqueeze(2)
            sample[self._get_field_name(field_type, "3d", variable)] = values
        for variable in vars_2d:
            values = torch.as_tensor(np.stack(arrays[variable]), dtype=torch.float32).unsqueeze(1).unsqueeze(2)
            sample[self._get_field_name(field_type, "2d", variable)] = values

    def _read_atmospheric_variable(self, ds: xr.Dataset, variable: str) -> np.ndarray:
        indices = self._level_indices(ds.sizes["lev"])
        if variable == "u_a":
            values = np.asarray(ds["u_s"].values)
            values = 0.5 * (values[:, :-1, :] + values[:, 1:, :])
        elif variable == "v_a":
            values = np.asarray(ds["v_w"].values)
            values = 0.5 * (values[:, :, :-1] + values[:, :, 1:])
        elif variable == "zh":
            values = np.asarray(ds["zh"].values)
            values = 0.5 * (values[:-1] + values[1:])
        else:
            values = np.asarray(ds[variable].values)
        return values[indices]

    def _level_indices(self, n_levels: int) -> list[int]:
        if self.levels is None:
            return list(range(n_levels))
        if any(level < 1 or level > n_levels for level in self.levels):
            raise ValueError(f"GEFS model levels must be one-based indices in [1, {n_levels}], got {self.levels}.")
        return [level - 1 for level in self.levels]

    def _cache_grid(self, lat_tiles: list[np.ndarray], lon_tiles: list[np.ndarray]) -> None:
        lat = np.concatenate([tile.reshape(-1) for tile in lat_tiles])
        lon = np.concatenate([tile.reshape(-1) for tile in lon_tiles])
        grid = {"grid_type": "unstructured", "lat": lat, "lon": lon}
        self.static_metadata["grid"] = grid
        write_source_grid_schema_if_missing(self.curr_source_name, grid, self.save_loc)

    def _extract_field_window(
        self,
        field_type: VALID_FIELD_TYPES,
        t_history: pd.DatetimeIndex,
        sample: dict[str, Any],
    ) -> None:
        if field_type == "static":
            step_sample: dict[str, Any] = {}
            self._extract_field(field_type, t_history[-1], step_sample)
            for key, value in step_sample.items():
                sample[key] = value.repeat(1, 1, len(t_history), 1)
            return

        per_step: list[dict[str, Any]] = []
        for timestamp in t_history:
            step_sample: dict[str, Any] = {}
            self._extract_field(field_type, timestamp, step_sample)
            per_step.append(step_sample)
        for key in per_step[0]:
            sample[key] = torch.cat([step[key] for step in per_step], dim=2)
