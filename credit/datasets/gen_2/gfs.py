"""
gfs.py
------
GFS and GDAS data loading for CREDIT Gen2.

This module provides ``GFSDataset``, a PyTorch dataset for the public GFS/GDAS
NetCDF files in Google Cloud Storage. A model run is represented by two files:
an atmospheric file containing model-level fields and a surface file
containing two-dimensional surface fields. ``GFSDataset`` discovers available
runs, reads only the requested variables and levels, and returns CREDIT's
standard input/target sample dictionaries.

The dataset deliberately does not derive pressure, geopotential, or other
diagnostics and does not regrid the native Gaussian grid. Those operations are
handled by CREDIT postblocks or preblocks so the raw GFS state remains
available to downstream processing.

Remote reads use anonymous ``obstore`` range requests, allowing xarray to read
selected portions of the large NetCDF objects without downloading each file in
full. Local mode reads files downloaded with
``credit.datasets.gen_2.gfs_download``. Both modes use the same native layout::

    {system}.YYYYMMDD/HH/atmos/{system}.tHHz.atmanl.nc
    {system}.YYYYMMDD/HH/atmos/{system}.tHHz.sfcanl.nc

Here ``system`` is ``gdas`` by default and may be changed to ``gfs`` in the
source configuration. Forecast files can be selected with ``forecast_hour``.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr

from credit.datasets.gen_2.base_dataset import VALID_FIELD_TYPES, BaseDataset
from credit.datasets.gen_2.grid_utils import write_source_grid_schema_if_missing

logger = logging.getLogger(__name__)

_GCS_BUCKET = "global-forecast-system"
VALID_SYSTEMS = Literal["gdas", "gfs"]
VALID_LEVEL_TYPES = Literal["model", "pressure"]
_ATM_2D_VARIABLES = {"hgtsfc", "pressfc"}


def _run_path(system: str, t: pd.Timestamp, base_path: str | None = None) -> str:
    prefix = f"{system}.{t:%Y%m%d}/{t:%H}/atmos"
    return os.path.join(base_path, prefix) if base_path else prefix


def _file_names(system: str, t: pd.Timestamp, forecast_hour: int | None) -> tuple[str, str]:
    suffix = "anl" if forecast_hour is None else f"f{forecast_hour:03d}"
    prefix = f"{system}.t{t:%H}z."
    return f"{prefix}atm{suffix}.nc", f"{prefix}sfc{suffix}.nc"


def _file_paths(
    system: str,
    t: pd.Timestamp,
    forecast_hour: int | None,
    base_path: str | None = None,
) -> tuple[str, str]:
    run_path = _run_path(system, t, base_path)
    atm_name, sfc_name = _file_names(system, t, forecast_hour)
    return os.path.join(run_path, atm_name), os.path.join(run_path, sfc_name)


class _ObstoreFile:
    """Adapt obstore's ``ReadableFile`` to the bytes interface expected by h5py."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._reader.read(size).to_bytes()

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def readline(self, size: int = -1) -> bytes:
        return self._reader.readline(size).to_bytes()

    def readlines(self, hint: int = -1) -> list[bytes]:
        return [line.to_bytes() for line in self._reader.readlines(hint)]

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._reader.seek(offset, whence)

    def tell(self) -> int:
        return self._reader.tell()

    def seekable(self) -> bool:
        return self._reader.seekable()

    def readable(self) -> bool:
        return True

    def flush(self) -> None:
        return None

    def close(self) -> None:
        if not self.closed:
            self._reader.close()
            self.closed = True


class GFSDataset(BaseDataset):
    """Read GFS or GDAS atmospheric and surface NetCDF output.

    ``GFSDataset`` reads the paired atmospheric and surface files published in
    the ``global-forecast-system`` Google Cloud bucket. Atmospheric files
    contain three-dimensional model-level fields and a small number of
    two-dimensional fields such as ``pressfc``. Surface files contain the
    two-dimensional surface fields such as ``tmp2m`` and ``land``. The dataset
    returns the same flat, slash-delimited tensor keys as the other Gen2
    datasets; derivations, regridding, and vertical interpolation are left to
    later preblocks or postblocks.

    Remote mode uses ``obstore`` for anonymous, range-based GCS reads. The
    remote NetCDF files are opened with xarray and h5netcdf because the
    obstore reader is a seekable file-like object. Local mode uses the
    ``netcdf4`` xarray engine by default and expects files laid out like the
    public bucket. Availability is checked during initialization by default,
    so missing model runs are removed from ``datetimes`` rather than failing
    later during sampling.

    Input settings:
        dataset_type (str): Must be ``"gfs"`` when routed through
            ``MultiSourceDataset``.
        system (str): Forecast system, either ``"gdas"`` or ``"gfs"``.
            Defaults to ``"gdas"``. The aliases ``model`` and ``gfs_type``
            are also accepted in source configuration.
        mode (str): ``"remote"`` to read from GCS or ``"local"`` to read
            downloaded files. Defaults to ``"remote"`` for this dataset.
        base_path (str): Root directory for local files. Required when
            ``mode`` is ``"local"``. The downloader creates the same
            ``{system}.YYYYMMDD/HH/atmos/`` layout used by the bucket.
        forecast_hour (int | None): Forecast lead hour. ``None`` selects the
            analysis files ``atmanl.nc`` and ``sfcanl.nc``. An integer selects
            files such as ``atmf003.nc`` and ``sfcf003.nc``.
        level_type (str): ``"model"`` selects one-based positions in the
            ``pfull`` dimension. ``"pressure"`` selects the nearest values of
            the file's ``pfull`` coordinate, in hPa. Defaults to ``"model"``.
        level_coord (str): Atmospheric vertical dimension. Defaults to
            ``"pfull"``.
        levels (list[int | float] | None): Requested model-level indices or
            pressure values, depending on ``level_type``. ``None`` reads all
            atmospheric levels.
        check_availability (bool): Check that the required atmospheric and
            surface objects exist before adding a timestamp to ``datetimes``.
            Defaults to ``True``.
        variables (dict): Field definitions grouped under ``prognostic``,
            ``dynamic_forcing``, ``static``, and ``diagnostic``. Each field
            may contain ``vars_3D`` and/or ``vars_2D`` using native GFS names.
        return_target (bool): Constructor argument controlling whether the
            sample includes the next timestep under ``target``.

    Attributes:
        dataset_type (str): The registered dataset type, ``"gfs"``.
        system (str): Active forecast system, ``"gdas"`` or ``"gfs"``.
        mode (str): Active storage mode, ``"remote"`` or ``"local"``.
        base_path (str | None): Expanded local storage root, if configured.
        forecast_hour (int | None): Active analysis or forecast lead hour.
        level_type (str): Active vertical-level interpretation.
        level_coord (str): Name of the atmospheric vertical dimension.
        levels (list[int | float] | None): Configured level selection.
        datetimes (pandas.DatetimeIndex): Available sampling timestamps after
            applying the configured clock and availability checks.
        file_dict (dict): Registered field types and their remote/local file
            source marker.
        var_dict (dict): Registered native GFS variables grouped by field type.
        static_metadata (dict): Calendar, grid, system, level, and forecast
            metadata exposed to ``MultiSourceDataset`` and downstream setup.

    Example YAML configuration::

        data:
          source:
            GDAS:
              dataset_type: "gfs"
              system: "gdas"
              mode: "remote"
              level_type: "model"
              levels: [1, 10, 30, 60, 90, 127]  # selected from the 127 available model levels (1-127)
              check_availability: true
              variables:
                prognostic:
                  vars_3D: [tmp, ugrd, vgrd, spfh]
                  vars_2D: [pressfc, tmp2m]
                dynamic_forcing: null
                static:
                  vars_2D: [land, orog]
                diagnostic: null
          start_datetime: "2024-01-01T00:00:00"
          end_datetime: "2024-01-31T18:00:00"
          timestep: "6h"
          forecast_len: 1

    For pressure-coordinate selection, change the source settings to
    ``level_type: "pressure"`` and provide values such as
    ``levels: [50, 100, 500, 850, 1000]``. To read downloaded files, use
    ``mode: "local"`` and set ``base_path`` to the downloader's output root.

    Command-line usage::

        # Download the configured files for local mode.
        python -m credit.datasets.gen_2.gfs_download -c config/gfs.yml

        # Instantiate GFSDataset directly from a YAML file.
        python - <<'PY'
        import yaml
        from credit.datasets.gen_2.gfs import GFSDataset

        with open("config/gfs.yml") as file:
            config = yaml.safe_load(file)
        dataset = GFSDataset(config["data"], return_target=True)
        print(len(dataset))
        PY

        # In normal training, MultiSourceDataset instantiates GFSDataset from
        # the same data block.
        credit_train_gen2 -c config/gfs.yml
    """

    def __init__(
        self,
        data_config: dict[str, Any],
        return_target: bool = False,
        gfs_type: VALID_SYSTEMS | None = None,
    ) -> None:
        source_name = next(iter(data_config.get("source", {})), None)
        source_cfg = data_config.get("source", {}).get(source_name, {}) if source_name else {}
        self.system: VALID_SYSTEMS = self._validate_system(
            gfs_type or source_cfg.get("gfs_type", source_cfg.get("system", source_cfg.get("model", "gdas")))
        )
        self.mode = source_cfg.get("mode", "remote")
        self.forecast_hour: int | None = source_cfg.get("forecast_hour")
        if self.forecast_hour is not None:
            self.forecast_hour = int(self.forecast_hour)
            if self.forecast_hour < 0 or self.forecast_hour > 999:
                raise ValueError(f"forecast_hour must be between 0 and 999, got {self.forecast_hour}")
        self.base_path = (
            os.path.expanduser(os.path.expandvars(source_cfg["base_path"])) if source_cfg.get("base_path") else None
        )
        self.level_type: VALID_LEVEL_TYPES = self._validate_level_type(source_cfg.get("level_type", "model"))
        self.levels: list[int | float] | None = source_cfg.get("levels")
        self.level_coord = source_cfg.get("level_coord", "pfull")
        self.check_availability = bool(source_cfg.get("check_availability", True))
        self._obstore = None
        self._run_objects: dict[str, set[str]] = {}
        self._initial_source_cfg = source_cfg

        super().__init__(data_config, return_target)

        if "mode" not in self.curr_source_cfg:
            self.mode = "remote"
        if self.mode not in ("local", "remote"):
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'local' or 'remote'.")
        if self.mode == "local" and self.base_path is None:
            raise ValueError(f"A base_path is required for local GFS mode in source '{self.curr_source_name}'.")

        self.dataset_type = "gfs"
        self.static_metadata = {
            "system": self.system,
            "forecast_hour": self.forecast_hour,
            "level_type": self.level_type,
            "levels": self.levels,
            "calendar": self.calendar,
            "datetime_fmt": "unix_ns",
        }
        self.init_register_all_fields()

    @staticmethod
    def _validate_system(value: str) -> VALID_SYSTEMS:
        system = str(value).lower()
        if system not in ("gdas", "gfs"):
            raise ValueError(f"Unknown GFS system '{value}'. Expected 'gdas' or 'gfs'.")
        return system  # type: ignore[return-value]

    @staticmethod
    def _validate_level_type(value: str) -> VALID_LEVEL_TYPES:
        level_type = str(value).lower()
        if level_type not in ("model", "pressure"):
            raise ValueError(f"Unknown GFS level_type '{value}'. Expected 'model' or 'pressure'.")
        return level_type  # type: ignore[return-value]

    def _build_timestamps(self) -> pd.DatetimeIndex:
        timestamps = super()._build_timestamps()
        if not isinstance(timestamps, pd.DatetimeIndex) or not self.check_availability or not len(timestamps):
            return timestamps

        available = [t for t in timestamps if self._run_is_available(pd.Timestamp(t))]
        missing = len(timestamps) - len(available)
        if missing:
            logger.warning(
                "GFSDataset '%s': excluded %d unavailable %s runs from the sampling clock.",
                getattr(self, "curr_source_name", "source"),
                missing,
                self.system.upper(),
            )
        return pd.DatetimeIndex(available)

    def _run_is_available(self, t: pd.Timestamp) -> bool:
        atm_name, sfc_name = _file_names(self.system, t, self.forecast_hour)
        atm_path, sfc_path = _file_paths(self.system, t, self.forecast_hour, self.base_path)
        required_atm, required_sfc = self._required_file_types()
        if self.mode == "local":
            return (not required_atm or os.path.isfile(atm_path)) and (not required_sfc or os.path.isfile(sfc_path))

        prefix = f"{self.system}.{t:%Y%m%d}/{t:%H}"
        if prefix not in self._run_objects:
            self._run_objects[prefix] = self._list_run_objects(prefix)
        objects = self._run_objects[prefix]
        return (not required_atm or atm_name in objects) and (not required_sfc or sfc_name in objects)

    def _required_file_types(self) -> tuple[bool, bool]:
        source_cfg = self.curr_source_cfg if hasattr(self, "curr_source_cfg") else self._initial_source_cfg
        need_atm = False
        need_sfc = False
        for field_cfg in (source_cfg.get("variables") or {}).values():
            if not isinstance(field_cfg, dict):
                continue
            need_atm |= bool(field_cfg.get("vars_3D"))
            for vname in field_cfg.get("vars_2D") or []:
                if vname in _ATM_2D_VARIABLES:
                    need_atm = True
                else:
                    need_sfc = True
        return need_atm, need_sfc

    def _list_run_objects(self, prefix: str) -> set[str]:
        import obstore
        from obstore.store import GCSStore

        if self._obstore is None:
            self._obstore = GCSStore(bucket=_GCS_BUCKET, config={"skip_signature": True})
        result = obstore.list_with_delimiter(self._obstore, prefix=f"{prefix}/atmos/")
        return {entry["path"].rsplit("/", 1)[-1] for entry in result.get("objects", [])}

    def _get_file_source(self, field_config: dict[str, Any]) -> bool:
        return True

    def _register_field(self, field_type: VALID_FIELD_TYPES, field_config: dict[str, Any] | None) -> None:
        super()._register_field(field_type, field_config)
        if field_config is not None:
            self.file_dict[field_type] = True

    def _open_dataset(self, path: str) -> tuple[xr.Dataset, Any | None]:
        if self.mode == "local":
            return xr.open_dataset(path, engine=self.engine or "netcdf4"), None

        import obstore

        if self._obstore is None:
            from obstore.store import GCSStore

            self._obstore = GCSStore(bucket=_GCS_BUCKET, config={"skip_signature": True})
        reader = obstore.open_reader(self._obstore, path)
        return xr.open_dataset(_ObstoreFile(reader), engine="h5netcdf"), reader

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
        atm_vars = vars_3d + [v for v in vars_2d if v in _ATM_2D_VARIABLES]
        sfc_vars = [v for v in vars_2d if v not in _ATM_2D_VARIABLES]
        atm_path, sfc_path = _file_paths(self.system, pd.Timestamp(t), self.forecast_hour, self.base_path)

        if atm_vars:
            self._extract_from_file(field_type, t, atm_path, atm_vars, vars_3d, sample)
        if sfc_vars:
            self._extract_from_file(field_type, t, sfc_path, sfc_vars, [], sample)

    def _extract_from_file(
        self,
        field_type: VALID_FIELD_TYPES,
        t: pd.Timestamp,
        path: str,
        variables: list[str],
        vars_3d: list[str],
        sample: dict[str, Any],
    ) -> None:
        del t
        ds, reader = self._open_dataset(path)
        try:
            if "grid" not in self.static_metadata:
                self._cache_grid(ds)
            missing = [v for v in variables if v not in ds.data_vars]
            if missing:
                raise KeyError(f"Variables {missing} were not found in GFS file '{path}'.")
            ds_t = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
            ds_t = self._select_levels(ds_t) if vars_3d else ds_t
            for vname in variables:
                arr = np.asarray(ds_t[vname].values)
                if vname in vars_3d:
                    key = self._get_field_name(field_type, "3d", vname)
                    sample[key] = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(1)
                else:
                    key = self._get_field_name(field_type, "2d", vname)
                    sample[key] = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        finally:
            ds.close()
            if reader is not None:
                reader.close()

    def _select_levels(self, ds: xr.Dataset) -> xr.Dataset:
        if self.level_coord not in ds.dims:
            raise ValueError(f"GFS atmospheric data does not contain level dimension '{self.level_coord}'.")
        if self.levels is None:
            if self.level_type == "model" and self.level_coord == "pfull":
                self.static_metadata["levels"] = list(range(1, ds.sizes[self.level_coord] + 1))
            else:
                self.static_metadata["levels"] = ds[self.level_coord].values.tolist()
            return ds
        if self.level_type == "model":
            indices = []
            for level in self.levels:
                if int(level) != level or not 1 <= int(level) <= ds.sizes[self.level_coord]:
                    raise ValueError(
                        f"GFS model levels must be one-based indices in [1, {ds.sizes[self.level_coord]}], got {self.levels}."
                    )
                indices.append(int(level) - 1)
            return ds.isel({self.level_coord: indices})
        return ds.sel({self.level_coord: self.levels}, method="nearest")

    def _cache_grid(self, ds: xr.Dataset) -> None:
        if "lat" not in ds or "lon" not in ds:
            return
        grid = {"grid_type": "curvilinear", "lat": ds["lat"].values, "lon": ds["lon"].values}
        self.static_metadata["grid"] = grid
        write_source_grid_schema_if_missing(self.curr_source_name, grid, self.save_loc)
