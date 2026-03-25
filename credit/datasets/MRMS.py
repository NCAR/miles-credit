"""
MRMS.py
-------------------------------------------------------
MRMSDataset with nested input/target structure.

Field type semantics (mirrors ERA5 conventions):
    prognostic      — input at step 0 AND target; model prediction fed back
                      at step > 0 (autoregressive rollout)
    diagnostic      — target only; not fed back into the model
    dynamic_forcing — input at every step; never a target

Sample structure returned by __getitem__:

    {
        "input": {
            "mrms/prognostic/2d/MultiSensor_QPE_01H_Pass2_00.00": tensor,
            "mrms/dynamic_forcing/2d/MultiSensor_QPE_06H_Pass2_00.00": tensor,
            ...
        },
        "target": {                                  # only when return_target=True
            "mrms/prognostic/2d/MultiSensor_QPE_01H_Pass2_00.00": tensor,
            ...
        },
        "metadata": {
            "input_datetime":  int,                  # nanoseconds since epoch
            "target_datetime": int,                  # only when return_target=True
        },
    }

All MRMS variables are 2D. Tensor shape (no batch dimension):
    (1, 1, lat, lon)   — singleton level dim, consistent with ERA5 2D convention

After DataLoader collation the batch dimension is prepended:
    (batch, 1, 1, lat, lon)

Modes:
    local  — load from NetCDF (.nc) or Zarr (.zarr) files on disk using
             the same ``filename_time_format`` strftime convention as ERA5.
    remote — stream directly from AWS S3 (noaa-mrms-pds, anonymous access)
             via s3fs + pygrib.

File naming (local mode):
    Controlled by the optional ``filename_time_format`` config key.
    Defaults to ``"%Y%m%d-%H%M%S"`` (one file per timestamp).

    Examples::

        filename_time_format: "%Y%m%d-%H%M%S"   # MRMS_20240601-060000.nc
        filename_time_format: "%Y%m%d"           # MRMS_20240601.nc  (daily)
        filename_time_format: "%Y%m"             # MRMS_202406.nc    (monthly)

    If only a single file matches the glob pattern, ``filename_time_format``
    is ignored and that file is used for all timestamps.
"""

from __future__ import annotations

import gzip
import logging
from glob import glob

import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from credit.datasets._file_utils import _find_file, _map_files

logger = logging.getLogger(__name__)

VALID_FIELD_TYPES = {"prognostic", "diagnostic", "dynamic_forcing"}

# S3 URI template for MRMS GRIB2 files
_S3_URI = "s3://noaa-mrms-pds/{region}/{varname}/{date_str}/MRMS_{varname}_{datetime_str}.grib2.gz"


def _apply_extent(da: xr.DataArray, extent: list[float] | None) -> xr.DataArray:
    """Subset *da* to a spatial extent if provided.

    Args:
        da: DataArray with ``lat`` and ``lon`` coordinates (0-360 longitude).
        extent: ``[min_lon, max_lon, min_lat, max_lat]`` in either -180–180 or
            0-360 format; normalised to 0-360 internally.  ``None`` returns
            *da* unchanged.

    Returns:
        Spatially subsetted DataArray, or *da* unchanged if *extent* is ``None``.
    """
    if extent is None:
        return da
    min_lon, max_lon, min_lat, max_lat = extent
    min_lon = min_lon % 360
    max_lon = max_lon % 360
    return da.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))


class MRMSDataset(Dataset):
    """PyTorch Dataset for MRMS data with nested input/target structure.

    Field types follow ERA5 conventions: ``prognostic`` variables appear in
    both input (at step 0) and target; ``dynamic_forcing`` appears in input
    at every step; ``diagnostic`` appears in target only.  At step ``i > 0``
    the model's own prognostic predictions are fed back — no disk read occurs
    for prognostic fields at those steps.

    Supports loading directly from AWS S3 (remote mode) or from local
    NetCDF / Zarr files (local mode). Spatial subsetting via ``extent``
    is applied at load time on the native MRMS grid.

    See module docstring for full description of output format and file naming.

    Example YAML configuration (local mode)::

        data:
          source:
            MRMS:
              mode: "local"
              variables:
                prognostic:                         # input at step 0 + target
                  vars_2D:
                    - "MultiSensor_QPE_01H_Pass2_00.00"
                  path: "/data/MRMS_*.nc"
                  filename_time_format: "%Y%m%d-%H%M%S"
                dynamic_forcing:                    # input every step
                  vars_2D:
                    - "MultiSensor_QPE_06H_Pass2_00.00"
                  path: "/data/MRMS_*.nc"
                  filename_time_format: "%Y%m%d-%H%M%S"
              extent: [-130, -60, 20, 55]   # [min_lon, max_lon, min_lat, max_lat]

          start_datetime: "2024-06-01"
          end_datetime:   "2024-07-01"
          timestep:       "6h"
          forecast_len:   0

    Example YAML configuration (remote mode)::

        data:
          source:
            MRMS:
              mode: "remote"
              region: "CONUS"
              variables:
                prognostic:
                  vars_2D:
                    - "MultiSensor_QPE_01H_Pass2_00.00"
              extent: [-130, -60, 20, 55]

    Assumptions:
        1. Local files have ``time``, ``lat``, ``lon`` dimensions/coordinates.
        2. Longitude coordinates are in the 0–360 convention (both local and remote).
        3. ``extent`` is specified as ``[min_lon, max_lon, min_lat, max_lat]``
           in either -180-180 or 0-360 format; it is normalised to 0-360 internally.
    """

    def __init__(self, config: dict, return_target: bool = False) -> None:
        source_cfg = config["source"]["MRMS"]

        self.source_name: str = "mrms"
        self.return_target: bool = return_target
        self.mode: str = source_cfg.get("mode", "local")
        self.region: str = source_cfg.get("region", "CONUS")
        self.extent: list[float] | None = source_cfg.get("extent", None)
        self.static_metadata: dict = {"datetime_fmt": "unix_ns"}

        self.dt = pd.Timedelta(config["timestep"])
        self.num_forecast_steps: int = config["forecast_len"]

        self.start_datetime = pd.Timestamp(config["start_datetime"])
        self.end_datetime = pd.Timestamp(config["end_datetime"])
        self.datetimes: pd.DatetimeIndex = self._build_timestamps()

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

        Prognostic fields are loaded into ``input`` only at step ``i == 0``
        (consistent with ERA5 autoregressive rollout semantics).  Dynamic
        forcing is loaded at every step.  Diagnostic fields never appear
        in ``input``.

        Args:
            args: ``(t, i)`` where *t* is the current timestamp (nanoseconds
                or pd.Timestamp) and *i* is the within-sequence step index
                produced by the sampler.

        Returns:
            Dict with keys ``"input"``, ``"metadata"``, and optionally
            ``"target"`` (when ``return_target=True``). Both ``"input"`` and
            ``"target"`` are dicts of per-variable tensors keyed by
            ``"mrms/{field_type}/2d/{varname}"``.
        """
        t, i = args
        t = pd.Timestamp(t)
        t_target = t + self.dt

        input_data: dict = {}

        # Dynamic forcing is loaded at every step
        self._extract_field("dynamic_forcing", t, input_data)

        # Prognostic is loaded only at the initial step; at i > 0 the model's
        # own prediction for this source is fed back (autoregressive rollout)
        if i == 0:
            self._extract_field("prognostic", t, input_data)

        sample: dict = {
            "input": input_data,
            "metadata": {"input_datetime": int(t.value)},
        }

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

        Args:
            field_type: One of ``"prognostic"``, ``"diagnostic"``,
                ``"dynamic_forcing"``.
            d: Field-type config dict, or ``None`` / null to disable the field.

        Raises:
            KeyError: If *field_type* is not a recognised MRMS field type.
            ValueError: If *d* defines no ``vars_2D``.
        """
        if field_type not in VALID_FIELD_TYPES:
            raise KeyError(
                f"Unknown field_type '{field_type}' in config['source']['MRMS']. "
                f"Valid options are: {sorted(VALID_FIELD_TYPES)}"
            )
        if not isinstance(d, dict):
            self.file_dict[field_type] = None
            return

        if not d.get("vars_2D"):
            raise ValueError(f"Field '{field_type}' must define vars_2D")

        self.var_dict[field_type] = {"vars_2D": d.get("vars_2D") or []}

        if self.mode == "local":
            files = sorted(glob(d.get("path", "")))
            time_fmt: str = d.get("filename_time_format", "%Y%m%d-%H%M%S")
            self.file_dict[field_type] = _map_files(files, time_fmt) if files else None
        else:
            # Remote mode: S3 path is constructed at runtime from the timestamp
            self.file_dict[field_type] = None

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
        """Load all variables for *field_type* at time *t* into *sample*.

        Dispatches to local or remote loading based on ``self.mode``.

        Args:
            field_type: Registered field type (e.g. ``"prognostic"``).
            t: Timestamp to load.
            sample: Dict to write variable tensors into (modified in place).
                Tensor shape (no batch dimension): ``(1, 1, lat, lon)``.
        """
        vd = self.var_dict.get(field_type)
        if not vd:
            return

        for vname in vd.get("vars_2D", []):
            if self.mode == "remote":
                arr = self._load_remote_var(vname, t)
            else:
                arr = self._load_local_var(field_type, vname, t)

            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sample[f"{self.source_name}/{field_type}/2d/{vname}"] = tensor

    def _load_local_var(self, field_type: str, vname: str, t: pd.Timestamp):
        """Load a single variable from a local NetCDF or Zarr file.

        Args:
            field_type: Field type key used to look up file intervals.
            vname: Variable name within the dataset.
            t: Timestamp to select.

        Returns:
            2-D numpy array ``(lat, lon)`` after optional extent subsetting.

        Raises:
            KeyError: If no files are registered for *field_type*.
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals:
            raise KeyError(
                f"No files registered for field_type '{field_type}'. Check that the path glob matches files on disk."
            )

        path = _find_file(file_intervals, t)
        if path.endswith(".zarr"):
            open_fn = xr.open_zarr
            open_kw: dict = {}
        else:
            open_fn = xr.open_dataset
            open_kw = {"engine": "netcdf4"}

        with open_fn(path, **open_kw) as ds:
            ds_t = ds.sel(time=t) if "time" in ds.dims else ds
            return _apply_extent(ds_t[vname], self.extent).values

    def _load_remote_var(self, vname: str, t: pd.Timestamp):
        """Stream a single variable from the MRMS S3 bucket.

        Imports ``s3fs`` and ``pygrib`` lazily so they are only required
        when remote mode is actually used.

        Args:
            vname: MRMS variable name (used in the S3 path).
            t: Timestamp to fetch.

        Returns:
            2-D numpy array ``(lat, lon)`` after optional extent subsetting.
        """
        import pygrib
        import s3fs

        s3_path = _S3_URI.format(
            region=self.region,
            varname=vname,
            date_str=t.strftime("%Y%m%d"),
            datetime_str=t.strftime("%Y%m%d-%H%M%S"),
        )

        with s3fs.S3FileSystem(anon=True).open(s3_path, "rb") as f:
            raw = pygrib.fromstring(gzip.decompress(f.read()))

        lats = raw.latitudes[:: raw.Nx][::-1]  # ascending south -> north
        lons = raw.longitudes[: raw.Nx]  # 0-360

        da = xr.DataArray(
            raw.values,
            coords={"lat": lats, "lon": lons},
            dims=["lat", "lon"],
        )
        return _apply_extent(da, self.extent).values
