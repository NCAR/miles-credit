"""
mrms.py
-------------------------------------------------------
MRMSDataset with nested input/target structure.

Sample structure returned by __getitem__:

    {
        "input": {
            "Example_MRMS/mrms/prognostic/2d/MultiSensor_QPE_01H_Pass2_00.00": tensor,
            "Example_MRMS/mrms/prognostic/2d/MultiSensor_QPE_06H_Pass2_00.00": tensor,
            ...
        },
        "target": {                                  # only when return_target=True
            "Example_MRMS/mrms/prognostic/2d/MultiSensor_QPE_01H_Pass2_00.00": tensor,
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

from typing import Any

from glob import glob
import gzip

import pandas as pd
import torch
import xarray as xr

from credit.datasets._utils import _find_file, _map_files
from credit.datasets.base_dataset import BaseDataset


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


class MRMSDataset(BaseDataset):
    """PyTorch Dataset for MRMS data with nested input/target structure.

    Field types follow CREDIT Gen2 conventions: ``prognostic`` variables appear in
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
            Example_MRMS:  # User-provided name (arbitrary key)
              dataset_type: "mrms"
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
            Example_MRMS:  # User-provided name (arbitrary key)
              dataset_type: "mrms"
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

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """Initialize MRMSDataset with config parsing, timestamp generation, file mapping from BaseDataset,
        then set MRMS-specific attributes.

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return target variables. Defaults to False.
        """

        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "mrms", (
            f"Expected dataset_type 'mrms' in config for MRMSDataset, got '{self.curr_source_cfg['dataset_type']}'"
        )
        # Set MRMS-specific attributes
        self.dataset_type: str = "mrms"
        self.region: str = self.curr_source_cfg.get("region", "CONUS")
        self.extent: list[float] | None = self.curr_source_cfg.get("extent", None)
        self.static_metadata: dict = {"datetime_fmt": "unix_ns"}

        # Initialize the field registration based on the provided config and populate
        #   dictionary of variables and file paths for each field type
        self.init_register_all_fields()

        # Initialize the s3fs on the first call to _extract_field within __getitem__
        self._fs = None

    def _init_fs(self):
        """Lazily initialize an anonymous ``s3fs.S3FileSystem`` instance.

        Called automatically on the first ``__getitem__`` invocation when
        ``mode`` is ``"remote"``. The filesystem object is cached in ``_fs``
        for re-use across later calls.

        Note:
            Mirrors the initialization pattern used in
            ``ARCOERA5Dataset._init_fs()``.
        """
        import s3fs

        fs_config = {
            "anon": True,
            "token": "anon",
            "default_block_size": 8**20,
        }
        self._fs = s3fs.S3FileSystem(**fs_config)

    def _get_file_source(
        self,
        field_config: dict[str, Any],
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, str]] | bool | None:
        """Return the file source for a field. Override in subclasses for different modes/backends.

        Args:
            field_config (dict[str, Any]): Validated field-type config dict.

        Raises:
            ValueError: If ``self.mode`` is not a recognised mode.

        Returns:
            list[tuple[pd.Timestamp, pd.Timestamp, str]] | bool | None: Depending on the mode and field type,
                this method may return a list of (start_time, end_time, file_path) tuples produced by _map_files,
                a boolean indicating the presence of the field (e.g., for remote data), or None if the field is disabled.
                The expected return type should be consistent within a dataset class.
        """
        if self.mode == "local":
            files = sorted(glob(field_config.get("path", "")))
            time_fmt: str = field_config.get("filename_time_format", "%Y%m%d-%H%M%S")
            return _map_files(files, time_fmt) if files else None
        else:
            # Remote mode: S3 path is constructed at runtime from the timestamp
            return True

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict,
    ) -> None:
        """Load all 2-D variables for *field_type* at time *t* into *sample*.

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
            key = self._get_field_name(field_type, "2d", vname)
            sample[key] = tensor

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

        # S3 URI template for MRMS GRIB2 files
        _S3_URI = "s3://noaa-mrms-pds/{region}/{varname}/{date_str}/MRMS_{varname}_{datetime_str}.grib2.gz"

        s3_path = _S3_URI.format(
            region=self.region,
            varname=vname,
            date_str=t.strftime("%Y%m%d"),
            datetime_str=t.strftime("%Y%m%d-%H%M%S"),
        )

        with self._fs.open(s3_path, "rb") as f:
            raw = pygrib.fromstring(gzip.decompress(f.read()))

        lats = raw.latitudes[:: raw.Nx][::-1]  # ascending south -> north
        lons = raw.longitudes[: raw.Nx]  # 0-360

        da = xr.DataArray(
            raw.values,
            coords={"lat": lats, "lon": lons},
            dims=["lat", "lon"],
        )
        return _apply_extent(da, self.extent).values
