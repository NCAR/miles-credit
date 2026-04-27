"""
GOES.py
-------------------------------------------------------
GOESDataset with nested input/target structure.

Sample structure returned by __getitem__:

    {
        "input": {
            "goes16/prognostic/2d/CMI_C04": tensor,
            "goes16/prognostic/2d/CMI_C07": tensor,
            ...
        },
        "target": {                                  # only when return_target=True
            "goes16/prognostic/2d/CMI_C04": tensor,
            "goes16/prognostic/2d/CMI_C07": tensor,
            ...
        },
        "metadata": {
            "input_datetime":  int,                  # nanoseconds since epoch
            "target_datetime": int,                  # only when return_target=True
        },
    }

All GOES variables are 2D. Tensor shape (no batch dimension):
    (1, 1, lat, lon)   — singleton level dim, consistent with ERA5/MRMS 2D convention

After DataLoader collation the batch dimension is prepended:
    (batch, 1, 1, lat, lon)
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

from credit.datasets._file_utils import _infer_period_freq, _find_file

logger = logging.getLogger(__name__)

VALID_FIELD_TYPES = {"prognostic", "diagnostic", "dynamic_forcing"}


def _build_spatial_slices(
    extent: list[int] | None,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
) -> tuple[slice, slice]:
    """Compute row (latitude) and column (longitude) slices that bound a geographic extent on a 2-D grid.

    Given an optional bounding box in geographic coordinates, returns a pair of
    ``slice`` objects that can be passed directly to ``xarray.Dataset.isel`` (or
    NumPy fancy-indexing) to crop a 2-D field to the requested region.

    Args:
        extent: Bounding box as ``[lon_min, lon_max, lat_min, lat_max]`` in
            decimal degrees. Pass ``None`` to select the entire grid (both
            slices become ``slice(None)``). Valid range for longitude: ``[-180, 180]``;
            and latitude: ``[-90, 90]``.
        lat2d: 2-D array of latitudes (degrees) with the same shape as the
            target grid. Required when ``extent`` is not ``None``.
        lon2d: 2-D array of longitudes (degrees) with the same shape as the
            target grid. Required when ``extent`` is not ``None``.

    Returns:
        A ``(y_slice, x_slice)`` tuple where ``y_slice`` indexes rows
        (latitude axis) and ``x_slice`` indexes columns (longitude axis) of
        the 2-D grid.

    Raises:
        ValueError: If ``extent`` is a list but ``lat2d`` or ``lon2d`` are
            ``None``.
        TypeError: If ``extent`` is neither ``None`` nor a list.
    """
    if extent is None:
        y_slice = slice(None)
        x_slice = slice(None)

    elif isinstance(extent, list):
        if lat2d is None or lon2d is None:
            raise ValueError(
                "A geographic extent requires lat2d and lon2d; pass 2-D coordinate arrays or set extent=None."
            )

        lon_min, lon_max, lat_min, lat_max = extent

        i_all, j_all = [], []
        for lat in (lat_min, lat_max):
            for lon in (lon_min, lon_max):
                i, j = _find_nearest_latlon(lat2d, lon2d, lat, lon)
                i_all.append(i)
                j_all.append(j)

        y_slice = slice(np.min(i_all), np.max(i_all) + 1)
        x_slice = slice(np.min(j_all), np.max(j_all) + 1)
    else:
        raise TypeError(f"Unsupported extent type: {type(extent)}")

    return y_slice, x_slice


def _find_nearest_latlon(lat2d: np.ndarray, lon2d: np.ndarray, lat_target: float, lon_target: float) -> tuple[int, int]:
    """Find the 2-D grid indices of the point nearest to a target lat/lon using Haversine distance.

    Args:
        lat2d: 2-D array of latitudes in decimal degrees.
        lon2d: 2-D array of longitudes in decimal degrees.
        lat_target: Target latitude in decimal degrees. Valid range for latitude: ``[-90, 90]``.
        lon_target: Target longitude in decimal degrees. Valid range for longitude: ``[-180, 180]``;

    Returns:
        A ``(i, j)`` tuple of the row and column indices of the nearest grid
        point.
    """
    lat2d_r = np.deg2rad(lat2d)
    lon2d_r = np.deg2rad(lon2d)
    lat_t_r = np.deg2rad(lat_target)
    lon_t_r = np.deg2rad(lon_target)

    dlat = lat2d_r - lat_t_r
    dlon = lon2d_r - lon_t_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_t_r) * np.cos(lat2d_r) * np.sin(dlon / 2) ** 2
    dist_sq = np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    idx_flat = np.nanargmin(dist_sq)
    i, j = np.unravel_index(idx_flat, lat2d.shape)

    return i, j


class GOESDataset(Dataset):
    """PyTorch Dataset for GOES-R ABI Level-2 (L2) satellite imagery.

    Field types follow ERA5/MRMS conventions: ``prognostic`` variables appear in
    both input (at step 0) and target; ``dynamic_forcing`` appears in input
    at every step; ``diagnostic`` appears in target only.  At step ``i > 0``
    the model's own prognostic predictions are fed back — no disk read occurs
    for prognostic fields at those steps.

    Supports loading directly from AWS S3 (remote mode) or from local
    NetCDF files (local mode). Spatial subsetting via ``extent``
    is applied at load time on the curvilinear GOES grid.

    See module docstring for full description of output format and file naming.

    Example YAML configuration (local mode):

        data:
            source:
                GOES:
                    goes_id: "goes16"
                    mode: "local"
                    product: "ABI-L2-MCMIPC"
                    variables:
                        prognostic:
                            vars_2D:
                                - "CMI_C04"
                                - "CMI_C07"
                                - "CMI_C08"
                                - "CMI_C09"
                                - "CMI_C10"
                                - "CMI_C13"
                            path: "/glade/derecho/scratch/kevinyang/datasets/goes/"
                    diagnostic: null
                    dynamic_forcing: null
                latlon2d_dir: "/glade/derecho/scratch/kevinyang/datasets/goes/"
                extent: [-130, -60, 20, 55]

            start_datetime: "2021-06-01"
            end_datetime: "2021-06-04"
            timestep: "6h"
            forecast_len: 0

    Example YAML configuration (remote mode):

        data:
            source:
                GOES:
                    goes_id: "goes16"
                    mode: "remote"
                    product: "ABI-L2-MCMIPC"
                    variables:
                        prognostic:
                            vars_2D:
                                - "CMI_C04"
                                - "CMI_C07"
                                - "CMI_C08"
                                - "CMI_C09"
                                - "CMI_C10"
                                - "CMI_C13"
                    diagnostic: null
                    dynamic_forcing: null
                latlon2d_dir: "/glade/derecho/scratch/kevinyang/datasets/goes/"
                extent: [-130, -60, 20, 55]

    Args:
        config: Top-level experiment configuration dictionary. The relevant
            sub-keys are:

            - ``config["source"]["GOES"]``: GOES-specific settings.

              - ``goes_id`` (str): Satellite identifier. One of ``"goes16"``,
                ``"goes17"``, ``"goes18"``, ``"goes19"``. Defaults to
                ``"goes16"``.
              - ``mode`` (str): ``"local"`` or ``"remote"`` (S3). Defaults to
                ``"local"``.
              - ``product`` (str): ABI product string, e.g.
                ``"ABI-L2-MCMIPC"``.
              - ``extent`` (list, optional): Bounding box
                ``[lon_min, lon_max, lat_min, lat_max]`` to spatially crop
                each field.
              - ``latlon2d_dir`` (str): Directory containing pre-computed
                lat/lon grid NetCDF files.
              - ``qc_path`` (str, optional): Path to a Parquet QC table.
                Timestamps that miss file or fail QC are replaced with ``None`` in the file
                map.
              - ``variables`` (dict): Mapping of field_type to variable spec,

            - ``config["timestep"]`` (str): Model timestep as a
              ``pandas.Timedelta``-parseable string (e.g. ``"1h"``).
            - ``config["forecast_len"]`` (int): Number of autoregressive
              forecast steps.
            - ``config["start_datetime"]`` (str): Start of the data range.
            - ``config["end_datetime"]`` (str): End of the data range.

        return_target: When ``True`` the sample also contains a ``"target"``
            key populated with prognostic and diagnostic fields at ``t + dt``.
            Defaults to ``False``.

    Attributes:
        datetimes (pd.DatetimeIndex): Valid input times for which samples can
            be fetched.
        file_dict (dict): Maps each field type to a list of
            ``(period_start, period_end, file path)`` tuples built during
            initialization.
        var_dict (dict): Maps each field type to
            ``{"vars_2D": [<variable names>]}``.
        y_slice (slice): Row crop derived from ``extent`` (or ``slice(None)``
            for the full grid).
        x_slice (slice): Column crop derived from ``extent`` (or
            ``slice(None)`` for the full grid).

    Raises:
        FileNotFoundError: If the lat/lon grid NetCDF cannot be found under
            ``latlon2d_dir``.
        ValueError: If ``goes_id`` or ``region`` are not recognized.
    """

    def __init__(self, config: dict, return_target: bool = False) -> None:
        source_cfg = config["source"]["GOES"]

        self.goes_id: str = source_cfg.get("goes_id", "goes16")
        self.return_target: bool = return_target
        self.mode: str = source_cfg.get("mode", "local")
        self.product: str = source_cfg.get("product", "ABI-L2-MCMIPC")

        self.static_metadata: dict = {"datetime_fmt": "unix_ns"}

        self.dt = pd.Timedelta(config["timestep"])
        self.num_forecast_steps: int = config["forecast_len"]

        self.start_datetime = pd.Timestamp(config["start_datetime"])
        self.end_datetime = pd.Timestamp(config["end_datetime"])
        self.datetimes: pd.DatetimeIndex = self._build_timestamps()

        self.file_dict: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]] | None] = {}
        self.var_dict: dict[str, dict[str, list[str]]] = {}

        self.qc_path: str = source_cfg.get("qc_path", None)

        # Hard-coded configurations
        self.tolerance = pd.Timedelta("3 minutes")  # to allow for searching the nearest GOES observation time

        # Initialize the s3fs on the first call to __getitem__
        self._fs = None

        # Build datetime-to-filepath lookup from source config variables
        for field_type, d in source_cfg["variables"].items():
            self._register_field(field_type, d)

        # Pre-compute spatial slices from GOES fixed lat/lon grids
        self.latlon2d_dir: str = source_cfg.get("latlon2d_dir", "")

        if self.goes_id in ("goes16", "goes19"):  # both GOES-East
            prefix = "goes19"
        elif self.goes_id in ("goes17", "goes18"):  # both GOES-West
            prefix = "goes18"
        else:
            raise ValueError(f"Unrecognized GOES ID: {self.goes_id!r}")

        if self.product[-1] == "C":
            suffix = "abi_conus_lat_lon.nc"
        elif self.product[-1] == "F":
            suffix = "abi_full_disk_lat_lon.nc"

        self.extent = source_cfg.get("extent", None)
        latlon2d_path = os.path.join(self.latlon2d_dir, f"{prefix}_{suffix}")

        try:
            with xr.open_dataset(latlon2d_path) as ds:
                self.y_slice, self.x_slice = _build_spatial_slices(
                    self.extent,
                    ds.latitude.values,
                    ds.longitude.values,
                )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Latitude/longitude grid file not found at {latlon2d_path}") from e

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.datetimes)

    def __getitem__(self, args: tuple) -> dict:
        """Return a nested input/target sample dict.

        Prognostic fields are loaded into ``input`` only at step ``i == 0``
        (consistent with ERA5/MRMS autoregressive rollout semantics).  Dynamic
        forcing is loaded at every step.  Diagnostic fields never appear
        in ``input``.

        Note: mirrors the MRMSDataset.__getitem__()

        Args:
            args: ``(t, i)`` where *t* is the current timestamp (nanoseconds
                or pd.Timestamp) and *i* is the within-sequence step index
                produced by the sampler.

        Returns:
            Dict with keys ``"input"``, ``"metadata"``, and optionally
            ``"target"`` (when ``return_target=True``). Both ``"input"`` and
            ``"target"`` are dicts of per-variable tensors keyed by
            ``"{goes_id}/{field_type}/2d/{vname}"``.
        """
        t, i = args
        t = pd.Timestamp(t)
        t_target = t + self.dt

        if self._fs is None:
            self._init_fs()

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

    def _collect_GOES_file_path(self, base_dir: str = "", verbose: bool = False):
        """Build a time-ordered file map for the dataset's datetime range.

        For each requested timestamp the method lists the appropriate S3 or
        local hourly directory, parses GOES L2 filenames, and associates each
        timestamp with the nearest file within ``tolerance`` (default 3
        minutes). QC filtering is applied automatically when ``self.qc_path``
        is set, masking bad intervals by setting their path entry to ``None``.

        Args:
            base_dir: Root directory prepended to relative paths when ``mode``
                is ``"local"``. Ignored for remote mode.
            verbose: When ``True``, print a warning for each hour directory
                that cannot be listed (missing data, permission errors, etc.).

        Returns:
            A list of ``(period_start, period_end, file_path)`` tuples, one
            per timestamp in ``datetimes``. ``file_path`` is ``"NONE"`` when
            no file was found within tolerance, or ``None`` when the interval
            was masked by the QC table at ``self.qc_path``.

        Raises:
            FileNotFoundError: If GOES L2 files are not found for
                 the requested datetime.
            ValueError: If the GOES L2 filenames do not match the expected
                naming convention (fewer than 6 underscore-separated tokens).
        """
        # -- Collect file paths from local or remote hourly directories --
        if self.mode == "remote":
            import s3fs

            fs = s3fs.S3FileSystem(anon=True, token="anon")

        file_paths = []
        for dt in self.datetimes.floor("h").unique():
            rel_path = os.path.join(
                f"noaa-{self.goes_id}/{self.product}",
                str(dt.year),
                dt.strftime("%j"),
                f"{dt.hour:02}",
            )
            try:
                if self.mode == "remote":
                    hourly_dir = f"s3://{rel_path}"
                    file_paths.extend(fs.ls(hourly_dir))
                elif self.mode == "local":
                    hourly_dir = os.path.join(base_dir, rel_path)
                    file_paths.extend(os.path.join(hourly_dir, f) for f in os.listdir(hourly_dir))
            except Exception as e:
                if verbose:
                    print(f"[WARN] No data at {dt}: {e}")

        if not file_paths:
            raise FileNotFoundError("No valid GOES-L2 files found for the given datetimes.")

        # -- Parse GOES L2 filenames and convert timestamp fields --
        # reference: _goes_file_df() under goes2go/src/goes2go/data.py
        df = pd.DataFrame(file_paths, columns=["file"])

        parts = df["file"].str.rsplit("_", expand=True, n=5)
        if parts.shape[1] < 6:
            raise ValueError("Unexpected GOES L2 filename structure encountered.")

        df[["product_mode", "satellite", "start", "end", "creation"]] = parts.loc[:, 1:]
        df["start"] = pd.to_datetime(df.start, format="s%Y%j%H%M%S%f")  # convert string times to datetime
        df["end"] = pd.to_datetime(df.end, format="e%Y%j%H%M%S%f")
        df["creation"] = pd.to_datetime(df.creation, format="c%Y%j%H%M%S%f.nc")

        df = df.dropna(subset=["start"]).sort_values("start").set_index("start")  # drop malformed rows if any

        # -- Match each requested timestamp to its nearest GOES file --
        unique_times = df.index.unique()
        nearest_indices = unique_times.get_indexer(self.datetimes, method="nearest", tolerance=self.tolerance)

        freq = _infer_period_freq("s%Y%j%H%M%S%f")

        time_file_map = []
        for query_time, nt_idx in zip(self.datetimes, nearest_indices):
            period = pd.Period(query_time, freq)
            if nt_idx == -1:
                time_file_map.append((period.start_time, period.end_time, "NONE"))
                continue
            files_at_time = df.loc[unique_times[nt_idx], "file"]
            time_file_map.append((period.start_time, period.end_time, files_at_time))

        # -- Apply QC mask if a QC table is provided --
        if self.qc_path is not None:
            df = pd.read_parquet(self.qc_path)
            time_file_map_qc = list(df.itertuples(index=False, name=None))

            list1 = time_file_map
            list2 = time_file_map_qc

            lookup = {(item[0], item[1]): item[2] for item in list2}
            merged = []
            for a, b, c in list1:
                val2 = lookup.get((a, b))
                merged_val = None if (c is None or val2 is None) else c
                merged.append((a, b, merged_val))
            return merged
        else:
            return time_file_map

    def _register_field(self, field_type: str, d: dict | None) -> None:
        """Validate and register a single field type from the source config.

        Populates ``var_dict`` with the variable list and ``file_dict`` with
        the time-ordered file map for ``field_type``.

        Args:
            field_type: One of ``"prognostic"``, ``"diagnostic"``, or
                ``"dynamic_forcing"``.
            d: Variable specification sub-dict from the config (keys ``path``
                and ``vars_2D``). Passing ``None`` or a non-dict value marks
                the field as absent (``file_dict[field_type] = None``).

        Raises:
            KeyError: If ``field_type`` is not in ``VALID_FIELD_TYPES``.
            ValueError: If ``d`` is a dict but ``vars_2D`` is empty or
                missing.
        """
        if field_type not in VALID_FIELD_TYPES:
            raise KeyError(
                f"Unknown field_type '{field_type}' in config['source']['GOES']. "
                f"Valid options are: {sorted(VALID_FIELD_TYPES)}"
            )
        if not isinstance(d, dict):
            self.file_dict[field_type] = None
            return

        if not d.get("vars_2D"):
            raise ValueError(f"Field '{field_type}' must define vars_2D")

        self.var_dict[field_type] = {"vars_2D": d.get("vars_2D") or []}

        base_dir = d.get("path", "")
        if self.mode == "local":
            self.file_dict[field_type] = self._collect_GOES_file_path(base_dir=base_dir)
        elif self.mode == "remote":
            self.file_dict[field_type] = self._collect_GOES_file_path()

    def _build_timestamps(self) -> pd.DatetimeIndex:
        """Construct the ``DatetimeIndex`` of valid input times.

        The index runs from ``start_datetime`` up to (but not including)
        ``end_datetime - num_forecast_steps * dt`` at intervals of ``dt``.

        Returns:
            A ``pandas.DatetimeIndex`` of valid input timestamps.
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
        """Load all 2-D variables for a field type at time ``t`` into ``sample``.

        Dispatches to ``_load_local_var`` or ``_load_remote_var`` depending on
        ``mode``, then stores each variable as a ``torch.Tensor`` of shape
        ``(1, 1, ny, nx)`` under the key
        ``"{goes_id}/{field_type}/2d/{vname}"`` in ``sample``. Does nothing if
        the field type has no registered variables.

        Args:
            field_type: One of ``"prognostic"``, ``"diagnostic"``, or
                ``"dynamic_forcing"``.
            t: Timestamp for which to load data.
            sample: Output dictionary that is updated in-place.
        """
        vd = self.var_dict.get(field_type)
        if not vd:
            return

        vnames = vd.get("vars_2D", [])
        if not vnames:
            return

        if self.mode == "remote":
            arrays = self._load_remote_var(field_type, vnames, t)
        elif self.mode == "local":
            arrays = self._load_local_var(field_type, vnames, t)
        else:
            return

        for vname, arr in arrays.items():
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sample[f"{self.goes_id}/{field_type}/2d/{vname}"] = tensor

    def _load_local_var(self, field_type: str, vnames: list[str], t: pd.Timestamp):
        """Load variables from a local NetCDF file and apply spatial cropping.

        Args:
            field_type: Field type used to look up the file map in
                ``file_dict``.
            vnames: Variable names to extract from the dataset.
            t: Timestamp used to locate the correct file via ``_find_file``.

        Returns:
            A dict mapping each variable name to its cropped ``numpy.ndarray``.

        Raises:
            KeyError: If no files are registered for ``field_type``.
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals:
            raise KeyError(
                f"No files registered for field_type '{field_type}'. Check that the path glob matches files on disk."
            )
        path = _find_file(file_intervals, t)
        with xr.open_dataset(path, engine="h5netcdf", chunks={}) as ds:
            sliced = ds[vnames].isel(y=self.y_slice, x=self.x_slice)
            return {v: sliced[v].values for v in vnames}

    def _load_remote_var(self, field_type: str, vnames: list[str], t: pd.Timestamp):
        """Load variables from a remote S3 NetCDF file and apply spatial cropping.

        Uses the cached ``_fs`` S3FileSystem to open the file as a byte stream
        and reads it with the ``h5netcdf`` engine.

        Args:
            field_type: Field type used to look up the file map in
                ``file_dict``.
            vnames: Variable names to extract from the dataset.
            t: Timestamp used to locate the correct file via ``_find_file``.

        Returns:
            A dict mapping each variable name to its cropped ``numpy.ndarray``.

        Raises:
            KeyError: If no files are registered for ``field_type``.
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals:
            raise KeyError(
                f"No files registered for field_type '{field_type}'. Check that the path glob matches files on disk."
            )

        path = _find_file(file_intervals, t)
        with xr.open_dataset(self._fs.open(path, "rb"), engine="h5netcdf", chunks={}) as ds:
            sliced = ds[vnames].isel(y=self.y_slice, x=self.x_slice)
            return {v: sliced[v].values for v in vnames}
