"""
goes.py
-------------------------------------------------------
GOESDataset: PyTorch Dataset for GOES data with nested input/target structure.

Sample structure returned by __getitem__::

    {
        "input":    {<user_provided_name>: {"<user_provided_name>/prognostic/2d/CMI_C04": tensor,
                                            "<user_provided_name>/prognostic/2d/CMI_C07": tensor}},
        "target":   {<user_provided_name>: {"<user_provided_name>/prognostic/2d/CMI_C04": tensor,
                                            "<user_provided_name>/prognostic/2d/CMI_C07": tensor}},  # only populated when return_target=True
        "metadata": {<user_provided_name>: {"input_datetime": int, "target_datetime": int}},
    }

All GOES variables are 2D. Tensor shape (no batch dimension)::

    (1, 1, lat, lon)   — singleton level dim, consistent with CREDIT Gen2 2D convention

After DataLoader collation the batch dimension is prepended::

    (batch, 1, 1, lat, lon)
"""

from __future__ import annotations

from typing import Any

import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

from credit.datasets._utils import _infer_period_freq, _find_file, _start_s3_fs
from credit.datasets.base_dataset import BaseDataset


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


class GOESDataset(BaseDataset):
    """PyTorch Dataset for GOES-R ABI Level-2 (L2) satellite imagery.

    Field types follow CREDIT Gen2 conventions: ``prognostic`` variables appear in
    both input (at step 0) and target; ``dynamic_forcing`` appears in input
    at every step; ``diagnostic`` appears in target only.  At step ``i > 0``
    the model's own prognostic predictions are fed back — no disk read occurs
    for prognostic fields at those steps.

    Supports loading directly from AWS S3 (remote mode) or from local
    NetCDF files (local mode). Spatial subsetting via ``extent``
    is applied at load time on the curvilinear GOES grid.

    See module docstring for full description of output format and file naming.

    Example YAML configuration (local mode)::

        data:
            source:
                Example_GOES:  # User-provided name (arbitrary key)
                    dataset_type: "goes"
                    goes_position: "east"  # or "west"
                    mode: "local"
                    product: "ABI-L2-MCMIPC"
                    variables:
                        prognostic:
                            vars_2D: ["CMI_C04", "CMI_C07", "CMI_C08", "CMI_C09", "CMI_C10", "CMI_C13"]
                            path: "/glade/derecho/scratch/kevinyang/datasets/goes/"
                        diagnostic: null
                        dynamic_forcing: null
                latlon2d_dir: "/glade/derecho/scratch/kevinyang/datasets/goes/"
                extent: [-130, -60, 20, 55]

            start_datetime: "2021-06-01"
            end_datetime: "2021-06-04"
            timestep: "6h"
            forecast_len: 0

    Example YAML configuration (remote mode)::

        data:
            source:
                Example_GOES:  # User-provided name (arbitrary key)
                    dataset_type: "goes"
                    goes_position: "east" # or "west"
                    mode: "remote"
                    product: "ABI-L2-MCMIPC"
                    variables:
                        prognostic:
                            vars_2D: ["CMI_C04", "CMI_C07", "CMI_C08", "CMI_C09", "CMI_C10", "CMI_C13"]
                        diagnostic: null
                        dynamic_forcing: null
                latlon2d_dir: "/glade/derecho/scratch/kevinyang/datasets/goes/"
                extent: [-130, -60, 20, 55]

    Args:
        config: Top-level experiment configuration dictionary. The relevant
            sub-keys are:

            - ``config["source"]["Example_GOES"]``: user-provided source name.

              - ``dataset_type`` (str): has to be "goes" to trigger this dataset class.
              - ``goes_position`` (str): Satellite position. One of ``"east"``, ``"west"``. Defaults to
                ``"east"``.
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
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """Initialize GOESDataset with config parsing, timestamp generation, file mapping from BaseDataset,
        then set GOES-specific attributes and spatial slice computation.

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return target variables. Defaults to False.

        Raises:
            FileNotFoundError: If the lat/lon grid NetCDF cannot be found under
                ``latlon2d_dir``.
        """
        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "goes", (
            f"Expected dataset_type 'goes' in config for GOESDataset, got '{self.curr_source_cfg['dataset_type']}'"
        )

        # Set GOES-specific attributes
        self.dataset_type = "goes"
        self.goes_position: str = self.curr_source_cfg.get("goes_position", "east")
        self.product: str = self.curr_source_cfg.get("product", "ABI-L2-MCMIPC")
        self.static_metadata: dict[str, Any] = {"datetime_fmt": "unix_ns"}
        self.qc_path: str = self.curr_source_cfg.get("qc_path", None)
        self.tolerance = pd.Timedelta("3 minutes")  # to allow for searching the nearest GOES observation time

        # Initialize the field registration based on the provided config and populate
        #   dictionary of variables and file paths for each field type
        self.init_register_all_fields()

        # Initialize the s3fs on the first call to _extract_field within __getitem__
        self._fs = None

        # Pre-compute spatial slices from GOES fixed lat/lon grids
        self.latlon2d_dir: str = self.curr_source_cfg.get("latlon2d_dir", "")

        if self.goes_position == "east":
            prefix = "goes19"
        elif self.goes_position == "west":
            prefix = "goes18"

        if self.product[-1] == "C":
            suffix = "abi_conus_lat_lon.nc"
        elif self.product[-1] == "F":
            suffix = "abi_full_disk_lat_lon.nc"

        self.extent = self.curr_source_cfg.get("extent", None)
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
            if (
                self.goes_position == "east"
            ):  # GOES-16 was replaced by GOES-19 at 15:10 UTC on April 7, 2025; https://www.ospo.noaa.gov/data/messages/2025/04/MSG_20250407_1510.html
                if dt < pd.Timestamp("2025-04-07 15:00:00"):
                    goes_id = "goes16"
                elif dt >= pd.Timestamp("2025-04-07 16:00:00"):
                    goes_id = "goes19"
                else:  # discard the hour containing the transition since it may contain files from both satellites and cause ambiguity in file mapping
                    print(
                        f"[WARN] Discarding observations in {dt} due to GOES-16/19 transition on 2025-04-07 15:10 UTC to avoid file mapping ambiguity during the hour."
                    )
                    continue
            elif (
                self.goes_position == "west"
            ):  # GOES-17 was replaced by GOES-18 at 18:00 UTC on January 4, 2023; https://www.ospo.noaa.gov/data/messages/2023/01/MSG_20230104_1805.html
                if dt < pd.Timestamp("2023-01-04 18:00:00"):
                    goes_id = "goes17"
                elif dt >= pd.Timestamp("2023-01-04 19:00:00"):
                    goes_id = "goes18"
                else:  # discard the hour containing the transition since it may contain files from both satellites and cause ambiguity in file mapping
                    print(
                        f"[WARN] Discarding observations in {dt} due to GOES-17/18 transition on 2023-01-04 18:00 UTC to avoid file mapping ambiguity during the hour."
                    )
                    continue

            rel_path = os.path.join(
                f"noaa-{goes_id}/{self.product}",
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
        base_dir = field_config.get("path", "")
        if self.mode == "local":
            return self._collect_GOES_file_path(base_dir=base_dir)
        elif self.mode == "remote":
            return self._collect_GOES_file_path()

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
        ``"{source_name}/{field_type}/2d/{vname}"`` in ``sample``. Does nothing if
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
            if self._fs is None:
                self._fs = _start_s3_fs()
            arrays = self._load_remote_var(field_type, vnames, t)
        elif self.mode == "local":
            arrays = self._load_local_var(field_type, vnames, t)
        else:
            return

        for vname, arr in arrays.items():
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            key = self._get_field_name(field_type, "2d", vname)
            sample[key] = tensor

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
