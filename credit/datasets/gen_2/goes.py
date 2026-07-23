"""
goes.py
-------------------------------------------------------
GOESDataset: PyTorch Dataset for GOES data with nested input/target structure.

Sample structure returned by __getitem__ (GOESDataset does not override this method;
see BaseDataset._load_sample for the implementation). Note this is the per-source
structure — when wrapped by MultiSourceDataset, an additional layer keyed by
<user_provided_name> is added around each of "input"/"target"/"metadata":

    {
        "input":    {"<user_provided_name>/prognostic/2d/CMI_C04": tensor,
                     "<user_provided_name>/prognostic/2d/CMI_C07": tensor},
        "target":   {"<user_provided_name>/prognostic/2d/CMI_C04": tensor,
                     "<user_provided_name>/prognostic/2d/CMI_C07": tensor},  # only populated when return_target=True
        "metadata": {"input_datetime": int, "target_datetime": int},
    }

All GOES variables are 2D. Tensor shape (no batch dimension):
    (level, time, lat, lon) = (1, 1, lat, lon)
    — level is singleton since GOES has no vertical levels; time is singleton
    since each sample covers a single timestep. Consistent with CREDIT Gen2
    convention, where 3D variables instead have shape (n_levels, time, lat, lon)
    (see e.g. ``credit/datasets/gen_2/era5.py``).

After DataLoader collation the batch dimension is prepended:
    (batch, level, time, lat, lon) = (batch, 1, 1, lat, lon)

Key features:
    * **I/O**: local (NetCDF) or remote (public, no-auth AWS S3) loading via
      ``mode``.
    * **Catalogs**: a pre-built JSON catalog (``file_catalog_path``, from
      ``quality_check_goes.py``) skips the directory/S3 scan and records
      per-timestamp availability -- ``MISSING``, ``QC_MASKED``, ``SKIP``,
      dropped from ``self.datetimes`` (see ``_filter_unavailable_timestamps``);
      ``SKIP`` doubles as a hook for custom sampling on top of (never instead
      of) QC. Catalogs can be merged, and reused for a narrower time window, a
      smaller ``extent``, or a subset of QC'd ``variables`` than they were
      built with (see ``_extent_covers``, ``_variables_covers``) -- gated by
      five "invariant" config fields shared between the writer and its
      readers (see ``catalog_invariant_metadata``).
    * **Spatial**: ``extent``-based cropping (bbox or NW/SE corners) resolved
      against GOES's curvilinear grid via nearest-neighbour search (see
      ``_build_spatial_slices``); CONUS vs. full-disk products use different
      precomputed lat/lon grids under ``latlon2d_dir``.
    * **Temporal**: only a single contiguous ``start_datetime``-``end_datetime``
      window is supported (random subsampling, if needed, happens one level up
      via the trainer's ``batches_per_epoch``). Multi-step (``forecast_len`` >
      1) rollout validates that every target step is available, not just the
      input timestamp. GOES-16->19 (east) / GOES-17->18 (west) satellite
      transitions are detected and their ambiguous hour dropped automatically.
    * **Data model**: field types follow the CREDIT Gen2 convention --
      ``prognostic`` in input and target, ``dynamic_forcing`` in input every
      step, ``diagnostic`` in target only, ``static`` in input only (never
      target); rollout feeds back the model's own prognostic predictions past
      step 0, with no disk read. Sample keys are
      ``"{source_name}/{field_type}/2d/{variable}"``; tensors are ``float32``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from credit.datasets.gen_2._utils import _infer_period_freq, _find_file, _start_s3_fs
from credit.datasets.gen_2.base_dataset import BaseDataset
from credit.datasets.gen_2.grid_utils import write_source_grid_schema_if_missing

logger = logging.getLogger(__name__)

# no file found within the tolerance for this timestamp during the scan
_FILE_NOT_FOUND = "MISSING"
# file exists, but failed QC (written by quality_check_goes.py, which lives in a
# different package and is not part of CREDIT)
_FILE_QC_MASKED = "QC_MASKED"
# timestamp manually excluded (e.g., special sampling)
_FILE_SKIP = "SKIP"
# valid flags; anything else is rejected as a typo
_KNOWN_FLAGS = frozenset({_FILE_NOT_FOUND, _FILE_QC_MASKED, _FILE_SKIP})


# Module-level helper functions below are ordered helper-before-caller: a function that
# depends on another is defined after the one it depends on, reading top-to-bottom.


# --- Catalog row validation ---


def _validate_catalog_row_path(path: str, t: "pd.Timestamp") -> None:
    """Raise a ValueError if *path* looks like a mistyped flag (sentinel) value.

    Real file paths always contain a '/' (local path or s3://) and end in '.nc'.
    A bare token that matches neither is presumed to be an attempted flag —
    if it's not one of the three recognized ones, it's almost certainly a typo.
    """
    if "/" not in path and not path.endswith(".nc") and path not in _KNOWN_FLAGS:
        raise ValueError(
            f"Unrecognized flag '{path}' in catalog row at {t}. "
            f"Expected a file path, or one of: {sorted(_KNOWN_FLAGS)}."
        )


# --- Spatial grid helpers ---


def _find_nearest_latlon(lat2d: np.ndarray, lon2d: np.ndarray, lat_target: float, lon_target: float) -> tuple[int, int]:
    """Find the 2-D grid indices of the point nearest to a target lat/lon using Haversine distance.

    Args:
        lat2d: 2-D array of latitudes in decimal degrees.
        lon2d: 2-D array of longitudes in decimal degrees.
        lat_target: Target latitude in decimal degrees. Valid range for latitude: ``[-90, 90]``.
        lon_target: Target longitude in decimal degrees. Valid range for longitude: ``[-180, 180]``.

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
    angular_dist = np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    idx_flat = np.nanargmin(angular_dist)
    i, j = np.unravel_index(idx_flat, lat2d.shape)

    return i, j


def _build_spatial_slices(
    extent: list[float] | dict | None,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
) -> tuple[slice, slice]:
    """Compute row (latitude) and column (longitude) slices that bound a geographic extent on a 2-D grid.

    Given an optional bounding box in geographic coordinates, returns a pair of
    ``slice`` objects that can be passed directly to ``xarray.Dataset.isel`` (or
    plain NumPy slicing) to crop a 2-D field to the requested region.

    Args:
        extent: One of three forms:

            * ``None`` — select the entire grid (both slices become
              ``slice(None)``).
            * ``[lon_min, lon_max, lat_min, lat_max]`` — rectangular bounding
              box in decimal degrees.  Four synthetic corners are evaluated via
              nearest-neighbour search.  Works well for compact, locally
              rectilinear regions but may overshoot on large curved grids (e.g.
              full-CONUS GOES).
            * ``{"nw": [lat, lon], "se": [lat, lon]}`` — explicit NW and SE
              corner points.  Only two nearest-neighbour lookups are performed,
              one per anchor, so the resulting slice is exact for any grid
              projection.

        lat2d: 2-D array of latitudes (degrees) with the same shape as the
            target grid. Required when ``extent`` is not ``None``.
        lon2d: 2-D array of longitudes (degrees) with the same shape as the
            target grid. Required when ``extent`` is not ``None``.

    Returns:
        A ``(y_slice, x_slice)`` tuple where ``y_slice`` indexes rows
        (latitude axis) and ``x_slice`` indexes columns (longitude axis) of
        the 2-D grid.

    Raises:
        ValueError: If ``extent`` is not ``None`` but ``lat2d`` or ``lon2d``
            are ``None``, or if a dict extent is missing ``"nw"`` or ``"se"``.
        TypeError: If ``extent`` is not ``None``, a list, or a dict.
    """
    if extent is None:
        y_slice = slice(None)
        x_slice = slice(None)

    elif isinstance(extent, dict):
        if lat2d is None or lon2d is None:
            raise ValueError(
                "A geographic extent requires lat2d and lon2d; pass 2-D coordinate arrays or set extent=None."
            )
        if "nw" not in extent or "se" not in extent:
            raise ValueError("Dict extent must have 'nw' and 'se' keys, each a [lat, lon] pair.")

        lat_nw, lon_nw = extent["nw"]
        lat_se, lon_se = extent["se"]

        i_nw, j_nw = _find_nearest_latlon(lat2d, lon2d, lat_nw, lon_nw)
        i_se, j_se = _find_nearest_latlon(lat2d, lon2d, lat_se, lon_se)

        y_slice = slice(min(i_nw, i_se), max(i_nw, i_se) + 1)
        x_slice = slice(min(j_nw, j_se), max(j_nw, j_se) + 1)

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


# --- Extent coverage comparison ---


def _extent_to_bbox(extent: list[float] | dict | None) -> tuple[float, float, float, float] | None:
    """Convert an ``extent`` value to a ``(lat_min, lat_max, lon_min, lon_max)`` bounding box.

    Returns ``None`` for ``extent=None`` (the unrestricted full-grid case). This is a plain
    numeric-bounds comparison — it does not consult the lat/lon grid arrays, so it ignores
    grid curvature (see ``_extent_covers``).

    Raises:
        ValueError: If a dict extent is missing ``"nw"`` or ``"se"``.
        TypeError: If ``extent`` is not ``None``, a list, or a dict.
    """
    if extent is None:
        return None
    if isinstance(extent, dict):
        if "nw" not in extent or "se" not in extent:
            raise ValueError(f"Dict extent must have 'nw' and 'se' keys, each a [lat, lon] pair. Got: {extent}")
        lat_nw, lon_nw = extent["nw"]
        lat_se, lon_se = extent["se"]
        return (min(lat_nw, lat_se), max(lat_nw, lat_se), min(lon_nw, lon_se), max(lon_nw, lon_se))
    if isinstance(extent, list):
        lon_min, lon_max, lat_min, lat_max = extent
        return (lat_min, lat_max, lon_min, lon_max)
    raise TypeError(f"Unsupported extent type: {type(extent)}")


# Fixed safety margin (degrees) for _extent_covers. A strictly-smaller (non-identical) request
# must be inset from the catalog's bounds by at least this much, not just numerically within
# them — see _extent_covers for why a plain boundary comparison isn't safe on a curvilinear grid.
# 0.5 deg is ~25 grid cells at CONUS/nadir resolution (~0.02 deg/cell) and ~5 grid cells at
# full-disk/near-limb resolution (~0.1 deg/cell, where resolution degrades with viewing angle) —
# comfortably above the ~2-3 cell nearest-neighbour snap error this margin guards against, at
# either scale. Does not apply to an exact-match request; only to a strictly smaller one.
_EXTENT_MARGIN_DEG = 0.5


def _extent_covers(catalog_extent: list[float] | dict | None, request_extent: list[float] | dict | None) -> bool:
    """Return whether *catalog_extent* geographically covers *request_extent*.

    A catalog's QC guarantees only hold within the extent it was built with
    (``quality_check_goes.py`` crops to ``extent`` before running its QC check).
    Requesting a **subdomain** of that extent (or the exact same extent) is
    safe to reuse; requesting anything **larger** is not, since the catalog
    never checked the extra region.

    Because this is a plain numeric lat/lon comparison, not a check against the
    actual (curvilinear) grid, a request extent that's only *barely* inside the
    catalog's extent isn't necessarily safe: nearest-neighbour snapping in
    ``_build_spatial_slices`` can round outward past what the catalog actually
    scanned. To guard against that, a non-identical (strictly smaller) request
    must be inset from the catalog's bounds by ``_EXTENT_MARGIN_DEG`` degrees,
    not just numerically within it. An exact-match request is always accepted
    regardless of margin, since identical corners snap to identical grid
    indices — there's nothing to guard against there. If a request is rejected
    only because it's too close to the catalog's boundary, consider using
    ``extent: null`` (the full grid) instead of a tightly-fitting subdomain.

    This function only handles the ``extent`` comparison. The caller
    (``_load_file_catalog``) also checks ``mode``, ``timestep``, ``product``, and
    ``goes_position`` for exact equality — and checks those *before* calling this
    function, so a mismatch on any of them (e.g. a different ``product``) fails
    fast without this function ever running. ``extent`` is the only one of
    these five config-invariant fields with this subdomain-vs-superset
    relaxation; see ``catalog_invariant_metadata``. ``variables`` gets a
    similar (but separate) relaxation — see ``_variables_covers``, used by
    ``_validate_catalog_variables`` rather than by this method's caller.

    Args:
        catalog_extent: The extent the catalog was built with
            (its ``metadata["extent"]``).
        request_extent: The extent of the dataset currently being
            constructed (``self.extent``).

    Returns:
        ``True`` if *request_extent* equals *catalog_extent*, or is contained
        within it with at least ``_EXTENT_MARGIN_DEG`` of margin (or
        *catalog_extent* is ``None``, i.e. the full grid). ``False`` otherwise.
    """
    catalog_bbox = _extent_to_bbox(catalog_extent)
    if catalog_bbox is None:
        return True  # catalog covers the full grid — any request is a subdomain of it

    request_bbox = _extent_to_bbox(request_extent)
    if request_bbox is None:
        return False  # request wants the full grid, but the catalog only checked a sub-region

    if request_bbox == catalog_bbox:
        return True  # identical extent — same corners snap to the same grid indices, no margin needed

    cat_lat_min, cat_lat_max, cat_lon_min, cat_lon_max = catalog_bbox
    req_lat_min, req_lat_max, req_lon_min, req_lon_max = request_bbox
    return (
        req_lat_min >= cat_lat_min + _EXTENT_MARGIN_DEG
        and req_lat_max <= cat_lat_max - _EXTENT_MARGIN_DEG
        and req_lon_min >= cat_lon_min + _EXTENT_MARGIN_DEG
        and req_lon_max <= cat_lon_max - _EXTENT_MARGIN_DEG
    )


# --- Variable coverage comparison ---


def _variables_covers(catalog_variables: dict | None, request_var_dict: dict) -> bool:
    """Return whether *catalog_variables* covers every channel *request_var_dict* needs.

    A catalog's QC guarantees only hold for the channels it actually checked
    (``quality_check_goes.py`` collects ``vars_2D``/``vars_3D`` across all
    active field types and runs the QC test on exactly those). Requesting a
    **subset** of the channels the catalog checked for a given field type is
    safe to reuse — the catalog's QC still covers that subset. Requesting a
    field type the catalog never checked at all, or a channel within a
    checked field type that the catalog didn't include, is not safe — the
    catalog's QC guarantee never covered it.

    Args:
        catalog_variables: The catalog's ``metadata["variables"]`` — a dict
            mapping field type to ``{"vars_3D": [...], "vars_2D": [...]}``,
            or ``None`` if the catalog's JSON predates this metadata field.
        request_var_dict: The dataset's own ``var_dict`` for the current
            config, same shape as *catalog_variables*.

    Returns:
        ``True`` if every field type and channel in *request_var_dict* is
        present in the corresponding entry of *catalog_variables*. ``False``
        otherwise (including when *catalog_variables* is ``None``, since
        there is then nothing to verify coverage against).
    """
    if catalog_variables is None:
        return False
    for field_type, spec in request_var_dict.items():
        catalog_spec = catalog_variables.get(field_type)
        if catalog_spec is None:
            return False
        for dim in ("vars_3D", "vars_2D"):
            requested = set(spec.get(dim) or [])
            checked = set(catalog_spec.get(dim) or [])
            if not requested.issubset(checked):
                return False
    return True


class GOESDataset(BaseDataset):
    """PyTorch Dataset for GOES-R ABI Level-2 (L2) satellite imagery.

    Field types follow CREDIT Gen2 conventions: ``prognostic`` variables appear in
    both input (at step 0) and target; ``dynamic_forcing`` appears in input
    at every step; ``diagnostic`` appears in target only; ``static`` appears in
    input at step 0 only, same timing as ``prognostic``, but never in target.
    At step ``i > 0`` the model's own prognostic predictions are fed back — no
    disk read occurs for prognostic fields at those steps.

    Supports loading directly from AWS S3 (remote mode) or from local
    NetCDF files (local mode). Spatial subsetting via ``extent``
    is applied at load time on the curvilinear GOES grid.

    See module docstring for full description of output format and file naming.

    GOES imager projection background (for deriving the ``latlon2d_dir`` grids):
    https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data_goes_imager_projection.php

    Example YAML configuration (remote/S3 mode):

        data:
            source:
                Example_GOES:  # user-provided name (arbitrary key)
                    dataset_type: "goes"
                    goes_position: "east"        # "east" (GOES-16/19) or "west" (GOES-17/18);
                                                 # satellite transitions are handled automatically
                    mode: "remote"               # streams directly from AWS S3 (public, no auth required)
                    product: "ABI-L2-MCMIPC"    # CONUS; use "ABI-L2-MCMIPF" for full disk
                    variables:
                        prognostic:
                            vars_2D: ["CMI_C04", "CMI_C07", "CMI_C08", "CMI_C09", "CMI_C10", "CMI_C13"]
                        diagnostic: null
                        dynamic_forcing: null
                    latlon2d_dir: "/glade/derecho/scratch/kevinyang/datasets/goes/"
                    # Three extent forms (pick one):
                    extent: {nw: [55, -130], se: [20, -60]}      # explicit NW/SE corners (more precise)
                    # extent: [-130, -60, 20, 55]                # [lon_min, lon_max, lat_min, lat_max]
                    # extent: null                                # no crop — load the full grid
                    # this catalog file is generated by quality_check_goes.py
                    file_catalog_path: "/path/to/goes_catalog_*.json"  # recommended for remote; avoids S3 listing
                    # scan_tolerance: "3 minutes"  # optional; max gap between requested time and nearest file

            start_datetime: "2021-06-01"
            end_datetime: "2021-06-04"
            timestep: "6h"
            forecast_len: 1  # 1 = single-step training

    For local mode, the same config applies with these differences: set
    ``mode: "local"``; add a ``path`` key under ``variables.prognostic``
    pointing to the local NetCDF directory to scan; ``file_catalog_path`` is
    optional rather than recommended, since a local directory scan is cheap.

    Args:
        data_config: Top-level experiment configuration dictionary. The relevant
            sub-keys are:

            - ``config["source"]["Example_GOES"]``: user-provided source name.

              - ``dataset_type`` (str): has to be "goes" to trigger this dataset class.
              - ``goes_position`` (str): Satellite position. One of ``"east"``, ``"west"``. Defaults to
                ``"east"``.
              - ``mode`` (str): ``"local"`` or ``"remote"`` (S3). Defaults to
                ``"local"``.
              - ``product`` (str): ABI product string, e.g.
                ``"ABI-L2-MCMIPC"``.
              - ``extent`` (list or dict, optional): Spatial crop. Either
                ``[lon_min, lon_max, lat_min, lat_max]`` or
                ``{"nw": [lat, lon], "se": [lat, lon]}``.
              - ``latlon2d_dir`` (str): Directory containing pre-computed
                lat/lon grid NetCDF files for GOES's curvilinear (satellite
                projection) grid (see class docstring above for background
                and how to derive these).
              - ``file_catalog_path`` (str, optional): Path or glob pattern to
                a pre-built JSON file catalog. When matched, skips the
                directory scan entirely. A catalog covering a wide time range
                (e.g. a full year, built once via ``quality_check_goes.py``)
                can be reused across many experiments — it only needs to be
                regenerated when ``mode``, ``timestep``, ``product``, or
                ``goes_position`` change. For ``extent``: an exact match
                always reuses the catalog; a *smaller* extent reuses it too,
                but only if inset from the catalog's bounds by a safety
                margin (see ``_EXTENT_MARGIN_DEG``) — too-close-to-boundary or
                *larger* requests are rejected, since the catalog's QC never
                checked outside (or reliably near the edge of) its own extent
                (see ``_extent_covers``).
                To train on a subset of a catalog's range, do **not** trim the
                catalog itself; instead narrow ``config["start_datetime"]`` /
                ``config["end_datetime"]`` (below). ``_build_timestamps``
                (see ``BaseDataset``) derives the actual sample pool from
                those two values, and ``_load_file_catalog`` only looks up
                rows that fall inside them — the catalog's full range does not
                have to match the training window. The dataset itself has no
                stride/random-subsample option — only a single contiguous
                ``start_datetime``–``end_datetime`` window. A form of random
                subsampling can still happen one level up, at the trainer: if
                ``trainer.batches_per_epoch`` is set smaller than a full
                epoch's batch count, the sampler (which reshuffles every
                epoch via ``set_epoch``) only draws that many batches, so
                each epoch trains on a random subset of the full window —
                but without direct control over which timestamps that is.

                Each catalog row's file path may instead be one of three
                flag (sentinel) values, causing that timestamp to be dropped from
                ``self.datetimes`` (see ``_filter_unavailable_timestamps``):

                * ``"MISSING"`` — no GOES file was found within
                  ``scan_tolerance`` during the scan (written automatically).
                * ``"QC_MASKED"`` — the file failed QC, or failed to open at
                  all (written automatically by ``quality_check_goes.py``).
                * ``"SKIP"`` — manually added by hand-editing the catalog
                  JSON (e.g. to exclude a timestamp for a reason outside the
                  automated QC check). Not written by any script — if you
                  regenerate a catalog from a fresh scan, any ``"SKIP"`` rows
                  you'd added are lost, since the scan has no way to know
                  about them. This can also be repurposed deliberately: an
                  application with its own strategic sampling logic can
                  assign ``"SKIP"`` to exactly the timestamps it wants left
                  out, on top of (never instead of) running QC first — QC
                  should always run before any such additional sampling.
              - ``scan_tolerance`` (str, optional): Maximum time difference
                between a requested timestamp and the nearest GOES file.
                Accepts any ``pandas.Timedelta``-parseable string (e.g.
                ``"5 minutes"``). Defaults to ``"3 minutes"``.
              - ``variables`` (dict): Mapping of field_type to variable spec.

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
            ``{"vars_3D": [], "vars_2D": [<variable names>]}``. GOES fields
            are always 2D, so ``vars_3D`` is always empty.
        y_slice (slice): Row crop derived from ``extent`` (or ``slice(None)``
            for the full grid).
        x_slice (slice): Column crop derived from ``extent`` (or
            ``slice(None)`` for the full grid).

    Raises:
        ValueError: If ``dataset_type`` is not ``"goes"``, if ``goes_position``
            is not ``"east"`` or ``"west"``, if ``product`` does not end in
            ``"C"`` (CONUS) or ``"F"`` (full disk), or if ``mode`` is not
            ``"local"`` or ``"remote"``.
        FileNotFoundError: If the lat/lon grid NetCDF cannot be found under
            ``latlon2d_dir``.

        See also ``init_register_all_fields`` and ``_register_field`` (both
        in ``BaseDataset``), and this class's own ``_load_file_catalog``, for
        errors raised during field registration and catalog loading.
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """Initialize GOESDataset.

        Calls ``super().__init__()`` for config parsing and timestamp
        generation, then sets GOES-specific attributes (``goes_position``,
        ``product``, ``extent``, etc.), then explicitly calls
        ``self.init_register_all_fields()`` to build the file mapping (unlike
        ``BaseDataset``, which only does this automatically when it is not
        subclassed). ``self._validate_catalog_variables()`` and
        ``self._filter_unavailable_timestamps()`` then run to drop any
        ``MISSING``/``QC_MASKED``/``SKIP`` timestamps from ``self.datetimes``,
        and finally the spatial crop slices are computed.

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return target variables. Defaults to False.

        Raises:
            ValueError: If ``data_config``'s ``dataset_type`` is not ``"goes"``,
                if ``goes_position`` is not ``"east"`` or ``"west"``, if
                ``product`` does not end in ``"C"`` or ``"F"``, or if ``mode``
                is not ``"local"`` or ``"remote"``.
            FileNotFoundError: If the lat/lon grid NetCDF cannot be found under
                ``latlon2d_dir``.
        """
        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)
        if self.curr_source_cfg["dataset_type"] != "goes":
            raise ValueError(
                f"Expected dataset_type 'goes' in config for GOESDataset, got '{self.curr_source_cfg['dataset_type']}'"
            )
        # Validate self.mode (set by BaseDataset.__init__) once here, rather than at every
        # call site — _get_file_source and _extract_field can then assume it's always valid.
        if self.mode not in ("local", "remote"):
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'local' or 'remote'.")
        # Set GOES-specific attributes
        self.dataset_type = "goes"
        self.goes_position: str = self.curr_source_cfg.get("goes_position", "east")
        self.product: str = self.curr_source_cfg.get("product", "ABI-L2-MCMIPC")
        if self.goes_position not in ("east", "west"):
            raise ValueError(f"goes_position must be 'east' or 'west', got '{self.goes_position}'.")
        if self.product[-1] not in ("C", "F"):
            raise ValueError(
                f"Unrecognized product '{self.product}': last character must be 'C' (CONUS) or 'F' (full disk)."
            )
        self.static_metadata: dict[str, Any] = {"datetime_fmt": "unix_ns"}
        self.file_catalog_path: str = self.curr_source_cfg.get("file_catalog_path", None)
        self.tolerance = pd.Timedelta(self.curr_source_cfg.get("scan_tolerance", "3 minutes"))
        # Must be set before init_register_all_fields() since _load_file_catalog validates it
        self.extent = self.curr_source_cfg.get("extent", None)
        # Cache for _load_file_catalog(): init_register_all_fields() calls _get_file_source
        # once per active field type, and every field type shares identical catalog rows
        # (GOES stores all channels in one file per timestamp), so only the first call
        # should actually read/parse/validate the JSON.
        self._catalog_cache: list[tuple] | None = None
        # Populated by _load_file_catalog() with the catalog's "variables" metadata (if a
        # catalog is loaded). Compared against var_dict in _validate_catalog_variables()
        # once var_dict is fully built, since it isn't complete until every field type
        # has been registered — see that method's docstring.
        self._catalog_variables_meta: dict[str, Any] | None = None
        # Lazily initialized on first remote use, by either _collect_GOES_file_path
        # (directory-listing scan, during field registration below) or _extract_field
        # (per-sample reads, during __getitem__) — whichever runs first. Set here, before
        # init_register_all_fields(), so both share one cached connection instead of
        # each opening its own.
        self._fs = None

        # Initialize the field registration based on the provided config and populate
        #   dictionary of variables and file paths for each field type
        self.init_register_all_fields()
        self._validate_catalog_variables()

        # Rebuild datetimes to exclude timestamps with missing, QC-masked, or
        # skipped files so that MultiSourceDataset's master clock naturally
        # drops those timestamps.
        self.datetimes = self._filter_unavailable_timestamps()

        # Pre-compute spatial slices from GOES fixed lat/lon curvilinear grids
        self.latlon2d_dir: str = self.curr_source_cfg.get("latlon2d_dir", "")

        if self.goes_position == "east":
            prefix = "goes19"
        elif self.goes_position == "west":
            prefix = "goes18"

        if self.product[-1] == "C":
            suffix = "abi_conus_lat_lon.nc"
        elif self.product[-1] == "F":
            suffix = "abi_full_disk_lat_lon.nc"

        latlon2d_path = os.path.join(self.latlon2d_dir, f"{prefix}_{suffix}")

        try:
            with xr.open_dataset(latlon2d_path) as ds:
                lat2d = ds.latitude.values
                lon2d = ds.longitude.values
                self.y_slice, self.x_slice = _build_spatial_slices(self.extent, lat2d, lon2d)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Latitude/longitude grid file not found at {latlon2d_path}") from e

        # Debugging aid: this source's native (post-extent-crop) grid. Not necessarily
        # the grid actually written to output — see credit.datasets.gen_2.grid_utils.
        grid = {
            "grid_type": "curvilinear",
            "lat": lat2d[self.y_slice, self.x_slice],
            "lon": lon2d[self.y_slice, self.x_slice],
        }
        self.static_metadata["grid"] = grid
        write_source_grid_schema_if_missing(self.curr_source_name, grid, self.save_loc)

    # ---------------
    # Catalog & timestamp validation
    # ---------------

    def _filter_unavailable_timestamps(self) -> pd.DatetimeIndex:
        """Return ``self.datetimes`` with missing, QC-masked, and skipped timestamps removed.

        Scans ``file_dict`` (populated by ``init_register_all_fields``) and
        collects interval start times whose path is ``_FILE_NOT_FOUND``,
        ``_FILE_QC_MASKED``, or ``_FILE_SKIP``, then excludes them from the
        current datetimes. Also excludes any input timestamp ``t`` whose
        forecast target steps are unavailable — if ``t + k*dt`` is bad for
        any ``k`` in ``1..num_forecast_steps``, ``t`` cannot be used either.
        """
        unavailable = set()
        n_missing, n_qc, n_skip = 0, 0, 0
        for intervals in self.file_dict.values():
            if intervals is None:
                continue
            for start, end, path in intervals:
                if path == _FILE_NOT_FOUND:
                    unavailable.add(start)
                    n_missing += 1
                elif path == _FILE_QC_MASKED:
                    unavailable.add(start)
                    n_qc += 1
                elif path == _FILE_SKIP:
                    unavailable.add(start)
                    n_skip += 1

        # Also exclude input timestamps whose target steps are unavailable:
        # if t+k*dt is bad for any k in 1..num_forecast_steps, t cannot be used.
        dt = self.datetimes.freq
        kept = [
            t
            for t in self.datetimes
            if t not in unavailable
            and all(t + k * dt not in unavailable for k in range(1, self.num_forecast_steps + 1))
        ]
        filtered = pd.DatetimeIndex(kept)
        logger.info(
            "Timestamps — total: %d, missing: %d, QC-masked: %d, skipped: %d, kept: %d",
            len(self.datetimes),
            n_missing,
            n_qc,
            n_skip,
            len(filtered),
        )
        return filtered

    def catalog_invariant_metadata(self) -> dict[str, Any]:
        """The five config fields relevant to catalog compatibility for this dataset.

        Notably excludes ``variables`` (the requested channel lists), even
        though that's also part of what makes a catalog compatible — see
        ``_validate_catalog_variables`` for that separate check, and why it's
        separate: these five fields are all known immediately (read once from
        config, never change), so they can be checked as soon as a catalog is
        loaded. ``variables`` (``self.var_dict``) is instead built up
        incrementally, one field type at a time, and a catalog gets loaded
        *in the middle of* that process — checking ``variables`` here would
        risk comparing against a still-incomplete list. It's checked
        separately, once that process finishes.

        This dict is the single definition of "does a catalog match my config,"
        shared by one writer and two readers so they can't drift out of sync:

        * Written into a new catalog's JSON ``"metadata"`` by
          ``quality_check_goes.py`` when it builds one.
        * Read by this dataset's own ``_load_file_catalog`` to decide whether
          an *existing* catalog is safe to load for training — does it cover
          what's being requested?
        * Read by ``quality_check_goes.py``'s pre-overwrite check to decide
          whether it's about to clobber an existing catalog that's already
          identical — is this the *same* catalog it'd be regenerating? This
          guards against silently losing hand-added ``"SKIP"`` rows.

        ``mode``, ``timestep``, ``product``, and ``goes_position`` must match
        a catalog exactly in all three contexts above. ``extent`` is looser
        only for the load check (see ``_load_file_catalog``/``_extent_covers``):
        the catalog's extent only needs to cover the requested extent (equal
        or a superset), since a smaller request is just a subdomain of what
        was already QC'd. The pre-overwrite check still compares ``extent``
        for exact equality, since "is this the same catalog" is a different
        question than "does this catalog cover my request."
        """
        return {
            "mode": self.mode,
            "timestep": str(self.dt),
            "product": self.product,
            "goes_position": self.goes_position,
            "extent": self.extent,
        }

    def _validate_catalog_variables(self) -> None:
        """Validate the loaded catalog's ``variables`` metadata against ``var_dict``.

        Must be called after ``init_register_all_fields()`` completes, since
        ``var_dict`` is only fully populated once every field type has been
        registered (``_load_file_catalog()`` runs mid-registration, when
        ``var_dict`` may still be partial, so it cannot do this check itself).

        The catalog's ``variables`` metadata only needs to *cover* the
        current config's channels (see ``_variables_covers``), not match
        them exactly — a catalog built checking a superset of channels (e.g.
        all GOES bands) safely covers a config that only requests a subset
        of them. A failure means the catalog's QC (which only checked the
        channels listed in its ``variables`` metadata) does not cover some
        channel the current config actually requests, so its QC-masking
        guarantee doesn't hold for that channel and the catalog must not be
        used silently.
        """
        if self._catalog_variables_meta is None:
            # Either no catalog was loaded for this dataset (e.g. no file_catalog_path
            # set), or one was loaded but its JSON predates the "variables" metadata
            # field. Either way, there's nothing to check coverage against.
            return
        if not _variables_covers(self._catalog_variables_meta, self.var_dict):
            raise ValueError(
                f"File catalog variables mismatch: catalog was built with variables "
                f"'{self._catalog_variables_meta}', which does not cover the current "
                f"config's variables '{self.var_dict}'. The catalog's quality checks "
                f"do not cover this set of channels. Re-run quality_check_goes.py to "
                f"regenerate the catalog."
            )

    # ---------------
    # File discovery & mapping (runs once at __init__, via init_register_all_fields)
    # ---------------

    def _load_file_catalog(self) -> list[tuple]:
        """Load and validate pre-built file catalogs matched by ``self.file_catalog_path``.

        ``file_catalog_path`` may be an exact path or a glob pattern (e.g.
        ``/path/to/goes_catalog_*.json``).  All matching files are loaded,
        their config-invariant metadata fields validated, their time ranges
        checked for overlaps, and their rows merged into a single sorted list.

        Returns:
            A list of ``(period_start, period_end, file_path)`` tuples ready
            to be used as a file map.

        Raises:
            FileNotFoundError: If no files match ``self.file_catalog_path``.
            ValueError: If any of the following occur:

                * A catalog's config-invariant metadata does not match the
                  current config — for ``extent`` this means the catalog's
                  extent does not cover the requested extent (see
                  ``_extent_covers``); a request for a subdomain of the
                  catalog's extent is accepted.
                * Matched catalog files disagree with each other on
                  ``variables`` metadata (they must all have been built from
                  the same config to be safely merged).
                * Two matched catalog files have overlapping time ranges.
                * The merged catalog is missing a row for any timestamp in
                  the full requested ``start_datetime``–``end_datetime`` range
                  (including target timestamps for the last forecast step;
                  this also catches a gap between non-overlapping catalog
                  files, not just insufficient coverage at either end).
                * A row's path is an unrecognized flag (see
                  ``_validate_catalog_row_path``).
        """
        # Reuse the result from the first call — every active field type shares
        # identical catalog rows, so there's no need to re-read/re-validate the JSON.
        if self._catalog_cache is not None:
            return self._catalog_cache

        import glob
        import json

        paths = sorted(glob.glob(self.file_catalog_path))
        if not paths:
            raise FileNotFoundError(f"No catalog files found matching: {self.file_catalog_path}")

        logger.info("Found %d catalog file(s): %s", len(paths), ", ".join(os.path.basename(p) for p in paths))

        # Fields every matched catalog file must satisfy against the current config
        # (exact match, except "extent" which only needs to cover the request — see below).
        # Note: "variables" is checked separately, against self.var_dict, in
        # _validate_catalog_variables() — var_dict isn't fully populated until every
        # field type has been registered, which happens after this method returns.
        invariant = self.catalog_invariant_metadata()

        file_metas = []
        all_rows = []
        for path in paths:
            with open(path) as f:
                catalog = json.load(f)

            meta = catalog["metadata"]

            # Validate config-invariant fields against current config. "extent" is special-cased:
            # the catalog's QC only covers its own extent, so a request for a *subdomain* of that
            # extent (or the same extent) is fine — only a request extending beyond it is rejected.
            for key, expected_val in invariant.items():
                if key == "extent":
                    if not _extent_covers(meta.get(key), expected_val):
                        raise ValueError(
                            f"File catalog mismatch in '{path}' for 'extent': "
                            f"catalog was QC'd for extent '{meta.get(key)}', which does not cover "
                            f"the requested extent '{expected_val}' (a request must either match "
                            f"exactly or be inset by at least {_EXTENT_MARGIN_DEG} deg — see "
                            f"_extent_covers). Re-run quality_check_goes.py with an extent that "
                            f"covers the request to regenerate, or consider using extent: null "
                            f"(the full grid) instead of a tightly-fitting subdomain."
                        )
                elif meta.get(key) != expected_val:
                    raise ValueError(
                        f"File catalog mismatch in '{path}' for '{key}': "
                        f"catalog has '{meta.get(key)}', config has '{expected_val}'. "
                        f"Re-run quality_check_goes.py to regenerate."
                    )

            # Stash "variables" for _validate_catalog_variables(), requiring it to
            # agree across all matched catalog files (they must all have been built
            # from the same config's var_dict to be safely merged).
            catalog_variables = meta.get("variables")
            if self._catalog_variables_meta is None:
                self._catalog_variables_meta = catalog_variables
            elif self._catalog_variables_meta != catalog_variables:
                raise ValueError(
                    f"File catalog mismatch in '{path}' for 'variables': "
                    f"catalog has '{catalog_variables}', other matched catalog file(s) have "
                    f"'{self._catalog_variables_meta}'. Re-run quality_check_goes.py to regenerate."
                )

            file_metas.append(
                {
                    "path": path,
                    "start": pd.Timestamp(meta["start_datetime"]),
                    "end": pd.Timestamp(meta["end_datetime"]),
                }
            )
            logger.debug(
                "  %s: %s → %s (%d rows)",
                os.path.basename(path),
                meta["start_datetime"],
                meta["end_datetime"],
                len(catalog["catalog"]),
            )
            all_rows.extend(catalog["catalog"])

        # Ensure no two catalog files cover overlapping time ranges
        file_metas.sort(key=lambda m: m["start"])
        for i in range(1, len(file_metas)):
            prev, curr = file_metas[i - 1], file_metas[i]
            if curr["start"] < prev["end"]:
                raise ValueError(
                    f"Overlapping catalog files detected:\n"
                    f"  {prev['path']} covers up to {prev['end']}\n"
                    f"  {curr['path']} starts at {curr['start']}"
                )

        # Merge and sort all rows by start time
        all_rows.sort(key=lambda r: r[0])
        merged = []
        for start, end, path in all_rows:
            start = pd.Timestamp(start)
            _validate_catalog_row_path(path, start)
            merged.append((start, pd.Timestamp(end), path))
        logger.info("Merged catalog: %d total rows, %s → %s", len(merged), merged[0][0], merged[-1][0])

        # Ensure the merged catalog has a row for every timestamp in the full requested
        # range, including target timestamps for the last forecast step — otherwise a
        # missing row would only surface later as a KeyError from _find_file during
        # training/rollout. Checking every timestamp (not just the merged catalog's
        # overall start/end) also catches a gap between two non-overlapping catalog
        # files, or rows removed by hand-editing — either of which the overlap check
        # above wouldn't catch, since neither produces an overlapping time range.
        required_start = self.datetimes[0]
        required_end = self.datetimes[-1] + self.num_forecast_steps * self.dt
        required_range = pd.date_range(required_start, required_end, freq=self.dt)
        merged_starts = {row[0] for row in merged}
        missing = [t for t in required_range if t not in merged_starts]
        if missing:
            raise ValueError(
                f"Catalog is missing {len(missing)} timestamp(s) within the requested "
                f"range [{required_start}, {required_end}] (start_datetime/end_datetime "
                f"plus forecast_len={self.num_forecast_steps} steps), starting at "
                f"{missing[0]}. This can happen from a gap between non-overlapping "
                f"catalog files, or rows removed by hand-editing. Regenerate the catalog "
                f"with quality_check_goes.py to cover this range, or narrow "
                f"start_datetime/end_datetime to fit within the catalog."
            )

        self._catalog_cache = merged  # store for subsequent field types' _load_file_catalog() calls
        return merged

    def _collect_GOES_file_path(self, base_dir: str = ""):
        """Build a time-ordered file map for the dataset's datetime range.

        If ``self.file_catalog_path`` points to an existing JSON catalog, it is
        loaded and validated directly — skipping directory scanning entirely.
        Otherwise, the method lists the appropriate S3 or local hourly
        directories, parses GOES L2 filenames, and associates each timestamp
        with the nearest file within ``tolerance`` (default 3 minutes).

        The scan range covers ``self.datetimes[0]`` through
        ``self.datetimes[-1] + num_forecast_steps * dt`` so that target files
        for all rollout steps are included, not just the first.

        Args:
            base_dir: Root directory prepended to relative paths when ``mode``
                is ``"local"``. Ignored for remote mode.

        Returns:
            A list of ``(period_start, period_end, file_path)`` tuples.
            For a live directory scan, this is exactly one tuple per timestamp
            in ``datetimes``, with ``file_path`` as ``_FILE_NOT_FOUND`` when no
            file was found within tolerance. When short-circuited through
            ``_load_file_catalog`` instead, the returned list is the full
            merged catalog, which may cover a wider time range than
            ``datetimes`` — a superset, not a 1:1 mapping — and may contain
            ``_FILE_QC_MASKED``/``_FILE_SKIP`` rows, which never originate
            from a live scan.

        Raises:
            FileNotFoundError: If GOES L2 files are not found for
                 the requested datetime.
            ValueError: If the GOES L2 filenames do not match the expected
                naming convention (fewer than 6 underscore-separated tokens).
        """
        # Short-circuit: load from pre-built catalog if available, skipping the
        # expensive directory scan (file_catalog_path may be a glob pattern)
        import glob

        if self.file_catalog_path:
            if glob.glob(self.file_catalog_path):
                logger.info("Loading from catalog: %s", self.file_catalog_path)
                return self._load_file_catalog()
            else:
                logger.warning(
                    "No catalog files found matching '%s' — falling back to directory scan.",
                    self.file_catalog_path,
                )

        # Extend by num_forecast_steps so target files for all rollout steps are included in the scan
        datetimes = pd.date_range(
            self.datetimes[0],
            self.datetimes[-1] + self.num_forecast_steps * self.datetimes.freq,
            freq=self.datetimes.freq,
        )

        # -- Collect file paths from local or remote hourly directories --
        if self.mode == "remote":
            if self._fs is None:
                self._fs = _start_s3_fs()
            fs = self._fs

        file_paths = []
        for dt in datetimes.floor("h").unique():
            # GOES-16 was replaced by GOES-19 at 15:10 UTC on April 7, 2025;
            # https://www.ospo.noaa.gov/data/messages/2025/04/MSG_20250407_1510.html
            if self.goes_position == "east":
                if dt < pd.Timestamp("2025-04-07 15:00:00"):
                    goes_id = "goes16"
                elif dt >= pd.Timestamp("2025-04-07 16:00:00"):
                    goes_id = "goes19"
                else:
                    # discard the hour containing the transition since it may contain files
                    # from both satellites and cause ambiguity in file mapping
                    logger.warning(
                        "Discarding observations in %s due to GOES-16/19 transition on 2025-04-07 "
                        "15:10 UTC to avoid file mapping ambiguity during the hour.",
                        dt,
                    )
                    continue
            # GOES-17 was replaced by GOES-18 at 18:00 UTC on January 4, 2023;
            # https://www.ospo.noaa.gov/data/messages/2023/01/MSG_20230104_1805.html
            elif self.goes_position == "west":
                if dt < pd.Timestamp("2023-01-04 18:00:00"):
                    goes_id = "goes17"
                elif dt >= pd.Timestamp("2023-01-04 19:00:00"):
                    goes_id = "goes18"
                else:
                    # discard the hour containing the transition since it may contain files
                    # from both satellites and cause ambiguity in file mapping
                    logger.warning(
                        "Discarding observations in %s due to GOES-17/18 transition on 2023-01-04 "
                        "18:00 UTC to avoid file mapping ambiguity during the hour.",
                        dt,
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
                logger.warning("No data at %s: %s", dt, e)

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
        nearest_indices = unique_times.get_indexer(datetimes, method="nearest", tolerance=self.tolerance)

        freq = _infer_period_freq("s%Y%j%H%M%S%f")

        time_file_map = []
        for query_time, nt_idx in zip(datetimes, nearest_indices):
            period = pd.Period(query_time, freq)
            if nt_idx == -1:
                time_file_map.append((period.start_time, period.end_time, _FILE_NOT_FOUND))
                continue
            files_at_time = df.loc[unique_times[nt_idx], "file"]
            time_file_map.append((period.start_time, period.end_time, files_at_time))

        return time_file_map

    def _get_file_source(
        self,
        field_config: dict[str, Any],
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, str]] | bool | None:
        """GOES's override of ``BaseDataset._get_file_source``: dispatch to ``_collect_GOES_file_path``.

        Passes ``field_config``'s ``path`` as ``base_dir`` in local mode; in
        remote mode there's no local path to resolve, so it's omitted.

        Args:
            field_config (dict[str, Any]): Validated field-type config dict.

        Returns:
            list[tuple[pd.Timestamp, pd.Timestamp, str]]: A list of
                ``(period_start, period_end, file_path)`` tuples produced by
                ``_collect_GOES_file_path``, for both local and remote modes.
        """
        base_dir = field_config.get("path", "")
        if self.mode == "local":
            return self._collect_GOES_file_path(base_dir=base_dir)
        else:
            return self._collect_GOES_file_path()

    # ---------------
    # Runtime data extraction (called from __getitem__, once per sample)
    # ---------------

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
            field_type: One of ``"prognostic"``, ``"diagnostic"``,
                ``"dynamic_forcing"``, or ``"static"``. A GOES channel is
                typically configured as ``"prognostic"``, but the same
                channel can instead be configured as ``"diagnostic"``,
                ``"dynamic_forcing"``, or ``"static"`` depending on the
                experiment — this method has no field-type-specific logic,
                so all four are handled identically.
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
        else:
            arrays = self._load_local_var(field_type, vnames, t)

        for vname, arr in arrays.items():
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            key = self._get_field_name(field_type, "2d", vname)
            sample[key] = tensor
