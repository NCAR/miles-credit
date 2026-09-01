"""
grid_utils.py
-------------
Everything about horizontal grid geometry for gen2: coordinate-pair detection,
rectilinear-vs-curvilinear classification, and GridSchema (the real-coordinate
contract for output, mirroring ChannelSchema in channel_utils.py).

find_coord_pair / infer_grid_type / resolve_source_grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Small, dependency-free helpers for locating a lon/lat coordinate pair in an
``xr.Dataset`` (by name) and classifying it as rectilinear (1D) or curvilinear
(2D). Used by the per-dataset grid detection in ``local.py``/``era5.py``, and
by ``credit.grid.scrip_from_netcdf`` (SCRIP-format grid generation for ESMF
regridding) — this is the shared home for both rather than duplicating the
logic in each.

``resolve_source_grid`` is the composed entry point the dataset classes call:
find the coordinate pair (honouring per-source ``lon_name``/``lat_name``
overrides), then classify it into the ``grid_type`` those classes publish via
``static_metadata["grid"]``.

A note on projected grids: ``x``/``y`` are deliberately **not** candidates for
the geographic coordinate pair. On a projected grid they carry projection
units (typically metres), so treating them as degrees would classify the grid
``rectilinear`` and write those metres out labelled ``latitude``/``longitude``
— silently wrong output, which is worse than failing. Their role is to be the
*dimensions* a real 2D lat/lon field is indexed on; a file carrying only x/y
has no geographic information to recover here.

GridSchema
~~~~~~~~~~
``ForecastWriter`` previously fabricated output lat/lon from
``model.image_height``/``image_width`` via a global ``[-90, 90] x [0, 360)``
linspace — wrong for regional domains and for curvilinear sources (e.g. HRRR),
which need a real 2D lat/lon field, not a fabricated 1D one. There is no
fabricated fallback anywhere in this pipeline anymore: if a real grid can't be
found or resolved, callers raise rather than guess.

Two distinct grid concepts:

* **Native input grid** — each dataset class (``LocalDataset``, ``era5.py``,
  ``goes.py``, ``hrrr.py``) exposes the real lat/lon it read from its own files
  via ``self.static_metadata["grid"]`` (a debugging aid, inspectable directly
  on the live dataset — not necessarily what ends up in the output file). The
  same dataset class also best-effort persists it to
  ``{save_loc}/{source}_grid_schema.nc`` the moment it's known, via
  ``write_source_grid_schema_if_missing`` — including from inside a DataLoader
  worker subprocess, which has filesystem access even though it can't
  propagate Python object state back to the main training process.
* **Resolved output grid** — what ``ForecastWriter`` actually writes. The
  model produces one flat tensor at one fixed ``(H, W)``, so there is exactly
  one output grid per run: the (single, in practice) source's native grid,
  overridden by an active ``Regridder`` preblock's real destination grid when
  regridding is in play (the common case where regridding is *not* used is
  the default: the native grid passes straight through unchanged).

Lifecycle
^^^^^^^^^
Training/rollout setup resolves the schema once — via ``GridSchema.resolve``
using the live dataset + preblocks (with a disk fallback to each source's
``{source}_grid_schema.nc`` when the live process never saw the data itself)
— and, only when an active ``Regridder`` actually changes the grid, saves the
result to ``{save_loc}/output_grid_schema.nc`` (skipped when no regridder is
active: the single source's own file already *is* the effective output grid,
so a second identical copy would be pure duplication). Later runs (or a
re-run without training) load whichever file is present instead of
re-resolving, via ``GridSchema.load_or_resolve``.

Scope: rectilinear, curvilinear and unstructured. An unstructured source
resolves and writes on its native mesh — a single ``ncol`` dimension with
per-cell ``lat``/``lon`` as non-dimension coordinates — so a ``Regridder``
preblock is an option for such a source rather than a requirement.

A curvilinear grid additionally carries its 1D projection axes (``y``/``x``,
typically in metres) when the source file has them, so projected output keeps
its native horizontal coordinates alongside the geographic ones. Deriving
lat/lon *from* a projection is explicitly out of scope: CRS/``grid_mapping``
handling would need a projection library, and plain lat/lon coordinates are
CF-valid on their own. A file carrying only x/y and no lat/lon must supply the
geography another way (see ``LocalDataset``'s ``coordinate_file``).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Literal

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinate detection / classification
# ---------------------------------------------------------------------------

# Supported name pairs in priority order: (lon_name, lat_name).
#
# ORDER IS BEHAVIOUR: the first two entries are the original pair and must stay
# first, so any file that resolves today keeps resolving identically. Append new
# conventions, never prepend. Matching is exact/case-sensitive; anything not
# covered here is reachable via the explicit lon_name/lat_name override.
#
# x/y are intentionally absent — see the module docstring's note on projected grids.
_COORD_CANDIDATES = [
    ("longitude", "latitude"),
    ("lon", "lat"),
    ("lons", "lats"),
    ("XLONG", "XLAT"),  # WRF
    ("nav_lon", "nav_lat"),  # NEMO
    ("grid_xt", "grid_yt"),  # GFDL / FV3
    ("geolon", "geolat"),  # MOM / GFDL ocean
    ("lonCell", "latCell"),  # MPAS (unstructured)
]


def find_coord_pair(ds, lon_name: str | None = None, lat_name: str | None = None):
    """
    Find a lon/lat coordinate pair in an xr.Dataset.

    Searches _COORD_CANDIDATES in order across both ds.coords and ds.data_vars,
    unless *lon_name* and *lat_name* name the variables explicitly — the escape
    hatch for naming conventions the candidate table does not cover.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to search.
    lon_name, lat_name : str, optional
        Explicit coordinate variable names. Must be given together; when
        present the candidate table is bypassed entirely.

    Returns
    -------
    (lon_array, lat_array, lon_name, lat_name)

    Raises
    ------
    ValueError
        If only one of lon_name/lat_name is given, if an explicitly named
        variable is absent, or if no recognised pair is found.
    """
    all_names = set(ds.data_vars) | set(ds.coords)

    if (lon_name is None) != (lat_name is None):
        raise ValueError(
            "find_coord_pair: lon_name and lat_name must be given together "
            f"(got lon_name={lon_name!r}, lat_name={lat_name!r})."
        )

    if lon_name is not None:
        missing = [n for n in (lon_name, lat_name) if n not in all_names]
        if missing:
            raise ValueError(
                f"Explicitly named coordinate variable(s) {missing} not found.\n"
                f"Available names: {sorted(all_names)}"
            )
        candidates = [(lon_name, lat_name)]
    else:
        candidates = _COORD_CANDIDATES

    for lon_key, lat_key in candidates:
        if lon_key in all_names and lat_key in all_names:
            return (
                ds[lon_key].values.astype(float),
                ds[lat_key].values.astype(float),
                lon_key,
                lat_key,
            )

    raise ValueError(
        "Could not find a recognised lon/lat coordinate pair.\n"
        f"Expected one of: {_COORD_CANDIDATES}\n"
        f"Available names: {sorted(all_names)}\n"
        "Set lon_name/lat_name explicitly in the source config, rename your "
        "coordinates, or call scrip_from_rectilinear / scrip_from_curvilinear directly."
    )


def infer_grid_type(lat, lon):
    """
    Classify a lat/lon coordinate pair as "rectilinear" or "curvilinear".

    1D lat and lon -> rectilinear.  2D lat and lon -> curvilinear.

    Returns
    -------
    str : "rectilinear" or "curvilinear"

    Raises
    ------
    ValueError if lat/lon are not both 1D or both 2D.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim == 1 and lon.ndim == 1:
        return "rectilinear"
    elif lat.ndim == 2 and lon.ndim == 2:
        return "curvilinear"
    raise ValueError(
        f"Unexpected coordinate shapes: lat={lat.shape}, lon={lon.shape}. "
        "Expected both 1D (rectilinear) or both 2D (curvilinear)."
    )


def resolve_source_grid(
    ds,
    source_cfg: dict[str, Any] | None = None,
    *,
    allow_unstructured: bool = True,
) -> dict[str, Any]:
    """Find and classify one source's native grid from an open dataset.

    The composed entry point the gen2 dataset classes call to populate
    ``static_metadata["grid"]``: locate the geographic coordinate pair
    (honouring ``lon_name``/``lat_name`` overrides from *source_cfg*), then
    decide the ``grid_type``.

    Classification order, first match wins:

    1. An explicit ``grid_type:`` in *source_cfg* — always authoritative.
    2. 1D lat/lon sharing the exact same dimension in the file (e.g. ``ncol``)
       → ``"unstructured"``. This is a reliable structural signal.
    3. 1D lat/lon of equal length on *different* dimensions — ambiguous on size
       alone — resolved by how the file's data variables are actually laid out:
       one horizontal dimension → ``"unstructured"``, two → rectilinear. See
       ``_data_spatial_rank``.
    4. Otherwise ``infer_grid_type``: 1D/1D → rectilinear, 2D/2D → curvilinear.

    Args:
        ds: Open dataset to read coordinates from. May be the source's own data
            file or a separate coordinate file — the classification is the same
            either way, since it depends only on the coordinates' own structure.
        source_cfg: This source's config block. Read for ``grid_type``,
            ``lon_name`` and ``lat_name``; all optional.
        allow_unstructured: When False, skip rules 2 and 3. Set by callers whose
            sources are known-rectilinear global stores (see ``era5.py``), where
            the same-length fallback could only ever be a false positive.

    Returns:
        ``{"grid_type": str, "lat": np.ndarray, "lon": np.ndarray}``, plus
        ``"x"``/``"y"`` (and ``"xy_attrs"``) for a curvilinear grid whose
        dimensions carry projection coordinates — see ``_find_projection_axes``.

    Raises:
        ValueError: if no coordinate pair can be found (see ``find_coord_pair``)
            or the shapes are not a recognised combination (see ``infer_grid_type``).
    """
    source_cfg = source_cfg or {}
    lon, lat, lon_name, lat_name = find_coord_pair(
        ds,
        lon_name=source_cfg.get("lon_name"),
        lat_name=source_cfg.get("lat_name"),
    )

    config_grid_type = source_cfg.get("grid_type")
    is_1d_pair = lat.ndim == 1 and lon.ndim == 1

    if config_grid_type:
        grid_type = config_grid_type
    elif allow_unstructured and is_1d_pair and ds[lon_name].dims == ds[lat_name].dims:
        grid_type = "unstructured"
    elif allow_unstructured and is_1d_pair and len(lat) == len(lon):
        # Same length but on different dimensions. A size match alone proves
        # nothing -- a square rectilinear grid (n_lat == n_lon) produces one
        # trivially -- so decide from how the data is actually indexed.
        grid_type = "unstructured" if _data_spatial_rank(ds, lat_name, lon_name, source_cfg) == 1 else infer_grid_type(lat, lon)
    else:
        grid_type = infer_grid_type(lat, lon)

    grid: dict[str, Any] = {"grid_type": grid_type, "lat": lat, "lon": lon}
    if grid_type == "curvilinear":
        grid.update(_find_projection_axes(ds, lat_name))
    return grid


def _data_spatial_rank(ds, lat_name: str, lon_name: str, source_cfg: dict[str, Any]) -> int | None:
    """How many horizontal dimensions this file's data variables actually use.

    The decisive signal for the one genuinely ambiguous case in
    ``resolve_source_grid``: 1D lat/lon of equal length sitting on different
    dimensions. Data on a single horizontal dimension is an unstructured mesh;
    data on two is a structured grid whose axes merely happen to be the same
    length.

    Time and level dimensions are excluded, so what remains is horizontal.
    Returns None when the file has no usable data variables (a pure coordinate
    file), or when its variables disagree — in both cases the caller falls back
    to shape-based classification, which treats lat/lon on separate dimensions
    as rectilinear. That is the right default: a genuine unstructured mesh puts
    lat and lon on the *same* dimension, which rule 2 has already caught.
    """
    time_coord = source_cfg.get("time_coord", "time")
    level_coord = source_cfg.get("level_coord")
    skip = {lat_name, lon_name}

    ranks = set()
    for name, da in ds.data_vars.items():
        if name in skip:
            continue
        spatial = [d for d in da.dims if d not in (time_coord, level_coord)]
        if spatial:
            ranks.add(len(spatial))

    return next(iter(ranks)) if len(ranks) == 1 else None


def _find_projection_axes(ds, lat_name: str) -> dict[str, Any]:
    """Recover the 1D projection axes a 2D lat/lon field is indexed on.

    On a projected grid (Lambert Conformal, polar stereographic, ...) the real
    geography is the 2D lat/lon pair, but the *dimensions* it lives on often
    carry their own 1D coordinate variables in projection units — ``x``/``y`` in
    metres, typically. Those are worth carrying through to the output file:
    they are the grid's native horizontal axes, and dropping them loses
    information a downstream user may need.

    They are found structurally, from the dimensions of the latitude variable,
    rather than by guessing names — so this works whether the dims are called
    ``y``/``x``, ``south_north``/``west_east``, or anything else. A file whose
    dimensions have no coordinate variables (common for WRF output) simply
    yields nothing, which is not an error.

    Attributes are copied verbatim from the file rather than invented: without
    reading the CRS we do not know the units, and fabricating CF metadata would
    be worse than omitting it.

    Returns:
        ``{}``, or ``{"y":, "x":, "xy_attrs": {"y": {...}, "x": {...}}}``.
    """
    dims = ds[lat_name].dims
    if len(dims) != 2:
        return {}
    y_dim, x_dim = dims
    if y_dim not in ds.coords or x_dim not in ds.coords:
        return {}

    y_var, x_var = ds[y_dim], ds[x_dim]
    if y_var.ndim != 1 or x_var.ndim != 1:
        return {}

    return {
        "y": y_var.values,
        "x": x_var.values,
        "xy_attrs": {"y": dict(y_var.attrs), "x": dict(x_var.attrs)},
    }


def expected_spatial_shape(grid: dict[str, Any]) -> tuple[int, ...]:
    """The horizontal shape a data variable must have to sit on *grid*.

    Used to cross-check a grid read from a separate ``coordinate_file`` against
    the data it is supposed to describe — the one failure mode a detached
    coordinate file introduces, and one that cannot arise when the coordinates
    live in the data file itself.

    Returns ``(n_lat, n_lon)`` for rectilinear, ``(ny, nx)`` for curvilinear,
    and ``(ncol,)`` for unstructured.
    """
    lat = np.asarray(grid["lat"])
    lon = np.asarray(grid["lon"])
    if grid["grid_type"] == "unstructured":
        return (lat.size,)
    if lat.ndim == 2:
        return tuple(lat.shape)
    return (lat.size, lon.size)


# Per-source native grid, written directly by the dataset class that read it.
SOURCE_GRID_SCHEMA_FILENAME = "{source}_grid_schema.nc"
# Resolved/effective output grid — only written when a Regridder preblock is
# actually active (see GridSchema.origin); otherwise the single source's own
# file above already is the effective output grid.
OUTPUT_GRID_SCHEMA_FILENAME = "output_grid_schema.nc"

GridType = Literal["rectilinear", "curvilinear", "unstructured"]
_VALID_GRID_TYPES = ("rectilinear", "curvilinear", "unstructured")


def write_source_grid_schema_if_missing(source_name: str, grid: dict[str, Any] | None, save_loc: str | None) -> None:
    """Best-effort persist one source's native grid to ``{save_loc}/{source}_grid_schema.nc``.

    Called from each dataset class right after ``static_metadata["grid"]`` is
    first populated — including from inside a DataLoader worker subprocess
    under ``num_workers > 0``, since workers have their own filesystem access
    even though they can't propagate Python object state back to the main
    process. Guarded by file existence, so redundant calls (e.g. multiple
    workers independently resolving the same grid) are cheap after the first
    successful write. Failures (no ``save_loc``, read-only filesystem, ...)
    are logged and swallowed — this must never break the data-loading path
    it's piggybacked onto.

    Every grid type ``GridSchema`` can represent is persisted, unstructured
    included; a type it cannot represent is logged and skipped rather than
    raising, since this runs inside the data-loading path.
    """
    if grid is None or not save_loc:
        return
    if grid["grid_type"] not in _VALID_GRID_TYPES:
        logger.info(
            "Source '%s' has a %r native grid; not persisted to %s (GridSchema represents %s).",
            source_name,
            grid["grid_type"],
            SOURCE_GRID_SCHEMA_FILENAME.format(source=source_name),
            _VALID_GRID_TYPES,
        )
        return
    path = os.path.join(os.path.expandvars(save_loc), SOURCE_GRID_SCHEMA_FILENAME.format(source=source_name))
    if os.path.isfile(path):
        return
    try:
        GridSchema(
            grid["grid_type"],
            grid["lat"],
            grid["lon"],
            y=grid.get("y"),
            x=grid.get("x"),
            xy_attrs=grid.get("xy_attrs"),
        ).save(path)
    except Exception as exc:
        logger.warning("Could not write grid schema for source '%s' to %s (%s).", source_name, path, exc)


def _load_source_grid_schema(source_name: str, save_loc: str | None) -> dict[str, Any] | None:
    """Disk fallback for one source's native grid, used when the live process
    never populated ``static_metadata["grid"]`` itself (e.g. a HRRR/remote-ERA5
    source read by a different DataLoader worker under ``num_workers > 0``)."""
    if not save_loc:
        return None
    path = os.path.join(os.path.expandvars(save_loc), SOURCE_GRID_SCHEMA_FILENAME.format(source=source_name))
    if not os.path.isfile(path):
        return None
    schema = GridSchema.load(path)
    return {
        "grid_type": schema.grid_type,
        "lat": schema.lat,
        "lon": schema.lon,
        "y": schema.y,
        "x": schema.x,
        "xy_attrs": schema.xy_attrs,
    }


def _native_grid(dataset: Any, save_loc: str | None = None) -> dict[str, Any] | None:
    """Return the resolved native grid dict for *dataset*.

    Handles both a single source dataset (``static_metadata`` is that source's
    own dict, with a top-level ``"grid"`` key) and ``MultiSourceDataset``
    (``static_metadata`` is ``{source_name: {..., "grid": ...}}``). For any
    source with no live grid, falls back to that source's persisted
    ``{source}_grid_schema.nc`` in *save_loc* before giving up on it.

    Raises:
        ValueError: if more than one source reports a native grid and they disagree.
    """
    static_metadata = getattr(dataset, "static_metadata", None) or {}

    if "grid" in static_metadata:
        # Single-source dataset: static_metadata is this source's own dict.
        grid = static_metadata["grid"]
        if grid is not None:
            return grid
        source_name = getattr(dataset, "curr_source_name", None)
        return _load_source_grid_schema(source_name, save_loc) if source_name else None

    grids: dict[str, dict[str, Any]] = {}
    for name, meta in static_metadata.items():
        grid = meta.get("grid") if meta else None
        if grid is None:
            grid = _load_source_grid_schema(name, save_loc)
        if grid is not None:
            grids[name] = grid

    if not grids:
        return None
    if len(grids) == 1:
        return next(iter(grids.values()))

    first_name, first_grid = next(iter(grids.items()))
    for name, grid in grids.items():
        if grid["grid_type"] != first_grid["grid_type"] or np.shape(grid["lat"]) != np.shape(first_grid["lat"]):
            raise ValueError(
                f"GridSchema.resolve: sources '{first_name}' and '{name}' report different "
                "native grids and no regridding preblock reconciles them. Add a Regridder "
                "preblock (which regrids every source onto one shared destination grid), "
                "or pass an explicit grid_schema."
            )
    return first_grid


def _find_regridder(ic_preblocks, step_preblocks):
    """Return the active ``Regridder`` instance with a resolved destination grid, if any.

    Walks both preblock ``nn.ModuleDict`` groups the same way
    ``credit.preblock.attach_channel_schema`` does. Returns ``None`` when no
    regridder is active — the common case.

    Raises:
        ValueError: if multiple active regridders resolve to different destination grids.
    """
    from credit.preblock.regrid import Regridder  # local import: avoid a hard torch dependency at module import time

    found = []
    for group in (ic_preblocks, step_preblocks):
        if group is None:
            continue
        for block in group.values():
            if isinstance(block, Regridder) and block.dst_lat is not None:
                found.append(block)

    if not found:
        return None

    first = found[0]
    for block in found[1:]:
        if block.dst_grid_type != first.dst_grid_type or block.dst_lat.shape != first.dst_lat.shape:
            raise ValueError(
                "GridSchema.resolve: multiple active Regridder preblocks resolve to different "
                "destination grids; the model expects one shared output grid."
            )
    return first


class GridSchema:
    """The resolved horizontal output grid: shared across every variable/source
    in one output file (the model produces one flat tensor at one fixed shape).

    Args:
        grid_type: ``"rectilinear"``, ``"curvilinear"`` or ``"unstructured"``.
        lat: 1D (rectilinear/unstructured) or 2D ``(y, x)`` (curvilinear) latitude array.
        lon: 1D (rectilinear/unstructured) or 2D ``(y, x)`` (curvilinear) longitude array.
            For unstructured, lat and lon are per-cell and must be the same length.
        origin: ``"native"`` (a source's own grid, unchanged) or
            ``"regridded"`` (an active ``Regridder`` preblock's destination
            grid). Set by ``.resolve()``; defaults to ``"native"`` for direct
            construction. Used by callers (e.g. the trainer) to decide whether
            this schema needs its own ``output_grid_schema.nc`` — a
            ``"native"`` schema is already fully covered by the source's own
            ``{source}_grid_schema.nc``.
        y, x: Optional 1D projection axes for a curvilinear grid, in the file's
            own units (see ``_find_projection_axes``). Carried through to the
            output file as dimension coordinates when present; ``None`` simply
            means the source had none, which is normal and not an error.
        xy_attrs: Optional ``{"y": {...}, "x": {...}}`` attribute dicts copied
            verbatim from the source file, so units/standard_name survive
            without this module inventing CF metadata it cannot verify.
    """

    def __init__(
        self,
        grid_type: GridType,
        lat: np.ndarray,
        lon: np.ndarray,
        origin: Literal["native", "regridded"] = "native",
        y: np.ndarray | None = None,
        x: np.ndarray | None = None,
        xy_attrs: dict[str, dict[str, Any]] | None = None,
    ):
        if grid_type not in _VALID_GRID_TYPES:
            raise ValueError(f"GridSchema: grid_type must be one of {_VALID_GRID_TYPES}, got {grid_type!r}")
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        expected_ndim = 2 if grid_type == "curvilinear" else 1
        if lat.ndim != expected_ndim or lon.ndim != expected_ndim:
            raise ValueError(
                f"GridSchema: grid_type={grid_type!r} expects {expected_ndim}D lat/lon, "
                f"got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
            )
        # Unstructured lat/lon are per-cell, so unequal lengths mean the two
        # arrays describe different meshes -- unlike rectilinear, where differing
        # lengths are simply the grid's two axes.
        if grid_type == "unstructured" and lat.size != lon.size:
            raise ValueError(
                f"GridSchema: unstructured lat/lon are per-cell and must be the same length, "
                f"got lat.size={lat.size}, lon.size={lon.size}"
            )
        self.grid_type: GridType = grid_type
        self.lat = lat
        self.lon = lon
        self.origin = origin

        # Projection axes are curvilinear-only: on a rectilinear grid the
        # dimension coordinates already *are* lat/lon, so a second pair would be
        # a duplicate; validated rather than silently dropped.
        if y is not None or x is not None:
            if grid_type != "curvilinear":
                raise ValueError(
                    f"GridSchema: y/x projection axes are only meaningful for a curvilinear "
                    f"grid, got grid_type={grid_type!r}."
                )
            y, x = np.asarray(y), np.asarray(x)
            if y.ndim != 1 or x.ndim != 1:
                raise ValueError(f"GridSchema: y/x must be 1D, got y.ndim={y.ndim}, x.ndim={x.ndim}")
            if (y.size, x.size) != lat.shape:
                raise ValueError(
                    f"GridSchema: y/x sizes {(y.size, x.size)} do not match lat/lon shape {lat.shape}."
                )
        self.y = y
        self.x = x
        self.xy_attrs = xy_attrs or {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def resolve(cls, dataset: Any, ic_preblocks=None, step_preblocks=None, save_loc: str | None = None) -> "GridSchema":
        """Resolve the effective output grid from a live dataset + preblocks.

        Starts from the dataset's native grid (``static_metadata["grid"]``,
        falling back to each source's persisted ``{source}_grid_schema.nc`` in
        *save_loc* when the live process doesn't have it); if an active
        ``Regridder`` preblock is found, its real destination grid is used
        instead. When no regridder is active (the common case), the native
        grid passes through unchanged and the returned schema's ``origin`` is
        ``"native"``.

        Raises:
            ValueError: if no native grid is available, or if grids disagree
                (see ``_native_grid`` / ``_find_regridder``).
        """
        native = _native_grid(dataset, save_loc)
        if native is None:
            raise ValueError(
                "GridSchema.resolve: no native grid available from dataset.static_metadata "
                f"or a saved {SOURCE_GRID_SCHEMA_FILENAME.format(source='<source>')} in save_loc. "
                "Ensure at least one source populates static_metadata['grid']."
            )

        regridder = _find_regridder(ic_preblocks, step_preblocks)
        if regridder is None:
            return cls(
                native["grid_type"],
                native["lat"],
                native["lon"],
                origin="native",
                y=native.get("y"),
                x=native.get("x"),
                xy_attrs=native.get("xy_attrs"),
            )

        # A regridded grid gets no projection axes: the Regridder's destination
        # comes from an ESMF weight file, which stores only cell centres, so the
        # source's own x/y no longer describe the output.
        return cls(regridder.dst_grid_type, regridder.dst_lat, regridder.dst_lon, origin="regridded")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the schema as NetCDF (atomically: temp file + rename).

        The staging file is per-process. Every rank — and every DataLoader
        worker under ``num_workers > 0`` — persists the grid it resolved (see
        ``write_source_grid_schema_if_missing``), so several processes routinely
        write the same *path* at once. A single shared ``<path>.tmp`` made them
        collide inside HDF5, surfacing as ``[Errno 13] Permission denied`` for
        every writer but the first. With one temp file each, the writes are
        independent and the renames are atomic; the content is identical, so
        last-writer-wins is harmless.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if self.grid_type == "rectilinear":
            ds = xr.Dataset(coords={"lat": ("lat", self.lat), "lon": ("lon", self.lon)})
        elif self.grid_type == "unstructured":
            # Both per-cell on the single mesh dimension.
            ds = xr.Dataset(coords={"lat": ("ncol", self.lat), "lon": ("ncol", self.lon)})
        else:
            coords = {}
            if self.y is not None and self.x is not None:
                coords = {"y": ("y", self.y), "x": ("x", self.x)}
            ds = xr.Dataset(
                data_vars={"lat": (("y", "x"), self.lat), "lon": (("y", "x"), self.lon)},
                coords=coords,
            )
            for axis, attrs in self.xy_attrs.items():
                if axis in ds.coords and attrs:
                    ds[axis].attrs.update(attrs)
        ds.attrs["grid_type"] = self.grid_type

        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            ds.to_netcdf(tmp)
            os.replace(tmp, path)
        except BaseException:
            # Never leave a half-written .tmp.<pid> behind for the next run to trip over.
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise
        logger.info("GridSchema saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "GridSchema":
        """Read a schema back. Projection axes are optional — a file written
        before they existed, or by a source that had none, loads unchanged."""
        with xr.open_dataset(path) as ds:
            grid_type = ds.attrs["grid_type"]
            lat = ds["lat"].values
            lon = ds["lon"].values
            y = x = None
            xy_attrs: dict[str, dict[str, Any]] = {}
            if grid_type == "curvilinear" and "y" in ds.coords and "x" in ds.coords:
                y, x = ds["y"].values, ds["x"].values
                xy_attrs = {"y": dict(ds["y"].attrs), "x": dict(ds["x"].attrs)}
        return cls(grid_type, lat, lon, y=y, x=x, xy_attrs=xy_attrs)

    @classmethod
    def load_or_resolve(
        cls,
        dataset: Any,
        ic_preblocks=None,
        step_preblocks=None,
        save_loc: str | None = None,
    ) -> "GridSchema | None":
        """Load ``output_grid_schema.nc`` from ``save_loc`` (only ever written when
        regridding was used), else resolve live from *dataset* (native grid,
        with a disk fallback to each source's own ``{source}_grid_schema.nc``,
        overridden by an active Regridder).

        Returns ``None`` (with a warning) when neither is possible — callers
        should treat that as fatal; there is no fabricated fallback.
        """
        path = os.path.join(os.path.expandvars(save_loc), OUTPUT_GRID_SCHEMA_FILENAME) if save_loc else None
        if path and os.path.isfile(path):
            logger.info("Loading grid schema from %s", path)
            return cls.load(path)
        try:
            schema = cls.resolve(dataset, ic_preblocks, step_preblocks, save_loc)
            logger.info(
                "No %s in %s — grid schema resolved from live dataset.",
                OUTPUT_GRID_SCHEMA_FILENAME,
                save_loc,
            )
            return schema
        except (ValueError, AttributeError) as e:
            logger.warning("No grid schema available (%s).", e)
            return None
