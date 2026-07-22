"""
grid_utils.py
-------------
Everything about horizontal grid geometry for gen2: coordinate-pair detection,
rectilinear-vs-curvilinear classification, and GridSchema (the real-coordinate
contract for output, mirroring ChannelSchema in channel_utils.py).

find_coord_pair / infer_grid_type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Small, dependency-free helpers for locating a lon/lat coordinate pair in an
``xr.Dataset`` (by name) and classifying it as rectilinear (1D) or curvilinear
(2D). Used by the per-dataset grid detection in ``local.py``/``era5.py``, and
by ``credit.grid.scrip_from_netcdf`` (SCRIP-format grid generation for ESMF
regridding) — this is the shared home for both rather than duplicating the
logic in each.

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

Scope: rectilinear and curvilinear only (no unstructured). Projection/CRS
metadata (e.g. HRRR's Lambert Conformal Conic) is deferred to a follow-on —
plain lat/lon coordinates are CF-valid on their own.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinate detection / classification
# ---------------------------------------------------------------------------

# Supported name pairs in priority order: (lon_name, lat_name)
_COORD_CANDIDATES = [
    ("longitude", "latitude"),
    ("lon", "lat"),
]


def find_coord_pair(ds):
    """
    Find a lon/lat coordinate pair in an xr.Dataset.

    Searches _COORD_CANDIDATES in order across both ds.coords and ds.data_vars.

    Returns
    -------
    (lon_array, lat_array, lon_name, lat_name)

    Raises
    ------
    ValueError if no recognised pair is found.
    """
    all_names = set(ds.data_vars) | set(ds.coords)
    for lon_name, lat_name in _COORD_CANDIDATES:
        if lon_name in all_names and lat_name in all_names:
            return (
                ds[lon_name].values.astype(float),
                ds[lat_name].values.astype(float),
                lon_name,
                lat_name,
            )

    raise ValueError(
        "Could not find a recognised lon/lat coordinate pair.\n"
        f"Expected one of: {_COORD_CANDIDATES}\n"
        f"Available names: {sorted(all_names)}\n"
        "Rename your coordinates or call scrip_from_rectilinear / "
        "scrip_from_curvilinear directly."
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


# Per-source native grid, written directly by the dataset class that read it.
SOURCE_GRID_SCHEMA_FILENAME = "{source}_grid_schema.nc"
# Resolved/effective output grid — only written when a Regridder preblock is
# actually active (see GridSchema.origin); otherwise the single source's own
# file above already is the effective output grid.
OUTPUT_GRID_SCHEMA_FILENAME = "output_grid_schema.nc"

GridType = Literal["rectilinear", "curvilinear"]
_VALID_GRID_TYPES = ("rectilinear", "curvilinear")


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
    """
    if grid is None or not save_loc:
        return
    path = os.path.join(os.path.expandvars(save_loc), SOURCE_GRID_SCHEMA_FILENAME.format(source=source_name))
    if os.path.isfile(path):
        return
    try:
        GridSchema(grid["grid_type"], grid["lat"], grid["lon"]).save(path)
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
    return {"grid_type": schema.grid_type, "lat": schema.lat, "lon": schema.lon}


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
        grid_type: ``"rectilinear"`` or ``"curvilinear"``.
        lat: 1D (rectilinear) or 2D ``(y, x)`` (curvilinear) latitude array.
        lon: 1D (rectilinear) or 2D ``(y, x)`` (curvilinear) longitude array.
        origin: ``"native"`` (a source's own grid, unchanged) or
            ``"regridded"`` (an active ``Regridder`` preblock's destination
            grid). Set by ``.resolve()``; defaults to ``"native"`` for direct
            construction. Used by callers (e.g. the trainer) to decide whether
            this schema needs its own ``output_grid_schema.nc`` — a
            ``"native"`` schema is already fully covered by the source's own
            ``{source}_grid_schema.nc``.
    """

    def __init__(
        self,
        grid_type: GridType,
        lat: np.ndarray,
        lon: np.ndarray,
        origin: Literal["native", "regridded"] = "native",
    ):
        if grid_type not in _VALID_GRID_TYPES:
            raise ValueError(f"GridSchema: grid_type must be one of {_VALID_GRID_TYPES}, got {grid_type!r}")
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        expected_ndim = 1 if grid_type == "rectilinear" else 2
        if lat.ndim != expected_ndim or lon.ndim != expected_ndim:
            raise ValueError(
                f"GridSchema: grid_type={grid_type!r} expects {expected_ndim}D lat/lon, "
                f"got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
            )
        self.grid_type: GridType = grid_type
        self.lat = lat
        self.lon = lon
        self.origin = origin

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
            return cls(native["grid_type"], native["lat"], native["lon"], origin="native")

        return cls(regridder.dst_grid_type, regridder.dst_lat, regridder.dst_lon, origin="regridded")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the schema as NetCDF (atomically: temp file + rename)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if self.grid_type == "rectilinear":
            ds = xr.Dataset(coords={"lat": ("lat", self.lat), "lon": ("lon", self.lon)})
        else:
            ds = xr.Dataset(data_vars={"lat": (("y", "x"), self.lat), "lon": (("y", "x"), self.lon)})
        ds.attrs["grid_type"] = self.grid_type

        tmp = path + ".tmp"
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
        logger.info("GridSchema saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "GridSchema":
        with xr.open_dataset(path) as ds:
            grid_type = ds.attrs["grid_type"]
            lat = ds["lat"].values
            lon = ds["lon"].values
        return cls(grid_type, lat, lon)

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
