"""Reduced Gaussian grid utilities.

ECMWF stores several native ERA5 products (e.g. the ARCO ERA5 cloud-optimized
``co/`` zarr stores) on a *reduced* Gaussian grid: latitude rings sit at the
Gaussian quadrature latitudes, and each ring carries its own number of
longitude points (fewer near the poles), always starting at 0°E with uniform
spacing. Points are stored as one flat vector, ring-major, north to south,
longitude increasing within each ring — N320 has 640 rings and 542,080 points.

This module provides:

- :class:`ReducedGaussianGrid` — the ring structure (latitudes, points per
  ring, flat offsets), built from per-point lat/lon coordinate vectors (as
  carried by the ARCO ``co/`` stores), from a saved NetCDF description, or
  from an explicit ring spec (for tests / synthetic grids).
- :func:`reduced_gaussian_to_latlon_bilinear_weights` — a sparse bilinear
  interpolation matrix from the reduced grid to a rectilinear lat/lon grid,
  written in the ESMF weight-file layout consumed by the ``regrid`` preblock
  (``credit.preblock.regrid.Regridder``). Pure numpy — no ESMF install needed.

Typical workflow (one-time setup for the ``arco_era5_co`` source)::

    from credit.reduced_gaussian import (
        fetch_arco_n320_grid, reduced_gaussian_to_latlon_bilinear_weights,
    )
    import numpy as np

    grid = fetch_arco_n320_grid("N320_grid.nc")            # ~9 MB of coords from GCS
    dst_lat = 90.0 - 0.25 * np.arange(721)                 # 0.25° equiangular, N->S
    dst_lon = 0.25 * np.arange(1440)
    reduced_gaussian_to_latlon_bilinear_weights(
        grid, dst_lat, dst_lon, "n320_to_0p25_bilinear.nc"
    )
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ReducedGaussianGrid:
    """Ring description of a reduced Gaussian grid.

    Attributes:
        ring_lats: ``(n_rings,)`` ring latitudes in degrees, strictly
            decreasing (north to south).
        ring_nlon: ``(n_rings,)`` integer number of longitude points per ring.
        ring_offset: ``(n_rings,)`` offset of each ring's first point in the
            flat point vector.
        n_points: total number of grid points (``ring_nlon.sum()``).
    """

    def __init__(self, ring_lats: np.ndarray, ring_nlon: np.ndarray):
        ring_lats = np.asarray(ring_lats, dtype=np.float64)
        ring_nlon = np.asarray(ring_nlon, dtype=np.int64)
        if ring_lats.ndim != 1 or ring_lats.shape != ring_nlon.shape:
            raise ValueError(
                f"ring_lats and ring_nlon must be 1D arrays of equal length, "
                f"got {ring_lats.shape} and {ring_nlon.shape}."
            )
        if not np.all(np.diff(ring_lats) < 0):
            raise ValueError("ring_lats must be strictly decreasing (north to south).")
        if np.any(ring_nlon < 1):
            raise ValueError("Every ring must have at least one longitude point.")
        self.ring_lats = ring_lats
        self.ring_nlon = ring_nlon
        self.ring_offset = np.concatenate([[0], np.cumsum(ring_nlon)[:-1]])
        self.n_points = int(ring_nlon.sum())

    @property
    def n_rings(self) -> int:
        return len(self.ring_lats)

    def lats_flat(self) -> np.ndarray:
        """Per-point latitude vector, ``(n_points,)``."""
        return np.repeat(self.ring_lats, self.ring_nlon)

    def lons_flat(self) -> np.ndarray:
        """Per-point longitude vector, ``(n_points,)``; each ring starts at 0°E."""
        return np.concatenate([360.0 * np.arange(n) / n for n in self.ring_nlon])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_points(cls, lat: np.ndarray, lon: np.ndarray) -> "ReducedGaussianGrid":
        """Build from per-point lat/lon vectors (ring-major, N->S, lon increasing).

        This is the coordinate layout of the ARCO ERA5 ``co/`` reduced-Gaussian
        stores (their ``latitude``/``longitude`` coords on the ``values`` dim).
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.ndim != 1 or lat.shape != lon.shape:
            raise ValueError(f"lat/lon must be equal-length 1D arrays, got {lat.shape}, {lon.shape}.")
        # Ring boundaries where the latitude changes.
        change = np.flatnonzero(np.diff(lat) != 0.0) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(lat)]])
        ring_lats = lat[starts]
        ring_nlon = ends - starts
        # Sanity: each ring starts at 0°E with uniform spacing.
        if not np.allclose(lon[starts], 0.0):
            raise ValueError("Expected each latitude ring to start at longitude 0.")
        return cls(ring_lats, ring_nlon)

    @classmethod
    def from_file(cls, path: str) -> "ReducedGaussianGrid":
        """Load a grid saved by :meth:`save`."""
        with xr.open_dataset(path) as ds:
            return cls(ds["ring_lats"].values, ds["ring_nlon"].values)

    def save(self, path: str) -> None:
        """Persist the ring description as a small NetCDF file."""
        xr.Dataset(
            {
                "ring_lats": (("ring",), self.ring_lats),
                "ring_nlon": (("ring",), self.ring_nlon),
            },
            attrs={"description": "Reduced Gaussian grid ring description (N->S)."},
        ).to_netcdf(path)


def reduced_gaussian_to_latlon_bilinear_weights(
    grid: ReducedGaussianGrid,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    out_file: str | None = None,
) -> xr.Dataset:
    """Sparse bilinear weights from a reduced Gaussian grid to a lat/lon grid.

    For each destination point: linear interpolation between the two bracketing
    latitude rings, and periodic linear interpolation in longitude within each
    ring (up to 4 source points total). Destination latitudes poleward of the
    first/last ring clamp to that ring (longitude interpolation only) — the
    same treatment ECMWF's MIR applies at the poles.

    Args:
        grid: source :class:`ReducedGaussianGrid`.
        dst_lat: ``(nlat,)`` destination latitudes in degrees (any order).
        dst_lon: ``(nlon,)`` destination longitudes in degrees.
        out_file: if given, write the ESMF-layout weight file (variables
            ``row``, ``col``, ``S``, ``xc_a``/``yc_a``, ``xc_b``/``yc_b``,
            ``src_grid_dims``, ``dst_grid_dims``) readable by
            ``credit.preblock.regrid.Regridder``.

    Returns:
        The weight file content as an :class:`xarray.Dataset`.
    """
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64) % 360.0
    nlat, nlon = len(dst_lat), len(dst_lon)

    ring_lats = grid.ring_lats  # strictly decreasing

    # --- latitude bracketing (vectorized over dst_lat) ---------------------
    # k = index of the last ring with ring_lat >= lat_d  (ring k above, k+1 below)
    k = np.searchsorted(-ring_lats, -dst_lat, side="right") - 1
    north_clamp = k < 0  # poleward of the first ring
    south_clamp = k >= grid.n_rings - 1  # on/last ring or poleward of it
    k = np.clip(k, 0, grid.n_rings - 2)
    denom = ring_lats[k] - ring_lats[k + 1]
    w_south = (ring_lats[k] - dst_lat) / denom  # weight of ring k+1
    w_south = np.where(north_clamp, 0.0, np.where(south_clamp, 1.0, w_south))
    w_south = np.clip(w_south, 0.0, 1.0)
    k = np.where(south_clamp, grid.n_rings - 2, k)

    rows, cols, vals = [], [], []
    dst_index = np.arange(nlat * nlon).reshape(nlat, nlon)

    for ring_side in (0, 1):  # 0 = northern ring k, 1 = southern ring k+1
        ring = k + ring_side
        w_ring = (1.0 - w_south) if ring_side == 0 else w_south
        active = w_ring > 0.0
        if not np.any(active):
            continue
        ring_a = ring[active]
        w_a = w_ring[active]
        n_j = grid.ring_nlon[ring_a]  # (n_active_lat,)
        off_j = grid.ring_offset[ring_a]
        # longitude interpolation on each active ring x every dst_lon
        pos = dst_lon[None, :] * n_j[:, None] / 360.0  # (n_active_lat, nlon)
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0
        i0 = i0 % n_j[:, None]
        i1 = (i0 + 1) % n_j[:, None]
        r = dst_index[active, :]  # (n_active_lat, nlon)
        for idx, w_lon in ((i0, 1.0 - frac), (i1, frac)):
            w_total = w_a[:, None] * w_lon
            keep = w_total > 0.0
            rows.append(r[keep])
            cols.append((off_j[:, None] + idx)[keep])
            vals.append(w_total[keep])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    # Merge duplicate (row, col) pairs (e.g. frac == 0 wrap cases).
    key = rows * np.int64(grid.n_points) + cols
    order = np.argsort(key, kind="stable")
    key, rows, cols, vals = key[order], rows[order], cols[order], vals[order]
    uniq, start = np.unique(key, return_index=True)
    vals = np.add.reduceat(vals, start)
    rows = rows[start]
    cols = cols[start]

    lat2d = np.repeat(dst_lat, nlon)
    lon2d = np.tile(dst_lon, nlat)
    ds = xr.Dataset(
        {
            # ESMF weight files are 1-based.
            "row": (("n_s",), rows.astype(np.int32) + 1),
            "col": (("n_s",), cols.astype(np.int32) + 1),
            "S": (("n_s",), vals),
            "xc_a": (("n_a",), grid.lons_flat()),
            "yc_a": (("n_a",), grid.lats_flat()),
            "xc_b": (("n_b",), lon2d),
            "yc_b": (("n_b",), lat2d),
            # ESMF convention: [nlon, nlat]; Regridder reverses to [nlat, nlon].
            "dst_grid_dims": (("dst_grid_rank",), np.array([nlon, nlat], dtype=np.int32)),
            "src_grid_dims": (("src_grid_rank",), np.array([grid.n_points], dtype=np.int32)),
        },
        attrs={"map_method": "Bilinear remapping (reduced Gaussian rings)", "conventions": "ESMF-like"},
    )
    if out_file is not None:
        ds.to_netcdf(out_file)
    return ds


def fetch_arco_n320_grid(out_file: str | None = None) -> ReducedGaussianGrid:
    """Fetch the N320 ring structure from the ARCO ERA5 ``co/`` store coordinates.

    Reads only the per-point ``latitude``/``longitude`` coordinate vectors
    (~9 MB, anonymous GCS access) from the single-level reanalysis store and
    optionally saves the ring description for the ``spectral_to_grid``
    preblock's ``grid_file`` argument. The ``arco_era5_co`` dataset also writes
    this file to ``save_loc`` automatically on first read; this helper exists
    for the chicken-and-egg case where the preblock (built at trainer init)
    needs the file before any data has been read.
    """
    import gcsfs

    fs = gcsfs.GCSFileSystem(token="anon")
    url = "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2"
    with xr.open_zarr(fs.get_mapper(url), chunks=None) as ds:
        grid = ReducedGaussianGrid.from_points(ds["latitude"].values, ds["longitude"].values)
    if out_file is not None:
        grid.save(out_file)
    return grid
