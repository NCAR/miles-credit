"""
scrip_generator.py
==================
Generate SCRIP-format NetCDF files for use with ESMFRegridWeightGen.

Supports two grid types:
    - Rectilinear  : 1D lon + 1D lat arrays (uniform, regional, or Gaussian spacing)
    - Curvilinear  : 2D lon + 2D lat arrays (HRRR, GOES, ocean tripolar, etc.)

All functions produce SCRIP files containing both cell centers and cell corners,
making them compatible with all ESMFRegridWeightGen methods (bilinear, patch,
nearest-neighbor, 1st- and 2nd-order conservative).

Entry points
------------
    scrip_from_netcdf(nc_file, scrip_file, grid_name=None, mask_var=None)
        Auto-detects grid type from file; preferred entry point.

    scrip_from_rectilinear(lon_1d, lat_1d, grid_name, grid_file, mask=None)
    scrip_from_curvilinear(lon_2d, lat_2d, grid_name, grid_file, mask=None)
        Direct API for when coordinates are already in hand.

Dependencies
------------
    numpy, xarray
"""

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Shared SCRIP writer
# ---------------------------------------------------------------------------


def _write_scrip(lon_centers, lat_centers, lon_corners, lat_corners, grid_name, grid_file, mask=None, grid_dims=None):
    """
    Write a SCRIP-format NetCDF file from pre-computed center and corner arrays.

    Parameters
    ----------
    lon_centers : np.ndarray, shape (grid_size,)
    lat_centers : np.ndarray, shape (grid_size,)
    lon_corners : np.ndarray, shape (grid_size, 4)
        Corner longitudes in CCW order: SW, SE, NE, NW.
    lat_corners : np.ndarray, shape (grid_size, 4)
        Corner latitudes in CCW order: SW, SE, NE, NW.
    grid_name   : str
    grid_file   : str
    mask        : np.ndarray of int32, shape (grid_size,), optional
        1 = valid, 0 = masked.  Defaults to all-ones (fully unmasked).

    Returns
    -------
    xr.Dataset written to grid_file.
    """
    grid_size = lon_centers.size
    if mask is None:
        mask = np.ones(grid_size, dtype=np.int32)
    else:
        mask = np.asarray(mask, dtype=np.int32).ravel()

    # grid_dims: [nlon, nlat] for structured grids (SCRIP convention: lon first).
    # ESMF uses len(grid_dims) to determine grid rank: 2 → structured (bilinear
    # works at poles without extra flags), 1 → unstructured.
    if grid_dims is None:
        _grid_dims = np.array([grid_size], dtype=np.int32)
    else:
        _grid_dims = np.asarray(grid_dims, dtype=np.int32)

    def _da(data, dims, name, units):
        da = xr.DataArray(data, dims=dims, name=name, attrs={"units": units})
        if np.issubdtype(data.dtype, np.floating):
            da.encoding["_FillValue"] = None
        return da

    ds = xr.Dataset(
        {
            "grid_dims": _da(_grid_dims, ("grid_rank",), "grid_dims", "unitless"),
            "grid_center_lon": _da(lon_centers.astype(np.float64), ("grid_size",), "grid_center_lon", "degrees"),
            "grid_center_lat": _da(lat_centers.astype(np.float64), ("grid_size",), "grid_center_lat", "degrees"),
            "grid_imask": _da(mask, ("grid_size",), "grid_imask", "unitless"),
            "grid_corner_lon": _da(
                lon_corners.astype(np.float64), ("grid_size", "grid_corners"), "grid_corner_lon", "degrees"
            ),
            "grid_corner_lat": _da(
                lat_corners.astype(np.float64), ("grid_size", "grid_corners"), "grid_corner_lat", "degrees"
            ),
        }
    )
    ds.attrs["title"] = grid_name
    ds.to_netcdf(grid_file, format="NETCDF3_64BIT")
    return ds


# ---------------------------------------------------------------------------
# Corner computation
# ---------------------------------------------------------------------------


def _corners_rectilinear(lons_1d, lats_1d):
    """
    Compute cell corners for a rectilinear grid from 1D center arrays.

    Edges are midpoints between adjacent centers.  Boundary edges are
    extrapolated using the nearest interior spacing.  Latitude edges are
    clamped to [-90, 90].  Longitude edges are kept monotonically increasing
    (not wrapped) so that west < east is guaranteed for every cell.

    Returns
    -------
    lon_corners, lat_corners : np.ndarray, shape (nlat*nlon, 4)
        CCW order: SW, SE, NE, NW.
    """
    nlat, nlon = lats_1d.size, lons_1d.size

    lat_edges = np.empty(nlat + 1)
    lat_edges[1:-1] = 0.5 * (lats_1d[:-1] + lats_1d[1:])
    lat_edges[0] = lats_1d[0] - (lat_edges[1] - lats_1d[0])
    lat_edges[-1] = lats_1d[-1] + (lats_1d[-1] - lat_edges[-2])
    lat_edges = np.clip(lat_edges, -90.0, 90.0)

    lon_edges = np.empty(nlon + 1)
    lon_edges[1:-1] = 0.5 * (lons_1d[:-1] + lons_1d[1:])
    lon_edges[0] = lons_1d[0] - (lon_edges[1] - lons_1d[0])
    lon_edges[-1] = lons_1d[-1] + (lons_1d[-1] - lon_edges[-2])

    jj, ii = np.indices((nlat, nlon))
    row, col = jj.ravel(), ii.ravel()

    south = lat_edges[row]
    north = lat_edges[row + 1]
    west = lon_edges[col]
    east = lon_edges[col + 1]

    lat_corners = np.stack([south, south, north, north], axis=1)
    lon_corners = np.stack([west, east, east, west], axis=1)
    return lon_corners, lat_corners


def _corners_curvilinear(lons_2d, lats_2d):
    """
    Compute cell corners for a curvilinear grid from 2D center arrays.

    Interior corners are the average of the four surrounding cell centers.
    Boundary corners are extrapolated using a one-cell ghost layer.

    Returns
    -------
    lon_corners, lat_corners : np.ndarray, shape (ny*nx, 4)
        CCW order: SW, SE, NE, NW.
    """
    ny, nx = lons_2d.shape

    def _extend(arr):
        """Pad array by one cell on all sides via linear extrapolation."""
        out = np.empty((ny + 2, nx + 2), dtype=np.float64)
        out[1:-1, 1:-1] = arr
        out[0, 1:-1] = 2 * arr[0, :] - arr[1, :]
        out[-1, 1:-1] = 2 * arr[-1, :] - arr[-2, :]
        out[1:-1, 0] = 2 * arr[:, 0] - arr[:, 1]
        out[1:-1, -1] = 2 * arr[:, -1] - arr[:, -2]
        out[0, 0] = 2 * arr[0, 0] - arr[1, 1]
        out[0, -1] = 2 * arr[0, -1] - arr[1, -2]
        out[-1, 0] = 2 * arr[-1, 0] - arr[-2, 1]
        out[-1, -1] = 2 * arr[-1, -1] - arr[-2, -2]
        return out

    lon_ext = _extend(lons_2d)
    lat_ext = _extend(lats_2d)

    # Average four surrounding centers to get corner positions
    c_lon = 0.25 * (lon_ext[:-1, :-1] + lon_ext[:-1, 1:] + lon_ext[1:, :-1] + lon_ext[1:, 1:])
    c_lat = 0.25 * (lat_ext[:-1, :-1] + lat_ext[:-1, 1:] + lat_ext[1:, :-1] + lat_ext[1:, 1:])

    jj, ii = np.indices((ny, nx))
    j, i = jj.ravel(), ii.ravel()

    lon_corners = np.stack([c_lon[j, i], c_lon[j, i + 1], c_lon[j + 1, i + 1], c_lon[j + 1, i]], axis=1)
    lat_corners = np.stack([c_lat[j, i], c_lat[j, i + 1], c_lat[j + 1, i + 1], c_lat[j + 1, i]], axis=1)
    return lon_corners, lat_corners


# ---------------------------------------------------------------------------
# Coordinate detection  (used by scrip_from_netcdf)
# ---------------------------------------------------------------------------

# Supported name pairs in priority order: (lon_name, lat_name)
_COORD_CANDIDATES = [
    ("longitude", "latitude"),
    ("lon", "lat"),
]


def _find_coord_pair(ds):
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


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def scrip_from_rectilinear(lon_1d, lat_1d, grid_name, grid_file, mask=None):
    """
    Generate a SCRIP file for a rectilinear (regular lat/lon) grid.

    Parameters
    ----------
    lon_1d    : array-like, shape (nlon,)
    lat_1d    : array-like, shape (nlat,)
    grid_name : str
    grid_file : str
    mask      : array-like, shape (nlat, nlon), optional  1=valid, 0=masked.

    Returns
    -------
    xr.Dataset
    """
    lons = np.mod(np.asarray(lon_1d, dtype=float), 360.0)
    lons = lons[np.argsort(lons)]
    lats = np.asarray(lat_1d, dtype=float)
    lats = lats[np.argsort(lats)]

    lon_corners, lat_corners = _corners_rectilinear(lons, lats)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    flat_mask = np.asarray(mask, dtype=np.int32).ravel() if mask is not None else None

    return _write_scrip(
        lon_grid.ravel(),
        lat_grid.ravel(),
        lon_corners,
        lat_corners,
        grid_name,
        grid_file,
        mask=flat_mask,
        grid_dims=np.array([lons.size, lats.size], dtype=np.int32),  # [nlon, nlat]
    )


def scrip_from_curvilinear(lon_2d, lat_2d, grid_name, grid_file, mask=None):
    """
    Generate a SCRIP file for a curvilinear grid.

    Parameters
    ----------
    lon_2d    : array-like, shape (ny, nx)
    lat_2d    : array-like, shape (ny, nx)
    grid_name : str
    grid_file : str
    mask      : array-like, shape (ny, nx), optional  1=valid, 0=masked.

    Returns
    -------
    xr.Dataset
    """
    lon_2d = np.asarray(lon_2d, dtype=float)
    lat_2d = np.asarray(lat_2d, dtype=float)
    assert lon_2d.shape == lat_2d.shape, "lon_2d and lat_2d must have the same shape"
    assert lon_2d.ndim == 2, "lon_2d and lat_2d must be 2D arrays"

    lon_corners, lat_corners = _corners_curvilinear(lon_2d, lat_2d)

    flat_mask = np.asarray(mask, dtype=np.int32).ravel() if mask is not None else None

    ny, nx = lon_2d.shape
    return _write_scrip(
        lon_2d.ravel(),
        lat_2d.ravel(),
        lon_corners,
        lat_corners,
        grid_name,
        grid_file,
        mask=flat_mask,
        grid_dims=np.array([nx, ny], dtype=np.int32),  # [nx, ny] — lon dim first
    )


def scrip_from_netcdf(nc_file, scrip_file, grid_name=None, mask_var=None):
    """
    Auto-detect grid type from a NetCDF file and write a SCRIP file.

    Coordinate names supported: lon/lat, longitude/latitude.
    1D coordinates → rectilinear.  2D coordinates → curvilinear.

    Parameters
    ----------
    nc_file    : str   Path to input NetCDF file.
    scrip_file : str   Path for output SCRIP file.
    grid_name  : str, optional  Defaults to the input filename stem.
    mask_var   : str, optional  Variable to use as mask (1=valid, 0=masked).

    Returns
    -------
    xr.Dataset
    """
    import os

    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"Input file not found: {nc_file}")

    if grid_name is None:
        grid_name = os.path.splitext(os.path.basename(nc_file))[0]

    ds = xr.open_dataset(nc_file)

    lon_arr, lat_arr, lon_name, lat_name = _find_coord_pair(ds)
    print(f"[scrip_from_netcdf] Coordinates : {lon_name} / {lat_name}")

    if lon_arr.ndim == 1 and lat_arr.ndim == 1:
        grid_type = "rectilinear"
    elif lon_arr.ndim == 2 and lat_arr.ndim == 2:
        grid_type = "curvilinear"
    else:
        raise ValueError(
            f"Unexpected coordinate shapes: lon={lon_arr.shape}, lat={lat_arr.shape}. "
            "Expected both 1D (rectilinear) or both 2D (curvilinear)."
        )
    print(f"[scrip_from_netcdf] Grid type   : {grid_type} (lon shape: {lon_arr.shape})")

    mask = None
    if mask_var is not None:
        if mask_var not in ds:
            raise ValueError(
                f"mask_var '{mask_var}' not found. Available: {sorted(set(ds.data_vars) | set(ds.coords))}"
            )
        mask = ds[mask_var].values.astype(np.int32)

    ds.close()

    if grid_type == "rectilinear":
        result = scrip_from_rectilinear(lon_arr, lat_arr, grid_name, scrip_file, mask=mask)
    else:
        result = scrip_from_curvilinear(lon_arr, lat_arr, grid_name, scrip_file, mask=mask)

    print(f"[scrip_from_netcdf] SCRIP written: {scrip_file}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Generate a SCRIP-format file from a NetCDF grid file.")
    parser.add_argument("-i", "--input", default=None, help="Input NetCDF file path")
    parser.add_argument("-o", "--output", default=None, help="Output SCRIP file path")
    parser.add_argument("-n", "--name", default=None, help="Grid name (default: input filename stem)")
    parser.add_argument("-m", "--mask", default=None, help="Mask variable name in input file")
    parser.add_argument("--test", action="store_true", help="Run synthetic tests")
    args = parser.parse_args()

    if args.input and args.output:
        scrip_from_netcdf(args.input, args.output, grid_name=args.name, mask_var=args.mask)

    elif args.test:
        outdir = "/tmp/scrip_test"
        os.makedirs(outdir, exist_ok=True)

        # Rectilinear
        scrip_from_rectilinear(
            np.arange(0.0, 360.0, 1.0),
            np.arange(-90.0, 91.0, 1.0),
            "test_rectilinear",
            f"{outdir}/test_rectilinear_scrip.nc",
        )
        print("Rectilinear SCRIP written.")

        # Curvilinear
        lo = np.linspace(-134.0, -60.0, 64)
        la = np.linspace(21.0, 52.0, 64)
        lo2, la2 = np.meshgrid(lo, la)
        lo2 += 0.01 * np.sin(np.radians(la2))
        la2 += 0.01 * np.cos(np.radians(lo2))
        scrip_from_curvilinear(lo2, la2, "test_curvilinear", f"{outdir}/test_curvilinear_scrip.nc")
        print("Curvilinear SCRIP written.")

        # scrip_from_netcdf: rectilinear (lon/lat names)
        lon_s, lat_s = np.arange(0.0, 10.0, 1.0), np.arange(30.0, 40.0, 1.0)
        ds_r = xr.Dataset(
            {"t": (["lat", "lon"], np.random.randn(lat_s.size, lon_s.size))},
            coords={"lon": lon_s, "lat": lat_s},
        )
        rect_nc = f"{outdir}/synthetic_rect.nc"
        ds_r.to_netcdf(rect_nc)
        print("\n--- scrip_from_netcdf: rectilinear ---")
        scrip_from_netcdf(rect_nc, f"{outdir}/auto_rect_scrip.nc")

        # scrip_from_netcdf: curvilinear (longitude/latitude names)
        lo2s, la2s = lo2[:8, :10].copy(), la2[:8, :10].copy()
        ds_c = xr.Dataset(
            {"r": (["y", "x"], np.random.randn(8, 10))},
            coords={"longitude": (["y", "x"], lo2s), "latitude": (["y", "x"], la2s)},
        )
        curv_nc = f"{outdir}/synthetic_curv.nc"
        ds_c.to_netcdf(curv_nc)
        print("\n--- scrip_from_netcdf: curvilinear ---")
        scrip_from_netcdf(curv_nc, f"{outdir}/auto_curv_scrip.nc")

        print(f"\nAll test SCRIP files written to {outdir}/")

    else:
        parser.print_help()
