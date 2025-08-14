import numpy as np
import xarray as xr


def scrip_from_latlon_grid(lons, lats, grid_name, grid_file):
    """

    Args:
        lons:
        lats:
        grid_name:
        grid_file:

    Returns:

    """
    dlat = np.abs(lats[1:] - lats[:-1])
    dlat = np.append(dlat, dlat[-1])
    dlon = np.abs(lons[1:] - lons[:-1])
    dlon = np.append(dlon, dlon[-1])
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    grid_indices = np.indices(lat_grid.shape)
    row_indices = grid_indices[0].ravel()
    col_indices = grid_indices[1].ravel()
    grid_size = lat_grid.size
    grid_corners = 4
    grid_dims = xr.DataArray(
        np.array(lat_grid.shape, dtype=np.int64), dims=("grid_rank",), name="grid_dims"
    )
    grid_center_lon = xr.DataArray(
        lon_grid.ravel(), dims=("grid_size",), name="grid_center_lon"
    )
    grid_center_lat = xr.DataArray(
        lat_grid.ravel(), dims=("grid_size",), name="grid_center_lat"
    )
    grid_imask = xr.DataArray(
        np.ones(grid_size, dtype=np.int64), dims=("grid_size",), name="grid_imask"
    )
    grid_corner_lat = xr.DataArray(
        np.zeros((grid_size, grid_corners)),
        dims=("grid_size", "grid_corners"),
        name="grid_corner_lat",
    )
    grid_corner_lon = xr.DataArray(
        np.zeros((grid_size, grid_corners)),
        dims=("grid_size", "grid_corners"),
        name="grid_corner_lat",
    )
    # corners are defined in counter clockwise order starting from the bottom left corner
    lon_sign = [-1, 1, 1, -1]
    lat_sign = [-1, -1, 1, 1]
    for i in range(grid_corners):
        grid_corner_lat[:, i] = lat_grid.ravel() + lat_sign[i] * dlat[row_indices] / 2.0
        grid_corner_lon[:, i] = lon_grid.ravel() + lon_sign[i] * dlon[col_indices] / 2.0
    scrip_ds = xr.Dataset(
        {
            "grid_dims": grid_dims,
            "grid_center_lon": grid_center_lon,
            "grid_center_lat": grid_center_lat,
            "grid_imask": grid_imask,
            "grid_corner_lat": grid_corner_lat,
            "grid_corner_lon": grid_corner_lon,
        }
    )
    scrip_ds.attrs["title"] = grid_name
    scrip_ds.to_netcdf(grid_file)
    return scrip_ds
