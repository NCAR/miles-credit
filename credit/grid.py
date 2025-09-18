import numpy as np
import xarray as xr


def scrip_from_latlon_grid(lons, lats, grid_name, grid_file):
    """
    Generate a SCRIP netCDF file defining the centers and corners of each element in a latitude-longitude grid.
    This function supports both equiangular lat-lon grids and full Gaussian grids.

    SCRIP is a legacy software package for regridding model output. The SCRIP grid format is supported by ESMF for
    regridding definitions. See https://github.com/SCRIP-Project/SCRIP/blob/master/SCRIP/doc/SCRIPusers.pdf for
    details about SCRIP and

    Args:
        lons: 1D array of longitudes
        lats: 1D array of latitudes
        grid_name: Name of the output grid map
        grid_file: path to netCDF file containing the grid information

    Returns:
        scrip_ds: xr.Dataset containing grid information in SCRIP format.
    """
    dlat = np.abs(lats[1:] - lats[:-1])
    dlat = np.append(dlat, dlat[-1])
    dlon = np.abs(lons[1:] - lons[:-1])
    dlon = np.append(dlon, dlon[-1])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_indices = np.indices(lat_grid.shape)
    row_indices = grid_indices[0].ravel()
    col_indices = grid_indices[1].ravel()
    grid_size = lat_grid.size
    grid_corners = 4
    grid_dims = xr.DataArray(
        np.array(lat_grid.shape[::-1], dtype=np.int32),
        dims=("grid_rank",),
        name="grid_dims",
    )
    grid_center_lon = xr.DataArray(
        lon_grid.ravel(),
        dims=("grid_size",),
        name="grid_center_lon",
        attrs=dict(units="degrees"),
    )
    grid_center_lat = xr.DataArray(
        lat_grid.ravel(),
        dims=("grid_size",),
        name="grid_center_lat",
        attrs=dict(units="degrees"),
    )
    grid_imask = xr.DataArray(
        np.ones(grid_size, dtype=np.int32),
        dims=("grid_size",),
        name="grid_imask",
        attrs=dict(units="unitless"),
    )
    grid_corner_lat = xr.DataArray(
        np.zeros((grid_size, grid_corners)),
        dims=("grid_size", "grid_corners"),
        name="grid_corner_lat",
        attrs=dict(units="degrees"),
    )
    grid_corner_lon = xr.DataArray(
        np.zeros((grid_size, grid_corners)),
        dims=("grid_size", "grid_corners"),
        name="grid_corner_lat",
        attrs=dict(units="degrees"),
    )
    grid_corner_lon.encoding["_FillValue"] = None
    grid_corner_lat.encoding["_FillValue"] = None
    grid_center_lat.encoding["_FillValue"] = None
    grid_center_lon.encoding["_FillValue"] = None
    # corners are defined in counter clockwise order starting from the bottom left corner
    lon_sign = [-1, 1, 1, -1]
    lat_sign = [-1, -1, 1, 1]
    for i in range(grid_corners):
        grid_corner_lat[:, i] = np.minimum(
            np.maximum(lat_grid.ravel() + lat_sign[i] * dlat[row_indices] / 2.0, -90),
            90,
        )
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
    scrip_ds.to_netcdf(grid_file, format="NETCDF3_64BIT")
    return scrip_ds


if __name__ == "__main__":
    grid_file = "/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/AIWQ_inits/b.e21.CREDIT_climate_branch_allvars_GEFSicefracandsst_2025-08-08.zarr"
    ds = xr.open_zarr(grid_file)
    print(ds)
    scrip_from_latlon_grid(
        ds["longitude"].values,
        ds["latitude"].values,
        "camulator_grid",
        "/glade/work/dgagne/camulator_grid_scrip.nc",
    )
