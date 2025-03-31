from credit.solar import get_solar_radiation_loc, get_toa_radiation
from mpi4py import MPI
import argparse
import xarray as xr
import numpy as np
import os
import pandas as pd


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    parser = argparse.ArgumentParser(description="Calculate global solar radiation")
    parser.add_argument("-s", "--start", type=str, default="2000-01-01", help="Start date (inclusive)")
    parser.add_argument("-e", "--end", type=str, default="2000-12-31 23:00", help="End date (inclusive")
    parser.add_argument("-t", "--step", type=str, default="1h", help="Step frequency")
    parser.add_argument("-u", "--sub", type=str, default="10Min", help="Sub Frequency")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="/glade/u/home/wchapman/MLWPS/DataLoader/LSM_static_variables_ERA5_zhght.nc",
        help="File containing longitudes, latitudes, and geopotential height.",
    )
    parser.add_argument(
        "-g",
        "--geo",
        type=str,
        default="Z_GDS4_SFC",
        help="Geopotential height variable.",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("-z", "--zarr", action="store_true", help="Output as zarr files.")
    args = parser.parse_args()
    grid_points_sub = None
    start_date_ts = pd.Timestamp(args.start)
    end_date_ts = pd.Timestamp(args.end)
    toa_radiation = None
    if rank == 0:
        dates = pd.date_range(start=start_date_ts, end=end_date_ts, freq=args.step)
        with xr.open_dataset(args.input) as static_ds:
            lons = static_ds["longitude"].values
            lats = static_ds["latitude"].values
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            solar_grid = xr.DataArray(
                data=np.zeros((dates.size, lats.size, lons.size), dtype=np.float32),
                coords={"time": dates, "longitude": lons, "latitude": lats},
                dims=("time", "latitude", "longitude"),
                name="tsi",
                attrs={"long_name": "total solar irradiance", "units": "J m-2"},
            )
            heights = static_ds[args.geo].values / 9.81
            grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel(), heights.ravel()]).T
            split_indices = np.round(np.linspace(0, grid_points.shape[0], size + 1)).astype(int)
            grid_points_sub = [
                grid_points[split_indices[s] : split_indices[s + 1]] for s in range(split_indices.size - 1)
            ]
        toa_radiation = get_toa_radiation(args.start, args.end, step_freq=args.step, sub_freq=args.sub)
    toa_radiation = comm.bcast(toa_radiation, root=0)
    rank_points = comm.scatter(grid_points_sub, root=0)
    print(rank_points.shape)
    for r, rank_point in enumerate(rank_points):
        if r % 100 == 0:
            print(f"{rank}: {r:d}/{rank_points.shape[0]:d}")
        solar_point = get_solar_radiation_loc(
            toa_radiation,
            rank_point[0],
            rank_point[1],
            rank_point[2],
            args.start,
            args.end,
            step_freq=args.step,
            sub_freq=args.sub,
        )
        if rank > 0:
            comm.Send(
                np.concatenate(
                    [
                        solar_point["latitude"].values,
                        solar_point["longitude"].values,
                        solar_point["tsi"].values.ravel(),
                    ]
                ),
                dest=0,
                tag=rank,
            )
        else:
            solar_grid.loc[:, solar_point["latitude"], solar_point["longitude"]] = solar_point["tsi"].values
            for sr in range(1, size):
                other_point = np.empty(2 + solar_grid.shape[0], dtype=solar_point["tsi"].dtype)
                comm.Recv(other_point, source=sr, tag=sr)
                solar_grid.loc[:, other_point[0], other_point[1]] = other_point[2:]

    if rank == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        date_format = "%Y-%m-%d_%H%M"
        start_date_str = start_date_ts.strftime(date_format)
        end_date_str = end_date_ts.strftime(date_format)
        print("Saving")
        if args.zarr:
            filename = f"solar_irradiance_{start_date_str}_{end_date_str}.zarr"
            solar_grid.to_zarr(
                os.path.join(args.output, filename),
                mode="w",
                encoding={
                    "tsi": {
                        "chunks": (
                            1,
                            solar_grid.shape[1],
                            solar_grid.shape[2],
                        )
                    }
                },
            )
        else:
            filename = f"solar_irradiance_{start_date_str}_{end_date_str}.nc"
            solar_grid.to_netcdf(
                os.path.join(args.output, filename),
                encoding={
                    "tsi": {
                        "zlib": True,
                        "complevel": 1,
                        "shuffle": True,
                        "chunksizes": (
                            1,
                            solar_grid.shape[1],
                            solar_grid.shape[2],
                        ),
                    }
                },
            )
    return


if __name__ == "__main__":
    main()
