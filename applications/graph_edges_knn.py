import xarray as xr
import argparse
import numpy as np
from os.path import join, exists
from os import makedirs
from sklearn.neighbors import BallTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--coord", help="Path to xarray file containing coordinates"
    )
    parser.add_argument("-o", "--out", help="Path to output directory")
    parser.add_argument("-k", "--k_neigh", type=int, help="Number of neighbors")
    # parser.add_argument("-p", "--procs", type=int, help="Number of processes")
    args = parser.parse_args()
    coords = xr.open_dataset(args.coord)
    lon = coords["longitude"].values
    lon[lon > 180] = lon[lon > 180] - 360.0
    lat = coords["latitude"].values

    resolution = "onedeg" if abs(lat[1] - lat[0]) > 0.5 else "quarter"

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    print("Size:", lon_flat.size)

    lat_lon = np.stack([lat_flat, lon_flat], axis=-1)
    rad_lat_lon = np.deg2rad(lat_lon)
    tree = BallTree(rad_lat_lon, metric="haversine")

    distances, indices = tree.query(rad_lat_lon, k=args.k_neigh)

    node_indices = np.arange(len(rad_lat_lon)).reshape(-1, 1)
    node_indices = np.tile(node_indices, reps=args.k_neigh)

    EARTH_RADIUS = 6_371  # in km
    dist_arr = distances.reshape(-1) * EARTH_RADIUS
    edge_indices_arr = np.stack(
        [indices.reshape(-1), node_indices.reshape(-1)], axis=-1
    )

    output_ds = xr.Dataset(
        {
            "edges": (("node", "pair"), edge_indices_arr),
            "distances": (("node",), dist_arr),
            "longitude": (("index",), lon_flat),
            "latitude": (("index",), lat_flat),
        },
        coords={"index": list(range(lon_flat.size))},
        attrs=dict(coord_file=args.coord, k_neighbors=args.k_neigh),
    )
    if not exists(args.out):
        makedirs(args.out)
    filename = join(args.out, f"grid_edge_pairs_k_{args.k_neigh}_{resolution}.nc")
    output_ds.to_netcdf(filename)
    print("Saved to " + filename)


if __name__ == "__main__":
    main()
