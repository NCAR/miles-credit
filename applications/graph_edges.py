from haversine import haversine_vector, Unit
import xarray as xr
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from os.path import join, exists
from os import makedirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--coord", help="Path to xarray file containing coordinates"
    )
    parser.add_argument("-o", "--out", help="Path to output directory")
    parser.add_argument(
        "-d", "--dist", type=float, help="Max distance for adjacency (km)"
    )
    parser.add_argument("-p", "--procs", type=int, help="Number of processes")
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
    chunk_size = int(lon_flat.size / args.procs)
    coord_set = np.array([[i, lat_flat[i], lon_flat[i]] for i in range(lon_flat.size)])
    calc_edge_p = partial(calc_edges, all_coords=coord_set, max_dist=args.dist)
    p = Pool(processes=args.procs, maxtasksperchild=1)
    results = p.map(calc_edge_p, coord_set, chunksize=chunk_size)
    p.close()
    p.join()
    edge_indices_list = [x[0] for x in results]
    dist_list = [x[1] for x in results]
    edge_indices_arr = np.vstack(edge_indices_list)
    dist_arr = np.concatenate(dist_list).ravel()
    output_ds = xr.Dataset(
        {
            "edges": (("node", "pair"), edge_indices_arr),
            "distances": (("node",), dist_arr),
            "longitude": (("index",), lon_flat),
            "latitude": (("index",), lat_flat),
        },
        coords={"index": list(range(lon_flat.size))},
        attrs=dict(coord_file=args.coord, max_distance=args.dist),
    )
    if not exists(args.out):
        makedirs(args.out)
    print("Saving to " + args.out)
    output_ds.to_netcdf(
        join(args.out, f"grid_edge_pairs_{args.dist:0.0f}_{resolution}.nc")
    )
    return


def calc_edges(coord, all_coords=None, max_dist=25.0):
    coord_distances = haversine_vector(
        coord[1:], all_coords[:, 1:], Unit.KILOMETERS, comb=True
    )
    close_coords = np.where(coord_distances < max_dist)[0]
    edge_distances = coord_distances[close_coords]
    edge_indices = [(int(np.round(coord[0])), c) for c in close_coords]
    if coord[0] % 100 == 0:
        print(coord, edge_distances.size)
    return edge_indices, edge_distances


if __name__ == "__main__":
    main()
