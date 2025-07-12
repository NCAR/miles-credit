"""Same as `graph_edges_spread.py` for reduced gaussian grid"""

import xarray as xr
import argparse
import numpy as np
from os.path import join, exists
from os import makedirs
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm
from graph_edges_spread import construct_graph, gen_per_lat_coords
import pandas as pd
import numpy as np


def get_spaced_lon(num):
    lon = np.linspace(0, 360, num, endpoint=False)
    assert num == len(lon)
    lon[lon > 180] = lon[lon > 180] - 360.0
    return lon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--coord", help="Path to csv file containing coordinates")
    # For example credit/models/graph_partition//graph_grid_gaussia/n48.csv 
    # Please see folder credit/models/graph_partition//graph_grid_gaussia for more details
    parser.add_argument("-o", "--out", help="Path to output directory")
    parser.add_argument("-r", "--radius", type=int, help="Number of hops")
    # parser.add_argument("-p", "--procs", type=int, help="Number of processes")
    args = parser.parse_args()
    df = pd.read_csv(args.coord)

    lat = df.latitude.values
    red_pts = df.reduced_points.values
    lat_lon_combo = [(l_i, get_spaced_lon(r_p_i)) for l_i, r_p_i in zip(lat, red_pts)]
    per_lat_coords = gen_per_lat_coords(lat_lon_combo)
    output_ds = construct_graph(per_lat_coords, args.radius)
    output_ds = output_ds.assign_attrs(coord_file=args.coord, radius=args.radius)
    if not exists(args.out):
        makedirs(args.out)
    gauss_name = f"n{int(len(lat) // 2)}"
    filename = join(args.out, f"grid_edge_pairs_r_{args.radius}_{gauss_name}.nc")
    output_ds.to_netcdf(filename)
    print("Saved to " + filename)


if __name__ == "__main__":
    main()
