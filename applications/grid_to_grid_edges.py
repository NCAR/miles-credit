"""
Generates edges between two grids or graph representations of the globe
"""
import xarray as xr
import argparse
import numpy as np
from os.path import join, exists
from os import makedirs
from sklearn.neighbors import BallTree
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help="Config file")
    parser.add_argument("-s", "--source", type=str, help="Source grid")
    parser.add_argument("-t", "--target", type=str, help="Target grid")
    parser.add_argument("-o", "--out", help="Path to output directory")
    parser.add_argument(
        "-k",
        "--k_neigh",
        type=int,
        help="Number of connections between source and target nodes",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    source = xr.open_dataset(config[args.source])
    target = xr.open_dataset(config[args.target])

    source_lat_lon = np.deg2rad(
        np.stack([source.latitude.values, source.longitude.values], axis=1)
    )
    target_lat_lon = np.deg2rad(
        np.stack([target.latitude.values, target.longitude.values], axis=1)
    )

    tree = BallTree(target_lat_lon, metric="haversine")

    distances, target_indices = tree.query(source_lat_lon, k=args.k_neigh)

    source_indices = source.node_idx.values.reshape(-1, 1)
    source_indices = np.tile(source_indices, reps=args.k_neigh)

    EARTH_RADIUS = 6_371  # in km
    dist_arr = distances.reshape(-1) * EARTH_RADIUS
    edge_indices_arr = np.stack(
        [source_indices.reshape(-1), target_indices.reshape(-1)], axis=-1
    )

    output_ds = xr.Dataset(
        {
            "edges": (("edge_idx", "pair"), edge_indices_arr),
            "distances": (("edge_idx",), dist_arr),
            "source_longitude": (("source_node_idx",), source.longitude.data),
            "source_latitude": (("source_node_idx",), source.latitude.data),
            "target_longitude": (("target_node_idx",), target.longitude.data),
            "target_latitude": (("target_node_idx",), target.latitude.data),
        },
        coords={
            "source_node_idx": np.arange(len(source.latitude)),
            "target_node_idx": np.arange(len(target.latitude)),
        },
        attrs=dict(
            source_grid=args.source, target_grid=args.target, k_neighbors=args.k_neigh
        ),
    )

    if not exists(args.out):
        makedirs(args.out)
    filename = join(
        args.out, f"grid_edge_pairs_{args.source}_to_{args.target}_k_{args.k_neigh}.nc"
    )
    output_ds.to_netcdf(filename)
    print("Saved to " + filename)


if __name__ == "__main__":
    main()
