"""
Generates edges evenly distributed across latitude bands.

This approach addresses the overconnectivity issue found in radius-based edge generation
(see `graph_edges_with_radius.py`), which tends to create excessive edge density near the poles
due to the convergence of grid points.
"""

import xarray as xr
import argparse
import numpy as np
from os.path import join, exists
from os import makedirs
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm


def gen_per_lat_coords(per_lat_lon_list):
    coord_list = []
    count = 0
    for lat, lon_list in per_lat_lon_list:
        coords = np.stack([np.repeat(lat, lon_list.shape[0]), lon_list], axis=1)
        assert coords.ndim == 2 and coords.shape[1] == 2
        node_index = count + np.arange(len(coords))
        count += len(coords)
        # coords is a 2-D lat-lon matrix of shape(lon, 2)
        # node_index gives a unique idx to each (lat, lon) combination
        coord_list.append((coords, node_index))
    return coord_list


def compute_haversine_dist(a, b):
    result = haversine_distances(np.deg2rad(np.asarray(a)), np.deg2rad(np.asarray(b)))
    return result * 6371000 / 1000  # multiply by Earth radius to get kilometers


def get_neighbors(current_coords, current_node_indx, neigh_coords, neigh_indx, radius):
    dist = compute_haversine_dist(current_coords, neigh_coords)
    assert (
        dist.shape[0] == current_coords.shape[0]
        and dist.shape[1] == neigh_coords.shape[0]
    )
    k_indx = np.argsort(dist, axis=1)[..., : radius * 2 + 1]
    k_nn_dist_arr = np.take_along_axis(dist, k_indx, axis=1)
    neighbors = np.repeat(neigh_indx[None], current_coords.shape[0], axis=0)
    neighbors = np.take_along_axis(neighbors, k_indx, axis=1)

    current_nodes = np.repeat(current_node_indx[:, None], k_indx.shape[1], axis=1)
    linked_coords = np.stack([neighbors, current_nodes], axis=-1)
    assert linked_coords.ndim == 3 and linked_coords.shape[2] == 2

    return linked_coords, k_nn_dist_arr
    # print(lat_i)


def construct_graph(per_lat_coords, radius):
    all_indices = np.concatenate([item_i[1] for item_i in per_lat_coords])
    all_coords = np.concatenate([item_i[0] for item_i in per_lat_coords])

    new_edges = []
    new_dist = []

    for lat_i in tqdm(range(len(per_lat_coords))):
        current_coords, current_node_indx = per_lat_coords[lat_i]

        # Connect nodes on the same latitude
        for i in range(len(current_coords)):
            # if i = 0, than node at i = 0 will have an edge that wraps to the rightmost node at i = -1
            coord_j = current_coords[i - 1]
            node_idx_j = current_node_indx[i - 1]
            coord_i = current_coords[i]
            node_idx_i = current_node_indx[i]
            dist = compute_haversine_dist([coord_j], [coord_i])
            # Every point is linked to its left neighbor on the same parallel to avoid duplicate edges
            new_edges.append(np.array([[node_idx_j, node_idx_i]]))
            new_dist.append(dist.reshape(-1))

        # nodes with positive lat are linked to above lats
        if lat_i > 0:
            for i in range(1, radius + 1):
                if lat_i - i >= 0:
                    up_coords, up_node_indx = per_lat_coords[lat_i - i]
                    linked_coords, k_nn_dist_arr = get_neighbors(
                        current_coords,
                        current_node_indx,
                        up_coords,
                        up_node_indx,
                        radius,
                    )
                    new_edges.append(linked_coords.reshape(-1, 2))
                    # print(len(new_edges), new_edges[-1].shape)
                    new_dist.append(k_nn_dist_arr.reshape(-1))

        # nodes with negative lat are linked to below lats
        if lat_i < len(per_lat_coords) - 1:
            for i in range(1, radius + 1):
                if lat_i + i < len(per_lat_coords):
                    down_coords, down_node_indx = per_lat_coords[lat_i + i]
                    linked_coords, k_nn_dist_arr = get_neighbors(
                        current_coords,
                        current_node_indx,
                        down_coords,
                        down_node_indx,
                        radius,
                    )
                    new_edges.append(linked_coords.reshape(-1, 2))
                    # print(len(new_edges), new_edges[-1].shape)
                    new_dist.append(k_nn_dist_arr.reshape(-1))

    new_edges = np.concatenate(new_edges)
    new_dist = np.concatenate(new_dist)
    new_edges_sym = np.concatenate([new_edges, new_edges[:, [1, 0]]])
    new_dist_sym = np.concatenate([new_dist, new_dist])

    _, uniq_indx = np.unique(new_edges_sym, axis=0, return_index=True)
    new_edges_sym = new_edges_sym[uniq_indx]
    new_dist_sym = new_dist_sym[uniq_indx]

    self_indices = np.stack([all_indices] * 2, axis=1)
    new_edges_sym = np.concatenate([new_edges_sym, self_indices])
    new_dist_sym = np.concatenate([new_dist_sym, np.zeros(len(all_indices))])

    output_ds = xr.Dataset(
        {
            "edges": (("edge_idx", "pair"), new_edges_sym),
            "distances": (("edge_idx",), new_dist_sym),
            "longitude": (("node_idx",), all_coords[:, 1]),
            "latitude": (("node_idx",), all_coords[:, 0]),
        },
        coords={"node_idx": all_indices},
    )
    return output_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--coord", help="Path to xarray file containing coordinates"
    )
    parser.add_argument("-o", "--out", help="Path to output directory")
    parser.add_argument("-r", "--radius", type=int, help="Number of hops")
    args = parser.parse_args()
    coords = xr.open_dataset(args.coord)
    lon = coords["longitude"].values
    lon[lon > 180] = lon[lon > 180] - 360.0
    lat = coords["latitude"].values

    # [(l, lon) for l in lat]: e.g., [(90, [-180,..., 180]), ..., (-90, [-180, ..., 180])]
    per_lat_coords = gen_per_lat_coords([(l, lon) for l in lat])

    output_ds = construct_graph(per_lat_coords, args.radius)
    output_ds = output_ds.assign_attrs(coord_file=args.coord, radius=args.radius)
    if not exists(args.out):
        makedirs(args.out)
    filename = join(
        args.out, f"grid_edge_pairs_r_{args.radius}_res_{len(lat)}_{len(lon)}.nc"
    )
    output_ds.to_netcdf(filename)
    print("Saved to " + filename)


if __name__ == "__main__":
    main()
