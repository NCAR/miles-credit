"""
Build AIFS mesh graph for CREDIT.

Constructs the torch_geometric HeteroData object used by CREDITAifs:
  - 'data' nodes    : the lat/lon ERA5 grid  (H × W)
  - 'hidden' nodes  : a coarser mesh (every mesh_stride degrees)
  - Grid2Mesh edges : k nearest-neighbour data → hidden
  - Mesh2Grid edges : k nearest-neighbour hidden → data

Edge attributes: [sin(Δlat), cos(Δlat), sin(Δlon), cos(Δlon), arc/max_arc]
  where Δlat/Δlon are coord differences from src to dst.

Node attributes (x): [sin(lat), cos(lat), sin(lon), cos(lon)]

Usage
-----
python -m credit.models.aifs.build_graph \\
    --lat_size 181 --lon_size 360 \\
    --mesh_stride 3 \\
    --k_g2m 4 --k_m2g 4 \\
    --save_path /path/to/aifs_graph_1deg.pt

The saved .pt file is loaded by CREDITAifs at model init via graph_path=...
"""

import argparse

import numpy as np
import torch
from scipy.spatial import KDTree
from torch_geometric.data import HeteroData


def _node_coords(lats_deg: np.ndarray, lons_deg: np.ndarray):
    """Return (N,2) lat/lon array in radians from 1D lat and lon arrays."""
    lat_g, lon_g = np.meshgrid(lats_deg, lons_deg, indexing="ij")
    return np.stack([lat_g.ravel(), lon_g.ravel()], axis=1)  # (N, 2) radians


def _node_attr(coords_rad: np.ndarray) -> torch.Tensor:
    """Node features: [sin(lat), cos(lat), sin(lon), cos(lon)]."""
    lat = coords_rad[:, 0]
    lon = coords_rad[:, 1]
    return torch.tensor(
        np.stack([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)], axis=1),
        dtype=torch.float32,
    )


def _edge_attr(
    src_coords: np.ndarray, dst_coords: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray
) -> torch.Tensor:
    """
    Edge features: [sin(Δlat), cos(Δlat), sin(Δlon), cos(Δlon), arc/max_arc].

    Δlat = lat_dst - lat_src,  Δlon = lon_dst - lon_src.
    arc = great-circle distance (haversine).
    """
    lat_s = src_coords[src_idx, 0]
    lon_s = src_coords[src_idx, 1]
    lat_d = dst_coords[dst_idx, 0]
    lon_d = dst_coords[dst_idx, 1]

    dlat = lat_d - lat_s
    dlon = lon_d - lon_s

    # haversine
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_s) * np.cos(lat_d) * np.sin(dlon / 2) ** 2
    arc = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    max_arc = arc.max() + 1e-8

    feats = np.stack(
        [
            np.sin(dlat),
            np.cos(dlat),
            np.sin(dlon),
            np.cos(dlon),
            arc / max_arc,
        ],
        axis=1,
    )
    return torch.tensor(feats, dtype=torch.float32)


def _knn_edges(src_coords: np.ndarray, dst_coords: np.ndarray, k: int, src_size: int, dst_size: int):
    """
    For each dst node find k nearest src nodes.

    Returns edge_index (2, E) [src, dst] and (src_idx, dst_idx) arrays.
    """

    # use 3D Cartesian for correct spherical distance
    def to_xyz(coords):
        lat, lon = coords[:, 0], coords[:, 1]
        return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)

    tree = KDTree(to_xyz(src_coords))
    dists, indices = tree.query(to_xyz(dst_coords), k=k)

    dst_idx = np.repeat(np.arange(dst_size), k)
    src_idx = indices.ravel()

    edge_index = torch.tensor(np.stack([src_idx, dst_idx], axis=0), dtype=torch.long)
    return edge_index, src_idx, dst_idx


def build_graph(
    lat_size: int = 181,
    lon_size: int = 360,
    mesh_stride: int = 3,
    k_g2m: int = 4,
    k_m2g: int = 4,
    radius_m2m: float = None,
    save_path: str = None,
) -> HeteroData:
    """
    Build and optionally save the AIFS mesh graph.

    Parameters
    ----------
    lat_size, lon_size : int
        Data grid shape (e.g. 181, 360 for 1° ERA5).
    mesh_stride : int
        Coarsening stride for the hidden mesh (degrees).  mesh_stride=3 gives
        a 61×120 = 7,320-node hidden mesh for 1° ERA5.
    k_g2m : int
        Number of nearest data nodes per hidden node (Grid2Mesh).
    k_m2g : int
        Number of nearest hidden nodes per data node (Mesh2Grid).
    radius_m2m : float or None
        Not used (Transformer processor handles Mesh2Mesh; no GNN edges needed).
    save_path : str or None
        If given, save the HeteroData as a .pt file.

    Returns
    -------
    HeteroData
    """
    # --- data nodes ---
    lats_data = np.linspace(90, -90, lat_size)  # degrees, north→south
    lons_data = np.linspace(0, 360 - 360 / lon_size, lon_size)
    data_coords_deg = _node_coords(lats_data, lons_data)
    data_coords_rad = np.deg2rad(data_coords_deg)
    N_data = lat_size * lon_size

    # --- hidden mesh nodes (coarsened lat/lon subgrid) ---
    mesh_lat = np.arange(-90, 90 + mesh_stride, mesh_stride, dtype=float)
    mesh_lon = np.arange(0, 360, mesh_stride, dtype=float)
    mesh_coords_deg = _node_coords(mesh_lat, mesh_lon)
    mesh_coords_rad = np.deg2rad(mesh_coords_deg)
    N_hidden = len(mesh_coords_rad)

    # --- Grid2Mesh  (data src → hidden dst) ---
    g2m_edge_index, g2m_src, g2m_dst = _knn_edges(
        data_coords_rad,
        mesh_coords_rad,
        k=k_g2m,
        src_size=N_data,
        dst_size=N_hidden,
    )
    g2m_attr = _edge_attr(data_coords_rad, mesh_coords_rad, g2m_src, g2m_dst)

    # --- Mesh2Grid  (hidden src → data dst) ---
    m2g_edge_index, m2g_src, m2g_dst = _knn_edges(
        mesh_coords_rad,
        data_coords_rad,
        k=k_m2g,
        src_size=N_hidden,
        dst_size=N_data,
    )
    m2g_attr = _edge_attr(mesh_coords_rad, data_coords_rad, m2g_src, m2g_dst)

    # --- HeteroData ---
    graph = HeteroData()
    graph["data"].num_nodes = N_data
    graph["data"].x = _node_attr(data_coords_rad)

    graph["hidden"].num_nodes = N_hidden
    graph["hidden"].x = _node_attr(mesh_coords_rad)

    graph["data", "to", "hidden"].edge_index = g2m_edge_index
    graph["data", "to", "hidden"].edge_attr = g2m_attr

    graph["hidden", "to", "data"].edge_index = m2g_edge_index
    graph["hidden", "to", "data"].edge_attr = m2g_attr

    if save_path:
        torch.save(graph, save_path)
        print(f"Saved AIFS graph → {save_path}")
        print(f"  data nodes   : {N_data:,}  ({lat_size}×{lon_size})")
        print(f"  hidden nodes : {N_hidden:,}  ({mesh_stride}° stride)")
        print(f"  Grid2Mesh    : {g2m_edge_index.shape[1]:,} edges  (k={k_g2m})")
        print(f"  Mesh2Grid    : {m2g_edge_index.shape[1]:,} edges  (k={k_m2g})")

    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AIFS mesh graph for CREDIT")
    parser.add_argument("--lat_size", type=int, default=181)
    parser.add_argument("--lon_size", type=int, default=360)
    parser.add_argument(
        "--mesh_stride", type=int, default=3, help="Hidden mesh stride in degrees (default 3 → ~7k nodes)"
    )
    parser.add_argument("--k_g2m", type=int, default=4, help="kNN neighbors: data → hidden (Grid2Mesh)")
    parser.add_argument("--k_m2g", type=int, default=4, help="kNN neighbors: hidden → data (Mesh2Grid)")
    parser.add_argument("--save_path", type=str, required=True, help="Output .pt file path")
    args = parser.parse_args()
    build_graph(**vars(args))
