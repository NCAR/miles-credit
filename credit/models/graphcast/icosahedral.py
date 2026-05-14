# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Icosahedral mesh construction originally Copyright 2023 DeepMind Technologies Limited
# (Apache 2.0). Ported from PhysicsNeMo (NVIDIA, Apache 2.0) into CREDIT.
# Distributed training stripped; PyG backend only.

"""Icosahedral mesh construction and grid↔mesh graph building for GraphCast."""

import itertools
import logging
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from scipy.spatial import transform as _scipy_transform

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    _scipy_transform = None

try:
    from sklearn.neighbors import NearestNeighbors

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    NearestNeighbors = None

try:
    import torch_geometric.data as _pyg_data
    import torch_geometric.utils as _pyg_utils

    PyGData = _pyg_data.Data
    PyGHeteroData = _pyg_data.HeteroData
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    PyGData = None
    PyGHeteroData = None


# ---------------------------------------------------------------------------
# Geometry utilities  (from PhysicsNeMo graph_utils.py)
# ---------------------------------------------------------------------------


def deg2rad(deg: Tensor) -> Tensor:
    return deg * np.pi / 180


def rad2deg(rad: Tensor) -> Tensor:
    return rad * 180 / np.pi


def latlon2xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    if unit == "deg":
        latlon = deg2rad(latlon)
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


def xyz2latlon(xyz: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == "deg":
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    return torch.stack((lat, lon), dim=1)


def geospatial_rotation(invar: Tensor, theta: Tensor, axis: str, unit: str = "rad") -> Tensor:
    if unit == "deg":
        invar = deg2rad(invar)
    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3))
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    if axis == "x":
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == "y":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == "z":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError("Invalid axis")
    outvar = torch.matmul(rotation, invar).squeeze()
    return outvar


def azimuthal_angle(lon: Tensor) -> Tensor:
    return torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)


def polar_angle(lat: Tensor) -> Tensor:
    return torch.where(lat >= 0.0, lat, 2 * np.pi + lat)


def max_edge_length(vertices, source_nodes, destination_nodes) -> float:
    vertices_np = np.array(vertices)
    source_coords = vertices_np[source_nodes]
    dest_coords = vertices_np[destination_nodes]
    squared_differences = np.sum((source_coords - dest_coords) ** 2, axis=1)
    return float(np.sqrt(np.max(squared_differences)))


def get_face_centroids(vertices, faces):
    centroids = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        centroid = (
            (v0[0] + v1[0] + v2[0]) / 3,
            (v0[1] + v1[1] + v2[1]) / 3,
            (v0[2] + v1[2] + v2[2]) / 3,
        )
        centroids.append(centroid)
    return centroids


# ---------------------------------------------------------------------------
# Icosahedral mesh  (from DeepMind via PhysicsNeMo icosahedral_mesh.py)
# ---------------------------------------------------------------------------


class TriangularMesh(NamedTuple):
    vertices: np.ndarray
    faces: np.ndarray


def merge_meshes(mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
    for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])
    return TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
    )


def get_hierarchy_of_triangular_meshes_for_sphere(splits: int) -> List[TriangularMesh]:
    current_mesh = get_icosahedron()
    output_meshes = [current_mesh]
    for _ in range(splits):
        current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
        output_meshes.append(current_mesh)
    return output_meshes


def get_icosahedron() -> TriangularMesh:
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for icosahedral mesh construction. Install with: pip install scipy")
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1 in [1.0, -1.0]:
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.0))
            vertices.append((0.0, c1, c2))
            vertices.append((c2, 0.0, c1))
    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1.0, phi])
    faces = [
        (0, 1, 2),
        (0, 6, 1),
        (8, 0, 2),
        (8, 4, 0),
        (3, 8, 2),
        (3, 2, 7),
        (7, 2, 1),
        (0, 4, 6),
        (4, 11, 6),
        (6, 11, 5),
        (1, 5, 7),
        (4, 10, 11),
        (4, 8, 10),
        (10, 8, 3),
        (10, 3, 9),
        (11, 10, 9),
        (11, 9, 5),
        (5, 9, 7),
        (9, 3, 7),
        (1, 6, 5),
    ]
    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    rotation = _scipy_transform.Rotation.from_euler(seq="y", angles=rotation_angle)
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)
    return TriangularMesh(
        vertices=vertices.astype(np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def _two_split_unit_sphere_triangle_faces(triangular_mesh: TriangularMesh) -> TriangularMesh:
    new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)
    new_faces = []
    for ind1, ind2, ind3 in triangular_mesh.faces:
        ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
        new_faces.extend(
            [
                [ind1, ind12, ind31],
                [ind12, ind2, ind23],
                [ind31, ind23, ind3],
                [ind12, ind23, ind31],
            ]
        )
    return TriangularMesh(
        vertices=new_vertices_builder.get_all_vertices(),
        faces=np.array(new_faces, dtype=np.int32),
    )


class _ChildVerticesBuilder:
    def __init__(self, parent_vertices):
        self._child_vertices_index_mapping = {}
        self._parent_vertices = parent_vertices
        self._all_vertices_list = list(parent_vertices)

    def _get_child_vertex_key(self, parent_vertex_indices):
        return tuple(sorted(parent_vertex_indices))

    def _create_child_vertex(self, parent_vertex_indices):
        child_vertex_position = self._parent_vertices[list(parent_vertex_indices)].mean(0)
        child_vertex_position /= np.linalg.norm(child_vertex_position)
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        self._child_vertices_index_mapping[child_vertex_key] = len(self._all_vertices_list)
        self._all_vertices_list.append(child_vertex_position)

    def get_new_child_vertex_index(self, parent_vertex_indices):
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        if child_vertex_key not in self._child_vertices_index_mapping:
            self._create_child_vertex(parent_vertex_indices)
        return self._child_vertices_index_mapping[child_vertex_key]

    def get_all_vertices(self):
        return np.array(self._all_vertices_list)


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert faces.ndim == 2 and faces.shape[-1] == 3
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


# ---------------------------------------------------------------------------
# PyG graph backend  (from PhysicsNeMo graph_backend.py, PyG path only)
# ---------------------------------------------------------------------------


def _add_edge_features_pyg(graph, pos, normalize: bool = True):
    """Compute rotated local-frame edge features and attach as graph.edge_attr."""
    if not _PYG_AVAILABLE:
        raise ImportError("torch_geometric is required for GraphCast.")
    if isinstance(pos, tuple):
        src_pos, dst_pos = pos
    else:
        src_pos = dst_pos = pos

    if isinstance(graph, PyGData):
        src_idx, dst_idx = graph.edge_index
    else:
        src_idx, dst_idx = graph[graph.edge_types[0]].edge_index

    src_pos_e = src_pos[src_idx.long()]
    dst_pos_e = dst_pos[dst_idx.long()]

    dst_latlon = xyz2latlon(dst_pos_e, unit="rad")
    dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]

    theta_azimuthal = azimuthal_angle(dst_lon)
    theta_polar = polar_angle(dst_lat)

    src_pos_e = geospatial_rotation(src_pos_e, theta=theta_azimuthal, axis="z", unit="rad")
    dst_pos_e = geospatial_rotation(dst_pos_e, theta=theta_azimuthal, axis="z", unit="rad")
    src_pos_e = geospatial_rotation(src_pos_e, theta=theta_polar, axis="y", unit="rad")
    dst_pos_e = geospatial_rotation(dst_pos_e, theta=theta_polar, axis="y", unit="rad")

    disp = src_pos_e - dst_pos_e
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    if normalize:
        max_disp_norm = torch.max(disp_norm)
        graph.edge_attr = torch.cat((disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1)
    else:
        graph.edge_attr = torch.cat((disp, disp_norm), dim=-1)
    return graph


def _add_node_features_pyg(graph, pos: Tensor):
    latlon = xyz2latlon(pos)
    lat, lon = latlon[:, 0], latlon[:, 1]
    graph.x = torch.stack((torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1)
    return graph


# ---------------------------------------------------------------------------
# Graph builder  (from PhysicsNeMo graph.py)
# ---------------------------------------------------------------------------


class Graph:
    """Builds the three GraphCast graphs: mesh, Grid2Mesh, Mesh2Grid."""

    def __init__(
        self,
        lat_lon_grid: Tensor,
        mesh_level: int = 6,
        multimesh: bool = True,
        dtype=torch.float,
    ) -> None:
        if not _PYG_AVAILABLE:
            raise ImportError("torch_geometric is required for GraphCast graph construction.")
        self.dtype = dtype
        self.lat_lon_grid_flat = lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

        _meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
        finest_mesh = _meshes[-1]
        self.finest_mesh_src, self.finest_mesh_dst = faces_to_edges(finest_mesh.faces)
        self.finest_mesh_vertices = np.array(finest_mesh.vertices)

        if multimesh:
            mesh = merge_meshes(_meshes)
            self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)
            self.mesh_vertices = np.array(mesh.vertices)
        else:
            mesh = finest_mesh
            self.mesh_src, self.mesh_dst = self.finest_mesh_src, self.finest_mesh_dst
            self.mesh_vertices = self.finest_mesh_vertices

        self.mesh_faces = mesh.faces

    def create_mesh_graph(self, verbose: bool = True):
        src_t = torch.from_numpy(self.mesh_src).long()
        dst_t = torch.from_numpy(self.mesh_dst).long()
        edge_index = torch.stack([src_t, dst_t], dim=0)
        edge_index = _pyg_utils.to_undirected(edge_index)

        mesh_graph = PyGData(edge_index=edge_index)
        mesh_pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)
        mesh_graph = _add_edge_features_pyg(mesh_graph, mesh_pos)
        mesh_graph = _add_node_features_pyg(mesh_graph, mesh_pos)
        mesh_graph.lat_lon = xyz2latlon(mesh_pos)
        mesh_graph.x = mesh_graph.x.to(dtype=self.dtype)
        mesh_graph.edge_attr = mesh_graph.edge_attr.to(dtype=self.dtype)

        if verbose:
            logger.info("mesh graph: %d nodes, %d edges", mesh_graph.num_nodes, mesh_graph.num_edges)
        return mesh_graph

    def create_g2m_graph(self, verbose: bool = True):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for Grid2Mesh graph construction. Install with: pip install scikit-learn"
            )
        max_edge_len = max_edge_length(self.finest_mesh_vertices, self.finest_mesh_src, self.finest_mesh_dst)
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 4
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(self.mesh_vertices)
        distances, indices = neighbors.kneighbors(cartesian_grid.numpy())

        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(n_nbrs):
                if distances[i][j] <= 0.6 * max_edge_len:
                    src.append(i)
                    dst.append(int(indices[i][j]))

        g2m_graph = PyGHeteroData()
        g2m_graph[("grid", "g2m", "mesh")].edge_index = torch.tensor([src, dst], dtype=torch.long)
        g2m_graph["grid"].pos = cartesian_grid.to(torch.float32)
        g2m_graph["mesh"].pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)
        g2m_graph["grid"].lat_lon = self.lat_lon_grid_flat
        g2m_graph["mesh"].lat_lon = xyz2latlon(g2m_graph["mesh"].pos)
        g2m_graph = _add_edge_features_pyg(g2m_graph, (g2m_graph["grid"].pos, g2m_graph["mesh"].pos))
        g2m_graph["grid"].pos = g2m_graph["grid"].pos.to(dtype=self.dtype)
        g2m_graph["mesh"].pos = g2m_graph["mesh"].pos.to(dtype=self.dtype)
        g2m_graph.edge_attr = g2m_graph.edge_attr.to(dtype=self.dtype)

        if verbose:
            n_edges = g2m_graph[("grid", "g2m", "mesh")].edge_index.shape[1]
            logger.info("g2m graph: %d edges", n_edges)
        return g2m_graph

    def create_m2g_graph(self, verbose: bool = True):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for Mesh2Grid graph construction. Install with: pip install scikit-learn"
            )
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        face_centroids = get_face_centroids(self.mesh_vertices, self.mesh_faces)
        n_nbrs = 1
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(face_centroids)
        _, indices = neighbors.kneighbors(cartesian_grid.numpy())
        indices = indices.flatten()

        src = [p for i in indices for p in self.mesh_faces[i]]
        dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]

        m2g_graph = PyGHeteroData()
        m2g_graph[("mesh", "m2g", "grid")].edge_index = torch.tensor([src, dst], dtype=torch.long)
        m2g_graph["mesh"].pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)
        m2g_graph["grid"].pos = cartesian_grid.to(torch.float32)
        m2g_graph["mesh"].lat_lon = xyz2latlon(m2g_graph["mesh"].pos)
        m2g_graph["grid"].lat_lon = self.lat_lon_grid_flat
        m2g_graph = _add_edge_features_pyg(m2g_graph, (m2g_graph["mesh"].pos, m2g_graph["grid"].pos))
        m2g_graph["mesh"].pos = m2g_graph["mesh"].pos.to(dtype=self.dtype)
        m2g_graph["grid"].pos = m2g_graph["grid"].pos.to(dtype=self.dtype)
        m2g_graph.edge_attr = m2g_graph.edge_attr.to(dtype=self.dtype)

        if verbose:
            n_edges = m2g_graph[("mesh", "m2g", "grid")].edge_index.shape[1]
            logger.info("m2g graph: %d edges", n_edges)
        return m2g_graph
