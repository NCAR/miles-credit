# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from NVIDIA PhysicsNeMo (Apache 2.0) into CREDIT.
# Stripped: nvfuser recompute_activation, distributed graph types, profiling decorator.

"""GNN primitives for GraphCast: MLP, edge/node blocks, encoder, decoder, embedders."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

try:
    import torch_geometric.data as _pyg_data

    PyGData = _pyg_data.Data
    PyGHeteroData = _pyg_data.HeteroData
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    PyGData = None
    PyGHeteroData = None

# scatter: prefer torch_scatter, fall back to manual scatter_add
try:
    from torch_scatter import scatter as _ts_scatter

    def _scatter(src, index, dim_size, reduce):
        return _ts_scatter(src, index, dim=0, dim_size=dim_size, reduce=reduce)
except ImportError:

    def _scatter(src, index, dim_size, reduce):
        out = src.new_zeros(dim_size, src.shape[1] if src.ndim > 1 else 1)
        if src.ndim == 1:
            src = src.unsqueeze(-1)
        out.scatter_add_(0, index.view(-1, 1).expand_as(src), src)
        if reduce == "mean":
            count = (
                src.new_zeros(dim_size).scatter_add_(0, index, torch.ones(len(index), device=src.device)).clamp(min=1)
            )
            out = out / count.unsqueeze(-1)
        return out


GraphType = Union["PyGData", "PyGHeteroData"]  # type alias


# ---------------------------------------------------------------------------
# Checkpoint helpers  (from PhysicsNeMo gnn_layers/utils.py)
# ---------------------------------------------------------------------------


def _checkpoint_identity(layer, *args, **kwargs):
    return layer(*args)


def set_checkpoint_fn(do_checkpointing: bool):
    return checkpoint if do_checkpointing else _checkpoint_identity


# ---------------------------------------------------------------------------
# Scatter/aggregate helpers  (from PhysicsNeMo gnn_layers/utils.py)
# ---------------------------------------------------------------------------


def _get_edge_index(graph):
    if isinstance(graph, PyGHeteroData):
        return graph[graph.edge_types[0]].edge_index.long()
    return graph.edge_index.long()


def concat_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor, Tensor]],
    graph: GraphType,
) -> Tensor:
    src_idx, dst_idx = _get_edge_index(graph)
    if isinstance(nfeat, Tensor):
        src_feat, dst_feat = nfeat, nfeat
    else:
        src_feat, dst_feat = nfeat
    return torch.cat((efeat, src_feat[src_idx], dst_feat[dst_idx]), dim=1)


@torch.compile
def _sum_edge_node_feat(efeat, src_feat, dst_feat, src_idx, dst_idx):
    return efeat + src_feat[src_idx] + dst_feat[dst_idx]


def sum_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor, Tensor]],
    graph: GraphType,
) -> Tensor:
    src_idx, dst_idx = _get_edge_index(graph)
    if isinstance(nfeat, Tensor):
        src_feat, dst_feat = nfeat, nfeat
    else:
        src_feat, dst_feat = nfeat
    return _sum_edge_node_feat(efeat, src_feat, dst_feat, src_idx, dst_idx)


def aggregate_and_concat(
    efeat: Tensor,
    nfeat: Tensor,
    graph: GraphType,
    aggregation: str,
) -> Tensor:
    _, dst_idx = _get_edge_index(graph)
    h_dest = _scatter(efeat, dst_idx, dim_size=nfeat.shape[0], reduce=aggregation)
    return torch.cat((h_dest, nfeat), dim=-1)


# ---------------------------------------------------------------------------
# MeshGraphMLP  (from PhysicsNeMo gnn_layers/mesh_graph_mlp.py)
# ---------------------------------------------------------------------------


class MeshGraphMLP(nn.Module):
    """MLP with optional LayerNorm, used throughout the GraphCast GNN layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: Optional[int] = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        if hidden_layers is not None:
            layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
            layers.append(nn.Linear(hidden_dim, output_dim))
            if norm_type is not None:
                layers.append(nn.LayerNorm(output_dim))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MeshGraphEdgeMLPConcat(MeshGraphMLP):
    """Edge MLP that concatenates (efeat, src_nfeat, dst_nfeat) before the MLP."""

    def __init__(
        self,
        efeat_dim: int = 512,
        src_dim: int = 512,
        dst_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__(
            efeat_dim + src_dim + dst_dim,
            output_dim,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor, Tensor]],
        graph: GraphType,
    ) -> Tensor:
        efeat = concat_efeat(efeat, nfeat, graph)
        return self.model(efeat)


class MeshGraphEdgeMLPSum(nn.Module):
    """Edge MLP with the 'concat trick': separate linear projections summed before MLP.

    Equivalent to MeshGraphEdgeMLPConcat but avoids a large concat for memory savings.
    Initialised from the same weight distribution as the concat version.
    """

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        # Initialise from the same distribution as a full concat linear
        tmp_lin = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim)
        w_efeat, w_src, w_dst = torch.split(tmp_lin.weight, [efeat_dim, src_dim, dst_dim], dim=1)
        self.lin_efeat = nn.Parameter(w_efeat)
        self.lin_src = nn.Parameter(w_src)
        self.lin_dst = nn.Parameter(w_dst)
        self.bias = tmp_lin.bias

        layers = [activation_fn]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))
        if norm_type is not None:
            layers.append(nn.LayerNorm(output_dim))
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        efeat: Tensor,
        nfeat: Union[Tensor, Tuple[Tensor, Tensor]],
        graph: GraphType,
    ) -> Tensor:
        mlp_sum = sum_efeat(
            torch.nn.functional.linear(efeat, self.lin_efeat, None),
            nfeat if isinstance(nfeat, Tensor) else nfeat,
            graph,
        )
        # bias added once (to dst projection)
        src_idx, dst_idx = _get_edge_index(graph)
        src_feat = nfeat if isinstance(nfeat, Tensor) else nfeat[0]
        dst_feat = nfeat if isinstance(nfeat, Tensor) else nfeat[1]
        mlp_sum = (
            torch.nn.functional.linear(efeat, self.lin_efeat, None)
            + torch.nn.functional.linear(src_feat, self.lin_src, None)[src_idx]
            + torch.nn.functional.linear(dst_feat, self.lin_dst, self.bias)[dst_idx]
        )
        return self.model(mlp_sum)


# ---------------------------------------------------------------------------
# MeshEdgeBlock  (from PhysicsNeMo gnn_layers/mesh_edge_block.py)
# ---------------------------------------------------------------------------


class MeshEdgeBlock(nn.Module):
    """Updates edge features from (efeat, src_nfeat, dst_nfeat) with residual."""

    def __init__(
        self,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()
        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(self, efeat: Tensor, nfeat: Tensor, graph: GraphType) -> Tuple[Tensor, Tensor]:
        efeat_new = self.edge_mlp(efeat, nfeat, graph) + efeat
        return efeat_new, nfeat


# ---------------------------------------------------------------------------
# MeshNodeBlock  (from PhysicsNeMo gnn_layers/mesh_node_block.py)
# ---------------------------------------------------------------------------


class MeshNodeBlock(nn.Module):
    """Updates node features by aggregating edge messages with residual."""

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.aggregation = aggregation
        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(self, efeat: Tensor, nfeat: Tensor, graph: GraphType) -> Tuple[Tensor, Tensor]:
        cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
        nfeat_new = self.node_mlp(cat_feat) + nfeat
        return efeat, nfeat_new


# ---------------------------------------------------------------------------
# MeshGraphEncoder  (from PhysicsNeMo gnn_layers/mesh_graph_encoder.py)
# ---------------------------------------------------------------------------


class MeshGraphEncoder(nn.Module):
    """Grid→Mesh bipartite GNN layer (Grid2Mesh encoder)."""

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_src_nodes: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation
        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.src_node_mlp = MeshGraphMLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.dst_node_mlp = MeshGraphMLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self,
        g2m_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: GraphType,
    ) -> Tuple[Tensor, Tensor]:
        efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
        cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
        mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
        return grid_nfeat, mesh_nfeat


# ---------------------------------------------------------------------------
# MeshGraphDecoder  (from PhysicsNeMo gnn_layers/mesh_graph_decoder.py)
# ---------------------------------------------------------------------------


class MeshGraphDecoder(nn.Module):
    """Mesh→Grid bipartite GNN layer (Mesh2Grid decoder)."""

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation
        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self,
        m2g_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: GraphType,
    ) -> Tensor:
        efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat, grid_nfeat), graph)
        cat_feat = aggregate_and_concat(efeat, grid_nfeat, graph, self.aggregation)
        return self.node_mlp(cat_feat) + grid_nfeat


# ---------------------------------------------------------------------------
# Embedders  (from PhysicsNeMo gnn_layers/embedder.py)
# ---------------------------------------------------------------------------


class GraphCastEncoderEmbedder(nn.Module):
    """Embeds grid nodes, mesh nodes, g2m edges, and mesh edges to hidden_dim."""

    def __init__(
        self,
        input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        kw = dict(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.grid_node_mlp = MeshGraphMLP(input_dim=input_dim_grid_nodes, **kw)
        self.mesh_node_mlp = MeshGraphMLP(input_dim=input_dim_mesh_nodes, **kw)
        self.mesh_edge_mlp = MeshGraphMLP(input_dim=input_dim_edges, **kw)
        self.grid2mesh_edge_mlp = MeshGraphMLP(input_dim=input_dim_edges, **kw)

    def forward(
        self,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        g2m_efeat: Tensor,
        mesh_efeat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.grid_node_mlp(grid_nfeat),
            self.mesh_node_mlp(mesh_nfeat),
            self.grid2mesh_edge_mlp(g2m_efeat),
            self.mesh_edge_mlp(mesh_efeat),
        )


class GraphCastDecoderEmbedder(nn.Module):
    """Embeds Mesh2Grid edge features to hidden_dim."""

    def __init__(
        self,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.mesh2grid_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(self, m2g_efeat: Tensor) -> Tensor:
        return self.mesh2grid_edge_mlp(m2g_efeat)
