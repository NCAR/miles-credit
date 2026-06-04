# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from NVIDIA PhysicsNeMo (Apache 2.0) into CREDIT.
# Stripped: distributed training (partition_size>1), GraphTransformer processor
# (transformer_engine), nvfuser recompute_activation, profiling decorator.
# Added: CREDITGraphCast wrapper for CREDIT's (B, C, T, H, W) interface.

"""GraphCast — icosahedral GNN encoder-processor-decoder for global weather forecasting.

Architecture (Lam et al. 2023 / PhysicsNeMo):
  Grid2Mesh bipartite encoder → icosahedral mesh processor (N x MeshEdgeBlock + MeshNodeBlock)
  -> Mesh2Grid bipartite decoder -> output MLP.
"""

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .gnn_layers import (
    GraphCastDecoderEmbedder,
    GraphCastEncoderEmbedder,
    MeshEdgeBlock,
    MeshGraphDecoder,
    MeshGraphEncoder,
    MeshGraphMLP,
    MeshNodeBlock,
    set_checkpoint_fn,
)
from .icosahedral import Graph

__all__ = ["GraphCastNet", "CREDITGraphCast"]


def _get_activation(name: str) -> nn.Module:
    _map = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
    }
    key = name.lower().replace("_", "")
    if key not in _map:
        raise ValueError(f"Unknown activation '{name}'. Options: {list(_map)}")
    return _map[key]()


# ---------------------------------------------------------------------------
# Processor  (from PhysicsNeMo graph_cast_processor.py)
# ---------------------------------------------------------------------------


class GraphCastProcessor(nn.Module):
    """N x (MeshEdgeBlock + MeshNodeBlock) on the icosahedral mesh."""

    def __init__(
        self,
        aggregation: str = "sum",
        processor_layers: int = 16,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()
        edge_kw = dict(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=input_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
        )
        node_kw = dict(
            aggregation=aggregation,
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=input_dim_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        layers = []
        for _ in range(processor_layers):
            layers.append(MeshEdgeBlock(**edge_kw))
            layers.append(MeshNodeBlock(**node_kw))
        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(layers)
        self.checkpoint_segments = [(0, self.num_processor_layers)]
        self.checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_segments(self, checkpoint_segments: int):
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError("processor_layers must be divisible by checkpoint_segments")
            seg_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = [(i, i + seg_size) for i in range(0, self.num_processor_layers, seg_size)]
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    def _run_segment(self, start: int, end: int):
        segment = self.processor_layers[start:end]

        def fn(efeat, nfeat, graph):
            for module in segment:
                efeat, nfeat = module(efeat, nfeat, graph)
            return efeat, nfeat

        return fn

    def forward(self, efeat: Tensor, nfeat: Tensor, graph) -> Tuple[Tensor, Tensor]:
        for start, end in self.checkpoint_segments:
            efeat, nfeat = self.checkpoint_fn(
                self._run_segment(start, end),
                efeat,
                nfeat,
                graph,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return efeat, nfeat


# ---------------------------------------------------------------------------
# GraphCastNet  (from PhysicsNeMo graph_cast_net.py)
# ---------------------------------------------------------------------------


class GraphCastNet(nn.Module):
    """GraphCast: Grid2Mesh GNN encoder -> icosahedral processor -> Mesh2Grid GNN decoder.

    Faithful port of PhysicsNeMo's GraphCastNet (Apache 2.0).
    Single-GPU only (distributed partition_size > 1 removed).

    Parameters
    ----------
    mesh_level : int
        Icosahedral refinement level (0=12 verts, 1=42, ..., 6=40962). Default 6.
    multimesh : bool
        Include edges from all mesh levels 0...mesh_level. Default True.
    input_res : Tuple[int, int]
        Grid (H, W). Default (721, 1440).
    input_dim_grid_nodes : int
        Input feature dim per grid node. Default 474.
    input_dim_mesh_nodes : int
        Mesh node position feature dim (always 3 xyz). Default 3.
    input_dim_edges : int
        Edge feature dim (always 4: dx,dy,dz,norm). Default 4.
    output_dim_grid_nodes : int
        Output feature dim per grid node. Default 227.
    processor_layers : int
        Number of processor layers (must be >= 3). Default 16.
    hidden_layers : int
        MLP hidden depth. Default 1.
    hidden_dim : int
        Hidden embedding size. Default 512.
    aggregation : str
        Message aggregation ("sum" or "mean"). Default "sum".
    activation_fn : str
        Activation name ("silu", "relu", "gelu"). Default "silu".
    norm_type : str
        "LayerNorm". Default "LayerNorm".
    do_concat_trick : bool
        Use the sum-instead-of-concat optimisation. Default False.
    """

    def __init__(
        self,
        mesh_level: int = 6,
        multimesh: bool = True,
        input_res: tuple = (721, 1440),
        input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 227,
        processor_layers: int = 16,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        activation_fn: str = "silu",
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()

        self.input_dim_grid_nodes = input_dim_grid_nodes
        self.output_dim_grid_nodes = output_dim_grid_nodes
        self.input_res = input_res

        act = _get_activation(activation_fn)

        # Build lat/lon grid and graphs
        latitudes = torch.linspace(-90, 90, steps=input_res[0])
        longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]
        lat_lon_grid = torch.stack(torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1)

        graph_builder = Graph(lat_lon_grid, mesh_level=mesh_level, multimesh=multimesh)
        self.mesh_graph = graph_builder.create_mesh_graph(verbose=False)
        self.g2m_graph = graph_builder.create_g2m_graph(verbose=False)
        self.m2g_graph = graph_builder.create_m2g_graph(verbose=False)

        self.g2m_edata = self.g2m_graph.edge_attr
        self.m2g_edata = self.m2g_graph.edge_attr
        self.mesh_ndata = self.mesh_graph.x
        self.mesh_edata = self.mesh_graph.edge_attr

        self.model_checkpoint_fn = set_checkpoint_fn(False)
        self.encoder_checkpoint_fn = set_checkpoint_fn(False)
        self.decoder_checkpoint_fn = set_checkpoint_fn(False)

        kw = dict(
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=act,
            norm_type=norm_type,
        )
        self.encoder_embedder = GraphCastEncoderEmbedder(
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            **kw,
        )
        self.decoder_embedder = GraphCastDecoderEmbedder(input_dim_edges=input_dim_edges, **kw)

        common_enc = dict(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_src_nodes=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=act,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
        )
        self.encoder = MeshGraphEncoder(**common_enc)

        if processor_layers <= 2:
            raise ValueError("processor_layers must be >= 3")

        proc_kw = dict(
            aggregation=aggregation,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=act,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
        )
        self.processor_encoder = GraphCastProcessor(processor_layers=1, **proc_kw)
        self.processor = GraphCastProcessor(processor_layers=processor_layers - 2, **proc_kw)
        self.processor_decoder = GraphCastProcessor(processor_layers=1, **proc_kw)

        self.decoder = MeshGraphDecoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=act,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
        )
        self.finale = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=output_dim_grid_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=act,
            norm_type=None,
        )

    def set_checkpoint_model(self, flag: bool):
        self.model_checkpoint_fn = set_checkpoint_fn(flag)
        if flag:
            self.processor.set_checkpoint_segments(-1)
            self.encoder_checkpoint_fn = set_checkpoint_fn(False)
            self.decoder_checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_processor(self, segments: int):
        self.processor.set_checkpoint_segments(segments)

    def set_checkpoint_encoder(self, flag: bool):
        self.encoder_checkpoint_fn = set_checkpoint_fn(flag)

    def set_checkpoint_decoder(self, flag: bool):
        self.decoder_checkpoint_fn = set_checkpoint_fn(flag)

    def encoder_forward(self, grid_nfeat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        (
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            g2m_efeat_embedded,
            mesh_efeat_embedded,
        ) = self.encoder_embedder(grid_nfeat, self.mesh_ndata, self.g2m_edata, self.mesh_edata)
        grid_nfeat_encoded, mesh_nfeat_encoded = self.encoder(
            g2m_efeat_embedded, grid_nfeat_embedded, mesh_nfeat_embedded, self.g2m_graph
        )
        mesh_efeat_processed, mesh_nfeat_processed = self.processor_encoder(
            mesh_efeat_embedded, mesh_nfeat_encoded, self.mesh_graph
        )
        return mesh_efeat_processed, mesh_nfeat_processed, grid_nfeat_encoded

    def decoder_forward(
        self,
        mesh_efeat_processed: Tensor,
        mesh_nfeat_processed: Tensor,
        grid_nfeat_encoded: Tensor,
    ) -> Tensor:
        _, mesh_nfeat_processed = self.processor_decoder(mesh_efeat_processed, mesh_nfeat_processed, self.mesh_graph)
        m2g_efeat_embedded = self.decoder_embedder(self.m2g_edata)
        grid_nfeat_decoded = self.decoder(m2g_efeat_embedded, grid_nfeat_encoded, mesh_nfeat_processed, self.m2g_graph)
        return self.finale(grid_nfeat_decoded)

    def custom_forward(self, grid_nfeat: Tensor) -> Tensor:
        mesh_efeat_proc, mesh_nfeat_proc, grid_nfeat_enc = self.encoder_checkpoint_fn(
            self.encoder_forward,
            grid_nfeat,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        mesh_efeat_proc, mesh_nfeat_proc = self.processor(mesh_efeat_proc, mesh_nfeat_proc, self.mesh_graph)
        return self.decoder_checkpoint_fn(
            self.decoder_forward,
            mesh_efeat_proc,
            mesh_nfeat_proc,
            grid_nfeat_enc,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def forward(self, grid_nfeat: Tensor) -> Tensor:
        if grid_nfeat.size(0) != 1:
            raise ValueError(f"GraphCastNet does not support batch size > 1. Got shape {tuple(grid_nfeat.shape)}")
        # (1, C, H, W) -> (H*W, C)
        invar = grid_nfeat[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)
        outvar = self.model_checkpoint_fn(
            self.custom_forward,
            invar,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        # (H*W, C_out) -> (1, C_out, H, W)
        outvar = outvar.permute(1, 0).view(self.output_dim_grid_nodes, *self.input_res)
        return outvar.unsqueeze(0)

    def to(self, *args: Any, **kwargs: Any) -> "GraphCastNet":
        self = super().to(*args, **kwargs)
        self.g2m_edata = self.g2m_edata.to(*args, **kwargs)
        self.m2g_edata = self.m2g_edata.to(*args, **kwargs)
        self.mesh_ndata = self.mesh_ndata.to(*args, **kwargs)
        self.mesh_edata = self.mesh_edata.to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.g2m_graph = self.g2m_graph.to(device)
            self.mesh_graph = self.mesh_graph.to(device)
            self.m2g_graph = self.m2g_graph.to(device)
        return self


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITGraphCast(nn.Module):
    """CREDIT wrapper for GraphCastNet.

    Accepts CREDIT's (B, C, T, H, W) or (B, C, H, W) tensors and returns
    (B, C_out, 1, H, W). The T dimension must be 1 for GraphCast (single step).

    Config keys (model section)
    ---------------------------
    in_channels         : int   -- C_in (grid node input dim)
    out_channels        : int   -- C_out (grid node output dim)
    img_size            : [H, W]
    mesh_level          : int   (default 6)
    multimesh           : bool  (default True)
    processor_layers    : int   (default 16)
    hidden_layers       : int   (default 1)
    hidden_dim          : int   (default 512)
    aggregation         : str   (default "sum")
    activation_fn       : str   (default "silu")
    do_concat_trick     : bool  (default False)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: list,
        mesh_level: int = 6,
        multimesh: bool = True,
        processor_layers: int = 16,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        activation_fn: str = "silu",
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        **kwargs,
    ):
        super().__init__()
        H, W = img_size
        self.model = GraphCastNet(
            mesh_level=mesh_level,
            multimesh=multimesh,
            input_res=(H, W),
            input_dim_grid_nodes=in_channels,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=out_channels,
            processor_layers=processor_layers,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            x = x[:, :, 0]  # (B, C, H, W)
        out = self.model(x)  # (B, C_out, H, W)
        return out.unsqueeze(2)  # (B, C_out, 1, H, W)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @classmethod
    def load_model(cls, conf):
        return cls(**conf["model"])

    @classmethod
    def load_model_name(cls):
        return "CREDITGraphCast"


# backward-compat alias
GraphCastModel = GraphCastNet
