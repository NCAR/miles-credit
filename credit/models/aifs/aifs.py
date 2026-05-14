"""
AIFS — Artificial Intelligence Forecasting System.

Ported from anemoi-models (Apache-2.0):
  https://github.com/ecmwf/anemoi-models

Paper: Lang et al., 2024.  arXiv:2406.01465

Architecture (from anemoi-models):
    Grid2Mesh GNN encoder  (GraphTransformerForwardMapper)
    → N TransformerProcessor layers  (MHSA + FF on hidden mesh nodes)
    → Mesh2Grid GNN decoder  (GraphTransformerBackwardMapper)

Requires torch-geometric.  The mesh graph must be pre-built with
credit/models/aifs/build_graph.py and its path supplied in the config.

Key differences from the anemoi-models original:
- No distributed training  (no shard_tensor / shard_heads / ProcessGroup)
- No Hydra / OmegaConf config
- No anemoi-graphs dependency  (graph built by the included build_graph.py)
- No activation checkpointing wrapper
- flash_attn optional  (falls back to F.scaled_dot_product_attention)
"""

import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

try:
    from flash_attn import flash_attn_func as _flash_attn_func

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Primitives (ported from anemoi.models.layers.utils / mlp / conv / block)
# ---------------------------------------------------------------------------


class AutocastLayerNorm(nn.LayerNorm):
    """LayerNorm that casts output back to the input dtype (AMP-safe)."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).type_as(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron.

    Ported from anemoi.models.layers.mlp.MLP.
    Default: in → hidden → out + AutocastLayerNorm (matches anemoi default).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        n_extra_layers: int = 0,
        activation: str = "SiLU",
        final_activation: bool = False,
        layer_norm: bool = True,
    ):
        super().__init__()
        act = getattr(nn, activation)
        layers = [nn.Linear(in_features, hidden_dim), act()]
        for _ in range(n_extra_layers + 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers.append(nn.Linear(hidden_dim, out_features))
        if final_activation:
            layers.append(act())
        if layer_norm:
            layers.append(AutocastLayerNorm(out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TrainableTensor(nn.Module):
    """
    Concatenates fixed edge/node attributes with a learned offset.

    Ported from anemoi.models.layers.graph.TrainableTensor.
    """

    def __init__(self, tensor_size: int, trainable_size: int):
        super().__init__()
        if trainable_size > 0:
            self.trainable = nn.Parameter(torch.zeros(tensor_size, trainable_size))
        else:
            self.trainable = None

    def forward(self, x: Tensor, batch_size: int) -> Tensor:
        parts = [einops.repeat(x, "e f -> (b e) f", b=batch_size)]
        if self.trainable is not None:
            parts.append(einops.repeat(self.trainable.to(x.device), "e f -> (b e) f", b=batch_size))
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Graph Transformer convolution  (anemoi.models.layers.conv.GraphTransformerConv)
# ---------------------------------------------------------------------------


class GraphTransformerConv(MessagePassing):
    """
    Attention-based message passing for bipartite GNN encode/decode.

    Message:  m = (V_j + edge_attr) * softmax(Q_i · (K_j + edge_attr) / sqrt(d))
    Aggregation: sum over neighbours.

    Ported from anemoi.models.layers.conv.GraphTransformerConv.
    """

    def __init__(self, out_channels: int, dropout: float = 0.0):
        super().__init__(aggr="add", node_dim=0)
        self.out_channels = out_channels
        self.dropout = dropout

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, edge_attr: Tensor, edge_index: Tensor, size=None
    ) -> Tensor:
        return self.propagate(
            edge_index,
            size=size,
            dim_size=query.shape[0],
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
        )

    def message(
        self, query_i: Tensor, key_j: Tensor, value_j: Tensor, edge_attr: Tensor, index: Tensor, ptr, size_i
    ) -> Tensor:
        # edge_attr shape: (E, heads, d)
        alpha = (query_i * (key_j + edge_attr)).sum(-1) / self.out_channels**0.5
        alpha = softmax(alpha, index, ptr, size_i)
        if self.dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout)
        return (value_j + edge_attr) * alpha.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Transformer self-attention  (anemoi.models.layers.attention.MultiHeadSelfAttention)
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """
    MHSA for the Transformer processor.

    Operates on (batch*grid, C) packed tensors; batch_size needed to reshape.
    Uses flash_attn if available, else F.scaled_dot_product_attention.

    Ported from anemoi.models.layers.attention.MultiHeadSelfAttention.
    """

    def __init__(self, num_heads: int, embed_dim: int, window_size: int = None, dropout_p: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = (window_size, window_size) if window_size else None
        self.dropout_p = dropout_p

        self.lin_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, batch_size: int) -> Tensor:
        # x: (batch*grid, C)
        query, key, value = self.lin_qkv(x).chunk(3, -1)

        # reshape to (batch, heads, grid, head_dim)
        query, key, value = (
            einops.rearrange(t, "(b g) (h d) -> b h g d", b=batch_size, h=self.num_heads) for t in (query, key, value)
        )

        dp = self.dropout_p if self.training else 0.0

        if _FLASH_AVAILABLE:
            query, key, value = (einops.rearrange(t, "b h g d -> b g h d") for t in (query, key, value))
            out = _flash_attn_func(
                query, key, value, causal=False, window_size=self.window_size or (-1, -1), dropout_p=dp
            )
            out = einops.rearrange(out, "b g h d -> b h g d")
        else:
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=dp)

        out = einops.rearrange(out, "b h g d -> (b g) (h d)")
        return self.projection(out)


# ---------------------------------------------------------------------------
# Processor block  (anemoi.models.layers.block.TransformerProcessorBlock)
# ---------------------------------------------------------------------------


class TransformerProcessorBlock(nn.Module):
    """
    Single Transformer layer: pre-norm MHSA + pre-norm MLP (SiLU, no final LN).

    Ported from anemoi.models.layers.block.TransformerProcessorBlock.
    """

    def __init__(
        self,
        num_channels: int,
        hidden_dim: int,
        num_heads: int,
        window_size: int = None,
        activation: str = "GELU",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(num_channels)
        self.attention = MultiHeadSelfAttention(num_heads, num_channels, window_size=window_size, dropout_p=dropout_p)
        act = getattr(nn, activation)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            act(),
            nn.Linear(hidden_dim, num_channels),
        )
        self.layer_norm2 = nn.LayerNorm(num_channels)

    def forward(self, x: Tensor, batch_size: int) -> Tensor:
        x = x + self.attention(self.layer_norm1(x), batch_size)
        x = x + self.mlp(self.layer_norm2(x))
        return x


# ---------------------------------------------------------------------------
# Mapper block  (anemoi.models.layers.block.GraphTransformerMapperBlock)
# ---------------------------------------------------------------------------


class GraphTransformerMapperBlock(nn.Module):
    """
    Bipartite Graph Transformer block for encode/decode mappers.

    src → dst message passing with attention-gated edge features.

    Ported from anemoi.models.layers.block.GraphTransformerMapperBlock.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        edge_dim: int,
        num_heads: int = 16,
        activation: str = "GELU",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.out_channels = out_channels

        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.layer_norm2 = nn.LayerNorm(in_channels)

        self.lin_query = nn.Linear(in_channels, out_channels)
        self.lin_key = nn.Linear(in_channels, out_channels)
        self.lin_value = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)

        self.conv = GraphTransformerConv(out_channels=self.head_dim)
        self.projection = nn.Linear(out_channels, out_channels)

        act = getattr(nn, activation)
        self.node_dst_mlp = nn.Sequential(
            AutocastLayerNorm(out_channels),
            nn.Linear(out_channels, hidden_dim),
            act(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(
        self, x_src: Tensor, x_dst: Tensor, edge_attr: Tensor, edge_index: Tensor, size=None
    ) -> tuple[Tensor, Tensor]:
        x_skip_src, x_skip_dst = x_src, x_dst

        x_src_n = self.layer_norm1(x_src)
        x_dst_n = self.layer_norm2(x_dst)

        x_r = self.lin_self(x_dst_n)

        # (N, heads, head_dim)
        query = einops.rearrange(self.lin_query(x_dst_n), "n (h d) -> n h d", h=self.num_heads)
        key = einops.rearrange(self.lin_key(x_src_n), "n (h d) -> n h d", h=self.num_heads)
        value = einops.rearrange(self.lin_value(x_src_n), "n (h d) -> n h d", h=self.num_heads)
        edges = einops.rearrange(self.lin_edge(edge_attr), "e (h d) -> e h d", h=self.num_heads)

        out = self.conv(query=query, key=key, value=value, edge_attr=edges, edge_index=edge_index, size=size)
        out = einops.rearrange(out, "n h d -> n (h d)")

        out = self.projection(out + x_r) + x_skip_dst
        out = self.node_dst_mlp(out) + out

        # src nodes pass through unchanged (update_src_nodes=False for backward mapper)
        return x_skip_src, out


# ---------------------------------------------------------------------------
# Graph edge registry mixin
# ---------------------------------------------------------------------------


class _EdgeMixin:
    """Registers edge_attr, edge_index, and batch-expansion increment."""

    def _register_edges(
        self, sub_graph: HeteroData, edge_attributes: list, src_size: int, dst_size: int, trainable_size: int
    ):
        edge_attr = torch.cat([sub_graph[a] for a in edge_attributes], dim=1)
        self.edge_dim = edge_attr.shape[1] + trainable_size
        self.register_buffer("edge_attr_base", edge_attr, persistent=False)
        self.register_buffer("edge_index_base", sub_graph.edge_index, persistent=False)
        import numpy as np

        inc = torch.from_numpy(np.asarray([[src_size], [dst_size]], dtype=np.int64))
        self.register_buffer("edge_inc", inc, persistent=True)

    def _expand_edges(self, edge_index: Tensor, edge_inc: Tensor, batch_size: int) -> Tensor:
        return torch.cat([edge_index + i * edge_inc for i in range(batch_size)], dim=1)


# ---------------------------------------------------------------------------
# Grid2Mesh encoder  (GraphTransformerForwardMapper)
# ---------------------------------------------------------------------------


class Grid2MeshMapper(nn.Module, _EdgeMixin):
    """
    GNN encoder: data-grid nodes → hidden-mesh nodes.

    Ported from anemoi.models.layers.mapper.GraphTransformerForwardMapper
    (single GPU, no distributed sharding).
    """

    def __init__(
        self,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        trainable_size: int = 8,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
        sub_graph: HeteroData = None,
        sub_graph_edge_attributes: list = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
    ):
        super().__init__()
        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)
        self.trainable = TrainableTensor(self.edge_attr_base.shape[0], trainable_size)

        self.emb_nodes_src = nn.Linear(in_channels_src, hidden_dim)
        self.emb_nodes_dst = nn.Linear(in_channels_dst, hidden_dim)

        self.proc = GraphTransformerMapperBlock(
            in_channels=hidden_dim,
            hidden_dim=mlp_hidden_ratio * hidden_dim,
            out_channels=hidden_dim,
            edge_dim=self.edge_dim,
            num_heads=num_heads,
            activation=activation,
        )

    def forward(self, x_src: Tensor, x_dst: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
        x_src_raw = x_src  # keep raw features for decoder's skip input
        x_src_emb = self.emb_nodes_src(x_src)
        x_dst_emb = self.emb_nodes_dst(x_dst)

        edge_attr = self.trainable(self.edge_attr_base, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)

        n_src = x_src_emb.shape[0]
        n_dst = x_dst_emb.shape[0]
        _, x_hidden = self.proc(x_src_emb, x_dst_emb, edge_attr, edge_index, size=(n_src, n_dst))
        # return raw src so decoder can re-embed it (matches anemoi ForwardMapper)
        return x_src_raw, x_hidden


# ---------------------------------------------------------------------------
# Mesh2Grid decoder  (GraphTransformerBackwardMapper)
# ---------------------------------------------------------------------------


class Mesh2GridMapper(nn.Module, _EdgeMixin):
    """
    GNN decoder: hidden-mesh nodes → data-grid nodes.

    Ported from anemoi.models.layers.mapper.GraphTransformerBackwardMapper
    (single GPU, no distributed sharding).
    """

    def __init__(
        self,
        in_channels_src: int,
        in_channels_dst: int,
        hidden_dim: int,
        out_channels_dst: int,
        trainable_size: int = 8,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        activation: str = "GELU",
        sub_graph: HeteroData = None,
        sub_graph_edge_attributes: list = None,
        src_grid_size: int = 0,
        dst_grid_size: int = 0,
    ):
        super().__init__()
        self._register_edges(sub_graph, sub_graph_edge_attributes, src_grid_size, dst_grid_size, trainable_size)
        self.trainable = TrainableTensor(self.edge_attr_base.shape[0], trainable_size)

        # dst = data nodes (we embed from the skip-connected input features)
        self.emb_nodes_dst = nn.Linear(in_channels_dst, hidden_dim)

        self.proc = GraphTransformerMapperBlock(
            in_channels=hidden_dim,
            hidden_dim=mlp_hidden_ratio * hidden_dim,
            out_channels=hidden_dim,
            edge_dim=self.edge_dim,
            num_heads=num_heads,
            activation=activation,
        )

        self.node_data_extractor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_channels_dst),
        )

    def forward(self, x_src: Tensor, x_dst: Tensor, batch_size: int) -> Tensor:
        x_dst = self.emb_nodes_dst(x_dst)

        edge_attr = self.trainable(self.edge_attr_base, batch_size)
        edge_index = self._expand_edges(self.edge_index_base, self.edge_inc, batch_size)

        n_src = x_src.shape[0]
        n_dst = x_dst.shape[0]
        _, x_dst = self.proc(x_src, x_dst, edge_attr, edge_index, size=(n_src, n_dst))
        return self.node_data_extractor(x_dst)


# ---------------------------------------------------------------------------
# Transformer processor  (anemoi.models.layers.processor.TransformerProcessor)
# ---------------------------------------------------------------------------


class TransformerProcessor(nn.Module):
    """
    N-layer Transformer processor operating on hidden mesh nodes.

    Ported from anemoi.models.layers.processor.TransformerProcessor.
    """

    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        window_size: int = None,
        activation: str = "GELU",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerProcessorBlock(
                    num_channels=num_channels,
                    hidden_dim=mlp_hidden_ratio * num_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    activation=activation,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, batch_size: int) -> Tensor:
        for layer in self.layers:
            x = layer(x, batch_size)
        return x


# ---------------------------------------------------------------------------
# Full AIFS model  (AnemoiModelEncProcDec — stripped)
# ---------------------------------------------------------------------------


class AIFSEncProcDec(nn.Module):
    """
    AIFS GNN encoder → Transformer processor → GNN decoder.

    The graph must be a torch_geometric.data.HeteroData with node types
    'data' and 'hidden', and edge types:
      ('data', 'to', 'hidden')  — Grid2Mesh
      ('hidden', 'to', 'hidden') — Mesh2Mesh (unused here; processor is Transformer)
      ('hidden', 'to', 'data')  — Mesh2Grid

    Each sub_graph must have:
      .edge_index : (2, E) long
      .edge_attr  : (E, edge_feat_dim) float  (sin/cos of lat/lon diffs + arc length)
    And each edge type must expose an 'edge_attr' attribute.

    Parameters
    ----------
    in_channels : int
        Input feature dimension per data node (all variables).
    out_channels : int
        Output feature dimension per data node (prognostic variables).
    num_channels : int
        Hidden embedding dimension (= num_channels in anemoi paper).
    num_layers : int
        Number of Transformer processor layers.
    num_heads : int
    mlp_hidden_ratio : int
        MLP hidden dim / num_channels.
    trainable_size : int
        Number of learnable edge-feature dimensions appended to fixed features.
    window_size : int or None
        Attention window size for Transformer processor (None = global).
    activation : str
    dropout_p : float
    graph_path : str
        Path to a saved HeteroData .pt file produced by build_graph.py.
    num_data_nodes : int
        N_data (= H × W for the lat/lon grid).
    num_hidden_nodes : int
        N_hidden (number of mesh nodes).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels: int = 512,
        num_layers: int = 16,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        trainable_size: int = 8,
        window_size: int = None,
        activation: str = "GELU",
        dropout_p: float = 0.0,
        graph_path: str = None,
    ):
        super().__init__()

        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        g2m = graph[("data", "to", "hidden")]
        m2g = graph[("hidden", "to", "data")]
        n_data = graph["data"].num_nodes
        n_hidden = graph["hidden"].num_nodes
        edge_attrs = ["edge_attr"]

        # node attributes (sin/cos lat/lon → 4 dims)
        n_data_attr = graph["data"].x.shape[1] if hasattr(graph["data"], "x") else 0
        n_hidden_attr = graph["hidden"].x.shape[1] if hasattr(graph["hidden"], "x") else 0

        self.encoder = Grid2MeshMapper(
            in_channels_src=in_channels + n_data_attr,
            in_channels_dst=n_hidden_attr,
            hidden_dim=num_channels,
            trainable_size=trainable_size,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            activation=activation,
            sub_graph=g2m,
            sub_graph_edge_attributes=edge_attrs,
            src_grid_size=n_data,
            dst_grid_size=n_hidden,
        )

        self.processor = TransformerProcessor(
            num_layers=num_layers,
            num_channels=num_channels,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            window_size=window_size,
            activation=activation,
            dropout_p=dropout_p,
        )

        self.decoder = Mesh2GridMapper(
            in_channels_src=num_channels,
            in_channels_dst=in_channels + n_data_attr,
            hidden_dim=num_channels,
            out_channels_dst=out_channels,
            trainable_size=trainable_size,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            activation=activation,
            sub_graph=m2g,
            sub_graph_edge_attributes=edge_attrs,
            src_grid_size=n_hidden,
            dst_grid_size=n_data,
        )

        # register node coordinate embeddings (sin/cos lat/lon)
        if n_data_attr > 0:
            self.register_buffer("data_node_attr", graph["data"].x, persistent=False)
        else:
            self.data_node_attr = None
        if n_hidden_attr > 0:
            self.register_buffer("hidden_node_attr", graph["hidden"].x, persistent=False)
        else:
            self.hidden_node_attr = None

        self.n_data = n_data
        self.n_hidden = n_hidden

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (batch*N_data, in_channels)
            Flattened lat/lon grid, all variables.

        Returns
        -------
        (batch*N_data, out_channels)
        """
        batch_size = x.shape[0] // self.n_data

        # append fixed node attributes (sin/cos lat/lon)
        if self.data_node_attr is not None:
            node_attr = einops.repeat(self.data_node_attr, "n f -> (b n) f", b=batch_size)
            x_data = torch.cat([x, node_attr], dim=-1)
        else:
            x_data = x

        if self.hidden_node_attr is not None:
            x_hidden = einops.repeat(self.hidden_node_attr, "n f -> (b n) f", b=batch_size)
        else:
            x_hidden = torch.zeros(batch_size * self.n_hidden, 1, device=x.device, dtype=x.dtype)

        # Grid2Mesh: encoder updates both data and hidden representations
        x_data_latent, x_hidden = self.encoder(x_data, x_hidden, batch_size)

        # Transformer processor on hidden mesh + skip (hidden → hidden)
        x_hidden_proc = self.processor(x_hidden, batch_size)
        x_hidden_proc = x_hidden_proc + x_hidden

        # Mesh2Grid: src=processed hidden, dst=skip-connected data latent
        x_out = self.decoder(x_hidden_proc, x_data_latent, batch_size)

        return x_out


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITAifs(nn.Module):
    """
    CREDIT wrapper for AIFS.  Flat (B, C, H, W) I/O; returns (B, C_out, 1, H, W).

    Requires a pre-built mesh graph (build_graph.py) at graph_path.
    """

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 76,
        img_size: tuple = (181, 360),
        frames: int = 1,
        num_channels: int = 512,
        num_layers: int = 16,
        num_heads: int = 16,
        mlp_hidden_ratio: int = 4,
        trainable_size: int = 8,
        window_size: int = None,
        activation: str = "GELU",
        dropout_p: float = 0.0,
        graph_path: str = None,
    ):
        super().__init__()
        self.H, self.W = img_size
        self.N = self.H * self.W

        if graph_path is None:
            raise ValueError("graph_path must point to a HeteroData .pt file from build_graph.py")

        self.model = AIFSEncProcDec(
            in_channels=in_channels * frames,
            out_channels=out_channels,
            num_channels=num_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            trainable_size=trainable_size,
            window_size=window_size,
            activation=activation,
            dropout_p=dropout_p,
            graph_path=os.path.expandvars(graph_path),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)
        B, C, H, W = x.shape

        # flatten spatial: (B*N, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        out_flat = self.model(x_flat)  # (B*N, out_channels)

        out = out_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C_out, H, W)
        return out.unsqueeze(2)

    @classmethod
    def load_model(cls, conf):
        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        save_loc = os.path.expandvars(conf["save_loc"])
        ckpt = os.path.join(save_loc, "model_checkpoint.pt")
        if not os.path.isfile(ckpt):
            ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model

    @classmethod
    def load_model_name(cls, conf, model_name):
        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        ckpt = os.path.join(os.path.expandvars(conf["save_loc"]), model_name)
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    _root = __import__("os").path.abspath(
        __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "..")
    )
    sys.path.insert(0, _root)

    # build a tiny graph inline for the smoke test
    from credit.models.aifs.build_graph import build_graph

    H, W = 16, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        graph_path = f.name
        build_graph(lat_size=H, lon_size=W, mesh_stride=2, k_g2m=4, k_m2g=4, radius_m2m=None, save_path=graph_path)

        C_in, C_out = 8, 6
        model = CREDITAifs(
            in_channels=C_in,
            out_channels=C_out,
            img_size=(H, W),
            num_channels=64,
            num_layers=2,
            num_heads=4,
            trainable_size=4,
            graph_path=graph_path,
        ).to(device)

        x = torch.randn(1, C_in, H, W, device=device)
        y = model(x)
        assert y.shape == (1, C_out, 1, H, W), f"unexpected shape {y.shape}"
        y.mean().backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITAifs OK — output {y.shape}, {n_params:.2f}M params, device {device}")
