import torch
import logging
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import MessagePassing, graclus
from torch_geometric.utils import coalesce, softmax, scatter

from credit.models.base_model import BaseModel
from credit.postblock import PostBlock

logger = logging.getLogger(__name__)


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


class NodeEmbedding(nn.Module):
    """Embeds input node features into the initial latent dimension."""

    def __init__(self, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        # x: (B, N, in_chans)
        return self.norm(self.proj(x))


class StaticGraphConv(MessagePassing):
    """
    Efficient Graph Convolution that processes batched node features (B, N, C)
    using a shared, static edge_index without duplicating the graph.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean", node_dim=0)
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Transform node features
        x = self.lin(x)  # (B, N, out_channels)

        # Transpose to (N, B, C) for efficient batched propagation in PyG
        x_trans = x.transpose(0, 1)

        # Propagate messages along the graph
        out_trans = self.propagate(edge_index, x=x_trans)  # (N, B, C)

        # Transpose back
        return out_trans.transpose(0, 1)  # (B, N, C)


class GraphAttention(MessagePassing):
    """
    Multi-head Graph Attention (GAT) capable of operating natively on
    batched (B, N, C) features over a static sparse edge_index.
    """

    def __init__(self, dim, heads=4, dim_head=32, dropout=0.0):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head

        self.lin_q = nn.Linear(dim, self.inner_dim)
        self.lin_k = nn.Linear(dim, self.inner_dim)
        self.lin_v = nn.Linear(dim, self.inner_dim)

        self.out_proj = nn.Linear(self.inner_dim, dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x: (B, N, dim) -> transpose to (N, B, dim)
        x_t = x.transpose(0, 1)

        q = self.lin_q(x_t).view(-1, x.size(0), self.heads, self.dim_head)
        k = self.lin_k(x_t).view(-1, x.size(0), self.heads, self.dim_head)
        v = self.lin_v(x_t).view(-1, x.size(0), self.heads, self.dim_head)

        # Output is (N, B, heads, dim_head)
        out_t = self.propagate(edge_index, q=q, k=k, v=v)

        out_t = out_t.reshape(-1, x.size(0), self.inner_dim)
        out = out_t.transpose(0, 1)  # (B, N, inner_dim)
        return self.out_proj(out)

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        # Compute attention scores
        alpha = (q_i * k_j).sum(dim=-1) / (self.dim_head**0.5)

        # Softmax over neighborhood
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return v_j * alpha.unsqueeze(-1)


class FeedForward(nn.Module):
    """Token-wise feedforward network (replaces 1x1 convolutions)."""

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.layers(x)


class GraphTransformer(nn.Module):
    """
    Replaces the image-based Transformer. Performs self/graph-attention
    over the arbitrary adjacency graph.
    """

    def __init__(self, dim, depth=4, heads=4, dim_head=32, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        GraphAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                        nn.LayerNorm(dim),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x, edge_index):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x), edge_index) + x
            x = ff(norm2(x)) + x
        return x


class GraphPool(nn.Module):
    """
    Smartly pools neighboring cells by finding clusters via Graclus based on
    the graph topology, then pooling features and sparsifying the edges.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x, edge_index):
        # 1. Cluster nodes (deterministic topological clustering via graclus)
        cluster = graclus(edge_index)
        _, cluster = torch.unique(cluster, return_inverse=True)

        # 2. Pool node features (B, N, C) -> (B, N_coarse, C)
        x_proj = self.proj(x)
        x_pooled = scatter(x_proj, cluster, dim=1, reduce="mean")

        # 3. Pool / Coarsen edges
        edge_index_pooled = cluster[edge_index]
        edge_index_pooled, _ = coalesce(edge_index_pooled, None, num_nodes=cluster.max().item() + 1)

        return x_pooled, edge_index_pooled, cluster


class GraphUpBlock(nn.Module):
    """
    Unpools features from a coarser grid using the stored clustering
    mapping, concatenates with the skip-connection, and refines them.
    """

    def __init__(self, in_chans, out_chans, num_residuals=2):
        super().__init__()
        self.proj = nn.Linear(in_chans, out_chans)

        self.res_blocks = nn.ModuleList()
        for _ in range(num_residuals):
            self.res_blocks.append(
                nn.ModuleList([StaticGraphConv(out_chans, out_chans), nn.LayerNorm(out_chans), nn.SiLU()])
            )

    def forward(self, x, edge_index):
        # x is the combined feature map (unpooled + skip)
        x = self.proj(x)

        shortcut = x
        for conv, norm, act in self.res_blocks:
            x = conv(x, edge_index)
            x = norm(x)
            x = act(x)

        return x + shortcut


class GraphCrossFormer(BaseModel):
    def __init__(
        self,
        frames: int = 2,
        output_frames: int = 1,
        channels: int = 4,
        surface_channels: int = 7,
        input_only_channels: int = 3,
        output_only_channels: int = 0,
        levels: int = 15,
        dim: tuple = (64, 128, 256, 512),
        depth: tuple = (2, 2, 8, 2),
        dim_head: int = 32,
        heads: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        post_conf: dict = None,
        **kwargs,
    ):
        """
        A Graph variant of CrossFormer designed to process arbitrary grids.
        Instead of regular spatial kernels, this operates purely on `edge_index`.
        """
        super().__init__()

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)

        self.frames = frames
        self.output_frames = output_frames

        self.input_only_channels = input_only_channels
        self.base_input_channels = channels * levels + surface_channels + input_only_channels
        self.input_channels = self.base_input_channels * frames

        self.base_output_channels = channels * levels + surface_channels + output_only_channels
        self.output_channels = self.base_output_channels * output_frames

        if kwargs.get("diffusion"):
            self.input_channels += self.output_channels

        self.use_post_block = post_conf is not None and post_conf.get("activate", False)

        # 1. Embed Features
        self.node_embedding = NodeEmbedding(self.input_channels, dim[0])

        # 2. Downsampling & Encoders
        self.down_layers = nn.ModuleList([])
        dims = [dim[0], *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        for (dim_in, dim_out), num_layers in zip(dim_in_and_out, depth):
            pool = GraphPool(dim_in, dim_out)
            transformer = GraphTransformer(
                dim=dim_out,
                depth=num_layers,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
            self.down_layers.append(nn.ModuleList([pool, transformer]))

        # 3. Upsampling & Decoders
        self.up_blocks = nn.ModuleList(
            [
                GraphUpBlock(dim[3] + dim[2], dim[2]),
                GraphUpBlock(dim[2] + dim[1], dim[1]),
                GraphUpBlock(dim[1] + dim[0], dim[0]),
                GraphUpBlock(dim[0] + dim[0], dim[0]),
            ]
        )

        # 4. Output Projection
        self.final_proj = nn.Linear(dim[0], self.output_channels)

        if self.use_post_block:
            self.postblock = PostBlock(post_conf)

    def forward(self, x, edge_index):
        """
        Args:
            x (torch.Tensor): Input features of shape (B, C, T, N) or (B, C, N).
            edge_index (torch.LongTensor): Sparse adjacency matrix of shape (2, E).
        """
        # Flatten Time into Channels if needed
        if x.dim() == 4:
            B, C, T, N = x.shape
            x = x.reshape(B, C * T, N)
        elif x.dim() == 3:
            B, C_T, N = x.shape
        else:
            raise ValueError(f"Expected input to have 3 or 4 dimensions, got {x.dim()}")

        x_copy = x.clone().detach() if self.use_post_block else None

        # (B, N, C_T)
        x = x.transpose(1, 2)
        x = self.node_embedding(x)

        encodings = []
        graphs = []
        current_edge_index = edge_index

        # --- ENCODER ---
        for pool, transformer in self.down_layers:
            encodings.append(x)  # Save skip-connection

            # Smartly pool neighboring cells into a coarser grid
            x, next_edge_index, cluster = pool(x, current_edge_index)
            graphs.append((current_edge_index, cluster))

            x = transformer(x, next_edge_index)
            current_edge_index = next_edge_index

        # --- DECODER ---
        for i in range(4):
            prev_edge_index, cluster = graphs[-(i + 1)]
            skip_x = encodings[-(i + 1)]

            # Unpool by mapping coarse nodes back to their finer topology
            x_up = x[:, cluster, :]

            # Concatenate skip-connection with unpooled features
            x_cat = torch.cat([x_up, skip_x], dim=-1)

            # Refine
            x = self.up_blocks[i](x_cat, prev_edge_index)

        x = self.final_proj(x)  # (B, N, C_out)
        x = x.transpose(1, 2)  # (B, C_out, N)

        B, _, N = x.shape
        x = x.view(B, self.base_output_channels, self.output_frames, N)

        if self.use_post_block:
            x_post = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x_post)

        return x

    def rk4(self, x, edge_index):
        def integrate_step(x, k, factor):
            return self.forward(x + k * factor, edge_index)

        k1 = self.forward(x, edge_index)
        k1 = torch.cat([x[:, :, -2:-1], k1], dim=2)

        k2 = integrate_step(x, k1, 0.5)
        k2 = torch.cat([x[:, :, -2:-1], k2], dim=2)

        k3 = integrate_step(x, k2, 0.5)
        k3 = torch.cat([x[:, :, -2:-1], k3], dim=2)

        k4 = integrate_step(x, k3, 1.0)

        return (k1 + 2 * k2 + 2 * k3 + k4) / 6
