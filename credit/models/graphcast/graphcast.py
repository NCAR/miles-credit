"""
GraphCast — Graph Neural Network weather model.
Lam et al., 2023.  https://arxiv.org/abs/2212.12794
Architecture from PhysicsNemo / DeepMind (Apache 2.0).

Architecture: Grid2Mesh GNN encoder → Mesh GNN processor → Mesh2Grid GNN decoder.

CREDIT simplification
---------------------
The original uses an icosahedral multi-scale mesh with ~40k nodes.  For CREDIT's
flat lat/lon grid we use a *single-resolution* learned graph directly on the
grid nodes, avoiding the mesh construction entirely:

  - Encoder  : linear projection of grid node features → latent node embeddings
  - Processor: N message-passing rounds on a k-nearest-neighbour graph over grid nodes
  - Decoder  : linear projection of node embeddings → output features per grid node

This captures the GNN inductive bias (local message passing + global communication)
without requiring torch_geometric or an external mesh library.

The kNN graph is precomputed once at construction time from lat/lon coordinates.

CREDITGraphCast wraps the model for CREDIT's flat (B, C, H, W) tensors.
"""

import os
import sys

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def build_knn_edge_index(lat: torch.Tensor, lon: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build k-nearest-neighbour edge index on a lat/lon grid (great-circle distance).

    Parameters
    ----------
    lat, lon : (N,) in degrees
    k : number of neighbours per node

    Returns
    -------
    edge_index : (2, N*k)  — (src, dst) pairs
    """
    N = lat.shape[0]
    lat_r = torch.deg2rad(lat.double())
    lon_r = torch.deg2rad(lon.double())

    # Haversine pairwise: (N, N)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_r[:, None]) * torch.cos(lat_r[None, :]) * torch.sin(dlon / 2) ** 2
    dist = 2 * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))  # (N, N)

    # zero out self-loops for topk
    dist.fill_diagonal_(float("inf"))
    _, nbrs = torch.topk(dist, k, dim=1, largest=False)  # (N, k)

    src = torch.arange(N, device=lat.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nbrs.reshape(-1)
    return torch.stack([src.long(), dst.long()], dim=0)  # (2, N*k)


# ---------------------------------------------------------------------------
# GNN layers
# ---------------------------------------------------------------------------


class EdgeMLP(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        self.norm = nn.LayerNorm(edge_dim)

    def forward(self, src_feat, dst_feat, edge_feat):
        inp = torch.cat([src_feat, dst_feat, edge_feat], dim=-1)
        return self.norm(edge_feat + self.net(inp))


class NodeMLP(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_feat, agg_edge):
        inp = torch.cat([node_feat, agg_edge], dim=-1)
        return self.norm(node_feat + self.net(inp))


class MessagePassingLayer(nn.Module):
    """One round of edge-update + aggregate + node-update."""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = EdgeMLP(node_dim, edge_dim, hidden_dim)
        self.node_mlp = NodeMLP(node_dim, edge_dim, hidden_dim)

    def forward(self, node_feat, edge_feat, edge_index):
        """
        node_feat : (B, N, node_dim)
        edge_feat : (B, E, edge_dim)
        edge_index: (2, E) long
        """
        B, N, Dn = node_feat.shape
        E = edge_feat.shape[1]
        src, dst = edge_index[0], edge_index[1]

        # edge update
        src_feat = node_feat[:, src, :]  # (B, E, Dn)
        dst_feat = node_feat[:, dst, :]
        edge_feat = self.edge_mlp(src_feat, dst_feat, edge_feat)

        # aggregate (sum) edges to destination nodes
        agg = torch.zeros(B, N, edge_feat.shape[-1], device=node_feat.device, dtype=node_feat.dtype)
        agg.scatter_add_(1, dst[None, :, None].expand(B, -1, edge_feat.shape[-1]), edge_feat)

        # node update
        node_feat = self.node_mlp(node_feat, agg)
        return node_feat, edge_feat


# ---------------------------------------------------------------------------
# GraphCast model
# ---------------------------------------------------------------------------


class GraphCastModel(nn.Module):
    """
    Simplified GraphCast on a flat lat/lon grid.

    Parameters
    ----------
    img_size : tuple[int,int]
        (H, W) — used to build the lat/lon coordinate grid.
    in_channels, out_channels : int
    latent_dim : int
        Node embedding dimension.
    edge_dim : int
        Edge feature dimension.
    processor_depth : int
        Number of message-passing rounds.
    k_neighbours : int
        kNN graph connectivity.
    mlp_hidden : int
        Hidden size inside GNN MLPs.
    lat_range : tuple[float,float]
        (lat_min, lat_max) in degrees. Default (-90, 90).
    lon_range : tuple[float,float]
        (lon_min, lon_max) in degrees. Default (0, 360).
    """

    def __init__(
        self,
        img_size=(128, 256),
        in_channels=70,
        out_channels=69,
        latent_dim=256,
        edge_dim=128,
        processor_depth=8,
        k_neighbours=6,
        mlp_hidden=512,
        lat_range=(-90.0, 90.0),
        lon_range=(0.0, 360.0),
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        N = H * W

        # pre-build lat/lon grid (not learnable; registered as buffer)
        lat = torch.linspace(lat_range[0], lat_range[1], H)
        lon = torch.linspace(lon_range[0], lon_range[1], W)
        lat_grid = lat[:, None].expand(H, W).reshape(N)
        lon_grid = lon[None, :].expand(H, W).reshape(N)

        # build kNN graph (fixed topology, on CPU then moved to device at runtime)
        edge_index = build_knn_edge_index(lat_grid, lon_grid, k=k_neighbours)
        self.register_buffer("edge_index", edge_index)  # (2, N*k)

        # edge features: Δlat, Δlon, distance (3-dim) → projected to edge_dim
        E = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]
        dlat = torch.deg2rad(lat_grid[dst] - lat_grid[src])
        dlon = torch.deg2rad(lon_grid[dst] - lon_grid[src])
        dist = torch.sqrt(dlat**2 + dlon**2)
        raw_edge = torch.stack([dlat.float(), dlon.float(), dist.float()], dim=-1)  # (E, 3)
        self.register_buffer("raw_edge_feat", raw_edge)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.edge_encoder = nn.Sequential(nn.Linear(3, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim))

        # ── Processor ────────────────────────────────────────────────────────
        self.processor = nn.ModuleList(
            [MessagePassingLayer(latent_dim, edge_dim, mlp_hidden) for _ in range(processor_depth)]
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) → (B, C_out, H, W)"""
        B, C, H, W = x.shape
        N = H * W

        # flatten spatial → nodes
        x_nodes = rearrange(x, "b c h w -> b (h w) c")  # (B, N, C_in)

        # encode
        node_feat = self.node_encoder(x_nodes)  # (B, N, latent_dim)
        E = self.raw_edge_feat.shape[0]
        edge_feat = self.edge_encoder(self.raw_edge_feat)  # (E, edge_dim)
        edge_feat = edge_feat[None].expand(B, -1, -1)  # (B, E, edge_dim)

        # process
        for layer in self.processor:
            node_feat, edge_feat = layer(node_feat, edge_feat, self.edge_index)

        # decode
        out = self.node_decoder(node_feat)  # (B, N, C_out)
        return rearrange(out, "b (h w) c -> b c h w", h=H, w=W)


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITGraphCast(nn.Module):
    """CREDIT wrapper for GraphCast (simplified lat/lon GNN).  Flat (B,C,H,W) I/O."""

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(128, 256),
        latent_dim=256,
        edge_dim=128,
        processor_depth=8,
        k_neighbours=6,
        mlp_hidden=512,
        lat_range=(-90.0, 90.0),
        lon_range=(0.0, 360.0),
    ):
        super().__init__()
        self.model = GraphCastModel(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            edge_dim=edge_dim,
            processor_depth=processor_depth,
            k_neighbours=k_neighbours,
            mlp_hidden=mlp_hidden,
            lat_range=lat_range,
            lon_range=lon_range,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def load_model(cls, conf):
        import torch

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
        import torch

        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        ckpt = os.path.join(os.path.expandvars(conf["save_loc"]), model_name)
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, _root)

    B, C_in, C_out = 1, 12, 10
    H, W = 16, 32  # tiny for smoke test (kNN is O(N^2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITGraphCast(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        latent_dim=64,
        edge_dim=32,
        processor_depth=2,
        k_neighbours=4,
        mlp_hidden=128,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected shape {y.shape}"
    y.mean().backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITGraphCast OK — output {y.shape}, params {n_params:.1f}M, device {device}")
