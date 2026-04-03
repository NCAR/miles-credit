"""AIFS-inspired global weather Transformer for CREDIT.

AIFS (Artificial Intelligence Forecasting System) — Lang et al., 2024.
https://arxiv.org/abs/2406.01465

The original AIFS uses a GNN encoder to map ERA5 data onto a reduced Gaussian
mesh, a 16-layer Transformer processor, and a GNN decoder back to the output
grid.  This CREDIT implementation simplifies the architecture by operating
directly on the regular lat/lon grid — removing the GNN encode/decode steps
and the reduced Gaussian mesh.  This keeps all the modelling power of the
Transformer processor while staying fully compatible with CREDIT's data pipeline.

Architecture:
    Linear token embedding  (C_in × H × W → hidden_dim)
    + Sinusoidal lat/lon position encoding
    → N × TransformerProcessorLayer  (MHSA + FF + residual + pre-norm)
    → Linear output projection  (hidden_dim → C_out × H × W)

The model operates on flattened spatial tokens ``(B, H*W, hidden_dim)`` —
equivalent to treating every grid cell as a "node" in the AIFS graph.

For very large grids an optional local attention window can be applied
(``window_size > 0``), which reduces the O(N²) complexity to O(N·W).

Input/output: ``(B, C, H, W)`` flat tensors.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

__all__ = ["AIFSProcessor", "CREDITAifs"]


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class LatLonPositionalEncoding(nn.Module):
    """Sinusoidal lat/lon positional encoding stored as a learnable-free buffer.

    Encodes (lat, lon) as sinusoids with multiple frequencies and projects to
    ``hidden_dim`` via a small linear layer.

    Args:
        hidden_dim: Target embedding dimension.
        num_freqs: Number of sinusoidal frequencies per coordinate.
    """

    def __init__(self, hidden_dim: int, num_freqs: int = 8) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        # 2 coords × 2 trig fns × num_freqs
        pos_dim = 4 * num_freqs
        self.proj = nn.Linear(pos_dim, hidden_dim, bias=False)

    def _encode(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N,) in radians → (N, 2*num_freqs)"""
        freqs = torch.arange(1, self.num_freqs + 1, dtype=coords.dtype, device=coords.device)
        angles = coords.unsqueeze(-1) * freqs.unsqueeze(0) * math.pi
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lat: ``(H,)`` latitudes in degrees.
            lon: ``(W,)`` longitudes in degrees.

        Returns:
            ``(H*W, hidden_dim)`` positional encodings.
        """
        lat_rad = lat * (math.pi / 180.0)
        lon_rad = lon * (math.pi / 180.0)
        # Build (H*W,) grids
        lat_grid = lat_rad.unsqueeze(1).expand(-1, lon.shape[0]).reshape(-1)  # H*W
        lon_grid = lon_rad.unsqueeze(0).expand(lat.shape[0], -1).reshape(-1)  # H*W
        enc = torch.cat([self._encode(lat_grid), self._encode(lon_grid)], dim=-1)  # H*W × 4*num_freqs
        return self.proj(enc)  # H*W × hidden_dim


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention (no flash-attn dependency).

    Args:
        hidden_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        window_size: If > 0, apply local sliding-window attention
            (reduces O(N²) to O(N·W)).  0 = global attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C)"""
        B, N, C = x.shape
        H = self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, head_dim)

        if self.window_size > 0:
            # Sliding window attention: attend only within local windows
            out = self._windowed_attn(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.drop(out)

    def _windowed_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Approximate local attention via chunked processing."""
        B, H, N, D = q.shape
        W = self.window_size
        # Pad N to a multiple of W
        pad = (W - N % W) % W
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        N2 = q.shape[2]
        n_wins = N2 // W

        q = q.reshape(B, H, n_wins, W, D)
        k = k.reshape(B, H, n_wins, W, D)
        v = v.reshape(B, H, n_wins, W, D)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, n_wins, W, W)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, H, n_wins, W, D)
        out = out.reshape(B, H, N2, D)[:, :, :N, :]
        return out


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer processor layer
# ---------------------------------------------------------------------------


class TransformerProcessorLayer(nn.Module):
    """Single Transformer layer: pre-norm MHSA + pre-norm FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout=dropout, window_size=window_size)
        self.ff = FeedForward(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class AIFSProcessor(nn.Module):
    """AIFS-inspired lat/lon Transformer processor.

    Replaces AIFS's GNN encoder/decoder with simple linear projections on the
    flattened lat/lon grid.  The Transformer processor — the core of AIFS — is
    kept intact.

    Args:
        n_input_channels: Number of input channels (all variables flattened).
        n_output_channels: Number of output channels.
        hidden_dim: Transformer hidden dimension.
        num_layers: Number of Transformer processor layers.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden / embedding ratio.
        dropout: Dropout rate.
        drop_path_rate: Max stochastic depth rate (linearly increases with depth).
        window_size: Local attention window size in tokens (0 = global).
        num_pos_freqs: Number of sinusoidal frequencies for lat/lon encoding.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        hidden_dim: int = 512,
        num_layers: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        window_size: int = 0,
        num_pos_freqs: int = 8,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # Encoder: project input features → hidden_dim
        self.input_proj = nn.Linear(n_input_channels, hidden_dim)

        # Positional encoding
        self.pos_enc = LatLonPositionalEncoding(hidden_dim, num_freqs=num_pos_freqs)

        # Transformer processor
        dp_rates = torch.linspace(0, drop_path_rate, num_layers).tolist()
        self.layers = nn.ModuleList(
            [
                TransformerProcessorLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dp_rates[i],
                    window_size=window_size,
                )
                for i in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Decoder: project hidden_dim → output features
        self.output_proj = nn.Linear(hidden_dim, n_output_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        lat: Optional[torch.Tensor] = None,
        lon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:   ``(B, C, H, W)`` input tensor.
            lat: ``(H,)`` latitudes in degrees.  If None, uses linspace(-90, 90).
            lon: ``(W,)`` longitudes in degrees.  If None, uses linspace(0, 359).

        Returns:
            ``(B, C_out, H, W)`` predicted tensor.
        """
        B, C, H, W = x.shape

        # Flatten spatial: (B, H*W, C)
        x_tok = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Embed tokens
        x_tok = self.input_proj(x_tok)  # (B, H*W, hidden_dim)

        # Add positional encoding
        if lat is None:
            lat = torch.linspace(90, -90, H, device=x.device, dtype=x.dtype)
        if lon is None:
            lon = torch.linspace(0, 359, W, device=x.device, dtype=x.dtype)
        pos = self.pos_enc(lat, lon)  # (H*W, hidden_dim)
        x_tok = x_tok + pos.unsqueeze(0)  # broadcast over batch

        # Transformer processor
        for layer in self.layers:
            x_tok = layer(x_tok)
        x_tok = self.norm(x_tok)

        # Project to output
        out = self.output_proj(x_tok)  # (B, H*W, C_out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C_out, H, W)
        return out


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITAifs(nn.Module):
    """AIFS processor wrapped for CREDIT's flat-tensor training pipeline.

    The model is fully flexible: any combination of surf/atmos/static variables
    can be used as input.  The output contains surf + atmos channels (static
    vars are not predicted).

    Channel layout (same convention as CREDITAurora / CREDITPangu):
        [surf_vars..., atmos_var_0_lev_0, ..., atmos_var_M_lev_P, static_vars...]

    Args:
        surf_vars: List of surface variable names.
        atmos_vars: List of atmospheric variable names.
        static_vars: List of static variable names (included as input only).
        atmos_levels: List of pressure levels in hPa.
        **aifs_kwargs: Forwarded to :class:`.AIFSProcessor`.
    """

    def __init__(
        self,
        surf_vars: List[str] = ("2t", "10u", "10v", "msl"),
        atmos_vars: List[str] = ("z", "u", "v", "t", "q"),
        static_vars: List[str] = ("lsm", "z", "slt"),
        atmos_levels: List[int] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        **aifs_kwargs,
    ) -> None:
        super().__init__()
        self.surf_vars = list(surf_vars)
        self.atmos_vars = list(atmos_vars)
        self.static_vars = list(static_vars)
        self.atmos_levels = list(atmos_levels)
        n_surf = len(surf_vars)
        n_atmos = len(atmos_vars)
        n_static = len(static_vars)
        n_levels = len(atmos_levels)
        self.n_surf = n_surf
        self.n_atmos = n_atmos
        self.n_static = n_static
        self.n_levels = n_levels

        n_in = n_surf + n_atmos * n_levels + n_static
        n_out = n_surf + n_atmos * n_levels

        self.model = AIFSProcessor(
            n_input_channels=n_in,
            n_output_channels=n_out,
            **aifs_kwargs,
        )

    @property
    def n_input_channels(self) -> int:
        return self.n_surf + self.n_atmos * self.n_levels + self.n_static

    @property
    def n_output_channels(self) -> int:
        return self.n_surf + self.n_atmos * self.n_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running AIFS smoke test on {device}")

    H, W = 32, 64
    surf_vars = ["2t", "10u", "10v", "msl"]
    atmos_vars = ["z", "u", "v", "t", "q"]
    static_vars = ["lsm", "z", "slt"]
    levels = [500, 850, 1000]
    n_surf = len(surf_vars)
    n_atmos = len(atmos_vars)
    n_static = len(static_vars)
    n_levels = len(levels)
    C_in = n_surf + n_atmos * n_levels + n_static
    C_out = n_surf + n_atmos * n_levels

    model = CREDITAifs(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        atmos_levels=levels,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path_rate=0.1,
        window_size=16,  # use local windows to keep memory low
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f} M")
    print(f"  Input channels : {C_in}")
    print(f"  Output channels: {C_out}")

    B = 2
    x = torch.randn(B, C_in, H, W, device=device)
    print(f"  Input  shape: {x.shape}")

    with torch.no_grad():
        y = model(x)
    print(f"  Output shape: {y.shape}")
    assert y.shape == (B, C_out, H, W), f"Unexpected output shape {y.shape}"

    x.requires_grad_(True)
    y = model(x)
    y.mean().backward()
    print(f"  Input grad shape: {x.grad.shape}")
    print("AIFS smoke test PASSED")
