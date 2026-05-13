"""
NextGen WXFormer: CrossFormer U-Net + SFNO bottleneck + column attention + level embeddings.

Design additions over WXFormer:
  - Learned pressure-level embeddings at input (Pangu/Aurora style)
  - Column attention at input for explicit vertical coupling (ArchesWeather inspired)
  - Spherical Fourier bottleneck for global spectral mixing (SFNO inspired)
  - Spectral normalization throughout
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


from credit.models.base_model import BaseModel
from credit.models.wxformer.crossformer import (
    CrossEmbedLayer,
    Transformer,
    UpBlockPS,
    apply_spectral_norm,
    cast_tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level embedding
# ---------------------------------------------------------------------------


class LevelEmbedding(nn.Module):
    """Learned per-level bias broadcast-added to atmospheric input channels."""

    def __init__(self, channels: int, levels: int):
        super().__init__()
        self.channels = channels
        self.levels = levels
        self.bias = nn.Parameter(torch.zeros(channels * levels))

    def forward(self, x_atmos: torch.Tensor) -> torch.Tensor:
        # x_atmos: (B, channels*levels, H, W)
        return x_atmos + self.bias[None, :, None, None]


# ---------------------------------------------------------------------------
# Column attention
# ---------------------------------------------------------------------------


class ColumnAttention(nn.Module):
    """Multi-head attention across pressure levels at each (H, W) location.

    Operates only on the atmospheric variable channels; surface and forcing
    channels pass through unchanged.

    spatial_stride > 1 pools the spatial grid before attending and upsamples
    the correction back — reduces memory from O(B·H·W·L²) to O(B·H·W·L²/s²).
    Use stride=8 for full 640×1280 grids to keep attention weights under ~100 MB.
    """

    def __init__(self, channels: int, levels: int, num_heads: int = 4, spatial_stride: int = 1):
        super().__init__()
        self.channels = channels
        self.levels = levels
        self.spatial_stride = spatial_stride
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x_atmos: torch.Tensor) -> torch.Tensor:
        # x_atmos: (B, channels*levels, H, W)
        B, C, H, W = x_atmos.shape

        if self.spatial_stride > 1:
            x_s = F.avg_pool2d(x_atmos, self.spatial_stride, self.spatial_stride)
        else:
            x_s = x_atmos
        h, w = x_s.shape[-2:]

        x = x_s.reshape(B, self.channels, self.levels, h, w)
        x = x.permute(0, 3, 4, 2, 1).reshape(B * h * w, self.levels, self.channels)
        x_n = self.norm(x)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        delta = self.proj(attn_out)
        delta = delta.reshape(B, h, w, self.levels, self.channels).permute(0, 4, 3, 1, 2).reshape(B, C, h, w)

        if self.spatial_stride > 1:
            delta = F.interpolate(delta, size=(H, W), mode="bilinear", align_corners=False)

        return x_atmos + delta


# ---------------------------------------------------------------------------
# Spherical Fourier bottleneck
# ---------------------------------------------------------------------------


class SpectralGNNBottleneck(nn.Module):
    """Grid-agnostic global bottleneck via a learned spectral graph convolution.

    Works on any node layout (equiangular, Gaussian, HEALPix, unstructured).
    No FFT, no SHT, no grid assumption.

    Strategy: pool all spatial nodes to K learned "virtual" spectral nodes via a
    soft attention aggregation, apply a channel MLP, then broadcast corrections
    back to every spatial node. Cost is O(N·K) not O(N²), and K << N.

    Args:
        dim: Channel dimension.
        nlat: Bottleneck height (spatial nodes = nlat × nlon).
        nlon: Bottleneck width.
        num_spectral_nodes: Number of virtual spectral nodes K. Default 64.
        mlp_ratio: Hidden-dim multiplier inside the spectral MLP.
    """

    def __init__(self, dim: int, nlat: int, nlon: int, num_spectral_nodes: int = 64, mlp_ratio: float = 2.0):
        super().__init__()
        self.N = nlat * nlon
        self.K = num_spectral_nodes
        hidden = max(1, int(dim * mlp_ratio))

        self.norm = nn.GroupNorm(1, dim)

        # Learned aggregation/scatter weights as Parameters so they move with the model.
        self.agg_w = nn.Parameter(torch.randn(self.K, self.N) * (self.N**-0.5))
        self.scatter_w = nn.Parameter(torch.randn(self.N, self.K) * (self.K**-0.5))

        # Spectral MLP: mix channels at each of the K virtual nodes.
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        residual = x
        B, C, H, W = x.shape
        x = self.norm(x)

        # Flatten spatial: (B, C, N)
        x_flat = x.reshape(B, C, -1)

        # Aggregate spatial → spectral: (B, C, K)
        s = torch.einsum("bcn,kn->bck", x_flat, self.agg_w)

        # Channel MLP at each spectral node
        s = self.mlp(s.permute(0, 2, 1)).permute(0, 2, 1)  # (B, C, K)

        # Scatter spectral → spatial: (B, C, N)
        delta = torch.einsum("bck,nk->bcn", s, self.scatter_w)

        delta = delta.reshape(B, C, H, W)
        return delta + residual


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class NextGenWXFormer(BaseModel):
    """CrossFormer U-Net with spectral GNN bottleneck, column attention,
    and pressure-level embeddings.

    Args:
        image_height: Grid cells in south-north direction.
        image_width: Grid cells in west-east direction.
        frames: Number of input time steps.
        channels: Number of 3D (pressure-level) variables.
        surface_channels: Number of surface (single-level) variables.
        input_only_channels: Forcing/static variables that are input-only.
        output_only_channels: Diagnostic variables that are output-only.
        levels: Number of vertical pressure levels.
        dim: Hidden dimension at each of the 4 encoder stages.
        depth: Transformer block depth at each encoder stage.
        dim_head: Attention head dimension.
        global_window_size: Long-range attention stride at each stage.
        local_window_size: Short-range window size (scalar, shared across stages).
        cross_embed_kernel_sizes: CrossEmbedLayer kernel sizes at each stage.
        cross_embed_strides: CrossEmbedLayer strides at each stage.
        col_attn_heads: Number of heads in column attention.
        col_attn_stride: Spatial pooling stride before column attention (1 = full resolution).
            Use 8 for 640×1280 grids to stay under ~100 MB attention memory.
        decoder_col_attn: Apply column attention to atmospheric output channels before the
            residual add. Only atmospheric channels are processed; surface channels pass through.
        num_spectral_nodes: Number of virtual spectral nodes K in the GNN bottleneck.
            Higher K = more global context, more memory. Default 64.
        use_spectral_norm: Apply spectral normalization to conv/linear layers.
    """

    def __init__(
        self,
        image_height: int = 640,
        image_width: int = 1280,
        frames: int = 2,
        channels: int = 4,
        surface_channels: int = 7,
        input_only_channels: int = 3,
        output_only_channels: int = 0,
        levels: int = 15,
        dim: tuple = (64, 128, 256, 512),
        depth: tuple = (2, 2, 8, 2),
        dim_head: int = 32,
        global_window_size: tuple = (5, 5, 2, 1),
        local_window_size: int = 10,
        cross_embed_kernel_sizes: tuple = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides: tuple = (4, 2, 2, 2),
        col_attn_heads: int = 4,
        col_attn_stride: int = 1,
        decoder_col_attn: bool = False,
        num_spectral_nodes: int = 64,
        use_spectral_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        dim = cast_tuple(tuple(dim), 4)
        depth = cast_tuple(tuple(depth), 4)
        global_window_size = cast_tuple(tuple(global_window_size), 4)
        local_window_size = cast_tuple(
            tuple(local_window_size) if isinstance(local_window_size, (list, tuple)) else local_window_size, 4
        )
        cross_embed_kernel_sizes = cast_tuple(tuple([tuple(k) for k in cross_embed_kernel_sizes]), 4)
        cross_embed_strides = cast_tuple(tuple(cross_embed_strides), 4)

        self.image_height = image_height
        self.image_width = image_width
        self.frames = frames
        self.channels = channels
        self.surface_channels = surface_channels
        self.input_only_channels = input_only_channels
        self.levels = levels
        self.use_spectral_norm = use_spectral_norm
        self.decoder_col_attn = decoder_col_attn

        atmos_channels = channels * levels
        input_channels = (atmos_channels + surface_channels + input_only_channels) * frames
        last_dim = dim[-1]

        self.output_channels = channels * levels + surface_channels + output_only_channels

        # ── Input processing ─────────────────────────────────────────────
        self.level_embedding = LevelEmbedding(channels, levels)
        self.col_attn = ColumnAttention(channels, levels, num_heads=col_attn_heads, spatial_stride=col_attn_stride)

        # ── Encoder ──────────────────────────────────────────────────────
        dims = [input_channels, *dim]
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList()
        for (dim_in, dim_out), n_layers, g_wsz, l_wsz, kernels, stride in zip(
            dim_pairs,
            depth,
            global_window_size,
            local_window_size,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            self.layers.append(
                nn.ModuleList(
                    [
                        CrossEmbedLayer(dim_in=dim_in, dim_out=dim_out, kernel_sizes=kernels, stride=stride),
                        Transformer(
                            dim=dim_out,
                            local_window_size=l_wsz,
                            global_window_size=g_wsz,
                            depth=n_layers,
                            dim_head=dim_head,
                        ),
                    ]
                )
            )

        # ── Spectral GNN bottleneck ──────────────────────────────────────
        bn_h, bn_w = image_height, image_width
        for s in cross_embed_strides:
            bn_h //= s
            bn_w //= s
        self.sfno_bottleneck = SpectralGNNBottleneck(last_dim, bn_h, bn_w, num_spectral_nodes=num_spectral_nodes)

        # ── Decoder ──────────────────────────────────────────────────────
        scale = 2
        self.up_block1 = UpBlockPS(last_dim, last_dim // 2, dim[0])
        self.up_block2 = UpBlockPS(2 * (last_dim // 2), last_dim // 4, dim[0])
        self.up_block3 = UpBlockPS(2 * (last_dim // 4), last_dim // 8, dim[0])
        self.up_block4 = nn.Sequential(
            nn.Conv2d(2 * (last_dim // 8), self.output_channels * scale**2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
        )

        if decoder_col_attn:
            # Reuses col_attn_stride — same memory budget as the encoder col_attn.
            # Only applied to atmospheric channels; surface channels are unaffected.
            self.dec_col_attn = ColumnAttention(
                channels, levels, num_heads=col_attn_heads, spatial_stride=col_attn_stride
            )

        if use_spectral_norm:
            apply_spectral_norm(self.layers)
            apply_spectral_norm(self.sfno_bottleneck)
            apply_spectral_norm(self.up_block1)
            apply_spectral_norm(self.up_block2)
            apply_spectral_norm(self.up_block3)
            apply_spectral_norm(self.up_block4)
            apply_spectral_norm(self.level_embedding)
            # col_attn excluded: MHA internal linear buffers conflict with spectral norm device placement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        # x: (B, C_in, T, H, W)
        B, C, T, H, W = x.shape

        if T > 1:
            x = x.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
        else:
            x = x.squeeze(2)

        # Residual base: prognostic channels of the last input frame.
        # Model predicts a delta; adding the base means output ≈ persistence at
        # random init, which gives non-zero ACC from the very first batch.
        total_per_frame = self.channels * self.levels + self.surface_channels + self.input_only_channels
        last_frame_offset = (T - 1) * total_per_frame
        x_res = x[:, last_frame_offset : last_frame_offset + self.output_channels]

        # Apply level embedding and column attention to each frame's atmos channels
        atmos_size = self.channels * self.levels

        frame_slices = []
        for t in range(T):
            offset = t * total_per_frame
            x_atmos = x[:, offset : offset + atmos_size]
            x_rest = x[:, offset + atmos_size : offset + total_per_frame]
            x_atmos = self.level_embedding(x_atmos)
            x_atmos = self.col_attn(x_atmos)
            frame_slices.extend([x_atmos, x_rest])

        x = torch.cat(frame_slices, dim=1)

        # Encode
        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        # Spectral GNN bottleneck
        x = self.sfno_bottleneck(x)

        # Decode
        x = self.up_block1(x)
        x = torch.cat([x, encodings[2]], dim=1)

        x = self.up_block2(x)
        x = torch.cat([x, encodings[1]], dim=1)

        x = self.up_block3(x)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear", align_corners=False)

        # Decoder column attention: refine vertical coupling in prediction space.
        # Only atmospheric channels are processed; surface channels pass through.
        if self.decoder_col_attn:
            atmos_size = self.channels * self.levels
            x_atmos = self.dec_col_attn(x[:, :atmos_size])
            x = torch.cat([x_atmos, x[:, atmos_size:]], dim=1)

        # Add residual: output is last-frame state + predicted delta
        x = x + x_res

        return x.unsqueeze(2)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, H, W = 1, 2, 64, 128
    channels, levels, surface_channels = 4, 5, 7
    input_only_channels, output_only_channels = 3, 0

    C_in = channels * levels + surface_channels + input_only_channels
    C_out = channels * levels + surface_channels + output_only_channels

    model = NextGenWXFormer(
        image_height=H,
        image_width=W,
        frames=T,
        channels=channels,
        surface_channels=surface_channels,
        input_only_channels=input_only_channels,
        output_only_channels=output_only_channels,
        levels=levels,
        dim=(32, 64, 128, 256),
        depth=(2, 2, 2, 2),
        dim_head=8,
        global_window_size=(4, 2, 2, 1),
        local_window_size=4,
        cross_embed_kernel_sizes=((2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        col_attn_heads=4,
        use_spectral_norm=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters : {n_params:.2f}M")
    print(f"Device     : {device}")

    x = torch.randn(B, C_in, T, H, W, device=device)
    print(f"Input      : {tuple(x.shape)}")

    y = model(x)
    print(f"Output     : {tuple(y.shape)}")

    expected = (B, C_out, 1, H, W)
    assert tuple(y.shape) == expected, f"shape mismatch: {tuple(y.shape)} != {expected}"
    assert not torch.isnan(y).any(), "NaN in output"

    y.mean().backward()
    print("Backward   : OK")
    print(f"Expected   : {expected}  ✓")
