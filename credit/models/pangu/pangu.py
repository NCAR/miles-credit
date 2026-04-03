"""Pangu-Weather: Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks.
Bi et al., Nature 2023.  https://doi.org/10.1038/s41586-023-06185-3

PyTorch implementation derived from the official pseudocode:
  https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

Architecture: 3D Earth-Specific Transformer with U-shaped encoder-decoder.
  PatchEmbedding (Conv3d + Conv2d)
  → EarthSpecificLayer × 2  (encoder stage 1, dim=192)
  → DownSample
  → EarthSpecificLayer × 6  (encoder stage 2, dim=384)
  → EarthSpecificLayer × 6  (decoder stage 1, dim=384)
  → UpSample
  → EarthSpecificLayer × 2  (decoder stage 2, dim=192)
  → Skip-concat  (dim=384)
  → PatchRecovery (ConvTranspose3d + ConvTranspose2d)

Input  : upper-air tensor  (B, n_atmos_vars, n_levels, H, W)
         surface  tensor   (B, n_surf_vars + n_static_vars, H, W)
Output : upper-air tensor  (B, n_atmos_vars, n_levels, H, W)
         surface  tensor   (B, n_surf_vars,  H, W)

The CREDIT-compatible wrapper ``CREDITPangu`` splices the flat (B, C, H, W)
channel tensor into these two inputs and reassembles the output.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

__all__ = ["PanguModel", "CREDITPangu"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_multiple(x: torch.Tensor, window: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Zero-pad a 5-D tensor ``(B, C, Z, H, W)`` so Z, H, W are multiples of window."""
    B, C, Z, H, W = x.shape
    wz, wh, ww = window
    pad_z = (wz - Z % wz) % wz
    pad_h = (wh - H % wh) % wh
    pad_w = (ww - W % ww) % ww
    # F.pad order: last dim first → (W_left, W_right, H_top, H_bot, Z_front, Z_back)
    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_z))
    return x, (pad_z, pad_h, pad_w)


def _crop(x: torch.Tensor, pads: Tuple[int, ...]) -> torch.Tensor:
    """Remove zero-padding added by ``_pad_to_multiple``."""
    pad_z, pad_h, pad_w = pads
    if pad_z:
        x = x[..., : x.shape[-3] - pad_z, :, :]
    if pad_h:
        x = x[..., : x.shape[-2] - pad_h, :]
    if pad_w:
        x = x[..., : x.shape[-1] - pad_w]
    return x


# ---------------------------------------------------------------------------
# Earth-Specific Position Bias
# ---------------------------------------------------------------------------


class EarthSpecificBias(nn.Module):
    """Learnable 3-D Earth-Specific relative position bias (Bi et al., 2023, Section 2.3).

    Unlike standard Swin position bias, the longitude axis is treated as periodic
    (shared between query and key) while the pressure and latitude axes are *not*
    periodic, giving the asymmetric parameterisation described in the paper.

    Args:
        window_size: (Wz, Wh, Ww) — window depth, height, width.
        num_heads: Number of attention heads.
        type_of_windows: Number of distinct window types = (Z//Wz) * (H//Wh).
    """

    def __init__(
        self,
        window_size: Tuple[int, int, int],
        num_heads: int,
        type_of_windows: int,
    ) -> None:
        super().__init__()
        Wz, Wh, Ww = window_size
        self.window_size = window_size
        self.num_heads = num_heads

        # Bias table: rows index unique relative positions; cols index window-type × head.
        n_unique = (2 * Ww - 1) * Wh * Wh * Wz * Wz
        self.earth_specific_bias = nn.Parameter(torch.zeros(n_unique, type_of_windows, num_heads))
        trunc_normal_(self.earth_specific_bias, std=0.02)

        # Pre-compute position index (Wz*Wh*Ww) × (Wz*Wh*Ww)
        self.register_buffer("position_index", self._build_index(window_size), persistent=False)

    @staticmethod
    def _build_index(window_size: Tuple[int, int, int]) -> torch.Tensor:
        Wz, Wh, Ww = window_size
        # Pressure (z): query uses [0..Wz-1], key offsets [0, -Wz, -2Wz, ...]
        coords_zi = torch.arange(Wz)
        coords_zj = -torch.arange(Wz) * Wz
        # Latitude (h)
        coords_hi = torch.arange(Wh)
        coords_hj = -torch.arange(Wh) * Wh
        # Longitude (w): shared between q and k (periodic → only relative offset matters)
        coords_w = torch.arange(Ww)

        # Build grids: each grid has shape (Wz, Wh, Ww)
        gz_i, gh_i, gw = torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij")
        gz_j, gh_j, _ = torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij")

        flat_i = torch.stack([gz_i.flatten(), gh_i.flatten(), gw.flatten()])  # 3 × N
        flat_j = torch.stack([gz_j.flatten(), gh_j.flatten(), gw.flatten()])  # 3 × N
        N = flat_i.shape[1]

        # Relative coords: (3, N, N)
        coords = flat_i[:, :, None] - flat_j[:, None, :]

        # Shift so all coords are non-negative
        coords[2] += Ww - 1  # lon: [-(Ww-1), Ww-1]
        coords[1] *= 2 * Ww - 1  # lat contribution
        coords[0] *= (2 * Ww - 1) * Wh * Wh  # pressure contribution

        position_index = coords.sum(dim=0)  # N × N
        return position_index.view(-1)  # N*N

    def forward(self, attn: torch.Tensor, window_type_index: int) -> torch.Tensor:
        """Add Earth-specific bias to attention logits.

        Args:
            attn: ``(B_w, n_heads, N, N)`` attention logits where N = Wz*Wh*Ww.
            window_type_index: Index of the current window type.

        Returns:
            attn with bias added, same shape.
        """
        Wz, Wh, Ww = self.window_size
        N = Wz * Wh * Ww
        bias = self.earth_specific_bias[self.position_index, window_type_index, :]  # N*N × heads
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0)  # 1 × heads × N × N
        return attn + bias


# ---------------------------------------------------------------------------
# 3-D Window Attention
# ---------------------------------------------------------------------------


class EarthAttention3D(nn.Module):
    """3-D shifted window self-attention with Earth-Specific position bias.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        window_size: (Wz, Wh, Ww).
        type_of_windows: Number of distinct window types.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        type_of_windows: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.earth_bias = EarthSpecificBias(window_size, num_heads, type_of_windows)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        window_type_index: int,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B_w, N, C)`` — flattened windows.
            mask: ``(B_w, 1, N, N)`` or ``None``.
            window_type_index: Index of window type for Earth-specific bias.
        """
        Bw, N, C = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(Bw, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (Bw, H, N, head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (Bw, H, N, N)
        attn = self.earth_bias(attn, window_type_index)

        if mask is not None:
            # mask: (n_windows, 1, N, N); attn: (B*n_windows, n_heads, N, N)
            n_w = mask.shape[0]
            if Bw != n_w:
                B = Bw // n_w
                mask = mask.repeat(B, 1, 1, 1)
            attn = attn + mask  # mask is -∞ for non-adjacent pairs

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# Earth-Specific Block
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _build_shift_mask(
    Z: int,
    H: int,
    W: int,
    window_size: Tuple[int, int, int],
    shift: Tuple[int, int, int],
    device: str,
) -> torch.Tensor:
    """Build attention mask for shifted-window attention. Cached by (Z,H,W,ws,shift,device)."""
    Wz, Wh, Ww = window_size
    sz, sh, sw = shift
    img_mask = torch.zeros(1, Z, H, W, 1, device=device)
    z_slices = (slice(0, -Wz), slice(-Wz, -sz), slice(-sz, None))
    h_slices = (slice(0, -Wh), slice(-Wh, -sh), slice(-sh, None))
    w_slices = (slice(0, -Ww), slice(-Ww, -sw), slice(-sw, None))
    cnt = 0
    for z in z_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, z, h, w, :] = cnt
                cnt += 1
    # Window partition: (n_windows, Wz*Wh*Ww)
    Bw = (Z // Wz) * (H // Wh) * (W // Ww)
    mask_windows = _window_partition_5d(img_mask, window_size)  # (Bw, N, 1)
    mask_windows = mask_windows.squeeze(-1)  # (Bw, N)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (Bw, N, N)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask.unsqueeze(1)  # (Bw, 1, N, N)


def _window_partition_5d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """Partition (B, Z, H, W, C) into windows (n_windows * B, Wz*Wh*Ww, C)."""
    B, Z, H, W, C = x.shape
    Wz, Wh, Ww = window_size
    x = x.view(B, Z // Wz, Wz, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, Wz * Wh * Ww, C)
    return windows


def _window_reverse_5d(
    windows: torch.Tensor,
    window_size: Tuple[int, int, int],
    B: int,
    Z: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Reverse of ``_window_partition_5d``."""
    Wz, Wh, Ww = window_size
    x = windows.view(B, Z // Wz, H // Wh, W // Ww, Wz, Wh, Ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, Z, H, W, -1)


class EarthSpecificBlock(nn.Module):
    """3-D window attention block with optional shift and Earth-Specific bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 6, 12),
        type_of_windows: int = 1,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        Wz, Wh, Ww = window_size
        self.shift_size = (Wz // 2, Wh // 2, Ww // 2) if shift else (0, 0, 0)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = EarthAttention3D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            type_of_windows=type_of_windows,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        Z: int,
        H: int,
        W: int,
        window_type_index: int,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, Z*H*W, C)`` sequence.
            Z, H, W: Spatial dimensions.
        """
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Z, H, W, C)

        # Pad to window multiples
        Wz, Wh, Ww = self.window_size
        pad_z = (Wz - Z % Wz) % Wz
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        if pad_z or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_z))
        _, Zp, Hp, Wp, _ = x.shape

        sz, sh, sw = self.shift_size
        if self.shift:
            x = torch.roll(x, shifts=(-sz, -sh, -sw), dims=(1, 2, 3))
            mask = _build_shift_mask(Zp, Hp, Wp, self.window_size, self.shift_size, str(x.device))
        else:
            mask = None

        # Count window type index based on position in (Z//Wz, H//Wh) grid
        windows = _window_partition_5d(x, self.window_size)  # (Bw, N, C)
        windows = self.attn(windows, mask, window_type_index)
        x = _window_reverse_5d(windows, self.window_size, B, Zp, Hp, Wp)

        if self.shift:
            x = torch.roll(x, shifts=(sz, sh, sw), dims=(1, 2, 3))

        # Crop padding
        if pad_z or pad_h or pad_w:
            x = x[:, :Z, :H, :W, :].contiguous()

        x = x.view(B, Z * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# EarthSpecificLayer
# ---------------------------------------------------------------------------


class EarthSpecificLayer(nn.Module):
    """Stack of EarthSpecificBlock layers with alternating shift."""

    def __init__(
        self,
        depth: int,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 6, 12),
        type_of_windows: int = 1,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | List[float] = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth
        self.blocks = nn.ModuleList(
            [
                EarthSpecificBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    type_of_windows=type_of_windows,
                    shift=(i % 2 == 1),
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, Z: int, H: int, W: int) -> torch.Tensor:
        """``x`` is ``(B, Z*H*W, C)``."""
        Wz, Wh = self.blocks[0].window_size[:2]
        # All windows in the same layer share the same type index (0) here;
        # Earth-specific bias is per (window_type, head) so this is a simplification
        # for non-0.25-deg grids.  For the 0.25-deg model, type_of_windows = (Z//Wz)*(H//Wh).
        for i, block in enumerate(self.blocks):
            # Vary window_type_index per spatial window: use 0 as a base (correct at 0.25 deg
            # when type_of_windows=1, approximate otherwise)
            x = block(x, Z, H, W, window_type_index=0)
        return x


# ---------------------------------------------------------------------------
# Down / Up Sampling
# ---------------------------------------------------------------------------


class DownSample(nn.Module):
    """Halve H and W, double channels."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, Z: int, H: int, W: int) -> Tuple[torch.Tensor, int, int, int]:
        B = x.shape[0]
        x = x.view(B, Z, H, W, -1)
        # Pad if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, _, H2, W2, C = x.shape
        # Rearrange: (B, Z, H//2, 2, W//2, 2, C) → (B, Z*(H//2)*(W//2), 4C)
        x = x.view(B, Z, H2 // 2, 2, W2 // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, Z * (H2 // 2) * (W2 // 2), 4 * C)
        x = self.norm(x)
        x = self.linear(x)
        return x, Z, H2 // 2, W2 // 2


class UpSample(nn.Module):
    """Double H and W, halve channels."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim * 4, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        Z: int,
        H: int,
        W: int,
        target_H: int,
        target_W: int,
    ) -> Tuple[torch.Tensor, int, int, int]:
        B = x.shape[0]
        x = self.linear1(x)
        # Rearrange: (B, Z, H, W, 4C) → (B, Z, H*2, W*2, C)
        x = x.view(B, Z, H, W, 2, 2, -1)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, Z * H * 2 * W * 2, -1)
        # Crop to target shape
        H2, W2 = H * 2, W * 2
        if H2 != target_H or W2 != target_W:
            x = x.view(B, Z, H2, W2, -1)[:, :, :target_H, :target_W, :].contiguous()
            x = x.view(B, Z * target_H * target_W, -1)
            H2, W2 = target_H, target_W
        x = self.norm(x)
        x = self.linear2(x)
        return x, Z, H2, W2


# ---------------------------------------------------------------------------
# Patch Embedding and Recovery
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """Embed upper-air and surface fields into a joint sequence of tokens.

    Upper air:  Conv3d with kernel (patch_size_z, patch_size_h, patch_size_w)
    Surface:    Conv2d with kernel (patch_size_h, patch_size_w)
                Surface gets n_static constant maps concatenated before the conv.

    Args:
        n_atmos_vars: Number of upper-air variables.
        n_surf_vars: Number of surface variables (excl. static constants).
        n_static: Number of static constant maps concatenated to surface input.
        n_levels: Number of pressure levels.
        patch_size: (Pz, Ph, Pw) — patch size.
        dim: Embedding dimension.
    """

    def __init__(
        self,
        n_atmos_vars: int,
        n_surf_vars: int,
        n_static: int,
        n_levels: int,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        dim: int = 192,
    ) -> None:
        super().__init__()
        Pz, Ph, Pw = patch_size
        self.patch_size = patch_size
        self.n_levels = n_levels
        self.Pz = Pz

        # Upper-air: (B, n_atmos_vars, n_levels, H, W) — treat as (B, C_atm, Z, H, W)
        self.conv_atmos = nn.Conv3d(n_atmos_vars, dim, kernel_size=patch_size, stride=patch_size)

        # Surface: (B, n_surf_vars + n_static, H, W)
        self.conv_surf = nn.Conv2d(n_surf_vars + n_static, dim, kernel_size=(Ph, Pw), stride=(Ph, Pw))
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        upper: torch.Tensor,
        surface: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int, int]:
        """
        Args:
            upper:   ``(B, n_atmos_vars, n_levels, H, W)``
            surface: ``(B, n_surf_vars + n_static, H, W)``

        Returns:
            x:   ``(B, (Zp+1)*Hp*Wp, dim)`` sequence
            Zp:  number of pressure patches + 1 (surface occupies level 0)
            Hp:  number of height patches
            Wp:  number of width patches
        """
        B, _, Pl, H, W = upper.shape
        Pz, Ph, Pw = self.patch_size

        # Pad upper air so Pl is a multiple of Pz
        pad_z = (Pz - Pl % Pz) % Pz
        pad_h = (Ph - H % Ph) % Ph
        pad_w = (Pw - W % Pw) % Pw
        if pad_z or pad_h or pad_w:
            upper = F.pad(upper, (0, pad_w, 0, pad_h, 0, pad_z))
        if pad_h or pad_w:
            surface = F.pad(surface, (0, pad_w, 0, pad_h))

        x_atm = self.conv_atmos(upper)  # (B, dim, Zp, Hp, Wp)
        x_surf = self.conv_surf(surface)  # (B, dim, Hp, Wp)

        _, dim, Zp, Hp, Wp = x_atm.shape

        # Concatenate surface as an extra "pressure level" (level 0)
        x_surf = x_surf.unsqueeze(2)  # (B, dim, 1, Hp, Wp)
        x = torch.cat([x_surf, x_atm], dim=2)  # (B, dim, Zp+1, Hp, Wp)

        # Reshape to sequence
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, Zp+1, Hp, Wp, dim)
        Z = Zp + 1
        x = x.view(B, Z * Hp * Wp, dim)
        x = self.norm(x)
        return x, Z, Hp, Wp


class PatchRecovery(nn.Module):
    """Recover upper-air and surface fields from the token sequence.

    Args:
        n_atmos_vars: Number of upper-air variables to predict.
        n_surf_vars: Number of surface variables to predict.
        patch_size: (Pz, Ph, Pw).
        dim: Embedding dimension (after skip-concat → 2×embed_dim).
        n_levels: Number of pressure levels in the original input.
    """

    def __init__(
        self,
        n_atmos_vars: int,
        n_surf_vars: int,
        patch_size: Tuple[int, int, int],
        dim: int,
        n_levels: int,
    ) -> None:
        super().__init__()
        Pz, Ph, Pw = patch_size
        self.patch_size = patch_size
        self.n_levels = n_levels
        self.Pz = Pz

        self.conv_atmos = nn.ConvTranspose3d(dim, n_atmos_vars, kernel_size=patch_size, stride=patch_size)
        self.conv_surf = nn.ConvTranspose2d(dim, n_surf_vars, kernel_size=(Ph, Pw), stride=(Ph, Pw))

    def forward(
        self,
        x: torch.Tensor,
        Z: int,
        H: int,
        W: int,
        orig_H: int,
        orig_W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B, Z*H*W, dim)`` sequence.
            Z, H, W: Patch-space dimensions.
            orig_H, orig_W: Original spatial dimensions (before patch embedding).

        Returns:
            upper:   ``(B, n_atmos_vars, n_levels, orig_H, orig_W)``
            surface: ``(B, n_surf_vars, orig_H, orig_W)``
        """
        B = x.shape[0]
        C = x.shape[-1]
        Pz, Ph, Pw = self.patch_size

        x = x.view(B, Z, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, Z, H, W)

        # Surface: level 0
        surf_tokens = x[:, :, 0, :, :]  # (B, C, H, W)
        surface = self.conv_surf(surf_tokens)

        # Upper air: levels 1..Z
        atm_tokens = x[:, :, 1:, :, :]  # (B, C, Zp, H, W)
        upper = self.conv_atmos(atm_tokens)

        # Crop to original spatial dimensions
        upper = upper[:, :, : self.n_levels, :orig_H, :orig_W]
        surface = surface[:, :, :orig_H, :orig_W]
        return upper, surface


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class PanguModel(nn.Module):
    """Pangu-Weather 3D Earth Transformer (trainable from scratch).

    Input:
        upper:   ``(B, n_atmos_vars, n_levels, H, W)``
        surface: ``(B, n_surf_vars + n_static_vars, H, W)``

    Output:
        upper_pred:   ``(B, n_atmos_vars, n_levels, H, W)``
        surface_pred: ``(B, n_surf_vars, H, W)``

    Args:
        n_atmos_vars: Number of 3-D atmospheric variables.
        n_surf_vars: Number of surface variables.
        n_static: Number of static constant maps concatenated to surface input.
        n_levels: Number of pressure levels.
        embed_dim: Base embedding dimension (192 in the paper).
        patch_size: 3-D patch size (default (2,4,4) as in the paper).
        window_size: Swin window size (default (2,6,12) as in the paper).
        depths: Tuple of (enc1_depth, enc2_depth, dec1_depth, dec2_depth).
        num_heads: Tuple of (enc1_heads, enc2_heads, dec1_heads, dec2_heads).
        drop_path_rate: Maximum stochastic depth rate.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        mlp_ratio: MLP hidden dim ratio.
    """

    def __init__(
        self,
        n_atmos_vars: int = 5,
        n_surf_vars: int = 4,
        n_static: int = 3,
        n_levels: int = 13,
        embed_dim: int = 192,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        window_size: Tuple[int, int, int] = (2, 6, 12),
        depths: Tuple[int, int, int, int] = (2, 6, 6, 2),
        num_heads: Tuple[int, int, int, int] = (6, 12, 12, 6),
        drop_path_rate: float = 0.2,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.n_levels = n_levels
        self.n_surf_vars = n_surf_vars

        # Linearly space drop-path rates
        total_depth = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_depth).tolist()
        d0, d1, d2, d3 = depths

        common = dict(
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )

        self.patch_embed = PatchEmbedding(
            n_atmos_vars=n_atmos_vars,
            n_surf_vars=n_surf_vars,
            n_static=n_static,
            n_levels=n_levels,
            patch_size=patch_size,
            dim=embed_dim,
        )

        self.layer1 = EarthSpecificLayer(d0, embed_dim, num_heads[0], drop_path=dp_rates[:d0], **common)
        self.downsample = DownSample(embed_dim)
        self.layer2 = EarthSpecificLayer(d1, embed_dim * 2, num_heads[1], drop_path=dp_rates[d0 : d0 + d1], **common)

        self.layer3 = EarthSpecificLayer(
            d2, embed_dim * 2, num_heads[2], drop_path=dp_rates[d0 + d1 : d0 + d1 + d2], **common
        )
        self.upsample = UpSample(embed_dim * 2, embed_dim)
        self.layer4 = EarthSpecificLayer(d3, embed_dim, num_heads[3], drop_path=dp_rates[d0 + d1 + d2 :], **common)

        self.patch_recovery = PatchRecovery(
            n_atmos_vars=n_atmos_vars,
            n_surf_vars=n_surf_vars,
            patch_size=patch_size,
            dim=embed_dim * 2,  # after skip-concat
            n_levels=n_levels,
        )

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
        upper: torch.Tensor,
        surface: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one forecasting step.

        Args:
            upper:   ``(B, n_atmos_vars, n_levels, H, W)``
            surface: ``(B, n_surf_vars + n_static_vars, H, W)``

        Returns:
            Predicted (upper, surface) tensors with the same shape as inputs
            except surface has ``n_surf_vars`` channels (static vars removed).
        """
        _, _, _, orig_H, orig_W = upper.shape

        # Patch embedding
        x, Z, H, W = self.patch_embed(upper, surface)

        # Encoder stage 1
        x = self.layer1(x, Z, H, W)
        skip = x

        # Downsample
        x, Z2, H2, W2 = self.downsample(x, Z, H, W)

        # Encoder stage 2
        x = self.layer2(x, Z2, H2, W2)

        # Decoder stage 1
        x = self.layer3(x, Z2, H2, W2)

        # Upsample back to (Z, H, W)
        x, Z3, H3, W3 = self.upsample(x, Z2, H2, W2, target_H=H, target_W=W)

        # Decoder stage 2
        x = self.layer4(x, Z3, H3, W3)

        # Skip connection: cat along channel dim
        x = torch.cat([skip, x], dim=-1)  # (B, Z*H*W, 2*embed_dim)

        # Patch recovery
        upper_pred, surface_pred = self.patch_recovery(x, Z3, H3, W3, orig_H, orig_W)
        return upper_pred, surface_pred


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITPangu(nn.Module):
    """Pangu-Weather adapted to CREDIT's flat ``(B, C, H, W)`` tensor convention.

    Channel layout (input tensor)::

        [surf_var_0, ..., surf_var_N,          ← n_surf_vars channels
         atmos_var_0_lev_0, ..., atmos_var_0_lev_P,
         ...
         atmos_var_M_lev_0, ..., atmos_var_M_lev_P,  ← n_atmos_vars * n_levels channels
         static_var_0, ..., static_var_S]      ← n_static_vars channels

    Output tensor::

        [surf_var_0, ..., surf_var_N,
         atmos_var_0_lev_0, ..., atmos_var_M_lev_P]

    Args:
        surf_vars: List of surface variable names (for documentation).
        atmos_vars: List of atmospheric variable names.
        static_vars: List of static variable names.
        atmos_levels: List of pressure levels in hPa.
        **pangu_kwargs: Forwarded to :class:`.PanguModel`.
    """

    def __init__(
        self,
        surf_vars: List[str] = ("2t", "10u", "10v", "msl"),
        atmos_vars: List[str] = ("z", "u", "v", "t", "q"),
        static_vars: List[str] = ("lsm", "z", "slt"),
        atmos_levels: List[int] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        **pangu_kwargs,
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

        self.model = PanguModel(
            n_atmos_vars=n_atmos,
            n_surf_vars=n_surf,
            n_static=n_static,
            n_levels=n_levels,
            **pangu_kwargs,
        )

    @property
    def n_input_channels(self) -> int:
        return self.n_surf + self.n_atmos * self.n_levels + self.n_static

    @property
    def n_output_channels(self) -> int:
        return self.n_surf + self.n_atmos * self.n_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, H, W)`` following the channel layout above.

        Returns:
            ``(B, C_out, H, W)`` with surf + atmos variables.
        """
        B, C, H, W = x.shape
        offset = 0
        # Surface
        surf = x[:, offset : offset + self.n_surf, :, :]
        offset += self.n_surf
        # Atmospheric: flatten to (B, n_atmos, n_levels, H, W)
        atm_flat = x[:, offset : offset + self.n_atmos * self.n_levels, :, :]
        atmos = atm_flat.view(B, self.n_atmos, self.n_levels, H, W)
        offset += self.n_atmos * self.n_levels
        # Static
        static = x[:, offset : offset + self.n_static, :, :]
        # Cat static to surface
        surface_in = torch.cat([surf, static], dim=1)  # (B, n_surf+n_static, H, W)

        upper_pred, surf_pred = self.model(atmos, surface_in)

        # Reassemble output: (B, n_atmos*n_levels, H, W)
        atm_out = upper_pred.view(B, self.n_atmos * self.n_levels, H, W)
        return torch.cat([surf_pred, atm_out], dim=1)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Pangu smoke test on {device}")

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

    model = CREDITPangu(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        atmos_levels=levels,
        embed_dim=64,
        patch_size=(1, 4, 4),
        window_size=(1, 4, 8),
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 4, 2),
        drop_path_rate=0.1,
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
    print("Pangu smoke test PASSED")
