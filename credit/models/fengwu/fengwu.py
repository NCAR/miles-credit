"""
FengWu — Multi-Modal Weather Forecasting.
Chen et al., 2023.  https://arxiv.org/abs/2304.02948

Faithful implementation based on:
  - The FengWu paper (Swin Transformer encoders / cross-modal fuser / decoders)
  - FengWu-Adas (OpenEarthLab/FengWu-Adas) Swin primitives

Architecture:
  G per-modality Swin Transformer encoders (2D patch embed + Swin blocks)
  → cross-modal fuser (each group attends to all others)
  → G per-modality Swin Transformer decoders + pixel-shuffle head

No official PyTorch weights exist (official FengWu inference weights are ONNX-only).
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Swin helpers
# ---------------------------------------------------------------------------


def _pad_to_multiple(x: torch.Tensor, window_size: int):
    """Pad H and W to multiples of window_size; return (padded_x, pad_H, pad_W)."""
    _, H, W, _ = x.shape
    pad_H = (window_size - H % window_size) % window_size
    pad_W = (window_size - W % window_size) % window_size
    if pad_H > 0 or pad_W > 0:
        x = F.pad(x, (0, 0, 0, pad_W, 0, pad_H))
    return x, pad_H, pad_W


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) → (B*nW, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws, ws, C) → (B, H, W, C)."""
    B = int(windows.shape[0] / (H // ws * W // ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _build_shift_mask(H: int, W: int, ws: int, shift: int, device) -> torch.Tensor:
    """Attention mask for shifted-window attention (SW-MSA)."""
    img_mask = torch.zeros((1, H, W, 1), device=device)
    slices_h = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    slices_w = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    cnt = 0
    for sh in slices_h:
        for sw in slices_w:
            img_mask[:, sh, sw, :] = cnt
            cnt += 1
    mask_windows = _window_partition(img_mask, ws).view(-1, ws * ws)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class RelativePositionBias(nn.Module):
    """Learnable relative position bias table for Swin attention."""

    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        self.ws = window_size
        self.table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).flatten(1)
        rel = grid[:, :, None] - grid[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("index", rel.sum(-1).view(-1))

    def forward(self) -> torch.Tensor:
        N = self.ws * self.ws
        return self.table[self.index].view(N, N, -1).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Swin block
# ---------------------------------------------------------------------------


class SwinMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class SwinBlock(nn.Module):
    """
    One Swin Transformer block — W-MSA or SW-MSA.

    Parameters
    ----------
    dim : int
    num_heads : int
    window_size : int
    shift : bool  — True → shifted-window (SW-MSA), False → standard (W-MSA)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.ws = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_bias = RelativePositionBias(window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(dim, mlp_ratio=mlp_ratio, drop=drop)

        self._mask_cache: dict = {}

    def _get_mask(self, H: int, W: int, device) -> torch.Tensor | None:
        if not self.shift:
            return None
        key = (H, W, device.type)
        if key not in self._mask_cache:
            self._mask_cache[key] = _build_shift_mask(H, W, self.ws, self.shift_size, device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # pad to window multiple
        x, pH, pW = _pad_to_multiple(x, self.ws)
        Hp, Wp = H + pH, W + pW

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # window partition
        x_win = _window_partition(x, self.ws)  # (B*nW, ws, ws, C)
        x_win = x_win.view(-1, self.ws * self.ws, C)  # (B*nW, N, C)

        # attention
        Bw, N, _ = x_win.shape
        qkv = self.qkv(x_win).reshape(Bw, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias().unsqueeze(0)  # (1, heads, N, N)

        mask = self._get_mask(Hp, Wp, x.device)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x_win = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        x_win = self.proj(x_win)

        # window reverse
        x_win = x_win.view(-1, self.ws, self.ws, C)
        x = _window_reverse(x_win, self.ws, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # unpad
        x = x[:, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Cross-modal fuser block (unchanged from paper description)
# ---------------------------------------------------------------------------


class CrossAttnBlock(nn.Module):
    """Query from x_q; keys + values from x_kv."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim), nn.Dropout(drop)
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B, Nq, C = x_q.shape
        Nkv = x_kv.shape[1]
        q = self.q(self.norm_q(x_q)).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(self.norm_kv(x_kv)).reshape(B, Nkv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x_q = x_q + self.proj(out)
        x_q = x_q + self.mlp(self.norm2(x_q))
        return x_q


# ---------------------------------------------------------------------------
# Per-group Swin encoder / decoder
# ---------------------------------------------------------------------------


class SwinEncoder(nn.Module):
    """Patch embed + Swin blocks for one variable group."""

    def __init__(
        self,
        n_ch: int,
        n_h: int,
        n_w: int,
        embed_dim: int,
        patch_size: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(n_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_h * n_w, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                SwinBlock(embed_dim, num_heads, window_size, shift=(i % 2 == 1), mlp_ratio=mlp_ratio, drop=drop)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.n_h = n_h
        self.n_w = n_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_ch, H, W)
        x = self.patch_embed(x)  # (B, D, n_h, n_w)
        B, D, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, n_h, n_w, D)
        tokens = x.view(B, h * w, D) + self.pos_embed
        x = tokens.view(B, h, w, D)
        for blk in self.blocks:
            x = blk(x)
        x = x.view(B, h * w, D)
        return self.norm(x)  # (B, n_patches, D)


class SwinDecoder(nn.Module):
    """Swin blocks + pixel-unshuffle head for one variable group."""

    def __init__(
        self,
        out_ch: int,
        n_h: int,
        n_w: int,
        embed_dim: int,
        patch_size: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.n_h = n_h
        self.n_w = n_w
        self.patch_size = patch_size
        self.blocks = nn.ModuleList(
            [
                SwinBlock(embed_dim, num_heads, window_size, shift=(i % 2 == 1), mlp_ratio=mlp_ratio, drop=drop)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_ch * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, n_patches, D)
        B, N, D = tokens.shape
        x = tokens.view(B, self.n_h, self.n_w, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(self.norm(x))  # (B, n_h, n_w, out_ch*p^2)
        x = rearrange(
            x,
            "b nh nw (c p1 p2) -> b c (nh p1) (nw p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x


# ---------------------------------------------------------------------------
# FengWu core model
# ---------------------------------------------------------------------------


class FengWuModel(nn.Module):
    """
    FengWu weather transformer.

    Parameters
    ----------
    img_size : tuple[int, int]
        Spatial dimensions after optional padding (must be divisible by patch_size).
    patch_size : int
    in_channels : int
    out_channels : int
    group_sizes : list[int]
        Input channels per group; must sum to in_channels.
    embed_dim : int
    encoder_depth : int
    fuser_depth : int
    decoder_depth : int
    num_heads : int
    window_size : int
        Swin window size (applied to the patch-grid, not pixel-grid).
    mlp_ratio : float
    drop_rate : float
    """

    def __init__(
        self,
        img_size: tuple = (184, 360),
        patch_size: int = 4,
        in_channels: int = 70,
        out_channels: int = 69,
        group_sizes: list = None,
        embed_dim: int = 192,
        encoder_depth: int = 8,
        fuser_depth: int = 4,
        decoder_depth: int = 8,
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        )
        n_h = H // patch_size
        n_w = W // patch_size

        if group_sizes is None:
            group_sizes = [in_channels]
        assert sum(group_sizes) == in_channels, f"group_sizes {group_sizes} must sum to in_channels {in_channels}"

        out_group_sizes = list(group_sizes)
        diff = sum(out_group_sizes) - out_channels
        if diff > 0:
            out_group_sizes[-1] -= diff
        elif diff < 0:
            out_group_sizes[-1] += abs(diff)

        self.group_sizes = group_sizes
        self.out_group_sizes = out_group_sizes
        n_groups = len(group_sizes)

        self.encoders = nn.ModuleList(
            [
                SwinEncoder(
                    gs, n_h, n_w, embed_dim, patch_size, encoder_depth, num_heads, window_size, mlp_ratio, drop_rate
                )
                for gs in group_sizes
            ]
        )

        # cross-modal fuser: each group attends to all others
        if n_groups > 1:
            self.fuser_cross = nn.ModuleList(
                [
                    nn.ModuleList([CrossAttnBlock(embed_dim, num_heads, drop=drop_rate) for _ in range(n_groups)])
                    for _ in range(fuser_depth)
                ]
            )
        else:
            self.fuser_cross = None

        self.decoders = nn.ModuleList(
            [
                SwinDecoder(
                    ogs, n_h, n_w, embed_dim, patch_size, decoder_depth, num_heads, window_size, mlp_ratio, drop_rate
                )
                for ogs in out_group_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = x.split(self.group_sizes, dim=1)
        tokens = [enc(g) for enc, g in zip(self.encoders, groups)]

        if self.fuser_cross is not None:
            for layer in self.fuser_cross:
                new_tokens = []
                for i, (t_q, cross_blk) in enumerate(zip(tokens, layer)):
                    others = torch.cat([tokens[j] for j in range(len(tokens)) if j != i], dim=1)
                    new_tokens.append(cross_blk(t_q, others))
                tokens = new_tokens

        outs = [dec(t) for dec, t in zip(self.decoders, tokens)]
        return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITFengWu(nn.Module):
    """CREDIT wrapper for FengWu.  Flat (B, C, H, W) I/O; returns (B, C_out, 1, H, W)."""

    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 69,
        img_size: tuple = (181, 360),
        frames: int = 1,
        patch_size: int = 4,
        window_size: int = 8,
        group_sizes: list = None,
        embed_dim: int = 192,
        encoder_depth: int = 8,
        fuser_depth: int = 4,
        decoder_depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W

        # pad so model sees a H×W divisible by patch_size * window_size
        align = patch_size * window_size
        pH = (align - H % align) % align
        pW = (align - W % align) % align
        self.pH, self.pW = pH, pW

        self.model = FengWuModel(
            img_size=(H + pH, W + pW),
            patch_size=patch_size,
            in_channels=in_channels * frames,
            out_channels=out_channels,
            group_sizes=group_sizes,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            fuser_depth=fuser_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)
        if self.pH > 0 or self.pW > 0:
            x = F.pad(x, (0, self.pW, 0, self.pH))
        out = self.model(x)
        if self.pH > 0 or self.pW > 0:
            out = out[:, :, : self.H, : self.W]
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
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, _root)

    B, C_in, C_out = 1, 12, 10
    H, W = 32, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITFengWu(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=4,
        window_size=4,
        group_sizes=[4, 4, 4],
        embed_dim=64,
        encoder_depth=2,
        fuser_depth=2,
        decoder_depth=2,
        num_heads=4,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, 1, H, W), f"unexpected shape {y.shape}"
    y.mean().backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITFengWu OK — output {y.shape}, params {n_params:.1f}M, device {device}")
