"""
SwinRNN — Hierarchical Swin Transformer with recurrent residual prediction.
Chen et al., 2023.  https://arxiv.org/abs/2307.09650
Architecture from PhysicsNemo / NVIDIA (Apache 2.0).

Architecture
------------
Encoder: 3-stage Swin hierarchy (patch embed → 3× Swin stages with downsampling)
Decoder: symmetric 3-stage upsampling with skip connections
Head:    linear projection to output channels

The "RNN" label refers to the temporal residual formulation used during
multi-step rollout (predicted Δstate added to input state), not a literal RNN.
Within a single step it is a pure encoder-decoder transformer.

CREDITSwinRNN wraps the model for CREDIT's flat (B, C, H, W) tensors.
No external dependencies beyond PyTorch + einops.
"""

import os
import sys

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Window attention helpers
# ---------------------------------------------------------------------------


def window_partition(x, window_size):
    """(B, H, W, C) → (num_windows*B, wh, ww, C)"""
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, wh, ww, C) → (B, H, W, C)"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.shift = shift

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position bias table
        self.rel_pos_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        )  # (2, ws, ws)
        coords_flat = coords.flatten(1)  # (2, ws^2)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws^2, ws^2)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_index", rel.sum(-1))  # (ws^2, ws^2)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # relative position bias
        bias = self.rel_pos_bias[self.rel_pos_index.reshape(-1)].reshape(N, N, -1)
        attn = attn + bias.permute(2, 0, 1)[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.reshape(B_, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    def __init__(
        self, dim, input_resolution, num_heads, window_size=7, shift=False, mlp_ratio=4.0, drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        H, W = input_resolution
        self.H, self.W = H, W
        self.ws = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, shift=shift, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

        # shifted-window attention mask
        if shift and self.shift_size > 0:
            img_mask = torch.zeros(1, H, W, 1)
            slices_h = (slice(0, -window_size), slice(-window_size, -self.shift_size), slice(-self.shift_size, None))
            slices_w = (slice(0, -window_size), slice(-window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for sh in slices_h:
                for sw in slices_w:
                    img_mask[:, sh, sw, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size).squeeze(-1)  # (nW, ws, ws)
            mask_windows = mask_windows.reshape(-1, window_size * window_size)
            attn_mask = mask_windows[:, :, None] - mask_windows[:, None, :]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B, self.H, self.W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_win = window_partition(x, self.ws).reshape(-1, self.ws * self.ws, C)
        shortcut = x.reshape(B, self.H * self.W, C)
        x_win = self.attn(self.norm1(x_win), mask=self.attn_mask)
        x = window_reverse(x_win.reshape(-1, self.ws, self.ws, C), self.ws, self.H, self.W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.reshape(B, self.H * self.W, C) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """2× spatial downsampling via patch merging."""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.H, self.W = input_resolution
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        x = x.reshape(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(torch.cat([x0, x1, x2, x3], dim=-1))).reshape(B, -1, 2 * C)


class PatchExpand(nn.Module):
    """2× spatial upsampling via patch expansion."""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.H, self.W = input_resolution
        self.expand = nn.Linear(dim, 4 * dim // 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        x = self.expand(x)  # (B, H*W, 2C)
        x = x.reshape(B, H, W, -1)
        # pixel-shuffle 2×
        x = rearrange(x, "b h w (c p1 p2) -> b (h p1) (w p2) c", p1=2, p2=2)
        return self.norm(x.reshape(B, H * 2 * W * 2, -1))


class SwinStage(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim,
                    input_resolution,
                    num_heads,
                    window_size=window_size,
                    shift=(i % 2 == 1),
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# SwinRNN encoder-decoder
# ---------------------------------------------------------------------------


class SwinRNNModel(nn.Module):
    """
    Swin Transformer encoder-decoder for weather forecasting.

    3-stage hierarchy:
      Stage 0: (H,  W ) @ dim
      Stage 1: (H/2,W/2) @ dim*2
      Stage 2: (H/4,W/4) @ dim*4

    Skip connections from encoder stages to decoder stages.

    Parameters
    ----------
    img_size : tuple[int,int]
    patch_size : int
        Initial patch embedding size.
    in_channels, out_channels : int
    embed_dim : int
        Base embedding dim at stage 0.
    depths : tuple[int,...]
        Number of Swin blocks at each of the 3 encoder stages.
    num_heads : tuple[int,...]
        Attention heads at each stage.
    window_size : int
    mlp_ratio : float
    drop_rate, attn_drop_rate : float
    """

    def __init__(
        self,
        img_size=(128, 256),
        patch_size=4,
        in_channels=70,
        out_channels=69,
        embed_dim=96,
        depths=(2, 2, 6),
        num_heads=(3, 6, 12),
        window_size=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0
        Ph, Pw = H // patch_size, W // patch_size

        # patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm0 = nn.LayerNorm(embed_dim)

        dims = [embed_dim, embed_dim * 2, embed_dim * 4]
        resolutions = [(Ph, Pw), (Ph // 2, Pw // 2), (Ph // 4, Pw // 4)]

        # ── Encoder ──────────────────────────────────────────────
        self.enc0 = SwinStage(
            dims[0],
            resolutions[0],
            depths[0],
            num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )
        self.down0 = PatchMerging(resolutions[0], dims[0])

        self.enc1 = SwinStage(
            dims[1],
            resolutions[1],
            depths[1],
            num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )
        self.down1 = PatchMerging(resolutions[1], dims[1])

        self.enc2 = SwinStage(
            dims[2],
            resolutions[2],
            depths[2],
            num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )

        # ── Decoder ──────────────────────────────────────────────
        self.up1 = PatchExpand(resolutions[2], dims[2])
        # after expand: dim = dims[2]//2 = dims[1]; concat with skip → dims[1]*2
        self.dec1_norm = nn.LayerNorm(dims[1] * 2)
        self.dec1_proj = nn.Linear(dims[1] * 2, dims[1])
        self.dec1 = SwinStage(
            dims[1],
            resolutions[1],
            depths[1],
            num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )

        self.up0 = PatchExpand(resolutions[1], dims[1])
        self.dec0_norm = nn.LayerNorm(dims[0] * 2)
        self.dec0_proj = nn.Linear(dims[0] * 2, dims[0])
        self.dec0 = SwinStage(
            dims[0],
            resolutions[0],
            depths[0],
            num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )

        # patch recovery (expand back to original resolution)
        self.final_expand = nn.Linear(dims[0], out_channels * patch_size * patch_size)
        self.patch_size = patch_size
        self.Ph, self.Pw = Ph, Pw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # patch embed
        x = self.patch_embed(x)  # (B, D, Ph, Pw)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.norm0(x)

        # encoder
        e0 = self.enc0(x)  # (B, Ph*Pw, D0)
        e1 = self.enc1(self.down0(e0))  # (B, Ph/2*Pw/2, D1)
        e2 = self.enc2(self.down1(e1))  # (B, Ph/4*Pw/4, D2)

        # decoder with skips
        d1 = self.up1(e2)  # (B, Ph/2*Pw/2, D1)
        d1 = self.dec1_proj(self.dec1_norm(torch.cat([d1, e1], dim=-1)))
        d1 = self.dec1(d1)

        d0 = self.up0(d1)  # (B, Ph*Pw, D0)
        d0 = self.dec0_proj(self.dec0_norm(torch.cat([d0, e0], dim=-1)))
        d0 = self.dec0(d0)

        # recover spatial resolution
        out = self.final_expand(d0)  # (B, Ph*Pw, C_out*p^2)
        out = rearrange(
            out,
            "b (ph pw) (c p1 p2) -> b c (ph p1) (pw p2)",
            ph=self.Ph,
            pw=self.Pw,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return out


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITSwinRNN(nn.Module):
    """CREDIT wrapper for SwinRNN.  Flat (B, C, H, W) I/O."""

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(128, 256),
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 6),
        num_heads=(3, 6, 12),
        window_size=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__()
        self.model = SwinRNNModel(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
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

    B, C_in, C_out = 1, 70, 69
    H, W = 64, 128  # small for smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITSwinRNN(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=4,
        embed_dim=32,
        depths=(2, 2, 2),
        num_heads=(2, 4, 4),
        window_size=4,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected shape {y.shape}"
    y.mean().backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITSwinRNN OK — output {y.shape}, params {n_params:.1f}M, device {device}")
