"""
FengWu — Multi-Modal and Multi-Task Weather Forecasting.
Chen et al., 2023.  https://arxiv.org/abs/2304.02948
Architecture from PhysicsNemo (Apache 2.0).

Key innovation: hierarchical ViT with a cross-variable fusion module.
Variables are grouped (e.g. Z-levels, T-levels, UV-levels, surface) and each
group gets its own ViT encoder/decoder.  A "fuser" transformer then lets
groups attend to each other before decoding.

Simplified CREDIT implementation:
  - Variable grouping: user-specified list of group sizes summing to in_channels.
    Default: 1 group (all channels together, degrades to a standard ViT).
  - Encoder: per-group ViT patch embed + transformer blocks.
  - Fuser: cross-group multi-head attention (all groups attend to all others).
  - Decoder: per-group ViT blocks + patch recovery.

CREDITFengWu wraps this for CREDIT's flat (B, C, H, W) tensors.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Shared ViT primitives
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
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


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = x + (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """Query from one sequence, keys+values from another."""

    def __init__(self, dim, num_heads, drop=0.0, attn_drop=0.0):
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
        self.mlp = MLP(dim, drop=drop)

    def forward(self, x_q, x_kv):
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
# Per-group encoder / decoder
# ---------------------------------------------------------------------------


class GroupEncoder(nn.Module):
    def __init__(self, n_ch, n_patches, embed_dim, patch_size, depth, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch_proj = nn.Conv2d(n_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [SelfAttnBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, n_ch, H, W)
        x = self.patch_proj(x)  # (B, D, nh, nw)
        x = rearrange(x, "b d h w -> b (h w) d") + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)  # (B, n_patches, D)


class GroupDecoder(nn.Module):
    def __init__(self, n_ch, n_patches, embed_dim, patch_size, depth, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.n_h = None  # set externally
        self.n_w = None
        self.patch_size = patch_size
        self.blocks = nn.ModuleList(
            [SelfAttnBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_ch * patch_size * patch_size)

    def forward(self, x, n_h, n_w):
        # x: (B, n_patches, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(self.norm(x))  # (B, n_patches, n_ch*p^2)
        x = rearrange(
            x,
            "b (nh nw) (c p1 p2) -> b c (nh p1) (nw p2)",
            nh=n_h,
            nw=n_w,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x


# ---------------------------------------------------------------------------
# FengWu core
# ---------------------------------------------------------------------------


class FengWuModel(nn.Module):
    """
    FengWu weather transformer.

    Parameters
    ----------
    img_size : tuple[int,int]
    patch_size : int
    in_channels : int
        Total input channels.
    out_channels : int
        Total output channels (should == sum of group output sizes).
    group_sizes : list[int]
        Number of input channels per group (must sum to in_channels).
        Output group sizes default to the same values (assumes symmetric).
    embed_dim : int
        Per-group embedding dimension.
    encoder_depth : int
        Swin/ViT blocks per group encoder.
    fuser_depth : int
        Cross-group fusion transformer layers.
    decoder_depth : int
        ViT blocks per group decoder.
    num_heads : int
    mlp_ratio : float
    drop_rate : float
    """

    def __init__(
        self,
        img_size=(128, 256),
        patch_size=4,
        in_channels=70,
        out_channels=69,
        group_sizes=None,
        embed_dim=256,
        encoder_depth=2,
        fuser_depth=4,
        decoder_depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        n_patches = self.n_h * self.n_w

        # default: single group (plain ViT)
        if group_sizes is None:
            group_sizes = [in_channels]
        assert sum(group_sizes) == in_channels, f"group_sizes {group_sizes} must sum to in_channels {in_channels}"
        # output groups same sizes as input (truncate last to match out_channels)
        out_group_sizes = list(group_sizes)
        diff = sum(out_group_sizes) - out_channels
        if diff > 0:
            # trim last group
            out_group_sizes[-1] -= diff
        elif diff < 0:
            out_group_sizes[-1] += abs(diff)

        self.group_sizes = group_sizes
        self.out_group_sizes = out_group_sizes
        n_groups = len(group_sizes)

        # per-group encoders
        self.encoders = nn.ModuleList(
            [
                GroupEncoder(
                    gs, n_patches, embed_dim, patch_size, encoder_depth, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate
                )
                for gs in group_sizes
            ]
        )

        # cross-group fuser: each group queries all other groups
        self.fuser = nn.ModuleList(
            [
                nn.ModuleList([CrossAttnBlock(embed_dim, num_heads, drop=drop_rate) for _ in range(n_groups - 1)])
                for _ in range(fuser_depth)
            ]
        )
        self.fuser_self = nn.ModuleList(
            [
                nn.ModuleList(
                    [SelfAttnBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate) for _ in range(n_groups)]
                )
                for _ in range(fuser_depth)
            ]
        )

        # per-group decoders
        self.decoders = nn.ModuleList(
            [
                GroupDecoder(
                    ogs, n_patches, embed_dim, patch_size, decoder_depth, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate
                )
                for ogs in out_group_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # split into groups along channel axis
        groups = x.split(self.group_sizes, dim=1)  # list of (B, gs, H, W)

        # encode each group
        tokens = [enc(g) for enc, g in zip(self.encoders, groups)]  # list of (B, N, D)

        # cross-group fusion
        for layer_cross, layer_self in zip(self.fuser, self.fuser_self):
            new_tokens = []
            for i, (t_q, cross_blk, self_blk) in enumerate(zip(tokens, layer_cross, layer_self)):
                # aggregate all other groups as context
                others = [tokens[j] for j in range(len(tokens)) if j != i]
                context = torch.cat(others, dim=1)  # (B, (G-1)*N, D)
                t_q = cross_blk(t_q, context)
                t_q = self_blk(t_q)
                new_tokens.append(t_q)
            # handle last group (no cross block for it in asymmetric list)
            if len(tokens) > len(layer_cross):
                idx = len(layer_cross)
                new_tokens.append(layer_self[idx](tokens[idx]))
            tokens = new_tokens

        # decode each group and concatenate
        outs = [dec(t, self.n_h, self.n_w) for dec, t in zip(self.decoders, tokens)]
        return torch.cat(outs, dim=1)  # (B, out_channels, H, W)


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITFengWu(nn.Module):
    """CREDIT wrapper for FengWu.  Flat (B, C, H, W) I/O."""

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(128, 256),
        patch_size=4,
        group_sizes=None,
        embed_dim=256,
        encoder_depth=2,
        fuser_depth=4,
        decoder_depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        pad_H = (patch_size - H % patch_size) % patch_size
        pad_W = (patch_size - W % patch_size) % patch_size
        self.pad_H, self.pad_W = pad_H, pad_W
        self.model = FengWuModel(
            img_size=(H + pad_H, W + pad_W),
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            group_sizes=group_sizes,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            fuser_depth=fuser_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:  # (B, C, T, H, W) from trainer → (B, C*T, H, W)
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)
        if self.pad_H > 0 or self.pad_W > 0:
            x = F.pad(x, (0, self.pad_W, 0, self.pad_H))
        out = self.model(x)
        if self.pad_H > 0 or self.pad_W > 0:
            out = out[:, :, : self.H, : self.W]
        return out

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
    H, W = 32, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3 variable groups
    model = CREDITFengWu(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=4,
        group_sizes=[4, 4, 4],
        embed_dim=64,
        encoder_depth=1,
        fuser_depth=2,
        decoder_depth=1,
        num_heads=4,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected shape {y.shape}"
    y.mean().backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITFengWu OK — output {y.shape}, params {n_params:.1f}M, device {device}")
