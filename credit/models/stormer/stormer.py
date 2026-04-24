"""
Stormer — Vision Transformer for weather forecasting (Nguyen et al., 2023).
https://arxiv.org/abs/2312.03876   MIT License

Architecture: plain ViT (patch embed → N transformer blocks → linear head).
No pressure-level special handling; treats all channels uniformly.

CREDITStormer wraps the core ViT for CREDIT's flat (B, C, H, W) tensors.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Core ViT building blocks
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.patch_h, self.patch_w = patch_size, patch_size
        self.n_patches = (self.img_h // self.patch_h) * (self.img_w // self.patch_w)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, n_patches, embed_dim)
        x = self.proj(x)
        return rearrange(x, "b d h w -> b (h w) d")


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Stormer ViT
# ---------------------------------------------------------------------------


class StormerViT(nn.Module):
    """
    Plain Vision Transformer weather model (Stormer architecture).

    Parameters
    ----------
    img_size : tuple[int, int]
        (H, W) of the input field.
    patch_size : int
        Patch size (square). Default 2.
    in_channels : int
        Number of input channels (surface + atmos + static).
    out_channels : int
        Number of predicted channels (surface + atmos, no static).
    embed_dim : int
        Transformer embedding dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Attention heads.
    mlp_ratio : float
        MLP hidden/embed ratio.
    drop_rate : float
        Dropout.
    """

    def __init__(
        self,
        img_size=(128, 256),
        patch_size=2,
        in_channels=70,
        out_channels=69,
        embed_dim=1024,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        )
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        self.n_patches = self.n_h * self.n_w

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # decode back to spatial: each patch → (out_channels * patch_size^2) pixels
        self.head = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        (B, out_channels, H, W)
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (B, n_patches, embed_dim)

        x = self.head(x)  # (B, n_patches, out_ch * p^2)
        # fold patches back to spatial
        x = rearrange(
            x,
            "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
            nh=self.n_h,
            nw=self.n_w,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return x


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITStormer(nn.Module):
    """
    Thin CREDIT wrapper around StormerViT.

    Accepts/returns flat (B, C, H, W) tensors matching CREDIT's channel layout.
    Static variables are appended to the input but not predicted.

    Parameters
    ----------
    in_channels : int
        Total input channels (surf + atmos + static).
    out_channels : int
        Output channels (surf + atmos, no static).
    img_size : tuple[int, int]
        (H, W). Default (128, 256).
    patch_size : int
        ViT patch size. Default 2.
    embed_dim : int
    depth : int
    num_heads : int
    mlp_ratio : float
    drop_rate : float
    """

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(128, 256),
        patch_size=2,
        embed_dim=1024,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        # Pad to next multiple of patch_size so the ViT assertion always passes
        pad_H = (patch_size - H % patch_size) % patch_size
        pad_W = (patch_size - W % patch_size) % patch_size
        self.pad_H, self.pad_W = pad_H, pad_W
        self.model = StormerViT(
            img_size=(H + pad_H, W + pad_W),
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
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

    B, C_in, C_out, H, W = 2, 70, 69, 128, 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITStormer(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=2,
        embed_dim=256,  # small for smoke test
        depth=4,
        num_heads=8,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected output shape {y.shape}"

    loss = y.mean()
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITStormer OK — output {y.shape}, params {n_params:.1f}M, device {device}")
