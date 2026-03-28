"""
ClimaX — Climate and Weather Forecasting ViT (Nguyen et al., 2023).
https://arxiv.org/abs/2301.10343   MIT License

Key idea: per-variable tokenization.  Each variable gets its own learned
embedding added to the patch embedding, so the model generalises across
different input variable sets.  An aggregation step folds per-variable
tokens into a single token sequence before the main ViT backbone.

CREDITClimaX wraps the core model for CREDIT's flat (B, C, H, W) tensors.
"""

import os
import sys

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Building blocks (shared with Stormer)
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
        attn = self.attn_drop(attn.softmax(dim=-1))
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
# ClimaX core
# ---------------------------------------------------------------------------


class ClimaX(nn.Module):
    """
    ClimaX Vision Transformer with per-variable tokenization.

    Each input channel is embedded independently (patch embed + variable embed),
    then aggregated (mean over variables at each spatial location) before the
    standard ViT backbone.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    img_size : tuple[int, int]
        (H, W).
    patch_size : int
        Patch size (square).
    embed_dim : int
        ViT embedding dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
    mlp_ratio : float
    drop_rate : float
    agg_depth : int
        Number of transformer blocks in the variable-aggregation step.
        Set to 0 to skip (pure mean aggregation).
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
        agg_depth=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        H, W = img_size
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        self.n_patches = self.n_h * self.n_w

        # per-variable patch projection (shared weights, independent variable embed)
        self.patch_proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        # per-variable learned embedding: one vector per input channel
        self.var_embed = nn.Embedding(in_channels, embed_dim)

        # spatial position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

        self.pos_drop = nn.Dropout(drop_rate)

        # optional aggregation transformer (runs over variable axis)
        self.agg_blocks = (
            nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate) for _ in range(agg_depth)])
            if agg_depth > 0
            else None
        )
        self.agg_norm = nn.LayerNorm(embed_dim) if agg_depth > 0 else None

        # main ViT backbone
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # prediction head
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
        x : (B, C_in, H, W)

        Returns
        -------
        (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        var_ids = torch.arange(C, device=x.device)  # (C,)
        v_emb = self.var_embed(var_ids)  # (C, D)

        # patch embed each channel independently
        # reshape to (B*C, 1, H, W) for a single shared Conv2d
        xc = x.reshape(B * C, 1, H, W)
        tokens = self.patch_proj(xc)  # (B*C, D, nh, nw)
        tokens = rearrange(tokens, "(b c) d nh nw -> b c (nh nw) d", b=B, c=C)

        # add variable embedding (broadcast over patches)
        tokens = tokens + v_emb[None, :, None, :]  # (B, C, n_patches, D)

        # add spatial position embedding (broadcast over variables)
        tokens = tokens + self.pos_embed[:, None, :, :]  # (B, C, n_patches, D)

        # optional aggregation transformer: run over variable axis at each patch
        if self.agg_blocks is not None:
            # treat (B * n_patches) as batch, C as sequence
            tokens = rearrange(tokens, "b c n d -> (b n) c d")
            for blk in self.agg_blocks:
                tokens = blk(tokens)
            tokens = self.agg_norm(tokens)
            # aggregate to single token per patch (mean over variables)
            tokens = tokens.mean(dim=1)  # (B*n_patches, D)
            tokens = rearrange(tokens, "(b n) d -> b n d", b=B)
        else:
            # simple mean aggregation
            tokens = tokens.mean(dim=1)  # (B, n_patches, D)

        tokens = self.pos_drop(tokens)

        # main backbone
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)  # (B, n_patches, D)

        # decode
        out = self.head(tokens)  # (B, n_patches, C_out*p^2)
        out = rearrange(
            out,
            "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
            nh=self.n_h,
            nw=self.n_w,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return out


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITClimaX(nn.Module):
    """
    CREDIT wrapper for ClimaX.  Accepts/returns flat (B, C, H, W) tensors.

    Parameters
    ----------
    in_channels, out_channels, img_size, patch_size, embed_dim,
    depth, num_heads, mlp_ratio, drop_rate, agg_depth : see ClimaX.
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
        agg_depth=2,
    ):
        super().__init__()
        self.model = ClimaX(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            agg_depth=agg_depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:  # (B, C, T, H, W) from trainer → (B, C*T, H, W)
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)
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

    B, C_in, C_out, H, W = 2, 70, 69, 128, 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITClimaX(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=2,
        embed_dim=256,
        depth=4,
        num_heads=8,
        agg_depth=1,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected output shape {y.shape}"

    loss = y.mean()
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITClimaX OK — output {y.shape}, params {n_params:.1f}M, device {device}")
