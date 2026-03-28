"""
FourCastNet v1 — Adaptive Fourier Neural Operator (AFNO) backbone.
Pathak et al., 2022.  https://arxiv.org/abs/2202.11214
Original code: NVIDIA/FourCastNet (Apache 2.0)

AFNO replaces self-attention with token mixing in Fourier space:
  x → FFT → element-wise MLP (shared across spatial freqs) → IFFT → residual

CREDITFourCastNet wraps the model for CREDIT's flat (B, C, H, W) tensors.
No external dependencies.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange


# ---------------------------------------------------------------------------
# AFNO token mixer
# ---------------------------------------------------------------------------


class AFNOLayer(nn.Module):
    """
    Adaptive Fourier Neural Operator token mixer.

    Divides embedding into `n_blocks` independent blocks and applies a shared
    2-layer MLP in Fourier space within each block.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    n_blocks : int
        Number of frequency blocks (dim must be divisible by n_blocks).
    sparsity_threshold : float
        Soft-thresholding in Fourier domain.
    hidden_size_factor : int
        Expansion factor for the Fourier-domain MLP.
    """

    def __init__(self, dim, n_blocks=8, sparsity_threshold=0.01, hidden_size_factor=1):
        super().__init__()
        assert dim % n_blocks == 0, "dim must be divisible by n_blocks"
        self.dim = dim
        self.n_blocks = n_blocks
        self.block_size = dim // n_blocks
        self.sparsity_threshold = sparsity_threshold
        hidden = int(self.block_size * hidden_size_factor)

        # weights for the complex-valued 2-layer MLP in Fourier space
        # real and imaginary parts handled separately (each linear)
        self.w1r = nn.Parameter(torch.empty(n_blocks, self.block_size, hidden))
        self.w1i = nn.Parameter(torch.empty(n_blocks, self.block_size, hidden))
        self.b1r = nn.Parameter(torch.zeros(n_blocks, hidden))
        self.b1i = nn.Parameter(torch.zeros(n_blocks, hidden))
        self.w2r = nn.Parameter(torch.empty(n_blocks, hidden, self.block_size))
        self.w2i = nn.Parameter(torch.empty(n_blocks, hidden, self.block_size))
        self.b2r = nn.Parameter(torch.zeros(n_blocks, self.block_size))
        self.b2i = nn.Parameter(torch.zeros(n_blocks, self.block_size))

        nn.init.trunc_normal_(self.w1r, std=0.02)
        nn.init.trunc_normal_(self.w1i, std=0.02)
        nn.init.trunc_normal_(self.w2r, std=0.02)
        nn.init.trunc_normal_(self.w2i, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, H, W, dim)
        returns : (B, H, W, dim)
        """
        B, H, W, D = x.shape
        residual = x

        # 2D FFT
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # (B, H, W//2+1, D)

        # reshape into blocks
        x = x.reshape(B, H, W // 2 + 1, self.n_blocks, self.block_size)
        # x: (B, H, Wf, nb, bs) where Wf = W//2+1

        xr, xi = x.real, x.imag

        # layer 1 (complex MLP): (B, H, Wf, nb, bs) @ (nb, bs, hidden)
        # subscripts: b=batch h=height w=width n=n_blocks s=block_size e=hidden
        o1r = (
            torch.einsum("bhwns, nse -> bhwne", xr, self.w1r)
            - torch.einsum("bhwns, nse -> bhwne", xi, self.w1i)
            + self.b1r[None, None, None]
        )
        o1i = (
            torch.einsum("bhwns, nse -> bhwne", xr, self.w1i)
            + torch.einsum("bhwns, nse -> bhwne", xi, self.w1r)
            + self.b1i[None, None, None]
        )
        o1r = torch.nn.functional.relu(o1r)
        o1i = torch.nn.functional.relu(o1i)

        # layer 2
        o2r = (
            torch.einsum("bhwne, nes -> bhwns", o1r, self.w2r)
            - torch.einsum("bhwne, nes -> bhwns", o1i, self.w2i)
            + self.b2r[None, None, None]
        )
        o2i = (
            torch.einsum("bhwne, nes -> bhwns", o1r, self.w2i)
            + torch.einsum("bhwne, nes -> bhwns", o1i, self.w2r)
            + self.b2i[None, None, None]
        )

        # soft threshold (sparsity)
        o2r = torch.nn.functional.softshrink(o2r, lambd=self.sparsity_threshold)
        o2i = torch.nn.functional.softshrink(o2i, lambd=self.sparsity_threshold)

        # reassemble
        x = torch.complex(o2r, o2i)  # (B, H, Wf, nb, bs)
        x = x.reshape(B, H, W // 2 + 1, D)

        # IFFT
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")  # (B, H, W, D)

        return x + residual


# ---------------------------------------------------------------------------
# AFNO block (mixer + MLP)
# ---------------------------------------------------------------------------


class AFNOBlock(nn.Module):
    def __init__(self, dim, n_blocks=8, mlp_ratio=4.0, drop=0.0, sparsity_threshold=0.01, hidden_size_factor=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = AFNOLayer(
            dim, n_blocks=n_blocks, sparsity_threshold=sparsity_threshold, hidden_size_factor=hidden_size_factor
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: (B, H, W, D)
        x = x + self.filter(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# FourCastNet (AFNO ViT)
# ---------------------------------------------------------------------------


class FourCastNet(nn.Module):
    """
    FourCastNet v1 — AFNO Vision Transformer.

    Parameters
    ----------
    img_size : tuple[int, int]
    patch_size : int
    in_channels : int
    out_channels : int
    embed_dim : int
    depth : int
    n_blocks : int
        AFNO frequency blocks (embed_dim // n_blocks = block size).
    mlp_ratio : float
    drop_rate : float
    sparsity_threshold : float
    hidden_size_factor : int
    """

    def __init__(
        self,
        img_size=(128, 256),
        patch_size=8,
        in_channels=20,
        out_channels=20,
        embed_dim=768,
        depth=12,
        n_blocks=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        sparsity_threshold=0.01,
        hidden_size_factor=1,
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_h, self.n_w, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [
                AFNOBlock(
                    embed_dim,
                    n_blocks=n_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    sparsity_threshold=sparsity_threshold,
                    hidden_size_factor=hidden_size_factor,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # head: project each patch token back to patch pixels
        self.head = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) → (B, C_out, H, W)"""
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, D, nh, nw)
        x = rearrange(x, "b d nh nw -> b nh nw d")
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (B, nh, nw, D)

        x = self.head(x)  # (B, nh, nw, C_out*p^2)
        x = rearrange(
            x,
            "b nh nw (c ph pw) -> b c (nh ph) (nw pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return x


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITFourCastNet(nn.Module):
    """
    CREDIT wrapper for FourCastNet (AFNO).  Flat (B, C, H, W) I/O.
    """

    def __init__(
        self,
        in_channels=20,
        out_channels=20,
        img_size=(128, 256),
        patch_size=8,
        embed_dim=768,
        depth=12,
        n_blocks=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        sparsity_threshold=0.01,
        hidden_size_factor=1,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        pad_H = (patch_size - H % patch_size) % patch_size
        pad_W = (patch_size - W % patch_size) % patch_size
        self.pad_H, self.pad_W = pad_H, pad_W
        self.model = FourCastNet(
            img_size=(H + pad_H, W + pad_W),
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            n_blocks=n_blocks,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            sparsity_threshold=sparsity_threshold,
            hidden_size_factor=hidden_size_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    B, C_in, C_out, H, W = 2, 20, 20, 128, 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITFourCastNet(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=8,
        embed_dim=256,
        depth=4,
        n_blocks=8,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected shape {y.shape}"

    y.mean().backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITFourCastNet OK — output {y.shape}, params {n_params:.1f}M, device {device}")
