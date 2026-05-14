"""
FourCastNet3 (FCN3) — Atmospheric Spherical Neural Operator.

Ported from NVIDIA/makani (Apache-2.0):
  makani/models/networks/fourcastnet3.py  →  class AtmoSphericNeuralOperatorNet

Paper: Kurth et al. 2025, arXiv:2507.12144
Original code: https://github.com/NVIDIA/makani

Architecture (from makani):
    DiscreteContinuousEncoder (DISCO conv, equiangular H×W → legendre-gauss h×w)
    → N NeuralOperatorBlocks (alternating local DISCO and global diagonal-harmonic)
    → DiscreteContinuousDecoder (bilinear ResampleS2 + DISCO conv, legendre-gauss h×w → equiangular H×W)

All spherical ops use torch-harmonics (DiscreteContinuousConvS2, RealSHT, ResampleS2).
No makani or physicsnemo dependencies required.

Key differences from the makani original:
- No distributed training (single-GPU only; no makani.utils.comm)
- No channel grouping by variable type (flat B×C×H×W interface)
- No SST imputation, water clamping, or auxiliary channels
"""

import math
import os
import sys

import torch
import torch.nn as nn

try:
    from torch_harmonics import (
        DiscreteContinuousConvS2,
        InverseRealSHT,
        RealSHT,
        ResampleS2,
    )

    _TH_AVAILABLE = True
except ImportError:
    _TH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _theta_cutoff(lmax: int, kernel_shape: tuple, margin: float = 1.0) -> float:
    """Heuristic cutoff radius for morlet DISCO kernels (from makani FCN3)."""
    return margin * kernel_shape[0] * math.pi / float(lmax)


class LayerScale(nn.Module):
    """Per-channel learnable scale factor (from makani)."""

    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, dim, 1, 1), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class MLP(nn.Module):
    """Point-wise 1×1 conv MLP (channel-first layout)."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(drop),
        )
        with torch.no_grad():
            self.net[0].weight *= 0.5
            self.net[3].weight *= 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Local and global convolution layers
# ---------------------------------------------------------------------------


class DISCOConv(nn.Module):
    """
    Local DISCO (Discrete-Continuous) spherical convolution.

    Operates within the legendre-gauss internal grid — the h×w downsampled space.
    Corresponds to conv_type='local' blocks in makani's NeuralOperatorBlock.
    """

    def __init__(self, dim: int, h: int, w: int, kernel_shape: tuple, lmax: int):
        super().__init__()
        theta_cutoff = _theta_cutoff(lmax, kernel_shape)
        self.conv = DiscreteContinuousConvS2(
            dim,
            dim,
            in_shape=(h, w),
            out_shape=(h, w),
            kernel_shape=kernel_shape,
            basis_type="morlet",
            basis_norm_mode="mean",
            grid_in="legendre-gauss",
            grid_out="legendre-gauss",
            bias=False,
            theta_cutoff=theta_cutoff,
        )
        with torch.no_grad():
            self.conv.weight *= 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self.conv(x.float()).to(x.dtype)


class DHConv(nn.Module):
    """
    Global diagonal-harmonic spectral convolution.

    SHT → learned per-channel complex filter → ISHT.
    Corresponds to conv_type='global' (operator_type='dhconv') in makani.
    AMP guard prevents cuFFT fp16 failure on non-power-of-2 grids.
    """

    def __init__(self, dim: int, h: int, w: int, lmax: int):
        super().__init__()
        self.sht = RealSHT(h, w, lmax=lmax, mmax=lmax, grid="legendre-gauss").float()
        self.isht = InverseRealSHT(h, w, lmax=lmax, mmax=lmax, grid="legendre-gauss").float()
        self.weight_r = nn.Parameter(torch.empty(dim, lmax, lmax))
        self.weight_i = nn.Parameter(torch.empty(dim, lmax, lmax))
        nn.init.trunc_normal_(self.weight_r, std=0.02)
        nn.init.trunc_normal_(self.weight_i, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            xf = x.float()
            w = torch.complex(self.weight_r.float(), self.weight_i.float())
            xc = self.sht(xf) * w[None]
            return self.isht(xc).to(x.dtype)


# ---------------------------------------------------------------------------
# NeuralOperatorBlock
# ---------------------------------------------------------------------------


class NeuralOperatorBlock(nn.Module):
    """
    Single processor block: norm(none) → conv → MLP → LayerScale → identity skip.

    Replicates makani's NeuralOperatorBlock with use_mlp=True, skip='identity',
    normalization_layer='none'.
    """

    def __init__(
        self,
        h: int,
        w: int,
        dim: int,
        lmax: int,
        conv_type: str = "local",
        mlp_ratio: float = 2.0,
        kernel_shape: tuple = (3, 3),
    ):
        super().__init__()
        if conv_type == "local":
            self.conv = DISCOConv(dim, h, w, kernel_shape, lmax)
        elif conv_type == "global":
            self.conv = DHConv(dim, h, w, lmax)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.layer_scale = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = self.conv(x)
        dx = self.mlp(dx)
        return x + self.layer_scale(dx)


# ---------------------------------------------------------------------------
# FCN3Net backbone
# ---------------------------------------------------------------------------


class FCN3Net(nn.Module):
    """
    FourCastNet3 backbone.

    Faithfully ports makani's AtmoSphericNeuralOperatorNet without distributed
    training, channel grouping, or NVIDIA-specific infrastructure.

    Parameters
    ----------
    in_channels, out_channels : int
    img_size : (H, W)  —  equiangular input/output grid.
    embed_dim : int    —  internal channel dimension.
    num_layers : int   —  total processor blocks (default 8, as in paper).
    scale_factor : int —  spatial downsampling factor for the internal grid.
    sfno_block_frequency : int — every k-th block is global (DHConv); others local (DISCO).
    kernel_shape : (int, int) — DISCO Morlet kernel footprint.
    mlp_ratio : float  —  MLP hidden / embed_dim ratio inside each block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: tuple = (181, 360),
        embed_dim: int = 256,
        num_layers: int = 8,
        scale_factor: int = 4,
        sfno_block_frequency: int = 2,
        kernel_shape: tuple = (3, 3),
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        H, W = img_size
        h, w = H // scale_factor, W // scale_factor
        self.H, self.W = H, W
        self.h, self.w = h, w
        lmax = min(h, w // 2 + 1)
        self.lmax = lmax
        tc = _theta_cutoff(lmax, kernel_shape)

        # encoder: equiangular (H, W) → legendre-gauss (h, w)
        self.encoder = DiscreteContinuousConvS2(
            in_channels,
            embed_dim,
            in_shape=(H, W),
            out_shape=(h, w),
            kernel_shape=kernel_shape,
            basis_type="morlet",
            basis_norm_mode="mean",
            grid_in="equiangular",
            grid_out="legendre-gauss",
            bias=False,
            theta_cutoff=tc,
        )
        with torch.no_grad():
            self.encoder.weight *= 0.5

        # processor
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            conv_type = "global" if (sfno_block_frequency > 0 and i % sfno_block_frequency == 0) else "local"
            self.blocks.append(
                NeuralOperatorBlock(
                    h, w, embed_dim, lmax, conv_type=conv_type, mlp_ratio=mlp_ratio, kernel_shape=kernel_shape
                )
            )

        # decoder: legendre-gauss (h, w) → equiangular (H, W) via bilinear resample + DISCO
        self.resample = ResampleS2(h, w, H, W, grid_in="legendre-gauss", grid_out="equiangular")
        self.decoder = DiscreteContinuousConvS2(
            embed_dim,
            out_channels,
            in_shape=(H, W),
            out_shape=(H, W),
            kernel_shape=kernel_shape,
            basis_type="morlet",
            basis_norm_mode="mean",
            grid_in="equiangular",
            grid_out="equiangular",
            bias=False,
            theta_cutoff=tc,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All DISCO and SHT ops require fp32 (sparse bmm / cuFFT limitation with AMP)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = self.encoder(x.float())

        for blk in self.blocks:
            x = blk(x)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = self.resample(x.float())
            x = self.decoder(x)

        return x


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITFourCastNetV3(nn.Module):
    """
    CREDIT wrapper for FourCastNet3.  Flat (B, C, H, W) I/O; returns (B, C_out, 1, H, W).

    Requires torch-harmonics (DiscreteContinuousConvS2, ResampleS2, RealSHT).
    Install with: pip install torch-harmonics
    """

    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 69,
        img_size: tuple = (181, 360),
        frames: int = 1,
        embed_dim: int = 256,
        num_layers: int = 8,
        scale_factor: int = 4,
        sfno_block_frequency: int = 2,
        kernel_shape: tuple = (3, 3),
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if not _TH_AVAILABLE:
            raise ImportError(
                "torch-harmonics is required for FourCastNet3.  Install with: pip install torch-harmonics"
            )
        self.H, self.W = img_size
        self.model = FCN3Net(
            in_channels=in_channels * frames,
            out_channels=out_channels,
            img_size=img_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            scale_factor=scale_factor,
            sfno_block_frequency=sfno_block_frequency,
            kernel_shape=tuple(kernel_shape),
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:  # (B, C, T, H, W) → (B, C*T, H, W)
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)
        out = self.model(x)
        return out.unsqueeze(2)  # (B, C_out, 1, H, W)

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

    B, C_in, C_out = 1, 8, 6
    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITFourCastNetV3(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        embed_dim=32,
        num_layers=4,
        scale_factor=4,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, 1, H, W), f"unexpected shape {y.shape}"
    y.mean().backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITFourCastNetV3 OK — output {y.shape}, {n_params:.1f}M params, device {device}")
