"""
FourCastNet3 (FCN3) — Geometric Spherical Neural Operator weather model.
Kurth et al., 2025.  https://research.nvidia.com/publication/2025-07_fourcastnet-3
Original: NVIDIA/PhysicsNeMo (Apache 2.0)

Architecture: U-Net encoder-decoder with Spherical Neural Operator (SNO) blocks.
The group-equivariant "geometric" aspect comes from using Spherical Harmonic
Transforms (SHTs) at every scale, making the spectral filter globally equivariant
under spherical rotations.

When torch-harmonics is installed, SHTs are used at each scale (true FCN3 behaviour).
Without it, rfft2 is used as a fallback (same as SFNO fallback).

Differences from SFNO:
  - U-Net multi-scale structure (3 encoder + bottleneck + 3 decoder stages)
  - Separate SHT at each spatial scale (lmax scales with resolution)
  - Skip connections + channel projections across scales
  - Group-normalisation instead of layer-norm (better for equivariance)

CREDITFourCastNetV3 wraps the model for CREDIT's flat (B, C, H, W) tensors.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.fft

try:
    from torch_harmonics import RealSHT, InverseRealSHT

    _HARMONICS_AVAILABLE = True
except ImportError:
    _HARMONICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Spherical Neural Operator block (reusable at any scale)
# ---------------------------------------------------------------------------


class SpectralConv2d(nn.Module):
    """
    Learnable spectral convolution on the sphere.

    Uses SHT when available; falls back to rfft2.
    At each scale, lmax = min(h, n_modes) to respect the Nyquist limit.
    """

    def __init__(self, h, w, dim, n_modes=None):
        super().__init__()
        self.h, self.w = h, w
        self.dim = dim
        if n_modes is None:
            n_modes = min(h, w // 2 + 1)
        self.n_modes = n_modes
        self.use_harmonics = _HARMONICS_AVAILABLE

        if self.use_harmonics:
            self.sht = RealSHT(h, w, lmax=n_modes, mmax=n_modes, grid="equiangular")
            self.isht = InverseRealSHT(h, w, lmax=n_modes, mmax=n_modes, grid="equiangular")
            self.weight_r = nn.Parameter(torch.empty(dim, n_modes, n_modes))
            self.weight_i = nn.Parameter(torch.empty(dim, n_modes, n_modes))
        else:
            wf = w // 2 + 1
            self.weight_r = nn.Parameter(torch.empty(dim, h, wf))
            self.weight_i = nn.Parameter(torch.empty(dim, h, wf))

        nn.init.trunc_normal_(self.weight_r, std=0.02)
        nn.init.trunc_normal_(self.weight_i, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, dim, H, W) → (B, dim, H, W)"""
        B, D, H, W = x.shape
        w = torch.complex(self.weight_r, self.weight_i)
        if self.use_harmonics:
            xc = self.sht(x) * w[None]
            return self.isht(xc)
        else:
            xc = torch.fft.rfft2(x, norm="ortho") * w[None]
            return torch.fft.irfft2(xc, s=(H, W), norm="ortho")


class SNOBlock(nn.Module):
    """
    Spherical Neural Operator block (channel-first layout).

    Structure: residual spectral branch + point-wise MLP branch.
    Uses GroupNorm for better equivariance under feature permutations.
    """

    def __init__(self, h, w, dim, n_modes=None, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        n_groups = min(8, dim)
        self.norm1 = nn.GroupNorm(n_groups, dim)
        self.spectral = SpectralConv2d(h, w, dim, n_modes=n_modes)
        self.skip = nn.Conv2d(dim, dim, 1)  # pointwise skip
        self.norm2 = nn.GroupNorm(n_groups, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden, dim, 1),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, dim, H, W)"""
        x = self.skip(x) + self.spectral(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------


class SNODown(nn.Module):
    """2× spatial downsampling: avg-pool + channel projection."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.proj = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        return self.proj(self.pool(x))


class SNOUp(nn.Module):
    """2× spatial upsampling: bilinear + channel projection."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(x)


# ---------------------------------------------------------------------------
# FCN3 model
# ---------------------------------------------------------------------------


class FCN3Model(nn.Module):
    """
    FourCastNet3 — multi-scale spherical neural operator U-Net.

    Parameters
    ----------
    img_size : tuple[int, int]
        (H, W).
    in_channels, out_channels : int
    base_dim : int
        Channel dim at the finest scale. Doubles at each stage.
    depth : int
        SNO blocks per stage (applied in both encoder and decoder).
    n_stages : int
        Number of downsampling stages. Default 3.
    n_modes : int, optional
        Spectral truncation at finest scale. Halved at each stage.
    mlp_ratio : float
    drop_rate : float
    """

    def __init__(
        self,
        img_size=(128, 256),
        in_channels=70,
        out_channels=69,
        base_dim=128,
        depth=2,
        n_stages=3,
        n_modes=None,
        mlp_ratio=2.0,
        drop_rate=0.0,
    ):
        super().__init__()
        H, W = img_size
        self.n_stages = n_stages

        dims = [base_dim * (2**i) for i in range(n_stages + 1)]  # [128, 256, 512, 1024]
        sizes = [(H // (2**i), W // (2**i)) for i in range(n_stages + 1)]

        if n_modes is None:
            n_modes = min(H, W // 2 + 1)
        modes_per_stage = [max(4, n_modes // (2**i)) for i in range(n_stages + 1)]

        # stem
        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        # encoder stages
        self.enc_stages = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        for i in range(n_stages):
            h, w = sizes[i]
            self.enc_stages.append(
                nn.Sequential(
                    *[
                        SNOBlock(h, w, dims[i], n_modes=modes_per_stage[i], mlp_ratio=mlp_ratio, drop=drop_rate)
                        for _ in range(depth)
                    ]
                )
            )
            self.down_layers.append(SNODown(dims[i], dims[i + 1]))

        # bottleneck
        h, w = sizes[n_stages]
        self.bottleneck = nn.Sequential(
            *[
                SNOBlock(h, w, dims[n_stages], n_modes=modes_per_stage[n_stages], mlp_ratio=mlp_ratio, drop=drop_rate)
                for _ in range(depth)
            ]
        )

        # decoder stages
        self.up_layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(n_stages - 1, -1, -1):
            h, w = sizes[i]
            self.up_layers.append(SNOUp(dims[i + 1], dims[i]))
            self.skip_projs.append(nn.Conv2d(dims[i] * 2, dims[i], 1))
            self.dec_stages.append(
                nn.Sequential(
                    *[
                        SNOBlock(h, w, dims[i], n_modes=modes_per_stage[i], mlp_ratio=mlp_ratio, drop=drop_rate)
                        for _ in range(depth)
                    ]
                )
            )

        self.head = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips = []
        for enc, down in zip(self.enc_stages, self.down_layers):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, skip_proj, dec, skip in zip(self.up_layers, self.skip_projs, self.dec_stages, reversed(skips)):
            x = up(x)
            # handle odd sizes from asymmetric pooling
            x = x[:, :, : skip.shape[2], : skip.shape[3]]
            x = skip_proj(torch.cat([x, skip], dim=1))
            x = dec(x)

        return self.head(x)


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITFourCastNetV3(nn.Module):
    """
    CREDIT wrapper for FourCastNet3 (FCN3).  Flat (B, C, H, W) I/O.

    torch-harmonics used automatically when installed; falls back to rfft2.
    """

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(128, 256),
        base_dim=128,
        depth=2,
        n_stages=3,
        n_modes=None,
        mlp_ratio=2.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.model = FCN3Model(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=base_dim,
            depth=depth,
            n_stages=n_stages,
            n_modes=n_modes,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        if not _HARMONICS_AVAILABLE:
            import logging

            logging.getLogger(__name__).warning(
                "torch-harmonics not installed; FCN3 falling back to rfft2 spectral conv. "
                "Install with: pip install torch-harmonics"
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

    B, C_in, C_out = 1, 70, 69
    H, W = 64, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITFourCastNetV3(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        base_dim=32,
        depth=2,
        n_stages=3,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected shape {y.shape}"
    y.mean().backward()

    harmonics_str = "SHT" if _HARMONICS_AVAILABLE else "rfft2 fallback"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITFourCastNetV3 OK ({harmonics_str}) — output {y.shape}, params {n_params:.1f}M, device {device}")
