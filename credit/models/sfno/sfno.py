"""
Spherical Fourier Neural Operator (SFNO) weather model.
Bonev et al., 2023.  https://arxiv.org/abs/2306.03838
Original code: NVIDIA/makani / PhysicsNemo (Apache 2.0)

SFNO replaces 2D FFT (in AFNO) with Spherical Harmonic Transforms (SHTs)
to respect the geometry of the sphere.  When torch-harmonics is available
it uses real SHTs; otherwise it falls back to standard rfft2 (same as AFNO).
This lets the model load and train without torch-harmonics installed, at the
cost of the spherical-geometry correction.

Architecture: patch embed → N SFNO blocks → patch recovery.
Each SFNO block: norm → SHT → learned spectral filter → ISHT → residual,
followed by a channel-MLP residual branch.

CREDITSfno wraps the model for CREDIT's flat (B, C, H, W) tensors.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

try:
    from torch_harmonics import RealSHT, InverseRealSHT

    _HARMONICS_AVAILABLE = True
except ImportError:
    _HARMONICS_AVAILABLE = False

from einops import rearrange


# ---------------------------------------------------------------------------
# Spherical / Cartesian filter helpers
# ---------------------------------------------------------------------------


class SpectralFilterLayer(nn.Module):
    """
    Learnable filter in spectral (spherical-harmonic) space.

    When torch-harmonics is available uses real SHTs on the sphere.
    Falls back to rfft2 otherwise (Cartesian AFNO-style).

    Parameters
    ----------
    h, w : int
        Spatial dimensions (latitude × longitude).
    embed_dim : int
    n_modes : int
        Number of spectral modes to keep (truncation level).
    use_harmonics : bool
        Force harmonics on/off (default: use if available).
    """

    def __init__(self, h, w, embed_dim, n_modes=None, use_harmonics=True):
        super().__init__()
        self.h = h
        self.w = w
        self.embed_dim = embed_dim
        self.use_harmonics = use_harmonics and _HARMONICS_AVAILABLE

        if n_modes is None:
            n_modes = min(h, w // 2 + 1)
        self.n_modes = n_modes

        if self.use_harmonics:
            self.sht = RealSHT(h, w, lmax=n_modes, mmax=n_modes, grid="equiangular")
            self.isht = InverseRealSHT(h, w, lmax=n_modes, mmax=n_modes, grid="equiangular")
            # spectral weight: (embed_dim, n_modes, n_modes) complex
            n_spec = n_modes * n_modes
            self.weight_r = nn.Parameter(torch.empty(embed_dim, n_modes, n_modes))
            self.weight_i = nn.Parameter(torch.empty(embed_dim, n_modes, n_modes))
        else:
            # rfft2 fallback
            self.wf = w // 2 + 1
            self.weight_r = nn.Parameter(torch.empty(embed_dim, h, self.wf))
            self.weight_i = nn.Parameter(torch.empty(embed_dim, h, self.wf))

        nn.init.trunc_normal_(self.weight_r, std=0.02)
        nn.init.trunc_normal_(self.weight_i, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, D) → (B, H, W, D)"""
        B, H, W, D = x.shape
        # work channel-last → (B, D, H, W) for SHT
        x = x.permute(0, 3, 1, 2)

        if self.use_harmonics:
            x_c = self.sht(x)  # (B, D, lmax, mmax) complex
            # element-wise complex multiply with learned weight
            w = torch.complex(self.weight_r, self.weight_i)  # (D, lmax, mmax)
            x_c = x_c * w[None]
            x = self.isht(x_c)  # (B, D, H, W)
        else:
            x_c = torch.fft.rfft2(x, norm="ortho")  # (B, D, H, Wf)
            w = torch.complex(self.weight_r, self.weight_i)
            x_c = x_c * w[None]
            x = torch.fft.irfft2(x_c, s=(H, W), norm="ortho")

        return x.permute(0, 2, 3, 1)  # (B, H, W, D)


# ---------------------------------------------------------------------------
# SFNO block
# ---------------------------------------------------------------------------


class SFNOBlock(nn.Module):
    def __init__(self, h, w, embed_dim, n_modes=None, mlp_ratio=2.0, drop=0.0, use_harmonics=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.filter = SpectralFilterLayer(h, w, embed_dim, n_modes=n_modes, use_harmonics=use_harmonics)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(drop),
        )
        # 1×1 conv skip in spectral branch to allow channel mixing
        self.skip = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: (B, H, W, D)
        x = self.skip(x) + self.filter(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# SFNO model
# ---------------------------------------------------------------------------


class SFNOModel(nn.Module):
    """
    Spherical FNO weather model.

    Parameters
    ----------
    img_size : tuple[int, int]
        (H, W) of the input.
    patch_size : int
        Patch size for initial embedding.
    in_channels : int
    out_channels : int
    embed_dim : int
    depth : int
    n_modes : int, optional
        Spectral truncation (defaults to min(H, W//2+1)).
    mlp_ratio : float
    drop_rate : float
    use_harmonics : bool
        Use torch-harmonics SHT when available (default True).
    """

    def __init__(
        self,
        img_size=(128, 256),
        patch_size=1,
        in_channels=70,
        out_channels=69,
        embed_dim=256,
        depth=12,
        n_modes=None,
        mlp_ratio=2.0,
        drop_rate=0.0,
        use_harmonics=True,
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_h, self.n_w, embed_dim))
        self.drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [
                SFNOBlock(
                    self.n_h,
                    self.n_w,
                    embed_dim,
                    n_modes=n_modes,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    use_harmonics=use_harmonics,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, D, nh, nw)
        x = rearrange(x, "b d h w -> b h w d")
        x = self.drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.head(x)  # (B, nh, nw, C*p^2)
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


class CREDITSfno(nn.Module):
    """
    CREDIT wrapper for SFNO.  Flat (B, C, H, W) I/O.

    torch-harmonics is used automatically when installed; falls back to rfft2.
    """

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(128, 256),
        patch_size=1,
        embed_dim=256,
        depth=12,
        n_modes=None,
        mlp_ratio=2.0,
        drop_rate=0.0,
        use_harmonics=True,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        pad_H = (patch_size - H % patch_size) % patch_size
        pad_W = (patch_size - W % patch_size) % patch_size
        self.pad_H, self.pad_W = pad_H, pad_W
        self.model = SFNOModel(
            img_size=(H + pad_H, W + pad_W),
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            n_modes=n_modes,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            use_harmonics=use_harmonics,
        )
        if not _HARMONICS_AVAILABLE and use_harmonics:
            import logging

            logging.getLogger(__name__).warning(
                "torch-harmonics not installed; SFNO falling back to rfft2. Install with: pip install torch-harmonics"
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

    B, C_in, C_out, H, W = 2, 70, 69, 64, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITSfno(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=1,
        embed_dim=64,
        depth=4,
        use_harmonics=True,  # will fall back to rfft2 if torch-harmonics absent
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected shape {y.shape}"
    y.mean().backward()

    harmonics_str = "SHT" if _HARMONICS_AVAILABLE else "rfft2 fallback"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITSfno OK ({harmonics_str}) — output {y.shape}, params {n_params:.1f}M, device {device}")
