"""
DLWP-HEALPix — Deep Learning Weather Prediction on the HEALPix sphere.

References
----------
Karlbauer et al. 2023, "Advancing Parsimonious Deep Learning Weather Prediction
using the HEALPix Mesh", JAMES. https://doi.org/10.1029/2023MS003700

Face-aware padding based on the Zephyr/PhysicsNeMo implementation by Noah
Brenowitz (nbren12) et al. at NVIDIA.
earth2grid: https://github.com/NVlabs/earth2grid  (Apache 2.0)
PhysicsNeMo: https://github.com/NVIDIA/physicsnemo  (Apache 2.0)

Architecture
------------
U-Net with HEALPix-aware convolutions that correctly pad across all 12 face
boundaries, including the rotated adjacencies at the polar faces.

CREDIT integration
------------------
CREDITHEALPix accepts flat (B, C, H, W) lat/lon tensors and:
1. Reprojects lat/lon → 12 HEALPix faces (nside×nside each) using healpy.
2. Folds 12 faces into batch dim: (B*12, C, nside, nside).
3. Runs the HEALPix U-Net with face-aware padding.
4. Unfolds + reprojects back to lat/lon.

healpy is required for correct pixel geometry.  If absent, falls back to an
approximate grid (not recommended for real runs).

nside : int  — HEALPix resolution.  N_pix = 12 * nside^2.
Suggested: nside=64 for 1.5° effective resolution.
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import healpy as hp

    _HEALPY = True
except ImportError:
    _HEALPY = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Face fold / unfold
# ---------------------------------------------------------------------------


class HEALPixFoldFaces(nn.Module):
    """(B, 12, C, H, W) → (B*12, C, H, W)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, C, H, W = x.shape
        return x.reshape(B * F, C, H, W)


class HEALPixUnfoldFaces(nn.Module):
    """(B*12, C, H, W) → (B, 12, C, H, W)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BF, C, H, W = x.shape
        return x.reshape(BF // 12, 12, C, H, W)


# ---------------------------------------------------------------------------
# Face-aware HEALPix padding (Zephyr / PhysicsNeMo approach)
#
# The 12 HEALPix faces are numbered:
#   0-3  : north polar cap
#   4-7  : equatorial belt
#   8-11 : south polar cap
#
# Polar faces have rotated adjacency (their neighbours must be rot90'd before
# copying into the padding region).  Equatorial faces have no rotation but
# their corner pixels are shared / interpolated.
# ---------------------------------------------------------------------------


class HEALPixPadding(nn.Module):
    """
    Topologically-correct padding for 12-face HEALPix tensors.

    Input:  (B*12, C, nside, nside)  — faces folded into batch
    Output: (B*12, C, nside+2p, nside+2p)

    Implements the face-adjacency rules from Gorski et al. 2005 / Calabretta
    & Roukema 2007, including the three rotation types needed at the polar
    faces (matching Zephyr/PhysicsNeMo).
    """

    def __init__(self, padding: int) -> None:
        super().__init__()
        self.p = padding
        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces()
        self.d = (-2, -1)  # spatial dims for rot90

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        x5d = self.unfold(x)  # (B, 12, C, ns, ns)
        f = [x5d[:, i] for i in range(12)]  # list of (B, C, ns, ns)

        # North polar faces (0-3): top neighbour rotated +90°, top-left +180°
        p00 = self._pn(f[0], t=f[1], tl=f[2], lft=f[3], bl=f[3], b=f[4], br=f[8], rgt=f[5], tr=f[1])
        p01 = self._pn(f[1], t=f[2], tl=f[3], lft=f[0], bl=f[0], b=f[5], br=f[9], rgt=f[6], tr=f[2])
        p02 = self._pn(f[2], t=f[3], tl=f[0], lft=f[1], bl=f[1], b=f[6], br=f[10], rgt=f[7], tr=f[3])
        p03 = self._pn(f[3], t=f[0], tl=f[1], lft=f[2], bl=f[2], b=f[7], br=f[11], rgt=f[4], tr=f[0])

        # Equatorial faces (4-7): no rotations, but interpolated corners
        p04 = self._pe(
            f[4],
            t=f[0],
            tl=self._tl(f[0], f[3]),
            lft=f[3],
            bl=f[7],
            b=f[11],
            br=self._br(f[11], f[8]),
            rgt=f[8],
            tr=f[5],
        )
        p05 = self._pe(
            f[5], t=f[1], tl=self._tl(f[1], f[0]), lft=f[0], bl=f[4], b=f[8], br=self._br(f[8], f[9]), rgt=f[9], tr=f[6]
        )
        p06 = self._pe(
            f[6],
            t=f[2],
            tl=self._tl(f[2], f[1]),
            lft=f[1],
            bl=f[5],
            b=f[9],
            br=self._br(f[9], f[10]),
            rgt=f[10],
            tr=f[7],
        )
        p07 = self._pe(
            f[7],
            t=f[3],
            tl=self._tl(f[3], f[2]),
            lft=f[2],
            bl=f[6],
            b=f[10],
            br=self._br(f[10], f[11]),
            rgt=f[11],
            tr=f[4],
        )

        # South polar faces (8-11): bottom rotated +90°, right -90°, br +180°
        p08 = self._ps(f[8], t=f[5], tl=f[0], lft=f[4], bl=f[11], b=f[11], br=f[10], rgt=f[9], tr=f[9])
        p09 = self._ps(f[9], t=f[6], tl=f[1], lft=f[5], bl=f[8], b=f[8], br=f[11], rgt=f[10], tr=f[10])
        p10 = self._ps(f[10], t=f[7], tl=f[2], lft=f[6], bl=f[9], b=f[9], br=f[8], rgt=f[11], tr=f[11])
        p11 = self._ps(f[11], t=f[4], tl=f[3], lft=f[7], bl=f[10], b=f[10], br=f[9], rgt=f[8], tr=f[8])

        stacked = torch.stack([p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11], dim=1)
        return self.fold(stacked)

    # ------------------------------------------------------------------
    # Padding helpers for the three face types
    # ------------------------------------------------------------------

    def _pn(self, c, t, tl, lft, bl, b, br, rgt, tr):
        """North polar face: top-neighbour rotated +90°, top-left +180°."""
        p, d = self.p, self.d
        c = torch.cat([t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]], dim=-2)
        left = torch.cat([tl.rot90(2, d)[..., -p:, -p:], lft.rot90(-1, d)[..., -p:], bl[..., :p, -p:]], dim=-2)
        right = torch.cat([tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]], dim=-2)
        return torch.cat([left, c, right], dim=-1)

    def _pe(self, c, t, tl, lft, bl, b, br, rgt, tr):
        """Equatorial face: no rotations."""
        p = self.p
        c = torch.cat([t[..., -p:, :], c, b[..., :p, :]], dim=-2)
        left = torch.cat([tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]], dim=-2)
        right = torch.cat([tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]], dim=-2)
        return torch.cat([left, c, right], dim=-1)

    def _ps(self, c, t, tl, lft, bl, b, br, rgt, tr):
        """South polar face: bottom +90°, right -90°, br +180°."""
        p, d = self.p, self.d
        c = torch.cat([t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]], dim=-2)
        left = torch.cat([tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]], dim=-2)
        right = torch.cat([tr[..., -p:, :p], rgt.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]], dim=-2)
        return torch.cat([left, c, right], dim=-1)

    def _tl(self, top, lft):
        """Interpolated top-left corner for equatorial faces (no shared pixel)."""
        p = self.p
        ret = torch.zeros_like(top)[..., :p, :p]
        ret[..., -1, -1] = 0.5 * top[..., -1, 0] + 0.5 * lft[..., 0, -1]
        for i in range(1, p):
            ret[..., -i - 1, -i:] = top[..., -i - 1, :i]
            ret[..., -i:, -i - 1] = lft[..., :i, -i - 1]
            ret[..., -i - 1, -i - 1] = 0.5 * top[..., -i - 1, 0] + 0.5 * lft[..., 0, -i - 1]
        return ret

    def _br(self, b, r):
        """Interpolated bottom-right corner for equatorial faces."""
        p = self.p
        ret = torch.zeros_like(b)[..., :p, :p]
        ret[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]
        for i in range(1, p):
            ret[..., :i, i] = r[..., -i:, i]
            ret[..., i, :i] = b[..., i, -i:]
            ret[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]
        return ret


# ---------------------------------------------------------------------------
# HEALPix-aware convolution layer
# ---------------------------------------------------------------------------


class HEALPixConv(nn.Module):
    """Conv2d preceded by topologically-correct HEALPix padding.

    For kernel_size=1 no padding is needed and the face-aware pad is skipped.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=True):
        super().__init__()
        p = kernel_size // 2
        self.pad = HEALPixPadding(p) if p > 0 else None
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net building blocks (all ops on folded faces: B*12, C, nside, nside)
# ---------------------------------------------------------------------------


class HEALPixBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        n_groups = min(8, dim)
        self.norm1 = nn.GroupNorm(n_groups, dim)
        self.conv1 = HEALPixConv(dim, dim)
        self.norm2 = nn.GroupNorm(n_groups, dim)
        self.conv2 = HEALPixConv(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class HEALPixDown(nn.Module):
    """2× downsample: AvgPool + channel projection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.proj = HEALPixConv(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.proj(self.pool(x))


class HEALPixUp(nn.Module):
    """2× upsample: bilinear + channel projection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = HEALPixConv(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(x)


class HEALPixUNet(nn.Module):
    """
    3-stage HEALPix U-Net.  All convolutions use face-aware HEALPix padding.

    Input/output: (B*12, C, nside, nside)
    """

    def __init__(self, in_channels, out_channels, embed_dim=64, depth=2, n_stages=3):
        super().__init__()
        dims = [embed_dim * (2**i) for i in range(n_stages + 1)]
        self.n_stages = n_stages

        self.stem = HEALPixConv(in_channels, dims[0])

        self.enc_stages = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        for i in range(n_stages):
            self.enc_stages.append(nn.Sequential(*[HEALPixBlock(dims[i]) for _ in range(depth)]))
            self.down_layers.append(HEALPixDown(dims[i], dims[i + 1]))

        self.bottleneck = nn.Sequential(*[HEALPixBlock(dims[n_stages]) for _ in range(depth)])

        self.up_layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(n_stages - 1, -1, -1):
            self.up_layers.append(HEALPixUp(dims[i + 1], dims[i]))
            self.skip_projs.append(HEALPixConv(dims[i] * 2, dims[i], kernel_size=1))
            self.dec_stages.append(nn.Sequential(*[HEALPixBlock(dims[i]) for _ in range(depth)]))

        self.head = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for enc, down in zip(self.enc_stages, self.down_layers):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, skip_proj, dec, skip in zip(self.up_layers, self.skip_projs, self.dec_stages, reversed(skips)):
            x = up(x)
            x = x[..., : skip.shape[-2], : skip.shape[-1]]
            x = skip_proj(torch.cat([x, skip], dim=1))
            x = dec(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Lat/lon ↔ HEALPix reprojection helpers
# ---------------------------------------------------------------------------


def _build_index_buffers(nside, H, W, lat_range, lon_range):
    """
    Returns:
        hp_to_ll : (12, nside, nside) long  — for each face pixel, flat ll index
        ll_to_hp : (H*W,) long              — for each ll pixel, face tensor flat index
    """
    n_pix = 12 * nside * nside
    all_pix = np.arange(n_pix)

    if _HEALPY:
        # Pixel centres in lat/lon degrees (NEST ordering)
        theta, phi = hp.pix2ang(nside, all_pix, nest=True)
        hp_lat = 90.0 - np.rad2deg(theta)  # (n_pix,)
        hp_lon = np.rad2deg(phi)  # (n_pix,) in [0, 360)

        # Face + xy coordinates for each NEST pixel
        x_hp, y_hp, f_hp = hp.pix2xyf(nside, all_pix, nest=True)
    else:
        # Fallback: approximate equal-area placement, no face structure
        fracs = (all_pix + 0.5) / n_pix
        hp_lat = np.rad2deg(np.arcsin(1.0 - 2.0 * fracs))
        hp_lon = (all_pix * 360.0 / n_pix) % 360.0
        # Assign pixels to faces in blocks
        f_hp = all_pix // (nside * nside)
        within = all_pix % (nside * nside)
        x_hp = within % nside
        y_hp = within // nside

    # lat/lon grid centres
    lat_grid = np.linspace(lat_range[0], lat_range[1], H)  # (H,)
    lon_grid = np.linspace(lon_range[0], lon_range[1], W)  # (W,)

    # ── hp_to_ll: for each hp face pixel, its nearest ll grid index ──────
    # For each hp pixel k: find argmin over lat_grid of |lat_grid - hp_lat[k]|,
    # similarly for lon_grid.
    i_lat = np.argmin(np.abs(lat_grid[:, None] - hp_lat[None, :]), axis=0)  # (n_pix,)
    j_lon = np.argmin(np.abs(lon_grid[:, None] - hp_lon[None, :]), axis=0)  # (n_pix,)
    ll_flat_idx = i_lat * W + j_lon  # (n_pix,) — flat ll index

    hp_to_ll = np.zeros((12, nside, nside), dtype=np.int64)
    hp_to_ll[f_hp, x_hp, y_hp] = ll_flat_idx

    # ── ll_to_hp: for each ll grid point, its face tensor flat index ──────
    if _HEALPY:
        lat2d = lat_grid[:, None] * np.ones((H, W))
        lon2d = lon_grid[None, :] * np.ones((H, W))
        theta2d = np.deg2rad(90.0 - lat2d).flatten()
        phi2d = np.deg2rad(lon2d).flatten()
        pix_nest = hp.ang2pix(nside, theta2d, phi2d, nest=True)  # (H*W,) NEST indices
        x_ll, y_ll, f_ll = hp.pix2xyf(nside, pix_nest, nest=True)
    else:
        pix_approx = (np.arange(H * W) * n_pix // (H * W)).astype(np.int64)
        f_ll = pix_approx // (nside * nside)
        within_ll = pix_approx % (nside * nside)
        x_ll = within_ll % nside
        y_ll = within_ll // nside

    # face tensor flat index = f * nside² + x * nside + y
    face_flat_idx = f_ll * nside * nside + x_ll * nside + y_ll  # (H*W,)
    ll_to_hp = face_flat_idx.astype(np.int64)

    return (
        torch.from_numpy(hp_to_ll).long(),  # (12, nside, nside)
        torch.from_numpy(ll_to_hp).long(),  # (H*W,)
    )


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITHEALPix(nn.Module):
    """
    CREDIT wrapper for DLWP-HEALPix.  Accepts/returns flat (B, C, H, W) lat/lon tensors.

    Internally reprojects lat/lon ↔ 12 HEALPix faces using healpy (required for
    correct pixel geometry; falls back to an approximate grid otherwise).

    Parameters
    ----------
    in_channels, out_channels : int
    img_size : (H, W)
    nside : int
        HEALPix resolution.  12*nside² total pixels.  Suggested nside=64 for
        ~1.5° resolution.
    embed_dim : int
        Base channel dim in the U-Net (doubles per stage).
    depth : int
        HEALPixBlocks per stage.
    n_stages : int
        U-Net downsampling stages (default 3).
    lat_range, lon_range : (float, float)
    """

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(128, 256),
        nside=32,
        embed_dim=64,
        depth=2,
        n_stages=3,
        lat_range=(-90.0, 90.0),
        lon_range=(0.0, 360.0),
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.nside = nside
        self.n_pix = 12 * nside * nside

        if not _HEALPY:
            logger.warning("healpy not installed; HEALPix grid is approximate. Install with: pip install healpy")

        hp_to_ll, ll_to_hp = _build_index_buffers(nside, H, W, lat_range, lon_range)
        # hp_to_ll: (12, nside, nside) — used in ll→hp direction
        # ll_to_hp: (H*W,)            — used in hp→ll direction
        self.register_buffer("hp_to_ll", hp_to_ll)
        self.register_buffer("ll_to_hp", ll_to_hp)

        self.unet = HEALPixUNet(
            in_channels,
            out_channels,
            embed_dim=embed_dim,
            depth=depth,
            n_stages=n_stages,
        )

        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces()

    def _ll_to_faces(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B*12, C, nside, nside)"""
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W)
        # hp_to_ll[f, px, py] = flat ll index of the nearest grid point
        idx = self.hp_to_ll.reshape(-1)  # (12*nside²,)
        hp_flat = x_flat[:, :, idx]  # (B, C, 12*nside²)
        faces = hp_flat.reshape(B, C, 12, self.nside, self.nside)
        # permute to (B, 12, C, nside, nside) then fold
        return self.fold(faces.permute(0, 2, 1, 3, 4))  # (B*12, C, nside, nside)

    def _faces_to_ll(self, x: torch.Tensor) -> torch.Tensor:
        """(B*12, C, nside, nside) → (B, C, H, W)"""
        B12, C, NS, _ = x.shape
        B = B12 // 12
        faces = self.unfold(x)  # (B, 12, C, nside, nside)
        hp_flat = faces.permute(0, 2, 1, 3, 4).reshape(B, C, -1)  # (B, C, 12*nside²)
        # ll_to_hp[i] = face-tensor flat index for ll grid point i
        ll_flat = hp_flat[:, :, self.ll_to_hp]  # (B, C, H*W)
        return ll_flat.reshape(B, C, self.H, self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        faces = self._ll_to_faces(x)
        out = self.unet(faces)
        return self._faces_to_ll(out)

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

    B, C_in, C_out = 1, 8, 6
    H, W = 32, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITHEALPix(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        nside=8,
        embed_dim=16,
        depth=1,
        n_stages=2,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"unexpected shape {y.shape}"
    y.mean().backward()

    healpy_str = "healpy (correct)" if _HEALPY else "approx (no healpy)"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITHEALPix OK ({healpy_str}) — output {y.shape}, params {n_params:.1f}M, device {device}")
