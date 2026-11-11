"""
cube_rope.py
------------
GnomonicRoPE: 2D axial rotary position embedding for within-face attention on a
cubed-sphere face, using an equiangular-corrected coordinate instead of raw
grid index.

Background
~~~~~~~~~~
The within-face attention blocks (``credit.models.wxformer.crossformer.Attention``)
currently use a translation-invariant relative-position bias
(``DynamicPositionBias``): the same bias is added for a given grid-index offset
regardless of where the attending window sits on the face. That assumption is
fine on a regular lat/lon grid, but not on a cubed-sphere face: a raw gnomonic
(central) projection is uniform in tangent-plane coordinate, not in true angle.
A fixed index step near a face center corresponds to a *larger* true angular
step than the same index step near a face edge/corner -- the well known ~2x
resolution non-uniformity of naive gnomonic cubed-sphere grids (coarsest at
face centers, finest near edges/corners). A translation-invariant bias can't
see this: it treats a k-cell window near the center the same as a k-cell
window near a corner, even though they cover different true angular extents.

GnomonicRoPE corrects this with the standard closed-form equiangular
reparametrization (Rancic et al. 1996, "A global shallow water model using an
expanded spherical cube"): the raw gnomonic tangent-plane coordinate
alpha = tan(theta) (theta = true angle from the face center), so
xi = arctan(alpha) recovers theta exactly along each axis (up to the small
cross-axis term dropped by this separable/diagonal approximation -- the same
approximation real equiangular cubed-sphere grids use). Rotary phases are built
from (xi, eta) instead of raw (row, col), so relative attention becomes aware
of true angular displacement, not just index displacement -- with no
face-adjacency bookkeeping needed (halo.py's cross-face ownership/ghost-cell
machinery is unrelated; this only needs a single face's own local grid).

This is a from-scratch, closed-form per-pixel coordinate correction plugged
into standard 2D axial RoPE -- not a port of any published tile-based /
stereographic-reprojection RoPE scheme; there is no tile decomposition or
per-tile reprojection here.

Scope (v1)
~~~~~~~~~~
Wired into the within-face windowed attention only (``crossformer.Attention``
/ ``Transformer``, both "short" and "long" window types). Cross-face attention
(``FaceAttention``, ``FaceEdgeAttention``) does not use this yet -- a token's
position there is "which of the 6 faces", not a location on one face's grid,
and deserves its own position scheme building on the true 3D adjacency already
in halo.py. Left for a follow-up.

Usage
-----
    rope = GnomonicRoPE(dim_head=32)
    q, k = rope(q, k)   # q, k: (..., H, W, dim_head) -> same shape, rotated
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GnomonicRoPE(nn.Module):
    """2D axial RoPE using equiangular-corrected cube-face coordinates.

    Parameters
    ----------
    dim_head : int
        Per-head channel dimension. Must be divisible by 4 (half the channels
        rotate with the xi-axis phase, half with the eta-axis phase, and each
        half needs an even number of channels for the standard rotate-half
        pairing).
    base : float
        RoPE frequency base (same role as the standard base=10000 in 1D RoPE),
        applied per axis over that axis's half of the channels.
    phase_scale : float
        Multiplies (xi, eta) before generating phases. xi, eta each live in
        (-pi/4, pi/4) for one cube face (arctan of a coordinate in [-1, 1]);
        the default expands that to roughly (-pi, pi) so the lowest frequency
        band completes about one full rotation across a whole face.
    """

    def __init__(self, dim_head: int, base: float = 10000.0, phase_scale: float = 4.0):
        super().__init__()
        if dim_head % 4 != 0:
            raise ValueError(f"GnomonicRoPE requires dim_head divisible by 4, got {dim_head}")
        self.dim_head = dim_head
        self.base = base
        self.phase_scale = phase_scale
        self._axis_dim = dim_head // 2  # channels allotted to each axis (xi, eta)

        freq_idx = torch.arange(0, self._axis_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (freq_idx / self._axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Lazy per-(h, w, device, dtype) cache -- not a persistent buffer since
        # it's cheap to rebuild and shape varies by encoder stage.
        self._cache_key = None
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None

    @staticmethod
    def _equiangular_coords(h: int, w: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-axis (xi, eta) for an h x w grid treated as one cube face.

        xi indexes columns (the alpha / "longitude-like" axis), eta indexes
        rows (the beta / "latitude-like" axis) -- same row/col convention as
        ``halo.py``'s alpha/beta.
        """
        col = torch.arange(w, device=device, dtype=dtype)
        row = torch.arange(h, device=device, dtype=dtype)
        alpha = 2.0 * col / max(w - 1, 1) - 1.0
        beta = 2.0 * row / max(h - 1, 1) - 1.0
        xi = torch.atan(alpha)  # (w,)
        eta = torch.atan(beta)  # (h,)
        return xi, eta

    def _build_cos_sin(self, h: int, w: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        xi, eta = self._equiangular_coords(h, w, device, dtype)
        xi = xi * self.phase_scale
        eta = eta * self.phase_scale

        inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        xi_phase = xi[:, None] * inv_freq[None, :]  # (w, axis_dim/2)
        eta_phase = eta[:, None] * inv_freq[None, :]  # (h, axis_dim/2)

        xi_phase = xi_phase[None, :, :].expand(h, w, -1)  # (h, w, axis_dim/2)
        eta_phase = eta_phase[:, None, :].expand(h, w, -1)  # (h, w, axis_dim/2)

        phase = torch.cat([xi_phase, eta_phase], dim=-1)  # (h, w, dim_head/2)
        # _rotate_half pairs channel i with channel i + dim_head/2 (split-half
        # convention, not interleaved), so cos/sin must repeat each phase value
        # across BOTH halves (torch.cat), not adjacent pairs (repeat_interleave).
        cos = torch.cat([torch.cos(phase), torch.cos(phase)], dim=-1)  # (h, w, dim_head)
        sin = torch.cat([torch.sin(phase), torch.sin(phase)], dim=-1)
        return cos, sin

    def _cos_sin(self, h: int, w: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, device, dtype)
        if key != self._cache_key:
            self._cos_cache, self._sin_cache = self._build_cos_sin(h, w, device, dtype)
            self._cache_key = key
        return self._cos_cache, self._sin_cache

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate q, k of shape (..., H, W, dim_head) at absolute grid position (H, W)."""
        *_, h, w, d = q.shape
        if d != self.dim_head:
            raise ValueError(f"GnomonicRoPE: expected last dim {self.dim_head}, got {d}")
        cos, sin = self._cos_sin(h, w, q.device, q.dtype)
        q_out = q * cos + self._rotate_half(q) * sin
        k_out = k * cos + self._rotate_half(k) * sin
        return q_out, k_out
