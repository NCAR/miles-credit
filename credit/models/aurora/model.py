"""CREDIT wrapper for the Aurora model.

Adapts Aurora's native :class:`.Batch` interface to CREDIT's flat
``(B, C, H, W)`` tensor convention so it can be trained with the standard
CREDIT data pipeline.

Channel layout (input and output tensors):
    channels = [surf_var_0, ..., surf_var_N,
                atmos_var_0_lev_0, atmos_var_0_lev_1, ..., atmos_var_0_lev_P,
                atmos_var_1_lev_0, ..., atmos_var_M_lev_P,
                static_var_0, ..., static_var_S]

where levels are ordered from ``atmos_levels[0]`` to ``atmos_levels[-1]``.
The output tensor contains surf + atmos channels only (static vars are
unchanged and dropped from the prediction).

Example config::

    model:
      type: aurora
      surf_vars: ["2t", "10u", "10v", "msl"]
      atmos_vars: ["z", "u", "v", "t", "q"]
      static_vars: ["lsm", "z", "slt"]
      atmos_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
      lat_min: -90.0
      lat_max: 90.0
      lon_min: 0.0
      lon_max: 359.0
      embed_dim: 256        # reduce for fast training
      encoder_depths: [2, 6, 2]
      encoder_num_heads: [4, 8, 16]
      decoder_depths: [2, 6, 2]
      decoder_num_heads: [16, 8, 4]
      use_lora: false
"""

from __future__ import annotations

import os
import sys

# Allow running this file directly: python credit/models/aurora/model.py
if __name__ == "__main__":
    _pkg_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    sys.path.insert(0, os.path.abspath(_pkg_root))

from datetime import datetime, timedelta
from typing import List, Optional

import torch
import torch.nn as nn

try:
    from .aurora import Aurora
    from .batch import Batch, Metadata
except ImportError:
    # Support running as __main__: python credit/models/aurora/model.py
    from credit.models.aurora.aurora import Aurora  # type: ignore[no-redef]
    from credit.models.aurora.batch import Batch, Metadata  # type: ignore[no-redef]

__all__ = ["CREDITAurora"]


class CREDITAurora(nn.Module):
    """Aurora wrapped for CREDIT's flat-tensor training pipeline.

    Args:
        surf_vars: Surface variable names (must match Aurora's normalisation table).
        atmos_vars: Atmospheric variable names.
        static_vars: Static variable names (lsm, z, slt …).
        atmos_levels: Pressure levels in hPa (e.g. ``[50, 100, …, 1000]``).
        lat_min: Southernmost latitude (default ``-90``).
        lat_max: Northernmost latitude (default ``90``).
        lon_min: Westernmost longitude in [0, 360) (default ``0``).
        lon_max: Easternmost longitude in [0, 360) (default ``359``).
        history_size: Number of history steps passed in the input tensor (default 1).
        timestep_hours: Model timestep in hours (default 6).
        **aurora_kwargs: Forwarded verbatim to :class:`.Aurora`.
    """

    def __init__(
        self,
        surf_vars: List[str] = ("2t", "10u", "10v", "msl"),
        atmos_vars: List[str] = ("z", "u", "v", "t", "q"),
        static_vars: List[str] = ("lsm", "z", "slt"),
        atmos_levels: List[int] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        lat_min: float = -90.0,
        lat_max: float = 90.0,
        lon_min: float = 0.0,
        lon_max: float = 359.0,
        n_lat: Optional[int] = None,
        n_lon: Optional[int] = None,
        history_size: int = 1,
        timestep_hours: int = 6,
        **aurora_kwargs,
    ) -> None:
        super().__init__()

        self.surf_vars = list(surf_vars)
        self.atmos_vars = list(atmos_vars)
        self.static_vars = list(static_vars)
        self.atmos_levels = tuple(int(lv) for lv in atmos_levels)
        self.history_size = history_size
        self.n_surf = len(surf_vars)
        self.n_atmos = len(atmos_vars)
        self.n_levels = len(atmos_levels)
        self.n_static = len(static_vars)

        self.aurora = Aurora(
            surf_vars=tuple(surf_vars),
            static_vars=tuple(static_vars),
            atmos_vars=tuple(atmos_vars),
            max_history_size=history_size,
            timestep=timedelta(hours=timestep_hours),
            **aurora_kwargs,
        )

        # Lat/lon grids are stored as buffers so they move with .to(device)
        # and are included in state_dict.
        if n_lat is None:
            n_lat = round(lat_max - lat_min) + 1
        if n_lon is None:
            n_lon = round(lon_max - lon_min) + 1
        lat = torch.linspace(lat_max, lat_min, n_lat)  # decreasing (Aurora convention)
        lon = torch.linspace(lon_min, lon_max, n_lon)  # increasing [0, 360)
        self.register_buffer("lat", lat)
        self.register_buffer("lon", lon)

    # ------------------------------------------------------------------
    # Channel layout helpers
    # ------------------------------------------------------------------

    @property
    def n_input_channels(self) -> int:
        """Total number of input channels in CREDIT flat tensor."""
        return self.history_size * (self.n_surf + self.n_atmos * self.n_levels) + self.n_static

    @property
    def n_output_channels(self) -> int:
        """Number of output channels (surf + atmos only, no static)."""
        return self.n_surf + self.n_atmos * self.n_levels

    def _flat_to_batch(self, x: torch.Tensor) -> Batch:
        """Split a CREDIT flat tensor ``(B, C, H, W)`` into an Aurora :class:`.Batch`.

        Channel layout::

            [hist_0_surf_vars... | hist_0_atmos_vars... |
             hist_1_surf_vars... | hist_1_atmos_vars... |   (repeated history_size times)
             static_vars...]

        The history axis runs from oldest (0) to most recent (T-1).
        """
        B, C, H, W = x.shape
        T = self.history_size

        surf_dict: dict[str, torch.Tensor] = {}
        atmos_dict: dict[str, torch.Tensor] = {}

        offset = 0
        # History steps
        surf_slices = []  # list over T of (B, n_surf, H, W)
        atmos_slices = []  # list over T of (B, n_atmos, n_levels, H, W)
        for _ in range(T):
            s = x[:, offset : offset + self.n_surf, :, :]
            surf_slices.append(s)
            offset += self.n_surf

            a = x[:, offset : offset + self.n_atmos * self.n_levels, :, :]
            a = a.view(B, self.n_atmos, self.n_levels, H, W)
            atmos_slices.append(a)
            offset += self.n_atmos * self.n_levels

        # Stack history: surf → (B, T, H, W) per var; atmos → (B, T, n_levels, H, W) per var
        surf_stacked = torch.stack(surf_slices, dim=1)  # (B, T, n_surf, H, W)
        atmos_stacked = torch.stack(atmos_slices, dim=1)  # (B, T, n_atmos, n_levels, H, W)

        for i, name in enumerate(self.surf_vars):
            surf_dict[name] = surf_stacked[:, :, i, :, :]  # (B, T, H, W)
        for i, name in enumerate(self.atmos_vars):
            atmos_dict[name] = atmos_stacked[:, :, i, :, :]  # (B, T, n_levels, H, W)

        # Static vars: (n_static, H, W) → per-var (H, W)
        static_dict: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.static_vars):
            static_dict[name] = x[0, offset + i, :, :]  # use first batch element

        # Dummy time metadata (Aurora only uses it for dynamic vars, off by default)
        dummy_time = tuple(datetime(2020, 1, 1) for _ in range(B))

        metadata = Metadata(
            lat=self.lat,
            lon=self.lon,
            time=dummy_time,
            atmos_levels=self.atmos_levels,
        )
        return Batch(
            surf_vars=surf_dict,
            static_vars=static_dict,
            atmos_vars=atmos_dict,
            metadata=metadata,
        )

    def _batch_to_flat(self, pred: Batch) -> torch.Tensor:
        """Reassemble Aurora :class:`.Batch` prediction into a flat ``(B, C, H, W)`` tensor.

        Output channel layout (no static vars, single predicted step)::

            [surf_var_0, ..., surf_var_N,
             atmos_var_0_lev_0, ..., atmos_var_0_lev_P,
             atmos_var_1_lev_0, ..., atmos_var_M_lev_P]
        """
        parts = []
        for name in self.surf_vars:
            # pred surf_vars: (B, T=1, H, W) → drop T
            parts.append(pred.surf_vars[name][:, 0, :, :].unsqueeze(1))  # (B,1,H,W)

        for name in self.atmos_vars:
            # pred atmos_vars: (B, T=1, n_levels, H, W) → drop T, flatten levels
            atmos = pred.atmos_vars[name][:, 0, :, :]  # (B, n_levels, H, W)
            parts.append(atmos)

        return torch.cat(parts, dim=1)  # (B, C_out, H, W)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next atmospheric state.

        Args:
            x: Input tensor of shape ``(B, C, H, W)`` following the channel
               layout described in the module docstring.

        Returns:
            Predicted tensor of shape ``(B, C_out, H, W)`` with
            ``C_out = n_surf + n_atmos * n_levels``.
        """
        batch = self._flat_to_batch(x)
        pred = self.aurora(batch)
        return self._batch_to_flat(pred)


# ---------------------------------------------------------------------------
# Junk-data smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Aurora smoke test on {device}")

    # Tiny grid so the test runs quickly even on CPU
    H, W = 32, 64  # 5.625-deg-like resolution

    surf_vars = ["2t", "10u", "10v", "msl"]
    atmos_vars = ["z", "u", "v", "t", "q"]
    static_vars = ["lsm", "z", "slt"]
    atmos_levels = [500, 850, 1000]  # 3 pressure levels
    n_surf = len(surf_vars)
    n_atmos = len(atmos_vars)
    n_levels = len(atmos_levels)
    n_static = len(static_vars)
    C_in = n_surf + n_atmos * n_levels + n_static
    C_out = n_surf + n_atmos * n_levels

    model = CREDITAurora(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        atmos_levels=atmos_levels,
        lat_min=-90.0,
        lat_max=90.0,
        lon_min=0.0,
        lon_max=354.375,  # 64 points × 5.625 deg/pt
        n_lat=H,
        n_lon=W,
        history_size=1,
        # Small config so it fits in memory
        embed_dim=128,
        num_heads=4,
        encoder_depths=(2, 2, 2),
        encoder_num_heads=(2, 4, 8),
        decoder_depths=(2, 2, 2),
        decoder_num_heads=(8, 4, 2),
        latent_levels=2,
        patch_size=4,
        enc_depth=1,
        dec_depth=1,
        use_lora=False,
        window_size=(2, 2, 4),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f} M")
    print(f"  Input channels : {C_in}")
    print(f"  Output channels: {C_out}")

    B = 2
    x = torch.randn(B, C_in, H, W, device=device)
    print(f"  Input  shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        y = model(x)
    print(f"  Output shape: {y.shape}")
    assert y.shape == (B, C_out, H, W), f"Unexpected output shape {y.shape}"

    # Backward pass
    x.requires_grad_(True)
    y = model(x)
    loss = y.mean()
    loss.backward()
    print(f"  Input grad shape: {x.grad.shape}")
    print("Aurora smoke test PASSED")
