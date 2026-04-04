"""
_preblock_wrf.py
-------------------------------------------------------
WRFPreBlock: pre-model data assembly for the WRF regional pipeline.

Encapsulates the input tensor assembly that is specific to the regional
WRF model, which takes two input streams:
  - Interior  (WRF state: upper-air + surface + optional static/forcing)
  - Boundary  (ERA5 boundary conditions: upper-air + surface, NaN-filled)

Usage in the trainer::

    preblock = WRFPreBlock()
    x, x_boundary, x_time = preblock(batch, device)
    y = preblock.assemble_target(batch, device)
    y_pred = model(x, x_boundary, x_time)
"""

import torch
import torch.nn as nn

from credit.data import concat_and_reshape, reshape_only


class WRFPreBlock(nn.Module):
    """Assemble model inputs from a WRF singlestep DataLoader batch.

    The boundary NaN fill (``torch.nan_to_num(..., nan=0.0)``) is the
    key WRF-specific operation: ERA5 boundary zarrs store NaN in the
    WRF interior (only the lateral edges are valid data).

    Args:
        boundary_fill_value (float): Value to substitute for NaN in the
            boundary tensor.  Default ``0.0``.
    """

    def __init__(self, boundary_fill_value: float = 0.0):
        super().__init__()
        self.boundary_fill_value = boundary_fill_value

    def forward(self, batch: dict, device: torch.device):
        """Assemble model inputs from a batch dict.

        Args:
            batch: Dict produced by the WRF DataLoader.  Expected keys:
                ``x``, optionally ``x_surf``, ``x_forcing_static``,
                ``x_boundary``, optionally ``x_surf_boundary``,
                ``x_time_encode``.
            device: Target device.

        Returns:
            Tuple ``(x, x_boundary, x_time_encode)``:
                - ``x``: interior state tensor ``(B, C, T, H, W)``
                - ``x_boundary``: boundary tensor ``(B, C, T, H, W)``,
                  NaN replaced with ``boundary_fill_value``
                - ``x_time_encode``: time-encoding tensor ``(B, D)``
        """
        # --- interior state ------------------------------------------------
        if "x_surf" in batch:
            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device)
        else:
            x = reshape_only(batch["x"]).to(device)

        if "x_forcing_static" in batch:
            x_forcing = batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4)
            x = torch.cat((x, x_forcing), dim=1)

        # --- boundary conditions -------------------------------------------
        if "x_surf_boundary" in batch:
            x_boundary = concat_and_reshape(batch["x_boundary"], batch["x_surf_boundary"]).to(device)
        else:
            x_boundary = reshape_only(batch["x_boundary"]).to(device)

        # ERA5 boundary zarrs have NaN in the WRF interior — fill to 0.
        x_boundary = torch.nan_to_num(x_boundary, nan=self.boundary_fill_value)

        # --- time encoding ------------------------------------------------
        x_time_encode = batch["x_time_encode"].to(device)

        return x, x_boundary, x_time_encode

    def assemble_target(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Assemble target tensor from a batch dict.

        Args:
            batch: Dict produced by the WRF DataLoader.  Expected keys:
                ``y``, optionally ``y_surf``, ``y_diag``.
            device: Target device.

        Returns:
            Target tensor ``(B, C, T, H, W)``.
        """
        if "y_surf" in batch:
            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device)
        else:
            y = reshape_only(batch["y"]).to(device)

        if "y_diag" in batch:
            y_diag = batch["y_diag"].to(device).permute(0, 2, 1, 3, 4)
            y = torch.cat((y, y_diag), dim=1)

        return y
