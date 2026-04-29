"""
MSLPCalculator postblock
------------------------
Computes mean sea level pressure (MSLP) from surface pressure, near-surface
temperature, and static surface geopotential using the Trenberth et al. (1993)
formula.  Fully vectorized in PyTorch — no numpy, no per-pixel loops.

Reference:
    Trenberth, K., J. Berry, and L. Buja, 1993: Vertical Interpolation and
    Truncation of Model-Coordinate Data. NCAR Tech. Note NCAR/TN-396+STR.
    https://doi.org/10.5065/D6HX19NH

Bug fixed vs. the original numpy implementation in credit/interp.py (merged
in PR #341): the sea-level temperature branch test used `LAPSE_RATE * sgp`
where sgp is geopotential (m² s⁻²); it must be `LAPSE_RATE * sgp / GRAVITY`
to convert geopotential to height in metres.
"""

import logging

import torch
from torch import nn

from credit.data import get_forward_data
from credit.physics_constants import GRAVITY, RDGAS
from credit.transforms import load_transforms

logger = logging.getLogger(__name__)

# Trenberth 1993 standard atmosphere lapse rate
_LAPSE_RATE = 0.0065  # K m⁻¹
_ALPHA_STD = _LAPSE_RATE * RDGAS / GRAVITY  # dimensionless


def mslp_from_surface_pressure(
    surface_pressure: torch.Tensor,
    temperature: torch.Tensor,
    surface_geopotential: torch.Tensor,
) -> torch.Tensor:
    """Vectorized MSLP from surface pressure, near-surface T, and PHIS.

    Implements the simplified Trenberth et al. (1993) formula.  All inputs
    must be broadcastable to the same shape.

    Args:
        surface_pressure: surface pressure in Pa, shape (..., H, W).
        temperature: near-surface temperature in K, shape (..., H, W).
        surface_geopotential: PHIS in m² s⁻², shape (..., H, W).

    Returns:
        MSLP in Pa, same shape as inputs.
    """
    sgp = surface_geopotential
    height = sgp / GRAVITY  # surface height in metres

    near_flat = height.abs() < 1e-4  # essentially at sea level

    # sea-level temperature (uncorrected)
    tto = temperature + _LAPSE_RATE * height

    # --- effective lapse-rate alpha and temperature ---
    # case 1: cold surface but warm sea level  (T_surf <= 290.5, T_sl > 290.5)
    mask1 = (temperature <= 290.5) & (tto > 290.5)
    alpha_case1 = RDGAS * (290.5 - temperature) / sgp.clamp(min=1e-6)

    # case 2: warm surface  (T_surf > 290.5)
    mask2 = temperature > 290.5

    # case 3: very cold surface  (T_surf < 255), only outside cases 1 & 2
    mask3 = (temperature < 255) & ~mask1 & ~mask2

    # defaults
    alpha = torch.full_like(temperature, _ALPHA_STD)
    temp_eff = temperature.clone()

    # apply cases
    alpha = torch.where(mask1, alpha_case1, alpha)
    alpha = torch.where(mask2, torch.zeros_like(alpha), alpha)
    temp_eff = torch.where(mask2, 0.5 * (290.5 + temperature), temp_eff)
    temp_eff = torch.where(mask3, 0.5 * (255.0 + temperature), temp_eff)

    x = sgp / (RDGAS * temp_eff.clamp(min=1.0))
    mslp_computed = surface_pressure * torch.exp(x * (1.0 - 0.5 * alpha * x + (alpha * x) ** 2 / 3.0))

    return torch.where(near_flat, surface_pressure, mslp_computed)


class MSLPCalculator(nn.Module):
    """Postblock that computes MSLP and appends it to y_pred.

    Reads static surface geopotential (PHIS) from the physics file at init
    time and registers it as a non-trainable buffer so it moves with the
    module to whatever device is in use.

    Config keys under ``post_conf["mslp_calculator"]``:

    .. code-block:: yaml

        mslp_calculator:
            activate: true
            sp_ind: 69              # channel index of surface pressure in y_pred
            t2m_ind: 70             # channel index of near-surface T in y_pred
            surface_geopotential_name: "geopotential_at_surface"
            denorm: true            # y_pred is normalized; denorm before computing
            append_output: true     # append MSLP as a new channel in y_pred

    Args:
        post_conf (dict): full post_conf dict (same object passed to PostBlock).
    """

    def __init__(self, post_conf: dict):
        super().__init__()

        cfg = post_conf["mslp_calculator"]
        self.sp_ind = int(cfg["sp_ind"])
        self.t2m_ind = int(cfg["t2m_ind"])
        self.append_output = bool(cfg.get("append_output", True))

        # --- static surface geopotential ---
        phis_name = cfg.get("surface_geopotential_name", "geopotential_at_surface")
        ds_physics = get_forward_data(post_conf["data"]["save_loc_physics"])
        phis = torch.from_numpy(ds_physics[phis_name].values).float()
        # store as buffer: shape (H, W) — broadcast over batch/time in forward
        self.register_buffer("phis", phis)

        # --- optional denormalization ---
        if cfg.get("denorm", False):
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None

    def forward(self, x: dict) -> dict:
        """Compute MSLP and optionally append to y_pred.

        Args:
            x: postblock state dict with at minimum ``x["y_pred"]``
               of shape (batch, vars, time, lat, lon).

        Returns:
            Updated x dict.  If ``append_output`` is True, a new MSLP
            channel is appended along dim=1 of ``y_pred``.
            ``x["mslp"]`` is always set to the raw (Pa) MSLP tensor
            of shape (batch, 1, time, lat, lon).
        """
        y_pred = x["y_pred"]  # (batch, vars, time, lat, lon)

        if self.state_trans is not None:
            y_pred = self.state_trans.inverse_transform(y_pred)

        # extract SP and T2m — shape (batch, time, lat, lon)
        sp = y_pred[:, self.sp_ind, ...]
        t2m = y_pred[:, self.t2m_ind, ...]

        # broadcast PHIS over batch and time dims
        phis = self.phis.to(y_pred.device)  # (H, W)
        phis = phis.expand_as(sp)  # (batch, time, lat, lon)

        mslp = mslp_from_surface_pressure(sp, t2m, phis)  # (batch, time, lat, lon)

        # re-normalize y_pred before storing back
        if self.state_trans is not None:
            y_pred = self.state_trans.transform_array(y_pred)

        # store raw MSLP in the dict (always, for downstream use)
        x["mslp"] = mslp.unsqueeze(1)  # (batch, 1, time, lat, lon)

        if self.append_output:
            x["y_pred"] = torch.cat([y_pred, x["mslp"]], dim=1)
        else:
            x["y_pred"] = y_pred

        return x
