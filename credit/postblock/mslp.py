"""
MSLPDiagnostic postblock
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

import torch

from credit.physics_constants import GRAVITY, RDGAS
from credit.postblock.base import BasePostblock

# Trenberth 1993 standard atmosphere lapse rate
_LAPSE_RATE = 0.0065  # K m⁻¹
_ALPHA_STD = _LAPSE_RATE * RDGAS / GRAVITY  # dimensionless

# Trenberth 1993 temperature thresholds for lapse-rate regime selection
_T_WARM = 290.5  # K — warm-surface branch (alpha → 0, T_eff blended)
_T_COLD = 255.0  # K — very-cold-surface branch (T_eff blended toward 255 K)


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
    # case 1: cold surface but warm sea level  (T_surf <= _T_WARM, T_sl > _T_WARM)
    mask1 = (temperature <= _T_WARM) & (tto > _T_WARM)
    alpha_case1 = RDGAS * (_T_WARM - temperature) / sgp.clamp(min=1e-6)

    # case 2: warm surface  (T_surf > _T_WARM)
    mask2 = temperature > _T_WARM

    # case 3: very cold surface  (T_surf < _T_COLD), only outside cases 1 & 2
    mask3 = (temperature < _T_COLD) & ~mask1 & ~mask2

    alpha = torch.full_like(temperature, _ALPHA_STD)
    alpha = torch.where(mask1, alpha_case1, alpha)
    alpha = torch.where(mask2, torch.zeros_like(alpha), alpha)

    temp_eff = torch.where(mask2, 0.5 * (_T_WARM + temperature), temperature)
    temp_eff = torch.where(mask3, 0.5 * (_T_COLD + temperature), temp_eff)

    x = sgp / (RDGAS * temp_eff.clamp(min=1.0))
    mslp_computed = surface_pressure * torch.exp(x * (1.0 - 0.5 * alpha * x + (alpha * x) ** 2 / 3.0))

    return torch.where(near_flat, surface_pressure, mslp_computed)


class MSLPDiagnostic(BasePostblock):
    """Postblock that computes MSLP from surface pressure, 2m temperature, and PHIS.

    Follows the same batch-dict protocol as ``Reconstruct`` and
    ``BridgeScalerTransform``: it operates on the nested output dict at
    ``batch_dict[key]`` (default ``"y_processed"``), which has the form
    ``{source: {var_key: tensor}}`` where ``var_key`` is
    ``"source/field_type/dim/varname"``. The source for each variable is
    derived from the first path component of its ``var_key``. The static
    surface geopotential is read from ``batch_dict[static_source_key]``
    (default ``"ic_raw"``), which has the same nested form, since static
    fields are not part of the reconstructed model output. The result is
    written back into ``batch_dict[key]`` under ``output_name``.

    Config example::

        type: "mslp_diagnostic"
        args:
            output_name: "ARCO_ERA5/derived_diagnostic/2d/mean_sea_level_pressure"
            surface_pressure_var: "ARCO_ERA5/prognostic/2d/surface_pressure"
            temperature_var: "ARCO_ERA5/prognostic/2d/2m_temperature"
            surface_geopotential_var: "ARCO_ERA5/static/2d/geopotential_at_surface"

    Args:
        output_name: var_key written into ``batch_dict[key]`` for the result.
        surface_pressure_var: variable name for surface pressure (Pa).
        temperature_var: variable name for near-surface temperature (K).
        surface_geopotential_var: variable name for PHIS (m² s⁻²).
        key: entry in ``batch_dict`` holding the nested output dict written
            by ``Reconstruct`` (default: ``"y_processed"``).
        static_source_key: entry in ``batch_dict`` holding the nested raw IC
            dict that provides static fields (default: ``"ic_raw"``).
    """

    def __init__(
        self,
        output_name: str = "ARCO_ERA5/derived_diagnostic/2d/mean_sea_level_pressure",
        surface_pressure_var: str = "ARCO_ERA5/prognostic/2d/surface_pressure",
        temperature_var: str = "ARCO_ERA5/prognostic/2d/2m_temperature",
        surface_geopotential_var: str = "ARCO_ERA5/static/2d/geopotential_at_surface",
        key: str = "y_processed",
        static_source_key: str = "ic_raw",
    ):
        super().__init__()
        self.output_name = output_name
        self.surface_pressure_var = surface_pressure_var
        self.temperature_var = temperature_var
        self.surface_geopotential_var = surface_geopotential_var
        self.key = key
        self.static_source_key = static_source_key

    def forward(self, batch_dict: dict) -> dict:
        """Compute MSLP and write it into the nested output dict.

        Args:
            batch_dict: batch dict containing ``key`` and ``static_source_key``
                entries, each of the form ``{source: {var_key: tensor}}``.
                Prognostic inputs must have shapes broadcastable to
                ``(B, 1, n_time, H, W)``; PHIS may have ``n_time == 1``.

        Returns:
            The same ``batch_dict`` with ``output_name`` added under
            ``batch_dict[key][source]``.
        """
        for required_key in (self.key, self.static_source_key):
            if required_key not in batch_dict:
                raise ValueError(f"Key {required_key!r} not found in batch_dict.")
        nested = batch_dict[self.key]  # {source: {var_key: tensor}}
        static_nested = batch_dict[self.static_source_key]
        sp = nested[self.surface_pressure_var.split("/")[0]][self.surface_pressure_var]  # (B, 1, n_time, H, W)
        t2m = nested[self.temperature_var.split("/")[0]][self.temperature_var]  # (B, 1, n_time, H, W)
        phis = static_nested[self.surface_geopotential_var.split("/")[0]][
            self.surface_geopotential_var
        ]  # (B, 1, 1_or_n_time, H, W)
        out_source = self.output_name.split("/")[0]
        nested.setdefault(out_source, {})[self.output_name] = mslp_from_surface_pressure(sp, t2m, phis.to(sp.device))
        return batch_dict
