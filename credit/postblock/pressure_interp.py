"""
PressureInterpDiagnostic postblock
----------------------------------
Interpolates model state variables from hybrid sigma-pressure model levels to
constant pressure levels, fully in PyTorch. Follows the interpolation logic of
``credit/interp.py``:

- Interpolation is linear in log(pressure), performed independently for each
  column and parallelized across columns with ``torch.vmap``.
- For pressure levels below ground (interp pressure > surface pressure), wind
  and moisture variables use constant extrapolation (the lowest-model-level
  value, matching ``np.interp`` behavior in ``interp_hybrid_to_pressure_levels``).
- Temperature and geopotential use the Trenberth et al. (1993) extrapolation
  procedures (Eqs. 15 and 16) as implemented in CESM CAM and ECMWF OpenIFS:
  a surface temperature is estimated from the model level nearest 150 m AGL
  (the ECMWF standard), and an effective lapse rate is chosen based on the
  surface elevation, with special handling above 2000 m and 2500 m.

Reference:
    Trenberth, K., J. Berry, and L. Buja, 1993: Vertical Interpolation and
    Truncation of Model-Coordinate Data. NCAR Tech. Note NCAR/TN-396+STR.
    https://doi.org/10.5065/D6HX19NH
"""

from functools import partial

import numpy as np
import torch
import xarray as xr

from credit.metadata import get_meta_file_path
from credit.physics_constants import GRAVITY, RDGAS
from credit.postblock._interp_utils import loglinear_interp_columns
from credit.postblock.base import BasePostblock

# Trenberth 1993 standard atmosphere lapse rate
_LAPSE_RATE = 0.0065  # K m⁻¹
_ALPHA_STD = _LAPSE_RATE * RDGAS / GRAVITY  # dimensionless

# Trenberth 1993 cap on the sea-level temperature used for lapse-rate selection
_T_PLATEAU = 298.0  # K


def interp_column_to_pressure_levels(
    fields: torch.Tensor,
    temperature: torch.Tensor,
    geopotential: torch.Tensor,
    surface_pressure: torch.Tensor,
    surface_geopotential: torch.Tensor,
    model_a: torch.Tensor,
    model_b: torch.Tensor,
    interp_pressures: torch.Tensor,
    temp_height: float = 150.0,
) -> torch.Tensor:
    """Interpolate one column from hybrid sigma-pressure levels to constant pressure levels.

    All interpolation is linear in log(pressure). Model levels must be ordered
    top-of-atmosphere to surface (pressure increasing with index). ``fields``
    are constantly extrapolated outside the model pressure range; temperature
    and geopotential are extrapolated below ground following Trenberth et al.
    (1993) Eqs. 16 and 15 respectively, as in ``credit/interp.py``.

    Designed to be wrapped in ``torch.vmap`` over columns: every tensor
    argument except ``model_a``, ``model_b``, and ``interp_pressures`` carries
    a batch dimension in the vmapped call.

    Args:
        fields: variables using constant extrapolation (e.g. u, v, q), shape (n_fields, n_levels).
        temperature: temperature in K, shape (n_levels,).
        geopotential: geopotential in m² s⁻², shape (n_levels,).
        surface_pressure: surface pressure in Pa, scalar tensor.
        surface_geopotential: surface geopotential (PHIS) in m² s⁻², scalar tensor.
        model_a: hybrid `a` coefficient at level midpoints in Pa, shape (n_levels,).
        model_b: hybrid `b` coefficient at level midpoints (unitless), shape (n_levels,).
        interp_pressures: target pressure levels in Pa, shape (n_plev,).
        temp_height: height AGL in meters of the level used to estimate surface
            temperature for the below-ground extrapolation (ECMWF standard: 150 m).

    Returns:
        Tensor of shape (n_fields + 2, n_plev): the interpolated ``fields``
        followed by temperature and geopotential.
    """
    pressure = model_a + model_b * surface_pressure  # (n_levels,)
    log_p = torch.log(pressure)
    log_plev = torch.log(interp_pressures)

    stacked = torch.cat([fields, temperature.unsqueeze(0), geopotential.unsqueeze(0)], dim=0)
    interped = loglinear_interp_columns(stacked, log_p, log_plev)  # (n_fields + 2, n_plev)

    # --- Below-ground extrapolation for temperature and geopotential ---------
    below_ground = interp_pressures > surface_pressure

    # Surface temperature estimated from the model level nearest temp_height AGL,
    # brought adiabatically to the surface (Trenberth 1993).
    height_agl = (geopotential - surface_geopotential) / GRAVITY
    h = torch.argmin(torch.abs(height_agl - temp_height)).unsqueeze(0)
    temp_h = torch.gather(temperature, 0, h).squeeze(0)
    pres_h = torch.gather(pressure, 0, h).squeeze(0)
    temp_surface = temp_h + _ALPHA_STD * temp_h * (surface_pressure / pres_h - 1.0)

    surface_height = surface_geopotential / GRAVITY
    temp_sea_level = temp_surface + _LAPSE_RATE * surface_height
    temp_pl = torch.clamp(temp_sea_level, max=_T_PLATEAU)

    # Effective lapse rate gamma: standard below 2000 m; blended between 2000
    # and 2500 m; capped for high plateaus above 2500 m (CESM cpslec / OpenIFS).
    # sgp only divides in the >= 2000 m branches, where it is large; the clamp
    # keeps the untaken torch.where branch finite at sea level.
    sgp_safe = surface_geopotential.clamp(min=1.0)
    t_adjusted = 0.002 * ((2500.0 - surface_height) * temp_sea_level + (surface_height - 2000.0) * temp_pl)
    gamma = torch.where(
        surface_height > 2500.0,
        GRAVITY / sgp_safe * torch.clamp(temp_pl - temp_surface, min=0.0),
        torch.where(
            surface_height >= 2000.0,
            GRAVITY / sgp_safe * (t_adjusted - temp_surface),
            torch.full_like(temp_surface, _LAPSE_RATE),
        ),
    )

    ln_p_ps = torch.log(interp_pressures / surface_pressure)  # (n_plev,)
    a_ln_p = gamma * RDGAS / GRAVITY * ln_p_ps
    # Trenberth Eq. 16: T(p) = T* (1 + a + a²/2 + a³/6)
    temp_extrap = temp_surface * (1.0 + a_ln_p + 0.5 * a_ln_p**2 + a_ln_p**3 / 6.0)
    # Trenberth Eq. 15: Z(p) = PHIS - Rd T* ln(p/ps) (1 + a/2 + a²/6)
    geo_extrap = surface_geopotential - RDGAS * temp_surface * ln_p_ps * (1.0 + 0.5 * a_ln_p + a_ln_p**2 / 6.0)

    temp_out = torch.where(below_ground, temp_extrap, interped[-2])
    geo_out = torch.where(below_ground, geo_extrap, interped[-1])
    return torch.cat([interped[:-2], temp_out.unsqueeze(0), geo_out.unsqueeze(0)], dim=0)


class PressureInterpDiagnostic(BasePostblock):
    """Postblock that interpolates model-level variables to constant pressure levels.

    Follows the same batch-dict protocol as ``Reconstruct`` and
    ``BridgeScalerTransform``: it operates on the nested output dict at
    ``batch_dict[key]`` (default ``"y_processed"``), which has the form
    ``{source: {var_key: tensor}}``. The source for each variable is derived
    from the first path component of its ``var_key``. The static surface
    geopotential is read from ``batch_dict[static_source_key]`` (default
    ``"ic_raw"``). One output is written back into ``batch_dict[key]`` per
    interpolated variable, named ``{source}/derived_diagnostic/{dim}/{varname}{output_suffix}``
    with shape ``(B, n_plev, n_time, H, W)``.

    ``geopotential_var`` must already be present in ``batch_dict[key]`` — chain
    a ``geopotential_diagnostic`` postblock before this one to derive it.

    Interpolation runs on the model's native (possibly reduced) level set,
    unlike ``credit/interp.py`` which first interpolates reduced levels to the
    full 137-level ERA5 grid. Level ordering (top→surface vs. surface→top) is
    detected automatically from the hybrid coefficients and flipped if needed.

    Config example::

        type: "pressure_interp_diagnostic"
        args:
            pressure_levels: [250.0, 500.0, 850.0]   # hPa
            interp_variables:
                - "ARCO_ERA5/prognostic/3d/u_component_of_wind"
                - "ARCO_ERA5/prognostic/3d/v_component_of_wind"
                - "ARCO_ERA5/prognostic/3d/specific_humidity"

    Args:
        pressure_levels: target pressure levels in hPa.
        interp_variables: 3D var_keys interpolated with constant below-ground
            extrapolation (winds, moisture). Temperature and geopotential are
            always processed with their special extrapolation and should not
            be listed here.
        temperature_var: variable name for temperature (K).
        geopotential_var: variable name for geopotential on model levels (m² s⁻²).
        surface_pressure_var: variable name for surface pressure (Pa).
        surface_geopotential_var: variable name for PHIS (m² s⁻²).
        output_suffix: appended to each output variable name (default ``"_PRES"``).
        temp_height: height AGL in meters of the level used to estimate surface
            temperature for below-ground extrapolation (default 150 m).
        chunk_size: vmap chunk size to bound memory usage.
        level_info_file: metadata file with hybrid level coefficients.
        model_a_var: name of the `a` coefficient at level midpoints (Pa).
        model_b_var: name of the `b` coefficient at level midpoints (unitless).
        key: entry in ``batch_dict`` holding the nested output dict written
            by ``Reconstruct`` (default: ``"y_processed"``).
        static_source_key: entry in ``batch_dict`` holding the nested raw IC
            dict that provides static fields (default: ``"ic_raw"``).
        levels: subset of 1-based model level numbers matching the data; None
            uses every level in ``level_info_file``.
    """

    def __init__(
        self,
        pressure_levels: list[float] = (500.0, 850.0),
        interp_variables: list[str] = (
            "ARCO_ERA5/prognostic/3d/u_component_of_wind",
            "ARCO_ERA5/prognostic/3d/v_component_of_wind",
            "ARCO_ERA5/prognostic/3d/specific_humidity",
        ),
        temperature_var: str = "ARCO_ERA5/prognostic/3d/temperature",
        geopotential_var: str = "ARCO_ERA5/derived_diagnostic/3d/geopotential",
        surface_pressure_var: str = "ARCO_ERA5/prognostic/2d/surface_pressure",
        surface_geopotential_var: str = "ARCO_ERA5/static/2d/geopotential_at_surface",
        output_suffix: str = "_PRES",
        temp_height: float = 150.0,
        chunk_size: int = 1000,
        level_info_file: str = "ERA5_Lev_Info.nc",
        model_a_var: str = "a_model",
        model_b_var: str = "b_model",
        key: str = "y_processed",
        static_source_key: str = "ic_raw",
        levels: list[int] | None = None,
    ):
        super().__init__()
        self.interp_variables = list(interp_variables)
        self.temperature_var = temperature_var
        self.geopotential_var = geopotential_var
        self.surface_pressure_var = surface_pressure_var
        self.surface_geopotential_var = surface_geopotential_var
        self.output_suffix = output_suffix
        self.temp_height = temp_height
        self.chunk_size = chunk_size
        self.level_info_file = get_meta_file_path(level_info_file)
        self.key = key
        self.static_source_key = static_source_key
        self.levels = levels
        self.pressure_levels_pa = torch.tensor([p * 100.0 for p in pressure_levels])  # hPa → Pa
        with xr.open_dataset(self.level_info_file) as level_info:
            a_all = torch.Tensor(level_info[model_a_var].values)
            b_all = torch.Tensor(level_info[model_b_var].values)
        if levels is not None:
            idx = [lv - 1 for lv in levels]
            self.model_a = a_all[idx]
            self.model_b = b_all[idx]
        else:
            self.model_a = a_all
            self.model_b = b_all
        # Interpolation needs pressure increasing with level index (top→surface);
        # detect orientation once from the coefficients at a reference surface pressure.
        ref_pressure = self.model_a + self.model_b * 101325.0
        self._flip = bool(ref_pressure[0] > ref_pressure[-1])
        if self._flip:
            self.model_a = torch.flip(self.model_a, (0,))
            self.model_b = torch.flip(self.model_b, (0,))

    def _output_key(self, var_key: str) -> str:
        parts = var_key.split("/")  # "source/field_type/dim/varname"
        return f"{parts[0]}/derived_diagnostic/{parts[2]}/{parts[3]}{self.output_suffix}"

    @staticmethod
    def _flatten_columns(tensor: torch.Tensor) -> torch.Tensor:
        """(B, n_levels, n_time, H, W) → (B * n_time * H * W, n_levels)."""
        per = torch.permute(tensor, (0, 2, 3, 4, 1))  # (B, n_time, H, W, n_levels)
        return per.reshape(int(np.prod(per.shape[:-1])), per.shape[-1])

    def forward(self, batch_dict: dict) -> dict:
        """Interpolate the configured variables to pressure levels and write them into the batch dict.

        Args:
            batch_dict: batch dict containing ``key`` and ``static_source_key``
                entries, each of the form ``{source: {var_key: tensor}}``. 3D
                variables have shape ``(B, n_levels, n_time, H, W)``; surface
                pressure ``(B, 1, n_time, H, W)``; PHIS may have ``n_time == 1``.

        Returns:
            The same ``batch_dict`` with one ``(B, n_plev, n_time, H, W)``
            output per interpolated variable added under ``batch_dict[key][source]``.

        Raises:
            ValueError: if ``key`` or ``static_source_key`` is not found in ``batch_dict``.
        """
        for required_key in (self.key, self.static_source_key):
            if required_key not in batch_dict:
                raise ValueError(f"Key {required_key!r} not found in batch_dict.")
        nested = batch_dict[self.key]  # {source: {var_key: tensor}}
        static_nested = batch_dict[self.static_source_key]

        def lookup(var_key: str, from_static: bool = False) -> torch.Tensor:
            src = static_nested if from_static else nested
            return src[var_key.split("/")[0]][var_key]

        temperature = lookup(self.temperature_var)  # (B, n_levels, n_time, H, W)
        geopotential = lookup(self.geopotential_var)
        surface_pressure = lookup(self.surface_pressure_var)  # (B, 1, n_time, H, W)
        phis = lookup(self.surface_geopotential_var, from_static=True)  # (B, 1, 1_or_n_time, H, W)

        batch, _, n_time, height, width = temperature.shape
        device = surface_pressure.device
        dtype = surface_pressure.dtype

        if self._flip:
            temperature = torch.flip(temperature, (1,))
            geopotential = torch.flip(geopotential, (1,))
        temp_flat = self._flatten_columns(temperature)  # (N, n_levels)
        geo_flat = self._flatten_columns(geopotential.to(device))
        sp_flat = self._flatten_columns(surface_pressure).squeeze(-1)  # (N,)
        phis = phis.expand(batch, 1, n_time, height, width).to(device)
        phis_flat = self._flatten_columns(phis).squeeze(-1)  # (N,)

        if self.interp_variables:
            field_tensors = [lookup(v) for v in self.interp_variables]
            if self._flip:
                field_tensors = [torch.flip(f, (1,)) for f in field_tensors]
            fields_flat = torch.stack(
                [self._flatten_columns(f.to(device)) for f in field_tensors], dim=1
            )  # (N, n_fields, n_levels)
        else:
            fields_flat = temp_flat.new_zeros((temp_flat.shape[0], 0, temp_flat.shape[1]))

        vinterp = torch.vmap(
            partial(interp_column_to_pressure_levels, temp_height=self.temp_height),
            in_dims=(0, 0, 0, 0, 0, None, None, None),
            chunk_size=self.chunk_size,
        )
        interped = vinterp(
            fields_flat,
            temp_flat,
            geo_flat,
            sp_flat,
            phis_flat,
            self.model_a.to(device=device, dtype=dtype),
            self.model_b.to(device=device, dtype=dtype),
            self.pressure_levels_pa.to(device=device, dtype=dtype),
        )  # (N, n_fields + 2, n_plev)

        n_plev = self.pressure_levels_pa.shape[0]
        interped = interped.reshape(batch, n_time, height, width, -1, n_plev)
        out_variables = self.interp_variables + [self.temperature_var, self.geopotential_var]
        for k, var_key in enumerate(out_variables):
            out_source = var_key.split("/")[0]
            # (B, n_time, H, W, n_plev) → (B, n_plev, n_time, H, W)
            nested.setdefault(out_source, {})[self._output_key(var_key)] = torch.permute(
                interped[..., k, :], (0, 4, 1, 2, 3)
            )
        return batch_dict
