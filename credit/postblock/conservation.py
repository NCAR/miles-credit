"""
conservation.py
---------------
Gen2 (name-based) physics conservation postblocks.

These mirror the gen1 fixers in ``credit/postblock/gen1.py`` but follow the
gen2 flagship convention established by ``geopotential.py`` / ``mslp.py``:

  * Variables are addressed by **name** from the nested super-dict
    (``batch_dict["y_processed"][source][var_key]``), never by channel index.
  * The blocks do **no** scaling.  They expect physical units in and out, so
    they run *after* the inverse ``bridgescaler`` and *before* the forward
    ``bridgescaler`` + ``FlattenToTensor`` that rebuilds ``y_pred`` for the
    loss.  (Scaling is the bridgescaler's job, not the fixer's.)
  * The previous-step / input state is read by name from the per-step physical
    input dict (default ``batch_dict["x_physical"]``): the named, physical input
    the model actually consumed this step. For ``forecast_len=1`` this equals the
    initial condition; for multistep it is the previous step's predicted physical
    state plus the current step's forcing, which is exactly the t0 state the
    conservation budgets need.

Gradient note: when ``Reconstruct`` is built with ``detach=False`` the
``y_processed`` tensors stay attached to ``y_pred``'s graph, so the corrections
these blocks apply propagate into the training loss.

Only the hybrid-sigma + midpoint path used by CAMulator is exercised here; the
pressure-level branch is supported for parity with gen1.
"""

import os
import torch
from torch import nn
from credit.data import get_forward_data
from credit.physics_core import physics_pressure_level, physics_hybrid_sigma_level
from credit.physics_constants import (
    GRAVITY,
    RHO_WATER,
    LH_WATER,
    CP_DRY,
    CP_VAPOR,
)


def _setup_physics_core(args: dict):
    """Build the vertical-integration core from a statics file.

    Reuses the same grid/coefficient setup as the gen1 fixers, driven by
    name-based ``args`` instead of an ``post_conf`` sub-dict.

    Returns a tuple ``(core, flag_sigma, midpoint, n_levels, coef_a, coef_b)``
    where ``coef_a`` / ``coef_b`` are ``None`` for the pressure-level grid.
    """
    ds_physics = get_forward_data(os.path.expandvars(args["save_loc_physics"]))
    lon_lat_level_name = args["lon_lat_level_name"]
    lon2d = torch.from_numpy(ds_physics[lon_lat_level_name[0]].values).float()
    lat2d = torch.from_numpy(ds_physics[lon_lat_level_name[1]].values).float()
    midpoint = bool(args.get("midpoint", False))

    if args.get("grid_type", "sigma") == "sigma":
        coef_a = torch.from_numpy(ds_physics[lon_lat_level_name[2]].values).float()
        coef_b = torch.from_numpy(ds_physics[lon_lat_level_name[3]].values).float()
        n_levels = len(coef_a) - 1 if midpoint else len(coef_a)
        core = physics_hybrid_sigma_level(lon2d, lat2d, coef_a, coef_b, midpoint=midpoint)
        return core, True, midpoint, n_levels, coef_a, coef_b

    p_level = torch.from_numpy(ds_physics[lon_lat_level_name[2]].values).float()
    core = physics_pressure_level(lon2d, lat2d, p_level, midpoint=midpoint)
    return core, False, midpoint, len(p_level), None, None


def _src(var_key: str) -> str:
    """Source (top-level super-dict key) is the first path segment of a var key."""
    return var_key.split("/")[0]


def _pred(batch_dict: dict, var_key: str) -> torch.Tensor:
    return batch_dict["y_processed"][_src(var_key)][var_key]


def _set_pred(batch_dict: dict, var_key: str, tensor: torch.Tensor) -> None:
    batch_dict["y_processed"][_src(var_key)][var_key] = tensor


class TracerFixer(nn.Module):
    """Clamp tracer fields to a lower (and optional upper) threshold by name.

    Args:
        tracer_vars: list of variable keys in ``y_processed`` to clamp.
        tracer_thres: lower threshold(s).  Scalar applied to all vars, or a
            per-variable list aligned with ``tracer_vars``.
        tracer_thres_max: optional upper threshold(s), same shape rule.
    """

    def __init__(self, tracer_vars, tracer_thres, tracer_thres_max=None):
        super().__init__()
        self.tracer_vars = list(tracer_vars)
        n = len(self.tracer_vars)
        self.tracer_thres = tracer_thres if isinstance(tracer_thres, (list, tuple)) else [tracer_thres] * n
        if tracer_thres_max is None:
            self.tracer_thres_max = [None] * n
        elif isinstance(tracer_thres_max, (list, tuple)):
            self.tracer_thres_max = list(tracer_thres_max)
        else:
            self.tracer_thres_max = [tracer_thres_max] * n

    def forward(self, batch_dict: dict) -> dict:
        for var_key, lo, hi in zip(self.tracer_vars, self.tracer_thres, self.tracer_thres_max):
            t = _pred(batch_dict, var_key)
            # out-of-place clamp: y_processed tensors are on the autograd graph
            t = torch.clamp(t, min=lo)
            if hi is not None:
                t = torch.clamp(t, max=hi)
            _set_pred(batch_dict, var_key, t)
        return batch_dict


class GlobalMassFixer(nn.Module):
    """Conserve global dry-air mass by correcting surface pressure (sigma grid).

    Mirrors ``gen1.GlobalMassFixer`` for the hybrid-sigma path: the dry-air
    mass at t0 (from the input state) sets the target, and surface pressure at
    t1 is rescaled so the predicted dry-air mass matches.

    Args:
        q_var: specific total water variable key.
        sp_var: surface pressure variable key.
        plus the physics-core setup keys (``save_loc_physics``,
        ``lon_lat_level_name``, ``grid_type``, ``midpoint``).
    """

    def __init__(self, q_var, sp_var, input_source_key="x_physical", **physics):
        super().__init__()
        self.q_var = q_var
        self.sp_var = sp_var
        self.input_source_key = input_source_key
        self.core, self.flag_sigma, self.midpoint, self.N_levels, self.coef_a, self.coef_b = _setup_physics_core(
            physics
        )

    def _input(self, batch_dict, var_key):
        return batch_dict[self.input_source_key][_src(var_key)][var_key]

    def forward(self, batch_dict: dict) -> dict:
        # prediction (t1): (B, L, T, H, W) -> (B, L, H, W); 2d vars squeeze level too
        q_pred = _pred(batch_dict, self.q_var)[:, :, 0, ...]
        sp_pred = _pred(batch_dict, self.sp_var)[:, 0, 0, ...]
        # input (t0): pick last input frame
        q_input = self._input(batch_dict, self.q_var)[:, :, -1, ...]
        sp_input = self._input(batch_dict, self.sp_var)[:, 0, -1, ...]
        device = q_pred.device

        if not self.flag_sigma:
            raise NotImplementedError("GlobalMassFixer (gen2) currently supports the hybrid-sigma grid only.")

        mass_dry_sum_t0 = self.core.total_dry_air_mass(q_input.to(device), sp_input.to(device))

        delta_coef_a = self.coef_a.diff().to(device)
        delta_coef_b = self.coef_b.diff().to(device)
        if self.midpoint:
            p_dry_a = ((delta_coef_a.view(1, -1, 1, 1)) * (1 - q_pred)).sum(1)
            p_dry_b = ((delta_coef_b.view(1, -1, 1, 1)) * (1 - q_pred)).sum(1)
        else:
            q_mid = (q_pred[:, :-1, ...] + q_pred[:, 1:, ...]) / 2
            p_dry_a = ((delta_coef_a.view(1, -1, 1, 1)) * (1 - q_mid)).sum(1)
            p_dry_b = ((delta_coef_b.view(1, -1, 1, 1)) * (1 - q_mid)).sum(1)

        grid_area = self.core.area.unsqueeze(0).to(device)
        mass_dry_a = (p_dry_a * grid_area).sum((-2, -1)) / GRAVITY
        mass_dry_b = (p_dry_b * sp_pred * grid_area).sum((-2, -1)) / GRAVITY

        sp_correct_ratio = (mass_dry_sum_t0 - mass_dry_a) / mass_dry_b
        sp_pred = sp_pred * sp_correct_ratio.unsqueeze(1).unsqueeze(2)

        # back to (B, 1, 1, H, W)
        _set_pred(batch_dict, self.sp_var, sp_pred.unsqueeze(1).unsqueeze(2))
        return batch_dict


class GlobalWaterFixer(nn.Module):
    """Close the global water budget by correcting precipitation.

    Mirrors ``gen1.GlobalWaterFixer``: the column-water tendency plus
    evaporation must balance precipitation; precip is rescaled to enforce it.

    Args:
        q_var, sp_var, precip_var, evapor_var: variable keys.
        lead_time_periods: forecast step length in hours (sets ``N_seconds``).
        plus the physics-core setup keys.
    """

    def __init__(
        self, q_var, sp_var, precip_var, evapor_var, lead_time_periods, input_source_key="x_physical", **physics
    ):
        super().__init__()
        self.q_var = q_var
        self.sp_var = sp_var
        self.precip_var = precip_var
        self.evapor_var = evapor_var
        self.N_seconds = int(lead_time_periods) * 3600
        self.input_source_key = input_source_key
        self.core, self.flag_sigma, self.midpoint, self.N_levels, self.coef_a, self.coef_b = _setup_physics_core(
            physics
        )

    def _input(self, batch_dict, var_key):
        return batch_dict[self.input_source_key][_src(var_key)][var_key]

    def forward(self, batch_dict: dict) -> dict:
        q_pred = _pred(batch_dict, self.q_var)[:, :, 0, ...]
        sp_pred = _pred(batch_dict, self.sp_var)[:, 0, 0, ...]
        precip = _pred(batch_dict, self.precip_var)[:, 0, 0, ...]
        evapor = _pred(batch_dict, self.evapor_var)[:, 0, 0, ...]
        q_input = self._input(batch_dict, self.q_var)[:, :, -1, ...]
        sp_input = self._input(batch_dict, self.sp_var)[:, 0, -1, ...]
        device = q_pred.device

        if not self.flag_sigma:
            raise NotImplementedError("GlobalWaterFixer (gen2) currently supports the hybrid-sigma grid only.")

        precip_flux = precip * RHO_WATER / self.N_seconds
        evapor_flux = evapor * RHO_WATER / self.N_seconds

        TWC_input = self.core.total_column_water(q_input.to(device), sp_input.to(device))
        TWC_pred = self.core.total_column_water(q_pred, sp_pred)
        dTWC_dt = (TWC_pred - TWC_input) / self.N_seconds

        TWC_sum = self.core.weighted_sum(dTWC_dt, axis=(-2, -1))
        E_sum = self.core.weighted_sum(evapor_flux, axis=(-2, -1))
        P_sum = self.core.weighted_sum(precip_flux, axis=(-2, -1))

        residual = -TWC_sum - E_sum - P_sum
        P_correct_ratio = (P_sum + residual) / P_sum
        precip = precip * P_correct_ratio.unsqueeze(-1).unsqueeze(-1)

        _set_pred(batch_dict, self.precip_var, precip.unsqueeze(1).unsqueeze(2))
        return batch_dict


class GlobalEnergyFixerUpDown(nn.Module):
    """Conserve global total energy using explicit up/down flux decomposition.

    Mirrors ``gen1.GlobalEnergyFixerUpDown``: the column total-energy tendency
    is forced to match the net TOA + surface energy fluxes, and temperature
    carries the correction.

    The TOA downwelling shortwave (SOLIN) is an input-only forcing absent from
    the prediction, so it is read from the input dict by name
    (``toa_down_solar_input_var``).

    Args:
        T_var, q_var, U_var, V_var, sp_var: atmosphere state keys.
        surface_geopotential_name: variable name of PHIS in the statics file.
        toa_down_solar_input_var: SOLIN key in the input dict.
        toa_up_solar_var, toa_up_olr_var: TOA upwelling flux keys.
        surf_down_solar_var, surf_up_solar_var, surf_down_lw_var,
        surf_up_lw_var, surf_sh_var, surf_lh_var: surface flux keys.
        lead_time_periods: forecast step length in hours.
        plus the physics-core setup keys.
    """

    def __init__(
        self,
        T_var,
        q_var,
        U_var,
        V_var,
        sp_var,
        surface_geopotential_name,
        toa_down_solar_input_var,
        toa_up_solar_var,
        toa_up_olr_var,
        surf_down_solar_var,
        surf_up_solar_var,
        surf_down_lw_var,
        surf_up_lw_var,
        surf_sh_var,
        surf_lh_var,
        lead_time_periods,
        input_source_key="x_physical",
        **physics,
    ):
        super().__init__()
        self.T_var = T_var
        self.q_var = q_var
        self.U_var = U_var
        self.V_var = V_var
        self.sp_var = sp_var
        self.toa_down_solar_input_var = toa_down_solar_input_var
        self.toa_up_solar_var = toa_up_solar_var
        self.toa_up_olr_var = toa_up_olr_var
        self.surf_down_solar_var = surf_down_solar_var
        self.surf_up_solar_var = surf_up_solar_var
        self.surf_down_lw_var = surf_down_lw_var
        self.surf_up_lw_var = surf_up_lw_var
        self.surf_sh_var = surf_sh_var
        self.surf_lh_var = surf_lh_var
        self.N_seconds = int(lead_time_periods) * 3600
        self.input_source_key = input_source_key
        self.core, self.flag_sigma, self.midpoint, self.N_levels, self.coef_a, self.coef_b = _setup_physics_core(
            physics
        )

        ds_physics = get_forward_data(os.path.expandvars(physics["save_loc_physics"]))
        gph_name = (
            surface_geopotential_name[0]
            if isinstance(surface_geopotential_name, (list, tuple))
            else (surface_geopotential_name)
        )
        self.GPH_surf = torch.from_numpy(ds_physics[gph_name].values).float()

    def _input(self, batch_dict, var_key):
        return batch_dict[self.input_source_key][_src(var_key)][var_key]

    def forward(self, batch_dict: dict) -> dict:
        T_pred = _pred(batch_dict, self.T_var)[:, :, 0, ...]
        q_pred = _pred(batch_dict, self.q_var)[:, :, 0, ...]
        U_pred = _pred(batch_dict, self.U_var)[:, :, 0, ...]
        V_pred = _pred(batch_dict, self.V_var)[:, :, 0, ...]
        sp_pred = _pred(batch_dict, self.sp_var)[:, 0, 0, ...]
        device = T_pred.device

        T_input = self._input(batch_dict, self.T_var)[:, :, -1, ...].to(device)
        q_input = self._input(batch_dict, self.q_var)[:, :, -1, ...].to(device)
        U_input = self._input(batch_dict, self.U_var)[:, :, -1, ...].to(device)
        V_input = self._input(batch_dict, self.V_var)[:, :, -1, ...].to(device)
        sp_input = self._input(batch_dict, self.sp_var)[:, 0, -1, ...].to(device)

        if not self.flag_sigma:
            raise NotImplementedError("GlobalEnergyFixerUpDown (gen2) currently supports the hybrid-sigma grid only.")

        GPH_surf = self.GPH_surf.to(device)

        CP_t0 = (1 - q_input) * CP_DRY + q_input * CP_VAPOR
        CP_t1 = (1 - q_pred) * CP_DRY + q_pred * CP_VAPOR

        ken_t0 = 0.5 * (U_input**2 + V_input**2)
        ken_t1 = 0.5 * (U_pred**2 + V_pred**2)

        E_qgk_t0 = LH_WATER * q_input + GPH_surf + ken_t0
        E_qgk_t1 = LH_WATER * q_pred + GPH_surf + ken_t1

        TOA_down_solar = (
            self._input(batch_dict, self.toa_down_solar_input_var)[:, 0, -1, ...].to(device) * self.N_seconds
        )
        TOA_up_solar = _pred(batch_dict, self.toa_up_solar_var)[:, 0, 0, ...] * self.N_seconds
        TOA_up_OLR = _pred(batch_dict, self.toa_up_olr_var)[:, 0, 0, ...] * self.N_seconds

        R_T = (TOA_down_solar - TOA_up_solar - TOA_up_OLR) / self.N_seconds
        R_T_sum = self.core.weighted_sum(R_T, axis=(-2, -1))

        surf_down_solar = _pred(batch_dict, self.surf_down_solar_var)[:, 0, 0, ...]
        surf_up_solar = _pred(batch_dict, self.surf_up_solar_var)[:, 0, 0, ...]
        surf_down_LW = _pred(batch_dict, self.surf_down_lw_var)[:, 0, 0, ...]
        surf_up_LW = _pred(batch_dict, self.surf_up_lw_var)[:, 0, 0, ...]
        surf_SH = _pred(batch_dict, self.surf_sh_var)[:, 0, 0, ...]
        surf_LH = _pred(batch_dict, self.surf_lh_var)[:, 0, 0, ...]
        F_S = (surf_down_solar - surf_up_solar + surf_down_LW - surf_up_LW + surf_SH + surf_LH) / self.N_seconds
        F_S_sum = self.core.weighted_sum(F_S, axis=(-2, -1))

        E_level_t0 = CP_t0 * T_input + E_qgk_t0
        E_level_t1 = CP_t1 * T_pred + E_qgk_t1

        TE_t0 = self.core.integral(E_level_t0, sp_input) / GRAVITY
        TE_t1 = self.core.integral(E_level_t1, sp_pred) / GRAVITY
        global_TE_t0 = self.core.weighted_sum(TE_t0, axis=(-2, -1))
        global_TE_t1 = self.core.weighted_sum(TE_t1, axis=(-2, -1))

        E_correct_ratio = (self.N_seconds * (R_T_sum - F_S_sum) + global_TE_t0) / global_TE_t1
        E_correct_ratio = E_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        E_t1_correct = E_level_t1 * E_correct_ratio
        T_pred = (E_t1_correct - E_qgk_t1) / CP_t1

        # back to (B, L, 1, H, W)
        _set_pred(batch_dict, self.T_var, T_pred.unsqueeze(2))
        return batch_dict
