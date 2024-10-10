"""
postblock.py
-------------------------------------------------------
Content:
    - PostBlock
    - TracerFixer
    - GlobalMassFixer
    - GlobalEnergyFixer
    - SKEBS

"""

import torch
from torch import nn
import torch_harmonics as harmonics
import numpy as np
import xarray as xr

from credit.data import get_forward_data
from torch.nn.parameter import Parameter
from credit.transforms import load_transforms
from credit.physics_core import compute_pressure_on_mlevs, physics_pressure_level
from credit.physics_constants import PI, RAD_EARTH, GRAVITY, RHO_WATER, LH_WATER, CP_DRY, CP_VAPOR

import logging

logger = logging.getLogger(__name__)


class PostBlock(nn.Module):
    def __init__(self, post_conf):
        """
        post_conf: dictionary with config options for PostBlock.
                   if post_conf is not specified in config,
                   defaults are set in the parser

        This class is a wrapper for all post-model operations.
        Registered modules:
            - SKEBS
            - TracerFixer
            - GlobalMassFixer
            - GlobalEnergyFixer

        """
        super().__init__()

        self.operations = nn.ModuleList()

        # The general order of postblock processes:
        # (1) negative tracer fixer --> global mass fixer --> SKEB --> global energy fixer

        # negative tracer fixer
        if post_conf["tracer_fixer"]["activate"]:
            logger.info("TracerFixer registered")
            opt = TracerFixer(post_conf)
            self.operations.append(opt)

        # stochastic kinetic energy backscattering (SKEB)
        if post_conf["skebs"]["activate"]:
            logging.info("using SKEBS")
            self.operations.append(SKEBS(post_conf))

        # global mass fixer
        if post_conf["global_mass_fixer"]["activate"]:
            logger.info("GlobalMassFixer registered")
            opt = GlobalMassFixer(post_conf)
            self.operations.append(opt)

        # global energy fixer
        if post_conf["global_energy_fixer"]["activate"]:
            logger.info("GlobalEnergyFixer registered")
            opt = GlobalEnergyFixer(post_conf)
            self.operations.append(opt)

    def forward(self, x):
        for op in self.operations:
            x = op(x)

        if isinstance(x, dict):
            # if output is a dict, return y_pred (if it exists), otherwise return x
            return x.get("y_pred", x)
        else:
            # if output is not a dict (assuming tensor), return x
            return x


class TracerFixer(nn.Module):
    """
    This module fixes tracer values by replacing their values to a given threshold
    (e.g., `tracer[tracer<thres] = thres`).

    Args:
        post_conf (dict): config dictionary that includes all specs for the tracer fixer.
    """

    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------ #
        # identify variables of interest
        self.tracer_indices = post_conf["tracer_fixer"]["tracer_inds"]
        self.tracer_thres = post_conf["tracer_fixer"]["tracer_thres"]

        # ------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf["tracer_fixer"]["denorm"]:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None

    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get y_pred
        # y_pred is channel first: (batch, var, time, lat, lon)
        y_pred = x["y_pred"]

        # if denorm is needed
        if self.state_trans:
            y_pred = self.state_trans.inverse_transform(y_pred)

        # ------------------------------------------------------------------------------ #
        # tracer correction
        for i, i_var in enumerate(self.tracer_indices):
            # get the tracers
            tracer_vals = y_pred[:, i_var, ...]

            # in-place modification of y_pred
            thres = self.tracer_thres[i]
            tracer_vals[tracer_vals < thres] = thres

        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


class GlobalMassFixer(nn.Module):
    """
    This module applies global mass conservation fixes for both dry air and water budget.
    The output ensures that the global dry air mass and global water budgets are conserved
    through correction ratios applied during model runs. Variables `specific total water`
    and `precipitation` will be corrected to close the budget. All corrections are done
    using float32 PyTorch tensors.

    Args:
        post_conf (dict): config dictionary that includes all specs for the global mass fixer.
    """

    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf["global_mass_fixer"]["simple_demo"]:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340])

            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.core_compute = physics_pressure_level(
                lon_demo, lat_demo, p_level_demo, midpoint=post_conf["global_mass_fixer"]["midpoint"]
            )
            self.N_levels = len(p_level_demo)
            self.N_seconds = int(post_conf["data"]["lead_time_periods"]) * 3600
            self.ind_fix = len(p_level_demo) - int(post_conf["global_mass_fixer"]["fix_level_num"]) + 1

        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf["data"]["save_loc_physics"])

            lon_lat_level_names = post_conf["global_mass_fixer"]["lon_lat_level_name"]
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()
            p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()

            self.core_compute = physics_pressure_level(
                lon2d, lat2d, p_level, midpoint=post_conf["global_mass_fixer"]["midpoint"]
            )
            self.N_levels = len(p_level)
            self.N_seconds = int(post_conf["data"]["lead_time_periods"]) * 3600
            self.ind_fix = len(p_level) - int(post_conf["global_mass_fixer"]["fix_level_num"]) + 1

        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.q_ind_start = int(post_conf["global_mass_fixer"]["q_inds"][0])
        self.q_ind_end = int(post_conf["global_mass_fixer"]["q_inds"][-1]) + 1
        self.precip_ind = int(post_conf["global_mass_fixer"]["precip_ind"])
        self.evapor_ind = int(post_conf["global_mass_fixer"]["evapor_ind"])
        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf["global_mass_fixer"]["denorm"]:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None

    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        # x_input does not have precip and evapor
        x_input = x["x"]
        y_pred = x["y_pred"]

        # detach x_input
        x_input = x_input.detach().to(y_pred.device)

        # other needed inputs
        N_vars = y_pred.shape[1]

        # if denorm is needed
        if self.state_trans:
            x_input = self.state_trans.inverse_transform_input(x_input)
            y_pred = self.state_trans.inverse_transform(y_pred)

        q_input = x_input[:, self.q_ind_start : self.q_ind_end, -1, ...]

        # y_pred (batch, var, time, lat, lon)
        # pick the first time-step, y_pred is expected to have the next step only
        q_pred = y_pred[:, self.q_ind_start : self.q_ind_end, 0, ...]
        precip = y_pred[:, self.precip_ind, 0, ...]
        evapor = y_pred[:, self.evapor_ind, 0, ...]

        # ------------------------------------------------------------------------------ #
        # global dry air mass conservation

        # total mass from q_input
        mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input)

        # total mass from q_pred
        mass_dry_sum_t1_hold = self.core_compute.weighted_sum(
            self.core_compute.integral_sliced(1 - q_pred, 0, self.ind_fix) / GRAVITY, axis=(-2, -1)
        )

        mass_dry_sum_t1_fix = self.core_compute.weighted_sum(
            self.core_compute.integral_sliced(1 - q_pred, self.ind_fix - 1, self.N_levels) / GRAVITY, axis=(-2, -1)
        )

        q_correct_ratio = (mass_dry_sum_t0 - mass_dry_sum_t1_hold) / mass_dry_sum_t1_fix
        q_correct_ratio = torch.clamp(q_correct_ratio, min=0.9, max=1.1)

        # broadcast: (batch, 1, 1, 1, 1)
        q_correct_ratio = q_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # ===================================================================== #
        # q fixes based on the ratio
        # fix lower atmosphere
        q_pred_fix = 1 - (1 - q_pred[:, self.ind_fix - 1 :, ...]) * q_correct_ratio
        # extract unmodified part from q_pred
        q_pred_hold = q_pred[:, : self.ind_fix - 1, ...]

        # concat upper and lower q vals
        # (batch, level, lat, lon)
        q_pred = torch.cat([q_pred_hold, q_pred_fix], dim=1)
        # ===================================================================== #

        # ------------------------------------------------------------------------------ #
        # global water balance
        precip_flux = precip * RHO_WATER / self.N_seconds
        evapor_flux = evapor * RHO_WATER / self.N_seconds

        # total water content (batch, var, time, lat, lon)
        TWC_input = self.core_compute.total_column_water(q_input)
        TWC_pred = self.core_compute.total_column_water(q_pred)

        dTWC_dt = (TWC_pred - TWC_input) / self.N_seconds

        # global sum of total water content tendency
        TWC_sum = self.core_compute.weighted_sum(dTWC_dt, axis=(-2, -1))

        # global evaporation source
        E_sum = self.core_compute.weighted_sum(evapor_flux, axis=(-2, -1))

        # global precip sink
        P_sum = self.core_compute.weighted_sum(precip_flux, axis=(-2, -1))

        # global water balance residual
        residual = -TWC_sum - E_sum - P_sum

        # compute correction ratio
        P_correct_ratio = (P_sum + residual) / P_sum
        P_correct_ratio = torch.clamp(P_correct_ratio, min=0.9, max=1.1)
        # broadcast: (batch_size, 1, 1, 1)
        P_correct_ratio = P_correct_ratio.unsqueeze(-1).unsqueeze(-1)

        # apply correction on precip
        precip = precip * P_correct_ratio

        # ===================================================================== #
        # return fixed q and precip back to y_pred

        # expand fixed vars to (batch level, time, lat, lon)
        q_pred = q_pred.unsqueeze(2)
        precip = precip.unsqueeze(1).unsqueeze(2)

        y_pred = concat_fix(y_pred, q_pred, self.q_ind_start, self.q_ind_end, N_vars)
        y_pred = concat_fix(y_pred, precip, self.precip_ind, self.precip_ind, N_vars)

        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


class GlobalEnergyFixer(nn.Module):
    """
    This module applys global energy conservation fixes. The output ensures that the global sum
    of total energy in the atmosphere is balanced by radiantion and energy fluxes at the top of
    the atmosphere and the surface. Variables `air temperature` will be modified to close the
    budget. All corrections are done using float32 Pytorch tensors.

    Args:
        post_conf (dict): config dictionary that includes all specs for the global energy fixer.
    """

    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf["global_energy_fixer"]["simple_demo"]:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340])

            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.core_compute = physics_pressure_level(
                lon_demo, lat_demo, p_level_demo, midpoint=post_conf["global_energy_fixer"]["midpoint"]
            )
            self.N_seconds = int(post_conf["data"]["lead_time_periods"]) * 3600

            gph_surf_demo = np.ones((10, 18))
            self.GPH_surf = torch.from_numpy(gph_surf_demo)

        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf["data"]["save_loc_physics"])

            lon_lat_level_names = post_conf["global_energy_fixer"]["lon_lat_level_name"]

            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()
            p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()

            self.core_compute = physics_pressure_level(
                lon2d, lat2d, p_level, midpoint=post_conf["global_energy_fixer"]["midpoint"]
            )
            self.N_seconds = int(post_conf["data"]["lead_time_periods"]) * 3600

            varname_gph = post_conf["global_energy_fixer"]["surface_geopotential_name"]
            self.GPH_surf = torch.from_numpy(ds_physics[varname_gph[0]].values).float()

        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.T_ind_start = int(post_conf["global_energy_fixer"]["T_inds"][0])
        self.T_ind_end = int(post_conf["global_energy_fixer"]["T_inds"][-1]) + 1

        self.q_ind_start = int(post_conf["global_energy_fixer"]["q_inds"][0])
        self.q_ind_end = int(post_conf["global_energy_fixer"]["q_inds"][-1]) + 1

        self.U_ind_start = int(post_conf["global_energy_fixer"]["U_inds"][0])
        self.U_ind_end = int(post_conf["global_energy_fixer"]["U_inds"][-1]) + 1

        self.V_ind_start = int(post_conf["global_energy_fixer"]["V_inds"][0])
        self.V_ind_end = int(post_conf["global_energy_fixer"]["V_inds"][-1]) + 1

        self.TOA_solar_ind = int(post_conf["global_energy_fixer"]["TOA_rad_inds"][0])
        self.TOA_OLR_ind = int(post_conf["global_energy_fixer"]["TOA_rad_inds"][1])

        self.surf_solar_ind = int(post_conf["global_energy_fixer"]["surf_rad_inds"][0])
        self.surf_LR_ind = int(post_conf["global_energy_fixer"]["surf_rad_inds"][1])

        self.surf_SH_ind = int(post_conf["global_energy_fixer"]["surf_flux_inds"][0])
        self.surf_LH_ind = int(post_conf["global_energy_fixer"]["surf_flux_inds"][1])

        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf["global_energy_fixer"]["denorm"]:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None

    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        # x_input does not have precip and evapor
        x_input = x["x"]
        y_pred = x["y_pred"]

        # detach x_input
        x_input = x_input.detach().to(y_pred.device)

        # other needed inputs
        GPH_surf = self.GPH_surf.to(y_pred.device)
        N_vars = y_pred.shape[1]

        # if denorm is needed
        if self.state_trans:
            x_input = self.state_trans.inverse_transform_input(x_input)
            y_pred = self.state_trans.inverse_transform(y_pred)

        T_input = x_input[:, self.T_ind_start : self.T_ind_end, -1, ...]
        q_input = x_input[:, self.q_ind_start : self.q_ind_end, -1, ...]
        U_input = x_input[:, self.U_ind_start : self.U_ind_end, -1, ...]
        V_input = x_input[:, self.V_ind_start : self.V_ind_end, -1, ...]

        # y_pred (batch, var, time, lat, lon)
        # pick the first time-step, y_pred is expected to have the next step only
        T_pred = y_pred[:, self.T_ind_start : self.T_ind_end, 0, ...]
        q_pred = y_pred[:, self.q_ind_start : self.q_ind_end, 0, ...]
        U_pred = y_pred[:, self.U_ind_start : self.U_ind_end, 0, ...]
        V_pred = y_pred[:, self.V_ind_start : self.V_ind_end, 0, ...]

        TOA_solar_pred = y_pred[:, self.TOA_solar_ind, 0, ...]
        TOA_OLR_pred = y_pred[:, self.TOA_OLR_ind, 0, ...]

        surf_solar_pred = y_pred[:, self.surf_solar_ind, 0, ...]
        surf_LR_pred = y_pred[:, self.surf_LR_ind, 0, ...]
        surf_SH_pred = y_pred[:, self.surf_SH_ind, 0, ...]
        surf_LH_pred = y_pred[:, self.surf_LH_ind, 0, ...]

        # ------------------------------------------------------------------------------ #
        # Latent heat, potential energy, kinetic energy

        # heat capacity on constant pressure
        CP_t0 = (1 - q_input) * CP_DRY + q_input * CP_VAPOR
        CP_t1 = (1 - q_pred) * CP_DRY + q_pred * CP_VAPOR

        # kinetic energy
        ken_t0 = 0.5 * (U_input**2 + V_input**2)
        ken_t1 = 0.5 * (U_pred**2 + V_pred**2)

        # packing latent heat + potential energy + kinetic energy
        E_qgk_t0 = LH_WATER * q_input + GPH_surf + ken_t0
        E_qgk_t1 = LH_WATER * q_input + GPH_surf + ken_t1

        # ------------------------------------------------------------------------------ #
        # energy source and sinks

        # TOA energy flux
        R_T = (TOA_solar_pred + TOA_OLR_pred) / self.N_seconds
        R_T_sum = self.core_compute.weighted_sum(R_T, axis=(-2, -1))

        # surface net energy flux
        F_S = (surf_solar_pred + surf_LR_pred + surf_SH_pred + surf_LH_pred) / self.N_seconds
        F_S_sum = self.core_compute.weighted_sum(F_S, axis=(-2, -1))

        # ------------------------------------------------------------------------------ #
        # thermal energy correction

        # total energy per level
        E_level_t0 = CP_t0 * T_input + E_qgk_t0
        E_level_t1 = CP_t1 * T_pred + E_qgk_t1

        # column integrated total energy
        TE_t0 = self.core_compute.integral(E_level_t0) / GRAVITY
        TE_t1 = self.core_compute.integral(E_level_t1) / GRAVITY
        # dTE_dt = (TE_t1 - TE_t0) / self.N_seconds

        global_TE_t0 = self.core_compute.weighted_sum(TE_t0, axis=(-2, -1))
        global_TE_t1 = self.core_compute.weighted_sum(TE_t1, axis=(-2, -1))

        # total energy correction ratio
        E_correct_ratio = (self.N_seconds * (R_T_sum - F_S_sum) + global_TE_t0) / global_TE_t1
        E_correct_ratio = torch.clamp(E_correct_ratio, min=0.9, max=1.1)
        # broadcast: (batch, 1, 1, 1, 1)
        E_correct_ratio = E_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # apply total energy correction
        E_t1_correct = E_level_t1 * E_correct_ratio

        # let thermal energy carry the corrected total energy amount
        T_pred = (E_t1_correct - E_qgk_t1) / CP_t1

        # ===================================================================== #
        # return fixed q and precip back to y_pred

        # expand fixed vars to (batch level, time, lat, lon)
        T_pred = T_pred.unsqueeze(2)

        y_pred = concat_fix(y_pred, T_pred, self.T_ind_start, self.T_ind_end, N_vars)

        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


def concat_fix(y_pred, q_pred_correct, q_ind_start, q_ind_end, N_vars):
    """
    this function use torch.concat to replace a specific subset of variable channels in `y_pred`.

    Given `q_pred = y_pred[:, ind_start:ind_end, ...]`, and `q_pred_correct` this function
    does: `y_pred[:, ind_start:ind_end, ...] = q_pred_correct`, but without using in-place
    modifications, so the graph of y_pred is maintained. It also handles
    `q_ind_start == q_ind_end cases`.

    All input tensors must have 5 dims of `batch, level-or-var, time, lat, lon`

    Args:
        y_pred (torch.Tensor): Original y_pred tensor of shape (batch, var, time, lat, lon).
        q_pred_correct (torch.Tensor): Corrected q_pred tensor.
        q_ind_start (int): Index where q_pred starts in y_pred.
        q_ind_end (int): Index where q_pred ends in y_pred.
        N_vars (int): Total number of variables in y_pred (i.e., y_pred.shape[1]).

    Returns:
        torch.Tensor: Concatenated y_pred with corrected q_pred.
    """
    # define a list that collects tensors
    var_list = []

    # vars before q_pred
    if q_ind_start > 0:
        var_list.append(y_pred[:, :q_ind_start, ...])

    # q_pred
    var_list.append(q_pred_correct)

    # vars after q_pred
    if q_ind_end < N_vars - 1:
        if q_ind_start == q_ind_end:
            var_list.append(y_pred[:, q_ind_end + 1 :, ...])
        else:
            var_list.append(y_pred[:, q_ind_end:, ...])

    return torch.cat(var_list, dim=1)


class SKEBS(nn.Module):
    """
    post_conf: dictionary with config options for PostBlock.
                if post_conf is not specified in config,
                defaults are set in the parser

    This class is currently a placeholder for SKEBS
    """

    def __init__(self, post_conf):
        """
        post_conf imputed by model init
        if predict_mode=True, batch_size=1
        """
        super().__init__()
        self.nlon = post_conf["model"]["image_width"]
        self.nlat = post_conf["model"]["image_height"]
        self.channels = post_conf["model"]["channels"]
        self.levels = post_conf["model"]["levels"]
        self.surface_channels = post_conf["model"]["surface_channels"]
        self.output_only_channels = post_conf["model"]["output_only_channels"]
        self.input_only_channels = post_conf["model"]["input_only_channels"]
        self.frames = post_conf["model"]["frames"]

        self.lmax = post_conf["skebs"]["lmax"]
        self.mmax = post_conf["skebs"]["mmax"]
        self.grid = post_conf["grid"]
        self.U_inds = post_conf["skebs"]["U_inds"]
        self.V_inds = post_conf["skebs"]["V_inds"]
        self.sp_index = post_conf["skebs"]["SP_ind"]

        # need this info
        self.timestep = post_conf["data"]["timestep"]
        self.level_info = xr.open_dataset(post_conf["data"]["level_info_file"])
        self.level_list = post_conf["data"]["level_list"]
        self.surface_area = xr.open_dataset(post_conf["data"]["save_loc_static"]).surface_area.to_numpy()
        self.device = None
        self.spec_coef = None

        self.initialize_sht()

    def initialize_sht(self):
        """
        Initialize spherical harmonics and inverse spherical harmonics transformations
        for both scalar and vector fields.
        """
        # Initialize spherical harmonics transformation objects
        # self.sht = harmonics.RealSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        self.isht = harmonics.InverseRealSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        # self.vsht = harmonics.RealVectorSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        # self.ivsht = harmonics.InverseRealVectorSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        self.lmax = self.isht.lmax
        self.mmax = self.isht.mmax

    def initialize_pattern(self, y_pred):
        """
        initialize the random pattern.
        in Berner et al
            m is zonal wavenumber -> mmax
            n is total wavenumber -> lmax
        """
        y_shape = y_pred.shape
        self.lrange = torch.arange(1, self.lmax + 1).unsqueeze(1)  # (lmax, 1)
        # assume (b, ... ,lat,lon)
        self.spec_coef = torch.zeros(
            (y_shape[0], 1, 1, self.lmax, self.mmax),  # b, 1, 1, lat, lon
            dtype=torch.cfloat,
        )
        # for _ in range(len(self.spec_coef.shape), len(y_shape)):
        #     # unsqueeze to achieve b, 1, ..., 1, lat, lon matching y_pred
        #     self.spec_coef = self.spec_coef.unsqueeze(1)

        # parameters we want to learn: (init to berner 2009 values for now)
        self.alpha = Parameter(torch.tensor(0.5, requires_grad=True))
        self.variance = Parameter(torch.tensor(0.083, requires_grad=True))
        self.p = Parameter(torch.tensor(-1.27, requires_grad=True))
        self.dE = Parameter(torch.tensor(10e-4, requires_grad=True))
        # initialize pattern todo: how many iters?
        for _ in range(10):
            self.spec_coef = self.cycle_pattern(self.spec_coef)

    def cycle_pattern(self, spec_coef):
        Gamma = torch.sum(self.lrange * (self.lrange + 1.0) * (self.lrange + 2.0) * self.lrange ** (2.0 * self.p))  # scalar
        b = torch.sqrt((4.0 * PI * RAD_EARTH**2.0) / (self.variance * Gamma) * self.alpha * self.dE)  # scalar
        g_n = b * self.lrange**self.p  # (lmax, 1)
        print(f"g_n: {g_n.shape}")
        noise = self.variance * torch.randn(spec_coef.shape)  # (b, 1, 1, lmax, mmax) std normal noise diff for all n?

        new_coef = (1.0 - self.alpha) * spec_coef + g_n * torch.sqrt(self.alpha) * noise  # (lmax, mmax)
        return new_coef

    def initialize_mass_calc(self):
        a_vals = self.level_info["a_model"].sel(level=self.level_list).to_numpy()
        b_vals = self.level_info["b_model"].sel(level=self.level_list).to_numpy()
        a_tensor = torch.from_numpy(a_vals).to(self.device).view(1, self.levels, 1, 1, 1) / 100
        b_tensor = torch.from_numpy(b_vals).to(self.device).view(1, self.levels, 1, 1, 1)
        self.surface_area_tensor = torch.from_numpy(self.surface_area).to(self.device).view(1, 1, 1, self.nlat, 1)
        print(f"surface_area_tensor: {self.surface_area_tensor.shape}")
        self.compute_plev_quantities = compute_pressure_on_mlevs(a_vals=a_tensor, b_vals=b_tensor, plev_dim=1)
        
    def calculate_mass(self, sp):
        # 1 / g * A * integral(dp) [thickness]
        return (1.0 / GRAVITY 
                * self.surface_area_tensor 
                * self.compute_plev_quantities.compute_mlev_thickness(sp) #same shape as sp but with size levels for dim=1
                )

    def forward(self, x):
        x = x["y_pred"]

        if not self.device and not self.spec_coef:  # if one is not set, a runtime error will be thrown later
            self.initialize_pattern(x)
            self.device = x.device
            self.initialize_mass_calc()

        self.spec_coef = self.cycle_pattern(self.spec_coef)  # cycle from prev step
        # b, 1, 1, lmax, mmax
        pattern_on_grid = self.isht(self.spec_coef) # b, 1, 1, lat, lon
        print(f"pattern on grid: {pattern_on_grid.shape}")
        # todo: placeholder for backscatter prediction
        backscatter_pred = torch.ones((x.shape[0], self.levels, 1, self.nlat, self.nlon)) 

        # skebs gives us an instantaneous forcing term, need to multiply by timestep
        total_forcing = self.timestep * backscatter_pred * pattern_on_grid 
        # shape (b, levels, t, lat, lon)

        mlev_mass = self.calculate_mass(x[:, self.sp_index : self.sp_index + 1])  # slice to keep dims
        # (b, levels, 1, lat, lon)
        u_squared, v_squared = x[:, self.U_inds] ** 2, x[:, self.V_inds] ** 2
        wind_squared = u_squared + v_squared
        u_frac = u_squared / wind_squared # (b, levels, 1, lat, lon)
        v_frac = v_squared / wind_squared

        add_wind_magnitude = torch.sqrt(2.0 * total_forcing / mlev_mass + wind_squared) - torch.sqrt(wind_squared)
        x[:, self.U_inds] += add_wind_magnitude * u_frac
        x[:, self.V_inds] += add_wind_magnitude * v_frac
        return x


# import yaml
# import os

# import torch
# from credit.parser import CREDIT_main_parser

# TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
# CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]),
#                     "config")

# def test_SKEBS_rand():
#     config = os.path.join(CONFIG_FILE_DIR, "example_skebs.yml")
#     with open(config) as cf:
#         conf = yaml.load(cf, Loader=yaml.FullLoader)

#     conf = CREDIT_main_parser(conf) # parser will copy model configs to post_conf
#     post_conf = conf['model']['post_conf']

#     image_height = post_conf["model"]["image_height"]
#     image_width = post_conf["model"]["image_width"]
#     channels = post_conf["model"]["channels"]
#     levels = post_conf["model"]["levels"]
#     surface_channels = post_conf["model"]["surface_channels"]
#     output_only_channels = post_conf["model"]["output_only_channels"]
#     input_only_channels = post_conf["model"]["input_only_channels"]
#     frames = post_conf["model"]["frames"]

#     in_channels = channels * levels + surface_channels + input_only_channels
#     x = torch.randn(1, in_channels, frames, image_height, image_width)
#     out_channels = channels * levels + surface_channels + output_only_channels
#     y_pred = torch.randn(1, out_channels, frames, image_height, image_width)

#     postblock = PostBlock(post_conf)
#     assert any([isinstance(module, SKEBS) for module in postblock.modules()])

#     input_dict = {"x": x,
#                   "y_pred": y_pred}

#     skebs_pred = postblock(input_dict)

#     assert skebs_pred.shape == y_pred.shape
#     assert not torch.isnan(skebs_pred).any()

# if __name__ == "__main__":
#     test_SKEBS_rand()
