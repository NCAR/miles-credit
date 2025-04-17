"""
postblock.py
-------------------------------------------------------
Content:
    - PostBlock
    - TracerFixer
    - GlobalMassFixer
    - GlobalWaterFixer
    - GlobalEnergyFixer

"""
import torch
from torch import nn

import numpy as np

from credit.data import get_forward_data
from credit.transforms import load_transforms
from credit.physics_core import physics_pressure_level, physics_hybrid_sigma_level
from credit.physics_constants import (
    GRAVITY,
    RHO_WATER,
    LH_WATER,
    CP_DRY,
    CP_VAPOR,
)
from credit.skebs import SKEBS

import logging
from math import pi
PI = pi
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
        # (1) tracer fixer --> mass fixer --> SKEB / water fixer --> energy fixer

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
        print('bingo bongo:', post_conf["global_mass_fixer"]["activate"])
        if post_conf["global_mass_fixer"]["activate"]:
            if post_conf["global_mass_fixer"]["activate_outside_model"] is False:
                logger.info("GlobalMassFixer registered")
                opt = GlobalMassFixer(post_conf)
                self.operations.append(opt)

        # global water fixer
        if post_conf["global_water_fixer"]["activate"]:
            if post_conf["global_water_fixer"]["activate_outside_model"] is False:
                logger.info("GlobalWaterFixer registered")
                opt = GlobalWaterFixer(post_conf)
                self.operations.append(opt)

        # global energy fixer
        if post_conf["global_energy_fixer"]["activate"]:
            if post_conf["global_energy_fixer"]["activate_outside_model"] is False:
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
            x_demo = np.array(
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    160,
                    180,
                    200,
                    220,
                    240,
                    260,
                    280,
                    300,
                    320,
                    340,
                ]
            )

            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(
                np.array([100, 30000, 50000, 70000, 80000, 90000, 100000])
            )
            self.flag_sigma_level = False
            self.flag_midpoint = post_conf["global_mass_fixer"]["midpoint"]
            self.core_compute = physics_pressure_level(
                lon_demo, lat_demo, p_level_demo, midpoint=self.flag_midpoint
            )

            self.N_levels = len(p_level_demo)
            self.ind_fix = (
                len(p_level_demo)
                - int(post_conf["global_mass_fixer"]["fix_level_num"])
                + 1
            )

        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf["data"]["save_loc_physics"])

            lon_lat_level_names = post_conf["global_mass_fixer"]["lon_lat_level_name"]
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()

            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = post_conf["global_mass_fixer"]["midpoint"]

            if post_conf["global_mass_fixer"]["grid_type"] == "sigma":
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(
                    ds_physics[lon_lat_level_names[2]].values
                ).float()
                self.coef_b = torch.from_numpy(
                    ds_physics[lon_lat_level_names[3]].values
                ).float()

                # get total number of levels
                self.N_levels = len(self.coef_a)
                if self.flag_midpoint:
                    self.N_levels = self.N_levels - 1

                self.core_compute = physics_hybrid_sigma_level(
                    lon2d, lat2d, self.coef_a, self.coef_b, midpoint=self.flag_midpoint
                )
            else:
                self.flag_sigma_level = False
                p_level = torch.from_numpy(
                    ds_physics[lon_lat_level_names[2]].values
                ).float()
                # get total number of levels
                self.N_levels = len(p_level)

                self.core_compute = physics_pressure_level(
                    lon2d, lat2d, p_level, midpoint=self.flag_midpoint
                )
            # -------------------------------------------------------------------------- #
            self.ind_fix = (
                self.N_levels - int(post_conf["global_mass_fixer"]["fix_level_num"]) + 1
            )

        # -------------------------------------------------------------------------- #
        if self.flag_midpoint:
            self.ind_fix_start = self.ind_fix
        else:
            self.ind_fix_start = self.ind_fix - 1

        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.q_ind_start = int(post_conf["global_mass_fixer"]["q_inds"][0])
        self.q_ind_end = int(post_conf["global_mass_fixer"]["q_inds"][-1]) + 1
        if self.flag_sigma_level:
            self.sp_ind = int(post_conf["global_mass_fixer"]["sp_inds"])
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

        # y_pred (batch, var, time, lat, lon)
        # pick the first time-step, y_pred is expected to have the next step only
        # !!! Note: time dimension is collapsed throughout !!!

        q_input = x_input[:, self.q_ind_start : self.q_ind_end, -1, ...]
        q_pred = y_pred[:, self.q_ind_start : self.q_ind_end, 0, ...]

        if self.flag_sigma_level:
            sp_input = x_input[:, self.sp_ind, -1, ...]
            sp_pred = y_pred[:, self.sp_ind, 0, ...]

        # ------------------------------------------------------------------------------ #
        # global dry air mass conservation

        if self.flag_sigma_level:
            # total dry air mass from q_input
            mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input, sp_input)

        else:
            # total dry air mass from q_input
            mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input)

            # total mass from q_pred
            mass_dry_sum_t1_hold = self.core_compute.weighted_sum(
                self.core_compute.integral_sliced(1 - q_pred, 0, self.ind_fix)
                / GRAVITY,
                axis=(-2, -1),
            )

            mass_dry_sum_t1_fix = self.core_compute.weighted_sum(
                self.core_compute.integral_sliced(
                    1 - q_pred, self.ind_fix_start, self.N_levels
                )
                / GRAVITY,
                axis=(-2, -1),
            )

            q_correct_ratio = (
                mass_dry_sum_t0 - mass_dry_sum_t1_hold
            ) / mass_dry_sum_t1_fix

            # broadcast: (batch, 1, 1, 1)
            q_correct_ratio = q_correct_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # ===================================================================== #
            # q fixes based on the ratio
            # fix lower atmosphere
            q_pred_fix = (
                1 - (1 - q_pred[:, self.ind_fix_start :, ...]) * q_correct_ratio
            )
            # extract unmodified part from q_pred
            q_pred_hold = q_pred[:, : self.ind_fix_start, ...]

            # concat upper and lower q vals
            # (batch, level, lat, lon)
            q_pred = torch.cat([q_pred_hold, q_pred_fix], dim=1)

            # ===================================================================== #
            # return fixed q back to y_pred

            # expand fixed vars to (batch, level, time, lat, lon)
            q_pred = q_pred.unsqueeze(2)
            y_pred = concat_fix(
                y_pred, q_pred, self.q_ind_start, self.q_ind_end, N_vars
            )

        # ===================================================================== #
        # surface pressure fixes on global dry air mass conservation
        # model level only

        if self.flag_sigma_level:
            delta_coef_a = self.coef_a.diff().to(q_pred.device)
            delta_coef_b = self.coef_b.diff().to(q_pred.device)

            if self.flag_midpoint:
                p_dry_a = (
                    (delta_coef_a.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_pred)
                ).sum(1)
                p_dry_b = (
                    (delta_coef_b.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_pred)
                ).sum(1)
            else:
                q_mid = (q_pred[:, :-1, ...] + q_pred[:, 1:, ...]) / 2
                p_dry_a = (
                    (delta_coef_a.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_mid)
                ).sum(1)
                p_dry_b = (
                    (delta_coef_b.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_mid)
                ).sum(1)

            grid_area = self.core_compute.area.unsqueeze(0).to(q_pred.device)
            mass_dry_a = (p_dry_a * grid_area).sum((-2, -1)) / GRAVITY
            mass_dry_b = (p_dry_b * sp_pred * grid_area).sum((-2, -1)) / GRAVITY

            # sp correction ratio using t0 dry air mass and t1 moisture
            sp_correct_ratio = (mass_dry_sum_t0 - mass_dry_a) / mass_dry_b
            sp_correct_ratio = sp_correct_ratio.unsqueeze(1).unsqueeze(2)
            sp_pred = sp_pred * sp_correct_ratio

            # expand fixed vars to (batch, level, time, lat, lon)
            sp_pred = sp_pred.unsqueeze(1).unsqueeze(2)
            y_pred = concat_fix(y_pred, sp_pred, self.sp_ind, self.sp_ind, N_vars)

        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


class GlobalWaterFixer(nn.Module):
    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf["global_water_fixer"]["simple_demo"]:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array(
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    160,
                    180,
                    200,
                    220,
                    240,
                    260,
                    280,
                    300,
                    320,
                    340,
                ]
            )

            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(
                np.array([100, 30000, 50000, 70000, 80000, 90000, 100000])
            )
            self.flag_sigma_level = False
            self.flag_midpoint = post_conf["global_water_fixer"]["midpoint"]
            self.core_compute = physics_pressure_level(
                lon_demo, lat_demo, p_level_demo, midpoint=self.flag_midpoint
            )
            self.N_levels = len(p_level_demo)
            self.N_seconds = int(post_conf["data"]["lead_time_periods"]) * 3600

        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf["data"]["save_loc_physics"])

            lon_lat_level_names = post_conf["global_mass_fixer"]["lon_lat_level_name"]
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()

            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = post_conf["global_mass_fixer"]["midpoint"]

            if post_conf["global_mass_fixer"]["grid_type"] == "sigma":
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(
                    ds_physics[lon_lat_level_names[2]].values
                ).float()
                self.coef_b = torch.from_numpy(
                    ds_physics[lon_lat_level_names[3]].values
                ).float()

                # get total number of levels
                self.N_levels = len(self.coef_a)

                if self.flag_midpoint:
                    self.N_levels = self.N_levels - 1

                self.core_compute = physics_hybrid_sigma_level(
                    lon2d, lat2d, self.coef_a, self.coef_b, midpoint=self.flag_midpoint
                )
            else:
                self.flag_sigma_level = False
                p_level = torch.from_numpy(
                    ds_physics[lon_lat_level_names[2]].values
                ).float()
                # get total number of levels
                self.N_levels = len(p_level)

                self.core_compute = physics_pressure_level(
                    lon2d, lat2d, p_level, midpoint=self.flag_midpoint
                )

            self.N_seconds = int(post_conf["data"]["lead_time_periods"]) * 3600

        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.q_ind_start = int(post_conf["global_water_fixer"]["q_inds"][0])
        self.q_ind_end = int(post_conf["global_water_fixer"]["q_inds"][-1]) + 1
        self.precip_ind = int(post_conf["global_water_fixer"]["precip_ind"])
        self.evapor_ind = int(post_conf["global_water_fixer"]["evapor_ind"])
        if self.flag_sigma_level:
            self.sp_ind = int(post_conf["global_water_fixer"]["sp_inds"])
        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf["global_water_fixer"]["denorm"]:
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

        if self.flag_sigma_level:
            sp_input = x_input[:, self.sp_ind, -1, ...]
            sp_pred = y_pred[:, self.sp_ind, 0, ...]

        # ------------------------------------------------------------------------------ #
        # global water balance
        precip_flux = precip * RHO_WATER / self.N_seconds
        evapor_flux = evapor * RHO_WATER / self.N_seconds

        # total water content (batch, var, time, lat, lon)
        if self.flag_sigma_level:
            TWC_input = self.core_compute.total_column_water(q_input, sp_input)
            TWC_pred = self.core_compute.total_column_water(q_pred, sp_pred)
        else:
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
        # P_correct_ratio = torch.clamp(P_correct_ratio, min=0.9, max=1.1)
        # broadcast: (batch_size, 1, 1, 1)
        P_correct_ratio = P_correct_ratio.unsqueeze(-1).unsqueeze(-1)

        # apply correction on precip
        precip = precip * P_correct_ratio

        # ===================================================================== #
        # return fixed precip back to y_pred
        precip = precip.unsqueeze(1).unsqueeze(2)
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
            x_demo = np.array(
                [
                    0,
                    20,
                    40,
                    60,
                    80,
                    100,
                    120,
                    140,
                    160,
                    180,
                    200,
                    220,
                    240,
                    260,
                    280,
                    300,
                    320,
                    340,
                ]
            )

            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(
                np.array([100, 30000, 50000, 70000, 80000, 90000, 100000])
            )
            self.flag_sigma_level = False
            self.flag_midpoint = post_conf["global_energy_fixer"]["midpoint"]
            self.core_compute = physics_pressure_level(
                lon_demo,
                lat_demo,
                p_level_demo,
                midpoint=self.flag_midpoint,
            )
            self.N_seconds = int(post_conf["data"]["lead_time_periods"]) * 3600

            gph_surf_demo = np.ones((10, 18))
            self.GPH_surf = torch.from_numpy(gph_surf_demo)

        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf["data"]["save_loc_physics"])

            lon_lat_level_names = post_conf["global_mass_fixer"]["lon_lat_level_name"]
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).float()
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).float()

            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = post_conf["global_mass_fixer"]["midpoint"]

            if post_conf["global_mass_fixer"]["grid_type"] == "sigma":
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(
                    ds_physics[lon_lat_level_names[2]].values
                ).float()
                self.coef_b = torch.from_numpy(
                    ds_physics[lon_lat_level_names[3]].values
                ).float()

                # get total number of levels
                self.N_levels = len(self.coef_a)

                if self.flag_midpoint:
                    self.N_levels = self.N_levels - 1

                self.core_compute = physics_hybrid_sigma_level(
                    lon2d, lat2d, self.coef_a, self.coef_b, midpoint=self.flag_midpoint
                )
            else:
                self.flag_sigma_level = False
                p_level = torch.from_numpy(
                    ds_physics[lon_lat_level_names[2]].values
                ).float()
                # get total number of levels
                self.N_levels = len(p_level)

                self.core_compute = physics_pressure_level(
                    lon2d, lat2d, p_level, midpoint=self.flag_midpoint
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

        if self.flag_sigma_level:
            self.sp_ind = int(post_conf["global_energy_fixer"]["sp_inds"])
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

        if self.flag_sigma_level:
            sp_input = x_input[:, self.sp_ind, -1, ...]
            sp_pred = y_pred[:, self.sp_ind, 0, ...]

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
        E_qgk_t1 = LH_WATER * q_pred + GPH_surf + ken_t1

        # ------------------------------------------------------------------------------ #
        # energy source and sinks

        # TOA energy flux
        R_T = (TOA_solar_pred + TOA_OLR_pred) / self.N_seconds
        R_T_sum = self.core_compute.weighted_sum(R_T, axis=(-2, -1))

        # surface net energy flux
        F_S = (
            surf_solar_pred + surf_LR_pred + surf_SH_pred + surf_LH_pred
        ) / self.N_seconds
        F_S_sum = self.core_compute.weighted_sum(F_S, axis=(-2, -1))

        # ------------------------------------------------------------------------------ #
        # thermal energy correction

        # total energy per level
        E_level_t0 = CP_t0 * T_input + E_qgk_t0
        E_level_t1 = CP_t1 * T_pred + E_qgk_t1

        # column integrated total energy
        if self.flag_sigma_level:
            TE_t0 = self.core_compute.integral(E_level_t0, sp_input) / GRAVITY
            TE_t1 = self.core_compute.integral(E_level_t1, sp_pred) / GRAVITY
        else:
            TE_t0 = self.core_compute.integral(E_level_t0) / GRAVITY
            TE_t1 = self.core_compute.integral(E_level_t1) / GRAVITY

        # dTE_dt = (TE_t1 - TE_t0) / self.N_seconds

        global_TE_t0 = self.core_compute.weighted_sum(TE_t0, axis=(-2, -1))
        global_TE_t1 = self.core_compute.weighted_sum(TE_t1, axis=(-2, -1))

        # total energy correction ratio
        E_correct_ratio = (
            self.N_seconds * (R_T_sum - F_S_sum) + global_TE_t0
        ) / global_TE_t1
        # E_correct_ratio = torch.clamp(E_correct_ratio, min=0.9, max=1.1)
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

