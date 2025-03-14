"""
postblock.py
-------------------------------------------------------
Content:
    - PostBlock
    - TracerFixer
    - GlobalMassFixer
    - GlobalWaterFixer
    - GlobalEnergyFixer
    - SKEBS

"""

import copy
import os
from os.path import join
from turtle import back
import torch
from torch import nn
import torch_harmonics as harmonics
import numpy as np
import xarray as xr

from credit.boundary_padding import TensorPadding
from credit.data import get_forward_data
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from credit.transforms import load_transforms
from credit.physics_core import compute_density, compute_pressure_on_mlevs, physics_pressure_level, physics_hybrid_sigma_level
from credit.physics_constants import PI, RAD_EARTH, GRAVITY, RHO_WATER, LH_WATER, CP_DRY, CP_VAPOR
import segmentation_models_pytorch as smp


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
        if post_conf['global_mass_fixer']['activate']:
            logger.info('GlobalMassFixer registered')
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
            self.sp_ind = int(post_conf["global_mass_fixer"]["sp_inds"])
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
            self.sp_ind = int(post_conf["global_mass_fixer"]["sp_inds"])
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

def concat_for_inplace_ops(y_orig, y_inplace_slice, ind_start, ind_end):
    """
    alternate way to concat tensors along first dim, 
    given a set of indices to replace that are contiguous 
    """
    tensors = [
        y_orig[:, :ind_start],
        y_inplace_slice,
        y_orig[:, ind_end + 1:]
    ]
    new_tensor = torch.concat(tensors, dim=1)
    return new_tensor


class Backscatter_FCNN(nn.Module):
    def __init__(self,
                 in_channels,
                 levels):
        # could also predict with x_prev and y
        super().__init__()
        self.in_channels = in_channels
        self.levels = levels
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 2, self.levels)
        self.relu2 = nn.ReLU()


        self.scale = torch.tensor(40.)


    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1) # put channels last

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.scale * x

        x = x.permute(0, -1, 1, 2, 3) # put channels back to 1st dim
        return x

class Backscatter_FCNN_wide(nn.Module):
    def __init__(self,
                 in_channels,
                 levels):
        # could also predict with x_prev and y
        super().__init__()
        self.in_channels = in_channels
        self.levels = levels

        self.fc1 = nn.Linear(in_channels, in_channels * 2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_channels * 2, in_channels * 4)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(in_channels * 4, in_channels * 2)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(in_channels * 2, levels)
        self.relu4 = nn.ReLU()


        self.scale = torch.tensor(40.)


    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1) # put channels last

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.relu4(x)

        x = self.scale * x

        x = torch.clamp(x, max=150.)

        x = x.permute(0, -1, 1, 2, 3) # put channels back to 1st dim
        return x

class Backscatter_CNN(nn.Module):
    def __init__(self,
                 in_channels,
                 levels,
                 nlat,
                 nlon):
        # could also predict with x_prev and y
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.in_channels = in_channels
        self.levels = levels

        # setup padding functions
        self.pad_lon = torch.nn.CircularPad2d((1,1,0,0))
        self.pad_lat = torch.nn.ReplicationPad2d((0,0,1,1))

        # setup conv layer
        self.conv = torch.nn.Conv2d(self.in_channels, self.levels, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

        self.scale = torch.tensor(70.)
        

    def pad(self, x):
        x = self.pad_lat(x) #reflection padding
        x[..., [0,-1], :] = torch.roll(x[..., [0,-1], :], self.nlon // 2, -1) # shift reflection by 180
        x = self.pad_lon(x) #padding across lon
        return x
    
    def unpad(self, x):
        return x[..., 1:-1, 1:-1]

    def forward(self, x):
        x = x.squeeze(2) # squeeze out time dim (see above)
        x = self.pad(x)
        # (b,c,lat+2,lon+2)
        x = self.conv(x) # should take out the pad
        x = self.sigmoid(x)
        x = self.scale * x

        x = x.unsqueeze(2)
        return x

supported_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
}

def load_premade_encoder_model(model_conf):
    model_conf = copy.deepcopy(model_conf)
    name = model_conf.pop("name")
    if name in supported_models:
        logger.info(f"Loading model {name} with settings {model_conf}")
        return supported_models[name](**model_conf)
    else:
        raise OSError(
            f"Model name {name} not recognized. Please choose from {supported_models.keys()}"
        )

class Backscatter_unet(nn.Module):
    def __init__(self,
                 in_channels,
                 levels,
                 nlat,
                 nlon,
                 architecture,
                 padding):
        # could also predict with x_prev and y
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.in_channels = in_channels
        self.levels = levels
        
        # setup padding functions
        self.pad = padding
        if self.pad:
            logger.info(f"padding size {self.pad} inside unet")
            self.boundary_padding = TensorPadding(pad_lat=(self.pad, self.pad),
                                                pad_lon=(self.pad, self.pad))

        self.relu = nn.ReLU()

        if architecture is None:
            architecture = {
                            "name": "unet++",
                            "encoder_name": "resnet34",
                            "encoder_weights": "imagenet",
                        }
        if architecture["name"] == "unet":
            architecture["decoder_attention_type"] = "scse"

        architecture["in_channels"] = in_channels
        architecture["classes"] = levels

        self.model = load_premade_encoder_model(architecture)
    
    def forward(self, x):
        x = x.squeeze(2) # squeeze out time dim (see above)

        if self.pad:
            x = self.boundary_padding.pad(x)
        
        x = self.model(x)

        if self.pad:
            x = self.boundary_padding.unpad(x)

        x = self.relu(x)


        x = x.unsqueeze(2)
        return x

class Backscatter_fixed_col(nn.Module):
    def __init__(self,
                 levels,):
        super().__init__()
        self.backscatter_array = Parameter(torch.full((1,levels,1,1,1), 2.5))

    def forward(self, x):
        # self.backscatter_array.data = self.backscatter_array.data.clamp(0., 10.)
        logger.debug(torch.flatten(self.backscatter_array))
        return self.backscatter_array  # this will be inside sqrt
    

class Backscatter_prescribed(nn.Module):
    """  min/max scaling
        # forcing / forcing_max * perturb_frac * sigma_max * std
        # perturb_max is the maximum perturbation fraction wrt to sigma_max*std
        # where we have pre-calculated forcing_max"""
    def __init__(self,
                 levels,
                 std_path,
                 sigma_max,
                 perturb_frac):
        super().__init__()

        self.backscatter_array = Parameter(torch.full((1,levels,1,1,1), 2.5))

        std = xr.open_dataset(std_path)
        std_wind = np.sqrt(std.U.values ** 2 + std.V.values ** 2)
        self.register_buffer("max_perturb",
                             (torch.tensor(std_wind * sigma_max * perturb_frac)
                            .view(1, levels, 1, 1, 1)),
                            persistent=False)

    def forward(self, x):
        return self.backscatter_array * (self.max_perturb / 2. ) ** 2  # this will be inside sqrt
        #add_wind_mag with backscatter = 1.0 is under ~80 for most of the distribution. target: 7% perturb fraction, 2 sigma level
    

class SKEBS(nn.Module):
    """
    post_conf: dictionary with config options for PostBlock.
                if post_conf is not specified in config,
                defaults are set in the parser

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

        self.forecast_len = post_conf["data"]["forecast_len"] + 1
        self.valid_forecast_len = post_conf["data"]["valid_forecast_len"] + 1
        self.multistep = self.forecast_len > 1
        self.lmax = post_conf["skebs"]["lmax"]
        self.mmax = post_conf["skebs"]["mmax"]
        self.grid = post_conf["grid"]
        self.U_inds = post_conf["skebs"]["U_inds"]
        self.V_inds = post_conf["skebs"]["V_inds"]
        self.T_inds = post_conf["skebs"]["T_inds"]
        self.Q_inds = post_conf["skebs"]["Q_inds"]
        self.sp_index = post_conf["skebs"]["SP_ind"]
        self.static_inds = post_conf["skebs"]["static_inds"]
        
        # setup coslat, gets expanded to ensemble size and to device in first forward pass
        cos_lat = np.cos(np.deg2rad(xr.open_dataset(post_conf["data"]["save_loc_static"])["latitude"])).values
        self.cos_lat = torch.tensor(cos_lat).reshape(1, 1, 1, cos_lat.shape[0], 1).expand(1,1,1,cos_lat.shape[0], 288)
        
        self.state_trans = load_transforms(post_conf, scaler_only=True)
        self.eps = 1e-12

        # check for contiguous indices, need this for concat operation
        assert np.all(np.diff(self.U_inds) == 1) and np.all(self.U_inds[:-1] <= self.U_inds[1:])
        assert np.all(np.diff(self.V_inds) == 1) and np.all(self.V_inds[:-1] <= self.V_inds[1:])

        # need this info
        self.timestep = post_conf["data"]["lead_time_periods"] * 3600
        self.alpha_init = post_conf["skebs"].get("alpha_init", 0.125)
        self.zero_out_levels_top_of_model = post_conf["skebs"].get("zero_out_levels_top_of_model", 3)
        logger.info(f"filtering out top {self.zero_out_levels_top_of_model} levels of skebs perturbation")
        # self.level_info = xr.open_dataset(post_conf["data"]["level_info_file"])
        # self.level_list = post_conf["data"]["level_list"]
        # self.surface_area = xr.open_dataset(post_conf["data"]["save_loc_static"])["surface_area"].to_numpy()

        self.tropics_only_dissipation = post_conf["skebs"].get("tropics_only_dissipation", False)
        self.filter_backscatter = post_conf["skebs"].get("filter_backscatter", True) # backscatter filter by default, trainable is set below using train_spectral_filter
        if self.filter_backscatter:
            logger.info("Filtering backscatter")
        else:
            logger.info("Filtering the pattern")

        self.initialize_sht()
        self.initialize_skebs_parameters()
        # self.initialize_plev_calc()
        
        # coeffs havent been spun up yet (indicates need to cycle the coeffs)
        self.spec_coef_is_initialized = False

        # freeze pattern weights before init backscatter
        self.freeze_pattern_weights = post_conf["skebs"].get("freeze_pattern_weights", False)
        if self.freeze_pattern_weights:
            logger.warning("freezing all skebs pattern weights")
            for param in self.parameters():
                param.requires_grad = False

        # initialize backscatter prediction
        num_channels = (self.channels * self.levels 
                        + post_conf["model"]["surface_channels"]
                        + post_conf["model"]["output_only_channels"]
                        + len(self.static_inds)
                        + 1 # this one for coslat
                        )
        # num_channels = (self.channels * self.levels 
        #                 + len(post_conf["data"]["surface_variables"])
        #                 + len(post_conf["data"]["diagnostic_variables"])
        # )
        self.relu = nn.ReLU() # use this to gaurantee positive backscatter

        dissipation_type = post_conf["skebs"]["dissipation_type"]
        if dissipation_type == "prescribed":
            self.backscatter_network = Backscatter_prescribed(self.levels,
                                                             post_conf["data"]["std_path"],
                                                             post_conf["skebs"]["sigma_max"],
                                                             post_conf["skebs"]["perturb_frac"])
        elif dissipation_type == "uniform":
            self.backscatter_network = Backscatter_fixed_col(self.levels)
        elif dissipation_type == "FCNN":
            self.backscatter_network = Backscatter_FCNN(num_channels, self.levels)
        elif dissipation_type == "FCNN_wide":
            self.backscatter_network = Backscatter_FCNN_wide(num_channels, self.levels)
        elif dissipation_type == "CNN":
            self.backscatter_network = Backscatter_CNN(num_channels, self.levels, self.nlat, self.nlon)
        elif dissipation_type == "unet":
            architecture = post_conf["skebs"].get("architecture", None)
            padding = post_conf["skebs"].get("padding", 0)
            self.backscatter_network = Backscatter_unet(num_channels,
                                                        self.levels,
                                                        self.nlat,
                                                        self.nlon,
                                                        architecture,
                                                        padding)
        else:
            raise RuntimeError(f"{dissipation_type} is a not a valid dissipation type, please modify config")
        
        logger.info(f"using dissipation type: {dissipation_type}")

        # freeze backscatter weights if needed
        if post_conf["skebs"].get("freeze_dissipation_weights", False):
            logger.warning("freezing all dissipation predictor weights")
            for param in self.backscatter_network.parameters():
                param.requires_grad = False
        # turn off training for all skebs params
        if not post_conf["skebs"].get("trainable", True):
            logger.warning("freezing all SKEBS parameters due to skebs config")
            for param in self.parameters():
                param.requires_grad = False

        # reset training for certain params
        self.train_alpha = post_conf["skebs"].get("train_alpha", False)
        if self.train_alpha:
            self.alpha.requires_grad = True
        
        self.train_spectral_filter = post_conf["skebs"].get("train_spectral_filter", False)
        if self.train_spectral_filter:
            if self.filter_backscatter:
                self.spectral_backscatter_filter.requires_grad = True
            else: 
                self.spectral_filter.requires_grad = True


        logger.info(f"trainable params{[name for name, param in self.named_parameters() if param.requires_grad]}")


        ########### debugging and analysis features #############
        self.write_debug_files = post_conf['skebs'].get('write_debug_files', False)
        self.write_every = 1000
        save_loc = post_conf['skebs']["save_loc"]
        self.debug_save_loc = join(save_loc, "debug_skebs")
        self.write_rollout_debug_files = post_conf["skebs"].get("write_rollout_debug_files", True)

        if self.write_debug_files or self.write_rollout_debug_files:
            os.makedirs(self.debug_save_loc, exist_ok=True)
            logger.info("writing SKEBS debugging files")

        

        self.iteration = 0


        ##### write out backscatter files #####
        self.is_training = False # will set this in first time into forward
        self.save_backscatter_prediction = post_conf["skebs"].get("save_backscatter", False)
        if self.save_backscatter_prediction:
            logger.info("writing backscatter files if not in training mode")
            self.backscatter_save_loc = join(save_loc, "backscatter")
            os.makedirs(self.backscatter_save_loc, exist_ok=True)

        ############# stop iteration ###################
        self.iteration_stop = post_conf['skebs'].get("iteration_stop", 0)
        if self.iteration_stop:
            logger.info(f"SKEBS is STOPPING at iteration {self.iteration_stop}")

    def initialize_sht(self):
        """
        Initialize spherical harmonics and inverse spherical harmonics transformations
        for both scalar and vector fields.
        """
        # Initialize spherical harmonics transformation objects
        self.sht = harmonics.RealSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        self.isht = harmonics.InverseRealSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        # self.vsht = harmonics.RealVectorSHT(self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False)
        self.ivsht = harmonics.InverseRealVectorSHT(
            self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False
        )
        self.lmax = self.isht.lmax
        self.mmax = self.isht.mmax

        # Compute quadrature weights and cosine of latitudes for the grid
        # cost, quad_weights = harmonics.quadrature.legendre_gauss_weights(
        #     self.nlat, -1, 1
        # )

        ## equiangular grid
        cost, w = harmonics.quadrature.clenshaw_curtiss_weights(self.nlat, -1, 1)
        self.lats = -torch.as_tensor(np.arcsin(cost))
        self.lons = torch.linspace(0, 2 * np.pi, self.nlon + 1, dtype=torch.float64)[
            : self.nlon
        ]

        l_arr = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l_arr = l_arr.expand(self.lmax, self.mmax)
        self.register_buffer("lap", -l_arr * (l_arr + 1) / RAD_EARTH**2,
                             persistent=False)
        self.register_buffer("invlap", -(RAD_EARTH**2) / l_arr / (l_arr + 1),
                             persistent=False)
        self.invlap[0] = 0.0  # Adjusting the first element to avoid division by zero

        logging.info(f"lmax: {self.lmax}, mmax: {self.mmax}")

    def initialize_skebs_parameters(self):
        self.register_buffer('backscatter_filter', # filter to zero out backscatter at top of model
                             torch.cat([
                                 torch.zeros(self.zero_out_levels_top_of_model),
                                 torch.ones(self.levels - self.zero_out_levels_top_of_model)
                             ]).view(1, self.levels, 1, 1, 1),
                             persistent=False) 
        if self.tropics_only_dissipation:
            self.register_buffer('tropics_backscatter_filter',
                                 torch.cat([
                                 torch.zeros(70),
                                 torch.ones(52),
                                 torch.zeros(70)
                                ]).view(1, 1, 1, self.nlat, 1),
                                persistent=False) 
        else:
            self.register_buffer('tropics_backscatter_filter',
                        torch.ones(self.nlat).view(1, 1, 1, self.nlat, 1),
                        persistent=False)
            

        self.register_buffer('lrange', 
                             torch.arange(1, self.lmax + 1).unsqueeze(1),
                             persistent=False) # (lmax, 1)
        # assume (b, c, t, ,lat,lon)
        # parameters we want to learn: (default init to berner 2009 values)
        if self.multistep:
            logger.info("multi-step skebs")
            self.alpha = Parameter(torch.tensor(self.alpha_init, requires_grad=True))
        else:
            logger.info("single-step skebs")
            self.alpha = Parameter(torch.tensor(1.0, requires_grad=False))
            self.alpha.requires_grad = False 
        self.variance = Parameter(torch.tensor(0.083, requires_grad=True))
        self.p = Parameter(torch.tensor(-1.27, requires_grad=True))
        self.dE = Parameter(torch.tensor(10e-4, requires_grad=True))
        self.r = Parameter(torch.tensor(0.01, requires_grad=False)) # see berner 2009, section 4a
        self.r.requires_grad = False
        # either filter the backscatter or the pattern
        self.spectral_filter = Parameter(torch.cat([
                                        torch.ones(90),
                                        torch.logspace(0., -3, 10),
                                        torch.zeros(self.lmax - 100)
                                        ]) # keep only first 100 wavenums
                                        .view(1,1,1,self.lmax, 1),
                                        requires_grad=False)
        if self.filter_backscatter:
            self.spectral_backscatter_filter = Parameter(torch.cat([
                                                    torch.ones(10),
                                                    torch.linspace(1., 0.2, 20),
                                                    torch.zeros(self.lmax - 30)
                                                    ])
                                                    .view(1,1,1,self.lmax, 1),
                                                    requires_grad=True)
        else:
            self.spectral_backscatter_filter = Parameter(torch.cat([
                                                torch.ones(60),
                                                torch.linspace(1., 0.2, 40),
                                                torch.zeros(self.lmax - 100)
                                                ])
                                                .view(1,1,1,self.lmax, 1),
                                                requires_grad=True)
            # self.spectral_filter = Parameter(torch.cat([
            #                                         torch.ones(30),
            #                                         torch.logspace(0., -3, 30),
            #                                         torch.zeros(self.lmax - 60)
            #                                         ])
            #                                         .view(1,1,1,self.lmax, 1),
            #                                         requires_grad=True)
        # self.spectral_filter = Parameter(torch.ones(self.lmax).view(1,1,1,self.lmax,1),
        #                                  requires_grad=True)

    def clip_parameters(self):
        self.alpha.data = self.alpha.data.clamp(self.eps, 1.)
        self.variance.data = self.variance.clamp(self.eps, 10.)
        self.p.data = self.p.data.clamp(-10, -self.eps)
        self.dE.data = self.dE.data.clamp(self.eps, 1.)
        self.r.data = self.r.data.clamp(self.eps, 1.)
        self.spectral_filter.data = self.spectral_filter.data.clamp(0., 1.)
        self.spectral_backscatter_filter.data = self.spectral_backscatter_filter.data.clamp(0., 1.)


    def initialize_pattern(self, y_pred):
        """
        initialize the random pattern.
        in Berner et al
            m is zonal wavenumber -> mmax
            n is total wavenumber -> lmax
        """
        if self.iteration > 0:
            self.spec_coef = self.spec_coef.detach()
        # self.spectral_filter = self.spectral_filter.detach()
        # self.alpha = self.alpha.detach()
        y_shape = y_pred.shape

        self.spec_coef = torch.zeros(
                                 (y_shape[0], 1, 1, self.lmax, self.mmax),  # b, 1, 1, lmax, mmax
                                 dtype=torch.cfloat,
                                 device=y_pred.device)
        self.multivariateNormal = MultivariateNormal(torch.zeros(2, device=y_pred.device), 
                                                     torch.eye(2, device=y_pred.device))
        # initialize pattern todo: how many iters?
        iters = 5
        logger.debug(f"initializing pattern with {iters} iterations")
        for i in range(iters):
            self.spec_coef = self.cycle_pattern(self.spec_coef)

    def cycle_pattern(self, spec_coef):
        Gamma = torch.sum(self.lrange * (self.lrange + 1.0) * (2 * self.lrange + 1.0) * self.lrange ** (2.0 * self.p))  # scalar
        self.b = torch.sqrt((4.0 * PI * RAD_EARTH**2.0) / (self.variance * Gamma) * self.alpha * self.dE)  # scalar
        self.g_n = self.b * self.lrange ** self.p  # (lmax, 1)

        cmplx_noise = torch.view_as_complex(self.multivariateNormal.sample(spec_coef.shape))
        noise = self.variance * cmplx_noise
        new_coef = (1.0 - self.alpha) * spec_coef + self.g_n * torch.sqrt(self.alpha) * noise  # (lmax, mmax)
        assert not new_coef.isnan().any()
        return new_coef * self.spectral_filter 

    def forward(self, x):
        if self.is_training: # this checks if we are in a training script
            # self.training is a torch level thing that checks if we are in train/validation mode of training
            # for inference, we don't need to reset the pattern
            if self.training and self.steps >= self.forecast_len:
                self.spec_coef_is_initialized = False
                self.spec_coef = self.spec_coef.detach()
                logger.debug(f"pattern is reset after train step {self.steps} total iter {self.iteration}")
            elif not self.training and self.steps >= self.valid_forecast_len:
                self.spec_coef_is_initialized = False
                self.spec_coef = self.spec_coef.detach()
                logger.debug(f"pattern is reset after valid step {self.steps} total iter {self.iteration}")
        
        # shutoff skebs if specified
        if self.iteration_stop > 0 and self.iteration > self.iteration_stop:
            if self.iteration == self.iteration_stop + 1:
                logger.info(f"skebs stopped at {self.iteration_stop} steps")
                
            return x
        
        ################### setup input data ################### 

        x_input_statics = x["x"][:, self.static_inds]
        x = x["y_pred"]

        if self.iteration == 0:
            self.cos_lat = self.cos_lat.to(x.device).expand(x.shape[0], *self.cos_lat.shape[1:])

        x = torch.cat([x, x_input_statics, self.cos_lat], dim=1) # add in static vars and coslat



        ################### BACKSCATTER ################### 
        # takes in raw (transformed) model output
        # TODO feed in prev step data?
        backscatter_pred = self.backscatter_filter * self.tropics_backscatter_filter * self.backscatter_network(x)
        backscatter_pred = self.relu(backscatter_pred) # make sure filter doesnt turn it negative

        backscatter_unfiltered = backscatter_pred.clone().detach()
        with torch.autocast(device_type="cuda", enabled=False): #isht cannot use amp
            backscatter_spec = self.sht(backscatter_pred) # b, levels, t, lmax, mmax
            backscatter_spec = self.spectral_backscatter_filter * backscatter_spec
            backscatter_pred = self.isht(backscatter_spec)

        backscatter_pred = self.relu(backscatter_pred) 

        logger.debug(f"max backscatter: {torch.max(torch.abs(backscatter_pred))}")

        if not self.is_training and torch.is_grad_enabled():
            self.is_training = True # set and forget to detect if we are in a training script

        if ((self.save_backscatter_prediction and not self.is_training)
            or (self.write_debug_files and self.iteration % self.write_every == 0)):
            logger.info(f"writing backscatter file for iter {self.iteration}")
            torch.save(backscatter_pred, join(self.backscatter_save_loc, f"backscatter_{self.iteration}"))
            torch.save(backscatter_unfiltered, join(self.backscatter_save_loc, f"backscatter_raw_{self.iteration}"))

        
        ################### SKEBS pattern ####################

        # inverse transform to get real output
        x = self.state_trans.inverse_transform(x)

        # (re)-initialize the pattern or clip the updated parameters
        if not self.spec_coef_is_initialized: #hacky way of doing lazymodulemixin
            self.steps = 0
            self.initialize_pattern(x)
            self.spec_coef_is_initialized = True

        else:
            self.clip_parameters()

        # cycle the pattern
        self.spec_coef = self.cycle_pattern(self.spec_coef) # b, 1, 1, lmax, mmax

        # transform pattern to grid space
        spec_coef = self.spec_coef.squeeze()
        u_chi, v_chi = self.getgrad(spec_coef)
        u_chi, v_chi = u_chi.unsqueeze(1).unsqueeze(1), v_chi.unsqueeze(1).unsqueeze(1)
        logger.debug(f"max pattern: {torch.max(torch.abs(u_chi))}")
        # compute the dissipation term
        dissipation_term = torch.sqrt(self.r * backscatter_pred * self.timestep / self.dE) # shape (b, levels, 1, lat, lon)

        ################### calculate the density ###################

        # sp = torch.ones_like(x[:, self.sp_index : self.sp_index + 1], device = x.device) * 1013.
        # sp = x[:, self.sp_index : self.sp_index + 1]  # slice to keep dims
        # t = x[:, self.T_inds]
        # q = x[:, self.Q_inds]
        # density = self.calculate_density(sp, t, q)
        # assert torch.min(density) >= 0., "ERROR: density is less than 0"
        # (b, levels, 1, lat, lon)

        # big forcing at top of atmosphere..
        # skebs gives us an instantaneous forcing term, need to multiply by timestep (euler step)
        # du/dt = 1 / rho * forcing
        # euler step: u_1 = u_0 + dt * 1/rho * forcing
        # add_wind_magnitude = total_forcing * self.timestep 

        # still debugging this part
        # add_wind_magnitude = (1. / density) * total_forcing * self.timestep  

        #############################################################
        ################### DEBUG perturbations #####################


        ## debug skebs, write out physical values 
        if self.write_debug_files and self.iteration % self.write_every == 0:
            torch.save(self.spectral_filter, join(self.debug_save_loc, f"spectral_filter_{self.iteration}"))
            # add_wind_magnitude = torch.sqrt(dissipation_term ** 2 * (u_chi ** 2 + v_chi ** 2))
            # logger.debug(f"perturb max/min: {add_wind_magnitude.max():.2f}, {add_wind_magnitude.min():.2f}")
            # torch.save(add_wind_magnitude, join(self.debug_save_loc, f"perturb_{self.iteration}"))
            # torch.save(pattern_on_grid, join(self.debug_save_loc, f"pattern_{self.iteration}"))
            # torch.save(x, join(self.debug_save_loc, f"x_{self.iteration}"))

        #############################################################
        ##################### perturb fields ########################

        u_perturb = dissipation_term * u_chi
        v_perturb = dissipation_term * v_chi

        if self.write_rollout_debug_files and not self.is_training: # always write these files in rollout
            torch.save(u_perturb, join(self.debug_save_loc, f"u_perturb_{self.iteration}"))
            torch.save(v_perturb, join(self.debug_save_loc, f"v_perturb_{self.iteration}"))

        x_u_wind = x[:, self.U_inds] + u_perturb
        x_v_wind = x[:, self.V_inds] + v_perturb
        
        x = concat_for_inplace_ops(x, x_u_wind, min(self.U_inds), max(self.U_inds))
        x = concat_for_inplace_ops(x, x_v_wind, min(self.V_inds), max(self.V_inds))
        
        # check for nans and transform back to model (transformed) output space
        assert not torch.isnan(x).any()
        x = self.state_trans.transform_array(x)
        assert not torch.isnan(x).any()

        #############################################################
        ################### setup next iteration #####################
        self.iteration += 1 # this one for total iterations
        self.steps += 1  # this one for skebs/model state
        
        return x
    
    def spec2grid(self, uspec):
        """
        spatial data from spectral coefficients
        """
        return self.isht(uspec)
    
    def getuv(self, vrtdivspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        return self.ivsht(self.invlap * vrtdivspec / RAD_EARTH)

    def getgrad(self, chispec):
        """
        compute vector gradient on grid given complex spectral coefficients.

        Args:
            chispec: rank 1 or 2 or 3 tensor complex array with shape
        `(ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt)` containing
        complex spherical harmonic coefficients (where ntrunc is the
        triangular truncation limit and nt is the number of spectral arrays
        to be transformed). If chispec is rank 1, nt is assumed to be 1.

        Returns:
            C{B{uchi, vchi}} - rank 2 or 3 numpy float32 arrays containing
        gridded zonal and meridional components of the vector gradient.
        Shapes are either (nlat,nlon) or (nlat,nlon,nt).
        """
        idim = chispec.ndim

        if (
            len(chispec.shape) != 1
            and len(chispec.shape) != 2
            and len(chispec.shape) != 3
        ):
            msg = "getgrad needs rank one or two arrays!"
            raise ValueError(msg)

        ntrunc = int(
            -1.5
            + 0.5
            * torch.sqrt(
                9.0 - 8.0 * (1.0 - torch.tensor(self.nlat))
            )
        )

        if len(chispec.shape) == 1:
            chispec = torch.view(chispec, ((ntrunc + 1) * (ntrunc + 2) // 2, 1))

        divspec2 = self.lap * chispec

        if idim == 1:
            uchi, vchi = self.getuv(
                torch.stack(
                    (
                        torch.zeros([divspec2.shape[0], divspec2.shape[1]]),
                        divspec2,
                    )
                ).to(divspec2.device)
            )
            return torch.squeeze(uchi), torch.squeeze(vchi)
        elif idim == 2:
            uchi, vchi = self.getuv(
                torch.stack(
                    (
                        torch.zeros([divspec2.shape[0], divspec2.shape[1]]).to(divspec2.device),
                        divspec2,
                    )
                )
            )
            return uchi, vchi
        elif idim == 3:
            # new_shape = (divspec2.shape[0], 2, *divspec2.shape[1:])
            # stacked_divspec = torch.zeros(
            #     new_shape, dtype=torch.complex64
            # ).to(divspec2.device)
            # # Copy the original data into the second slice of the new dimension
            # stacked_divspec[:, 1, :, :] = divspec2
            stacked_divspec = torch.concat([divspec2.unsqueeze(1),
                                            divspec2.unsqueeze(1)],
                                            dim=1)

            backy = self.getuv(stacked_divspec)
            uchi = backy[:, 0, :, :]
            vchi = backy[:, 1, :, :]
            return uchi, vchi
        else:
            print("nothing happening here")

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
