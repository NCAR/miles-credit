import os
import gc
import sys
import yaml
import time
import logging
import warnings
import copy
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from collections import defaultdict
import cftime
from cftime import DatetimeNoLeap
import json
import pickle
import argparse
from typing import Dict



# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# ---------- #
import torch
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity
from torch import nn
from torchvision import transforms as tforms


# ---------- #
# credit
from credit.models import load_model, load_model_name
from credit.seed import seed_everything
from credit.loss import latitude_weights

from credit.data import (
    concat_and_reshape,
    reshape_only,
    drop_var_from_dataset,
    generate_datetime,
    nanoseconds_to_year,
    hour_to_nanoseconds,
    get_forward_data,
    extract_month_day_hour,
    find_common_indices,
)

from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.metrics import LatWeightedMetrics
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state
from credit.parser import credit_main_parser, predict_data_check
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer, GlobalDryMassFixer


from credit.physics_constants import (RAD_EARTH, GRAVITY, 
                                      RHO_WATER, LH_WATER, 
                                      RVGAS, RDGAS, CP_DRY, CP_VAPOR)


# credit
from credit.data import (
    Sample,
    concat_and_reshape,
    reshape_only,
    ERA5_and_Forcing_Dataset,
    get_forward_data
)

from credit.transforms import (
    Normalize_ERA5_and_Forcing,
    ToTensor_ERA5_and_Forcing,
    load_transforms
)

from credit.parser import (
    credit_main_parser,
    training_data_check
)

from credit.physics_core import physics_hybrid_sigma_level

from credit.physics_constants import (RAD_EARTH, GRAVITY, 
                                      RHO_WATER, LH_WATER, 
                                      RVGAS, RDGAS, CP_DRY, CP_VAPOR)

from credit.postblock import (
    PostBlock,
    SKEBS,
    TracerFixer,
    GlobalMassFixer,
    GlobalWaterFixer,
    GlobalEnergyFixer
)

RHO_WATER = torch.tensor(1000.0).to(torch.float64)
GRAVITY = torch.tensor(9.80665).to(torch.float64)

class GlobalDryMassFixer_here(nn.Module):
    '''
    This module applies global dry mass conservation fixes for the dry mass budget.
    The output ensures that the global dry air mass is conserved through correction ratios 
    applied during model runs. Variables `SP` (surface pressure) will be corrected to close the budget.
    All corrections are done using float32 PyTorch tensors.
    
    Args:
        post_conf (dict): config dictionary that includes all specs for the global dry mass fixer.
    '''
    
    def __init__(self, post_conf):
        super().__init__()

        # ------------------------------------------------------------------------------------ #
        # initialize physics computation
        print('DOING just DRY MASSS taco')
        # provide example data if it is a unit test
        if post_conf['global_drymass_fixer']['simple_demo']:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 
                               200, 220, 240, 260, 280, 300, 320, 340])
            
            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.flag_sigma_level = False
            self.flag_midpoint = post_conf['global_drymass_fixer']['midpoint']
            self.core_compute = physics_pressure_level_here(lon_demo, lat_demo, p_level_demo, 
                                                       midpoint=self.flag_midpoint)
            self.N_levels = len(p_level_demo)
            self.ind_fix = len(p_level_demo) - int(post_conf['global_drymass_fixer']['fix_level_num']) + 1
            self.requires_scaling = False
            
        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf['data']['save_loc_physics'])
            
            lon_lat_level_names = post_conf['global_drymass_fixer']['lon_lat_level_name']
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).to(torch.float64)
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).to(torch.float64)
            
            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = post_conf['global_drymass_fixer']['midpoint']
            
            if post_conf['global_drymass_fixer']['grid_type'] == 'sigma':
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).to(torch.float64)
                self.coef_b = torch.from_numpy(ds_physics[lon_lat_level_names[3]].values).to(torch.float64)
                
                # get total number of levels
                self.N_levels = len(self.coef_a)
                if self.flag_midpoint:
                    self.N_levels = self.N_levels -1
                    
                self.core_compute = physics_hybrid_sigma_level_here(lon2d.to(torch.float64), 
                                                               lat2d.to(torch.float64), 
                                                               self.coef_a.to(torch.float64), 
                                                               self.coef_b.to(torch.float64), 
                                                               midpoint=self.flag_midpoint)
            else:
                self.flag_sigma_level = False        
                p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                # get total number of levels
                self.N_levels = len(p_level)
                
                self.core_compute = physics_pressure_level(lon2d, 
                                                           lat2d, 
                                                           p_level, 
                                                           midpoint=self.flag_midpoint)

            self.requires_scaling = post_conf['requires_scaling']
            # -------------------------------------------------------------------------- #
            self.ind_fix = self.N_levels - int(post_conf['global_drymass_fixer']['fix_level_num']) + 1

        # -------------------------------------------------------------------------- #
        if self.flag_midpoint:
            self.ind_fix_start = self.ind_fix
        else:
            self.ind_fix_start = self.ind_fix-1
            
        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.q_ind_start = int(post_conf['global_drymass_fixer']['q_inds'][0])
        self.q_ind_end = int(post_conf['global_drymass_fixer']['q_inds'][-1]) + 1
        if self.flag_sigma_level:
            self.sp_ind = int(post_conf['global_drymass_fixer']['sp_inds'])
        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf['global_drymass_fixer']['denorm']:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None

        self.post_conf = post_conf
            
    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        x_input = x['x'].to(torch.float64)
        y_pred = x["y_pred"].to(torch.float64)

        # detach x_input
        x_input = x_input.detach().to(y_pred.device).to(torch.float64)

        # other needed inputs
        N_vars = y_pred.shape[1]
        
        # if denorm is needed
        if self.state_trans:
            print('state trans 1')
            x_input = self.state_trans.inverse_transform_input(x_input.to(torch.float64)).to(torch.float64)
            y_pred = self.state_trans.inverse_transform(y_pred.to(torch.float64)).to(torch.float64)

        # y_pred (batch, var, time, lat, lon) 
        # pick the first time-step, y_pred is expected to have the next step only
        # !!! Note: time dimension is collapsed throughout !!!
        
        q_input = x_input[:, self.q_ind_start:self.q_ind_end, -1, ...].to(torch.float64)
        q_pred = y_pred[:, self.q_ind_start:self.q_ind_end, 0, ...].to(torch.float64)

        if self.flag_sigma_level:
            sp_input = x_input[:, self.sp_ind, -1, ...].to(torch.float64)
            sp_pred = y_pred[:, self.sp_ind, 0, ...].to(torch.float64)

        if self.requires_scaling:
            #print('scaling coef Q: ',self.post_conf['scaling_coefs']['Q'])
            q_input = q_input*self.post_conf['scaling_coefs']['Q'].to(torch.float64)
            q_pred = q_pred*self.post_conf['scaling_coefs']['Q'].to(torch.float64)
            sp_input = sp_input*self.post_conf['scaling_coefs']['SP'].to(torch.float64)
            sp_pred = sp_pred*self.post_conf['scaling_coefs']['SP'].to(torch.float64)
        # ------------------------------------------------------------------------------ #
        # global dry air mass conservation
        # ------------------------------------------------------------------------------ #
        #print('GRAV', GRAVITY.dtype)
        if self.flag_sigma_level:
            #print('hey hey', GRAVITY.dtype)
            # total dry air mass from q_input
            #print('q_input', (1-q_input).shape)
            #print('sp_input' ,sp_input.shape)
            mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input.to(torch.float64), sp_input.to(torch.float64)).to(torch.float64)
            
            # total mass from q_pred
            mass_dry_sum_t1_hold = self.core_compute.weighted_sum(
                self.core_compute.integral_sliced(1-q_pred, sp_pred, 0, self.ind_fix).to(torch.float64) / GRAVITY, 
                axis=(-2, -1)).to(torch.float64)

            #print('1-q_pred', (1-q_pred).shape)
            #print('sp_pred' ,sp_pred.shape)
           
        else:
            # total dry air mass from q_input
            mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input)
        
            # total mass from q_pred
            mass_dry_sum_t1_hold = self.core_compute.weighted_sum(
                self.core_compute.integral_sliced(1-q_pred, 0, self.ind_fix) / GRAVITY, 
                axis=(-2, -1))
            
        # ===================================================================== #
        # surface pressure fixes on global dry air mass conservation
        # model level only
        
        if self.flag_sigma_level:
            #print('in GlobalDryMassFixer flag sigma level')
            delta_coef_a = self.coef_a.diff().to(q_pred.device).to(torch.float64)
            delta_coef_b = self.coef_b.diff().to(q_pred.device).to(torch.float64)

            #print('delta_coef_b.shape', delta_coef_b.shape)
            #print('delta_coef_a.shape', delta_coef_a.shape)
            #print('(1 - q_pred).shape:', (1 - q_pred).shape)
            
            if self.flag_midpoint:
                #print('in flag midpoint Dry')
                p_dry_a = ((delta_coef_a.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_pred)).sum(1)
                p_dry_b = ((delta_coef_b.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_pred)).sum(1)
            else:
                q_mid = (q_pred[:, :-1, ...] + q_pred[:, 1:, ...]) / 2
                p_dry_a = ((delta_coef_a.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_mid)).sum(1)
                p_dry_b = ((delta_coef_b.unsqueeze(0).unsqueeze(2).unsqueeze(3)) * (1 - q_mid)).sum(1)

            #print('p_dry_a.shape',p_dry_a.shape)

            grid_area = self.core_compute.area.unsqueeze(0).to(q_pred.device).to(torch.float64)

            #print('shape of calc', ((p_dry_a * grid_area).shape))
            
            mass_dry_a = (p_dry_a * grid_area).sum((-2, -1)) / GRAVITY
            mass_dry_b = (p_dry_b * sp_pred * grid_area).sum((-2, -1)) / GRAVITY

            #print('GRAVITY dtype', ((p_dry_a * grid_area).shape))
            
            # sp correction ratio using t0 dry air mass and t1 moisture
            #print('mass_dry_sum_t0 dtype:', mass_dry_sum_t0.dtype)
            #print('mass_dry_a dtype:', mass_dry_a.dtype)
            #print(' mass_dry_b dtype:',  mass_dry_b.dtype)
            #print('mass_dry_sum_t0 total:', mass_dry_sum_t0)
            #print('mass_dry_a total:', mass_dry_a)
            #print(' mass_dry_b total:',  mass_dry_b)

            
            sp_correct_ratio = (mass_dry_sum_t0 - mass_dry_a) / mass_dry_b
            
            #print('sp_correct_ratio:', sp_correct_ratio)
            #print('sp_correct_ratio dtype:', sp_correct_ratio.dtype)
            #print('sp_pred dtype:', sp_pred.dtype)
            sp_correct_ratio = sp_correct_ratio.unsqueeze(1).unsqueeze(2)
            sp_pred = sp_pred * sp_correct_ratio

            # Set the print options to display full precision
            torch.set_printoptions(precision=45, sci_mode=False)
            #print('!!!! sp_correct_ratio dry !!!!', sp_correct_ratio)

            # expand fixed vars to (batch, level, time, lat, lon)
            sp_pred = sp_pred.unsqueeze(1).unsqueeze(2)

            #print('1-q_pred', (1-q_pred).shape)
            #print('sp_pred' ,sp_pred.shape)

            # mass_dry_sum_t0 = self.core_compute.total_dry_air_mass(q_input.to(torch.float64), sp_input.to(torch.float64)).to(torch.float64)
            mass_dry_sum_after = self.core_compute.total_dry_air_mass(q_pred.to(torch.float64), sp_pred[0,0,:,:,:].to(torch.float64)).to(torch.float64)

            #print(f'{mass_dry_sum_t0} {mass_dry_sum_after} diff: {np.array(mass_dry_sum_t0 - mass_dry_sum_after)}')
            
            if self.requires_scaling:
                sp_input = sp_input/self.post_conf['scaling_coefs']['SP']
                sp_pred = sp_pred/self.post_conf['scaling_coefs']['SP']
            
            y_pred = concat_fix(y_pred, sp_pred, self.sp_ind, self.sp_ind, N_vars)
            
        # ===================================================================== #        
        if self.state_trans:
            #print(y_pred.dtype)
            y_pred = self.state_trans.transform_array(y_pred)
            #print(y_pred.dtype)
        
        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x

class GlobalWaterFixer_here(nn.Module):
    
    def __init__(self, post_conf):
        super().__init__()
        torch.set_default_dtype(torch.float64)  # Use higher precision for all tensors
        # ------------------------------------------------------------------------------------ #
        # initialize physics computation

        # provide example data if it is a unit test
        if post_conf['global_water_fixer']['simple_demo']:
            y_demo = np.array([90, 70, 50, 30, 10, -10, -30, -50, -70, -90])
            x_demo = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 
                               200, 220, 240, 260, 280, 300, 320, 340])
            
            lon_demo, lat_demo = np.meshgrid(x_demo, y_demo)
            lon_demo = torch.from_numpy(lon_demo)
            lat_demo = torch.from_numpy(lat_demo)

            p_level_demo = torch.from_numpy(np.array([100, 30000, 50000, 70000, 80000, 90000, 100000]))
            self.flag_sigma_level = False
            self.flag_midpoint = post_conf['global_water_fixer']['midpoint']
            self.core_compute = physics_pressure_level_here(lon_demo, lat_demo, p_level_demo, 
                                                       midpoint=self.flag_midpoint)
            self.N_levels = len(p_level_demo)
            self.N_seconds = int(post_conf['data']['lead_time_periods']) * 3600
            self.requires_scaling = False
            
        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf['data']['save_loc_physics'])
            
            lon_lat_level_names = post_conf['global_mass_fixer']['lon_lat_level_name']
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).to(torch.float64)
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).to(torch.float64)
            
            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = post_conf['global_mass_fixer']['midpoint']
            
            if post_conf['global_mass_fixer']['grid_type'] == 'sigma':
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).to(torch.float64)
                self.coef_b = torch.from_numpy(ds_physics[lon_lat_level_names[3]].values).to(torch.float64)
                
                # get total number of levels
                self.N_levels = len(self.coef_a)
                
                if self.flag_midpoint:
                    self.N_levels = self.N_levels -1
                    
                self.core_compute = physics_hybrid_sigma_level_here(lon2d.to(torch.float64), 
                                                               lat2d.to(torch.float64), 
                                                               self.coef_a.to(torch.float64), 
                                                               self.coef_b.to(torch.float64), 
                                                               midpoint=self.flag_midpoint)
            else:
                self.flag_sigma_level = False        
                p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).float()
                # get total number of levels
                self.N_levels = len(p_level)
                
                self.core_compute = physics_pressure_level(lon2d, 
                                                           lat2d, 
                                                           p_level, 
                                                           midpoint=self.flag_midpoint)

            self.requires_scaling = post_conf['requires_scaling']
            self.N_seconds = torch.tensor(int(post_conf['data']['lead_time_periods']) * 3600.).to(torch.float64)
            
        # ------------------------------------------------------------------------------------ #
        # identify variables of interest
        self.q_ind_start = int(post_conf['global_water_fixer']['q_inds'][0])
        self.q_ind_end = int(post_conf['global_water_fixer']['q_inds'][-1]) + 1
        self.precip_ind = int(post_conf['global_water_fixer']['precip_ind'])
        self.evapor_ind = int(post_conf['global_water_fixer']['evapor_ind'])
        if self.flag_sigma_level:
            self.sp_ind = int(post_conf['global_drymass_fixer']['sp_inds'])            
        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf['global_water_fixer']['denorm']:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None
        self.post_conf = post_conf
            
    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors
        torch.set_default_dtype(torch.float64)  # Use higher precision for all tensors
        # x_input (batch, var, time, lat, lon)
        # x_input does not have precip and evapor
        x_input = x['x'].to(torch.float64)
        y_pred = x["y_pred"].to(torch.float64)

        # detach x_input
        x_input = x_input.detach().to(y_pred.device)

        # other needed inputs
        N_vars = y_pred.shape[1]
        
        # if denorm is needed
        if self.state_trans:
            x_input = self.state_trans.inverse_transform_input(x_input).to(torch.float64)
            y_pred = self.state_trans.inverse_transform(y_pred).to(torch.float64)
            
        q_input = x_input[:, self.q_ind_start:self.q_ind_end, -1, ...].to(torch.float64)
        
        # y_pred (batch, var, time, lat, lon) 
        # pick the first time-step, y_pred is expected to have the next step only
        q_pred = y_pred[:, self.q_ind_start:self.q_ind_end, 0, ...].to(torch.float64)
        precip = y_pred[:, self.precip_ind, 0, ...].to(torch.float64)
        evapor = y_pred[:, self.evapor_ind, 0, ...].to(torch.float64)


        if self.flag_sigma_level:
            sp_input = x_input[:, self.sp_ind, -1, ...].to(torch.float64)
            sp_pred = y_pred[:, self.sp_ind, 0, ...].to(torch.float64)

        if self.requires_scaling:
            q_input = q_input*self.post_conf['scaling_coefs']['Q']
            q_pred = q_pred*self.post_conf['scaling_coefs']['Q']
            sp_input = sp_input*self.post_conf['scaling_coefs']['SP']
            sp_pred = sp_pred*self.post_conf['scaling_coefs']['SP']
            evapor = evapor*self.post_conf['scaling_coefs']['evap'] 
            precip = precip*self.post_conf['scaling_coefs']['tot_precip'] 
                    
        # ------------------------------------------------------------------------------ #
        # global water balance
        precip_flux = precip * RHO_WATER.to(torch.float64) / self.N_seconds
        evapor_flux = evapor * RHO_WATER.to(torch.float64) / self.N_seconds
        
        # total water content (batch, var, time, lat, lon)
        if self.flag_sigma_level:
            TWC_input = self.core_compute.total_column_water(q_input, sp_input).to(torch.float64)
            TWC_pred = self.core_compute.total_column_water(q_pred, sp_pred).to(torch.float64)
        else:
            TWC_input = self.core_compute.total_column_water(q_input).to(torch.float64)
            TWC_pred = self.core_compute.total_column_water(q_pred).to(torch.float64)


        # Initial step
        step = 0
        
        # # Loop to find an available step
        # while os.path.exists(f'/glade/work/wchapman/miles_branchs/CESM_physics/q_input_{step:03}.pt'):
        #     step += 1
        # torch.save(q_input, f'/glade/work/wchapman/miles_branchs/CESM_physics/q_input_{step:03}.pt')
        # torch.save(sp_input, f'/glade/work/wchapman/miles_branchs/CESM_physics/sp_input_{step:03}.pt')
        # torch.save(q_pred, f'/glade/work/wchapman/miles_branchs/CESM_physics/q_pred_{step:03}.pt')
        # torch.save(sp_pred, f'/glade/work/wchapman/miles_branchs/CESM_physics/sp_pred_{step:03}.pt')
        # torch.save(precip, f'/glade/work/wchapman/miles_branchs/CESM_physics/precip_input_{step:03}.pt')
        # torch.save(evapor, f'/glade/work/wchapman/miles_branchs/CESM_physics/evapor_input_{step:03}.pt')
        
        dTWC_dt = (TWC_pred - TWC_input) / self.N_seconds
        
        # global sum of total water content tendency
        TWC_sum = self.core_compute.weighted_sum(dTWC_dt, axis=(-2, -1))
        #print('dTWC_dt: ', TWC_sum )
        
        # global evaporation source
        E_sum = self.core_compute.weighted_sum(evapor_flux, axis=(-2, -1))

        #print('Esum:', E_sum)
        
        # global precip sink
        P_sum = self.core_compute.weighted_sum(precip_flux, axis=(-2, -1))

        #print('Psum:', P_sum)
        
        # global water balance residual
        residual = -TWC_sum.to(torch.float64) - E_sum.to(torch.float64) - P_sum.to(torch.float64)

        #print('resid:', residual.dtype)

        # compute correction ratio
        P_correct_ratio = (P_sum.to(torch.float64) + residual.to(torch.float64)) / P_sum.to(torch.float64)
        #P_correct_ratio = torch.clamp(P_correct_ratio, min=0.9, max=1.1)
        # broadcast: (batch_size, 1, 1, 1)

        # print('water fix P_correct_ratio :', P_correct_ratio)
        P_correct_ratio = P_correct_ratio.to(torch.float64).unsqueeze(-1).unsqueeze(-1)

        #print('P correct ratio:', P_correct_ratio)
        
        precip = precip.to(torch.float64) * P_correct_ratio

        # apply correction on precip
        if self.requires_scaling:
            precip = precip /self.post_conf['scaling_coefs']['tot_precip'] 
            q_input = q_input/self.post_conf['scaling_coefs']['Q']
            q_pred = q_pred/self.post_conf['scaling_coefs']['Q']
            sp_input = sp_input/self.post_conf['scaling_coefs']['SP']
            sp_pred = sp_pred/self.post_conf['scaling_coefs']['SP']
            evapor = evapor/self.post_conf['scaling_coefs']['evap'] 

        # ===================================================================== #
        # return fixed precip back to y_pred
        precip = precip.unsqueeze(1).unsqueeze(2)
        precip = precip.to(torch.float64)
        y_pred = concat_fix(y_pred, precip, self.precip_ind, self.precip_ind, N_vars)
        
        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred).to(torch.float64)
        
        # give it back to x
        x["y_pred"] = y_pred.to(torch.float64)
        
        # return dict, 'x' is not touched
        return x


class GlobalEnergyFixer_here(nn.Module):
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
        print('DOING just ENERGY taco')
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
            self.flag_midpoint = post_conf['global_energy_fixer']['midpoint']
            self.core_compute = physics_pressure_level(
                lon_demo,
                lat_demo,
                p_level_demo,
                midpoint=self.flag_midpoint,
            )
            self.N_seconds = torch.tensor(int(post_conf['data']['lead_time_periods']) * 3600.).to(torch.float64)

            gph_surf_demo = np.ones((10, 18))
            self.GPH_surf = torch.from_numpy(gph_surf_demo)
            self.requires_scaling = False

        else:
            # the actual setup for model runs
            ds_physics = get_forward_data(post_conf['data']['save_loc_physics'])
            
            lon_lat_level_names = post_conf['global_energy_fixer']['lon_lat_level_name']
            lon2d = torch.from_numpy(ds_physics[lon_lat_level_names[0]].values).to(torch.float64)
            lat2d = torch.from_numpy(ds_physics[lon_lat_level_names[1]].values).to(torch.float64)
            
            # -------------------------------------------------------------------------- #
            # pick physics core
            self.flag_midpoint = post_conf['global_energy_fixer']['midpoint']
            
            if post_conf['global_energy_fixer']['grid_type'] == 'sigma':
                self.flag_sigma_level = True
                self.coef_a = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).to(torch.float64)
                self.coef_b = torch.from_numpy(ds_physics[lon_lat_level_names[3]].values).to(torch.float64)
                
                # get total number of levels
                self.N_levels = len(self.coef_a)
                
                if self.flag_midpoint:
                    self.N_levels = self.N_levels -1
                    
                self.core_compute = physics_hybrid_sigma_level_here(lon2d, 
                                                               lat2d, 
                                                               self.coef_a, 
                                                               self.coef_b, 
                                                               midpoint=self.flag_midpoint)
            else:
                self.flag_sigma_level = False        
                p_level = torch.from_numpy(ds_physics[lon_lat_level_names[2]].values).to(torch.float64)
                # get total number of levels
                self.N_levels = len(p_level)
                
                self.core_compute = physics_pressure_level(lon2d, 
                                                           lat2d, 
                                                           p_level, 
                                                           midpoint=self.flag_midpoint)
                
            self.N_seconds = torch.tensor(int(post_conf['data']['lead_time_periods']) * 3600.).to(torch.float64)

            varname_gph = post_conf["global_energy_fixer"]["surface_geopotential_name"]
            self.GPH_surf = torch.from_numpy(ds_physics[varname_gph[0]].values).to(torch.float64)

            self.requires_scaling = post_conf['requires_scaling']
            
            if self.requires_scaling:
                self.GPH_surf = self.GPH_surf*torch.tensor(post_conf['scaling_coefs']['gph_surf']).to(torch.float64)

            self.post_conf = post_conf

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
            self.sp_ind = int(post_conf['global_energy_fixer']['sp_inds'])  
        # ------------------------------------------------------------------------------------ #
        # setup a scaler
        if post_conf["global_energy_fixer"]["denorm"]:
            self.state_trans = load_transforms(post_conf, scaler_only=True)
        else:
            self.state_trans = None
        self.post_conf = post_conf
    def forward(self, x):
        # ------------------------------------------------------------------------------ #
        # get tensors

        # x_input (batch, var, time, lat, lon)
        # x_input does not have precip and evapor
        x_input = x["x"].to(torch.float64)
        y_pred = x["y_pred"].to(torch.float64)

        # detach x_input
        x_input = x_input.detach().to(y_pred.device).to(torch.float64)

        # other needed inputs
        GPH_surf = self.GPH_surf.to(y_pred.device).to(torch.float64)
        N_vars = y_pred.shape[1]

        # if denorm is needed
        if self.state_trans:
            x_input = self.state_trans.inverse_transform_input(x_input).to(torch.float64)
            y_pred = self.state_trans.inverse_transform(y_pred).to(torch.float64)

        T_input = x_input[:, self.T_ind_start : self.T_ind_end, -1, ...].to(torch.float64)
        q_input = x_input[:, self.q_ind_start : self.q_ind_end, -1, ...].to(torch.float64)
        U_input = x_input[:, self.U_ind_start : self.U_ind_end, -1, ...].to(torch.float64)
        V_input = x_input[:, self.V_ind_start : self.V_ind_end, -1, ...].to(torch.float64)

        # y_pred (batch, var, time, lat, lon)
        # pick the first time-step, y_pred is expected to have the next step only
        T_pred = y_pred[:, self.T_ind_start : self.T_ind_end, 0, ...].to(torch.float64)
        q_pred = y_pred[:, self.q_ind_start : self.q_ind_end, 0, ...].to(torch.float64)
        U_pred = y_pred[:, self.U_ind_start : self.U_ind_end, 0, ...].to(torch.float64)
        V_pred = y_pred[:, self.V_ind_start : self.V_ind_end, 0, ...].to(torch.float64)

        TOA_solar_pred = y_pred[:, self.TOA_solar_ind, 0, ...].to(torch.float64)
        TOA_OLR_pred = y_pred[:, self.TOA_OLR_ind, 0, ...].to(torch.float64)

        surf_solar_pred = y_pred[:, self.surf_solar_ind, 0, ...].to(torch.float64)
        surf_LR_pred = y_pred[:, self.surf_LR_ind, 0, ...].to(torch.float64)
        surf_SH_pred = y_pred[:, self.surf_SH_ind, 0, ...].to(torch.float64)
        surf_LH_pred = y_pred[:, self.surf_LH_ind, 0, ...].to(torch.float64)


        if self.flag_sigma_level:
            sp_input = x_input[:, self.sp_ind, -1, ...].to(torch.float64)
            sp_pred = y_pred[:, self.sp_ind, 0, ...].to(torch.float64)

        if self.requires_scaling:
            q_input = q_input*torch.tensor(self.post_conf['scaling_coefs']['Q']).to(torch.float64)
            q_pred = q_pred*torch.tensor(self.post_conf['scaling_coefs']['Q']).to(torch.float64)
            T_input = T_input*torch.tensor(self.post_conf['scaling_coefs']['T']).to(torch.float64)
            T_pred = T_pred*torch.tensor(self.post_conf['scaling_coefs']['T']).to(torch.float64)
            U_input = U_input*torch.tensor(self.post_conf['scaling_coefs']['U']).to(torch.float64)
            U_pred = U_pred*torch.tensor(self.post_conf['scaling_coefs']['U']).to(torch.float64)
            V_input = V_input*torch.tensor(self.post_conf['scaling_coefs']['V']).to(torch.float64)
            V_pred = V_pred*torch.tensor(self.post_conf['scaling_coefs']['V']).to(torch.float64)
            TOA_solar_pred = TOA_solar_pred*torch.tensor(self.post_conf['scaling_coefs']['top_net_solar']).to(torch.float64)
            TOA_OLR_pred = TOA_OLR_pred*torch.tensor(self.post_conf['scaling_coefs']['top_net_therm']).to(torch.float64)
            surf_solar_pred = surf_solar_pred*torch.tensor(self.post_conf['scaling_coefs']['surf_net_solar']).to(torch.float64)
            surf_LR_pred = surf_LR_pred*torch.tensor(self.post_conf['scaling_coefs']['surf_net_therm']).to(torch.float64)
            surf_SH_pred = surf_SH_pred*torch.tensor(self.post_conf['scaling_coefs']['surf_shflx']).to(torch.float64)
            surf_LH_pred = surf_LH_pred*torch.tensor(self.post_conf['scaling_coefs']['surf_lhflx']).to(torch.float64)
            sp_input = sp_input*torch.tensor(self.post_conf['scaling_coefs']['SP']).to(torch.float64)
            sp_pred = sp_pred*torch.tensor(self.post_conf['scaling_coefs']['SP']).to(torch.float64)
            
        
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

        if self.requires_scaling:
            q_input = q_input/torch.tensor(self.post_conf['scaling_coefs']['Q']).to(torch.float64)
            q_pred = q_pred/torch.tensor(self.post_conf['scaling_coefs']['Q']).to(torch.float64)
            T_input = T_input/torch.tensor(self.post_conf['scaling_coefs']['T']).to(torch.float64)
            T_pred = T_pred/torch.tensor(self.post_conf['scaling_coefs']['T']).to(torch.float64)
            U_input = U_input/torch.tensor(self.post_conf['scaling_coefs']['U']).to(torch.float64)
            U_pred = U_pred/torch.tensor(self.post_conf['scaling_coefs']['U']).to(torch.float64)
            V_input = V_input/torch.tensor(self.post_conf['scaling_coefs']['V']).to(torch.float64)
            V_pred = V_pred/torch.tensor(self.post_conf['scaling_coefs']['V']).to(torch.float64)
            TOA_solar_pred = TOA_solar_pred/torch.tensor(self.post_conf['scaling_coefs']['top_net_solar']).to(torch.float64)
            TOA_OLR_pred = TOA_OLR_pred/torch.tensor(self.post_conf['scaling_coefs']['top_net_therm']).to(torch.float64)
            surf_solar_pred = surf_solar_pred/torch.tensor(self.post_conf['scaling_coefs']['surf_net_solar']).to(torch.float64)
            surf_LR_pred = surf_LR_pred/torch.tensor(self.post_conf['scaling_coefs']['surf_net_therm']).to(torch.float64)
            surf_SH_pred = surf_SH_pred/torch.tensor(self.post_conf['scaling_coefs']['surf_shflx']).to(torch.float64)
            surf_LH_pred = surf_LH_pred/torch.tensor(self.post_conf['scaling_coefs']['surf_lhflx']).to(torch.float64)
            sp_input = sp_input/torch.tensor(self.post_conf['scaling_coefs']['SP']).to(torch.float64)
            sp_pred = sp_pred/torch.tensor(self.post_conf['scaling_coefs']['SP']).to(torch.float64)

        y_pred = concat_fix(y_pred, T_pred, self.T_ind_start, self.T_ind_end, N_vars)

        if self.state_trans:
            y_pred = self.state_trans.transform_array(y_pred)

        # give it back to x
        x["y_pred"] = y_pred

        # return dict, 'x' is not touched
        return x


class physics_hybrid_sigma_level_here:
    '''
    Hybrid sigma-pressure level physics

    Attributes:
        lon (torch.Tensor): Longitude in degrees.
        lat (torch.Tensor): Latitude in degrees.
        surface_pressure (torch.Tensor): Surface pressure in Pa.
        coef_a (torch.Tensor): Hybrid sigma-pressure coefficient 'a' [Pa].
        coef_b (torch.Tensor): Hybrid sigma-pressure coefficient 'b' [unitless].
        area (torch.Tensor): Area of grid cells [m^2].
        integral (function): Vertical integration method (midpoint or trapezoidal).
    '''

    def __init__(self,
                 lon: torch.Tensor,
                 lat: torch.Tensor,
                 coef_a: torch.Tensor,
                 coef_b: torch.Tensor,
                 midpoint: bool = False):
        '''
        Initialize the class with longitude, latitude, and hybrid sigma-pressure levels.
        All inputs must be on the same torch device.
        Full order of dimensions: (batch, level, time, latitude, longitude)
        Accepted dimensions: (batch, level, latitude, longitude)
        
        Args:
            lon (torch.Tensor): Longitude in degrees.
            lat (torch.Tensor): Latitude in degrees.
            coef_a (torch.Tensor): Hybrid sigma-pressure coefficient 'a' [Pa] (level,).
            coef_b (torch.Tensor): Hybrid sigma-pressure coefficient 'b' [unitless] (level,).
            midpoint (bool): True if vertical level quantities are midpoint values; otherwise False.

        Note:
            pressure = coef_a + coef_b * surface_pressure
        '''
        self.lon = lon.to(torch.float64)
        self.lat = lat.to(torch.float64)
        self.coef_a = coef_a.to(torch.float64)  # (level,)
        self.coef_b = coef_b.to(torch.float64)  # (level,)

        # ========================================================================= #
        # Compute pressure on each hybrid sigma level
        # Reshape coef_a and coef_b for broadcasting
        self.coef_a = coef_a.view(1, -1, 1, 1)  # (1, level, 1, 1)
        self.coef_b = coef_b.view(1, -1, 1, 1)  # (1, level, 1, 1)
        
        # ========================================================================= #
        # compute gtid area
        # area = R^2 * d_sin(lat) * d_lon
        lat_rad = torch.deg2rad(self.lat).to(torch.float64)
        lon_rad = torch.deg2rad(self.lon).to(torch.float64)
        sin_lat_rad = torch.sin(lat_rad)
        d_phi = torch.gradient(sin_lat_rad, dim=0, edge_order=2)[0].to(torch.float64)
        d_lambda = torch.gradient(lon_rad, dim=1, edge_order=2)[0].to(torch.float64)
        d_lambda = (d_lambda + torch.pi) % (2 * torch.pi) - torch.pi
        self.area = torch.abs(RAD_EARTH**2 * d_phi * d_lambda).to(torch.float64)

        # ========================================================================== #
        # Vertical integration method
        if midpoint:
            self.integral = self.pressure_integral_midpoint
            self.integral_sliced = self.pressure_integral_midpoint_sliced
        else:
            self.integral = self.pressure_integral_trapz
            self.integral_sliced = self.pressure_integral_trapz_sliced

    def pressure_integral_midpoint(self, 
                                   q_mid: torch.Tensor,
                                   surface_pressure: torch.Tensor,) -> torch.Tensor:
        '''
        Compute the pressure level integral of a given quantity; assuming its mid-point
        values are pre-computed.

        Args:
            q_mid: The quantity with dims of (batch, level-1, time, latitude, longitude)
            surface_pressure: Surface pressure in Pa (batch, time, latitude, longitude).

        Returns:
            Pressure level integrals of q
        '''
        # (batch, 1, lat, lon)
        surface_pressure = surface_pressure.to(torch.float64).unsqueeze(1)

        # (batch, level, lat, lon)
        pressure = self.coef_a.to(q_mid.device) + self.coef_b.to(q_mid.device) * surface_pressure

        # (batch, level-1, lat, lon)
        delta_p = pressure.diff(dim=1).to(q_mid.device).to(torch.float64)

        # Element-wise multiplication
        q_area = q_mid.to(torch.float64) * delta_p

        # Sum over level dimension
        q_integral = torch.sum(q_area, dim=1)
        
        return q_integral
        
    def pressure_integral_midpoint_sliced(self,
                                          q_mid: torch.Tensor,
                                          surface_pressure: torch.Tensor,
                                          ind_start: int,
                                          ind_end: int) -> torch.Tensor:
        '''
        As in `pressure_integral_midpoint`, but supports pressure level indexing,
        so it can calculate integrals of a subset of levels.
        '''
        # (batch, 1, lat, lon)
        surface_pressure = surface_pressure.unsqueeze(1)

        # (batch, level, lat, lon)
        pressure = self.coef_a.to(q_mid.device) + self.coef_b.to(q_mid.device) * surface_pressure

        # (batch, level-1, lat, lon)
        pressure_thickness = pressure.diff(dim=1)
        
        delta_p = pressure_thickness[:, ind_start:ind_end, ...].to(q_mid.device)
        
        q_mid_sliced = q_mid[:, ind_start:ind_end, ...]
        q_area = q_mid_sliced * delta_p
        q_integral = torch.sum(q_area, dim=1)
        return q_integral

    def pressure_integral_trapz(self, 
                                q: torch.Tensor,
                                surface_pressure: torch.Tensor) -> torch.Tensor:
        '''
        Compute the pressure level integral of a given quantity using the trapezoidal rule.

        Args:
            q: The quantity with dims of (batch, level, time, latitude, longitude)

        Returns:
            Pressure level integrals of q
        '''
        # (batch, 1, lat, lon)
        surface_pressure = surface_pressure.unsqueeze(1)

        # (batch, level, lat, lon)
        pressure = self.coef_a.to(q.device) + self.coef_b.to(q.device) * surface_pressure

        # (batch, level-1, lat, lon)
        delta_p = pressure.diff(dim=1).to(q.device)

        # trapz
        q1 = q[:, :-1, ...]
        q2 = q[:, 1:, ...]
        q_area = 0.5 * (q1 + q2) * delta_p
        q_trapz = torch.sum(q_area, dim=1)
        
        return q_trapz

    def pressure_integral_trapz_sliced(self,
                                       q: torch.Tensor,
                                       surface_pressure: torch.Tensor,
                                       ind_start: int,
                                       ind_end: int) -> torch.Tensor:
        '''
        As in `pressure_integral_trapz`, but supports pressure level indexing,
        so it can calculate integrals of a subset of levels.
        '''
        # (batch, 1, lat, lon)
        surface_pressure = surface_pressure.unsqueeze(1)

        # (batch, level, lat, lon)
        pressure = self.coef_a.to(q.device) + self.coef_b.to(q.device) * surface_pressure
        
        delta_p = pressure[:, ind_start:ind_end, ...].diff(dim=1).to(q.device)

        # trapz
        q_slice = q[:, ind_start:ind_end, ...]
        q1 = q_slice[:, :-1, ...]
        q2 = q_slice[:, 1:, ...]
        q_area = 0.5 * (q1 + q2) * delta_p
        q_trapz = torch.sum(q_area, dim=1)
        
        return q_trapz

    def weighted_sum(self,
                     q: torch.Tensor,
                     axis: Dict[tuple, None] = None, 
                     keepdims: bool = False) -> torch.Tensor:
        '''
        Compute the weighted sum of a given quantity for PyTorch tensors.
        
        Args:
            data: the quantity to be summed (PyTorch tensor)
            axis: dims to compute the sum (can be int or tuple of ints)
            keepdims: whether to keep the reduced dimensions or not
    
        Returns:
            Weighted sum (PyTorch tensor)
        '''
        q_w = q.to(torch.float64) * self.area.to(torch.float64).to(q.device)
        q_sum = torch.sum(q_w, dim=axis, keepdim=keepdims)
        return q_sum

    def total_dry_air_mass(self, 
                           q: torch.Tensor,
                           surface_pressure: torch.Tensor) -> torch.Tensor:
        '''
        Compute the total mass of dry air over the entire globe [kg]
        '''
        mass_dry_per_area = self.integral(1-q.to(torch.float64), surface_pressure.to(torch.float64)) / GRAVITY # kg/m^2
        # weighted sum on latitude and longitude dimensions
        mass_dry_sum = self.weighted_sum(mass_dry_per_area, axis=(-2, -1)) # kg
        
        return mass_dry_sum

    def total_column_water(self, 
                           q: torch.Tensor,
                           surface_pressure: torch.Tensor,) -> torch.Tensor:
        '''
        Compute total column water (TCW) per air column [kg/m2]
        '''
        TWC = self.integral(q, surface_pressure) / GRAVITY # kg/m^2
        
        return TWC


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


class TracerFixer_here(nn.Module):
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

        print('tracer inds:', self.tracer_indices, 'len:', len(self.tracer_indices))

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


def mass_residual_verif(x, y_pred, opt, post_conf):
    state_trans = load_transforms(post_conf, scaler_only=True)
    
    x = state_trans.inverse_transform_input(x.to(torch.float64))
    y_pred = state_trans.inverse_transform(y_pred.to(torch.float64))
    
    q_ind_start = opt.q_ind_start
    q_ind_end = opt.q_ind_end
    sp_ind = opt.sp_ind
    sp_input = x[:, sp_ind, -1, ...]
    sp_pred = y_pred[:, sp_ind, 0, ...]
    
    ds_physics = get_forward_data(post_conf['data']['save_loc_physics'])        
    lon2d = torch.from_numpy(ds_physics['lon2d'].values).to(torch.float64)
    lat2d = torch.from_numpy(ds_physics['lat2d'].values).to(torch.float64)
    coef_a = torch.from_numpy(ds_physics['hyai'].values).to(torch.float64)
    coef_b = torch.from_numpy(ds_physics['hybi'].values).to(torch.float64)
    
    core_compute = physics_hybrid_sigma_level_here(lon2d, lat2d, coef_a, coef_b, midpoint=True)
    
    mass_dry_sum_t0 = core_compute.total_dry_air_mass(x[:, q_ind_start:q_ind_end, -1, ...].to(torch.float64), sp_input.to(torch.float64))
    mass_dry_sum_t1 = core_compute.total_dry_air_mass(y_pred[:, q_ind_start:q_ind_end, 0, ...].to(torch.float64), sp_pred.to(torch.float64))
    mass_residual = mass_dry_sum_t1 - mass_dry_sum_t0

    # print(f"Residual [float64]: {np.array(mass_residual):.16e}")
    mass_residual_np = mass_residual.detach().cpu().numpy()
    #print(mass_residual_np.dtype)
    
    #print(f'Residual to conserve mass budget [kg]: {mass_residual}')
    #print(f"Residual [float64]: {mass_residual_np[0]:.6f}")  # Format as float with 6 decimal places
    return mass_residual, mass_dry_sum_t1, mass_dry_sum_t0

def water_budget_verif(x, y_pred, opt, post_conf):
    x = x.to(torch.float64)
    y_pred = y_pred.to(torch.float64)
    
    state_trans = load_transforms(post_conf, scaler_only=True)
    RHO_WATER = 1000.
    N_seconds = torch.tensor(3600 * 6, dtype=torch.float64)
    RHO_WATER = torch.tensor(RHO_WATER, dtype=torch.float64)
    
    x = state_trans.inverse_transform_input(x)
    y_pred = state_trans.inverse_transform(y_pred)
    y_pred = y_pred.to(torch.float64)
    x = x.to(torch.float64)


    precip_ind = opt.precip_ind
    q_ind_start = opt.q_ind_start
    q_ind_end = opt.q_ind_end
    evapor_ind = opt.evapor_ind
    sp_ind = opt.sp_ind
    
    sp_input = x[:, sp_ind, -1, ...]
    sp_pred = y_pred[:, sp_ind, 0, ...]
    
    ds_physics = get_forward_data(post_conf['data']['save_loc_physics'])
    lon2d = torch.from_numpy(ds_physics['lon2d'].values).to(torch.float64)
    lat2d = torch.from_numpy(ds_physics['lat2d'].values).to(torch.float64)
    coef_a = torch.from_numpy(ds_physics['hyai'].values).to(torch.float64)
    coef_b = torch.from_numpy(ds_physics['hybi'].values).to(torch.float64)
    
    core_compute = physics_hybrid_sigma_level_here(lon2d.to(torch.float64), 
    lat2d.to(torch.float64), 
    coef_a.to(torch.float64), 
    coef_b.to(torch.float64), 
    midpoint=True
    )

    q_input = x[:, q_ind_start:q_ind_end, -1, ...]
    q_pred = y_pred[:, q_ind_start:q_ind_end, 0, ...]
    precip = y_pred[:, precip_ind, 0, ...]
    evapor = y_pred[:, evapor_ind, 0, ...]

    precip_flux = (precip * RHO_WATER / N_seconds).to(torch.float64)
    evapor_flux = (evapor * RHO_WATER / N_seconds).to(torch.float64)
    
    TWC_input = core_compute.total_column_water(q_input, sp_input)
    TWC_pred = core_compute.total_column_water(q_pred, sp_pred)
    dTWC_dt = (TWC_pred - TWC_input) / N_seconds

    TWC_sum = core_compute.weighted_sum(dTWC_dt, axis=(-2, -1)).to(torch.float64)
    E_sum = core_compute.weighted_sum(evapor_flux, axis=(-2, -1)).to(torch.float64)
    P_sum = core_compute.weighted_sum(precip_flux, axis=(-2, -1)).to(torch.float64)

    residual = -TWC_sum - E_sum - P_sum

    # If residual is below threshold, set to zero
    residual_threshold = 1e-5
    if abs(residual.item()) < residual_threshold:
        residual = torch.tensor(0.0, dtype=torch.float64)

    P_correct_ratio = ((P_sum + residual) / P_sum).to(torch.float64)
    # P_correct_ratio = torch.clamp(P_correct_ratio, min=1e-10, max=1e10)
    P_correct_ratio = P_correct_ratio.unsqueeze(-1).unsqueeze(-1).to(torch.float64)

    adjusted_P_sum = P_sum + residual  # Recompute adjusted sum for debugging

    #print('TWC_input dtype:', TWC_input.dtype)
    #print('TWC_pred dtype:', TWC_pred.dtype)
    #print(f"Residual [float64]: {residual.item():.16e}")
    #print('P correct ratio:', P_correct_ratio, 1 - P_correct_ratio)
    #print(f"P_correct_ratio [float64]: {P_correct_ratio.item():.16e}")
    #print(f'Residual to conserve water budget [kg]: {residual.item():.16e}')
    #print(f"Adjusted P_sum [float64]: {adjusted_P_sum.item():.16e}")

    return residual, adjusted_P_sum, P_sum

def energy_residual_verif(x, y_pred, opt, post_conf):

    state_trans = load_transforms(post_conf, scaler_only=True)
    
    x = state_trans.inverse_transform_input(x).to(torch.float64)
    y_pred = state_trans.inverse_transform(y_pred).to(torch.float64)
    
    N_seconds = torch.tensor(3600 * 6).to(torch.float64)
    
    T_ind_start = opt.T_ind_start
    T_ind_end = opt.T_ind_end
    
    q_ind_start = opt.q_ind_start
    q_ind_end = opt.q_ind_end
    
    U_ind_start = opt.U_ind_start
    U_ind_end = opt.U_ind_end
    
    V_ind_start = opt.V_ind_start
    V_ind_end = opt.V_ind_end
    
    TOA_solar_ind = opt.TOA_solar_ind
    TOA_OLR_ind = opt.TOA_OLR_ind
    
    surf_solar_ind = opt.surf_solar_ind
    surf_LR_ind = opt.surf_LR_ind
    
    surf_SH_ind = opt.surf_SH_ind
    surf_LH_ind = opt.surf_LH_ind

    sp_ind = opt.sp_ind

    sp_input = x[:, sp_ind, -1, ...].to(torch.float64)
    sp_pred = y_pred[:, sp_ind, 0, ...].to(torch.float64)
    
    ds_physics = get_forward_data(post_conf['data']['save_loc_physics'])        
    lon2d = torch.from_numpy(ds_physics['lon2d'].values).to(torch.float64)
    lat2d = torch.from_numpy(ds_physics['lat2d'].values).to(torch.float64)
    coef_a = torch.from_numpy(ds_physics['hyai'].values).to(torch.float64)
    coef_b = torch.from_numpy(ds_physics['hybi'].values).to(torch.float64)
    
    GPH_surf = torch.from_numpy(ds_physics['PHIS'].values).to(torch.float64)
    
    core_compute = physics_hybrid_sigma_level_here(lon2d, lat2d, coef_a, coef_b, midpoint=True)
    
    T_input = x[:, T_ind_start:T_ind_end, -1, ...].to(torch.float64)
    q_input = x[:, q_ind_start:q_ind_end, -1, ...].to(torch.float64)
    U_input = x[:, U_ind_start:U_ind_end, -1, ...].to(torch.float64)
    V_input = x[:, V_ind_start:V_ind_end, -1, ...].to(torch.float64)
    
    T_pred = y_pred[:, T_ind_start:T_ind_end, 0, ...].to(torch.float64)
    q_pred = y_pred[:, q_ind_start:q_ind_end, 0, ...].to(torch.float64)
    U_pred = y_pred[:, U_ind_start:U_ind_end, 0, ...].to(torch.float64)
    V_pred = y_pred[:, V_ind_start:V_ind_end, 0, ...].to(torch.float64)
            
    TOA_solar_pred = y_pred[:, TOA_solar_ind, 0, ...].to(torch.float64)
    TOA_OLR_pred = y_pred[:, TOA_OLR_ind, 0, ...].to(torch.float64)
            
    surf_solar_pred = y_pred[:, surf_solar_ind, 0, ...].to(torch.float64)
    surf_LR_pred = y_pred[:, surf_LR_ind, 0, ...].to(torch.float64)
    surf_SH_pred = y_pred[:, surf_SH_ind, 0, ...].to(torch.float64)
    surf_LH_pred = y_pred[:, surf_LH_ind, 0, ...].to(torch.float64)
    
    CP_t0 = (1 - q_input) * CP_DRY + q_input * CP_VAPOR
    CP_t1 = (1 - q_pred) * CP_DRY + q_pred * CP_VAPOR
    
    # kinetic energy
    ken_t0 = 0.5 * (U_input ** 2 + V_input ** 2)
    ken_t1 = 0.5 * (U_pred ** 2 + V_pred ** 2)

    #print(ken_t1.dtype)
    
    # packing latent heat + potential energy + kinetic energy
    E_qgk_t0 = torch.tensor(LH_WATER).to(q_pred.device) * q_input.to(q_pred.device) + GPH_surf.to(q_pred.device) + ken_t0.to(q_pred.device)
    E_qgk_t1 = torch.tensor(LH_WATER).to(q_pred.device) * q_input.to(q_pred.device) + GPH_surf.to(q_pred.device) + ken_t1.to(q_pred.device)

    #print(E_qgk_t0.dtype)
    
    # TOA energy flux
    R_T = (TOA_solar_pred + TOA_OLR_pred) / N_seconds
    R_T_sum = core_compute.weighted_sum(R_T, axis=(-2, -1))

    
    # print(R_T.dtype)
    # print(R_T_sum.dtype)
    
    # surface net energy flux
    F_S = (surf_solar_pred + surf_LR_pred + surf_SH_pred + surf_LH_pred) / N_seconds
    F_S_sum = core_compute.weighted_sum(F_S, axis=(-2, -1))

    # print(F_S.dtype)
    # print(F_S_sum.dtype)

    E_level_t0 = CP_t0 * T_input + E_qgk_t0
    E_level_t1 = CP_t1 * T_pred + E_qgk_t1

    # print(E_level_t0.dtype)
    # print(E_level_t1.dtype)

    # column integrated total energy
    TE_t0 = core_compute.integral(E_level_t0, sp_input) / GRAVITY
    TE_t1 = core_compute.integral(E_level_t1, sp_pred) / GRAVITY

    # print(TE_t0.dtype)
    # print(TE_t1.dtype)
    
    dTE_dt = (TE_t1 - TE_t0) / N_seconds

    print(dTE_dt.dtype)
    
    dTE_sum = core_compute.weighted_sum(dTE_dt, axis=(1, 2), keepdims=False)
    
    delta_dTE_sum = (R_T_sum - F_S_sum) - dTE_sum

    # print(delta_dTE_sum.dtype)
    # print(R_T_sum.dtype)
    # print(F_S_sum.dtype)
    # print(dTE_sum.dtype)
    
    # print('Residual to conserve energy budget [Watts]: {}'.format(delta_dTE_sum))
    return delta_dTE_sum, dTE_sum, (R_T_sum - F_S_sum)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def save_task(data, meta_data, conf):
    """Wrapper function for saving data in parallel."""
    darray_upper_air, darray_single_level, init_datetime_str, lead_time, forecast_hour = data
    save_netcdf_increment(
        darray_upper_air,
        darray_single_level,
        init_datetime_str,
        lead_time * forecast_hour,
        meta_data,
        conf,
    )

def run_year_rmse(p, config, input_shape, forcing_shape, output_shape, device, model_name=None, init_noise=None):
    """
    Function to compute RMSE for a year-long climate model prediction. 
    
    Parameters:
    - config: str
        Path to the YAML configuration file.
    - input_shape: tuple
        Shape of the input tensor to the model.
    - forcing_shape: tuple
        Shape of the forcing tensor for the model.
    - output_shape: tuple
        Shape of the output tensor from the model.
    - device: torch.device
        Device to run the model on (CPU/GPU).
    - model_name: str, optional
        Name of the model to load, if specified.

    The function handles:
    - Loading configurations, model, and transforms
    - Setting up data pre-processing steps
    - Handling dynamic/static variables and conservation constraints
    - Preparing input/output tensors for prediction and RMSE computation
    - Tracing the model for optimized execution
    - running for a specified amount of time and saving mean 
    """
    # Load configuration from the YAML file
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Parse and preprocess the configuration for prediction
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    conf["predict"]["mode"] = None
    
    # Extract the history length for the data
    history_len = conf["data"]["history_len"]
    
    # Load transformation utilities and scalers
    transform = load_transforms(conf)
    if conf["data"]["scaler_type"] == "std_new":
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        print("Scaler type {} not supported".format(conf["data"]["scaler_type"]))
        raise

    # Load the model (optionally a custom model) and configure distributed mode
    if model_name is not None:
        print('loading custom: ', model_name)
        model = load_model_name(conf, model_name, load_weights=True).to(device)
    else:
        model = load_model(conf, load_weights=True).to(device)
    
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        model = distributed_model_wrapper(conf, model, device)
        if conf["predict"]["mode"] == "fsdp":
            model = load_model_state(conf, model, device)
    
    model.eval()
    post_conf = conf["model"]["post_conf"]
    
    # Extract conservation flags from the configuration
    flag_mass_conserve, flag_water_conserve, flag_energy_conserve = False, False, False
    if post_conf["activate"]:
        if post_conf["global_drymass_fixer"]["activate"] and post_conf["global_drymass_fixer"]["activate_outside_model"]:
            flag_mass_conserve = True
            opt_mass = GlobalDryMassFixer_here(post_conf)
        if post_conf["global_water_fixer"]["activate"] and post_conf["global_water_fixer"]["activate_outside_model"]:
            flag_water_conserve = True
            opt_water = GlobalWaterFixer_here(post_conf)
        if post_conf["global_energy_fixer"]["activate"] and post_conf["global_energy_fixer"]["activate_outside_model"]:
            flag_energy_conserve = True
            opt_energy = GlobalEnergyFixer_here(post_conf)
    
    # Extract variable names from configuration
    df_variables = conf["data"]["dynamic_forcing_variables"]
    sf_variables = conf["data"]["static_variables"]
    varnum_diag = len(conf["data"]["diagnostic_variables"])
    lead_time_periods = conf["data"]["lead_time_periods"]

    # Load initial condition and forcing dataset
    x = torch.load(conf['predict']['init_cond_fast_climate'], map_location=torch.device(device)).to(device)
    DSforc = xr.open_dataset(conf["predict"]["forcing_file"])

    # Set up metrics and transformations
    metrics = LatWeightedMetrics(conf)
    DSforc_norm = state_transformer.transform_dataset(DSforc)

    # Extract indices for RMSE computation and load the truth field
    inds_to_rmse = conf['predict']['inds_rmse_fast_climate']
    truth_field = torch.load(conf['predict']['seasonal_mean_fast_climate'], map_location=torch.device(device))
    print(f'torch loaded truth: {conf["predict"]["seasonal_mean_fast_climate"]}')

    # Load metadata and set up static and dynamic forcing variables
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])
    meta_data = load_metadata(conf)
    num_ts = conf["predict"]["timesteps_fast_climate"]

    if conf["data"]["static_first"]:
        df_sf_variables = conf["data"]["static_variables"] + conf["data"]["dynamic_forcing_variables"]
    else:
        df_sf_variables = conf["data"]["dynamic_forcing_variables"] + conf["data"]["static_variables"]

    y_diag_present = len(conf["data"]["diagnostic_variables"]) > 0

    DS_forcx_static = DSforc[sf_variables].load()

    # Initialize prediction start date
    conf_pred = conf['predict']
    conf_pred_for = conf_pred['forecasts']
    year_ = conf_pred_for.get('start_year', 1970)  # Default to 1970 if missing
    month_ = conf_pred_for.get('start_month', 1)   # Default to January
    day_ = conf_pred_for.get('start_day', 1)       # Default to 1st
    hours_ = conf_pred_for.get('start_hours', [0]) # Default to [0]
    
    # Ensure hours is a list and take the first value
    hour_ = hours_[0] if isinstance(hours_, list) and hours_ else 0
    init_date_obj = datetime(year_, month_, day_, hour_)

    init_date_obj = DatetimeNoLeap(year_, month_, day_, hour_)
    print('init date obj: ', init_date_obj)

    # Align forcing dataset to the initial date
    nearest_time = DSforc_norm.sel(time=init_date_obj, method='nearest')

    index = DSforc_norm.indexes['time'].get_loc(init_date_obj)
    indx_start = index
    print('index start: ', indx_start)
    init_datetime_str = init_date_obj.strftime('%Y-%m-%d %H:%M:%S')
    print('init_datetime_str: ', init_datetime_str)
    print(f"Index of start time: {indx_start}")

    # Select dynamic forcing data for prediction window
    DS_forcx_dynamic = DSforc_norm[df_variables].isel(time=slice(indx_start, indx_start+num_ts+10)).load()
    DS_forcx_dynamic_time = DS_forcx_dynamic['time'].values

    if init_noise is not None:
        print('adding forecast noise')
        # Define the standard deviation for the noise (e.g., 0.01)
        noise_std = 0.05
        
        # Generate random noise tensor with the same shape as `x`
        # Use `torch.randn` for a normal distribution with mean=0 and std=1
        noise = torch.randn_like(x) * noise_std

        # Add the noise to `x`
        x = x + noise.to(device)
        
    # Initialize RMSE computation parameters
    if inds_to_rmse is None:
        inds_to_rmse = np.arange(0, output_shape[1])

    test_tensor_rmse = torch.zeros(output_shape).to(device)
    x_forcing_batch = torch.zeros(forcing_shape).to(device)

    # Prepare forcing dictionary for input tensors
    forcing_dict = {sfv: torch.tensor(DS_forcx_static[sfv].values).to(device) for sfv in sf_variables}
    forcing_dict.update({dfv: torch.tensor(DS_forcx_dynamic[dfv].values).to(device) for dfv in df_variables})

    forecast_hour = 1
    add_it = 0
    num_steps = 0
    
    # Trace the model for optimized execution
    #print('jit trace model start')
    #model = torch.jit.trace(model, x)
    #print('jit trace model done')

    # model = model.to(torch.float64)

    # for name, param in model.named_parameters():
    #     print(f"{name} dtype: {param.dtype}")
    # for name, buffer in model.named_buffers():
    #     print(f"{name} dtype: {buffer.dtype}")

    # for param in model.parameters():
    #     param.data = param.data.to(torch.float64)
    # for buffer in model.buffers():
    #     buffer.data = buffer.data.to(torch.float64)


    resid_wa = []
    resid_en = []
    en_ = []
    resid_ma = []
    date_vec = []

    for k in range(num_ts):
        loop_start = time.time()
        if (k+1)%20 == 0:
            print(f'model step: {k:05}')
        
        start_forcing = time.time()
        if k != 0:
            if conf["data"]["static_first"]:
                for bb, sfv in enumerate(sf_variables):
                    x_forcing_batch[:, bb, :, :, :] = forcing_dict[sfv]
                for bb, dfv in enumerate(df_variables):
                    x_forcing_batch[:, len(sf_variables) + bb, :, :, :] = forcing_dict[dfv][k, :, :]
            else:
                for bb, dfv in enumerate(df_variables):
                    x_forcing_batch[:, bb, :, :, :] = forcing_dict[dfv][k, :, :]
                for bb, sfv in enumerate(sf_variables):
                    x_forcing_batch[:, len(df_variables) + bb, :, :, :] = forcing_dict[sfv]
            x = torch.cat((x, x_forcing_batch), dim=1)

        if k == 0:
            cftime_obj = DS_forcx_dynamic_time[k]
            init_datetime_str = cftime_obj.strftime('%Y-%m-%dT%HZ')
            
        cftime_obj = DS_forcx_dynamic_time[k+1]
        # Convert to string directly using str()
        cftime_str = str(cftime_obj)  # This will look like 'YYYY-MM-DD HH:SS:MM'
        # Parse the string into a standard datetime object
        utc_datetime = datetime.strptime(cftime_str, '%Y-%m-%d %H:%M:%S')
        
        # print('Forcing time:', time.time() - start_forcing)

        start_model = time.time()
        x = x.contiguous()
        print('x dtype', x.dtype)
        x = x.to(torch.float32)
        print('x dtype', x.dtype)
        with torch.no_grad():
            y_pred = model(x)
        # print('Model inference time:', time.time() - start_model)

        x = x.to(torch.float64)
        y_pred = y_pred.to(torch.float64)

        start_postprocess = time.time()
        # test_tensor_rmse.add_(y_pred)
        
        if flag_mass_conserve:
            if forecast_hour == 1:
                x_init = x.clone()
            input_dict = {"y_pred": y_pred, "x": x_init}
            #input_dict = opt_mass(input_dict)
            y_pred = input_dict["y_pred"]

            y_pred_clone = y_pred.clone()
            x_clone = x.clone()
            residual_mass, ressidy_sum, P_sum = mass_residual_verif(x_init, y_pred_clone, opt_mass, post_conf)
            print('residual mass:', residual_mass)
            print('total mass:', ressidy_sum)
            resid_ma.append(residual_mass.cpu().numpy())
            
        if flag_water_conserve:
            input_dict = {"y_pred": y_pred, "x": x}
            #input_dict = opt_water(input_dict)
            y_pred = input_dict["y_pred"]

            y_pred_clone = y_pred.clone()
            x_clone = x.clone()
            residual_water, ressidy_sum, P_sum = water_budget_verif(x_clone, y_pred_clone, opt_water, post_conf)
            print('residual water:', residual_water)
            resid_wa.append(residual_water.cpu().numpy())
            
        if flag_energy_conserve:
            input_dict = {"y_pred": y_pred, "x": x}
            #input_dict = opt_energy(input_dict)
            y_pred = input_dict["y_pred"]
            y_pred_clone = y_pred.clone()
            x_clone = x.clone()
            residual_energy, ressidy_sum, P_sum = energy_residual_verif(x_clone, y_pred_clone, opt_energy, post_conf)
            print('residual energy:', residual_energy)
            resid_en.append(residual_energy.cpu().numpy())
            en_.append(ressidy_sum.cpu().numpy())


        # print('Postprocess time:', time.time() - start_postprocess)
        test_tensor_rmse.add_(y_pred)
        darray_upper_air, darray_single_level = make_xarray(
            y_pred.cpu(),
            utc_datetime,
            latlons.latitude.values,
            latlons.longitude.values,
            conf,
        )

        result = p.apply_async(
            save_netcdf_increment,
            (
                darray_upper_air,
                darray_single_level,
                init_datetime_str,
                lead_time_periods * forecast_hour,
                meta_data,
                conf,
            ),
        )
        print('I did it once!')
        date_vec.append(utc_datetime)
        
        # cuda_empty_start = time.time()
        # torch.cuda.empty_cache()
        # gc.collect()
        add_it += 1
        num_steps += 1
        forecast_hour += 1
        # print('CUDA empty time:', time.time() - cuda_empty_start)
        start_switch = time.time()
        # ============================================================ #
        # use previous step y_pred as the next step input
        if history_len == 1:
            # cut diagnostic vars from y_pred, they are not inputs
            if y_diag_present:
                x = y_pred[:, :-varnum_diag, ...].detach()
            else:
                x = y_pred.detach()

        # multi-step in
        else:
            if static_dim_size == 0:
                x_detach = x[:, :, 1:, ...].detach()
            else:
                x_detach = x[:, :-static_dim_size, 1:, ...].detach()

            # cut diagnostic vars from y_pred, they are not inputs
            if y_diag_present:
                x = torch.cat(
                    [x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2
                )
            else:
                x = torch.cat([x_detach, y_pred.detach()], dim=2)
                
        # print('Total loop time:', time.time() - loop_start)
        # ============================================================ #
        # print('Switch time:', time.time() - start_switch)
        # print('Total loop time:', time.time() - loop_start)
        # print('add_it: ', add_it)
        # if add_it == 365:
        #     break

    


    # Convert residual lists to numpy arrays
    resid_wa_array = np.array([val.item() for val in resid_wa])
    resid_en_array = np.array([val.item() for val in resid_en])
    en_array_ = np.array([val.item() for val in en_])
    resid_ma_array = np.array([val.item() for val in resid_ma])

    # Create the time vector using utc_datetime
    time_vector = np.array(date_vec)
    print(time_vector)
    
    # Create an xarray Dataset
    residual_dataset = xr.Dataset(
        {
            "resid_water": (["time"], resid_wa_array),
            "resid_energy": (["time"], resid_en_array),
            "resid_mass": (["time"], resid_ma_array),
            "energy_tend": (["time"], en_array_),
        },
        coords={
            "time": time_vector,
        },
        attrs={
            "description": "Residuals for water, energy, and mass conservation",
            "init_date": str(init_date_obj),
        },
    )
    
    # Generate a unique filename based on `init_date_obj`
    unique_filename = f'{conf["save_loc"]}/residuals_energyincl_{init_datetime_str}_noconserve_nopostblock_{str(flag_mass_conserve)}.nc'
    
    # Save the Dataset to a NetCDF file
    residual_dataset.to_netcdf(unique_filename)
    
    print(f"Residual dataset saved to {unique_filename}")


    if model_name is not None:
        fsout = f'{os.path.basename(config)}_{model_name}'
    else:
        fsout = os.path.basename(config)
    
    METS = metrics(test_tensor_rmse.cpu()/add_it,truth_field.cpu())

    #save out: test_tensor_rmse/num_ts, MET, plots  
    with open(f'{conf["save_loc"]}/{fsout}_quick_climate_METS.pkl', 'wb') as f:
        pickle.dump(METS, f)

    torch.save(test_tensor_rmse.cpu()/add_it,f'{conf["save_loc"]}/{fsout}_quick_climate_avg_tensor.pt')

    fig, axes = plt.subplots(2, 3, figsize=(25, 15))

    p1 = axes[0,0].pcolor(test_tensor_rmse.squeeze()[-16, :, :].cpu() / add_it, vmin=-2, vmax=2, cmap='RdBu_r')
    fig.colorbar(p1, ax=axes[0,0])
    axes[0,0].set_title('Test Tensor RMSE')
        
    p2 = axes[0,1].pcolor(truth_field.squeeze()[-16, :, :].cpu(), vmin=-2, vmax=2, cmap='RdBu_r')
    fig.colorbar(p2, ax=axes[0,1])
    axes[0,1].set_title('Truth Field')
        
    p3 = axes[0,2].pcolor((test_tensor_rmse.squeeze()[-16, :, :].cpu() / add_it) - truth_field.squeeze()[-16, :, :].cpu(), vmin=-0.1, vmax=0.1, cmap='RdBu_r')
    fig.colorbar(p3, ax=axes[0,2])
    axes[0,2].set_title('Difference (RMSE - Truth)')
        
    p1 = axes[1,0].pcolor(test_tensor_rmse.squeeze()[-15, :, :].cpu() / add_it, vmin=-2, vmax=2, cmap='RdBu_r')
    fig.colorbar(p1, ax=axes[1,0])
    axes[1,0].set_title('Test Tensor RMSE')
        
    p2 = axes[1,1].pcolor(truth_field.squeeze()[-15, :, :].cpu(), vmin=-2, vmax=2, cmap='RdBu_r')
    fig.colorbar(p2, ax=axes[1,1])
    axes[1,1].set_title('Truth Field')
        
    p3 = axes[1,2].pcolor((test_tensor_rmse.squeeze()[-15, :, :].cpu() / add_it) - truth_field.squeeze()[-15, :, :].cpu(), vmin=-0.5, vmax=0.5, cmap='RdBu_r')
    fig.colorbar(p3, ax=axes[1,2])
    axes[1,2].set_title('Difference (RMSE - Truth)')
    
    plt.tight_layout()
    plt.savefig(f'{conf["save_loc"]}/{fsout}_quick_climate_plot_slow.png', bbox_inches='tight')
    plt.show()

    
    return test_tensor_rmse, truth_field, inds_to_rmse, metrics, conf, METS



def main():
    parser = argparse.ArgumentParser(description="Run year RMSE for WxFormer model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration YAML file.')
    parser.add_argument('--input_shape', type=int, nargs='+', required=True, help='Input shape as a list of integers.')
    parser.add_argument('--forcing_shape', type=int, nargs='+', required=True, help='Forcing shape as a list of integers.')
    parser.add_argument('--output_shape', type=int, nargs='+', required=True, help='Output shape as a list of integers.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu).')
    parser.add_argument('--model_name', type=str, default=None, help='Optional model checkpoint name.')
    parser.add_argument('--init_noise', type=int, default=None, help='init model noise')

    args = parser.parse_args()

    start_time = time.time()

    # # Call the run_year_rmse function with parsed arguments
    # test_tensor_rmse, truth_field, inds_to_rmse, metrics, conf, METS = run_year_rmse(
    #     config=args.config,
    #     input_shape=args.input_shape,
    #     forcing_shape=args.forcing_shape,
    #     output_shape=args.output_shape,
    #     device=args.device,
    #     model_name=args.model_name
    # )
    
    num_cpus = 8
    with mp.Pool(num_cpus) as p:
        run_year_rmse(p, config=args.config, input_shape=args.input_shape,
        forcing_shape=args.forcing_shape,
        output_shape=args.output_shape,
        device=args.device,
        model_name=args.model_name,
        init_noise=args.init_noise)

    end_time = time.time()
    elapsed_time = end_time - start_time
    # How to run:
    # python Quick_Climate_Year.py --config /path/to/config.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt
    print(f"Run completed. Results saved to configured location. Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Run completed. Results saved to configured location. Elapsed time: {elapsed_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
