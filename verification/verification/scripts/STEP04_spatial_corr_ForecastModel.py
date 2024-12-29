'''
This script computes spatial correlation of model level upper air variables 
[V, U, T, Q] on 6 hourly Wxformer. It runs with config file `verif_config_1h.yml`. 
The script produces a netCDF4 file on `path_verif`.
'''

import os
import sys
import yaml
import argparse
from glob import glob
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

config_name = os.path.realpath('../verif_config_1h.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('verif_ind_start', help='verif_ind_start')
parser.add_argument('verif_ind_end', help='verif_ind_end')
args = vars(parser.parse_args())

verif_ind_start = int(args['verif_ind_start'])
verif_ind_end = int(args['verif_ind_end'])
# ====================== #
model_name = 'wxformer'
lead_range = conf[model_name]['lead_range']
verif_lead_range = [40,]

leads_exist = list(np.arange(lead_range[0], lead_range[-1]+lead_range[0], lead_range[0]))
leads_verif = list(np.arange(verif_lead_range[0], verif_lead_range[-1]+verif_lead_range[0], verif_lead_range[0]))
ind_lead = vu.lead_to_index(leads_exist, leads_verif)

print('Verifying lead times: {}'.format(leads_verif))
print('Verifying lead indices: {}'.format(ind_lead))
# ====================== #

path_verif = conf[model_name]['save_loc_verif']+'spatial_corr_{:04d}_{:04d}_{:03d}h_{}.nc'.format(
    verif_ind_start, verif_ind_end, verif_lead_range[0], model_name)

# ---------------------------------------------------------------------------------------- #
filename_OURS = sorted(glob(conf[model_name]['save_loc_gather']+'*.nc'))

# pick years
year_range = conf[model_name]['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_OURS = [fn for fn in filename_OURS if any(year in fn for year in years_pick)]

L_max = len(filename_OURS)
assert verif_ind_end <= L_max, 'verified indices (days) exceeds the max index available'

filename_OURS = filename_OURS[verif_ind_start:verif_ind_end]
variables_levels = {'V': None, 'U': None, 'T': None, 'Q': None}

# ---------------------------------------------------------------------------------------- #
# allocate spatial corrlation on every 6 hour
corr_ds_list = []

# variables (time, level, latitude, longtude) to compute corr
var_4D = list(variables_levels.keys())

# loop over 6 hourly indices
for idx, fn_ours in enumerate(filename_OURS):
    
    ds_ours = xr.open_dataset(fn_ours)
    ds_ours = vu.ds_subset_everything(ds_ours, variables_levels)
    ds_ours = ds_ours.isel(time=ind_lead[0])
    ds_ours = ds_ours.load()

    if idx == 0:
        # get level information
        levels = ds_ours['level'].values
    
    # create a list of var with level info
    var_info = [(f"{varname}_{level_num}", varname, level_num) for varname in var_4D for level_num in levels]
    
    # get var names to label output xarray
    varname_full = [var[0] for var in var_info]
    N_vars = len(varname_full)
    
    # allocate xr.DataArray for corr output
    corr_array = xr.DataArray(
        np.full((N_vars, N_vars), np.nan),
        coords={'var1': varname_full, 'var2': varname_full},
        dims=['var1', 'var2']
    )

    # loop over variable pairs, compute corr
    for (var1_info, var2_info) in combinations(var_info, 2):
        
        # var1
        varname1, var1, lev1 = var1_info
        data_target1 = ds_ours[var1].sel(level=lev1)

        # var2
        varname2, var2, lev2 = var2_info
        data_target2 = ds_ours[var2].sel(level=lev2)
        
        # convert to numpy and ravel
        data_target1 = data_target1.values.ravel()
        data_target2 = data_target2.values.ravel()

        # corr
        corr, _ = pearsonr(data_target1, data_target2)
        
        # assign corr coef to corr_array
        corr_array.loc[dict(var1=varname1, var2=varname2)] = corr
        corr_array.loc[dict(var1=varname2, var2=varname1)] = corr

    # set diagonal elements to 1
    for varname in varname_full:
        corr_array.loc[dict(var1=varname, var2=varname)] = 1.0

    # convert xr.DataArray to xr.Dataset
    corr_ds = xr.Dataset({'correlation': corr_array})

    # Append the dataset to the list
    corr_ds_list.append(corr_ds)

# concat all 6 hourly result
corr_ds_combined = xr.concat(corr_ds_list, dim='day')
corr_ds_combined.to_netcdf(path_verif)





