'''
This script computes zonal energy spectrum of kinetic energy and potential temperature
energy from 6 hourly Wxformer outputs. It runs with config file `verif_config_6h.yml` 
with hard coded forecast lead time of [24, 120, 240] hours. 

The script calculates energy spectrum on a given range of initializations:
```
python STEP03_ZES_wxformer_6h.py 0 365
```
where 0 and 365 are the first and the last initialization.

The script produces a netCDF4 file on `path_verif`.

Note: the script assumes coordinate names are `latitude` and `longitude`. Potential
temperature is computed using an approximation, but it does not impact the energy
distributions.
'''

import os
import sys
import yaml
import argparse
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu
import score_utils as su

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
verif_lead_range = conf[model_name]['verif_lead_range']

leads_exist = list(np.arange(lead_range[0], lead_range[-1]+lead_range[0], lead_range[0]))
leads_verif = [24, 120,]
#list(np.arange(verif_lead_range[0], verif_lead_range[-1]+verif_lead_range[0], verif_lead_range[0]))
ind_lead = vu.lead_to_index(leads_exist, leads_verif)

print('Verifying lead times: {}'.format(leads_verif))
print('Verifying lead indices: {}'.format(ind_lead))

variables_levels = conf['ERA5_ours']['verif_variables']
varnames = list(variables_levels.keys())
# ====================== #

path_verif = conf[model_name]['save_loc_verif']+'combined_zes_{:04d}_{:04d}_{}'.format(
    verif_ind_start, verif_ind_end, model_name)

# ---------------------------------------------------------------------------------------- #
# forecast
filename_OURS = sorted(glob(conf[model_name]['save_loc_gather']+'*.nc'))

# pick years
year_range = conf[model_name]['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_OURS = [fn for fn in filename_OURS if any(year in fn for year in years_pick)]

L_max = len(filename_OURS)
assert verif_ind_end <= L_max, 'verified indices (days) exceeds the max index available'

filename_OURS = filename_OURS[verif_ind_start:verif_ind_end]

# ---------------------------------------------------------------------------------------- #
# loop over lead time, init time, variables to compute zes
for i, ind_pick in enumerate(ind_lead):
    # allocate result for the current lead time
    verif_results = []
    
    for fn_ours in filename_OURS:
        ds_ours = xr.open_dataset(fn_ours)
        ds_ours = vu.ds_subset_everything(ds_ours, variables_levels)
        ds_ours = ds_ours.isel(time=ind_pick)
        ds_ours = ds_ours.compute()
        
        # -------------------------------------------------------------- #
        # potential temperature
        ds_ours['theta'] = ds_ours['T500'] * (1000/500)**(287.0/1004)

        # -------------------------------------------------------------- #
        zes_temp = []
        for var in ['U500', 'V500', 'theta']:
            zes = su.zonal_energy_spectrum_sph(ds_ours, var)
            zes_temp.append(zes)
            
        verif_results.append(xr.merge(zes_temp))
        
    ds_verif = xr.concat(verif_results, dim='time')
    save_name = path_verif+'_lead{}.nc'.format(leads_verif[i])
    ds_verif.to_netcdf(save_name)
