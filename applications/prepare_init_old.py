import xarray as xr
import cftime
import pandas as pd
import argparse
from os.path import join
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", required=True, help="Date in YYYY-mm-ddTHHMM format")
args = parser.parse_args()

# pick initialization:
model = 'CAMulator'
# init = '2025-08-14'
init = pd.Timestamp(args.date).strftime("%Y-%m-%d")
init_cftime = cftime.DatetimeNoLeap(int(init[:4]), int(init[5:7]), int(init[8:])) # cesm data calendar
initdate = pd.to_datetime(init).strftime('%Y%m%d')
YEAR = init[:4]


# last forecast day:
# https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html
day32_cftime = init_cftime + pd.Timedelta(days=4+(7*3)+6)
day32 = pd.to_datetime(init) + pd.Timedelta(days=4+(7*3)+6)


# will use this to convert cftime calendar in dynamic forcing to datetime
# --> required for credit realtime rollout
timestamp_forecastlead = pd.date_range(
            start=init,
            end=day32,
            freq="6h",)


# Load CESM Dynamic Forcing File
cesmpath = '/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/'
dynamicfile = 'b.e21.CREDIT_climate_branch_allvars_'+YEAR+'.zarr'
ds_dynamic_forcings = xr.open_dataset(cesmpath+dynamicfile, engine='zarr')

# Load a GEFS initialization file
ddir_GEFS = '/glade/derecho/scratch/cbecker/GEFS_CAM/gefs.'+initdate+'/00/'
GEFS_ICS_fi = 'gefs_cam_grid_c00.nc'
ds_GEFS_ICS = xr.open_dataset(ddir_GEFS+GEFS_ICS_fi)

# replace CESM ICEFRAC with ICEFRAC from GEFS
ds_dynamic_forcings["ICEFRAC"].loc[dict(time=slice(init_cftime, day32_cftime))] = ds_GEFS_ICS["ICEFRAC"][0].values
if model == 'CAMulator':
    ds_dynamic_forcings["SST"] = xr.where(ds_dynamic_forcings["SST"] == 283.0, x=283.0, y=ds_GEFS_ICS['TS'][0].values)

# subset dynamic_forcings for forecast time (init through day 32; 6 hourly timesteps)
ds_dynamic_forcings_forecast = ds_dynamic_forcings.sel(time=slice(init_cftime, day32_cftime))
ds_dynamic_forcings_forecast['time'] = timestamp_forecastlead

# Save
savefi = 'b.e21.CREDIT_climate_branch_allvars_GEFSicefracandsst_'+init+'.zarr'

# test_path = "/glade/derecho/scratch/cbecker/test_kirsten_file.zarr"
dir_path = join(cesmpath, 'AIWQ_inits', init)
os.makedirs(dir_path, exist_ok=True)
ds_dynamic_forcings_forecast.to_zarr(join(dir_path, savefi), mode='w', consolidated=True)
# ds_dynamic_forcings_forecast.to_zarr(test_path, mode='w', consolidated=True)
