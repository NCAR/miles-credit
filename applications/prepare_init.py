import xarray as xr
import cftime
import pandas as pd
import os
import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", required=True, help="Date in YYYY-mm-ddTHHMM format")
args = parser.parse_args()

# pick initialization
model = 'CAMulator'
# model = 'subCESMulator'
init = pd.Timestamp(args.date).strftime("%Y-%m-%d")
# init_times = pd.date_range("2024-03-01", "2024-12-31", freq="W-THU")
# print(init_times)
# for init in init_times.strftime("%Y-%m-%d"):
#     print(init)
init_cftime = cftime.DatetimeNoLeap(int(init[:4]), int(init[5:7]), int(init[8:])) # cesm data calendar
initdate = pd.to_datetime(init).strftime('%Y%m%d')
YEAR = init[:4]

# last forecast day:
# https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html
# need to add in one more day for the precip accumulation which goes to 00Z on the 33rd day
# we account for this extra 6 hours in prepare_forecast.py
day32_cftime = init_cftime + pd.Timedelta(days=4+(7*3)+6+1) #pd.Timedelta(days=4+(7*3)+6, hours=18)
day32 = pd.to_datetime(init) + pd.Timedelta(days=4+(7*3)+6+1) #pd.Timedelta(days=4+(7*3)+6, hours=18)


# will use this to convert cftime calendar in dynamic forcing to datetime
# --> required for credit realtime rollout
timestamp_forecastlead = pd.date_range(
            start=init,
            end=day32,
            freq="6h",)


# Load CESM Dynamic Forcing File
cesmpath = '/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/'
dynamicfile = 'b.e21.CREDIT_climate_branch_allvars_'+YEAR+'.zarr'

# Load a GEFS initialization file
ddir_GEFS = '/glade/derecho/scratch/cbecker/GEFS_CAM/gefs.'+initdate+'/00/'
GEFS_ICS_fi = 'gefs_cam_grid_c00.nc'
ds_GEFS_ICS = xr.open_dataset(ddir_GEFS+GEFS_ICS_fi)

## Handle initializing in one year where forecast extends into following year
forecast_end_year = (pd.Timestamp(args.date) + pd.Timedelta(days=35)).strftime('%Y')
if YEAR != forecast_end_year:
    dynamicfile_new_year = 'b.e21.CREDIT_climate_branch_allvars_' + forecast_end_year + '.zarr'
    ds_dynamic_forcings = xr.open_mfdataset([cesmpath + dynamicfile,  cesmpath + dynamicfile_new_year], engine='zarr',
                                            concat_dim='time', combine='nested')
    # print('Rechunking')
    # ds_dynamic_forcings = ds_dynamic_forcings.chunk({"time": 12, "latitude": 192, "longitude": 288,
    #                                                  "level": 32, "ilev": 33})
    # print('complete')
else:
    ds_dynamic_forcings = xr.open_dataset(cesmpath + dynamicfile, engine='zarr')

# replace CESM ICEFRAC with ICEFRAC from GEFS
ds_dynamic_forcings["ICEFRAC"].loc[dict(time=slice(init_cftime, day32_cftime))] = ds_GEFS_ICS["ICEFRAC"][0].values
if model == 'CAMulator':
    ds_dynamic_forcings["SST"] = xr.where(ds_dynamic_forcings["SST"] == 283.0, x=283.0, y=ds_GEFS_ICS['TS'][0].values)
# elif model == 'subCESMulator':
#     ds_GEFS_ICS['SST'] = xr.where(ds_dynamic_forcings["SST"][0].expand_dims(time=[ds_GEFS_ICS.time.values[0]]) == 283.0,
#                                   x=283.0, y=ds_GEFS_ICS['TS'].values)
    # ds_dynamic_forcings['SST'] = xr.where(ds_dynamic_forcings["SST"][0].expand_dims(time=[ds_GEFS_ICS.time.values[0]]) == 283.0,
    #                               x=283.0, y=ds_GEFS_ICS['TS'].values)
# subset dynamic_forcings for forecast time (init through day 32; 6 hourly timesteps)
ds_dynamic_forcings_forecast = ds_dynamic_forcings.sel(time=slice(init_cftime, day32_cftime))
ds_dynamic_forcings_forecast['time'] = timestamp_forecastlead
# ds_dynamic_forcings = ds_dynamic_forcings.chunk({"time": 12, "latitude": 192, "longitude": 288,
#                                                      "level": 32, "ilev": 33})
for var in ds_dynamic_forcings_forecast.data_vars:
    ds_dynamic_forcings_forecast[var].encoding = {}
ds_dynamic_forcings_forecast = ds_dynamic_forcings_forecast.chunk({"time": 12, "latitude": 192, "longitude": 288})
print('chunked')
# Save
savefi = 'b.e21.CREDIT_climate_branch_allvars_GEFSicefracandsst_'+init+'.zarr'
ds_dynamic_forcings_forecast.to_zarr(cesmpath+'AIWQ_inits/'+savefi, mode='w', consolidated=True)

# Fix levels in GEFS
# ddir_GEFS_save = '/glade/derecho/scratch/cbecker/CREDIT_runs_historical/AIWQ_init/'+initdate+'/'
ddir_GEFS_save = '/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/AIWQ_inits/'+initdate+'/'

if not os.path.exists(ddir_GEFS_save):
    os.makedirs(ddir_GEFS_save)

all_mems = os.listdir(ddir_GEFS)

# stdfi = '/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/std_6h_coupled_2000-2014_allvars_nonan.nc'
# std = xr.open_dataset(stdfi)
# mask = std['SOILWATER_10CM'] > 0.1

SWfi = 'b.e21.CREDIT_climate_branch_SWupdate_2000-2049climo.nc'
SWclimo = xr.open_dataset(cesmpath+SWfi)

datetime_index = pd.to_datetime([init])
time_dataarray = xr.DataArray(datetime_index, coords={"time": datetime_index}, dims=["time"])
doy = time_dataarray.dt.dayofyear

all_mems = os.listdir(ddir_GEFS)
for GEFS_fi in all_mems:
    print(GEFS_fi)
    ds_GEFS_ICS = xr.open_dataset(ddir_GEFS+GEFS_fi)
    if model == 'subCESMulator':
        # ds_GEFS_ICS['SOILWATER_10CM'] = ds_GEFS_ICS['SOILWATER_10CM']*mask
        ds_GEFS_ICS['SOILWATER_10CM'] = SWclimo['SOILWATER_10CM'][doy.values[0]-1]
        ds_GEFS_ICS['SST'] = xr.where(
            ds_dynamic_forcings["SST"][0].expand_dims(time=[ds_GEFS_ICS.time.values[0]]) == 283.0,
            x=283.0, y=ds_GEFS_ICS['TS'].values)
    ds_GEFS_ICS['level'] = ds_dynamic_forcings.level.values
    ds_GEFS_ICS.to_netcdf(ddir_GEFS_save + GEFS_fi)
