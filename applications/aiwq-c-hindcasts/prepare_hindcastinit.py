import numpy as np
import xarray as xr
import cftime
import datetime
import pandas as pd
import os
import argparse
import yaml

print('Loaded libraries')

def days_since_noleap(days_since, start_date=datetime.datetime(1979, 1, 1)):
    startdate_noleap = cftime.datetime(start_date.year,
                                       start_date.month,
                                       start_date.day,
                                       calendar='noleap')
    return startdate_noleap + datetime.timedelta(days=days_since)


print('Made it')
LEAD = 46

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", required=True, help="Date in YYYY-mm-ddTHHMM format")
args = parser.parse_args()

print('Loaded Args')

init = pd.Timestamp(args.date).strftime("%Y-%m-%d")
init_cftime = cftime.DatetimeNoLeap(int(init[:4]), int(init[5:7]), int(init[8:])) # cesm data calendar
init_datetime = pd.to_datetime(init)

# last forecast day:
daylead_cftime = init_cftime + pd.Timedelta(days=LEAD)
daylead = init_datetime + pd.Timedelta(days=LEAD)

# will use this to convert cftime calendar in dynamic forcing to datetime
# --> required for credit realtime rollout
timestamp_forecastlead = pd.date_range(
            start=init,
            end=daylead,
            freq="6h",)


# Update config file - realtime and forecast sections:
# CONFIG_FI = '/glade/work/kjmayer/research/AIWQ_subCESMulator/miles-credit/config/S2Shindcast.yml'
# with open(CONFIG_FI, 'r') as file:
#     configdata = yaml.safe_load(file)

# configdata['predict']['forecasts']['start_year'] = int(init[:4])
# configdata['predict']['forecasts']['start_month'] = int(init[5:7])
# configdata['predict']['forecasts']['start_day'] = int(init[-2:])
# configdata['predict']['forecasts']['days'] = LEAD

# in hindcast_wrapper.py
# configdata['predict']['realtime']['forecast_start_time'] = "" #init
# configdata['predict']['realtime']['forecast_end_time'] = "" #daylead.strftime("%Y-%m-%d")

# with open(CONFIG_FI, 'w') as file:
#     yaml.safe_dump(configdata, file, sort_keys=False)
#     print('Updated config file for forecast')


# Check whether files exist:
USER = os.environ['USER']
output_base = f'/glade/derecho/scratch/{USER}/DATA/CESMS2S_inits/'
input_base = '/glade/derecho/scratch/kjmayer/DATA/CESMS2S_inits/'  # read-only forcing data
init_savefi = output_base+init+'/CESM_'+init+'.nc'

if os.path.isfile(init_savefi):
    print('Files for '+init+' already exist... continuing with presaved files')

else:
    print('Generating files for '+init)
    # Get CESM hindcast initialization data
    ICpath = '/glade/campaign/cesm/development/cross-wg/S2S/sglanvil/forKirsten/subCESMulator/final/'
    finames = 'CESM_'+init+'.nc' # load only 2000+ because I don't have SOLIN or CO2 for 1999
    initdata = xr.open_dataset(ICpath+finames,
                               decode_times=False)
    
    # Drop extra vars in CESM init data
    initdata = initdata.drop_vars(['SST','ICEFRAC','SST_lowResLnd'])
    
    initdata['time'] = np.array([days_since_noleap(days_since)
                                 for days_since in initdata.time.values])
    initdata = initdata.rename({'Q': 'Qtot','lev':'level','lat':'latitude','lon':'longitude'})
    initdata = initdata.chunk({'time': 1})
    initdata['time'] = [init_datetime]

    # Include all years spanned by the forecast (init date + 46-day lead)
    init_year = init_datetime.year
    end_year = daylead.year
    years = range(init_year, end_year + 1)

    # Load CESM dynamical forcing data for SOLIN and CO2
    # & CESM Hindcast SST 6hrly interpolated init
    cesm_ddir = '/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/'

    cesm_years = []
    initcesm_dyn_years = []

    for YEAR in years:
        cesm_finame = 'b.e21.CREDIT_climate_branch_allvars_'+str(YEAR)+'_SWupdate.zarr'
        cesm_years.append(xr.open_zarr(cesm_ddir+cesm_finame)[['co2vmr_3d','SOLIN']])

        init_dynfiname = 'CESM_DYNFORCEinit_6hr_'+str(YEAR)+'.nc'
        initcesm_dyn_years.append(xr.open_dataset(input_base+init_dynfiname)[['SST','ICEFRAC']])

    ds_dynamical_forcing = xr.concat(cesm_years, dim='time').sel(time=slice(init_cftime,daylead_cftime))
    ds_cesminit_forcing = xr.concat(initcesm_dyn_years, dim='time').sel(time=slice(init_cftime,daylead_cftime))

    # Reindex both to exact 6-hourly noleap timestamps covering the forecast window.
    # This fixes cross-year-boundary mismatches where the yearly zarr files and
    # DYNFORCEinit files produce different numbers of timesteps after slicing.
    target_times_noleap = xr.cftime_range(
        start=init_cftime, periods=len(timestamp_forecastlead),
        freq="6h", calendar="noleap"
    )
    ds_dynamical_forcing = ds_dynamical_forcing.reindex(time=target_times_noleap, method='nearest')
    ds_cesminit_forcing = ds_cesminit_forcing.reindex(time=target_times_noleap, method='nearest')

    # --> add in ICEFRAC and SST from CESM init data + persist for length of forecast
    ds_dynamical_forcing["ICEFRAC"] = ds_cesminit_forcing['ICEFRAC']
    ds_dynamical_forcing["SST"] = ds_cesminit_forcing['SST']
    ds_dynamical_forcing = ds_dynamical_forcing.chunk({'time': 1})
    ds_dynamical_forcing['time'] = timestamp_forecastlead
    
    
    # Save
    save_dir = output_base+init+'/'
    os.makedirs(save_dir, exist_ok=True)
    print('saving: '+init)
    
    init_fidata = initdata.isel(time=[0])
    init_fidata.to_netcdf(save_dir+'CESM_'+init+'.nc')
    
    ds_dynamical_forcing.to_netcdf(save_dir+'CESM_dynamicforcing_'+init+'.nc')
