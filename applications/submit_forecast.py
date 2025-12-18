import xarray as xr
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from AI_WQ_package import forecast_submission as fs
from AI_WQ_package.plotting_forecast import plot_forecast
from AI_WQ_package.forecast_submission import AI_WQ_check_submission


# ------- GET INIT & LEADS -------
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", required=True, help="Date in YYYY-mm-ddTHHMM format")
args = parser.parse_args()
# ----------------------- USER SPECIFIES --------------------------
init = pd.Timestamp(args.date).strftime("%Y-%m-%d")
#init = '2025-09-04' # YYYY-MM-DD for CREDIT rollout files
init_dt = pd.to_datetime(init) # dt object for timedelta calc
initdate = init_dt.strftime('%Y%m%d') # YYYYMMDD - must be a Thursday

# get lead times
day19 = init_dt + pd.Timedelta(days=4+(7*2))
day25 = init_dt + pd.Timedelta(days=4+(7*2)+6, hours=18)

day26 = init_dt + pd.Timedelta(days=4+(7*3))
day32 = init_dt + pd.Timedelta(days=4+(7*3)+6, hours=18)

# convert times to YYYYMMDD
day19_date = day19.strftime('%Y%m%d')
day25_date = day25.strftime('%Y%m%d')
day26_date = day26.strftime('%Y%m%d')
day32_date = day32.strftime('%Y%m%d')

teamname = 'NSFNCAR'
modelname = 'subCESMulator'
password = ''


# ------- CREATE EMPTY DATASET FOR FORECASTS -------
# ----------- 2m temperature -------------
tas_p1_fc = fs.AI_WQ_create_empty_dataarray(
    'tas', initdate, '1', teamname, modelname, password
)
tas_p2_fc = fs.AI_WQ_create_empty_dataarray(
    'tas', initdate, '2', teamname, modelname, password
)

# ----------- MSLP -------------
mslp_p1_fc = fs.AI_WQ_create_empty_dataarray(
    'mslp', initdate, '1', teamname, modelname, password
)
mslp_p2_fc = fs.AI_WQ_create_empty_dataarray(
    'mslp', initdate, '2', teamname, modelname, password
)

# ----------- Accumulated Precip -------------
pr_p1_fc = fs.AI_WQ_create_empty_dataarray(
    'pr', initdate, '1', teamname, modelname, password
)
pr_p2_fc = fs.AI_WQ_create_empty_dataarray(
    'pr', initdate, '2', teamname, modelname, password
)


# ------- LOAD IN PREDICTIONS -------
ddir = '/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/submitted_forecasts/'
finame_week3 = 'init'+initdate+'_forecast'+day19_date+'-'+day25_date+'.nc'
finame_week4 = 'init'+initdate+'_forecast'+day26_date+'-'+day32_date+'.nc'

tas1_subCESMulator = xr.open_dataarray(ddir+'tas_'+finame_week3).isel(latitude=slice(None, None, -1))
tas2_subCESMulator = xr.open_dataarray(ddir+'tas_'+finame_week4).isel(latitude=slice(None, None, -1))

pr1_subCESMulator = xr.open_dataarray(ddir+'pr_'+finame_week3).isel(latitude=slice(None, None, -1))
pr2_subCESMulator = xr.open_dataarray(ddir+'pr_'+finame_week4).isel(latitude=slice(None, None, -1))

mslp1_subCESMulator = xr.open_dataarray(ddir+'mslp_'+finame_week3).isel(latitude=slice(None, None, -1))
mslp2_subCESMulator = xr.open_dataarray(ddir+'mslp_'+finame_week4).isel(latitude=slice(None, None, -1))


# ------- POPULATE EMPTY ARRAYS WITH FORECAST -------
# ----------- 2m temperature -------------
tas_p1_fc.values = tas1_subCESMulator
tas_p2_fc.values = tas2_subCESMulator
# ----------- MSLP -------------
mslp_p1_fc.values = mslp1_subCESMulator
mslp_p2_fc.values = mslp2_subCESMulator
# ----------- Accumulated Precip -------------
pr_p1_fc.values = pr1_subCESMulator
pr_p2_fc.values = pr2_subCESMulator

tas_p1_fc.to_netcdf("/glade/derecho/scratch/cbecker/tas_20251127_p1_NSFNCAR_subCESMulator.nc")
tas_p2_fc.to_netcdf("/glade/derecho/scratch/cbecker/tas_20251127_p2_NSFNCAR_subCESMulator.nc")
mslp_p1_fc.to_netcdf("/glade/derecho/scratch/cbecker/mslp_20251127_p1_NSFNCAR_subCESMulator.nc")
mslp_p2_fc.to_netcdf("/glade/derecho/scratch/cbecker/mslp_20251127_p2_NSFNCAR_subCESMulator.nc")
pr_p1_fc.to_netcdf("/glade/derecho/scratch/cbecker/pr_20251127_p1_NSFNCAR_subCESMulator.nc")
pr_p2_fc.to_netcdf("/glade/derecho/scratch/cbecker/pr_20251127_p2_NSFNCAR_subCESMulator.nc")
# ------- SUBMIT FORECAST -------
# ----------- 2m temperature -------------
tas_p1_fc_submit = fs.AI_WQ_forecast_submission(
    tas_p1_fc, 'tas', initdate, '1', teamname, modelname, password
)
tas_p2_fc_submit = fs.AI_WQ_forecast_submission(
    tas_p2_fc, 'tas', initdate, '2', teamname, modelname, password
)
# ----------- MSLP -------------
mslp_p1_fc_submit = fs.AI_WQ_forecast_submission(
    mslp_p1_fc, 'mslp', initdate, '1', teamname, modelname, password
)
mslp_p2_fc_submit = fs.AI_WQ_forecast_submission(
    mslp_p2_fc, 'mslp', initdate, '2', teamname, modelname, password
)
# ----------- Accumulated Precip -------------
pr_p1_fc_submit = fs.AI_WQ_forecast_submission(
    pr_p1_fc, 'pr', initdate, '1', teamname, modelname, password
)
pr_p2_fc_submit = fs.AI_WQ_forecast_submission(
    pr_p2_fc, 'pr', initdate, '2', teamname, modelname, password
)


# ------- CONFIRM SUBMISSION (print statements) -------
AI_WQ_check_submission('tas',initdate,'1',teamname,modelname,password)
AI_WQ_check_submission('tas',initdate,'2',teamname,modelname,password)

AI_WQ_check_submission('pr',initdate,'1',teamname,modelname,password)
AI_WQ_check_submission('pr',initdate,'2',teamname,modelname,password)

AI_WQ_check_submission('mslp',initdate,'1',teamname,modelname,password)
AI_WQ_check_submission('mslp',initdate,'2',teamname,modelname,password)


# ------- PLOT & SAVE SUBMISSION -------
for i in range(1,6):
    plot_forecast(
        tas_p1_fc_submit,
        quintile_num = i,
        local_destination='/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/figures/')
    plot_forecast(
        tas_p2_fc_submit,
        quintile_num = i,
        local_destination='/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/figures/')
    plot_forecast(
        pr_p1_fc_submit,
        quintile_num = i,
        local_destination='/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/figures/')
    plot_forecast(
        pr_p2_fc_submit,
        quintile_num = i,
        local_destination='/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/figures/')
    plot_forecast(
        mslp_p1_fc_submit,
        quintile_num = i,
        local_destination='/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/figures/')
    plot_forecast(
        mslp_p2_fc_submit,
        quintile_num = i,
        local_destination='/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/figures/')