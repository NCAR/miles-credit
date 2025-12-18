import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe
import matplotlib.pyplot as plt
import sys
import glob
import argparse
from AI_WQ_package import retrieve_evaluation_data as red
from AI_WQ_package import forecast_evaluation as fe

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", required=True, help="Date in YYYY-mm-ddTHHMM format")
args = parser.parse_args()
init = pd.Timestamp(args.date).strftime("%Y-%m-%d")

def make_quintile_array():
    return xr.DataArray(
        coords={"quintile": [0.2,0.4,0.6,0.8,1.0], "latitude": tas1.latitude,
                "longitude": tas1.longitude},
        dims=["quintile", "latitude", "longitude"]
    )

def grab_quintiles(array, quintile_cutoff):
    quintile_array = make_quintile_array()

    quintile_array[0] = xr.where(
        array<quintile_cutoff[0],
        x=1,y=0
    ).sum('member') / array.member.shape[0]

    quintile_array[1] = xr.where(
        (array>=quintile_cutoff[0]) & (array<quintile_cutoff[1]),
        x=1,y=0
    ).sum('member') / array.member.shape[0]

    quintile_array[2] = xr.where(
        (array>=quintile_cutoff[1]) & (array<quintile_cutoff[2]),
        x=1,y=0
    ).sum('member') / array.member.shape[0]

    quintile_array[3] = xr.where(
        (array>=quintile_cutoff[2]) & (array<quintile_cutoff[3]),
        x=1,y=0
    ).sum('member') / array.member.shape[0]

    quintile_array[4] = xr.where(
        array>=quintile_cutoff[3],
        x=1,y=0
    ).sum('member') / array.member.shape[0]

    return quintile_array


# ------------ DEFINE LEADS ------------
password = ""

# pick initialization:
#init = "2025-09-04"  # YYYY-MM-DD for CREDIT rollout files
init_dt = pd.to_datetime(init)  # dt object for timedelta calc
initdate = init_dt.strftime("%Y%m%d")  # YYYYMMDD for AIWQ functions

# get to the next Monday then add number of weeks
# https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html
day19 = init_dt + pd.Timedelta(days=4 + (7 * 2))
day25 = init_dt + pd.Timedelta(days=4 + (7 * 2) + 6, hours=18)

day26 = init_dt + pd.Timedelta(days=4 + (7 * 3))
day32 = init_dt + pd.Timedelta(days=4 + (7 * 3) + 6, hours=18)

# convert times into YYYYMMDD
day19_date = day19.strftime("%Y%m%d")
day25_date = day25.strftime("%Y%m%d")
day26_date = day26.strftime("%Y%m%d")
day32_date = day32.strftime("%Y%m%d")


# ------------ LOAD FORECAST ------------
ddir_rollout = "/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/AIWQ_rollout/camulator/"
ddir_init = ddir_rollout + init + "T00Z/"

fc_all = []
for mem in np.arange(1, 342):
    print(mem)
    file = "pred_" + str(mem).zfill(3) + "_*.nc"
    files = sorted(glob.glob(ddir_init + file))
    dataset = []
    for f in files:
        data = xr.open_dataset(f, engine="netcdf4")
        dataset.append(data)
    all_data = xr.concat(dataset, dim="time")
    fc_all.append(all_data)
fc = xr.concat(fc_all, dim="member")


# ------------ GRAB WEEK 3-4 & CALC MEAN/SUM ------------
# .sel() is inclusive of both the start and end values in the slice
tas1 = fc["TREFHT"].sel(time=slice(day19, day25)).mean("time")
tas2 = fc["TREFHT"].sel(time=slice(day26, day32)).mean("time")

pr1 = fc["PRECT"].sel(time=slice(day19, day25 + pd.Timedelta(hours=18))).sum("time")*1000
pr2 = fc["PRECT"].sel(time=slice(day26, day32 + pd.Timedelta(hours=18))).sum("time")*1000

mslp1 = fc["mean_sea_level_pressure"].sel(time=slice(day19, day25)).mean("time")
mslp2 = fc["mean_sea_level_pressure"].sel(time=slice(day26, day32)).mean("time")


# ------------ LOAD CESM QUINTILES ------------
ddir_quintiles = "/glade/work/jaye/aiwq/"
quintile_finame = "aiwq.20yr.quintiles.weekly.nc"
quintile_cutoffs_all = xr.open_dataset(ddir_quintiles + quintile_finame)

# SELECT quintile_cutoffs DAY HERE - use forecast/lead day
quintile_cutoffs1 = quintile_cutoffs_all.sel(time=day19)
quintile_cutoffs2 = quintile_cutoffs_all.sel(time=day26)

# convert forecast to quintiles
tas1_pbs = grab_quintiles(tas1, quintile_cutoffs1["tas"])
tas2_pbs = grab_quintiles(tas2, quintile_cutoffs2["tas"])

pr1_pbs = grab_quintiles(pr1, quintile_cutoffs1["pr"])
pr2_pbs = grab_quintiles(pr2, quintile_cutoffs2["pr"])

mslp1_pbs = grab_quintiles(mslp1, quintile_cutoffs1["mslp"])
mslp2_pbs = grab_quintiles(mslp2, quintile_cutoffs2["mslp"])


# ------------ REGRID DATA ------------
# NOTE: need to install conda install -c conda-forge xesmf
# AIWQ Required lat-lon grid:
# Latitude: Ranges from 90.0°N to -90.0°N with a step of -1.5° latitude.
# Longitude: Ranges from 0.0° to 358.5° longitude with a step of 1.5°.

ds_in = xr.Dataset(
    {"latitude": tas1_pbs["latitude"], "longitude": tas1_pbs["longitude"]}
)

ds_out = xr.Dataset(
    {
        "latitude": (["latitude"], np.arange(-90, 91.5, 1.5)),
        "longitude": (["longitude"], np.arange(0, 360, 1.5)),
    }
)

regridder = xe.Regridder(ds_in, ds_out, "bilinear")

tas1_pbs_regrid = regridder(tas1_pbs, keep_attrs=True)
tas2_pbs_regrid = regridder(tas2_pbs, keep_attrs=True)

pr1_pbs_regrid = regridder(pr1_pbs, keep_attrs=True)
pr2_pbs_regrid = regridder(pr2_pbs, keep_attrs=True)

mslp1_pbs_regrid = regridder(mslp1_pbs, keep_attrs=True)
mslp2_pbs_regrid = regridder(mslp2_pbs, keep_attrs=True)


# ------------ SAVE UPDATED FORECAST ------------
# directory and file names
ddir_forecast = (
    "/glade/derecho/scratch/cbecker/CREDIT_runs_realtime/submitted_forecasts/"
)
file_forecast_week3 = (
    "init" + initdate + "_forecast" + day19_date + "-" + day25_date + ".nc"
)
file_forecast_week4 = (
    "init" + initdate + "_forecast" + day26_date + "-" + day32_date + ".nc"
)

# Save files
tas1_pbs_regrid.to_netcdf(ddir_forecast + "tas_" + file_forecast_week3)
tas2_pbs_regrid.to_netcdf(ddir_forecast + "tas_" + file_forecast_week4)

pr1_pbs_regrid.to_netcdf(ddir_forecast + "pr_" + file_forecast_week3)
pr2_pbs_regrid.to_netcdf(ddir_forecast + "pr_" + file_forecast_week4)

mslp1_pbs_regrid.to_netcdf(ddir_forecast + "mslp_" + file_forecast_week3)
mslp2_pbs_regrid.to_netcdf(ddir_forecast + "mslp_" + file_forecast_week4)