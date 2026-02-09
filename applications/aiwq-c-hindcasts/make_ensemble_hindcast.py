import xarray as xr
import os
import random
import subprocess
from datetime import datetime
import argparse
import pandas as pd
import xesmf as xe

# from applications.AIWQ_realtime.make_ensemble_aiwq import original_file

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", required=True, help="Date in YYYY-mm-ddTHHMM format")
args = parser.parse_args()
# ----------------------- USER SPECIFIES --------------------------
DATE = pd.Timestamp(args.date).strftime("%Y-%m-%d")
# DATE = "2025-08-14"  # format: YYYY-MM-DD
ensembleSize = 11
USER = os.environ['USER']
FINAL_IC_DIR = f'/glade/derecho/scratch/{USER}/DATA/CESMS2S_inits/'+DATE+'/'
# -----------------------------------------------------------------
os.makedirs(FINAL_IC_DIR, exist_ok=True)

date_obj = datetime.strptime(DATE, "%Y-%m-%d")
YEAR = date_obj.year
MONTH = date_obj.month
DAY = date_obj.day

original_file = FINAL_IC_DIR+'CESM_'+DATE+'.nc'
# ------------------- CREATE LIST OF PERTURBATION FILE VALUES ----------------------
numbers = []
seen = set()
while len(numbers) < ensembleSize // 2:
    rand_num = random.randint(1, 999)
    if rand_num not in seen:
        numbers.append(rand_num)
        seen.add(rand_num)
filename = os.path.join(FINAL_IC_DIR, f"camic_{DATE}.txt")
with open(filename, "w") as f:
    f.write(" ".join(map(str, numbers)))
with open(filename, "r") as f:
    print('random perts:', f.read())
# --------------------------- MAKE THE ENSEMBLE MEMBERS ---------------------------
create_regridder = True
for i in range(1, ensembleSize + 1):
    print()
    member = f"{i:03}"
    final_file = os.path.join(FINAL_IC_DIR, f"ICs.{member}.{DATE}-00000.nc")
    print(f"final: {final_file}")
    if i == 1:
        print(f"member 1: use original: {original_file}")
        subprocess.run(["cp", original_file, final_file], check=True)
    else:
        if i % 2 == 0:
            inx = (i - 2) // 2
            weight = 0.15
        else:
            inx = (i - 3) // 2
            weight = -0.15
        pert_number = numbers[inx]
        pert_file = f"/glade/campaign/cesm/development/cross-wg/S2S/CESM2/CAMI/RP/{MONTH:02d}/CESM2.cam.i.M{MONTH:02d}.diff.{pert_number}.nc"
        pert_file_tmp = f"/glade/derecho/scratch/{USER}/pert_file_"+DATE+"_tmp.nc"
        ds = xr.open_dataset(pert_file)
        ds = ds.rename({'US': 'U', 'VS': 'V'})

        ###
        ds_in_U = xr.Dataset(
            {"slat": ds['slat'],
             "lon": ds['lon']})
        ds_in_V = xr.Dataset(
            {"lat": ds['lat'],
             "slon": ds['slon']})

        ds_out_U = xr.Dataset(
            {"slat": ds['lat'],
             "lon": ds['lon']})
        ds_out_V = xr.Dataset(
            {"lat": ds['lat'],
             "slon": ds['lon']})
        if create_regridder:
            U_regridder = xe.Regridder(ds_in_U, ds_out_U, "bilinear")
            V_regridder = xe.Regridder(ds_in_V, ds_out_V, "bilinear")
            create_regridder = False
        ds['U'] = U_regridder(ds['U'], keep_attrs=True)
        ds['U'].loc[dict(lat=-90)] = ds['U'].isel(lat=1).values
        ds['U'].loc[dict(lat=90)] = ds['U'].isel(lat=-2).values

        ds['V'] = V_regridder(ds['V'], keep_attrs=True)
        ds['V'].loc[dict(lon=358.75)] = ds['V'].isel(lon=0).values

        # ds['U'] = ds['U'].interp(slat=ds['lat'], lon=ds['lon'], method='nearest')
        # ds['V'] = ds['V'].interp(slon=ds['lon'], lat=ds['lat'], method='nearest')
        
        ds = ds.rename({'Q': 'Qtot', 'lev': 'level', 'lat': 'latitude', 'lon': 'longitude'})
        ds.to_netcdf(pert_file_tmp)
        print(f"mbr: {i}, inx: {inx}, weight: {weight}, use pert: {pert_file}")
        cmd = f'module load nco && ncflint -O -C -v latitude,longitude,level,U,V,T,Qtot,PS -w {weight},1.0 {pert_file_tmp} {original_file} {final_file}'
        os.system(f'bash -lc "{cmd} > /dev/null 2>&1"')
        
        ds_interp = xr.open_dataset(final_file).load()
        ds_interp = ds_interp.drop_vars(['latitude', 'longitude', 'level'], errors='ignore')
        ds_orig = xr.open_dataset(original_file).load()
        ds_final = xr.merge([ds_interp, ds_orig.drop_vars(ds_interp.data_vars.keys())])
        ###
        os.remove(final_file)
        ds_final.to_netcdf(final_file)
