import xarray as xr
import os
import random
import subprocess
from datetime import datetime
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", required=True, help="Date in YYYY-mm-ddTHHMM format")
args = parser.parse_args()
# ----------------------- USER SPECIFIES --------------------------
DATE = pd.Timestamp(args.date).strftime("%Y-%m-%d")
# DATE = "2025-08-14"  # format: YYYY-MM-DD
ensembleSize = 11
# CAMIC_TXT_DIR = "/glade/work/sglanvil/CCR/kjmayer/camic_txt_files/"
CAMIC_TXT_DIR = "/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/AIWQ_inits/camic_txt_files/"
FINAL_IC_DIR = f"/glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/AIWQ_inits/{DATE}/"
# FINAL_IC_DIR = '/glade/derecho/scratch/cbecker/GEFS/'
# -----------------------------------------------------------------
os.makedirs(CAMIC_TXT_DIR, exist_ok=True)
os.makedirs(FINAL_IC_DIR, exist_ok=True)

date_obj = datetime.strptime(DATE, "%Y-%m-%d")
YEAR = date_obj.year
MONTH = date_obj.month
DAY = date_obj.day


counter=-11
GEFSmbrs = ['c00'] + [f'p{str(i).zfill(2)}' for i in range(1, 31)]
print(GEFSmbrs)
for GEFSmbr in GEFSmbrs:
    print()
    # original_file = f'/glade/derecho/scratch/dgagne/GEFS_CAM/gefs.{YEAR}{MONTH}{DAY}/00/gefs_cam_grid_{GEFSmbr}.nc'
    original_file = f'/glade/derecho/scratch/cbecker/GEFS_CAM/gefs.{YEAR}{MONTH:02}{DAY}/00/gefs_cam_grid_{GEFSmbr}.nc'
    print(f'GEFS mbr: {GEFSmbr} -- GEFS file: {original_file}')
    # ------------------- CREATE LIST OF PERTURBATION FILE VALUES ----------------------
    numbers = []
    seen = set()
    while len(numbers) < ensembleSize // 2:
        rand_num = random.randint(1, 999)
        if rand_num not in seen:
            numbers.append(rand_num)
            seen.add(rand_num)
    filename = os.path.join(CAMIC_TXT_DIR, f"camic_{DATE}.1-10_GEFS{GEFSmbr}.txt")
    with open(filename, "w") as f:
        f.write(" ".join(map(str, numbers)))
    with open(filename, "r") as f:
        print('random perts:',f.read())
    # --------------------------- MAKE THE ENSEMBLE MEMBERS ---------------------------
    counter=counter+11
    for i in range(1, ensembleSize + 1):
        print()
        mbr = f"{i:03d}"
        mbr_counter = i + counter
        mbr_forFilename = f"{mbr_counter:03d}"
        final_file = os.path.join(FINAL_IC_DIR, f"GEFS_ICs.{mbr_forFilename}.i.{DATE}-00000.nc")
        print(f"final: {final_file}")
        if mbr == "001":
            print(f"mbr: 001, use original: {original_file}")
            subprocess.run(["cp", original_file, final_file], check=True)
        else:
            mbr_num = int(mbr)
            if mbr_num % 2 == 0:
                inx = (mbr_num - 2) // 2
                weight = 0.15
            else:
                inx = (mbr_num - 3) // 2
                weight = -0.15
            pert_number = numbers[inx]
            pert_file = f"/glade/campaign/cesm/development/cross-wg/S2S/CESM2/CAMI/RP/{MONTH:02d}/CESM2.cam.i.M{MONTH:02d}.diff.{pert_number}.nc"
            pert_file_tmp = "/glade/derecho/scratch/cbecker/pert_file_tmp.nc"
            ds = xr.open_dataset(pert_file)
            ds = ds.rename({'US': 'U', 'VS': 'V'})
            ds['U'] = ds['U'].interp(slat=ds['lat'], lon=ds['lon'], method='nearest')
            ds['V'] = ds['V'].interp(slon=ds['lon'], lat=ds['lat'], method='nearest')
            ds = ds.rename({'Q': 'Qtot', 'lev': 'level', 'lat': 'latitude', 'lon': 'longitude'})
            ds.to_netcdf(pert_file_tmp)
            print(f"mbr: {mbr}, inx: {inx}, weight: {weight}, use pert: {pert_file}")
            # cmd = f'module load nco && ncflint -O -C -v lat,lon,slat,slon,lev,ilev,hyai,hybi,hyam,hybm,U,V,T,Q,PS -w {weight},1.0 {pert_file_tmp} {original_file} {final_file}'
            cmd = f'module load nco && ncflint -O -C -v latitude,longitude,level,U,V,T,Qtot,PS -w {weight},1.0 {pert_file_tmp} {original_file} {final_file}'
            os.system(f'bash -lc "{cmd} > /dev/null 2>&1"')
            ds_interp = xr.open_dataset(final_file).load()
            ds_interp = ds_interp.drop_vars(['latitude', 'longitude', 'level'], errors='ignore')
            ds_orig = xr.open_dataset(original_file).load()
            ds_final = xr.merge([ds_interp, ds_orig.drop_vars(ds_interp.data_vars.keys())])
            ###
            ds_final['level'] = ds_orig['level'].values # MODIFICATION BY CB TO FIX LEVELS FROM INDEX SPACE TO PRESSURE (?)
            ###
            os.remove(final_file)
            ds_final.to_netcdf(final_file)