from __future__ import annotations
import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import warnings
import psutil
import calendar
import xesmf as xe

# Suppress metadata noise
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- DIAGNOSTICS & RESUME LOGIC ---

def print_mem(label=""):
    mem = psutil.virtual_memory()
    print(f"--- [MEM {label}] Used: {mem.used / 1e9:.2f} GB | Avail: {mem.available / 1e9:.2f} GB ---")

def get_completed_months(out_zarr, year):
    if not os.path.exists(out_zarr):
        return []
    try:
        ds_check = xr.open_zarr(out_zarr)
        completed = []
        for m in range(1, 13):
            hours_present = (ds_check.time.dt.month == m).sum().values
            if hours_present == 0:
                continue
            expected_hours = calendar.monthrange(year, m)[1] * 24
            if hours_present == expected_hours:
                completed.append(m)
            else:
                raise RuntimeError(
                    f"\n!!! CORRUPTION RISK !!!\n"
                    f"Month {m} has {hours_present}/{expected_hours} hours.\n"
                    f"The previous job timed out mid-write. Delete the incomplete Zarr store."
                )
        return completed
    except RuntimeError as e:
        raise e  
    except Exception as e:
        print(f"Could not read existing Zarr metadata: {e}")
        return []

# --- PROCESSING LOGIC ---

def process_month(year, month, out_zarr, target_ds, target_levels, weight_file):
    ym = f"{year}{month:02d}"
    print(f"\n{'#'*70}\nSTARTING {ym} (xESMF Regridding)\n{'#'*70}")

    # Calculate previous month string to grab overlap data
    if month == 1:
        prev_ym = f"{year-1}12"
    else:
        prev_ym = f"{year}{month-1:02d}"

    ml_path = f"/glade/campaign/collections/rda/data/d633006/e5.oper.an.ml/{ym}/"
    pl_path = f"/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/{ym}/"
    sfc_an_path = f"/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/{ym}/"
    
    sfc_fc_path = f"/glade/campaign/collections/rda/data/d633000/e5.oper.fc.sfc.accumu/{ym}/"
    prev_sfc_fc_path = f"/glade/campaign/collections/rda/data/d633000/e5.oper.fc.sfc.accumu/{prev_ym}/"
    
    tsi_dir = "/glade/derecho/scratch/bagherio/cloud.dir/datasets/solar_forcing/"

    def clean(ds):
        drop_list = ['utc_date', 'utc_time', 'data_origin', 'expver', 'number']
        return ds.drop_vars(drop_list, errors='ignore')

    ds_list = []
    
    # ---------------------------------------------------------
    # 1. 3D Fields (ML & PL)
    # ---------------------------------------------------------
    v_ml = {'0_5_0_0_0_t': 'T', '0_5_0_1_0_q': 'Q', '0_5_0_2_2_u': 'U', '0_5_0_2_3_v': 'V', '128_134_sp': 'SP'}
    for c, name in v_ml.items():
        ds_v = clean(xr.open_mfdataset(sorted(glob.glob(f"{ml_path}*{c}*.nc")), chunks={'time': 24}))
        if 'level' in ds_v.dims: ds_v = ds_v.sel(level=target_levels, method='nearest')
        ds_list.append(ds_v[[(c.split('_')[-1].upper() if 'sp' not in c else 'SP')]])

    v_pl = {'128_133_q': 'Q500', '128_130_t': 'T500', '128_131_u': 'U500', '128_132_v': 'V500', '128_129_z': 'Z500'}
    for c, n in v_pl.items():
        ds_v = clean(xr.open_mfdataset(sorted(glob.glob(f"{pl_path}*{c}*.nc")), chunks={'time': 24}))
        ds_list.append(ds_v.sel(level=500).drop_vars('level').rename({c.split('_')[-1].upper(): n}))

    # ---------------------------------------------------------
    # 2. Surface Analysis Variables (Instantaneous)
    # ---------------------------------------------------------
    v_sfc_an = {
        '128_167_2t': 't2m', '128_034_sstk': 'sst', '128_151_msl': 'msl',
        '128_168_2d': 'd2m', '128_165_10u': 'u10', '128_166_10v': 'v10',
        '128_136_tcw': 'tcw', '128_244_fsr': 'fsr',    
        '128_067_laihv': 'lai_hv', '128_066_lailv': 'lai_lv'  
    }
    for c, n in v_sfc_an.items():
        files = sorted(glob.glob(f"{sfc_an_path}*{c}*.nc"))
        if files:
            ds_v = clean(xr.open_mfdataset(files, chunks={'time': 24}))
            var_name = list(ds_v.data_vars)[0] 
            ds_list.append(ds_v.rename({var_name: n}))

    # ---------------------------------------------------------
    # 3. Surface Forecast Variables (Accumulations/Fluxes)
    # ---------------------------------------------------------
    v_sfc_fc = {
        '128_169_ssrd': 'ssrd', '128_175_strd': 'strd', 
        '128_142_lsp': 'lsp', '128_143_cp': 'cp'      
    }
    for c, n in v_sfc_fc.items():
        # Load both the current month and the previous month's files to catch the overlap
        current_files = sorted(glob.glob(f"{sfc_fc_path}*{c}*.nc"))
        prev_files = sorted(glob.glob(f"{prev_sfc_fc_path}*{c}*.nc"))
        
        all_files = prev_files + current_files
        
        if all_files:
            ds_v = clean(xr.open_mfdataset(all_files))
            
            # Flatten the forecast matrix safely
            if 'forecast_initial_time' in ds_v.dims and 'forecast_hour' in ds_v.dims:
                ds_v = ds_v.stack(stacked_time=('forecast_initial_time', 'forecast_hour'))
                
                if np.issubdtype(ds_v.forecast_hour.dtype, np.number):
                    hours = pd.to_timedelta(ds_v.forecast_hour.values, unit='h')
                else:
                    hours = ds_v.forecast_hour.values
                
                true_times = ds_v.forecast_initial_time.values + hours
                
                ds_v = ds_v.assign_coords(time=('stacked_time', true_times))
                ds_v = ds_v.swap_dims({'stacked_time': 'time'})
                ds_v = ds_v.drop_vars(['stacked_time', 'forecast_initial_time', 'forecast_hour'], errors='ignore')
                
                ds_v = ds_v.drop_duplicates(dim='time').sortby('time')
                ds_v = ds_v.chunk({'time': 24})

            var_name = list(ds_v.data_vars)[0]
            ds_list.append(ds_v.rename({var_name: n}))

    # 4. Solar Irradiance (TSI)
    tsi_file = glob.glob(f"{tsi_dir}solar_irradiance_{year}*.nc")[0]
    ds_list.append(clean(xr.open_dataset(tsi_file, chunks={'time': 24})))

    # ---------------------------------------------------------
    # Align Grids & Load
    # ---------------------------------------------------------
    print("Aligning and Loading into RAM...")
    master_coords = {'latitude': ds_list[0].latitude, 'longitude': ds_list[0].longitude, 'time': ds_list[0].time}
    
    # We use join='right' against the master coordinates (which are exactly the current month's hours)
    # This automatically trims off the extra data from the previous month that we don't need!
    ds_combined = xr.merge([d.interp(master_coords, method="linear") for d in ds_list], join='right')
    ds_combined = ds_combined.load()

    # ---------------------------------------------------------
    # Regridding & Saving
    # ---------------------------------------------------------
    print("Regridding variables sequentially with xESMF weights...")
    regridder = xe.Regridder(
        ds_combined, target_ds, method='bilinear', 
        weights=weight_file, reuse_weights=True, unmapped_to_nan=True 
    )
    
    spatial_vars = [var for var in ds_combined.data_vars 
                    if 'latitude' in ds_combined[var].dims and 'longitude' in ds_combined[var].dims]
    
    regridded_vars = {}
    for var in spatial_vars:
        regridded_vars[var] = regridder(ds_combined[var]).astype(np.float32)

    ds_final = xr.Dataset(regridded_vars)
    
    if 'level' in ds_final.dims:
        ds_final = ds_final.transpose("time", "level", "latitude", "longitude")
    else:
        ds_final = ds_final.transpose("time", "latitude", "longitude")
        
    ds_final = ds_final.assign_coords({
        "latitude": target_ds.latitude,
        "longitude": target_ds.longitude,
        "time": ds_combined.time,
        "level": ds_combined.level if 'level' in ds_combined.dims else None
    })
    if ds_final.level is None: ds_final = ds_final.drop_vars("level", errors='ignore')

    encoding = {v: {'chunks': (1, 16, 1024, 960) if 'level' in ds_final[v].dims else (1, 1024, 960)} 
                for v in ds_final.data_vars}

    if not os.path.exists(out_zarr):
        print(f"Initializing Zarr: {out_zarr}")
        ds_final.to_zarr(out_zarr, mode='w', encoding=encoding, consolidated=True)
    else:
        print(f"Appending {ym}...")
        ds_final.to_zarr(out_zarr, append_dim='time', consolidated=True)
    
    print_mem(f"End {ym}")

if __name__ == "__main__":
    ref_ds = xr.open_zarr("/glade/derecho/scratch/dkimpara/goes-cloud-dataset/era5_regrid/era5_interp_2022.zarr")
    target_ds = xr.Dataset({
        'latitude': (['latitude'], ref_ds.latitude.values),
        'longitude': (['longitude'], ref_ds.longitude.values),
    })
    t_levs = ref_ds.level.values
    weight_file = "/glade/derecho/scratch/dkimpara/goes-cloud-dataset/regrid_files/regrid_era5_to_goes.nc"
    
    for year in [2025]: 
        out = f"/glade/derecho/scratch/bagherio/cloud.dir/datasets/era5_dataset/era5_interp_{year}.zarr"
        completed_months = get_completed_months(out, year)
        
        for m in range(1, 13):
            if m in completed_months:
                print(f"\n--- [SKIP] {year}-{m:02d} is 100% complete. Moving to next. ---")
                continue
            process_month(year, m, out, target_ds, t_levs, weight_file)