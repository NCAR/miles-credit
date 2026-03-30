"""
WXFormer v2 WeatherBench-style ACC and RMSE verification.

Output format matches the CREDIT-arXiv verification files exactly:
  ACC_006h_240h_wxformer_v2.nc  : dims (days, time=40), vars Z500/T500/U500/V500/Q500/SP/t2m
  RMSE_006h_240h_wxformer_v2.nc : same 2D vars + U/V/T/Q at level=120

Methodology:
  - Spatial average: da.weighted(cos(lat)/mean(cos(lat))).mean(['latitude','longitude'])
  - ACC = sp_avg(pred_anom * true_anom) / sqrt(sp_avg(pred_anom²) * sp_avg(true_anom²))
  - RMSE = sqrt(sp_avg((pred - truth)²))
  - Climatology indexed by (dayofyear, hour); ERA5 1990-2019 6-hourly

Truth source: ERA5_mlevel_cesm_stage1 (192x288 Gaussian grid)
  - Z500, SP, t2m: pre-computed 2D fields
  - T/U/V/Q at 500 hPa: log-pressure interpolation from hybrid model levels (numpy, no numba)

Usage
-----
python applications/wxformer_wb2_verif.py --forecast /path/to/netcdf/ --out /path/to/out/
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Path to ERA5 hybrid level coefficients (relative to credit package root)
_CREDIT_ROOT = os.path.join(os.path.dirname(__file__), "..")
LEV_INFO_FILE = os.path.join(_CREDIT_ROOT, "credit", "metadata", "ERA5_Lev_Info.nc")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CESM_GLOB = (
    "/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/all_in_one/ERA5_mlevel_cesm_6h_lev16_*.zarr"
)
DEFAULT_CLIM_PATH = "/glade/campaign/cisl/aiml/ksha/CREDIT_CESM/VERIF/ERA5_clim/ERA5_clim_1990_2019_6h_cesm_interp.nc"
DEFAULT_OUT = "/glade/work/schreck/CREDIT_verif/wxformer_v2"

# 40 lead times: 6h...240h (matches CREDIT-arXiv convention)
LEAD_HOURS = list(range(6, 241, 6))

# Climatology level index for 500 hPa (level 0–12 = 50–1000 hPa; level 7 = 500 hPa)
CLIM_LEV_500 = 7

# Model level saved for U/V/T/Q RMSE (must exist in forecast level coord)
MODEL_LEV = 120


# ---------------------------------------------------------------------------
# Spatial averaging — exact match to CREDIT-arXiv sp_avg
# ---------------------------------------------------------------------------
def _sp_avg(da, w_lat):
    return da.weighted(w_lat).mean(["latitude", "longitude"], skipna=False)


def _make_w_lat(da):
    w = np.cos(np.deg2rad(da.latitude.values))
    return xr.DataArray(w / w.mean(), dims=["latitude"], coords={"latitude": da.latitude})


def _acc(pred_anom, true_anom, w_lat):
    top = _sp_avg(pred_anom * true_anom, w_lat)
    bottom = np.sqrt(_sp_avg(pred_anom**2, w_lat) * _sp_avg(true_anom**2, w_lat))
    return top / (bottom + 1e-12)


def _rmse(pred, truth, w_lat):
    return np.sqrt(_sp_avg((pred - truth) ** 2, w_lat))


# ---------------------------------------------------------------------------
# Pure-numpy log-pressure interpolation to 500 hPa (no numba)
# ---------------------------------------------------------------------------
def _interp_to_500hpa(ds_truth, lev_file=LEV_INFO_FILE, fields=("T", "U", "V", "Q"), p_target_hpa=500.0):
    """
    Interpolate T/U/V/Q from hybrid model levels to a target pressure level
    using log-pressure linear interpolation.  Pure numpy — no numba.

    Parameters
    ----------
    ds_truth : xr.Dataset  (time, level, latitude, longitude)
    lev_file  : path to ERA5_Lev_Info.nc
    fields    : variable names to interpolate
    p_target_hpa : target pressure in hPa

    Returns
    -------
    xr.Dataset with variables {v}_PRES of shape (time, 1, lat, lon),
    pressure coordinate = [p_target_hpa].
    """
    SP = ds_truth["SP"].values.astype(np.float64)  # (T, lat, lon)  Pa
    levels = ds_truth.level.values  # e.g. [1,2,...,16]

    with xr.open_dataset(lev_file) as lev_ds:
        valid = np.isin(lev_ds.level.values, levels)
        a = lev_ds["a_model"].values[valid]  # (nlev,) Pa
        b = lev_ds["b_model"].values[valid]  # (nlev,) dimensionless

    # Pressure at every model level: (T, nlev, lat, lon)
    P = a[None, :, None, None] + b[None, :, None, None] * SP[:, None, :, :]

    p_target_pa = p_target_hpa * 100.0
    log_P = np.log(P)
    log_pt = np.log(p_target_pa)

    # Upper bracketing index: number of levels with P <= p_target, minus 1
    # (levels ordered top→bottom so P increases with index)
    above = np.sum(P <= p_target_pa, axis=1) - 1  # (T, lat, lon)
    above = np.clip(above, 0, len(levels) - 2)

    T_idx = np.arange(SP.shape[0])[:, None, None]
    lat_idx = np.arange(SP.shape[1])[None, :, None]
    lon_idx = np.arange(SP.shape[2])[None, None, :]

    log_Pa = log_P[T_idx, above, lat_idx, lon_idx]
    log_Pb = log_P[T_idx, above + 1, lat_idx, lon_idx]
    denom = log_Pb - log_Pa
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    alpha = np.clip((log_pt - log_Pa) / denom, 0.0, 1.0)

    out_ds = xr.Dataset()
    for v in fields:
        fld = ds_truth[v].values.astype(np.float64)  # (T, nlev, lat, lon)
        fa = fld[T_idx, above, lat_idx, lon_idx]
        fb = fld[T_idx, above + 1, lat_idx, lon_idx]
        interp = (fa + alpha * (fb - fa)).astype(np.float32)

        out_ds[v + "_PRES"] = xr.DataArray(
            interp[:, np.newaxis, :, :],
            dims=["time", "pressure", "latitude", "longitude"],
            coords={
                "time": ds_truth.time,
                "pressure": np.array([p_target_hpa], dtype=np.float32),
                "latitude": ds_truth.latitude,
                "longitude": ds_truth.longitude,
            },
        )
    return out_ds


# ---------------------------------------------------------------------------
# Build output dataset in Kyle's exact format
# ---------------------------------------------------------------------------
def _build_dataset(results_list, kind):
    n_days = len(results_list)
    n_time = len(LEAD_HOURS)

    var_names = list(results_list[0][kind].keys())
    doy = np.array([r["dayofyear"] for r in results_list])  # (days, time)
    hour = np.array([r["hour"] for r in results_list])

    ds = xr.Dataset()
    for v in var_names:
        data = np.array([r[kind][v] for r in results_list])  # (days, time)
        if v in ("T", "U", "V", "Q"):
            ds[v] = xr.DataArray(data[:, :, np.newaxis], dims=["days", "time", "level"])
        else:
            ds[v] = xr.DataArray(data, dims=["days", "time"])

    ds = ds.assign_coords(
        dayofyear=(["days", "time"], doy),
        hour=(["days", "time"], hour),
    )
    if kind == "rmse":
        ds = ds.assign_coords(level=("level", [MODEL_LEV]))
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WXFormer v2 WeatherBench ACC+RMSE verification")
    parser.add_argument("--forecast", required=True, help="Root dir of forecast NetCDFs (one subdir per init date)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cesm", default=DEFAULT_CESM_GLOB)
    parser.add_argument("--clim", default=DEFAULT_CLIM_PATH)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect init dirs
    # ------------------------------------------------------------------
    init_dirs = sorted(
        [
            os.path.join(args.forecast, d)
            for d in os.listdir(args.forecast)
            if os.path.isdir(os.path.join(args.forecast, d)) and d != "metrics"
        ]
    )
    logger.info(f"{len(init_dirs)} init dirs found")

    # ------------------------------------------------------------------
    # Pre-load truth: open cesm_stage1 lazily, collect all verif times,
    # run pressure interpolation ONCE across all inits
    # ------------------------------------------------------------------
    logger.info("Opening cesm_stage1 zarr (lazy)...")
    cesm_files = sorted(glob.glob(args.cesm))
    ds_cesm = xr.open_mfdataset(cesm_files, combine="by_coords", engine="zarr")

    # gather all verification timestamps needed
    all_times = set()
    init_to_times = {}
    for init_dir in init_dirs:
        name = os.path.basename(init_dir)
        lead_files = []
        ok = True
        for lh in LEAD_HOURS:
            pat = os.path.join(init_dir, f"pred_{name}_{lh:03d}.nc")
            m = glob.glob(pat)
            if not m:
                ok = False
                break
            lead_files.append(m[0])
        if not ok:
            logger.warning(f"Missing lead files for {name}, skipping")
            continue
        # peek at verification times from first file (extend by lead offsets)
        t0 = pd.Timestamp(name.replace("T", " ").replace("Z", ""))
        vtimes = [np.datetime64(t0 + pd.Timedelta(hours=lh)) for lh in LEAD_HOURS]
        init_to_times[init_dir] = (lead_files, vtimes)
        all_times.update(vtimes)

    all_times_sorted = sorted(all_times)
    logger.info(f"Loading {len(all_times_sorted)} unique truth timesteps from cesm...")
    ds_truth_raw = ds_cesm.sel(time=all_times_sorted).load()
    logger.info("cesm load complete")

    # pressure-interpolate T/U/V/Q at 500 hPa for ALL times at once
    # (pure numpy log-pressure interpolation; no numba required)
    logger.info("Interpolating T/U/V/Q to 500 hPa (numpy)...")
    ds_pres_truth = _interp_to_500hpa(ds_truth_raw)
    logger.info("Pressure interpolation complete")

    # ------------------------------------------------------------------
    # Load climatology once
    # ------------------------------------------------------------------
    logger.info("Loading climatology...")
    ds_clim_full = xr.open_dataset(args.clim).load()
    logger.info("Climatology loaded")

    # ------------------------------------------------------------------
    # Loop over inits: load forecast, slice truth, compute ACC + RMSE
    # ------------------------------------------------------------------
    results = []
    n_inits = len(init_to_times)

    for i, (init_dir, (lead_files, vtimes)) in enumerate(sorted(init_to_times.items())):
        name = os.path.basename(init_dir)

        # load forecast (40 files → concat along time)
        ds_fc = xr.open_mfdataset(lead_files, combine="nested", concat_dim="time").load()

        # slice pre-computed truth for this init's 40 timestamps
        truth_times = np.array(vtimes)
        ds_truth = ds_truth_raw.sel(time=truth_times)
        ds_pres_t = ds_pres_truth.sel(time=truth_times)

        # build climatology for these (dayofyear, hour) values
        dt_index = pd.DatetimeIndex(truth_times)
        doy_arr = dt_index.day_of_year.values
        hour_arr = dt_index.hour.values
        clim_list = [ds_clim_full.sel(dayofyear=int(d), hour=int(h)) for d, h in zip(doy_arr, hour_arr)]
        ds_clim = xr.concat(clim_list, dim="time")
        ds_clim["time"] = ds_fc.time

        # latitude weights
        w = _make_w_lat(ds_fc["Z500"])

        # variable pairs: (forecast, truth, climatology)
        var_map = {
            "Z500": (ds_fc["Z500"].squeeze(), ds_truth["Z500"], ds_clim["geopotential"].isel(level=CLIM_LEV_500)),
            "T500": (
                ds_fc["T_PRES"].sel(pressure=500.0).squeeze(),
                ds_pres_t["T_PRES"].sel(pressure=500.0),
                ds_clim["temperature"].isel(level=CLIM_LEV_500),
            ),
            "U500": (
                ds_fc["U_PRES"].sel(pressure=500.0).squeeze(),
                ds_pres_t["U_PRES"].sel(pressure=500.0),
                ds_clim["u_component_of_wind"].isel(level=CLIM_LEV_500),
            ),
            "V500": (
                ds_fc["V_PRES"].sel(pressure=500.0).squeeze(),
                ds_pres_t["V_PRES"].sel(pressure=500.0),
                ds_clim["v_component_of_wind"].isel(level=CLIM_LEV_500),
            ),
            "Q500": (
                ds_fc["Q_PRES"].sel(pressure=500.0).squeeze(),
                ds_pres_t["Q_PRES"].sel(pressure=500.0),
                ds_clim["specific_humidity"].isel(level=CLIM_LEV_500),
            ),
            "SP": (ds_fc["SP"].squeeze(), ds_truth["SP"], ds_clim["surface_pressure"]),
            "t2m": (ds_fc["t2m"].squeeze(), ds_truth["t2m"], ds_clim["2m_temperature"]),
        }
        mlev_map = {
            "T": (ds_fc["T"].sel(level=MODEL_LEV), ds_truth["T"].sel(level=MODEL_LEV)),
            "U": (ds_fc["U"].sel(level=MODEL_LEV), ds_truth["U"].sel(level=MODEL_LEV)),
            "V": (ds_fc["V"].sel(level=MODEL_LEV), ds_truth["V"].sel(level=MODEL_LEV)),
            "Q": (ds_fc["Q"].sel(level=MODEL_LEV), ds_truth["Q"].sel(level=MODEL_LEV)),
        }

        acc_vars = {}
        rmse_vars = {}
        for vname, (pred, truth, clim) in var_map.items():
            acc_vars[vname] = _acc(pred - clim, truth - clim, w).values
            rmse_vars[vname] = _rmse(pred, truth, w).values
        for vname, (pred, truth) in mlev_map.items():
            rmse_vars[vname] = _rmse(pred, truth, w).values

        results.append(
            {
                "acc": acc_vars,
                "rmse": rmse_vars,
                "dayofyear": doy_arr,
                "hour": hour_arr,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == n_inits:
            logger.info(f"[{i + 1}/{n_inits}] {name}  Z500_acc6h={acc_vars['Z500'][0]:.5f}")

        ds_fc.close()

    logger.info(f"Processed {len(results)} inits successfully")

    # sort chronologically by first verification time
    results.sort(key=lambda r: (r["dayofyear"][0], r["hour"][0]))

    # ------------------------------------------------------------------
    # Save Kyle-format output files
    # ------------------------------------------------------------------
    ds_acc = _build_dataset(results, "acc")
    ds_rmse = _build_dataset(results, "rmse")

    acc_path = os.path.join(args.out, "ACC_006h_240h_wxformer_v2.nc")
    rmse_path = os.path.join(args.out, "RMSE_006h_240h_wxformer_v2.nc")
    ds_acc.to_netcdf(acc_path)
    ds_rmse.to_netcdf(rmse_path)
    logger.info(f"ACC  → {acc_path}")
    logger.info(f"RMSE → {rmse_path}")

    # per-100-init batch files (matches CREDIT-arXiv naming convention)
    n = len(results)
    for start in range(0, n, 100):
        end = min(start + 100, n)
        batch = results[start:end]
        _build_dataset(batch, "acc").to_netcdf(
            os.path.join(args.out, f"combined_acc_{start:04d}_{end:04d}_006h_240h_wxformer_v2.nc")
        )
        _build_dataset(batch, "rmse").to_netcdf(
            os.path.join(args.out, f"combined_rmse_{start:04d}_{end:04d}_006h_240h_wxformer_v2.nc")
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
