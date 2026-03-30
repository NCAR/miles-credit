"""
CREDIT ensemble WeatherBench-style verification: CRPS, spread, RMSE, ACC.

Compares 100-member CREDIT ensemble forecasts vs ERA5 cesm_stage1.

Output format matches wxformer_wb2_verif.py:
  ACC_006h_240h_{tag}.nc    : dims (days, time=40), vars Z500/T500/U500/V500/Q500/SP/t2m
  RMSE_006h_240h_{tag}.nc   : ensemble-mean RMSE
  CRPS_006h_240h_{tag}.nc   : spatially-averaged CRPS
  SPREAD_006h_240h_{tag}.nc : spatially-averaged ensemble std

CRPS uses the sorted-ensemble formula (O(n log n), memory-efficient):
  CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
  E[|X - X'|] = (2/n²) * sum_i (2i - n + 1) * x_{(i)}  (i: 0-indexed sorted members)

Usage
-----
python applications/ensemble_wb2_verif.py \\
    --forecast /glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/netcdf_pressure_interp \\
    --out /glade/work/schreck/CREDIT_verif/ensemble
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

DEFAULT_FORECAST = "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/netcdf_pressure_interp"
DEFAULT_CESM_GLOB = (
    "/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/all_in_one/ERA5_mlevel_cesm_6h_lev16_*.zarr"
)
DEFAULT_CLIM_PATH = "/glade/campaign/cisl/aiml/ksha/CREDIT_CESM/VERIF/ERA5_clim/ERA5_clim_1990_2019_6h_cesm_interp.nc"
DEFAULT_OUT = "/glade/work/schreck/CREDIT_verif/ensemble"

# 40 lead times: 6h..240h (matches det verification convention)
LEAD_HOURS = list(range(6, 241, 6))

# Level index in ERA5_clim for 500 hPa (levels 50..1000 hPa, 13 levels; 500 hPa = index 7)
CLIM_LEV_500 = 7


# ---------------------------------------------------------------------------
# Spatial averaging — identical to wxformer_wb2_verif.py
# ---------------------------------------------------------------------------
def _sp_avg(da, w_lat):
    return da.weighted(w_lat).mean(["latitude", "longitude"], skipna=False)


def _make_w_lat_xr(da):
    w = np.cos(np.deg2rad(da.latitude.values))
    return xr.DataArray(w / w.mean(), dims=["latitude"], coords={"latitude": da.latitude})


def _acc(pred_anom, true_anom, w_lat):
    top = _sp_avg(pred_anom * true_anom, w_lat)
    bottom = np.sqrt(_sp_avg(pred_anom**2, w_lat) * _sp_avg(true_anom**2, w_lat))
    return top / (bottom + 1e-12)


def _rmse(pred, truth, w_lat):
    return np.sqrt(_sp_avg((pred - truth) ** 2, w_lat))


# ---------------------------------------------------------------------------
# CRPS + spread — vectorized sorted-ensemble formula
# ---------------------------------------------------------------------------
def _crps_spread(pred_ens_np, truth_np, w_np):
    """
    Spatially-averaged CRPS and ensemble spread for one lead time.

    Parameters
    ----------
    pred_ens_np : (n_members, lat, lon)  float64
    truth_np    : (lat, lon)             float64
    w_np        : (lat,)                 normalized cos-lat weights

    Returns
    -------
    crps   : scalar
    spread : scalar  (spatial avg of ensemble std, ddof=1)
    """
    n, n_lat, n_lon = pred_ens_np.shape
    w2d = w_np[:, None]  # (lat, 1)

    def sp_avg(arr2d):
        return (arr2d * w2d).sum() / (n_lat * n_lon)

    # E[|X - y|]: mean over members, then spatial avg
    abs_err = np.abs(pred_ens_np - truth_np[None]).mean(axis=0)  # (lat, lon)
    term1 = sp_avg(abs_err)

    # E[|X - X'|] via sorted ensemble: O(n log n), memory O(n * lat * lon)
    sorted_ens = np.sort(pred_ens_np, axis=0)
    wk = (2 * np.arange(n) - n + 1).reshape(n, 1, 1).astype(np.float64)
    energy_map = (2.0 / (n * n)) * (sorted_ens * wk).sum(axis=0)
    term2 = sp_avg(energy_map)

    crps = float(term1 - 0.5 * term2)

    std_map = pred_ens_np.std(axis=0, ddof=1)
    spread = float(sp_avg(std_map))

    return crps, spread


# ---------------------------------------------------------------------------
# Build output dataset — same format as wxformer_wb2_verif.py
# ---------------------------------------------------------------------------
def _build_dataset(results_list, kind):
    var_names = list(results_list[0][kind].keys())
    doy = np.array([r["dayofyear"] for r in results_list])
    hour = np.array([r["hour"] for r in results_list])

    ds = xr.Dataset()
    for v in var_names:
        data = np.array([r[kind][v] for r in results_list])  # (days, time)
        ds[v] = xr.DataArray(data, dims=["days", "time"])
    ds = ds.assign_coords(
        dayofyear=(["days", "time"], doy),
        hour=(["days", "time"], hour),
    )
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CREDIT ensemble WeatherBench CRPS/RMSE/ACC verification")
    parser.add_argument(
        "--forecast", default=DEFAULT_FORECAST, help="Root dir of ensemble NetCDFs (one subdir per init date)"
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cesm", default=DEFAULT_CESM_GLOB)
    parser.add_argument("--clim", default=DEFAULT_CLIM_PATH)
    parser.add_argument("--tag", default="ensemble", help="Tag appended to output filenames (e.g. 'wxformer_v2_ens')")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect init dirs and verify lead file availability
    # ------------------------------------------------------------------
    init_dirs = sorted(
        [
            os.path.join(args.forecast, d)
            for d in os.listdir(args.forecast)
            if os.path.isdir(os.path.join(args.forecast, d))
        ]
    )
    logger.info(f"{len(init_dirs)} init dirs found")

    init_to_meta = {}
    all_times = set()
    for init_dir in init_dirs:
        name = os.path.basename(init_dir)
        lead_files, ok = [], True
        for lh in LEAD_HOURS:
            m = glob.glob(os.path.join(init_dir, f"pred_{name}_{lh:03d}.nc"))
            if not m:
                ok = False
                break
            lead_files.append(m[0])
        if not ok:
            logger.warning(f"Missing lead files for {name}, skipping")
            continue
        t0 = pd.Timestamp(name.replace("T", " ").replace("Z", ""))
        vtimes = [np.datetime64(t0 + pd.Timedelta(hours=lh)) for lh in LEAD_HOURS]
        init_to_meta[init_dir] = (lead_files, vtimes)
        all_times.update(vtimes)

    # ------------------------------------------------------------------
    # Pre-load truth for all verification times at once
    # cesm_stage1 has T500/U500/V500/Q500/Z500/SP/t2m pre-computed — no interpolation
    # ------------------------------------------------------------------
    all_times_sorted = sorted(all_times)
    logger.info(f"Loading {len(all_times_sorted)} truth timesteps from cesm_stage1...")
    cesm_files = sorted(glob.glob(args.cesm))
    ds_cesm = xr.open_mfdataset(cesm_files, combine="by_coords", engine="zarr")
    truth_vars = ["Z500", "T500", "U500", "V500", "Q500", "SP", "t2m"]
    ds_truth_all = ds_cesm[truth_vars].sel(time=all_times_sorted).load()
    logger.info("cesm load complete")

    logger.info("Loading climatology...")
    ds_clim_full = xr.open_dataset(args.clim).load()
    logger.info("Climatology loaded")

    # latitude weights from truth grid
    w_np = np.cos(np.deg2rad(ds_truth_all.latitude.values))
    w_np = w_np / w_np.mean()
    w_xr = xr.DataArray(w_np, dims=["latitude"], coords={"latitude": ds_truth_all.latitude})

    # Climatology variable name + level index for each output variable
    clim_var_map = {
        "Z500": ("geopotential", CLIM_LEV_500),
        "T500": ("temperature", CLIM_LEV_500),
        "U500": ("u_component_of_wind", CLIM_LEV_500),
        "V500": ("v_component_of_wind", CLIM_LEV_500),
        "Q500": ("specific_humidity", CLIM_LEV_500),
        "SP": ("surface_pressure", None),
        "t2m": ("2m_temperature", None),
    }

    # Ensemble forecast variable + pressure selector for each output variable
    fc_var_map = {
        "Z500": ("Z_PRES", 500.0),
        "T500": ("T_PRES", 500.0),
        "U500": ("U_PRES", 500.0),
        "V500": ("V_PRES", 500.0),
        "Q500": ("Q_PRES", 500.0),
        "SP": ("SP", None),
        "t2m": ("t2m", None),
    }

    # ------------------------------------------------------------------
    # Main loop: one init at a time
    # ------------------------------------------------------------------
    results, n_inits = [], len(init_to_meta)

    for i, (init_dir, (lead_files, vtimes)) in enumerate(sorted(init_to_meta.items())):
        name = os.path.basename(init_dir)
        truth_times = np.array(vtimes)
        ds_truth = ds_truth_all.sel(time=truth_times)

        dt_idx = pd.DatetimeIndex(truth_times)
        doy_arr = dt_idx.day_of_year.values
        hour_arr = dt_idx.hour.values

        clim_list = [ds_clim_full.sel(dayofyear=int(d), hour=int(h)) for d, h in zip(doy_arr, hour_arr)]
        ds_clim = xr.concat(clim_list, dim="time")
        ds_clim["time"] = ds_truth.time

        crps_v = {v: np.zeros(len(LEAD_HOURS)) for v in fc_var_map}
        spread_v = {v: np.zeros(len(LEAD_HOURS)) for v in fc_var_map}
        rmse_v = {v: np.zeros(len(LEAD_HOURS)) for v in fc_var_map}
        acc_v = {v: np.zeros(len(LEAD_HOURS)) for v in fc_var_map}

        for li, lf in enumerate(lead_files):
            ds_lf = xr.open_dataset(lf, engine="netcdf4")

            for vname, (fc_var, pres) in fc_var_map.items():
                # forecast ensemble: (n_ens, lat, lon)
                if pres is not None:
                    pred_da = ds_lf[fc_var].sel(pressure=pres).squeeze("time")
                else:
                    pred_da = ds_lf[fc_var].squeeze("time")
                pred_np = pred_da.values.astype(np.float64)  # (n_ens, lat, lon)

                truth_da = ds_truth[vname].isel(time=li)
                truth_np = truth_da.values.astype(np.float64)

                clim_var, clim_lev = clim_var_map[vname]
                if clim_lev is not None:
                    clim_da = ds_clim[clim_var].isel(level=clim_lev, time=li)
                else:
                    clim_da = ds_clim[clim_var].isel(time=li)

                crps_val, spread_val = _crps_spread(pred_np, truth_np, w_np)
                crps_v[vname][li] = crps_val
                spread_v[vname][li] = spread_val

                # ensemble mean for RMSE + ACC
                pred_mean_da = xr.DataArray(
                    pred_np.mean(axis=0),
                    dims=["latitude", "longitude"],
                    coords={"latitude": truth_da.latitude, "longitude": truth_da.longitude},
                )
                rmse_v[vname][li] = float(_rmse(pred_mean_da, truth_da, w_xr).values)
                acc_v[vname][li] = float(_acc(pred_mean_da - clim_da, truth_da - clim_da, w_xr).values)

            ds_lf.close()

        results.append(
            {
                "crps": crps_v,
                "spread": spread_v,
                "rmse": rmse_v,
                "acc": acc_v,
                "dayofyear": doy_arr,
                "hour": hour_arr,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == n_inits:
            logger.info(
                f"[{i + 1}/{n_inits}] {name}  Z500_crps6h={crps_v['Z500'][0]:.1f}  Z500_acc6h={acc_v['Z500'][0]:.5f}"
            )

    logger.info(f"Processed {len(results)} inits")
    results.sort(key=lambda r: (r["dayofyear"][0], r["hour"][0]))

    # ------------------------------------------------------------------
    # Save outputs: combined + per-100-init batch files
    # ------------------------------------------------------------------
    tag = args.tag
    for kind in ("crps", "spread", "rmse", "acc"):
        ds_out = _build_dataset(results, kind)
        path = os.path.join(args.out, f"{kind.upper()}_006h_240h_{tag}.nc")
        ds_out.to_netcdf(path)
        logger.info(f"{kind.upper()} → {path}")

        n = len(results)
        for start in range(0, n, 100):
            end = min(start + 100, n)
            batch = results[start:end]
            _build_dataset(batch, kind).to_netcdf(
                os.path.join(
                    args.out,
                    f"combined_{kind}_{start:04d}_{end:04d}_006h_240h_{tag}.nc",
                )
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
