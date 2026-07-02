"""
HRES deterministic WeatherBench-style ACC + RMSE verification.

Compares IFS HRES forecasts (from HRES.zarr) vs ERA5 cesm_stage1 analysis,
evaluated at the CREDIT 1.25° (192×288) Gaussian grid via xesmf bilinear regridding.

Output format matches wxformer_wb2_verif.py:
  ACC_006h_240h_{tag}.nc  : dims (days, time=40), vars Z500/T500/U500/V500/Q500/SP/t2m
  RMSE_006h_240h_{tag}.nc : same

Usage
-----
python applications/hres_wb2_verif.py \\
    [--hres /glade/derecho/scratch/schreck/HRES.zarr] \\
    [--out /glade/work/schreck/CREDIT_verif/hres] \\
    [--year 2022] [--init-hour 0]
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

logger = logging.getLogger(__name__)

DEFAULT_HRES = "/glade/derecho/scratch/schreck/HRES.zarr"
DEFAULT_CESM_GLOB = (
    "/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/all_in_one/ERA5_mlevel_cesm_6h_lev16_*.zarr"
)
DEFAULT_CLIM_PATH = "/glade/campaign/cisl/aiml/ksha/CREDIT_CESM/VERIF/ERA5_clim/ERA5_clim_1990_2019_6h_cesm_interp.nc"
DEFAULT_OUT = "/glade/work/schreck/CREDIT_verif/hres"

LEAD_HOURS = list(range(6, 241, 6))  # 40 leads, 6h..240h
CLIM_LEV_500 = 7  # 500 hPa index in ERA5_clim (0-indexed)


# ---------------------------------------------------------------------------
# Spatial averaging — identical to wxformer_wb2_verif.py
# ---------------------------------------------------------------------------
def _sp_avg(da, w_lat):
    return da.weighted(w_lat).mean(["latitude", "longitude"], skipna=False)


def _make_w_lat_xr(lats):
    w = np.cos(np.deg2rad(lats))
    return xr.DataArray(w / w.mean(), dims=["latitude"], coords={"latitude": lats})


def _acc(pred_anom, true_anom, w_lat):
    top = _sp_avg(pred_anom * true_anom, w_lat)
    bottom = np.sqrt(_sp_avg(pred_anom**2, w_lat) * _sp_avg(true_anom**2, w_lat))
    return top / (bottom + 1e-12)


def _rmse(pred, truth, w_lat):
    return np.sqrt(_sp_avg((pred - truth) ** 2, w_lat))


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
    parser = argparse.ArgumentParser(description="HRES deterministic WeatherBench ACC/RMSE verification")
    parser.add_argument("--hres", default=DEFAULT_HRES, help="Path to HRES.zarr")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cesm", default=DEFAULT_CESM_GLOB)
    parser.add_argument("--clim", default=DEFAULT_CLIM_PATH)
    parser.add_argument("--tag", default="hres", help="Tag appended to output filenames (e.g. 'hres_2022')")
    parser.add_argument("--year", type=int, default=None, help="Restrict to a single calendar year (e.g. 2022)")
    parser.add_argument(
        "--init-hour", type=int, default=None, help="Restrict to specific init hour: 0 (00Z) or 12 (12Z)"
    )
    parser.add_argument("--weights-dir", default="/tmp", help="Cache dir for xesmf regridder weight files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------
    # Open data sources lazily
    # ------------------------------------------------------------------
    logger.info("Opening HRES.zarr (lazy)...")
    ds_hres = xr.open_zarr(args.hres)

    logger.info("Opening cesm_stage1 zarr (lazy)...")
    cesm_files = sorted(glob.glob(args.cesm))
    ds_cesm = xr.open_mfdataset(cesm_files, combine="by_coords", engine="zarr")
    truth_vars = ["Z500", "T500", "U500", "V500", "Q500", "SP", "t2m"]

    logger.info("Loading climatology...")
    ds_clim_full = xr.open_dataset(args.clim).load()
    logger.info("Climatology loaded")

    # ------------------------------------------------------------------
    # Filter HRES init times
    # ------------------------------------------------------------------
    hres_times = pd.DatetimeIndex(ds_hres.time.values)
    mask = np.ones(len(hres_times), dtype=bool)
    if args.year is not None:
        mask &= hres_times.year == args.year
    if args.init_hour is not None:
        mask &= hres_times.hour == args.init_hour
    hres_times_sel = hres_times[mask]
    logger.info(f"{len(hres_times_sel)} HRES init times selected")

    if len(hres_times_sel) == 0:
        raise ValueError("No HRES init times matched filter criteria.")

    # ------------------------------------------------------------------
    # Build xesmf regridder: HRES (0.25°, 721×1440) → cesm_stage1 (192×288)
    # Rename latitude/longitude → lat/lon for xesmf
    # ------------------------------------------------------------------
    logger.info("Building xesmf regridder (HRES → cesm_stage1)...")
    ds_hres_grid = xr.Dataset(
        {
            "lat": ("lat", ds_hres.latitude.values.astype(np.float64)),
            "lon": ("lon", ds_hres.longitude.values.astype(np.float64)),
        }
    )
    cesm_lats = ds_cesm.latitude.values
    cesm_lons = ds_cesm.longitude.values
    ds_out_grid = xr.Dataset(
        {
            "lat": ("lat", cesm_lats.astype(np.float64)),
            "lon": ("lon", cesm_lons.astype(np.float64)),
        }
    )
    weights_file = os.path.join(
        args.weights_dir,
        f"xesmf_hres{len(ds_hres.latitude)}x{len(ds_hres.longitude)}"
        f"_to_credit{len(cesm_lats)}x{len(cesm_lons)}_bilinear.nc",
    )
    regridder = xe.Regridder(
        ds_hres_grid,
        ds_out_grid,
        method="bilinear",
        periodic=True,
        filename=weights_file,
        reuse_weights=os.path.exists(weights_file),
    )
    logger.info(f"Regridder ready (weights: {weights_file})")

    w_xr = _make_w_lat_xr(cesm_lats)

    # HRES variable name + pressure level (hPa int, or None for 2D vars)
    hres_var_map = {
        "Z500": ("geopotential", 500),
        "T500": ("temperature", 500),
        "U500": ("u_component_of_wind", 500),
        "V500": ("v_component_of_wind", 500),
        "Q500": ("specific_humidity", 500),
        "SP": ("surface_pressure", None),
        "t2m": ("2m_temperature", None),
    }

    # Climatology variable name + level index for ACC computation
    clim_var_map = {
        "Z500": ("geopotential", CLIM_LEV_500),
        "T500": ("temperature", CLIM_LEV_500),
        "U500": ("u_component_of_wind", CLIM_LEV_500),
        "V500": ("v_component_of_wind", CLIM_LEV_500),
        "Q500": ("specific_humidity", CLIM_LEV_500),
        "SP": ("surface_pressure", None),
        "t2m": ("2m_temperature", None),
    }

    # Pre-build timedeltas for HRES lead times
    hres_tds = [pd.Timedelta(hours=lh) for lh in LEAD_HOURS]

    # ------------------------------------------------------------------
    # Main loop: one HRES init at a time
    # ------------------------------------------------------------------
    results, n_inits = [], len(hres_times_sel)

    for i, t0_pd in enumerate(hres_times_sel):
        t0_np = np.datetime64(t0_pd)
        vtimes = np.array([t0_np + np.timedelta64(lh, "h") for lh in LEAD_HOURS])

        # load 40 truth timesteps; skip if any are outside cesm_stage1 coverage
        try:
            ds_truth = ds_cesm[truth_vars].sel(time=vtimes).load()
        except (KeyError, ValueError):
            logger.debug(f"  {t0_pd}: truth coverage incomplete, skipping")
            continue

        dt_idx = pd.DatetimeIndex(vtimes)
        doy_arr = dt_idx.day_of_year.values
        hour_arr = dt_idx.hour.values

        clim_list = [ds_clim_full.sel(dayofyear=int(d), hour=int(h)) for d, h in zip(doy_arr, hour_arr)]
        ds_clim = xr.concat(clim_list, dim="time")
        ds_clim["time"] = ds_truth.time

        rmse_v = {v: np.zeros(len(LEAD_HOURS)) for v in hres_var_map}
        acc_v = {v: np.zeros(len(LEAD_HOURS)) for v in hres_var_map}

        for li, (lh, td) in enumerate(zip(LEAD_HOURS, hres_tds)):
            hres_slice = ds_hres.sel(
                time=t0_pd,
                prediction_timedelta=np.timedelta64(int(td.total_seconds() * 1e9), "ns"),
            )

            for vname, (hres_var, hres_lev) in hres_var_map.items():
                if hres_lev is not None:
                    fc_raw = hres_slice[hres_var].sel(level=hres_lev).load()
                else:
                    fc_raw = hres_slice[hres_var].load()

                # regrid HRES (0.25°) → cesm_stage1 (1.25°)
                ds_in = xr.Dataset(
                    {
                        "data": xr.DataArray(
                            fc_raw.values.astype(np.float64),
                            dims=["lat", "lon"],
                            coords={
                                "lat": ds_hres.latitude.values.astype(np.float64),
                                "lon": ds_hres.longitude.values.astype(np.float64),
                            },
                        )
                    }
                )
                fc_out = xr.DataArray(
                    regridder(ds_in)["data"].values,
                    dims=["latitude", "longitude"],
                    coords={"latitude": cesm_lats, "longitude": cesm_lons},
                )

                truth_da = ds_truth[vname].isel(time=li)
                clim_var, clim_lev = clim_var_map[vname]
                if clim_lev is not None:
                    clim_da = ds_clim[clim_var].isel(level=clim_lev, time=li)
                else:
                    clim_da = ds_clim[clim_var].isel(time=li)

                rmse_v[vname][li] = float(_rmse(fc_out, truth_da, w_xr).values)
                acc_v[vname][li] = float(_acc(fc_out - clim_da, truth_da - clim_da, w_xr).values)

        results.append(
            {
                "rmse": rmse_v,
                "acc": acc_v,
                "dayofyear": doy_arr,
                "hour": hour_arr,
            }
        )

        if (i + 1) % 50 == 0 or (i + 1) == n_inits:
            logger.info(
                f"[{i + 1}/{n_inits}] {t0_pd.strftime('%Y-%m-%dT%HZ')}  "
                f"Z500_rmse6h={rmse_v['Z500'][0]:.1f}  "
                f"Z500_acc6h={acc_v['Z500'][0]:.5f}"
            )

    logger.info(f"Processed {len(results)} inits")
    if not results:
        logger.warning("No results to save.")
        return

    results.sort(key=lambda r: (r["dayofyear"][0], r["hour"][0]))

    # ------------------------------------------------------------------
    # Save outputs: combined + per-100-init batch files
    # ------------------------------------------------------------------
    tag = args.tag
    for kind in ("rmse", "acc"):
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
