"""
WeatherBench2-style deterministic evaluation for CREDIT forecasts.

Two input modes:
  --csv   : aggregate pre-computed per-init metrics CSVs (fast, no GPU/ERA5 needed)
  --netcdf: compute metrics directly from forecast netCDFs against ERA5 zarr (full path)

Outputs a single CSV with RMSE, ACC, bias by variable and lead time, plus
latitude-region breakdowns (global / tropics / extratropics).

Usage
-----
# From existing per-init CSVs (fast path):
python eval_weatherbench.py --csv /path/to/metrics/ --out wb2_scores.csv

# From raw forecast netCDFs + ERA5:
python eval_weatherbench.py --netcdf /path/to/forecasts/ \\
    --era5 /glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_1deg/all_in_one/ERA5_mlevel_1deg_6h* \\
    --out wb2_scores.csv
"""

import argparse
import glob
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from credit.verification.deterministic import latitude_slices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WB2 standard variable definitions
# level_idx: index into CREDIT's 18-level pressure axis
# ---------------------------------------------------------------------------
# Default climatology (1990-2019, 6-hourly, from Ankur's CREDIT verification data)
DEFAULT_CLIM_PATH = "/glade/campaign/cisl/aiml/akn7/CREDIT_CESM/VERIF/ERA5_clim/ERA5_clim_1990_2019_6h_cesm_interp.nc"

# WB2 target levels.
# Each entry: label → (era5_var, credit_var, era5_level, credit_coord, credit_level)
#   era5_var     : variable name in ERA5 zarr (long names)
#   credit_var   : variable name in CREDIT forecast netCDF
#   era5_level   : value of the ERA5 model-level coord (CREDIT model-level index, e.g. 69≈500hPa)
#   credit_coord : 'pressure' for _PRES vars, 'level' for model-level vars
#   credit_level : coordinate value to select in the forecast (pressure in hPa or model-level idx)
# ERA5 mlevel zarr has no geopotential, so Z500 is omitted.
WB2_TARGET_LEVELS = {
    "T500": ("temperature", "T_PRES", 69.0, "pressure", 500.0),
    "T700": ("temperature", "T_PRES", 79.0, "pressure", 700.0),
    "T850": ("temperature", "T_PRES", 89.0, "pressure", 850.0),
    "U500": ("u_component_of_wind", "U", 69.0, "level", 69.0),
    "V500": ("v_component_of_wind", "V", 69.0, "level", 69.0),
    "U850": ("u_component_of_wind", "U", 89.0, "level", 89.0),
    "V850": ("v_component_of_wind", "V", 89.0, "level", 89.0),
    "Q500": ("specific_humidity", "Q", 69.0, "level", 69.0),
    "Q850": ("specific_humidity", "Q", 89.0, "level", 89.0),
}

WB2_SURFACE_VARS = ["t2m", "SP", "U500", "V500"]

# ERA5 variable name mapping (zarr → CREDIT short name used in CSVs)
# CREDIT ERA5 mlevel zarr files use the same short names as CREDIT output,
# so upper-air variables are identity-mapped.
ERA5_VAR_MAP = {
    # upper-air (model-level ERA5 zarr uses CREDIT short names directly)
    "Z": "Z",
    "T": "T",
    "U": "U",
    "V": "V",
    "Q": "Q",
    # surface / diagnostic
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "temperature": "T",
    "specific_humidity": "Q",
    "VAR_2T": "t2m",
    "SP": "SP",
    "VAR_10U": "u10",
    "VAR_10V": "v10",
}


# ---------------------------------------------------------------------------
# CSV fast path
# ---------------------------------------------------------------------------


def load_csv_metrics(csv_dir):
    """Load all per-init CSV files from csv_dir, return concatenated DataFrame."""
    files = [
        f
        for f in glob.glob(os.path.join(csv_dir, "*.csv"))
        if "average" not in os.path.basename(f) and "ensemble" not in os.path.basename(f)
    ]
    if not files:
        raise FileNotFoundError(f"No per-init CSV files found in {csv_dir}")
    logger.info(f"Loading {len(files)} init date CSVs from {csv_dir}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def csv_to_wb2_scores(df, lead_time_hours=6):
    """
    Aggregate per-init CSV metrics into WB2-style scores by lead time.

    Parameters
    ----------
    df : pd.DataFrame
        Concatenated per-init metrics (from load_csv_metrics).
    lead_time_hours : int
        Hours per forecast step (default 6).

    Returns
    -------
    pd.DataFrame
        Columns: lead_time_hours, variable, rmse, acc, mae, n_inits
    """
    # identify variable columns
    var_pattern = re.compile(r"^(rmse|acc|mae|bias|std)_(.+)$")
    vars_found = set()
    for col in df.columns:
        m = var_pattern.match(col)
        if m:
            vars_found.add(m.group(2))

    # aggregate by forecast_step
    records = []
    grouped = df.groupby("forecast_step")
    for step, grp in grouped:
        lead_h = step * lead_time_hours
        n = len(grp)
        row = {"lead_time_hours": lead_h, "forecast_step": step, "n_inits": n}
        for var in vars_found:
            for metric in ["rmse", "mae", "std"]:
                col = f"{metric}_{var}"
                if col in grp.columns:
                    row[col] = grp[col].mean()
            # acc_* columns in per-init CSVs are Pearson correlations, not true
            # WB2 anomaly ACC — store as pearson_r_* to avoid confusion
            acc_col = f"acc_{var}"
            if acc_col in grp.columns:
                row[f"pearson_r_{var}"] = grp[acc_col].mean()
        records.append(row)

    result = pd.DataFrame(records).sort_values("lead_time_hours").reset_index(drop=True)
    return result


def load_climatology(clim_path=None):
    """
    Load ERA5 climatology for ACC computation.

    Returns an xr.Dataset with dims (hour, dayofyear, [level,] latitude, longitude).
    Default: Ankur's 1990-2019 6-hourly climatology on campaign storage.
    """
    path = clim_path or DEFAULT_CLIM_PATH
    logger.info(f"Loading climatology from {path}")
    return xr.open_dataset(path)


def select_clim(ds_clim, valid_times, variable, level=None):
    """
    Select climatology values matching a set of valid forecast times.

    Parameters
    ----------
    ds_clim : xr.Dataset
        Climatology with dims (hour, dayofyear, ...).
    valid_times : array-like of np.datetime64
        Forecast valid times to match.
    variable : str
        Variable name in ds_clim.
    level : float, optional
        Pressure level to select (for 3D variables).

    Returns
    -------
    xr.DataArray
        Climatology values with a 'time' dimension matching valid_times.
    """
    times = pd.DatetimeIndex(valid_times)
    clim_slices = []
    for t in times:
        doy = t.dayofyear
        hour = t.hour
        da = ds_clim[variable].sel(dayofyear=doy, hour=hour)
        if level is not None and "level" in da.dims:
            da = da.sel(level=level, method="nearest")
        clim_slices.append(da)
    da_clim = xr.concat(clim_slices, dim="time")
    da_clim["time"] = valid_times
    return da_clim


def print_wb2_summary(scores_df):
    """Print WB2-style scorecard for key variables and lead times."""
    key_vars = ["T500", "T850", "t2m", "U500", "V500", "Q850"]
    key_days = [1, 3, 5, 7, 10]
    key_steps = {d: d * 24 // 6 for d in key_days}  # assumes 6h steps

    print("\n" + "=" * 70)
    print("WeatherBench2-style scorecard (mean over all init dates)")
    print("=" * 70)

    for var in key_vars:
        rmse_col = f"rmse_{var}"
        acc_col = f"acc_{var}"
        if rmse_col not in scores_df.columns:
            continue
        print(f"\n  {var}")
        print(f"  {'Day':>4}  {'RMSE':>10}  {'ACC':>8}")
        print(f"  {'-' * 4}  {'-' * 10}  {'-' * 8}")
        for day, step in key_steps.items():
            row = scores_df[scores_df["forecast_step"] == step]
            if row.empty:
                continue
            rmse = row[rmse_col].values[0]
            acc = row[acc_col].values[0] if acc_col in row.columns else float("nan")
            print(f"  {day:>4}  {rmse:>10.2f}  {acc:>8.4f}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# NetCDF full path
# ---------------------------------------------------------------------------


def open_era5_year(era5_glob, year):
    """Open ERA5 zarr/nc for the given year."""
    pattern = str(era5_glob).replace("*", f"*{year}*")
    files = sorted(glob.glob(pattern))
    if not files:
        # fallback: open all and select year
        files = sorted(glob.glob(str(era5_glob)))
    if not files:
        raise FileNotFoundError(f"No ERA5 files matching {era5_glob}")
    ds = xr.open_mfdataset(
        files, engine="zarr" if files[0].endswith(".zarr") else "netcdf4", combine="by_coords", chunks={"time": 4}
    )
    if year is not None:
        ds = ds.sel(time=str(year))
    return ds


def _score_step_worker(args):
    """
    Top-level worker for ProcessPoolExecutor (must be picklable).

    Scores all forecasts at a single lead-time step against ERA5.

    Parameters
    ----------
    args : tuple
        (step, files, era5_glob, clim_path)

    Returns
    -------
    dict  Row of WB2 scores for this step, or None on failure.
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    step, files, era5_glob, clim_path = args
    lead_h = step  # filename suffix is already lead-time hours

    # Load climatology inside worker (no shared state across processes)
    ds_clim = None
    try:
        ds_clim = load_climatology(clim_path)
    except Exception as e:
        _log.warning(f"Step {step}: could not load climatology ({e}); ACC skipped")

    # Per-worker ERA5 cache (keyed by year; at most 2 years per step in practice)
    era5_cache = {}

    pred_list, true_list = [], []
    for f in files:
        try:
            ds_pred = xr.open_dataset(f)
        except Exception as e:
            _log.warning(f"Step {step}: failed to open {f}: {e}")
            continue
        if "ensemble" in ds_pred.dims:
            ds_pred = ds_pred.mean(dim="ensemble")
        valid_time = ds_pred.time.values[0]
        year = pd.Timestamp(valid_time).year

        if year not in era5_cache:
            era5_cache[year] = open_era5_year(era5_glob, year)
            if len(era5_cache) > 2:
                del era5_cache[min(era5_cache.keys())]

        ds_true = era5_cache[year].sel(time=valid_time, method="nearest")
        pred_list.append(ds_pred)
        true_list.append(ds_true)

    if not pred_list:
        return None

    ds_pred_all = xr.concat(pred_list, dim="time")
    ds_true_all = xr.concat(true_list, dim="time")

    row = {"lead_time_hours": lead_h, "forecast_step": step, "n_inits": len(pred_list)}
    valid_times = ds_pred_all.time.values

    # upper-air
    for label, (era5_var, credit_var, era5_level, credit_coord, credit_level) in WB2_TARGET_LEVELS.items():
        if era5_var not in ds_true_all or credit_var not in ds_pred_all:
            continue
        da_pred = ds_pred_all[credit_var].sel({credit_coord: credit_level}, method="nearest")
        da_true = ds_true_all[era5_var].sel(level=era5_level, method="nearest")
        if da_true.latitude.shape != da_pred.latitude.shape or not np.allclose(
            da_true.latitude.values, da_pred.latitude.values, atol=0.1
        ):
            da_true = da_true.interp(latitude=da_pred.latitude, longitude=da_pred.longitude, method="linear")
        da_clim_var = None
        if ds_clim is not None and era5_var in ds_clim:
            da_clim_var = select_clim(ds_clim, valid_times, era5_var, level=era5_level)
            da_clim_var = da_clim_var.interp(latitude=da_pred.latitude, longitude=da_pred.longitude)
        _add_scores(row, da_pred, da_true, label, da_clim=da_clim_var)

    # surface
    surface_map = {"t2m": ("VAR_2T", "2m_temperature"), "SP": ("SP", "SP")}
    for credit_var, (era5_var, clim_var) in surface_map.items():
        if credit_var not in ds_pred_all or era5_var not in ds_true_all:
            continue
        da_pred = ds_pred_all[credit_var]
        da_true = ds_true_all[era5_var]
        if da_true.latitude.shape != da_pred.latitude.shape or not np.allclose(
            da_true.latitude.values, da_pred.latitude.values, atol=0.1
        ):
            da_true = da_true.interp(latitude=da_pred.latitude, longitude=da_pred.longitude, method="linear")
        da_clim_var = None
        if ds_clim is not None and clim_var in ds_clim:
            da_clim_var = select_clim(ds_clim, valid_times, clim_var)
            da_clim_var = da_clim_var.interp(latitude=da_pred.latitude, longitude=da_pred.longitude)
        _add_scores(row, da_pred, da_true, credit_var, da_clim=da_clim_var)

    _log.info(f"Step {step} ({lead_h}h): done ({len(pred_list)} inits)")
    return row


def compute_netcdf_scores(forecast_dir, era5_glob, clim_path=None, lead_time_hours=6, max_inits=None, workers=None):
    """
    Compute WB2 deterministic scores from forecast netCDFs against ERA5.

    Each lead-time step is scored in parallel via ProcessPoolExecutor.

    Expects forecast_dir structured as:
        forecast_dir/{init_datetime}/pred_{init}_{step:03d}.nc

    Parameters
    ----------
    forecast_dir : str or Path
        Root directory of forecast netCDFs.
    era5_glob : str
        Glob pattern for ERA5 zarr files.
    lead_time_hours : int
        Hours per step (kept for signature compatibility; step suffix is authoritative).
    max_inits : int, optional
        Limit number of init dates (for testing).
    workers : int, optional
        Number of parallel workers.  Defaults to os.cpu_count().

    Returns
    -------
    pd.DataFrame
        WB2 scores by lead time and variable.
    """
    forecast_dir = Path(forecast_dir)
    init_dirs = sorted([d for d in forecast_dir.iterdir() if d.is_dir()])
    if max_inits:
        init_dirs = init_dirs[:max_inits]
    logger.info(f"Found {len(init_dirs)} init dates in {forecast_dir}")

    # collect forecast files grouped by step
    step_files = {}
    for init_dir in init_dirs:
        for f in sorted(init_dir.glob("pred_*.nc")):
            parts = f.stem.split("_")
            step = int(parts[-1])
            step_files.setdefault(step, []).append(f)

    n_steps = len(step_files)
    n_workers = workers or os.cpu_count() or 1
    logger.info(f"Scoring {n_steps} steps with {n_workers} workers")

    work_items = [(step, files, era5_glob, clim_path) for step, files in sorted(step_files.items())]

    all_records = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_score_step_worker, item): item[0] for item in work_items}
        for fut in as_completed(futures):
            step = futures[fut]
            try:
                row = fut.result()
                if row is not None:
                    all_records.append(row)
            except Exception as exc:
                logger.error(f"Step {step} raised: {exc}")

    result = pd.DataFrame(all_records).sort_values("lead_time_hours").reset_index(drop=True)
    return result


def _add_scores(row, da_pred, da_true, label, da_clim=None):
    """
    Compute RMSE, bias, and ACC per region and add to row dict in-place.

    ACC uses anomaly correlation if da_clim is provided (true WB2 ACC).
    Without climatology, ACC is Pearson correlation (reported as acc_pearson_*).
    """
    w_lat = np.cos(np.deg2rad(da_pred.latitude))
    sq_err = (da_pred - da_true) ** 2
    diff = da_pred - da_true

    for region, s in latitude_slices.items():
        w_r = w_lat.sel(latitude=s)
        sq_r = sq_err.sel(latitude=s)
        diff_r = diff.sel(latitude=s)
        w_sum = float(w_r.sum())

        rmse_val = float(np.sqrt((sq_r.mean(dim="longitude") * w_r).sum(dim="latitude") / w_sum).mean(dim="time"))
        bias_val = float((diff_r.mean(dim="longitude") * w_r).sum(dim="latitude").mean(dim="time") / w_sum)

        row[f"rmse_{label}_{region}"] = rmse_val
        row[f"bias_{label}_{region}"] = bias_val

        # true ACC (anomaly correlation)
        if da_clim is not None:
            pred_anom = da_pred.sel(latitude=s) - da_clim.sel(latitude=s)
            true_anom = da_true.sel(latitude=s) - da_clim.sel(latitude=s)
            num = (w_r * pred_anom * true_anom).sum(dim=["latitude", "longitude"])
            denom = np.sqrt(
                (w_r * pred_anom**2).sum(dim=["latitude", "longitude"])
                * (w_r * true_anom**2).sum(dim=["latitude", "longitude"])
            )
            row[f"acc_{label}_{region}"] = float((num / (denom + 1e-12)).mean(dim="time"))

    # global columns (matches existing CSV format)
    w_sum_global = float(w_lat.sum())
    row[f"rmse_{label}"] = float(
        np.sqrt((sq_err.mean(dim="longitude") * w_lat).sum(dim="latitude") / w_sum_global).mean(dim="time")
    )
    row[f"bias_{label}"] = float(
        (diff.mean(dim="longitude") * w_lat).sum(dim="latitude").mean(dim="time") / w_sum_global
    )

    if da_clim is not None:
        pred_anom = da_pred - da_clim
        true_anom = da_true - da_clim
        num = (w_lat * pred_anom * true_anom).sum(dim=["latitude", "longitude"])
        denom = np.sqrt(
            (w_lat * pred_anom**2).sum(dim=["latitude", "longitude"])
            * (w_lat * true_anom**2).sum(dim=["latitude", "longitude"])
        )
        row[f"acc_{label}"] = float((num / (denom + 1e-12)).mean(dim="time"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="WeatherBench2-style evaluation for CREDIT forecasts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=str, help="Directory of per-init metrics CSVs (fast path)")
    group.add_argument("--netcdf", type=str, help="Directory of forecast netCDFs (full path)")

    parser.add_argument(
        "--era5",
        type=str,
        default="/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_1deg/all_in_one/ERA5_mlevel_1deg_6h*",
        help="Glob pattern for ERA5 zarr files (required for --netcdf mode)",
    )
    parser.add_argument(
        "--clim",
        type=str,
        default=None,
        help=f"Path to ERA5 climatology netCDF for true ACC. Defaults to {DEFAULT_CLIM_PATH}",
    )
    parser.add_argument("--out", type=str, default="wb2_scores.csv", help="Output CSV path")
    parser.add_argument("--lead-time-hours", type=int, default=6, help="Hours per forecast step")
    parser.add_argument("--max-inits", type=int, default=None, help="Limit init dates (for testing)")
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="PLOT_DIR",
        help="If set, also generate WB2 scorecard figures in this directory",
    )
    parser.add_argument(
        "--label", type=str, default="CREDIT", help="Model label used in plot legends (default: CREDIT)"
    )
    parser.add_argument("--no-refs", action="store_true", help="Omit IFS/Pangu/GraphCast reference lines in plots")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for --netcdf mode (default: os.cpu_count())",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    if args.csv:
        df_raw = load_csv_metrics(args.csv)
        scores = csv_to_wb2_scores(df_raw, lead_time_hours=args.lead_time_hours)
    else:
        if not args.era5:
            parser.error("--era5 is required when using --netcdf mode")
        scores = compute_netcdf_scores(
            args.netcdf,
            args.era5,
            clim_path=args.clim,
            lead_time_hours=args.lead_time_hours,
            max_inits=args.max_inits,
            workers=args.workers,
        )

    scores.to_csv(args.out, index=False)
    logger.info(f"Saved WB2 scores to {args.out}")

    print_wb2_summary(scores)

    if args.plot:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from plot_weatherbench import plot_all

        plot_all(args.out, args.plot, label=args.label, show_refs=not args.no_refs)


if __name__ == "__main__":
    main()
