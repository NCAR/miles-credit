"""Load and align gen2 CAMulator forecast output against CESM truth data.

Prediction output (credit/output_gen2.py::ForecastWriter) already uses bare
variable names and latitude/longitude/level dims, matching what
shared_utils.py's plotting helpers expect. The CESM truth zarr under
data.source.CESM uses CESM-native names (lat/lon/lev) and a noleap calendar,
so it needs renaming and exact time alignment before it can share code with
the prediction dataset.
"""

import glob
import os

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from credit.datasets.gen_2.grid_utils import find_coord_pair


def _years_from_times(valid_times) -> list[int]:
    """Distinct calendar years spanned by *valid_times*.

    Prediction/truth time coords decode to cftime.datetime objects (noleap
    calendar, see camulator_metadata.yaml), which pd.to_datetime() cannot
    convert -- it only understands standard-calendar datetimes. cftime
    objects already carry `.year` directly.
    """
    times = np.asarray(valid_times).ravel()
    if len(times) and isinstance(times[0], cftime.datetime):
        return sorted({t.year for t in times})
    return sorted(pd.DatetimeIndex(pd.to_datetime(times)).year.unique())


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def bare_output_variables(conf: dict) -> list[str]:
    """Bare variable names declared in inference.output.variables.

    Output entries are named e.g. "CESM/prognostic/3d/U"; the writer flattens
    these down to the trailing component ("U") on disk.
    """
    names = [entry["name"] for entry in conf["inference"]["output"]["variables"]]
    return [n.rsplit("/", 1)[-1] for n in names]


def _fmt_init(init_time) -> str:
    return pd.Timestamp(init_time).strftime("%Y%m%d_%H%MZ")


def find_prediction_dir(conf: dict, init_time: str | None = None) -> str:
    """Resolve save_forecast/<init_time_dir>/ for one forecast's output files."""
    save_dir = os.path.expandvars(conf["inference"]["save_forecast"])
    if init_time is None:
        init_time = conf["inference"]["single_forecast"]["start_datetime"]
    pred_dir = os.path.join(save_dir, _fmt_init(init_time))
    if not os.path.isdir(pred_dir):
        available = sorted(os.listdir(save_dir)) if os.path.isdir(save_dir) else "(save_dir does not exist)"
        raise FileNotFoundError(f"No forecast output directory at {pred_dir}. Available init times: {available}")
    return pred_dir


def load_prediction_dataset(conf: dict, init_time: str | None = None, variables: list[str] | None = None) -> xr.Dataset:
    """Open one forecast's output files (all group_by chunks) as a single Dataset."""
    pred_dir = find_prediction_dir(conf, init_time)
    files = sorted(glob.glob(os.path.join(pred_dir, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found under {pred_dir}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    if variables is not None:
        ds = ds[[v for v in variables if v in ds.data_vars]]
    return ds


def _variable_groups(conf: dict, source: str = "CESM") -> dict:
    return conf["data"]["source"][source]["variables"]


def _find_variable_group(conf: dict, var: str, source: str = "CESM") -> tuple[str, dict, bool]:
    """Return (group_name, field_cfg, is_3d) for the group containing *var*."""
    for group_name, field_cfg in _variable_groups(conf, source).items():
        if var in (field_cfg.get("vars_3D") or []):
            return group_name, field_cfg, True
        if var in (field_cfg.get("vars_2D") or []):
            return group_name, field_cfg, False
    raise KeyError(f"Variable {var!r} not found in any data.source.{source}.variables group")


def load_truth_dataset(conf: dict, valid_times, variables: list[str], source: str = "CESM") -> xr.Dataset:
    """Open truth CESM zarr(s) spanning *valid_times*, renamed to match prediction naming."""
    source_cfg = conf["data"]["source"][source]
    level_coord = source_cfg.get("level_coord", "level")
    years = _years_from_times(valid_times)

    path_to_vars: dict[str, list[str]] = {}
    for var in variables:
        _, field_cfg, _ = _find_variable_group(conf, var, source)
        path_to_vars.setdefault(field_cfg["path"], []).append(var)

    per_path = []
    for path_template, vars_in_path in path_to_vars.items():
        yearly = [xr.open_zarr(path_template.replace("%Y", f"{year:04d}"))[vars_in_path] for year in years]
        per_path.append(xr.concat(yearly, dim="time") if len(yearly) > 1 else yearly[0])

    truth = xr.merge(per_path, join="exact")

    lon, lat, lon_name, lat_name = find_coord_pair(truth)
    rename_map = {lon_name: "longitude", lat_name: "latitude"}
    if level_coord in truth.dims:
        rename_map[level_coord] = "level"
    truth = truth.rename(rename_map)

    # Stash the real CESM hybrid-sigma level values on a separate aux coordinate
    # before align_truth_to_prediction overwrites "level" itself with prediction's
    # plain 0..31 index (needed so .sel(level=idx) means the same thing on both
    # datasets -- see align_truth_to_prediction). "level_value" is for display
    # only (dropdown labels, axis ticks); xarray carries it along automatically
    # through any .sel()/.isel() on the "level" dimension.
    if "level" in truth.coords:
        truth = truth.assign_coords(level_value=("level", truth["level"].values))

    return truth


def align_truth_to_prediction(truth_ds: xr.Dataset, pred_ds: xr.Dataset, time_coord: str = "time") -> xr.Dataset:
    """Select truth at exactly the prediction's valid times, and harmonize the level coordinate.

    No `method="nearest"` on time: both are 6-hourly noleap-calendar steps sharing the
    same origin, so an exact match should always exist -- a KeyError here means the
    calendars/time grids actually disagree and should surface loudly.

    The level coordinate is overwritten with the prediction's rather than matched by
    value: truth's `level` holds real CESM hybrid-sigma pressure values (e.g. 3.6..992.6
    hPa) while the prediction's is a plain 0..31 index -- both describe the same 32 model
    levels in the same order, just labeled differently. Without this, `.sel(level=idx)`
    (used for level selection elsewhere) would mean different things on the two datasets.
    """
    truth_ds = truth_ds.sel({time_coord: pred_ds[time_coord].values})
    if "level" in truth_ds.dims and "level" in pred_ds.dims:
        truth_ds = truth_ds.assign_coords(level=pred_ds["level"].values)
    return truth_ds


def load_truth_and_prediction(
    conf: dict, variables: list[str] | None = None, init_time: str | None = None, source: str = "CESM"
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load and align (truth, prediction) for the given config in one call."""
    if variables is None:
        variables = bare_output_variables(conf)
    pred = load_prediction_dataset(conf, init_time=init_time, variables=variables)
    truth = load_truth_dataset(conf, pred["time"].values, variables=variables, source=source)
    truth = align_truth_to_prediction(truth, pred)
    return truth, pred


def load_ic_snapshot(
    conf: dict, variables: list[str], init_time: str | None = None, source: str = "CESM"
) -> xr.Dataset:
    """Load the single input timestep a forecast would start from, before any rollout has run.

    Reads straight from the source zarr(s) (same path resolution as load_truth_dataset) and
    selects the nearest timestep to *init_time* -- useful for displaying e.g. SST so a user can
    click a sensible location on it, without waiting on a model run.
    """
    if init_time is None:
        init_time = conf["inference"]["single_forecast"]["start_datetime"]
    ds = load_truth_dataset(conf, [init_time], variables=variables, source=source)

    # CFTimeIndex.sel(method="nearest") compares raw values rather than coercing a string label
    # (unlike exact/label indexing, which does parse ISO strings) -- so it errors on a plain str
    # against cftime.datetime entries. Build a cftime object of the index's own subclass instead.
    time_vals = ds["time"].values
    target = pd.Timestamp(init_time)
    if len(time_vals) and isinstance(time_vals[0], cftime.datetime):
        cf_type = type(time_vals[0])
        target = cf_type(target.year, target.month, target.day, target.hour, target.minute, target.second)
    return ds.sel(time=target, method="nearest")


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "../config/gen_2/camulator/camulator_cesm_tutorial_casper.yml"
    conf = load_config(config_path)
    variables = bare_output_variables(conf)
    print(f"Config: {config_path}")
    print(f"Output variables ({len(variables)}): {variables}")

    pred_dir = find_prediction_dir(conf)
    print(f"Prediction dir: {pred_dir}")

    truth, pred = load_truth_and_prediction(conf, variables=variables)
    print("\n--- Prediction dataset ---")
    print(pred)
    print("\n--- Truth dataset (aligned) ---")
    print(truth)

    for var in variables:
        if var in pred.data_vars and var in truth.data_vars:
            print(f"{var}: pred {pred[var].dims}{pred[var].shape}  truth {truth[var].dims}{truth[var].shape}")
