"""
rollout_to_netcdf_v2.py
-----------------------
Rollout entry point for the v2 data schema (conf["data"]["source"]).

Key differences from rollout_to_netcdf.py (v1):
  - No credit_main_parser / predict_data_check (those are v1-only)
  - Uses ERA5Dataset + ERA5Normalizer + ConcatPreblock for data loading
  - Inverse-normalization built from mean/std NC files (per-channel)
  - Injects flat data schema keys for output.py compatibility
  - forecast_len semantics: N steps at lead_time_periods hours per step

Usage:
    torchrun --standalone --nnodes=1 --nproc-per-node=<N_GPUS> \\
        applications/rollout_to_netcdf_v2.py -c config/wxformer_025deg_6hr_v2.yml

Or submit via PBS (see scripts/casper_v2.sh with SCRIPT=applications/rollout_to_netcdf_v2.py).
"""

import os
import sys
import yaml
import logging
import warnings
import traceback
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from datetime import datetime, timedelta

import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

from credit.datasets.era5 import ERA5Dataset
from credit.preblock import ERA5Normalizer, ConcatPreblock, apply_preblocks
from credit.models import load_model
from credit.seed import seed_everything
from credit.distributed import get_rank_info, setup, distributed_model_wrapper
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
from credit.forecast import load_forecasts
from credit.pbs import launch_script, launch_script_mpi

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _inject_flat_schema(conf):
    """Inject v1-style flat keys into conf['data'] so output.py utilities work."""
    if "variables" in conf["data"]:
        return
    src = conf["data"]["source"]["ERA5"]
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}
    conf["data"]["variables"] = prog.get("vars_3D", [])
    conf["data"]["surface_variables"] = prog.get("vars_2D", [])
    # diagnostic_variables is the list of names shown in xarray output
    conf["data"]["diagnostic_variables"] = diag.get("vars_2D", []) if diag else []
    # level_ids: actual pressure/model-level values for xarray coordinate
    conf["data"]["level_ids"] = src.get("levels", list(range(conf["model"]["levels"])))
    # scaler_type needed by save_netcdf_increment
    if "scaler_type" not in conf["data"]:
        conf["data"]["scaler_type"] = "std_new"


def _inject_tracer_inds(conf):
    """Compute tracer_inds for TracerFixer from v2 variable layout (same as train_v2.py)."""
    tracer_conf = conf.get("model", {}).get("post_conf", {}).get("tracer_fixer", {})
    if not tracer_conf.get("activate", False) or "tracer_inds" in tracer_conf:
        return
    src = conf["data"]["source"]["ERA5"]
    n_levels = len(src.get("levels", []))
    v = src["variables"]
    vars_3d = (v.get("prognostic") or {}).get("vars_3D", [])
    vars_2d = (v.get("prognostic") or {}).get("vars_2D", [])
    diag_2d = (v.get("diagnostic") or {}).get("vars_2D", [])
    output_vars = [vn for vn in vars_3d for _ in range(n_levels)] + vars_2d + diag_2d
    tracer_names = tracer_conf.get("tracer_name", [])
    tracer_thres_cfg = tracer_conf.get("tracer_thres", [])
    thres_map = dict(zip(tracer_names, tracer_thres_cfg))
    tracer_inds, tracer_thres = [], []
    for i, vn in enumerate(output_vars):
        if vn in thres_map:
            tracer_inds.append(i)
            tracer_thres.append(thres_map[vn])
    conf["model"]["post_conf"]["tracer_fixer"]["tracer_inds"] = tracer_inds
    conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"] = tracer_thres
    conf["model"]["post_conf"]["tracer_fixer"]["denorm"] = False


def _build_output_denorm(conf, device, dtype=torch.float32):
    """Build (mean, std) tensors of shape (1, C_out, 1, 1, 1) for inverse-normalizing y_pred.

    Channel order matches ConcatPreblock _TARGET_FIELD_ORDER:
        prognostic/3d (each var × n_levels), prognostic/2d, diagnostic/2d
    """
    data_conf = conf["data"]
    src = data_conf["source"]["ERA5"]
    levels = src["levels"]
    level_coord = src["level_coord"]
    n_levels = len(levels)
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}

    mean_ds = xr.open_dataset(data_conf["mean_path"]).load()
    std_ds = xr.open_dataset(data_conf["std_path"]).load()

    def _stats(varname, is_3d):
        if varname not in mean_ds:
            n = n_levels if is_3d else 1
            return torch.zeros(n, dtype=dtype), torch.ones(n, dtype=dtype)
        if is_3d:
            m = torch.tensor(mean_ds[varname].sel({level_coord: levels}).values, dtype=dtype)
            s = torch.tensor(std_ds[varname].sel({level_coord: levels}).values, dtype=dtype)
        else:
            m = torch.tensor([float(mean_ds[varname].values)], dtype=dtype)
            s = torch.tensor([float(std_ds[varname].values)], dtype=dtype)
        return m, s

    means, stds = [], []
    for vname in prog.get("vars_3D", []):
        m, s = _stats(vname, True)
        means.append(m)
        stds.append(s)
    for vname in prog.get("vars_2D", []):
        m, s = _stats(vname, False)
        means.append(m)
        stds.append(s)
    for vname in diag.get("vars_3D", []):
        m, s = _stats(vname, True)
        means.append(m)
        stds.append(s)
    for vname in diag.get("vars_2D", []):
        m, s = _stats(vname, False)
        means.append(m)
        stds.append(s)

    mean_ds.close()
    std_ds.close()

    mean_all = torch.cat(means).view(1, -1, 1, 1, 1).to(device)
    std_all = torch.cat(stds).view(1, -1, 1, 1, 1).to(device)
    return mean_all, std_all


def _sample_to_batch(sample):
    """Wrap a single ERA5Dataset sample (no batch dim) into preblock-compatible dict."""
    return {"era5": {"input": {k: v.unsqueeze(0) for k, v in sample["input"].items()}, "metadata": sample["metadata"]}}


# ---------------------------------------------------------------------------
# Async save worker
# ---------------------------------------------------------------------------


def _save_worker(y_pred_np, init_datetime_str, forecast_step, lead_time_periods, lat, lon, meta_data, conf):
    """Called via pool.apply_async — converts numpy array to xarray and saves."""
    try:
        utc_dt = datetime.strptime(init_datetime_str, "%Y-%m-%dT%HZ") + timedelta(
            hours=lead_time_periods * forecast_step
        )
        y_pred_t = torch.from_numpy(y_pred_np)
        darray_upper_air, darray_single_level = make_xarray(y_pred_t, utc_dt, lat, lon, conf)
        save_netcdf_increment(
            darray_upper_air,
            darray_single_level,
            init_datetime_str,
            lead_time_periods * forecast_step,
            meta_data,
            conf,
        )
    except Exception:
        print(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------


def predict(rank, world_size, conf, p):
    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["predict"]["mode"])

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    seed_everything(conf["seed"])

    # ---- Data schema helpers ----
    src = conf["data"]["source"]["ERA5"]
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}
    dyn = v.get("dynamic_forcing") or {}
    static_v = v.get("static") or {}
    n_levels = len(src["levels"])

    # channel counts
    n_prog = len(prog.get("vars_3D", [])) * n_levels + len(prog.get("vars_2D", []))
    n_dyn = len(dyn.get("vars_2D", []))
    varnum_diag = len(diag.get("vars_2D", [])) + len(diag.get("vars_3D", [])) * n_levels

    lead_time_periods = conf["data"]["lead_time_periods"]
    forecast_steps = conf["predict"].get("forecast_steps", conf["predict"].get("days", 1) * (24 // lead_time_periods))

    # ---- Preblocks ----
    preblocks_dict = {}
    if conf["data"].get("scaler_type") == "std_new":
        preblocks_dict["norm"] = ERA5Normalizer(conf)
    preblocks_dict["concat"] = ConcatPreblock()
    preblocks = nn.ModuleDict(preblocks_dict)

    # ---- Inverse normalization ----
    denorm_mean, denorm_std = _build_output_denorm(conf, device)

    # ---- Lat/lon for output ----
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"]).load()
    lat = latlons.latitude.values
    lon = latlons.longitude.values

    meta_data = load_metadata(conf)

    # ---- Postblocks (conservation fixers) ----
    post_conf = conf["model"]["post_conf"]
    flag_mass_conserve = flag_water_conserve = flag_energy_conserve = False
    if post_conf.get("activate", False):
        if post_conf.get("global_mass_fixer", {}).get("activate_outside_model", False):
            flag_mass_conserve = True
            opt_mass = GlobalMassFixer(post_conf)
        if post_conf.get("global_water_fixer", {}).get("activate_outside_model", False):
            flag_water_conserve = True
            opt_water = GlobalWaterFixer(post_conf)
        if post_conf.get("global_energy_fixer", {}).get("activate_outside_model", False):
            flag_energy_conserve = True
            opt_energy = GlobalEnergyFixer(post_conf)

    # ---- Model ----
    _inject_tracer_inds(conf)
    model = _load_model(conf, device)
    model.eval()

    # ---- Dataset (predict uses full data range; we select specific inits below) ----
    dataset_conf = dict(conf["data"])
    dataset_conf["forecast_len"] = 1
    dataset = ERA5Dataset(dataset_conf, return_target=False)
    dt = pd.Timedelta(conf["data"]["timestep"])

    # ---- Forecast init times: distribute across ranks ----
    all_forecasts = conf["predict"]["forecasts"]
    forecasts = [f for i, f in enumerate(all_forecasts) if i % world_size == rank]

    # ---- Rollout ----
    with torch.no_grad():
        for init_str in forecasts:
            t0 = pd.Timestamp(init_str)

            # Step 0: load full initial state
            sample_full = dataset[(t0, 0)]
            batch_full = _sample_to_batch(sample_full)
            batch_full = apply_preblocks(preblocks, batch_full)
            x = batch_full["x"].to(device).float()  # (1, C_in, 1, H, W)

            x_init = None
            results = []

            for step in range(1, forecast_steps + 1):
                y_pred = model(x)

                if flag_mass_conserve:
                    if step == 1:
                        x_init = x.clone()
                    y_pred = opt_mass({"y_pred": y_pred, "x": x_init})["y_pred"]
                if flag_water_conserve:
                    y_pred = opt_water({"y_pred": y_pred, "x": x})["y_pred"]
                if flag_energy_conserve:
                    y_pred = opt_energy({"y_pred": y_pred, "x": x})["y_pred"]

                # Inverse-normalize → physical space; squeeze T dim for output.py
                y_pred_phys = (y_pred * denorm_std + denorm_mean).squeeze(2)  # (1, C_out, H, W)

                result = p.apply_async(
                    _save_worker,
                    (
                        y_pred_phys.cpu().numpy(),
                        init_str,
                        step,
                        lead_time_periods,
                        lat,
                        lon,
                        meta_data,
                        conf,
                    ),
                )
                results.append(result)

                # Update x for next step
                if step < forecast_steps:
                    t_next = t0 + step * dt
                    sample_frc = dataset[(t_next, 1)]  # loads only dynamic_forcing
                    batch_frc = _sample_to_batch(sample_frc)
                    batch_frc = apply_preblocks(preblocks, batch_frc)
                    x_frc = batch_frc["x"].to(device).float()  # (1, n_dyn, 1, H, W)

                    # Replace prognostic with y_pred, forcing with new data, static unchanged
                    x[:, :n_prog, ...] = y_pred[:, :n_prog, ...].detach()
                    x[:, n_prog : n_prog + n_dyn, ...] = x_frc
                    # x[:, n_prog+n_dyn:, ...] stays (static)

            for result in results:
                result.get()

            if conf["predict"]["mode"] in ["ddp", "fsdp"]:
                torch.distributed.barrier()

    return 1


def _load_model(conf, device):
    mode = conf["predict"]["mode"]
    if mode == "none":
        return load_model(conf, load_weights=True).to(device)
    elif mode == "ddp":
        model = load_model(conf).to(device)
        model = distributed_model_wrapper(conf, model, device)
        ckpt_path = os.path.join(os.path.expandvars(conf["save_loc"]), "checkpoint.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        load_msg = model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        load_state_dict_error_handler(load_msg)
        return model
    elif mode == "fsdp":
        model = load_model(conf, load_weights=True).to(device)
        model = distributed_model_wrapper(conf, model, device)
        return load_model_state(conf, model, device)
    raise ValueError(f"Unsupported predict mode: {mode}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(description="Rollout AI-NWP forecasts (v2 data schema)")
    parser.add_argument("-c", dest="model_config", type=str, required=True, help="Path to v2 model configuration YAML.")
    parser.add_argument("-l", dest="launch", type=int, default=0, help="Submit to PBS if 1.")
    parser.add_argument("-m", "--mode", type=str, default="none", help="Override predict mode: none | ddp | fsdp")
    parser.add_argument("-cpus", "--num_cpus", type=int, default=4, help="Number of CPU workers for async save pool.")
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)

    with open(args.model_config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    assert "source" in conf["data"], (
        "rollout_to_netcdf_v2.py requires the v2 nested data schema (conf['data']['source']). "
        "For v1 configs use rollout_to_netcdf.py."
    )

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    _inject_flat_schema(conf)

    assert "save_forecast" in conf["predict"], "conf['predict']['save_forecast'] is required."
    os.makedirs(os.path.expandvars(conf["predict"]["save_forecast"]), exist_ok=True)

    if args.mode in ["none", "ddp", "fsdp"]:
        conf["predict"]["mode"] = args.mode

    if args.launch:
        script_path = Path(__file__).absolute()
        if conf.get("pbs", {}).get("queue") == "casper":
            launch_script(args.model_config, script_path)
        else:
            launch_script_mpi(args.model_config, script_path)
        sys.exit()

    conf["predict"]["forecasts"] = load_forecasts(conf)
    seed_everything(conf["seed"])

    local_rank, world_rank, world_size = get_rank_info(conf["predict"]["mode"])

    with mp.Pool(args.num_cpus) as p:
        if conf["predict"]["mode"] in ["fsdp", "ddp"]:
            predict(world_rank, world_size, conf, p=p)
        else:
            predict(0, 1, conf, p=p)
        p.close()
        p.join()


if __name__ == "__main__":
    main()
