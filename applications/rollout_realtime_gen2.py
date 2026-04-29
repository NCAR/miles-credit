"""
rollout_realtime_gen2.py
----------------------
Operational/realtime forecast rollout for CREDIT v2 models.

Designed to be run with a single command — no pre-editing of config files:

    python rollout_realtime_gen2.py \\
        -c config/wxformer_025deg_6hr_gen2.yml \\
        --init-time 2024-01-15T00 \\
        --steps 40 \\
        --save-dir /path/to/output

All forecast-specific parameters (init time, steps, save dir) come from
CLI args and override whatever is in the config.  The config only needs to
describe the model, the data source paths, and the normalization files.

The script uses ERA5Dataset directly — the same dataset class used for
training — so there is no separate "predict dataset" to maintain.

Output: one NetCDF file per forecast step saved to
    <save_dir>/<YYYY-MM-DDTHH>Z/pred_<YYYY-MM-DDTHH>Z_<FHR:03d>.nc
"""

import os
import yaml
import logging
import warnings
import traceback
from argparse import ArgumentParser
from datetime import datetime, timedelta
from multiprocessing.shared_memory import SharedMemory

import pandas as pd
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from credit.datasets.era5 import ERA5Dataset
from credit.preblock import build_preblocks, apply_preblocks
from credit.models import load_model
from credit.seed import seed_everything
from credit.distributed import get_rank_info, setup, distributed_model_wrapper
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
from credit.nwp import build_GFS_init

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Config helpers (shared with rollout_to_netcdf_gen2)
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
    conf["data"]["diagnostic_variables"] = diag.get("vars_2D", []) if diag else []
    conf["data"]["level_ids"] = src.get("levels", list(range(conf["model"]["levels"])))
    if "scaler_type" not in conf["data"]:
        conf["data"]["scaler_type"] = "std_new"


def _inject_tracer_inds(conf):
    """Compute tracer_inds for TracerFixer from v2 variable layout."""
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
    thres_map = dict(zip(tracer_conf.get("tracer_name", []), tracer_conf.get("tracer_thres", [])))
    inds, thres = [], []
    for i, vn in enumerate(output_vars):
        if vn in thres_map:
            inds.append(i)
            thres.append(thres_map[vn])
    conf["model"]["post_conf"]["tracer_fixer"]["tracer_inds"] = inds
    conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"] = thres
    conf["model"]["post_conf"]["tracer_fixer"]["denorm"] = False


def _build_output_denorm(conf, device, dtype=torch.float32):
    """Return (mean, std) of shape (1, C_out, 1, 1, 1) for inverse-normalizing y_pred."""
    data_conf = conf["data"]
    src = data_conf["source"]["ERA5"]
    levels = src["levels"]
    level_coord = src["level_coord"]
    n_levels = len(levels)
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}

    norm_args = conf.get("preblocks", {}).get("norm", {}).get("args", data_conf)
    mean_ds = xr.open_dataset(norm_args.get("mean_path", data_conf.get("mean_path"))).load()
    std_ds = xr.open_dataset(norm_args.get("std_path", data_conf.get("std_path"))).load()

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
    for groups in [("vars_3D", True), ("vars_2D", False)]:
        for vname in prog.get(groups[0], []):
            m, s = _stats(vname, groups[1])
            means.append(m)
            stds.append(s)
    for groups in [("vars_3D", True), ("vars_2D", False)]:
        for vname in diag.get(groups[0], []):
            m, s = _stats(vname, groups[1])
            means.append(m)
            stds.append(s)

    mean_ds.close()
    std_ds.close()
    return (
        torch.cat(means).view(1, -1, 1, 1, 1).to(device),
        torch.cat(stds).view(1, -1, 1, 1, 1).to(device),
    )


def _sample_to_batch(sample):
    """Add batch dim and wrap ERA5Dataset sample for preblock input."""
    return {"era5": {"input": {k: v.unsqueeze(0) for k, v in sample["input"].items()}, "metadata": sample["metadata"]}}


# ---------------------------------------------------------------------------
# Async save worker (uses SharedMemory to avoid pickling large tensors)
# ---------------------------------------------------------------------------


def _save_worker(shm_name, arr_shape, arr_dtype, init_str, step, fhr_per_step, lat, lon, meta_data, conf):
    try:
        shm = SharedMemory(shm_name)
        y_np = np.ndarray(arr_shape, dtype=arr_dtype, buffer=shm.buf).copy()
        shm.unlink()

        utc_dt = datetime.strptime(init_str, "%Y-%m-%dT%HZ") + timedelta(hours=fhr_per_step * step)
        y_t = torch.from_numpy(y_np)
        darray_upper_air, darray_single_level = make_xarray(y_t, utc_dt, lat, lon, conf)
        save_netcdf_increment(
            darray_upper_air,
            darray_single_level,
            init_str,
            fhr_per_step * step,
            meta_data,
            conf,
        )
        print(f"  step={step:3d}  valid={utc_dt.strftime('%Y-%m-%d %HZ')}  fhr={fhr_per_step * step:3d}h")
    except Exception:
        print(traceback.format_exc())


# ---------------------------------------------------------------------------
# GFS initial condition fetch
# ---------------------------------------------------------------------------


def run_gfs_init(conf, init_time: pd.Timestamp, n_procs: int = 1) -> str:
    """Download GFS analysis for init_time, regrid to CREDIT grid, save as zarr.

    Patches conf['data']['source']['ERA5']['variables']['prognostic']['path']
    to the generated zarr so ERA5Dataset loads the GFS IC at step 0.
    Returns the zarr path.
    """
    rt_conf = conf["predict"]["realtime"]
    ic_path = rt_conf["initial_condition_path"]
    os.makedirs(ic_path, exist_ok=True)

    zarr_path = os.path.join(ic_path, f"gfs_init_{init_time.strftime('%Y%m%d_%H00')}.zarr")

    if not os.path.exists(zarr_path):
        base = os.path.abspath(os.path.dirname(__file__))
        parent = os.path.basename(os.path.abspath(os.path.join(base, os.pardir)))
        metadata_path = os.path.join(base, os.pardir, "metadata" if parent == "credit" else "credit/metadata")

        lev_info_file = os.path.join(metadata_path, "ERA5_Lev_Info.nc")
        credit_grid = xr.open_dataset(lev_info_file)
        model_level_csv = os.path.join(metadata_path, "L137_model_level_indices.csv")
        model_level_indices = pd.read_csv(model_level_csv)["model_level_indices"].values

        prog = conf["data"]["source"]["ERA5"]["variables"].get("prognostic", {})
        variables = prog.get("vars_3D", []) + prog.get("vars_2D", [])

        now = pd.Timestamp.utcnow().tz_localize(None)
        init_naive = init_time.tz_localize(None) if init_time.tzinfo is not None else init_time
        gdas_base = (
            "gs://global-forecast-system/"
            if now - init_naive >= pd.Timedelta(days=10)
            else "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        )

        logger.info(f"Building GFS init for {init_time} from {gdas_base}")
        gfs_ds = build_GFS_init(
            output_grid=credit_grid,
            date=init_time,
            variables=variables,
            model_level_file=lev_info_file,
            model_levels=model_level_indices,
            gdas_base_path=gdas_base,
            n_procs=n_procs,
        )
        logger.info(f"Saving GFS init zarr to {zarr_path}")
        gfs_ds.to_zarr(zarr_path, mode="w")
    else:
        logger.info(f"GFS init zarr already exists: {zarr_path}")

    conf["data"]["source"]["ERA5"]["variables"]["prognostic"]["path"] = zarr_path
    return zarr_path


# ---------------------------------------------------------------------------
# Core rollout
# ---------------------------------------------------------------------------


def run_forecast(conf, init_time: pd.Timestamp, n_steps: int, save_dir: str, pool, rank=0, world_size=1):
    """
    Run a single autoregressive forecast from `init_time` for `n_steps` steps.

    Args:
        conf:        Full configuration dict (v2 schema, flat keys already injected).
        init_time:   Forecast initialization timestamp.
        n_steps:     Number of autoregressive steps to run.
        save_dir:    Directory for output NetCDF files.
        pool:        multiprocessing.Pool for async saves.
        rank/world_size: For DDP; single-GPU callers use (0, 1).
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    dt = pd.Timedelta(conf["data"]["timestep"])
    fhr_per_step = int(dt.total_seconds() / 3600)

    # ---- v2 channel bookkeeping ----
    src = conf["data"]["source"]["ERA5"]
    v = src["variables"]
    prog = v.get("prognostic") or {}
    dyn = v.get("dynamic_forcing") or {}
    n_levels = len(src["levels"])
    # ERA5Dataset input insertion order: [dynfrc | static | prog]
    n_prog = len(prog.get("vars_3D", [])) * n_levels + len(prog.get("vars_2D", []))
    n_dyn = len(dyn.get("vars_2D", []))
    n_static = len(v.get("static", {}).get("vars_2D", []))
    static_dim_size = n_dyn + n_static

    # ---- Preblocks ----
    preblocks = build_preblocks(conf.get("preblocks", {}))

    # ---- Inverse normalizer ----
    denorm_mean, denorm_std = _build_output_denorm(conf, device)

    # ---- Lat/lon for output ----
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"]).load()
    lat, lon = latlons.latitude.values, latlons.longitude.values

    meta_data = load_metadata(conf)

    # ---- Postblocks ----
    post_conf = conf["model"].get("post_conf", {})
    flag_mass = flag_water = flag_energy = False
    if post_conf.get("activate", False):
        if post_conf.get("global_mass_fixer", {}).get("activate_outside_model", False):
            flag_mass = True
            opt_mass = GlobalMassFixer(post_conf)
        if post_conf.get("global_water_fixer", {}).get("activate_outside_model", False):
            flag_water = True
            opt_water = GlobalWaterFixer(post_conf)
        if post_conf.get("global_energy_fixer", {}).get("activate_outside_model", False):
            flag_energy = True
            opt_energy = GlobalEnergyFixer(post_conf)

    # ---- Model ----
    _inject_tracer_inds(conf)
    mode = conf["predict"]["mode"]
    if mode == "none":
        model = load_model(conf, load_weights=True).to(device)
    elif mode == "ddp":
        model = load_model(conf).to(device)
        model = distributed_model_wrapper(conf, model, device)
        ckpt = torch.load(
            os.path.join(os.path.expandvars(conf["save_loc"]), "checkpoint.pt"),
            map_location=device,
        )
        load_state_dict_error_handler(model.module.load_state_dict(ckpt["model_state_dict"], strict=False))
    elif mode == "fsdp":
        model = load_model(conf, load_weights=True).to(device)
        model = distributed_model_wrapper(conf, model, device)
        model = load_model_state(conf, model, device)
    else:
        raise ValueError(f"Unsupported predict mode: {mode}")
    model.eval()

    # ---- Dataset: build a config that covers the requested init time ----
    # ERA5Dataset uses start/end datetimes only for __len__; we access by timestamp directly.
    dataset_conf = dict(conf["data"])
    dataset_conf["start_datetime"] = str(init_time.date())
    dataset_conf["end_datetime"] = str((init_time + n_steps * dt).date())
    dataset_conf["forecast_len"] = 1
    dataset = ERA5Dataset(dataset_conf, return_target=False)

    init_str = init_time.strftime("%Y-%m-%dT%HZ")
    conf["predict"]["save_forecast"] = save_dir

    logger.info(f"Forecast init: {init_str}  steps: {n_steps}  fhr_max: {n_steps * fhr_per_step}h")
    logger.info(f"Saving to: {save_dir}/{init_str}/")

    # ---- Load full initial state ----
    sample_full = dataset[(init_time, 0)]
    batch_full = _sample_to_batch(sample_full)
    x, _ = apply_preblocks(preblocks, batch_full)
    x = x.to(device).float()  # (1, C_in, 1, H, W)

    results = []
    x_init = None

    with torch.no_grad():
        for step in tqdm(range(1, n_steps + 1), desc=f"Rollout {init_str}"):
            y_pred = model(x)

            if flag_mass:
                if step == 1:
                    x_init = x.clone()
                y_pred = opt_mass({"y_pred": y_pred, "x": x_init})["y_pred"]
            if flag_water:
                y_pred = opt_water({"y_pred": y_pred, "x": x})["y_pred"]
            if flag_energy:
                y_pred = opt_energy({"y_pred": y_pred, "x": x})["y_pred"]

            # Inverse-normalize → physical space; squeeze T dim for output.py
            y_phys = (y_pred * denorm_std + denorm_mean).squeeze(2).cpu().numpy()  # (1, C, H, W)

            # Async save via SharedMemory
            shm = SharedMemory(create=True, size=y_phys.nbytes)
            shm_arr = np.ndarray(y_phys.shape, dtype=y_phys.dtype, buffer=shm.buf)
            shm_arr[:] = y_phys
            result = pool.apply_async(
                _save_worker,
                (shm.name, y_phys.shape, y_phys.dtype, init_str, step, fhr_per_step, lat, lon, meta_data, conf),
            )
            results.append(result)

            # Update x for next step (if not last)
            if step < n_steps:
                t_next = init_time + step * dt
                sample_frc = dataset[(t_next, 1)]  # only dynamic_forcing
                batch_frc = _sample_to_batch(sample_frc)
                x_frc, _ = apply_preblocks(preblocks, batch_frc)
                x_frc = x_frc.to(device).float()

                # ERA5Dataset insertion order: [dynfrc | static | prog]
                x[:, :n_dyn, ...] = x_frc
                x[:, static_dim_size:, ...] = torch.from_numpy(y_phys[:, :n_prog, np.newaxis]).to(device)
                # x[:, n_dyn:static_dim_size, ...] — static unchanged

    for r in results:
        r.get()

    if conf["predict"]["mode"] in ["ddp", "fsdp"]:
        torch.distributed.barrier()

    logger.info(f"Done: {init_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(
        description="CREDIT v2 realtime/operational forecast rollout.",
        epilog="""
Examples:
  Single GPU, one init time:
    python rollout_realtime_gen2.py -c config/wxformer_025deg_6hr_gen2.yml --init-time 2024-01-15T00

  With explicit step count and save dir:
    python rollout_realtime_gen2.py -c config.yml --init-time 2024-01-15T00 --steps 60 --save-dir /scratch/$USER/fcst

  Multi-GPU DDP (via torchrun):
    torchrun --standalone --nnodes=1 --nproc-per-node=4 rollout_realtime_gen2.py -c config.yml --init-time 2024-01-15T00
        """,
    )
    parser.add_argument(
        "-c", "--config", dest="model_config", type=str, required=True, help="Path to v2 model configuration YAML."
    )
    parser.add_argument(
        "--init-time",
        type=str,
        required=True,
        help="Forecast initialization time. ISO format: YYYY-MM-DDTHH (e.g. 2024-01-15T00)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of autoregressive steps. Default: conf['predict']['days'] × steps_per_day.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Output directory. Default: conf['predict']['save_forecast']."
    )
    parser.add_argument(
        "-m", "--mode", type=str, default=None, help="Override predict mode: none | ddp | fsdp. Default: from config."
    )
    parser.add_argument(
        "-p", "--procs", dest="num_cpus", type=int, default=4, help="CPU workers for async NetCDF saves."
    )

    args = parser.parse_args()

    # ---- Logging ----
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)

    # ---- Load config ----
    with open(args.model_config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    assert "source" in conf["data"], (
        "rollout_realtime_gen2.py requires the v2 nested data schema. For v1 configs use rollout_realtime.py."
    )

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    _inject_flat_schema(conf)

    # ---- CLI overrides ----
    if args.mode in ["none", "ddp", "fsdp"]:
        conf["predict"]["mode"] = args.mode
    if "mode" not in conf.get("predict", {}):
        conf.setdefault("predict", {})["mode"] = "none"

    # ---- Parse init time ----
    init_time_str = args.init_time.replace("T", " ").replace("Z", "")
    # Accept YYYY-MM-DDTHH or YYYY-MM-DD HH
    for fmt in ("%Y-%m-%d %H", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            init_time = pd.Timestamp(datetime.strptime(init_time_str, fmt))
            break
        except ValueError:
            continue
    else:
        init_time = pd.Timestamp(args.init_time)

    # ---- Number of steps ----
    if args.steps is not None:
        n_steps = args.steps
    else:
        fhr_per_step = int(pd.Timedelta(conf["data"]["timestep"]).total_seconds() / 3600)
        days = conf.get("predict", {}).get("forecasts", {}).get("days", 10)
        n_steps = days * (24 // fhr_per_step)

    # ---- Save directory ----
    save_dir = args.save_dir or conf.get("predict", {}).get("save_forecast")
    assert save_dir, "Specify --save-dir or set conf['predict']['save_forecast']."
    save_dir = os.path.expandvars(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if conf.get("predict", {}).get("realtime", {}).get("use_gfs_init", False):
        logger.info("use_gfs_init=True: fetching GFS analysis as initial condition")
        run_gfs_init(conf, init_time, n_procs=args.num_cpus)

    seed_everything(conf["seed"])
    local_rank, world_rank, world_size = get_rank_info(conf["predict"]["mode"])

    import multiprocessing as mp

    with mp.Pool(args.num_cpus) as pool:
        if conf["predict"]["mode"] in ["fsdp", "ddp"]:
            setup(world_rank, world_size, conf["predict"]["mode"])
        run_forecast(conf, init_time, n_steps, save_dir, pool, rank=world_rank, world_size=world_size)
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
