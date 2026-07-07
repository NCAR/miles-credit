import pandas as pd
import xarray as xr

from torch.utils.data import DataLoader
import torch

from credit.datasets.goes10km_dataset import GOES10kmDataset
from credit.datasets.era5 import ERA5Dataset
from credit.samplers import DistributedMultiStepBatchSampler

import logging

from credit.transform import load_era5_transforms

logger = logging.getLogger(__name__)

def load_era5_forcing(conf, start_datetime, end_datetime, transforms=None):
    if not transforms:
        logger.warning("NOT loading ERA5 transforms")

    time_config = {
        "timestep": pd.Timedelta("1h"),
        "num_forecast_steps": 1,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
    }
    return ERA5Dataset(conf, time_config, "ERA5", transforms=transforms)

def load_dataset(conf, rank, world_size, device, is_train=True):
    logger.info("loading a GOES 10km dataset")

    data_config = conf["data"]
    padding_conf = conf["model"].get("padding_conf", {})
    if padding_conf:
        padding = not padding_conf["activate"]
    else:
        padding = True

    # FIX: Do NOT open zarr here. Pass the raw path string so GOES10kmDataset
    # can open its own lazy per-worker handle (the refactored lazy-open design).
    zarr_path = data_config["save_loc"]

    mode = "train" if is_train else "valid"

    time_config = {
        "timestep": pd.Timedelta(data_config["timestep"]),
        "num_forecast_steps": data_config["forecast_len"] + 1,
        "start_datetime": pd.Timestamp(data_config[mode]["start_datetime"]),
        "end_datetime": pd.Timestamp(data_config[mode]["end_datetime"]),
    }

    if "ERA5" in data_config.get("source", {}).keys():
        logger.info("loading an era5 dataset for forcing")
        era5dataset = load_era5_forcing(
            conf,
            time_config["start_datetime"].values if hasattr(time_config["start_datetime"], "values") else time_config["start_datetime"],
            time_config["end_datetime"].values if hasattr(time_config["end_datetime"], "values") else time_config["end_datetime"] + pd.Timedelta("1D"),
            transforms=load_era5_transforms(conf, device)
        )
    else:
        era5dataset = None

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        if rank == 0: # make sure init times are setup first by rank 0, otherwise will try to concurrent write to same netcdf
            dataset = GOES10kmDataset(
                zarr_path,   # <-- path string, not Dataset
                data_config,
                time_config,
                padding=padding,
                era5dataset=era5dataset,
            )
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            dataset = GOES10kmDataset(
                zarr_path,   # <-- path string, not Dataset
                data_config,
                time_config,
                padding=padding,
                era5dataset=era5dataset,
            )
    else:
        dataset = GOES10kmDataset(
            zarr_path, data_config, time_config, padding=padding, era5dataset=era5dataset
        )

    # NEW: final sync — wait for every rank to finish creating its dataset
    # before any rank races ahead to model loading or DDP init
    torch.distributed.barrier()
    return dataset

def load_predict_dataset(conf, rank, world_size, rollout_init_times, device):
    logger.info("loading a GOES 10km dataset for rollout")

    data_config = conf["data"]
    padding_conf = conf["model"].get("padding_conf", {})
    if padding_conf:
        padding = not padding_conf["activate"]
    else:
        padding = True

    zarr_path = data_config["save_loc"]

    # FIX: Open a lightweight metadata-only dataset solely to read the time
    # coordinate bounds (t.min / t.max). Use chunks=None so no Dask pool is
    # created. Discard immediately after — GOES10kmDataset will do its own open.
    try:
        _bounds_ds = xr.open_zarr(zarr_path, consolidated=True, chunks=None)
        logger.info("Successfully loaded GOES Zarr bounds with consolidated metadata.")
    except Exception as e:
        logger.warning(f"Consolidated metadata failed for bounds. Falling back: {e}")
        _bounds_ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)

    t_min = _bounds_ds.t.min()
    t_max = _bounds_ds.t.max()
    del _bounds_ds   # discard — we only needed t.min/t.max

    # set default forecast steps
    num_forecast_steps = conf["predict"]["forecasts"].get("num_forecast_steps", None)
    num_forecast_hours = conf["predict"]["forecasts"].get("num_forecast_hours", None)

    if conf["predict"]["forecasts"].get("type", "debugger") in ["debugger", "standard"]:
        num_forecast_hours = conf["predict"]["forecasts"].get("num_forecast_hours", 24)

    timestep = pd.Timedelta(data_config["timestep"])

    # forecast hours overrides forecast_steps
    if num_forecast_hours:
        remainder = num_forecast_hours * 3600 % timestep.seconds
        num_forecast_steps = num_forecast_hours * 3600 // timestep.seconds
        if remainder:
            num_forecast_steps += 1

    time_tol = conf["predict"]["forecasts"].get("time_tol", (1, "d"))

    time_config = {
        "timestep": timestep,
        "num_forecast_steps": num_forecast_steps,
        "start_datetime": t_min,
        "end_datetime": t_max,
        "rollout_init_times": rollout_init_times,
        "time_tol": time_tol,
    }

    if "ERA5" in data_config.get("source", {}).keys():
        logger.info("loading an era5 dataset for forcing")
        era5dataset = load_era5_forcing(
            conf,
            time_config["start_datetime"].values if hasattr(time_config["start_datetime"], "values") else time_config["start_datetime"],
            time_config["end_datetime"].values if hasattr(time_config["end_datetime"], "values") else time_config["end_datetime"] + pd.Timedelta("1D"),
            transforms=load_era5_transforms(conf, device),
        )
    else:
        era5dataset = None

    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        if rank == 0: # make sure init times are setup first by rank 0, otherwise will try to concurrent write to same netcdf
            dataset = GOES10kmDataset(
                zarr_path,   # <-- path string, not Dataset
                data_config,
                time_config,
                padding=padding,
                era5dataset=era5dataset,
            )
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            dataset = GOES10kmDataset(
                zarr_path,   # <-- path string, not Dataset
                data_config,
                time_config,
                padding=padding,
                era5dataset=era5dataset,
            )
    else:
        dataset = GOES10kmDataset(
            zarr_path, data_config, time_config, padding=padding, era5dataset=era5dataset
        )

    return dataset

def load_dataloader(
    conf, train_dataset, rank, world_size, is_train=True, is_predict=False
):
    """
    is_predict will override is_train no matter what is_train is.
    It will grab num_workers from validation config as the rollout times should be the same
    """
    logger.info("loading a GOES 10km dataloader")

    if (
        conf["trainer"]["type"] == "goes10km-distributed-ensemble"
    ):  # want every device to see the same batch
        world_size = 1
        rank = 0

    if not is_predict:
        sampling_modes = conf["data"]["sampling_modes"]
    else:
        sampling_modes = generate_rollout_sampling_modes(
            train_dataset, conf["predict"].get("compute_metrics", False)
        )
    if not sampling_modes:
        sampling_modes = generate_default_sampling_modes(train_dataset)

    seed = conf["seed"]
    training_type = "train" if is_train else "valid"
    batch_size = conf["trainer"][f"{training_type}_batch_size"]
    logger.info(f"loading {training_type} dataloader with batch size {batch_size}")

    if is_predict:
        batch_size = conf["predict"]["batch_size"]
        is_train = False
        num_workers = conf["predict"].get("thread_workers", 0)
        prefetch_factor = conf["predict"].get("prefetch_factor", None)
    else:
        num_workers = (
            conf["trainer"]["thread_workers"]
            if is_train
            else conf["trainer"]["valid_thread_workers"]
        )
        prefetch_factor = conf["trainer"].get("prefetch_factor")

    if prefetch_factor is None:
        logger.warning(
            "prefetch_factor not found in config. Using default value of 4. "
            "Please specify prefetch_factor in the 'trainer' section of your config."
        )
        prefetch_factor = 4

    sampler = DistributedMultiStepBatchSampler(
        train_dataset,
        batch_size,
        sampling_modes,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=(not is_predict),
    )

    dataloader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    logger.info(
        f"dataloader workers: {dataloader.num_workers},  prefetch factor: {dataloader.prefetch_factor}"
    )

    return dataloader


def generate_default_sampling_modes(dataset):
    num_forecast_steps = dataset.num_forecast_steps

    return ["init"] + ["y"] * (num_forecast_steps - 1) + ["stop"]


def generate_rollout_sampling_modes(dataset, compute_metrics=False):
    if compute_metrics:
        return generate_default_sampling_modes(dataset)

    num_forecast_steps = dataset.num_forecast_steps

    return ["init"] + ["forcing"] * (num_forecast_steps - 1) + ["stop"]


def load_verification_dataset(conf):
    logger.info("loading a GOES 10km dataset for evaluation")

    data_config = conf["data"]
    padding_conf = conf["model"].get("padding_conf", {})
    if padding_conf:
        padding = not padding_conf["activate"] # opposite of what the model does
    else:
        padding = True

    # FIX: pass the path string directly
    zarr_path = data_config["save_loc"]

    dataset = GOES10kmDataset(
        zarr_path,   # <-- path string, not Dataset
        data_config,
        time_config={},
        padding=padding,
        evaluate=True,
        era5dataset=None,
    )
    return dataset