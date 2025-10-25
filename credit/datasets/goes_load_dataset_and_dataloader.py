import pandas as pd
import xarray as xr

from torch.utils.data import DataLoader

from credit.datasets.goes10km_dataset import GOES10kmDataset
from credit.samplers import DistributedMultiStepBatchSampler

import logging
logger = logging.getLogger(__name__)

def load_dataset(conf, rank, world_size, is_train=True):

    logger.info("loading a GOES 10km dataset")

    data_config = conf["data"]

    zarr_ds = xr.open_dataset(data_config["save_loc"])

    years = data_config["train_years"] if is_train else data_config["valid_years"]

    time_config = {
        "timestep": pd.Timedelta(data_config["lead_time_periods"], "h"),
        "num_forecast_steps": data_config["forecast_len"] + 1,
        "years": years,
    }

    return GOES10kmDataset(zarr_ds, time_config)


def load_dataloader(conf, train_dataset, rank, world_size, is_train=True):

    logger.info("loading a GOES 10km dataloader")

    sampling_modes = conf["data"]["sampling_modes"]

    seed = conf["seed"]
    training_type = "train" if is_train else "valid"
    batch_size = conf["trainer"][f"{training_type}_batch_size"]

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

    if not sampling_modes:
        sampling_modes = generate_default_sampling_modes(train_dataset)

    sampler = DistributedMultiStepBatchSampler(train_dataset,
                                                  batch_size,
                                                  sampling_modes,
                                                  num_replicas=world_size,
                                                  rank=rank,
                                                  seed=seed
                                                  )
    
    dataloader = DataLoader(train_dataset,
                            batch_sampler=sampler,
                            pin_memory=True,
                            persistent_workers=True if num_workers > 0 else False,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor if num_workers > 0 else None,
                            )
    logger.debug(f"dataloader: workers: {dataloader.num_workers} ,  prefetch factor: {dataloader.prefetch_factor}")
    
    return dataloader

def generate_default_sampling_modes(dataset):
    num_forecast_steps = dataset.num_forecast_steps

    return ["init"] + ["y"] * (num_forecast_steps - 1) + ["stop"]
