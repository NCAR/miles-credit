from credit.datasets.era5_multistep import ERA5_and_Forcing_MultiStep
from credit.datasets.era5_singlestep import ERA5_and_Forcing_SingleStep
from credit.datasets.era5_multistep_batcher import (
    ERA5_MultiStep_Batcher,
    MultiprocessingBatcher,
    MultiprocessingBatcherPrefetch
)
from credit.data import ERA5_and_Forcing_Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from credit.transforms import load_transforms
from credit.parser import credit_main_parser, training_data_check
from credit.datasets import setup_data_loading, set_globals
import logging
import re


class DataLoaderLite:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for sample in self.dataset:  # Directly iterate over the dataset
            yield sample

    def __len__(self):
        if hasattr(self.dataset, "batches_per_epoch"):
            return self.dataset.batches_per_epoch()  # Use the dataset's method if available
        else:
            return len(self.dataset)  # Otherwise, fall back to the dataset's length


def collate_fn(batch):
    # Only used with ERA5_MultiStep_Batcher
    # Prevents time and batch dimension from getting flipped
    return batch[0]


def load_dataset(conf, rank=0, world_size=1, is_train=True):
    """
    Load the dataset based on the configuration.

    Args:
        conf (dict): Configuration dictionary containing dataset and training parameters.
        rank (int, optional): Rank of the current process. Default is 0.
        world_size (int, optional): Number of processes participating in the job. Default is 1.
        is_train (bool, optional): Flag indicating whether the dataset is for training or validation.
                                  Default is True.

    Returns:
        Dataset: The loaded dataset.
    """
    seed = conf["seed"]
    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)  # this is redundant once load_dataset is called from train_*.py
    data_config = setup_data_loading(conf)

    training_type = "train" if is_train else "valid"
    dataset_type = conf["data"].get("dataset_type", )
    batch_size = conf["trainer"][f"{training_type}_batch_size"]
    shuffle = is_train
    num_workers = conf["trainer"]["thread_workers"] if is_train else conf["trainer"]["valid_thread_workers"]

    prefetch_factor = conf["trainer"].get("prefetch_factor")
    if prefetch_factor is None:
        logging.warning(
            "prefetch_factor not found in config under 'trainer'. Using default value of 4. "
            "Please specify prefetch_factor in the 'trainer' section of your config."
        )
        prefetch_factor = 4

    # Instantiate the dataset based on the provided class name
    if dataset_type == "ERA5_and_Forcing_SingleStep":  # forecast-len = 0 dataset
        dataset = ERA5_and_Forcing_SingleStep(
            varname_upper_air=conf["data"]["variables"],
            varname_surface=conf["data"]["surface_variables"],
            varname_dyn_forcing=conf["data"]["dynamic_forcing_variables"],
            varname_forcing=conf["data"]["forcing_variables"],
            varname_static=conf["data"]["static_variables"],
            varname_diagnostic=conf["data"]["diagnostic_variables"],
            filenames=data_config["all_ERA_files"],
            filename_surface=data_config["surface_files"],
            filename_dyn_forcing=data_config["dyn_forcing_files"],
            filename_forcing=conf["data"]["save_loc_forcing"],
            filename_static=conf["data"]["save_loc_static"],
            filename_diagnostic=data_config["diagnostic_files"],
            history_len=data_config['history_len'],
            forecast_len=data_config['forecast_len'],
            skip_periods=conf["data"]["skip_periods"],
            one_shot=conf["data"]["one_shot"],
            max_forecast_len=conf["data"]["max_forecast_len"],
            transform=load_transforms(conf),
            sst_forcing=data_config['sst_forcing']
        )
    # All datasets from here on are multi-step examples
    elif dataset_type == "ERA5_and_Forcing_MultiStep":
        dataset = ERA5_and_Forcing_MultiStep(
            varname_upper_air=conf["data"]["variables"],
            varname_surface=conf["data"]["surface_variables"],
            varname_dyn_forcing=conf["data"]["dynamic_forcing_variables"],
            varname_forcing=conf["data"]["forcing_variables"],
            varname_static=conf["data"]["static_variables"],
            varname_diagnostic=conf["data"]["diagnostic_variables"],
            filenames=data_config["all_ERA_files"],
            filename_surface=data_config["surface_files"],
            filename_dyn_forcing=data_config["dyn_forcing_files"],
            filename_forcing=conf["data"]["save_loc_forcing"],
            filename_static=conf["data"]["save_loc_static"],
            filename_diagnostic=data_config["diagnostic_files"],
            history_len=data_config['history_len'],
            forecast_len=data_config['forecast_len'],
            skip_periods=conf["data"]["skip_periods"],
            max_forecast_len=conf["data"]["max_forecast_len"],
            transform=load_transforms(conf),
            rank=rank,
            world_size=world_size,
            seed=seed
        )
    elif dataset_type == "ERA5_MultiStep_Batcher":
        dataset = ERA5_MultiStep_Batcher(
            varname_upper_air=data_config['varname_upper_air'],
            varname_surface=data_config['varname_surface'],
            varname_dyn_forcing=data_config['varname_dyn_forcing'],
            varname_forcing=data_config['varname_forcing'],
            varname_static=data_config['varname_static'],
            varname_diagnostic=data_config['varname_diagnostic'],
            filenames=data_config['all_ERA_files'],
            filename_surface=data_config['surface_files'],
            filename_dyn_forcing=data_config['dyn_forcing_files'],
            filename_forcing=data_config['forcing_files'],
            filename_static=data_config['static_files'],
            filename_diagnostic=data_config['diagnostic_files'],
            history_len=data_config['history_len'],
            forecast_len=data_config['forecast_len'],
            skip_periods=data_config['skip_periods'],
            max_forecast_len=data_config['max_forecast_len'],
            transform=load_transforms(conf),
            batch_size=batch_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size
        )
    elif dataset_type == "MultiprocessingBatcher":
        dataset = MultiprocessingBatcher(
            varname_upper_air=data_config['varname_upper_air'],
            varname_surface=data_config['varname_surface'],
            varname_dyn_forcing=data_config['varname_dyn_forcing'],
            varname_forcing=data_config['varname_forcing'],
            varname_static=data_config['varname_static'],
            varname_diagnostic=data_config['varname_diagnostic'],
            filenames=data_config['all_ERA_files'],
            filename_surface=data_config['surface_files'],
            filename_dyn_forcing=data_config['dyn_forcing_files'],
            filename_forcing=data_config['forcing_files'],
            filename_static=data_config['static_files'],
            filename_diagnostic=data_config['diagnostic_files'],
            history_len=data_config['history_len'],
            forecast_len=data_config['forecast_len'],
            skip_periods=data_config['skip_periods'],
            max_forecast_len=data_config['max_forecast_len'],
            transform=load_transforms(conf),
            batch_size=batch_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers
        )
    elif dataset_type == "MultiprocessingBatcherPrefetch":
        dataset = MultiprocessingBatcherPrefetch(
            varname_upper_air=data_config['varname_upper_air'],
            varname_surface=data_config['varname_surface'],
            varname_dyn_forcing=data_config['varname_dyn_forcing'],
            varname_forcing=data_config['varname_forcing'],
            varname_static=data_config['varname_static'],
            varname_diagnostic=data_config['varname_diagnostic'],
            filenames=data_config['all_ERA_files'],
            filename_surface=data_config['surface_files'],
            filename_dyn_forcing=data_config['dyn_forcing_files'],
            filename_forcing=data_config['forcing_files'],
            filename_static=data_config['static_files'],
            filename_diagnostic=data_config['diagnostic_files'],
            history_len=data_config['history_len'],
            forecast_len=data_config['forecast_len'],
            skip_periods=data_config['skip_periods'],
            max_forecast_len=data_config['max_forecast_len'],
            transform=load_transforms(conf),
            batch_size=batch_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    train_flag = "training" if is_train else "validation"

    logging.info(f"Loaded a {train_flag} {dataset_type} ERA dataset (forecast length = {data_config['forecast_len'] + 1})")

    return dataset


def load_dataloader(conf, dataset, rank=0, world_size=1, is_train=True):
    """
    Load the DataLoader based on the dataset type.

    Args:
        conf (dict): Configuration dictionary containing dataloader parameters.
        dataset (Dataset): The dataset to be used in the DataLoader.
        rank (int, optional): Rank of the current process. Default is 0.
        world_size (int, optional): Number of processes participating in the job. Default is 1.
        is_train (bool, optional): Flag indicating whether the dataset is for training or validation.
                                  Default is True.

    Returns:
        DataLoader: The loaded DataLoader.
    """
    seed = conf["seed"]
    training_type = "train" if is_train else "valid"
    batch_size = conf["trainer"][f"{training_type}_batch_size"]
    shuffle = is_train
    num_workers = conf["trainer"]["thread_workers"] if is_train else conf["trainer"]["valid_thread_workers"]
    prefetch_factor = conf["trainer"].get("prefetch_factor")
    if prefetch_factor is None:
        logging.warning(
            "prefetch_factor not found in config. Using default value of 4. "
            "Please specify prefetch_factor in the 'trainer' section of your config."
        )
        prefetch_factor = 4

    if type(dataset) is ERA5_and_Forcing_SingleStep:
        # This is the single-step dataset, original version
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=shuffle,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            num_workers=num_workers
        )
    elif type(dataset) is ERA5_and_Forcing_MultiStep:
        # This is the deprecated dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=is_train,
            drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=False,
            num_workers=1,  # set to one so prefetch is working
            prefetch_factor=prefetch_factor
        )
    elif type(dataset) is ERA5_MultiStep_Batcher:
        dataloader = DataLoader(
            dataset,
            num_workers=1,  # Must be 1 to use prefetching
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor
        )
    elif type(dataset) is MultiprocessingBatcher:
        dataloader = DataLoaderLite(
            dataset
        )
    elif type(dataset) is MultiprocessingBatcherPrefetch:
        dataloader = DataLoaderLite(
            dataset
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    # Extract the class name using regular expression
    match = re.search(r"\.(\w+) object at", str(dataset))
    class_name = match.group(1)

    train_flag = "training" if is_train else "validation"

    logging.info(f"Loaded a {train_flag} DataLoader for the {class_name} ERA dataset.")

    return dataloader


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py [dataset_type]")
        sys.exit(1)

    dataset_id = int(sys.argv[1])

    import time
    import yaml

    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    with open(
        "/glade/derecho/scratch/schreck/repos/miles-credit/production/multistep/wxformer_6h/model.yml"
    ) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)
    data_config = setup_data_loading(conf)

    # options
    dataset_type = [
        "ERA5_and_Forcing_SingleStep",
        "ERA5_and_Forcing_MultiStep",
        "ERA5_MultiStep_Batcher",
        "MultiprocessingBatcher",
        "MultiprocessingBatcherPrefetch"
    ][dataset_id]

    epoch = 0
    rank = 0
    world_size = 2
    conf["trainer"]["start_epoch"] = epoch
    conf["trainer"]["train_batch_size"] = 1  # batch_size
    conf["trainer"]["valid_batch_size"] = 1  # batch_size
    conf["trainer"]["thread_workers"] = 0   # num_workers
    conf["trainer"]["valid_thread_workers"] = 0   # num_workers
    conf["trainer"]["prefetch_factor"] = 4  # Add prefetch_factor
    conf["data"]["forecast_len"] = 0
    conf["data"]["valid_forecast_len"] = 0
    conf["data"]["dataset_type"] = dataset_type

    set_globals(data_config, namespace=globals())

    try:
        # Load the dataset using the provided dataset_type
        dataset = load_dataset(conf, rank=rank, world_size=world_size)

        # Load the dataloader
        dataloader = load_dataloader(conf, dataset, rank=rank, world_size=world_size)

        # Must set the epoch before the dataloader will work for some datasets
        if hasattr(dataloader.dataset, 'set_epoch'):
            dataloader.dataset.set_epoch(epoch)
        elif hasattr(dataloader, 'set_epoch'):
            dataloader.set_epoch(epoch)

        start_time = time.time()

        # Iterate through the dataloader and print samples
        for (k, sample) in enumerate(dataloader):
            print(k, sample['index'], sample['datetime'], sample['forecast_step'], sample['stop_forecast'], sample["x"].shape, sample["x_surf"].shape)
            if k == 20:
                break

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time for fetching 20 batches: {elapsed_time:.2f} seconds")

    except ValueError as e:
        print(e)
        sys.exit(1)