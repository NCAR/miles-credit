from credit.datasets.era5_multistep import (
    ERA5_and_Forcing_MultiStep,
    RepeatingIndexSampler,
)
from credit.datasets.era5_singlestep import ERA5_and_Forcing_SingleStep
from credit.datasets.era5_multistep_batcher import (
    ERA5_MultiStep_Batcher,
    MultiprocessingBatcher,
    MultiprocessingBatcherPrefetch,
)
from credit.data_downscaling import DownscalingDataset
from credit.datasets import setup_data_loading
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from credit.transforms import load_transforms
import numpy as np
import logging
import torch
import sys
import re


class BatchForecastLenSampler:
    def __init__(self, dataset):
        """
        A PyTorch Sampler designed to preserve the forecast length from the dataset.

        This sampler is tailored for datasets with a non-trivial forecast length,
        ensuring compatibility with batch sampling by adjusting the total number
        of iterations based on the dataset's forecast length and batches per epoch.

        Attributes:
            dataset: The dataset object, which must have `forecast_len` and
                     `batches_per_epoch()` attributes.
            forecast_len: The forecast length incremented by 1.
            len: The total number of iterations determined by the forecast length
                 and the number of batches per epoch.
        """
        self.dataset = dataset
        self.forecast_len = dataset.forecast_len + 1
        self.len = self.dataset.batches_per_epoch() * self.forecast_len

    def __iter__(self):
        """
        Returns an iterator for the sampler.

        The iterator generates a sequence of zeros with a length equal
        to the calculated `len`. This is primarily a placeholder to
        satisfy the interface requirements of a PyTorch Sampler.

        Returns:
            An iterator over a sequence of zeros.
        """
        return iter(np.zeros(self.len))

    def __len__(self):
        """
        Returns the length of the sampler.

        The length is the total number of iterations based on the forecast
        length and batches per epoch from the dataset.

        Returns:
            int: The total number of iterations.
        """
        return self.len


class BatchForecastLenDataLoader:
    def __init__(self, dataset):
        """
        A custom DataLoader that supports datasets with a non-trivial forecast length.

        This DataLoader is designed to iterate over datasets that provide a
         `forecast_len` attribute, optionally incorporating batch-specific
         properties like `batches_per_epoch()` if available.

        Attributes:
            dataset: The dataset object, which must have a `forecast_len` attribute
                      and may optionally have a `batches_per_epoch()` method.
            forecast_len: The forecast length incremented by 1.
        """
        self.dataset = dataset
        self.forecast_len = dataset.forecast_len + 1

    def __iter__(self):
        """
        Iterates over the dataset.

        This method directly yields samples from the dataset. The forecast
         length is not explicitly handled here; it is assumed to be accounted
         for in the dataset's structure or sampling.

        Yields:
            sample: A single sample from the dataset.
        """
        dataset_iter = iter(self.dataset)
        for _ in range(len(self)):
            yield next(dataset_iter)

    def __len__(self):
        """
        Returns the length of the DataLoader.

        The length is determined by the forecast length and either the
         dataset's `batches_per_epoch()` method (if available) or the dataset's
         overall length.

        Returns:
            int: The total number of samples or iterations.
        """
        if hasattr(self.dataset, "batches_per_epoch"):
            return (
                self.dataset.batches_per_epoch() * self.forecast_len
            )  # Use the dataset's method if available
        else:
            return (
                len(self.dataset) * self.forecast_len
            )  # Otherwise, fall back to the dataset's length


def collate_fn(batch):
    """
    Custom collate function for use with the ERA5_MultiStep_Batcher dataset.

    This function ensures that the time and batch dimensions are not flipped
     during data loading. It assumes that the dataset is structured such that
     the first element of the batch contains the correctly formatted data.

    Args:
        batch (list): A list of samples from the dataset, where each sample
                       is expected to be identically structured.

    Returns:
        Any: The first element of the batch, which contains the correctly
             formatted data.
    """
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
    if 'datasets' in conf['data']:  # if true, this is a downscaling dataset
        ## todo: check with parser here
        ## todo: set different first_date and last_date for training vs validation
        dataset = DownscalingDataset(**conf['data'])
        if not is_train:
            dataset.mode = "infer"
    else:
        try:
            data_config = setup_data_loading(conf)
        except KeyError:
            logging.warning(
                "You must run credit.parser.credit_main_parser(conf) before loading data. Exiting."
            )
            sys.exit()
            seed = conf["seed"]
            shuffle = is_train
            training_type = "train" if is_train else "valid"
            dataset_type = conf["data"].get(
                "dataset_type",
            )
            batch_size = conf["trainer"][f"{training_type}_batch_size"]
            shuffle = is_train
            num_workers = (
                conf["trainer"]["thread_workers"]
            if is_train
            else conf["trainer"]["valid_thread_workers"]
            )
            prefetch_factor = conf["trainer"].get(
                "prefetch_factor",
            )
            history_len = (
                data_config["history_len"] if is_train else data_config["valid_history_len"]
            )
            forecast_len = (
                data_config["forecast_len"] if is_train else data_config["valid_forecast_len"]
            )
        if prefetch_factor is None:
            logging.warning(
                "prefetch_factor not found in config under 'trainer'. Using default value of 4. "
                "Please specify prefetch_factor in the 'trainer' section of your config."
            )
            prefetch_factor = 4

        # Instantiate the dataset based on the provided class name
        if dataset_type == "ERA5_and_Forcing_SingleStep":  # forecast-len = 0 dataset
            logging.warning(
                "ERA5_and_Forcing_SingleStep is deprecated. Use ERA5_MultiStep_Batcher or MultiprocessingBatcher for all forecast lengths"
            )
            dataset = ERA5_and_Forcing_SingleStep(
                varname_upper_air=conf["data"]["variables"],
                varname_surface=conf["data"]["surface_variables"],
                varname_dyn_forcing=conf["data"]["dynamic_forcing_variables"],
                varname_forcing=conf["data"]["forcing_variables"],
                varname_static=conf["data"]["static_variables"],
                varname_diagnostic=conf["data"]["diagnostic_variables"],
                filenames=data_config[f"{training_type}_files"],
                filename_surface=data_config[f"{training_type}_surface_files"],
                filename_dyn_forcing=data_config[f"{training_type}_dyn_forcing_files"],
                filename_forcing=conf["data"]["save_loc_forcing"],
                filename_static=conf["data"]["save_loc_static"],
                filename_diagnostic=data_config[f"{training_type}_diagnostic_files"],
                history_len=history_len,
                forecast_len=forecast_len,
                skip_periods=conf["data"]["skip_periods"],
                one_shot=conf["data"]["one_shot"],
                max_forecast_len=conf["data"]["max_forecast_len"],
                transform=load_transforms(conf),
                sst_forcing=data_config["sst_forcing"],
            )
            # All datasets from here on are multi-step examples
        elif dataset_type == "ERA5_and_Forcing_MultiStep":
            logging.warning(
                "ERA5_and_Forcing_MultiStep is deprecated -- it only supports batch size = 1 (ignoring whats in your config).\n"
                "Use ERA5_MultiStep_Batcher or MultiprocessingBatcher for all forecast lengths and batch sizes"
            )
            dataset = ERA5_and_Forcing_MultiStep(
                varname_upper_air=conf["data"]["variables"],
                varname_surface=conf["data"]["surface_variables"],
                varname_dyn_forcing=conf["data"]["dynamic_forcing_variables"],
                varname_forcing=conf["data"]["forcing_variables"],
                varname_static=conf["data"]["static_variables"],
                varname_diagnostic=conf["data"]["diagnostic_variables"],
                filenames=data_config[f"{training_type}_files"],
                filename_surface=data_config[f"{training_type}_surface_files"],
                filename_dyn_forcing=data_config[f"{training_type}_dyn_forcing_files"],
                filename_forcing=conf["data"]["save_loc_forcing"],
                filename_static=conf["data"]["save_loc_static"],
                filename_diagnostic=data_config[f"{training_type}_diagnostic_files"],
                history_len=history_len,
                forecast_len=forecast_len,
                skip_periods=conf["data"]["skip_periods"],
                max_forecast_len=conf["data"]["max_forecast_len"],
                transform=load_transforms(conf),
                rank=rank,
                world_size=world_size,
                seed=seed,
            )
        elif dataset_type == "ERA5_MultiStep_Batcher":
            dataset = ERA5_MultiStep_Batcher(
                varname_upper_air=data_config["varname_upper_air"],
                varname_surface=data_config["varname_surface"],
                varname_dyn_forcing=data_config["varname_dyn_forcing"],
                varname_forcing=data_config["varname_forcing"],
                varname_static=data_config["varname_static"],
                varname_diagnostic=data_config["varname_diagnostic"],
                filenames=data_config[f"{training_type}_files"],
                filename_surface=data_config[f"{training_type}_surface_files"],
                filename_dyn_forcing=data_config[f"{training_type}_dyn_forcing_files"],
                filename_forcing=data_config["forcing_files"],
                filename_static=data_config["static_files"],
                filename_diagnostic=data_config[f"{training_type}_diagnostic_files"],
                history_len=history_len,
                forecast_len=forecast_len,
                skip_periods=data_config["skip_periods"],
                max_forecast_len=data_config["max_forecast_len"],
                transform=load_transforms(conf),
                batch_size=batch_size,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
            )
        elif dataset_type == "MultiprocessingBatcher":
            dataset = MultiprocessingBatcher(
                varname_upper_air=data_config["varname_upper_air"],
                varname_surface=data_config["varname_surface"],
                varname_dyn_forcing=data_config["varname_dyn_forcing"],
                varname_forcing=data_config["varname_forcing"],
                varname_static=data_config["varname_static"],
                varname_diagnostic=data_config["varname_diagnostic"],
                filenames=data_config[f"{training_type}_files"],
                filename_surface=data_config[f"{training_type}_surface_files"],
                filename_dyn_forcing=data_config[f"{training_type}_dyn_forcing_files"],
                filename_forcing=data_config["forcing_files"],
                filename_static=data_config["static_files"],
                filename_diagnostic=data_config[f"{training_type}_diagnostic_files"],
                history_len=history_len,
                forecast_len=forecast_len,
                skip_periods=data_config["skip_periods"],
                max_forecast_len=data_config["max_forecast_len"],
                transform=load_transforms(conf),
                batch_size=batch_size,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
                num_workers=num_workers,
            )
        elif dataset_type == "MultiprocessingBatcherPrefetch":
            dataset = MultiprocessingBatcherPrefetch(
                varname_upper_air=data_config["varname_upper_air"],
                varname_surface=data_config["varname_surface"],
                varname_dyn_forcing=data_config["varname_dyn_forcing"],
                varname_forcing=data_config["varname_forcing"],
                varname_static=data_config["varname_static"],
                varname_diagnostic=data_config["varname_diagnostic"],
                filenames=data_config[f"{training_type}_files"],
                filename_surface=data_config[f"{training_type}_surface_files"],
                filename_dyn_forcing=data_config[f"{training_type}_dyn_forcing_files"],
                filename_forcing=data_config["forcing_files"],
                filename_static=data_config["static_files"],
                filename_diagnostic=data_config[f"{training_type}_diagnostic_files"],
                history_len=history_len,
                forecast_len=forecast_len,
                skip_periods=data_config["skip_periods"],
                max_forecast_len=data_config["max_forecast_len"],
                transform=load_transforms(conf),
                batch_size=batch_size,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        train_flag = "training" if is_train else "validation"

        logging.info(
            f"Loaded a {train_flag} {dataset_type} dataset (forecast length = {data_config['forecast_len'] + 1})"
        )

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
    num_workers = (
        conf["trainer"]["thread_workers"]
        if is_train
        else conf["trainer"]["valid_thread_workers"]
    )
    forecast_len = (
        conf["data"]["forecast_len"] #if is_train else conf["data"]["valid_forecast_len"]
    )
    prefetch_factor = conf["trainer"].get("prefetch_factor")
    if prefetch_factor is None:
        logging.warning(
            "prefetch_factor not found in config. Using default value of 4. "
            "Please specify prefetch_factor in the 'trainer' section of your config."
        )
        prefetch_factor = 4

    # If loss is CRPS, we need all samplers-dataloaders to return the same (x, y)
    # pair as the CDF is computed across GPUs. Randomness is handled by adding noise
    # to the input x to create different samples. There are many other ways to do this
    # but using the same rank and world_size is the fastest as far as communication.
    if conf["loss"]["training_loss"] == "KCRPS":
        rank = 0
        world_size = 1
        logging.info(
            "For CRPS loss, we maintain identical rank and world size across all "
            "GPUs to ensure proper CDF calculation during synchronous distributed processing."
        )

    if type(dataset) is ERA5_and_Forcing_SingleStep:
        # This is the single-step dataset, original version
        def custom_collate_fn(batch):
            # Only return length 1 tensors for forecast_step and stop_forecast
            keys = batch[0].keys()
            collated_batch = {}
            for key in keys:
                items = [item[key] for item in batch]
                if torch.is_tensor(items[0]):
                    collated_batch[key] = torch.stack(items)
                elif isinstance(items[0], (int, float, bool)):
                    collated_batch[key] = torch.tensor(
                        [items[0]]
                        if key in ["forecast_step", "stop_forecast"]
                        else items
                    )
                else:
                    collated_batch[key] = items
            return collated_batch

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
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
        )
    elif type(dataset) is ERA5_and_Forcing_MultiStep:
        # This is the deprecated dataset
        sampler = RepeatingIndexSampler(
            dataset,
            forecast_len=forecast_len,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=is_train,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=False,
            num_workers=1,  # set to one so prefetch is working
            prefetch_factor=prefetch_factor,
        )
    elif type(dataset) is ERA5_MultiStep_Batcher:
        dataloader = DataLoader(
            dataset,
            num_workers=1,  # Must be 1 to use prefetching
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
            sampler=BatchForecastLenSampler(dataset),  # Ensure len is correct
        )
    elif type(dataset) is MultiprocessingBatcher:
        dataloader = BatchForecastLenDataLoader(dataset)
    elif type(dataset) is MultiprocessingBatcherPrefetch:
        dataloader = BatchForecastLenDataLoader(dataset)
    elif type(dataset) is DownscalingDataset:
        print("we made it here!")
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
            num_workers=1,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
            sampler=sampler,
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
    import time
    import yaml
    from credit.parser import credit_main_parser, training_data_check

    if len(sys.argv) != 2:
        print("Usage: python script.py [dataset_type]")
        sys.exit(1)

    dataset_id = int(sys.argv[1])

    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    with open("../../config/example-v2025.2.0.yml") as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)


    # options
    dataset_type = [
        "ERA5_and_Forcing_SingleStep",
        "ERA5_and_Forcing_MultiStep",
        "ERA5_MultiStep_Batcher",
        "MultiprocessingBatcher",
        "MultiprocessingBatcherPrefetch",
    ][dataset_id]


    epoch = 0
    rank = 0
    world_size = 2
    conf["trainer"]["start_epoch"] = epoch
    conf["trainer"]["train_batch_size"] = 2  # batch_size
    conf["trainer"]["valid_batch_size"] = 2  # batch_size
    conf["trainer"]["thread_workers"] = 4  # num_workers
    conf["trainer"]["valid_thread_workers"] = 4  # num_workers
    conf["trainer"]["prefetch_factor"] = 4  # Add prefetch_factor
    conf["data"]["history_len"] = 1
    conf["data"]["valid_history_len"] = conf["data"]["history_len"]
    conf["data"]["forecast_len"] = 5
    conf["data"]["valid_forecast_len"] = 0
    conf["data"]["dataset_type"] = dataset_type

    try:
        # Load the dataset using the provided dataset_type
        dataset = load_dataset(conf, rank=rank, world_size=world_size)

        logger.info(f"datset: {dataset}")

        # Load the dataloader
        dataloader = load_dataloader(conf, dataset, rank=rank, world_size=world_size)

        # Must set the epoch before the dataloader will work for some datasets
        if hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(epoch)
        elif hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)

        start_time = time.time()

        # Iterate through the dataloader and print samples
        for k, sample in enumerate(dataloader):
            print(
                k,
                sample["index"],
                sample["datetime"],
                sample["forecast_step"],
                sample["stop_forecast"],
                sample["x"].shape,
                sample["x_surf"].shape,
            )
            if k == 20:
                break

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time for fetching 20 batches: {elapsed_time:.2f} seconds")

    except ValueError as e:
        print(e)
        sys.exit(1)
