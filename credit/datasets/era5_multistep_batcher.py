import logging
import math
import multiprocessing
import queue
import signal
import time
from datetime import datetime, timedelta
from functools import partial
from queue import Queue
from threading import Thread

import numpy as np
import xarray as xr
import torch
from torch.utils.data import DistributedSampler

from credit.data import (
    drop_var_from_dataset,
    keep_dataset_vars,
    extract_month_day_hour,
    find_common_indices,
    generate_datetime,
    get_forward_data,
    hour_to_nanoseconds,
    nanoseconds_to_year,
)
from credit.datasets.era5_multistep import worker

logger = logging.getLogger(__name__)


class ERA5_MultiStep_Batcher(torch.utils.data.Dataset):
    """
    A Pytorch Dataset class that works on:
        - upper-air variables (time, level, lat, lon)
        - surface variables (time, lat, lon)
        - dynamic forcing variables (time, lat, lon)
        - foring variables (time, lat, lon)
        - diagnostic variables (time, lat, lon)
        - static variables (lat, lon)
    """

    def __init__(
        self,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames,
        filename_surface=None,
        filename_dyn_forcing=None,
        filename_forcing=None,
        filename_static=None,
        filename_diagnostic=None,
        sst_forcing=None,
        history_len=2,
        forecast_len=0,
        transform=None,
        seed=42,
        rank=0,
        world_size=1,
        skip_periods=None,
        max_forecast_len=None,
        batch_size=1,
        shuffle=True,
    ):
        """
        Initialize the ERA5_and_Forcing_Dataset

        Parameters:
        - varname_upper_air (list): List of upper air variable names.
        - varname_surface (list): List of surface variable names.
        - varname_dyn_forcing (list): List of dynamic forcing variable names.
        - varname_forcing (list): List of forcing variable names.
        - varname_static (list): List of static variable names.
        - varname_diagnostic (list): List of diagnostic variable names.
        - filenames (list): List of filenames for upper air data.
        - filename_surface (list, optional): List of filenames for surface data.
        - filename_dyn_forcing (list, optional): List of filenames for dynamic forcing data.
        - filename_forcing (str, optional): Filename for forcing data.
        - filename_static (str, optional): Filename for static data.
        - filename_diagnostic (list, optional): List of filenames for diagnostic data.
        - history_len (int, optional): Length of the history sequence. Default is 2.
        - forecast_len (int, optional): Length of the forecast sequence. Default is 0.
        - transform (callable, optional): Transformation function to apply to the data.
        - seed (int, optional): Random seed for reproducibility. Default is 42.
        - skip_periods (int, optional): Number of periods to skip between samples.
        - max_forecast_len (int, optional): Maximum length of the forecast sequence.
        - shuffle (bool, optional): Whether to shuffle the data. Default is True.
        - sst_forcing (optional):
        Returns:
        - sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.
        """

        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle

        # skip periods
        self.skip_periods = skip_periods
        if self.skip_periods is None:
            self.skip_periods = 1

        # total number of needed forecast lead times
        self.total_seq_len = self.history_len + self.forecast_len

        # set random seed
        self.rng = np.random.default_rng(seed=seed)

        # max possible forecast len
        self.max_forecast_len = max_forecast_len

        # sst forcing
        self.sst_forcing = sst_forcing

        # =================================================================== #
        # flags to determin if any of the [surface, dyn_forcing, diagnostics]
        # variable groups share the same file as upper air variables
        flag_share_surf = False
        flag_share_dyn = False
        flag_share_diag = False

        all_files = []
        filenames = sorted(filenames)

        # ------------------------------------------------------------------ #
        # blocks that can handle no-sharing (each group has it own file)
        # surface
        surface_files = None
        if filename_surface is not None:
            surface_files = []
            filename_surface = sorted(filename_surface)

            if filenames == filename_surface:
                flag_share_surf = True
            else:
                for fn in filename_surface:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_surf = keep_dataset_vars(ds, varname_surface)
                    surface_files.append(ds_surf)

                self.surface_files = surface_files
        else:
            self.surface_files = None

        # dynamic forcing
        dyn_forcing_files = None
        if filename_dyn_forcing is not None:
            dyn_forcing_files = []
            filename_dyn_forcing = sorted(filename_dyn_forcing)

            if filenames == filename_dyn_forcing:
                flag_share_dyn = True
            else:
                for fn in filename_dyn_forcing:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_dyn = keep_dataset_vars(ds, varname_dyn_forcing)
                    dyn_forcing_files.append(ds_dyn)

                self.dyn_forcing_files = dyn_forcing_files
        else:
            self.dyn_forcing_files = None

        # diagnostics
        diagnostic_files = None
        if filename_diagnostic is not None:
            diagnostic_files = []
            filename_diagnostic = sorted(filename_diagnostic)

            if filenames == filename_diagnostic:
                flag_share_diag = True
            else:
                for fn in filename_diagnostic:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_diag = keep_dataset_vars(ds, varname_diagnostic)
                    diagnostic_files.append(ds_diag)

                self.diagnostic_files = diagnostic_files
        else:
            self.diagnostic_files = None

        # ------------------------------------------------------------------ #
        # blocks that can handle file sharing (share with upper air file)
        for fn in filenames:
            # drop variables if they are not in the config
            ds = get_forward_data(filename=fn)
            ds_upper = keep_dataset_vars(ds, varname_upper_air)

            if flag_share_surf:
                ds_surf = keep_dataset_vars(ds, varname_surface)
                surface_files.append(ds_surf)

            if flag_share_dyn:
                ds_dyn = keep_dataset_vars(ds, varname_dyn_forcing)
                dyn_forcing_files.append(ds_dyn)

            if flag_share_diag:
                ds_diag = keep_dataset_vars(ds, varname_diagnostic)
                diagnostic_files.append(ds_diag)

            all_files.append(ds_upper)

        self.all_files = all_files

        if flag_share_surf:
            self.surface_files = surface_files
        if flag_share_dyn:
            self.dyn_forcing_files = dyn_forcing_files
        if flag_share_diag:
            self.diagnostic_files = diagnostic_files

        # -------------------------------------------------------------------------- #
        # get sample indices from ERA5 upper-air files:
        ind_start = 0
        self.ERA5_indices = {}  # <------ change
        for ind_file, ERA5_xarray in enumerate(self.all_files):
            # [number of samples, ind_start, ind_end]
            self.ERA5_indices[str(ind_file)] = [
                ERA5_xarray["time"].shape[0],
                ind_start,
                ind_start + ERA5_xarray["time"].shape[0],
            ]
            ind_start += ERA5_xarray["time"].shape[0] + 1

        # ======================================================== #
        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_forcing)
            xarray_dataset = keep_dataset_vars(xarray_dataset, varname_forcing)

            self.xarray_forcing = xarray_dataset
        else:
            self.xarray_forcing = None

        # ======================================================== #
        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_static)
            xarray_dataset = keep_dataset_vars(xarray_dataset, varname_static)

            self.xarray_static = xarray_dataset
        else:
            self.xarray_static = None

        self.worker = partial(
            worker,
            ERA5_indices=self.ERA5_indices,
            all_files=self.all_files,
            surface_files=self.surface_files,
            dyn_forcing_files=self.dyn_forcing_files,
            diagnostic_files=self.diagnostic_files,
            xarray_forcing=self.xarray_forcing,
            xarray_static=self.xarray_static,
            sst_forcing=self.sst_forcing,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
            skip_periods=self.skip_periods,
            transform=self.transform,
        )

        # Set an epoch flag so that if set_epoch is not called, a warning will be issued
        self.current_epoch = None

        # Use DistributedSampler for index management
        self.sampler = DistributedSampler(
            self,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )

        # Initialize state variables for batch management
        self.batch_size = batch_size
        self.batch_indices = None  # To track initial indices for each batch item
        self.time_steps = None  # Tracks time steps for each batch index
        self.forecast_step_counts = None  # Track forecast step counts for each batch item

        # initialze the batch indices by faking the epoch number here and resetting to None
        # this is mainly a feature for working with smaller datasets / testing purposes
        self.sampler.set_epoch(0)
        self.batch_indices = list(self.sampler)
        if len(self.batch_indices) < batch_size:
            logger.warning(
                f"Note that the batch size ({batch_size}) is larger than the number of data indices ({len(self.batch_indices)})"
                f"Resetting the batch size to {len(self.batch_indices)}."
            )
            self.batch_size = len(self.batch_indices)

    def initialize_batch(self):
        """
        Initializes batch indices using DistributedSampler's indices.
        Ensures proper cycling when shuffle=False.
        """
        # Initialize the call count if not already present
        if not hasattr(self, "batch_call_count"):
            self.batch_call_count = 0

        # Set epoch for DistributedSampler to ensure consistent shuffling across devices
        if self.current_epoch is None:
            logging.warning("You must first set the epoch number using set_epoch method.")

        # Retrieve indices for this GPU
        total_indices = len(self.batch_indices)

        # Select batch indices based on call count (deterministic cycling)
        start = self.batch_call_count * self.batch_size
        end = start + self.batch_size

        if not self.shuffle:
            if end > total_indices:
                # Simple wraparound by incrementing start index
                start = start % total_indices
                end = min(start + self.batch_size, total_indices)
            indices = self.batch_indices[start:end]
        else:
            if end > total_indices:
                # Wrap-around to ensure no index is skipped
                indices = self.batch_indices[start:] + self.batch_indices[: (end % total_indices)]
            else:
                indices = self.batch_indices[start:end]

        # Increment batch_call_count, reset when all indices are cycled
        self.batch_call_count += 1
        if start + self.batch_size >= total_indices:
            self.batch_call_count = 0  # Reset for next cycle

        # Assign batch indices
        self.current_batch_indices = list(indices)  # this will be the local indices used in getitem
        self.time_steps = [0 for _ in self.batch_indices]
        self.forecast_step_counts = [0 for _ in self.batch_indices]

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += ERA5_xarray["time"].shape[0] - self.total_seq_len + 1
        return total_len

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.sampler.set_epoch(epoch)
        self.batch_indices = list(self.sampler)
        self.batch_call_count = 0
        self.initialize_batch()

    def batches_per_epoch(self):
        return math.ceil(len(list(self.batch_indices)) / self.batch_size)

    def __getitem__(self, _):
        """
        Fetches the current forecast step data for each item in the batch.
        Resets items when their forecast length is exceeded.
        """
        batch = {}

        # If the forecast_step_count exceeds forecast_len, reset the item
        # If one exceeds, they all exceed and all neet reset
        if self.forecast_step_counts[0] == self.forecast_len + 1:
            # Get a new starting index for this item (randomly selected)
            self.initialize_batch()

        for k, idx in enumerate(self.current_batch_indices):
            # Get the current time step for this batch item
            current_t = self.time_steps[k]
            index_pair = (idx, idx + current_t)  # Correctly construct the index_pair

            # Fetch the current sample for this batch item
            sample = self.worker(index_pair)

            # Add index to the sample
            sample["index"] = idx + current_t

            # Concatenate data by common keys in sample
            for key, value in sample.items():
                if isinstance(value, np.ndarray):  # If it's a numpy array
                    value = torch.tensor(value)
                elif isinstance(value, np.int64):  # If it's a numpy scalar (int64)
                    value = torch.tensor(value, dtype=torch.int64)
                elif isinstance(value, (int, float)):  # If it's a native Python scalar (int/float)
                    value = torch.tensor(value, dtype=torch.float32)  # Ensure tensor is float for scalar
                elif not isinstance(value, torch.Tensor):  # If it's not already a tensor
                    value = torch.tensor(value)  # Ensure conversion to tensor

                # Convert zero-dimensional tensor (scalar) to 1D tensor
                if value.ndimension() == 0:
                    value = value.unsqueeze(0)  # Unsqueeze to make it a 1D tensor

                # add the time, which is 1 in all datasets in this example
                # this is needed since we use BatchForecastLenDataLoader, which does not add
                # the extra dimension and a mix up between batch and time-dim happens.
                if value.ndim in (4, 5):
                    value = value.unsqueeze(0)

                if key not in batch:
                    batch[key] = value  # Initialize the key in the batch dictionary
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)  # Concatenate values along the batch dimension

            # Increment time steps and forecast step counts for this batch item
            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = torch.tensor([self.forecast_step_counts[0]])
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len + 1
        batch["datetime"] = batch["datetime"].view(-1, len(self.current_batch_indices))

        return batch


class MultiprocessingBatcher(ERA5_MultiStep_Batcher):
    def __init__(self, *args, num_workers=4, **kwargs):
        """
        Initialize the MultiprocessingBatcher with a configurable number of workers.

        Args:
            num_workers (int): Number of workers to use for multiprocessing.
            *args, **kwargs: Arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)  # Initialize the parent class
        self.num_workers = num_workers if num_workers > 0 else 1
        self.num_workers = min(self.num_workers, self.batch_size)

        # Shared dictionary to collect results from multiple processes (keyed by index)
        self.manager = multiprocessing.Manager()
        self.results = self.manager.dict()

    def __getitem__(self, _):
        """
        Fetches the current forecast step data for each item in the batch.
        Utilizes multiprocessing to parallelize calls to `self.worker`.
        Ensures the results are returned in the correct order.
        """
        batch = {}

        # Reset items if forecast step count exceeds forecast length
        if self.forecast_step_counts[0] == self.forecast_len + 1:
            self.initialize_batch()
            self.results.clear()

        # Prepare arguments for processing
        args = []
        for k, idx in enumerate(self.current_batch_indices):
            current_t = self.time_steps[k]
            index_pair = (idx, idx + current_t)
            args.append((k, index_pair, self.results))

        # Worker function
        def worker_process(start_idx, end_idx):
            for i in range(start_idx, end_idx):
                k, index_pair = args[i][:2]
                sample = self.worker(index_pair)
                sample["index"] = index_pair[1]  # Add index to the sample
                self.results[k] = sample  # Store the result keyed by its order index

        # Split tasks among workers
        processes = []
        splits = np.array_split(range(len(args)), self.num_workers)
        start_ends = [(split[0], split[-1] + 1) for split in splits if len(split)]
        for start_idx, end_idx in start_ends:
            p = multiprocessing.Process(target=worker_process, args=(start_idx, end_idx))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Sort results by their original order
        ordered_results = [self.results[k] for k in sorted(self.results.keys())]

        # Process sorted results and build the batch
        for k, sample in enumerate(ordered_results):
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    value = torch.tensor(value)
                elif isinstance(value, np.int64):
                    value = torch.tensor(value, dtype=torch.int64)
                elif isinstance(value, (int, float)):
                    value = torch.tensor(value, dtype=torch.float32)
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)

                if value.ndimension() == 0:
                    value = value.unsqueeze(0)

                # add the time, which is 1 in all datasets in this example
                # this is needed since we use BatchForecastLenDataLoader, which does not add
                # the extra dimension and a mix up between batch and time-dim happens.
                if value.ndim in (4, 5):
                    value = value.unsqueeze(0)

                if key not in batch:
                    batch[key] = value
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)

            # Increment time steps and forecast step counts for this batch item
            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = torch.tensor([self.forecast_step_counts[0]])
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len + 1
        batch["datetime"] = batch["datetime"].view(-1, len(self.current_batch_indices))

        return batch

    def __del__(self):
        """Cleanup the manager when the object is destroyed"""
        if hasattr(self, "manager"):
            self.manager.shutdown()


class MultiprocessingBatcherPrefetch(ERA5_MultiStep_Batcher):
    def __init__(self, *args, num_workers=4, prefetch_factor=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers if num_workers > 0 else 1
        self.prefetch_factor = prefetch_factor
        self.prefetch_queue = Queue(maxsize=prefetch_factor)
        self.stop_signal = multiprocessing.Event()

        # Manager for shared resources
        self.manager = multiprocessing.Manager()
        self.results = self.manager.dict()

        # Register signal handler
        self.stop_event = multiprocessing.Event()
        # signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        self.prefetch_thread = None

    def handle_signal(self, signum, frame):
        logging.info("Received signal, shutting down worker processes...")
        self.stop_event.set()  # Signal worker processes to stop

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.sampler.set_epoch(epoch)
        self.batch_indices = list(self.sampler)
        self.batch_call_count = 0
        self.initialize_batch()

        # Start prefetching thread
        if self.prefetch_thread is None:
            self.prefetch_thread = Thread(target=self.prefetch_batches, daemon=True)
            self.prefetch_thread.start()

    def prefetch_batches(self):
        """
        Prefetch batches asynchronously and store them in a queue.
        Stops when the `stop_signal` is set.
        """
        try:
            while not self.stop_signal.is_set():
                if not self.prefetch_queue.full():  # Only prefetch if the queue has space
                    try:
                        batch = self._fetch_batch()
                        self.prefetch_queue.put(batch)  # Add batch to the queue
                    except Exception as e:
                        print(f"Error during prefetching: {e}")
                else:
                    # Wait briefly to avoid busy waiting when the queue is full
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in prefetch thread: {e}")

    def worker_process(self, k, index_pair, result_dict):
        """
        Worker function that processes individual tasks, with error handling for specific exceptions.
        """
        try:
            sample = self.worker(index_pair)
            sample["index"] = index_pair[1]  # Add index to the sample
            result_dict[k] = sample  # Store the result keyed by its order index
        except FileNotFoundError:
            # Log the error but continue processing
            # logger.warning(f"Ignoring transient connection error for index {k}.")
            self.stop_event.set()
            return
        except Exception as e:
            logger.warning(
                f"Error in worker process for index {k}: {e}.\n"
                "This is likely due to the end of training, or you killed the program. Exiting!\n"
                "Not what you expected? Email schreck@ucar.edu for support"
            )
            self.stop_event.set()
            # logger.error(f"Error in worker process for index {k}: {e}")
            # Ensure proper cleanup before exiting worker process
            if hasattr(self, "shutdown") and callable(self.shutdown):
                logger.info("Initiating shutdown sequence.")
                self.shutdown()
            return

    def _fetch_batch(self):
        """
        Fetches a batch using multiprocessing workers and splits the work efficiently.
        """
        batch = {}

        # Reset items if forecast step count exceeds forecast length
        if self.forecast_step_counts[0] == self.forecast_len + 1 or self.batch_indices is None:
            self.initialize_batch()

        # Prepare arguments for processing
        tasks = []
        for k, idx in enumerate(self.current_batch_indices):
            current_t = self.time_steps[k]
            index_pair = (idx, idx + current_t)
            tasks.append((k, index_pair))

        # Split tasks among workers (efficient chunking)
        chunk_size = max(1, len(tasks) // self.num_workers)
        task_chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

        # Shared dictionary to collect results from workers
        results = self.results

        processes = []
        for chunk in task_chunks:
            p = multiprocessing.Process(target=self._process_chunk, args=(chunk, results))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Sort results by original order
        ordered_results = [results[k] for k in sorted(results.keys())]

        # Process sorted results and build the batch
        for k, sample in enumerate(ordered_results):
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    value = torch.tensor(value)
                elif isinstance(value, np.int64):
                    value = torch.tensor(value, dtype=torch.int64)
                elif isinstance(value, (int, float)):
                    value = torch.tensor(value, dtype=torch.float32)
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)

                if value.ndimension() == 0:
                    value = value.unsqueeze(0)

                # add the time, which is 1 in all datasets in this example
                # this is needed since we use BatchForecastLenDataLoader, which does not add
                # the extra dimension and a mix up between batch and time-dim happens.
                if value.ndim in (4, 5):
                    value = value.unsqueeze(0)

                if key not in batch:
                    batch[key] = value
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)

            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = torch.tensor([self.forecast_step_counts[0]])
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len + 1
        batch["datetime"] = batch["datetime"].view(-1, len(self.current_batch_indices))

        return batch

    def _process_chunk(self, task_chunk, result_dict):
        """
        Process a chunk of tasks and update the shared results dictionary.
        """
        for k, index_pair in task_chunk:
            self.worker_process(k, index_pair, result_dict)

    def __getitem__(self, _):
        """
        Get a batch from the prefetch queue.
        """
        return self.prefetch_queue.get()

    def __del__(self):
        """
        Cleanup processes and threads when the object is destroyed.
        """
        try:
            # Terminate running processes
            if hasattr(self, "processes"):
                for p in self.processes:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=1)

            # Close and remove any open managers
            if hasattr(self, "manager"):
                self.manager.shutdown()
                del self.manager

            # Clear queues
            if hasattr(self, "prefetch_queue"):
                while not self.prefetch_queue.empty():
                    try:
                        self.prefetch_queue.get_nowait()
                    except queue.Empty:
                        break

            # Stop threads and events
            if hasattr(self, "stop_signal"):
                self.stop_signal.set()

            if hasattr(self, "prefetch_thread"):
                self.prefetch_thread.join(timeout=2)

        except Exception as e:
            print(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()


class Predict_Dataset_Batcher(torch.utils.data.Dataset):
    """
    A Pytorch Dataset class that works on:
        - upper-air variables (time, level, lat, lon)
        - surface variables (time, lat, lon)
        - dynamic forcing variables (time, lat, lon)
        - foring variables (time, lat, lon)
        - diagnostic variables (time, lat, lon)
        - static variables (lat, lon)
    """

    def __init__(
        self,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames,
        filename_surface=None,
        filename_dyn_forcing=None,
        filename_forcing=None,
        filename_static=None,
        filename_diagnostic=None,
        sst_forcing=None,
        fcst_datetime=None,
        lead_time_periods=6,
        history_len=1,
        transform=None,
        seed=42,
        rank=0,
        world_size=1,
        skip_periods=None,
        batch_size=1,
        skip_target=False,
    ):
        """
        Initialize the ERA5_and_Forcing_Dataset

        Parameters:
        - varname_upper_air (list): List of upper air variable names.
        - varname_surface (list): List of surface variable names.
        - varname_dyn_forcing (list): List of dynamic forcing variable names.
        - varname_forcing (list): List of forcing variable names.
        - varname_static (list): List of static variable names.
        - varname_diagnostic (list): List of diagnostic variable names.
        - filenames (list): List of filenames for upper air data.
        - filename_surface (list, optional): List of filenames for surface data.
        - filename_dyn_forcing (list, optional): List of filenames for dynamic forcing data.
        - filename_forcing (str, optional): Filename for forcing data.
        - filename_static (str, optional): Filename for static data.
        - filename_diagnostic (list, optional): List of filenames for diagnostic data.
        - history_len (int, optional): Length of the history sequence. Default is 2.
        - transform (callable, optional): Transformation function to apply to the data.
        - seed (int, optional): Random seed for reproducibility. Default is 42.
        - skip_periods (int, optional): Number of periods to skip between samples.
        - max_forecast_len (int, optional): Maximum length of the forecast sequence.
        - sst_forcing (optional):
        - skip_target: Do not return y truth data
        Returns:
        - sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.
        """

        self.history_len = history_len
        self.transform = transform
        self.init_datetime = fcst_datetime
        self.lead_time_periods = lead_time_periods
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.skip_target = skip_target

        # skip periods
        self.skip_periods = skip_periods
        if self.skip_periods is None:
            self.skip_periods = 1

        # set random seed
        self.rng = np.random.default_rng(seed=seed)

        # sst forcing
        self.sst_forcing = sst_forcing

        # flags to determine if any of the [surface, dyn_forcing, diagnostics]
        # variable groups share the same file as upper air variables
        flag_share_surf = False
        flag_share_dyn = False
        flag_share_diag = False

        all_files = []
        filenames = sorted(filenames)

        # blocks that can handle no-sharing (each group has it own file)
        # surface
        surface_files = None
        if filename_surface is not None:
            surface_files = []
            filename_surface = sorted(filename_surface)

            if filenames == filename_surface:
                flag_share_surf = True
            else:
                for fn in filename_surface:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_surf = keep_dataset_vars(ds, varname_surface)
                    surface_files.append(ds_surf)

                self.surface_files = surface_files
        else:
            self.surface_files = None

        # dynamic forcing
        dyn_forcing_files = None
        if filename_dyn_forcing is not None:
            dyn_forcing_files = []
            filename_dyn_forcing = sorted(filename_dyn_forcing)

            if filenames == filename_dyn_forcing:
                flag_share_dyn = True
            else:
                for fn in filename_dyn_forcing:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_dyn = keep_dataset_vars(ds, varname_dyn_forcing)
                    dyn_forcing_files.append(ds_dyn)

                self.dyn_forcing_files = dyn_forcing_files
        else:
            self.dyn_forcing_files = None

        # diagnostics
        diagnostic_files = None
        if filename_diagnostic is not None:
            diagnostic_files = []
            filename_diagnostic = sorted(filename_diagnostic)

            if filenames == filename_diagnostic:
                flag_share_diag = True
            else:
                for fn in filename_diagnostic:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_diag = keep_dataset_vars(ds, varname_diagnostic)
                    diagnostic_files.append(ds_diag)

                self.diagnostic_files = diagnostic_files
        else:
            self.diagnostic_files = False

        # blocks that can handle file sharing (share with upper air file)
        for fn in filenames:
            # drop variables if they are not in the config
            ds = get_forward_data(filename=fn)
            ds_upper = keep_dataset_vars(ds, varname_upper_air)

            if flag_share_surf:
                ds_surf = keep_dataset_vars(ds, varname_surface)
                surface_files.append(ds_surf)

            if flag_share_dyn:
                ds_dyn = keep_dataset_vars(ds, varname_dyn_forcing)
                dyn_forcing_files.append(ds_dyn)

            if flag_share_diag:
                ds_diag = keep_dataset_vars(ds, varname_diagnostic)
                diagnostic_files.append(ds_diag)

            all_files.append(ds_upper)

        # file names
        self.all_files = all_files
        self.filenames = filenames
        self.filename_surface = filename_surface
        self.filename_dyn_forcing = filename_dyn_forcing
        self.filename_forcing = filename_forcing
        self.filename_static = filename_static
        self.filename_diagnostic = filename_diagnostic

        # var names
        self.varname_upper_air = varname_upper_air
        self.varname_surface = varname_surface
        self.varname_dyn_forcing = varname_dyn_forcing
        self.varname_forcing = varname_forcing
        self.varname_static = varname_static
        self.varname_diagnostic = varname_diagnostic

        if flag_share_surf:
            self.surface_files = surface_files
        if flag_share_dyn:
            self.dyn_forcing_files = dyn_forcing_files
        if flag_share_diag:
            self.diagnostic_files = diagnostic_files

        # get sample indices from ERA5 upper-air files:
        ind_start = 0
        self.ERA5_indices = {}  # <------ change
        for ind_file, ERA5_xarray in enumerate(self.all_files):
            # [number of samples, ind_start, ind_end]
            self.ERA5_indices[str(ind_file)] = [
                ERA5_xarray["time"].shape[0],
                ind_start,
                ind_start + ERA5_xarray["time"].shape[0],
            ]
            ind_start += ERA5_xarray["time"].shape[0] + 1

        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_forcing)
            xarray_dataset = keep_dataset_vars(xarray_dataset, varname_forcing)

            self.xarray_forcing = xarray_dataset
        else:
            self.xarray_forcing = None

        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_static)
            xarray_dataset = keep_dataset_vars(xarray_dataset, varname_static)

            self.xarray_static = xarray_dataset.load()
        else:
            self.xarray_static = None

        # Initialize the first forecast so we can get the forecast_len
        # which up to here is not defined. Needed in __len__ so DataLoader knows when to stop
        # convert to datetime object
        fcst_datetime_0 = datetime.strptime(self.init_datetime[0][0], "%Y-%m-%d %H:%M:%S")
        fcst_datetime_1 = datetime.strptime(self.init_datetime[0][1], "%Y-%m-%d %H:%M:%S")
        # convert the 1st & last init times to a list of init times
        self.forecast_period = len(
            generate_datetime(
                fcst_datetime_0,
                fcst_datetime_1,
                self.lead_time_periods,
            )
        )
        self.forecast_len = self.forecast_period - 1  # For consistency with other datasets that use  forecast_len
        # logger.info(f'rank={self.rank} {self.forecast_period=:} {self.forecast_len=:}')

        all_indices_splits = torch.tensor_split(torch.arange(len(self.init_datetime)), world_size)
        self.batch_indices = all_indices_splits[rank].long()
        self.batch_indices_splits = []
        total_indices = len(self.batch_indices)
        if (
            total_indices == 0
        ):  #  Some processes will not have indices if world_size is greater then len(self.init_datetime),
            return

        chunks = math.ceil(total_indices / self.batch_size)
        self.batch_indices_splits = torch.tensor_split(self.batch_indices, chunks)
        self.batch_call_count = 0
        self.data_lookup = None
        self.initialize_batch()

    def __len__(self):
        total_forecasts = len(self.init_datetime)
        return total_forecasts  # conf["predict"]["duration"] x len(conf["predict"]["start_hours"])

    def ds_read_and_subset(self, filename, time_start, time_end, varnames):
        sliced_x = get_forward_data(filename)
        sliced_x = sliced_x.isel(time=slice(time_start, time_end))
        sliced_x = drop_var_from_dataset(sliced_x, varnames)
        return sliced_x

    def get_time_variable(self, filename, time_start, time_end) -> xr.Dataset:
        """
        Open NetCDF or Zarr file and return only the time variable.
        """
        if filename.endswith((".nc", ".nc4")):
            dataset = xr.open_dataset(filename, decode_times=False, engine="h5netcdf")
        else:
            dataset = xr.open_zarr(filename, consolidated=True)
        
        dataset = dataset.drop_vars(list(dataset.data_vars))
        dataset = dataset.isel(time=slice(time_start, time_end))
        
        return dataset

    def load_zarr_as_input(self, i_file, i_init_start, i_init_end, mode="input"):
        # get the needed file from a list of zarr files
        # open the zarr file as xr.dataset and subset based on the needed time
        # if mode == "forcing":
        #     sliced_x = self.get_time_variable(self.filenames[i_file], i_init_start, i_init_end + 1)
        sliced_x = self.all_files[i_file]["time"][i_init_start : i_init_end + 1]
        if mode in ["input", "target"]:
            # sliced_x: the final output, starts with an upper air xr.dataset
            # sliced_x = self.ds_read_and_subset(
            #     self.filenames[i_file], i_init_start, i_init_end + 1, self.varname_upper_air
            # )
            sliced_x = self.all_files[i_file].isel(time=slice(i_init_start, i_init_end + 1)).load()
            # surface variables
            if self.filename_surface is not None:
                # sliced_surface = self.ds_read_and_subset(
                #    self.filename_surface[i_file],
                #    i_init_start,
                #    i_init_end + 1,
                #    self.varname_surface,
                # )
                # merge surface to sliced_x
                sliced_surface = self.surface_files[i_file].isel(time=slice(i_init_start, i_init_end + 1)).load()
                # sliced_surface["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_surface)

        if mode in ["input", "forcing"]:
            # dynamic forcing variables
            if self.filename_dyn_forcing is not None:
                sliced_dyn_forcing = (
                    self.dyn_forcing_files[i_file].isel(time=slice(i_init_start, i_init_end + 1)).load()
                )
                # sliced_dyn_forcing = self.ds_read_and_subset(
                #    self.filename_dyn_forcing[i_file],
                #    i_init_start,
                #    i_init_end + 1,
                #    self.varname_dyn_forcing,
                # )
                # merge surface to sliced_x
                sliced_dyn_forcing["time"] = sliced_x["time"]
                if mode == "forcing":
                    sliced_x = sliced_dyn_forcing
                else:
                    sliced_x = sliced_x.merge(sliced_dyn_forcing)

            # forcing / periodic static
            if self.filename_forcing is not None:
                sliced_forcing = self.xarray_forcing
                # See also `ERA5_and_Forcing_Dataset`
                # matching month, day, hour between forcing and upper air [time]
                # this approach handles leap year forcing file and non-leap-year upper air file
                month_day_forcing = extract_month_day_hour(np.array(sliced_forcing["time"]))
                month_day_inputs = extract_month_day_hour(np.array(sliced_x["time"]))
                # indices to subset
                ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
                sliced_forcing = sliced_forcing.isel(time=ind_forcing).load()
                # forcing and upper air have different years but the same mon/day/hour
                # safely replace forcing time with upper air time
                sliced_forcing["time"] = sliced_x["time"]
                
                # merge forcing to sliced_x
                sliced_x = sliced_x.merge(sliced_forcing)
                
            if self.filename_static is not None:
                # sliced_static = get_forward_data(self.filename_static)
                # sliced_static = drop_var_from_dataset(sliced_static, self.varname_static)
                sliced_static = self.xarray_static
                sliced_static = sliced_static.expand_dims(dim={"time": sliced_x["time"].shape[0]})
                sliced_static["time"] = sliced_x["time"]
                # merge static to sliced_x
                sliced_x = sliced_x.merge(sliced_static)

        elif mode == "target":
            # diagnostic
            if self.filename_diagnostic is not None:
                sliced_diagnostic = self.ds_read_and_subset(
                    self.filename_diagnostic[i_file],
                    i_init_start,
                    i_init_end + 1,
                    self.varname_diagnostic,
                )
                # merge diagnostics to sliced_x
                sliced_diagnostic["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_diagnostic)

        return sliced_x

    def find_start_stop_indices(self, index):
        # shift hours for history_len > 1, because more than one init times are needed
        # <--- !! it MAY NOT work when self.skip_period != 1
        shifted_hours = self.lead_time_periods * self.skip_periods * (self.history_len - 1)

        # subtrack shifted_hour form the 1st & last init times
        # convert to datetime object
        self.init_datetime[index][0] = datetime.strptime(self.init_datetime[index][0], "%Y-%m-%d %H:%M:%S") - timedelta(
            hours=shifted_hours
        )
        self.init_datetime[index][1] = datetime.strptime(self.init_datetime[index][1], "%Y-%m-%d %H:%M:%S") - timedelta(
            hours=shifted_hours
        )

        # convert the 1st & last init times to a list of init times
        self.init_datetime[index] = generate_datetime(
            self.init_datetime[index][0],
            self.init_datetime[index][1],
            self.lead_time_periods,
        )
        # convert datetime obj to nanosecondes
        init_time_list_dt = [np.datetime64(date.strftime("%Y-%m-%d %H:%M:%S")) for date in self.init_datetime[index]]

        # init_time_list_np: a list of python datetime objects, each is a forecast step
        # init_time_list_np[0]: the first initialization time
        # init_time_list_np[t]: the forcasted time of the (t-1)th step; the initialization time of the t-th step
        self.init_time_list_np = [
            np.datetime64(str(dt_obj) + ".000000000").astype(datetime) for dt_obj in init_time_list_dt
        ]

        info = []
        for init_time in self.init_time_list_np:
            for i_file, ds in enumerate(self.all_files):
                # get the year of the current file
                # looks messy because extra code needed to handle cftime
                ds_year = int((np.datetime_as_string(ds["time"][0].astype("datetime64[ns]").values, unit="Y")))

                # get the first and last years of init times
                init_year0 = nanoseconds_to_year(init_time)

                # found the right yearly file
                if init_year0 == ds_year:
                    N_times = len(ds["time"])
                    # convert ds['time'] to a list of nanosecondes
                    ds_time_list = [ds_time.astype("datetime64[ns]").values.astype(datetime) for ds_time in ds["time"]]
                    ds_start_time = ds_time_list[0]
                    ds_end_time = ds_time_list[-1]

                    init_time_start = init_time
                    # if initalization time is within this (yearly) xr.Dataset
                    if ds_start_time <= init_time_start <= ds_end_time:
                        # try getting the index of the first initalization time
                        i_init_start = ds_time_list.index(init_time_start)

                        # for multiple init time inputs (history_len > 1), init_end is different for init_start
                        init_time_end = init_time_start + hour_to_nanoseconds(shifted_hours)

                        # see if init_time_end is alos in this file
                        if ds_start_time <= init_time_end <= ds_end_time:
                            # try getting the index
                            i_init_end = ds_time_list.index(init_time_end)
                        else:
                            # this set of initalizations have crossed years
                            # get the last element of the current file
                            # we have anthoer section that checks additional input data
                            i_init_end = len(ds_time_list) - 1

                        info.append([i_file, i_init_start, i_init_end, N_times])
        return info

    def initialize_batch(self):
        """
        Initializes batch indices using DistributedSampler's indices.
        Ensures proper cycling when shuffle=False.
        """
        # Initialize the call count if not already present
        if not hasattr(self, "batch_call_count"):
            self.batch_call_count = 0

        indices = self.batch_indices_splits[self.batch_call_count]
        # Assign batch indices
        self.current_batch_indices = list(indices)  # this will be the local indices used in getitem
        self.time_steps = [0] * len(self.current_batch_indices)
        self.forecast_step_counts = [0] * len(self.current_batch_indices)

        self.batch_call_count += 1
        if self.batch_call_count == len(self.batch_indices_splits):
            self.batch_call_count = 0

    def batches_per_epoch(self):
        return len(self.batch_indices_splits)  # math.ceil(len(self) / self.batch_size)

    def __getitem__(self, _):
        batch = {}

        # logger.info(f"rank={self.rank} current_idx={self.current_batch_indices} all_indices={self.batch_indices}")
        if self.forecast_step_counts[0] == self.forecast_period:
            self.initialize_batch()

        if self.forecast_step_counts[0] == 0:
            self.data_lookup = [self.find_start_stop_indices(idx) for idx in self.current_batch_indices]
            mode = "input"
        else:
            mode = "forcing"

        for k, idx in enumerate(self.current_batch_indices):
            # Get data for current timestep
            current_t = self.time_steps[k]
            i_file, i_init_start, i_init_end, N_times = self.data_lookup[k][current_t]

            # Load input data
            sliced_x = self.load_zarr_as_input(i_file, i_init_start, i_init_end, mode=mode)
            # Handle cross-file data if needed
            if (len(sliced_x["time"]) < self.history_len) or (i_init_end + 1 >= N_times):
                next_file_idx = self.filenames.index(self.filenames[i_file]) + 1
                if next_file_idx >= len(self.filenames):
                    raise OSError("End of available data reached.")
                # Input data
                sliced_x_next = self.load_zarr_as_input(next_file_idx, 0, self.history_len, mode=mode)
                sliced_x = xr.concat([sliced_x, sliced_x_next], dim="time").isel(time=slice(0, self.history_len))
                # Truth data
                if not self.skip_target:
                    sliced_y = self.load_zarr_as_input(i_file, i_init_end, i_init_end, mode="target")
                    sliced_y_next = self.load_zarr_as_input(next_file_idx, 0, 1, mode="target")
                    sliced_y = xr.concat([sliced_y, sliced_y_next], dim="time").isel(
                        time=slice(self.history_len, self.history_len + 1)
                    )
            elif not self.skip_target:
                sliced_y = self.load_zarr_as_input(i_file, i_init_end + 1, i_init_end + 1, mode="target")

            # Transform data
            sample = {"historical_ERA5_images": sliced_x}
            if not self.skip_target:
                sample["target_ERA5_images"] = sliced_y
            if self.transform:
                sample = self.transform(sample)

            # Add metadata
            sample["index"] = idx + current_t
            sample["datetime"] = sliced_x.time.values.astype("datetime64[s]").astype(int)[-1]

            # Convert and add to batch
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    value = torch.tensor(value)
                elif isinstance(value, np.int64):
                    value = torch.tensor(value, dtype=torch.int64)
                elif isinstance(value, (int, float)):
                    value = torch.tensor(value, dtype=torch.float32)
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)

                if value.ndimension() == 0:
                    value = value.unsqueeze(0)

                if value.ndim in (4, 5):
                    value = value.unsqueeze(0)

                if key not in batch:
                    batch[key] = value
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)

            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = torch.tensor([self.forecast_step_counts[0]])
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_period

        return batch


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py [1|2|3]")
        sys.exit(1)

    option = sys.argv[1]

    import logging

    import torch
    import yaml
    from torch.utils.data import DataLoader

    from credit.datasets import set_globals, setup_data_loading
    from credit.parser import credit_main_parser, training_data_check
    from credit.transforms import load_transforms

    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)  # Create a logger with the module name

    with open("/glade/derecho/scratch/schreck/repos/miles-credit/production/multistep/wxformer_6h/model.yml") as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)
    training_data_check(conf, print_summary=False)
    data_config = setup_data_loading(conf)

    epoch = 0
    batch_size = 5
    data_config["forecast_len"] = 6
    data_config["history_len"] = 1
    shuffle = True

    rank = 0
    world_size = 2
    num_workers = 4

    set_globals(data_config, namespace=globals())

    # globals().update(data_config)
    # for key, value in data_config.items():
    #     globals()[key] = value
    #     logger.info(f"Creating global variable in the namespace: {key}")

    def collate_fn(batch):
        # Only used with ERA5_MultiStep_Batcher
        # Prevents time and batch dimension from getting flipped
        return batch[0]

    ##########
    # Option 1
    ##########

    if option == "1":
        logger.info("Option 1: ERA5_MultiStep_Batcher")
        logger.info(
            """
            dataset_multi = ERA5_MultiStep_Batcher(
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
                sst_forcing=data_config['sst_forcing'],
                batch_size=batch_size,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size
            )

            dataloader = DataLoader(
                dataset_multi,
                num_workers=1,  # Must be 1 to use prefetching
                prefetch_factor=4,
                collate_fn=collate_fn  # Strip the batch_dim added by Dataloader. Batch is handled in dataset.
            )
            """
        )
        start_time = time.time()
        dataset_multi = ERA5_MultiStep_Batcher(
            varname_upper_air=data_config["varname_upper_air"],
            varname_surface=data_config["varname_surface"],
            varname_dyn_forcing=data_config["varname_dyn_forcing"],
            varname_forcing=data_config["varname_forcing"],
            varname_static=data_config["varname_static"],
            varname_diagnostic=data_config["varname_diagnostic"],
            filenames=data_config["all_ERA_files"],
            filename_surface=data_config["surface_files"],
            filename_dyn_forcing=data_config["dyn_forcing_files"],
            filename_forcing=data_config["forcing_files"],
            filename_static=data_config["static_files"],
            filename_diagnostic=data_config["diagnostic_files"],
            sst_forcing=data_config["sst_forcing"],
            history_len=data_config["history_len"],
            forecast_len=data_config["forecast_len"],
            skip_periods=data_config["skip_periods"],
            max_forecast_len=data_config["max_forecast_len"],
            transform=load_transforms(conf),
            batch_size=batch_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
        )

        dataloader = DataLoader(
            dataset_multi,
            num_workers=1,  # Must be 1 to use prefetching
            prefetch_factor=4,
            collate_fn=collate_fn,
        )

    ##########
    # Option 2
    ##########

    elif option == "2":
        logger.info("Testing 2: MultiprocessingBatcher")
        logger.info(
            """
            dataset_multi = MultiprocessingBatcher(
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
                sst_forcing=data_config['sst_forcing'],
                batch_size=batch_size,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
                num_workers=4
            )

            dataloader = DataLoader(
                dataset_multi,
                num_workers=0,  # Cannot use multiprocessing in both
                collate_fn=collate_fn  # Strip the batch_dim added by Dataloader. Batch is handled in dataset.
            )
            """
        )

        start_time = time.time()
        dataset_multi = MultiprocessingBatcher(
            varname_upper_air=data_config["varname_upper_air"],
            varname_surface=data_config["varname_surface"],
            varname_dyn_forcing=data_config["varname_dyn_forcing"],
            varname_forcing=data_config["varname_forcing"],
            varname_static=data_config["varname_static"],
            varname_diagnostic=data_config["varname_diagnostic"],
            filenames=data_config["all_ERA_files"],
            filename_surface=data_config["surface_files"],
            filename_dyn_forcing=data_config["dyn_forcing_files"],
            filename_forcing=data_config["forcing_files"],
            filename_static=data_config["static_files"],
            filename_diagnostic=data_config["diagnostic_files"],
            sst_forcing=data_config["sst_forcing"],
            history_len=data_config["history_len"],
            forecast_len=data_config["forecast_len"],
            skip_periods=data_config["skip_periods"],
            max_forecast_len=data_config["max_forecast_len"],
            transform=load_transforms(conf),
            batch_size=batch_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
        )

        dataloader = DataLoader(
            dataset_multi,
            num_workers=0,  # Cannot use multiprocessing in both
            collate_fn=collate_fn,
        )

    ##########
    # Option 3
    ##########

    elif option == "3":
        logger.info("Testing 3: MultiprocessingBatcherPrefetch")
        logger.info(
            """
            dataset_multi = MultiprocessingBatcherPrefetch(
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
                sst_forcing=data_config['sst_forcing'],
                history_len=data_config['history_len'],
                forecast_len=data_config['forecast_len'],
                skip_periods=data_config['skip_periods'],
                max_forecast_len=data_config['max_forecast_len'],
                transform=load_transforms(conf),
                batch_size=batch_size,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
                num_workers=6,
                prefetch_factor=6,
            )

            dataloader = DataLoader(
                dataset_multi,
                collate_fn=collate_fn  # Strip the batch_dim added by Dataloader. Batch is handled in dataset.
            )
            """
        )

        start_time = time.time()
        dataset_multi = MultiprocessingBatcherPrefetch(
            varname_upper_air=data_config["varname_upper_air"],
            varname_surface=data_config["varname_surface"],
            varname_dyn_forcing=data_config["varname_dyn_forcing"],
            varname_forcing=data_config["varname_forcing"],
            varname_static=data_config["varname_static"],
            varname_diagnostic=data_config["varname_diagnostic"],
            filenames=data_config["all_ERA_files"],
            filename_surface=data_config["surface_files"],
            filename_dyn_forcing=data_config["dyn_forcing_files"],
            filename_forcing=data_config["forcing_files"],
            filename_static=data_config["static_files"],
            filename_diagnostic=data_config["diagnostic_files"],
            sst_forcing=data_config["sst_forcing"],
            history_len=data_config["history_len"],
            forecast_len=data_config["forecast_len"],
            skip_periods=data_config["skip_periods"],
            max_forecast_len=data_config["max_forecast_len"],
            transform=load_transforms(conf),
            batch_size=batch_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
            num_workers=6,
            prefetch_factor=6,
        )

        dataloader = DataLoader(dataset_multi, collate_fn=collate_fn)

    else:
        print(f"Invalid option: {option}. Please choose 1, 2, or 3.")
        sys.exit(1)

    dataloader.dataset.set_epoch(epoch)
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

    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Log the elapsed time
    logger.info(f"Elapsed time for fetching 20 batches: {elapsed_time:.2f} seconds")
