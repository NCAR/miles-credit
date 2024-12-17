import logging
import time
import queue
import multiprocessing
from threading import Thread
from queue import Queue

import numpy as np
import torch
from functools import partial
from torch.utils.data import DistributedSampler

from credit.data import drop_var_from_dataset, get_forward_data
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
        history_len=2,
        forecast_len=0,
        transform=None,
        seed=42,
        rank=0,
        world_size=1,
        skip_periods=None,
        max_forecast_len=None,
        batch_size=1,
        shuffle=True
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
        self.rng = np.random.default_rng(seed=seed+rank)

        # max possible forecast len
        self.max_forecast_len = max_forecast_len

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
        if filename_surface is not None:
            surface_files = []
            filename_surface = sorted(filename_surface)

            if filenames == filename_surface:
                flag_share_surf = True
            else:
                for fn in filename_surface:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_surf = drop_var_from_dataset(ds, varname_surface)
                    surface_files.append(ds_surf)

                self.surface_files = surface_files
        else:
            self.surface_files = False

        # dynamic forcing
        if filename_dyn_forcing is not None:
            dyn_forcing_files = []
            filename_dyn_forcing = sorted(filename_dyn_forcing)

            if filenames == filename_dyn_forcing:
                flag_share_dyn = True
            else:
                for fn in filename_dyn_forcing:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_dyn = drop_var_from_dataset(ds, varname_dyn_forcing)
                    dyn_forcing_files.append(ds_dyn)

                self.dyn_forcing_files = dyn_forcing_files
        else:
            self.dyn_forcing_files = False

        # diagnostics
        if filename_diagnostic is not None:
            diagnostic_files = []
            filename_diagnostic = sorted(filename_diagnostic)

            if filenames == filename_diagnostic:
                flag_share_diag = True
            else:
                for fn in filename_diagnostic:
                    # drop variables if they are not in the config
                    ds = get_forward_data(filename=fn)
                    ds_diag = drop_var_from_dataset(ds, varname_diagnostic)
                    diagnostic_files.append(ds_diag)

                self.diagnostic_files = diagnostic_files
        else:
            self.diagnostic_files = False

        # ------------------------------------------------------------------ #
        # blocks that can handle file sharing (share with upper air file)
        for fn in filenames:
            # drop variables if they are not in the config
            ds = get_forward_data(filename=fn)
            ds_upper = drop_var_from_dataset(ds, varname_upper_air)

            if flag_share_surf:
                ds_surf = drop_var_from_dataset(ds, varname_surface)
                surface_files.append(ds_surf)

            if flag_share_dyn:
                ds_dyn = drop_var_from_dataset(ds, varname_dyn_forcing)
                dyn_forcing_files.append(ds_dyn)

            if flag_share_diag:
                ds_diag = drop_var_from_dataset(ds, varname_diagnostic)
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
                len(ERA5_xarray["time"]),
                ind_start,
                ind_start + len(ERA5_xarray["time"]),
            ]
            ind_start += len(ERA5_xarray["time"]) + 1

        # ======================================================== #
        # forcing file
        self.filename_forcing = filename_forcing

        if self.filename_forcing is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_forcing)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_forcing)

            self.xarray_forcing = xarray_dataset
        else:
            self.xarray_forcing = False

        # ======================================================== #
        # static file
        self.filename_static = filename_static

        if self.filename_static is not None:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_static)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_static)

            self.xarray_static = xarray_dataset
        else:
            self.xarray_static = False

        self.worker = partial(
            worker,
            ERA5_indices=self.ERA5_indices,
            all_files=self.all_files,
            surface_files=self.surface_files,
            dyn_forcing_files=self.dyn_forcing_files,
            diagnostic_files=self.diagnostic_files,
            xarray_forcing=self.xarray_forcing,
            xarray_static=self.xarray_static,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
            skip_periods=self.skip_periods,
            transform=self.transform,
        )

        self.total_length = len(self.ERA5_indices)
        self.current_epoch = None
        self.current_index = None
        self.initial_index = None

        # Use DistributedSampler for index management
        self.sampler = DistributedSampler(
            self,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )

        # Initialize state variables for batch management
        self.batch_size = batch_size
        self.batch_indices = None  # To track initial indices for each batch item
        self.time_steps = None  # Tracks time steps for each batch index
        self.forecast_step_counts = None  # Track forecast step counts for each batch item
        self.initial_indices = None  # Tracks the initial index for each forecast item in the batch

        # Initialize batch once when the dataset is created
        self.initialize_batch()

    # def initialize_batch(self):
    #     """
    #     Initializes batch indices using DistributedSampler's indices.
    #     Resets the time steps and forecast step counts.
    #     Ensures proper cycling when shuffle=False.
    #     """
    #     # Initialize the call count if not already present
    #     if not hasattr(self, "batch_call_count"):
    #         self.batch_call_count = 0

    #     # Set epoch for DistributedSampler to ensure consistent shuffling across devices
    #     if self.current_epoch is not None:
    #         self.sampler.set_epoch(self.current_epoch)

    #     # Retrieve indices for this GPU
    #     indices = list(self.sampler)
    #     total_indices = len(indices)

    #     # Select batch indices based on call count (deterministic cycling)
    #     start = self.batch_call_count * self.batch_size
    #     end = start + self.batch_size

    #     if end > total_indices:
    #         # Wrap-around to ensure no index is skipped
    #         indices = indices[start:] + indices[:(end % total_indices)]
    #     else:
    #         indices = indices[start:end]

    #     # Increment batch_call_count, reset when all indices are cycled
    #     self.batch_call_count += 1
    #     if start + self.batch_size >= total_indices:
    #         self.batch_call_count = 0  # Reset for next cycle

    #     # Assign batch indices
    #     self.batch_indices = indices
    #     self.time_steps = [0 for _ in self.batch_indices]
    #     self.forecast_step_counts = [0 for _ in self.batch_indices]
    #     self.initial_indices = list(self.batch_indices)

    def initialize_batch(self):
        """
        Initializes batch indices using DistributedSampler's indices.
        Ensures proper cycling when shuffle=False.
        """
        # Initialize the call count if not already present
        if not hasattr(self, "batch_call_count"):
            self.batch_call_count = 0

        # Set epoch for DistributedSampler to ensure consistent shuffling across devices
        if self.current_epoch is not None:
            self.sampler.set_epoch(self.current_epoch)

        # Retrieve indices for this GPU
        indices = list(self.sampler)
        total_indices = len(indices)

        # Select batch indices based on call count (deterministic cycling)
        start = self.batch_call_count * self.batch_size
        end = start + self.batch_size

        if not self.shuffle:
            if end > total_indices:
                # Simple wraparound by incrementing start index
                start = start % total_indices
                end = min(start + self.batch_size, total_indices)
            indices = indices[start:end]
        else:
            if end > total_indices:
                # Wrap-around to ensure no index is skipped
                indices = indices[start:] + indices[:(end % total_indices)]
            else:
                indices = indices[start:end]

        # Increment batch_call_count, reset when all indices are cycled
        self.batch_call_count += 1
        if start + self.batch_size >= total_indices:
            self.batch_call_count = 0  # Reset for next cycle

        # Assign batch indices
        self.batch_indices = indices
        self.time_steps = [0 for _ in self.batch_indices]
        self.forecast_step_counts = [0 for _ in self.batch_indices]
        self.initial_indices = list(self.batch_indices)

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += len(ERA5_xarray["time"]) - self.total_seq_len + 1
        return total_len

    # def set_epoch(self, epoch):
    #     self.current_epoch = epoch
    #     self.current_index = None
    #     self.initial_index = None

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.sampler.set_epoch(epoch)

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

        for k, idx in enumerate(self.batch_indices):
            # Get the current time step for this batch item
            current_t = self.time_steps[k]
            initial_idx = self.initial_indices[k]  # Correctly find the initial index
            index_pair = (initial_idx, idx + current_t)  # Correctly construct the index_pair

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

                if key not in batch:
                    batch[key] = value  # Initialize the key in the batch dictionary
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)  # Concatenate values along the batch dimension

            # Increment time steps and forecast step counts for this batch item
            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = self.forecast_step_counts[0]
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len + 1
        batch["datetime"] = batch["datetime"].view(-1, self.batch_size)  # reshape

        return batch


# This version does not control the num_workers. Kept here for now for benchmarking, remove it soon.
# class MultiprocessingBatcher(ERA5_MultiStep_Batcher):

#     def __init__(self, *args, num_workers=4, **kwargs):
#         """
#         Initialize the MultiprocessingBatcher with a configurable number of workers.

#         Args:
#             num_workers (int): Number of workers to use for multiprocessing.
#             *args, **kwargs: Arguments passed to the parent class.
#         """
#         super().__init__(*args, **kwargs)  # Initialize the parent class
#         self.num_workers = num_workers

#     def __getitem__(self, _):
#         """
#         Fetches the current forecast step data for each item in the batch.
#         Utilizes multiprocessing to parallelize calls to self.worker.
#         Ensures the results are returned in the correct order.
#         """
#         batch = {}

#         # Reset items if forecast step count exceeds forecast length
#         if self.forecast_step_counts[0] == self.forecast_len + 1:
#             self.initialize_batch()

#         # Shared dictionary to collect results from multiple processes (keyed by index)
#         manager = multiprocessing.Manager()
#         results = manager.dict()

#         # Prepare arguments for processing
#         args = []
#         for k, idx in enumerate(self.batch_indices):
#             current_t = self.time_steps[k]
#             initial_idx = self.initial_indices[k]
#             index_pair = (initial_idx, idx + current_t)
#             args.append((k, index_pair, results))

#         # Define a function for processing each worker call
#         def worker_process(k, index_pair, result_dict):
#             sample = self.worker(index_pair)
#             sample["index"] = index_pair[1]  # Add index to the sample
#             result_dict[k] = sample  # Store the result keyed by its order index

#         # Split the tasks among the available workers
#         processes = []
#         for arg in args:
#             p = multiprocessing.Process(target=worker_process, args=arg)
#             processes.append(p)
#             p.start()

#         # Wait for all processes to finish
#         for p in processes:
#             p.join()

#         # Sort results by their original order
#         ordered_results = [results[k] for k in sorted(results.keys())]

#         # Process sorted results and build the batch
#         for k, sample in enumerate(ordered_results):
#             for key, value in sample.items():
#                 if isinstance(value, np.ndarray):
#                     value = torch.tensor(value)
#                 elif isinstance(value, np.int64):
#                     value = torch.tensor(value, dtype=torch.int64)
#                 elif isinstance(value, (int, float)):
#                     value = torch.tensor(value, dtype=torch.float32)
#                 elif not isinstance(value, torch.Tensor):
#                     value = torch.tensor(value)

#                 if value.ndimension() == 0:
#                     value = value.unsqueeze(0)

#                 if key not in batch:
#                     batch[key] = value
#                 else:
#                     batch[key] = torch.cat((batch[key], value), dim=0)

#             # Increment time steps and forecast step counts for this batch item
#             self.time_steps[k] += 1
#             self.forecast_step_counts[k] += 1

#         batch["forecast_step"] = self.forecast_step_counts[0]
#         batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len + 1
#         batch["datetime"] = batch["datetime"].view(-1, self.batch_size)

#         return batch


class MultiprocessingBatcher(ERA5_MultiStep_Batcher):

    def __init__(self, *args, num_workers=4, **kwargs):
        """
        Initialize the MultiprocessingBatcher with a configurable number of workers.

        Args:
            num_workers (int): Number of workers to use for multiprocessing.
            *args, **kwargs: Arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)  # Initialize the parent class
        self.num_workers = num_workers

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

        # Shared dictionary to collect results from multiple processes (keyed by index)
        manager = multiprocessing.Manager()
        results = manager.dict()

        # Prepare arguments for processing
        args = []
        for k, idx in enumerate(self.batch_indices):
            current_t = self.time_steps[k]
            initial_idx = self.initial_indices[k]
            index_pair = (initial_idx, idx + current_t)
            args.append((k, index_pair, results))

        # Worker function
        def worker_process(start_idx, end_idx):
            for i in range(start_idx, end_idx):
                k, index_pair = args[i][:2]
                sample = self.worker(index_pair)
                sample["index"] = index_pair[1]  # Add index to the sample
                results[k] = sample  # Store the result keyed by its order index

        # Split tasks among workers
        tasks_per_worker = max(1, len(args) // self.num_workers)
        processes = []

        for i in range(self.num_workers):
            start_idx = i * tasks_per_worker
            end_idx = min((i + 1) * tasks_per_worker, len(args))
            if start_idx < end_idx:  # Check if there are tasks for this worker
                p = multiprocessing.Process(target=worker_process, args=(start_idx, end_idx))
                processes.append(p)
                p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Sort results by their original order
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

                if key not in batch:
                    batch[key] = value
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)

            # Increment time steps and forecast step counts for this batch item
            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = self.forecast_step_counts[0]
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len + 1
        batch["datetime"] = batch["datetime"].view(-1, self.batch_size)

        return batch


class MultiprocessingBatcherPrefetch(ERA5_MultiStep_Batcher):
    def __init__(self, *args, num_workers=4, prefetch_factor=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.prefetch_queue = Queue(maxsize=prefetch_factor)
        self.stop_signal = multiprocessing.Event()

        # Manager for shared resources
        self.manager = multiprocessing.Manager()
        self.results = self.manager.dict()

        # Start prefetching thread
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
            logger.warning(f"Ignoring transient connection error for index {k}.")
        except Exception as e:
            logger.error(f"Error in worker process for index {k}: {e}")
            raise RuntimeError(f"Error in worker process for index {k}: {e}") from e

    def _fetch_batch(self):
        """
        Fetches a batch using multiprocessing workers and splits the work efficiently.
        """
        batch = {}

        # Reset items if forecast step count exceeds forecast length
        if self.forecast_step_counts[0] == self.forecast_len + 1:
            self.initialize_batch()

        # Prepare arguments for processing
        tasks = []
        for k, idx in enumerate(self.batch_indices):
            current_t = self.time_steps[k]
            initial_idx = self.initial_indices[k]
            index_pair = (initial_idx, idx + current_t)
            tasks.append((k, index_pair))

        # Split tasks among workers (efficient chunking)
        chunk_size = max(1, len(tasks) // self.num_workers)
        task_chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

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

                if key not in batch:
                    batch[key] = value
                else:
                    batch[key] = torch.cat((batch[key], value), dim=0)

            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = self.forecast_step_counts[0]
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len + 1
        batch["datetime"] = batch["datetime"].view(-1, self.batch_size)

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
            if hasattr(self, 'processes'):
                for p in self.processes:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=1)

            # Close and remove any open managers
            if hasattr(self, 'manager'):
                self.manager.shutdown()
                del self.manager

            # Clear queues
            if hasattr(self, 'prefetch_queue'):
                while not self.prefetch_queue.empty():
                    try:
                        self.prefetch_queue.get_nowait()
                    except queue.Empty:
                        break

            # Stop threads and events
            if hasattr(self, 'stop_signal'):
                self.stop_signal.set()

            if hasattr(self, 'prefetch_thread'):
                self.prefetch_thread.join(timeout=2)

        except Exception as e:
            print(f"Cleanup error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()


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
    from credit.transforms import load_transforms
    from credit.parser import credit_main_parser, training_data_check
    from credit.datasets import setup_data_loading, set_globals

    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)  # Create a logger with the module name

    with open(
        "/glade/derecho/scratch/schreck/repos/miles-credit/production/multistep/wxformer_6h/model.yml"
    ) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)
    data_config = setup_data_loading(conf)

    batch_size = 2
    data_config["forecast_len"] = 6
    rank = 0
    world_size = 2
    shuffle = True

    set_globals(data_config, namespace=globals())

    # globals().update(data_config)
    # for key, value in data_config.items():
    #     globals()[key] = value
    #     logger.info(f"Creating global variable in the namespace: {key}")

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
                batch_size=batch_size,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size
            )

            dataloader = DataLoader(
                dataset_multi,
                num_workers=1,  # Must be 1 to use prefetching
                drop_last=True,  # Drop the last incomplete batch if not divisible by batch_size,
                prefetch_factor=4
            )
            """
        )
        start_time = time.time()
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
            batch_size=batch_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size
        )

        dataloader = DataLoader(
            dataset_multi,
            num_workers=1,  # Must be 1 to use prefetching
            drop_last=True,  # Drop the last incomplete batch if not divisible by batch_size,
            prefetch_factor=4
        )

        dataloader.dataset.set_epoch(0)
        for (k, sample) in enumerate(dataloader):
            print(k, sample['index'], sample['datetime'], sample['forecast_step'], sample['stop_forecast'])
            if k == 20:
                break

        # End the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Log the elapsed time
        logger.info(f"Elapsed time for fetching 20 batches: {elapsed_time:.2f} seconds")

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
                batch_size=batch_size,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
                num_workers=4
            )

            dataloader = DataLoader(
                dataset_multi,
                num_workers=0,  # Cannot use multiprocessing in both
                drop_last=True  # Drop the last incomplete batch if not divisible by batch_size
            )
            """
        )

        start_time = time.time()
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
            batch_size=batch_size,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
            num_workers=4
        )

        dataloader = DataLoader(
            dataset_multi,
            num_workers=0,  # Cannot use multiprocessing in both
            drop_last=True  # Drop the last incomplete batch if not divisible by batch_size
        )

        dataloader.dataset.set_epoch(0)
        for (k, sample) in enumerate(dataloader):
            print(k, sample['index'], sample['datetime'], sample['forecast_step'], sample['stop_forecast'])
            if k == 20:
                break

        # End the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Log the elapsed time
        logger.info(f"Elapsed time for fetching 20 batches: {elapsed_time:.2f} seconds")

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
                drop_last=True,  # Drop the last incomplete batch if not divisible by batch_size,
            )
            """
        )

        start_time = time.time()
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
            prefetch_factor=6
        )

        dataloader = DataLoader(
            dataset_multi,
            drop_last=True,  # Drop the last incomplete batch if not divisible by batch_size,
        )

        dataloader.dataset.set_epoch(0)
        for (k, sample) in enumerate(dataloader):
            print(k, sample['index'], sample['datetime'], sample['forecast_step'], sample['stop_forecast'])
            if k == 20:
                break

        # End the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Log the elapsed time
        logger.info(f"Elapsed time for fetching 20 batches: {elapsed_time:.2f} seconds")

    else:
        print(f"Invalid option: {option}. Please choose 1, 2, or 3.")
        sys.exit(1)
