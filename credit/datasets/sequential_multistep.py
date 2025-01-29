"""
Pytorch IterableDataset for multi-step training

Reference:
    Non-daemonic Python pool process
    https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic

    Pytorch Iterable Dataset
    https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-?usp=sharing#scrollTo=CxSCQPOMHgwo
"""

import os
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

from credit.data import (
    Sample,
    find_key_for_number,
    get_forward_data,
    drop_var_from_dataset,
    extract_month_day_hour,
    find_common_indices,
)

logger = logging.getLogger(__name__)


class DistributedSequentialDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        varname_upper_air: List[str],
        varname_surface: List[str],
        varname_dyn_forcing: List[str],
        varname_forcing: List[str],
        varname_static: List[str],
        varname_diagnostic: List[str],
        filenames: List[str],
        filename_surface: Optional[List[str]] = None,
        filename_dyn_forcing: Optional[List[str]] = None,
        filename_forcing: Optional[str] = None,
        filename_static: Optional[str] = None,
        filename_diagnostic: Optional[List[str]] = None,
        rank: int = 0,
        world_size: int = 1,
        history_len: int = 2,
        forecast_len: int = 0,
        transform: Optional[Callable] = None,
        seed: int = 42,
        skip_periods: Optional[int] = None,
        max_forecast_len: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        """
        Initialize the DistributedSequentialDatasetV2.

        Parameters:
        - varname_upper_air (list): List of upper air variable names.
        - varname_surface (list): List of surface variable names.
        - varname_dyn_forcing (list): List of dynamic forcing variable names.
        - varname_forcing (list): List of forcing variable names.
        - varname_static (list): List of static variable names.
        - varname_diagnostic (list): List of diagnostic variable names.
        - filenames (list): List of filenames for upper air data.
        - filename_surface (list, optional): List of filenames for surface data.
        - filename_dyn_forcing (list, optional): List of filenames for dynamic forcing.
        - filename_forcing (str, optional): Filename for forcing data.
        - filename_static (str, optional): Filename for static data.
        - filename_diagnostic (list, optional): List of filenames for diagnostic data.
        - rank (int, optional): Rank of the current process. Default is 0.
        - world_size (int, optional): Total number of processes. Default is 1.
        - history_len (int, optional): Length of the history sequence. Default is 2.
        - forecast_len (int, optional): Length of the forecast sequence. Default is 0.
        - transform (callable, optional): Transformation function to apply to the data.
        - seed (int, optional): Random seed for reproducibility. Default is 42.
        - skip_periods (int, optional): Number of periods to skip between samples.
        - max_forecast_len (int, optional): Maximum length of the forecast sequence.
        - shuffle (bool, optional): Whether to shuffle the data. Default is True.
        - num_workers (int, optional): Number of worker processes. Default is 0.

        Returns:
        - sample (dict): A dictionary containing historical ERA5 images, target ERA5 images, datetime index, and additional information.
        """

        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.current_epoch = 0
        self.num_workers = num_workers

        logger.info(f"Using {num_workers} workers in the iterable dataset")

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
        ## surface
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

        ## dynamic forcing
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

        ## diagnostics
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
            assert os.path.isfile(
                filename_forcing
            ), "Cannot find forcing file [{}]".format(filename_forcing)

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
            assert os.path.isfile(
                filename_static
            ), "Cannot find static file [{}]".format(filename_static)

            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename_static)
            xarray_dataset = drop_var_from_dataset(xarray_dataset, varname_static)

            self.xarray_static = xarray_dataset

        else:
            self.xarray_static = False

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self) -> int:
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += len(ERA5_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def __iter__(self):
        # ------------------------------------------------------------------- #
        # get worker info
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # distributed sampler with worker info
        sampler = DistributedSampler(
            self,
            num_replicas=num_workers * self.world_size,
            rank=self.rank * num_workers + worker_id,
            shuffle=self.shuffle,
        )
        sampler.set_epoch(self.current_epoch)

        # ------------------------------------------------------------------- #
        # worker process
        process_index_partial = partial(
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

        # Dont use multi-processing
        if self.num_workers <= 1:
            for index in iter(sampler):
                # Explicit inner (time step) loop
                indices = list(
                    range(index, index + self.history_len + self.forecast_len)
                )
                for ind_start_current_step in indices:
                    sample = process_index_partial((index, ind_start_current_step))
                    yield sample
                    if sample["stop_forecast"]:
                        break

        else:  # use multi-processing
            with Pool(self.num_workers) as p:
                batch_size = 2 * self.num_workers  # limit the size of the "queue"
                for index in iter(sampler):
                    indices = list(
                        range(index, index + self.history_len + self.forecast_len)
                    )

                    # Process indices in batches to avoid potential memory problems if indices is very long
                    for i in range(0, len(indices), batch_size):
                        batch_indices = indices[i : i + batch_size]
                        batch_tasks = [
                            (index, ind_start_current_step)
                            for ind_start_current_step in batch_indices
                        ]

                        # Process the batch
                        batch_results = p.map(process_index_partial, batch_tasks)

                        # Yield results from the batch
                        for sample in batch_results:
                            yield sample
                            if sample["stop_forecast"]:
                                return


def worker(
    tuple_index: Tuple[int, int],
    ERA5_indices: Dict[str, List[int]],
    all_files: List[Any],
    surface_files: Optional[List[Any]],
    dyn_forcing_files: Optional[List[Any]],
    diagnostic_files: Optional[List[Any]],
    xarray_forcing: Optional[Any],
    xarray_static: Optional[Any],
    history_len: int,
    forecast_len: int,
    skip_periods: int,
    transform: Optional[Callable],
) -> Dict[str, Any]:
    """
    Processes a given index to extract and transform data for a specific time slice.

    Parameters:
    - tuple_index (Tuple[int, int]): Tuple containing the current index and sub-index for processing.
    - ERA5_indices (Dict[str, List[int]]): Dictionary containing ERA5 indices metadata.
    - all_files (List[Any]): List of xarray datasets containing upper air data.
    - surface_files (Optional[List[Any]]): List of xarray datasets containing surface data.
    - dyn_forcing_files (Optional[List[Any]]): List of xarray datasets containing dynamic forcing data.
    - diagnostic_files (Optional[List[Any]]): List of xarray datasets containing diagnostic data.
    - history_len (int): Length of the history sequence.
    - forecast_len (int): Length of the forecast sequence.
    - skip_periods (int): Number of periods to skip between samples.
    - xarray_forcing (Optional[Any]): xarray dataset containing forcing data.
    - xarray_static (Optional[Any]): xarray dataset containing static data.

    - transform (Optional[Callable]): Transformation function to apply to the data.

    Returns:
    - Dict[str, Any]: A dictionary containing historical ERA5 images, target ERA5 images, datetime index, and additional information.
    """

    index, ind_start_current_step = tuple_index

    try:
        # select the ind_file based on the iter index
        ind_file = find_key_for_number(ind_start_current_step, ERA5_indices)

        # get the ind within the current file
        ind_start = ERA5_indices[ind_file][1]
        ind_start_in_file = ind_start_current_step - ind_start

        # handle out-of-bounds
        ind_largest = len(all_files[int(ind_file)]["time"]) - (
            history_len + forecast_len + 1
        )
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        # ========================================================================== #
        # subset xarray on time dimension & load it to the memory

        ind_end_in_file = ind_start_in_file + history_len + forecast_len

        # ERA5_subset: a xarray dataset that contains training input and target (for the current batch)
        ERA5_subset = all_files[int(ind_file)].isel(
            time=slice(ind_start_in_file, ind_end_in_file + 1)
        )

        if surface_files:
            # subset surface variables
            surface_subset = surface_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )

            # merge upper-air and surface here:
            ERA5_subset = ERA5_subset.merge(
                surface_subset
            )  # <-- lazy merge, ERA5 and surface both not loaded

        # ==================================================== #
        # split ERA5_subset into training inputs and targets
        #   + merge with forcing and static

        # the ind_end of the ERA5_subset
        # ind_end_time = len(ERA5_subset['time'])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = ERA5_subset.time.values.astype("datetime64[s]").astype(int)

        # ==================================================== #
        # xarray dataset as input
        # historical_ERA5_images: the final input

        historical_ERA5_images = ERA5_subset.isel(
            time=slice(0, history_len, skip_periods)
        ).load()

        # ========================================================================== #
        # merge dynamic forcing inputs
        if dyn_forcing_files:
            dyn_forcing_subset = dyn_forcing_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(
                time=slice(0, history_len, skip_periods)
            ).load()

            historical_ERA5_images = historical_ERA5_images.merge(dyn_forcing_subset)

        # ========================================================================== #
        # merge forcing inputs
        if xarray_forcing:
            # =============================================================================== #
            # matching month, day, hour between forcing and upper air [time]
            # this approach handles leap year forcing file and non-leap-year upper air file
            month_day_forcing = extract_month_day_hour(np.array(xarray_forcing["time"]))
            month_day_inputs = extract_month_day_hour(
                np.array(historical_ERA5_images["time"])
            )
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = xarray_forcing.isel(time=ind_forcing).load()
            # forcing and upper air have different years but the same mon/day/hour
            # safely replace forcing time with upper air time
            forcing_subset_input["time"] = historical_ERA5_images["time"]
            # =============================================================================== #

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(forcing_subset_input)

        # ========================================================================== #
        # merge static inputs
        if xarray_static:
            # expand static var on time dim
            N_time_dims = len(ERA5_subset["time"])
            static_subset_input = xarray_static.expand_dims(dim={"time": N_time_dims})
            # assign coords 'time'
            static_subset_input = static_subset_input.assign_coords(
                {"time": ERA5_subset["time"]}
            )

            # slice + load to the GPU
            static_subset_input = static_subset_input.isel(
                time=slice(0, history_len, skip_periods)
            ).load()

            # update
            static_subset_input["time"] = historical_ERA5_images["time"]

            # merge
            historical_ERA5_images = historical_ERA5_images.merge(static_subset_input)

        # ==================================================== #
        # xarray dataset as target
        # target_ERA5_images: the final target

        # get the next forecast step
        target_ERA5_images = ERA5_subset.isel(
            time=slice(history_len, history_len + skip_periods, skip_periods)
        ).load()

        # merge diagnoisc input here:
        if diagnostic_files:
            # subset diagnostic variables
            diagnostic_subset = diagnostic_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )

            # get the next forecast step
            diagnostic_subset = diagnostic_subset.isel(
                time=slice(history_len, history_len + skip_periods, skip_periods)
            ).load()  # <-- load into memory

            # merge into the target dataset
            target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

        # create a dict object with input/output tensors
        sample = Sample(
            historical_ERA5_images=historical_ERA5_images,
            target_ERA5_images=target_ERA5_images,
            datetime_index=datetime_as_number,
        )

        # data normalization
        if transform:
            sample = transform(sample)

        sample["index"] = index
        stop_forecast = (ind_start_current_step - index) == forecast_len
        sample["forecast_step"] = ind_start_current_step - index + 1
        sample["index"] = index
        sample["stop_forecast"] = stop_forecast
        sample["datetime"] = [
            int(
                historical_ERA5_images.time.values[0]
                .astype("datetime64[s]")
                .astype(int)
            ),
            int(target_ERA5_images.time.values[0].astype("datetime64[s]").astype(int)),
        ]

    except Exception as e:
        logger.error(f"Error processing index {tuple_index}: {e}")
        raise

    return sample


class DistributedSequentialDatasetBasic(DistributedSequentialDataset):
    def __iter__(self):
        # ------------------------------------------------------------------- #
        # get worker info
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # distributed sampler with worker info
        sampler = DistributedSampler(
            self,
            num_replicas=num_workers * self.world_size,
            rank=self.rank * num_workers + worker_id,
            shuffle=self.shuffle,
        )
        sampler.set_epoch(self.current_epoch)

        for index in iter(sampler):
            indices = list(range(index, index + self.history_len + self.forecast_len))
            stop_forecast = False

            for k, ind_start_current_step in enumerate(indices):
                # select the ind_file based on the iter index
                ind_file = find_key_for_number(
                    ind_start_current_step, self.ERA5_indices
                )

                # get the ind within the current file
                ind_start = self.ERA5_indices[ind_file][1]
                ind_start_in_file = ind_start_current_step - ind_start

                # handle out-of-bounds
                ind_largest = len(self.all_files[int(ind_file)]["time"]) - (
                    self.history_len + self.forecast_len + 1
                )
                if ind_start_in_file > ind_largest:
                    ind_start_in_file = ind_largest
                # ========================================================================== #
                # subset xarray on time dimension & load it to the memory

                ind_end_in_file = (
                    ind_start_in_file + self.history_len + self.forecast_len
                )

                # ERA5_subset: a xarray dataset that contains training input and target (for the current batch)
                ERA5_subset = self.all_files[int(ind_file)].isel(
                    time=slice(ind_start_in_file, ind_end_in_file + 1)
                )

                if self.surface_files:
                    # subset surface variables
                    surface_subset = self.surface_files[int(ind_file)].isel(
                        time=slice(ind_start_in_file, ind_end_in_file + 1)
                    )

                    # merge upper-air and surface here:
                    ERA5_subset = ERA5_subset.merge(surface_subset)

                # ==================================================== #
                # split ERA5_subset into training inputs and targets + merge with forcing and static

                # the ind_end of the ERA5_subset
                # ind_end_time = len(ERA5_subset['time'])

                # datetiem information as int number (used in some normalization methods)
                datetime_as_number = ERA5_subset.time.values.astype(
                    "datetime64[s]"
                ).astype(int)

                # ==================================================== #
                # xarray dataset as input
                # historical_ERA5_images: the final input

                historical_ERA5_images = ERA5_subset.isel(
                    time=slice(0, self.history_len, self.skip_periods)
                ).load()

                # ========================================================================== #
                # merge dynamic forcing inputs
                if self.dyn_forcing_files:
                    dyn_forcing_subset = self.dyn_forcing_files[int(ind_file)].isel(
                        time=slice(ind_start_in_file, ind_end_in_file + 1)
                    )
                    dyn_forcing_subset = dyn_forcing_subset.isel(
                        time=slice(0, self.history_len, self.skip_periods)
                    ).load()

                    historical_ERA5_images = historical_ERA5_images.merge(
                        dyn_forcing_subset
                    )

                # ========================================================================== #
                # merge forcing inputs
                if self.xarray_forcing:
                    # =============================================================================== #
                    # matching month, day, hour between forcing and upper air [time]
                    # this approach handles leap year forcing file and non-leap-year upper air file
                    month_day_forcing = extract_month_day_hour(
                        np.array(self.xarray_forcing["time"])
                    )
                    month_day_inputs = extract_month_day_hour(
                        np.array(historical_ERA5_images["time"])
                    )  # <-- upper air
                    # indices to subset
                    ind_forcing, _ = find_common_indices(
                        month_day_forcing, month_day_inputs
                    )
                    forcing_subset_input = self.xarray_forcing.isel(
                        time=ind_forcing
                    ).load()
                    # forcing and upper air have different years but the same mon/day/hour
                    # safely replace forcing time with upper air time
                    forcing_subset_input["time"] = historical_ERA5_images["time"]
                    # =============================================================================== #

                    # merge
                    historical_ERA5_images = historical_ERA5_images.merge(
                        forcing_subset_input
                    )

                # ========================================================================== #
                # merge static inputs
                if self.xarray_static:
                    # expand static var on time dim
                    N_time_dims = len(ERA5_subset["time"])
                    static_subset_input = self.xarray_static.expand_dims(
                        dim={"time": N_time_dims}
                    )
                    # assign coords 'time'
                    static_subset_input = static_subset_input.assign_coords(
                        {"time": ERA5_subset["time"]}
                    )

                    # slice + load to the GPU
                    static_subset_input = static_subset_input.isel(
                        time=slice(0, self.history_len, self.skip_periods)
                    ).load()

                    # update
                    static_subset_input["time"] = historical_ERA5_images["time"]

                    # merge
                    historical_ERA5_images = historical_ERA5_images.merge(
                        static_subset_input
                    )

                # ==================================================== #
                # xarray dataset as target
                # target_ERA5_images: the final target

                # get the next forecast step
                target_ERA5_images = ERA5_subset.isel(
                    time=slice(
                        self.history_len,
                        self.history_len + self.skip_periods,
                        self.skip_periods,
                    )
                ).load()

                # merge diagnoisc input here:
                if self.diagnostic_files:
                    # subset diagnostic variables
                    diagnostic_subset = self.diagnostic_files[int(ind_file)].isel(
                        time=slice(ind_start_in_file, ind_end_in_file + 1)
                    )

                    # get the next forecast step
                    diagnostic_subset = diagnostic_subset.isel(
                        time=slice(
                            self.history_len,
                            self.history_len + self.skip_periods,
                            self.skip_periods,
                        )
                    ).load()

                    # merge into the target dataset
                    target_ERA5_images = target_ERA5_images.merge(diagnostic_subset)

                # pipe xarray datasets to the sampler
                sample = Sample(
                    historical_ERA5_images=historical_ERA5_images,
                    target_ERA5_images=target_ERA5_images,
                    datetime_index=datetime_as_number,
                )

                # ==================================== #
                # data normalization
                if self.transform:
                    sample = self.transform(sample)

                # assign sample index
                sample["index"] = index

                stop_forecast = k == self.forecast_len

                sample["forecast_step"] = k + 1
                sample["index"] = index
                sample["stop_forecast"] = stop_forecast
                sample["datetime"] = [
                    int(
                        historical_ERA5_images.time.values[0]
                        .astype("datetime64[s]")
                        .astype(int)
                    ),
                    int(
                        target_ERA5_images.time.values[0]
                        .astype("datetime64[s]")
                        .astype(int)
                    ),
                ]

                yield sample

                if stop_forecast:
                    break

                if k == self.forecast_len:
                    break
