import torch
import logging
import numpy as np
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
from credit.data import (
    drop_var_from_dataset,
    get_forward_data,
    Sample,
    find_key_for_number,
    extract_month_day_hour,
    find_common_indices,
)


logger = logging.getLogger(__name__)


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
    sst_forcing: Optional[Any] = None,
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
            ERA5_subset = ERA5_subset.merge(surface_subset)

        # ==================================================== #
        # split ERA5_subset into training inputs and targets
        #   + merge with forcing and static

        # the ind_end of the ERA5_subset
        # ind_end_time = len(ERA5_subset['time'])

        # datetiem information as int number (used in some normalization methods)
        datetime_as_number = ERA5_subset.time.astype("datetime64[s]").values.astype(int)

        # ==================================================== #
        # xarray dataset as input
        # historical_ERA5_images: the final input

        historical_ERA5_images = ERA5_subset.isel(
            time=slice(0, history_len, skip_periods)
        ).load()  # <-- load into memory

        # ========================================================================== #
        # merge dynamic forcing inputs
        if dyn_forcing_files:
            dyn_forcing_subset = dyn_forcing_files[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(
                time=slice(0, history_len, skip_periods)
            ).load()  # <-- load into memory

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
            )  # <-- upper air
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = xarray_forcing.isel(
                time=ind_forcing
            ).load()  # <-- load into memory
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
            ).load()  # <-- load into memory

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
        ).load()  # <-- load into memory

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

        # sst forcing operations
        if sst_forcing is not None:
            # get xr.dataset keys
            varname_skt = sst_forcing["varname_skt"]
            varname_ocean_mask = sst_forcing["varname_ocean_mask"]

            # get xr.dataarray from the dataset
            ocean_mask = historical_ERA5_images[varname_ocean_mask]
            input_skt = historical_ERA5_images[varname_skt]
            target_skt = target_ERA5_images[varname_skt]

            # for multi-input cases, use time=-1 ocean mask for all times
            if history_len > 1:
                ocean_mask[: history_len - 1] = ocean_mask.isel(time=-1)

            # get ocean mask
            ocean_mask_bool = ocean_mask.isel(time=-1) == 0

            # for multi-input cases, use time=-1 ocean SKT for all times
            if history_len > 1:
                input_skt[: history_len - 1] = input_skt[: history_len - 1].where(
                    ~ocean_mask_bool, input_skt.isel(time=-1)
                )

            # for target skt, replace ocean values using time=-1 input SKT
            target_skt = target_skt.where(~ocean_mask_bool, input_skt.isel(time=-1))

            # Update the target_ERA5_images dataset with the modified target_skt
            historical_ERA5_images[varname_ocean_mask] = ocean_mask
            historical_ERA5_images[varname_skt] = input_skt
            target_ERA5_images[varname_skt] = target_skt

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
        # stop_forecast = ((ind_start_current_step - index) == forecast_len)
        # sample['forecast_step'] = ind_start_current_step - index + 1
        # sample['stop_forecast'] = stop_forecast
        sample["datetime"] = [
            int(
                historical_ERA5_images.time[0]
                .astype("datetime64[s]")
                .values
                .astype(int)
            ),
            int(target_ERA5_images.time[0].astype("datetime64[s]").values.astype(int)),
        ]

    except Exception as e:
        logger.error(f"Error processing index {tuple_index}: {e}")
        raise

    return sample


class RepeatingIndexSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        forecast_len,
        skip_periods=1,
        shuffle=True,
        seed=42,
        rank=0,
        num_replicas=1,
    ):
        """
        Sampler that yields each starting index repeated (forecast_len + 1) times,
        ensuring indices don't exceed the valid range for a full forecast sequence.
        Supports distributed sampling.

        Args:
        - dataset (Dataset): The dataset to sample from.
        - forecast_len (int): Length of each forecast sequence minus one.
        - skip_periods (int): Number of periods to skip between sequences.
        - shuffle (bool): Whether to shuffle the starting indices.
        - seed (int): Random seed for reproducibility.
        - rank (int): Rank of the current process (for distributed training).
        - world_size (int): Total number of processes (for distributed training).
        """
        self.dataset = dataset
        self.forecast_len = forecast_len + 1  # Total steps in the forecast sequence
        self.skip_periods = skip_periods
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.num_replicas = num_replicas

        # Compute valid starting indices ensuring full sequences fit
        all_start_indices = list(range(0, len(self.dataset), skip_periods))

        num_indices = len(
            all_start_indices
        )  # Trim the number of indices to ensure it's divisible by world_size
        num_indices_per_rank = num_indices // self.num_replicas
        all_start_indices = all_start_indices[
            : num_indices_per_rank * self.num_replicas
        ]
        self.all_start_indices = all_start_indices
        self.num_indices_per_rank = num_indices_per_rank

        if self.shuffle:
            self.rng = np.random.default_rng(seed)
            # rng.shuffle(self.start_indices)

    def __len__(self):
        """Returns the total number of indices for this rank."""
        # return len(self.start_indices) * self.forecast_len
        return self.num_indices_per_rank * self.forecast_len

    def __iter__(self):
        """
        Yields each start index repeated (forecast_len + 1) times.
        """
        all_indices = self.all_start_indices
        if self.shuffle:
            all_indices = self.rng.permutation(all_indices)

        self.start_indices = all_indices[self.rank :: self.num_replicas]
        assert len(self.start_indices) == self.num_indices_per_rank
        for idx in self.start_indices:
            for _ in range(self.forecast_len):
                yield idx

    def batches_per_epoch(self):
        """
        Computes the number of batches per epoch for a given batch size.

        Returns:
        - int: Number of batches per epoch.
        """
        return self.num_indices_per_rank


class ERA5_and_Forcing_MultiStep(torch.utils.data.Dataset):
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
        one_shot=None,
        max_forecast_len=None,
        sst_forcing=None,
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
        - one_shot(bool, optional): Whether to return all states or just
                                    the final state of the training target. Default is None
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

        # skip periods
        self.skip_periods = skip_periods
        if self.skip_periods is None:
            self.skip_periods = 1

        # one shot option
        self.one_shot = one_shot

        # total number of needed forecast lead times
        self.total_seq_len = self.history_len + self.forecast_len

        # sst forcing
        self.sst_forcing = sst_forcing

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
            sst_forcing=self.sst_forcing,
        )

        self.total_length = len(self.ERA5_indices)
        self.current_epoch = None
        self.forecast_step_count = 0
        self.current_index = None
        self.initial_index = None

    def __post_init__(self):
        # Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        # compute the total number of length
        total_len = 0
        for ERA5_xarray in self.all_files:
            total_len += len(ERA5_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.forecast_step_count = 0
        self.current_index = None
        self.initial_index = None

    def __getitem__(self, index):
        if (self.forecast_step_count == self.forecast_len + 1) or (
            self.current_index is None
        ):
            # We've completed the last forecast or we're starting for the first time
            # Start a new forecast using the sampler index
            self.current_index = index  # self._get_random_start_index()
            self.forecast_step_count = 0
            index = self.current_index
            self.initial_index = self.current_index
        else:
            # Ignore the sampler index and continue the forecast
            self.current_index += 1
            index = self.current_index

        # Worker process
        sample = self.worker((self.initial_index, index))

        # assign sample index
        sample["forecast_step"] = self.forecast_step_count + 1
        sample["index"] = index
        sample["stop_forecast"] = self.forecast_step_count == self.forecast_len

        # update the step count
        self.forecast_step_count += 1

        return sample


if __name__ == "__main__":
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from credit.transforms import load_transforms
    from credit.parser import credit_main_parser, training_data_check
    from credit.datasets import setup_data_loading, set_globals

    # filename = "/glade/derecho/scratch/schreck/finetune/arnold/model_xform.yml"
    filename = "../../config/example-v2025.2.0.yml"
    with open(filename) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )
    training_data_check(conf, print_summary=False)

    data_config = setup_data_loading(conf)

    data_config["forecast_len"] = 5
    batch_size = 1
    training_type = "train"

    set_globals(data_config, namespace=globals())

    dataset_multi = ERA5_and_Forcing_MultiStep(
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
        history_len=data_config["history_len"],
        forecast_len=data_config["forecast_len"],
        skip_periods=data_config["skip_periods"],
        one_shot=False,
        max_forecast_len=data_config["max_forecast_len"],
        sst_forcing=data_config["sst_forcing"],
        transform=load_transforms(conf),
    )

    sampler = RepeatingIndexSampler(
        dataset_multi,
        forecast_len=data_config["forecast_len"],
        num_replicas=1,
        rank=0,
        seed=1000,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset_multi,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=False,
        num_workers=1,  # set to one so prefetch is working
        prefetch_factor=4,
    )

    dataloader.dataset.set_epoch(0)
    for k, sample in enumerate(dataloader):
        print(
            k,
            sample["index"],
            sample["datetime"],
            sample["forecast_step"],
            sample["stop_forecast"],
            sample["x"].shape,
            sample["x_surf"].shape,
            # sample["x_forcing_static"].shape,
        )
        if k == 500:
            break
