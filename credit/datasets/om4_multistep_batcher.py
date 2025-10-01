from credit.ocean.samudra_constants import PROG_VARS_MAP, BOUND_VARS_MAP, TensorMap
from credit.ocean.samudra_data import validate_data, extract_wet_mask
import xarray as xr
import torch
import logging 
import math 
import numpy as np
from torch.utils.data import DistributedSampler
from pathlib import Path
import cftime
import pandas as pd


class StandardScaler:
    def __init__(
        self,
        data_mean: xr.Dataset,
        data_std: xr.Dataset,
        prognostic_vars: str,
        boundary_vars: str,
        wet_mask: torch.Tensor,
    ) -> None:

        """Store normalization parameters and pre-compute numpy arrays."""
        self.prognostic_mean = data_mean[prognostic_vars]
        self.prognostic_std = data_std[prognostic_vars]
        self.boundary_mean = data_mean[boundary_vars]
        self.boundary_std = data_std[boundary_vars]
        self.wet_mask = wet_mask

        # Pre-compute numpy arrays for faster access
        self._prognostic_mean_np = (
            self.prognostic_mean.to_array().to_numpy().reshape(-1)
        )
        self._prognostic_std_np = self.prognostic_std.to_array().to_numpy().reshape(-1)
        self._wet_mask_np = self.wet_mask.numpy()

    def _to_tensor(self, array: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert numpy array to tensor on specified device."""
        return torch.from_numpy(array).to(device)

    def normalize_prognostics(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize input dataset."""
        norm = (data - self.prognostic_mean) / self.prognostic_std
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_boundary(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize boundary conditions."""
        norm = (data - self.boundary_mean) / self.boundary_std
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def unnormalize_prognostics(self, data: xr.Dataset) -> xr.Dataset:
        """Unnormalize output dataset."""
        data_unnorm = data * self.prognostic_std + self.prognostic_mean
        data_unnorm = data_unnorm * xr.DataArray(self._wet_mask_np)
        return data_unnorm

    def normalize_tensor_prognostics(
        self, data: torch.Tensor, fill_nan=True, fill_value=0.0
    ) -> torch.Tensor:
        """Normalize tensor."""
        tensor_mean = self._to_tensor(self._prognostic_mean_np, data.device)
        tensor_std = self._to_tensor(self._prognostic_std_np, data.device)

        if data.ndim == 4:  # (B, C, H, W)
            tensor_mean = tensor_mean.reshape([1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, -1, 1, 1])
        elif data.ndim == 5:  # (B, T, C, H, W)
            tensor_mean = tensor_mean.reshape([1, 1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, 1, -1, 1, 1])

        norm = (data - tensor_mean) / tensor_std
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        return norm

    def unnormalize_tensor_prognostics(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize tensor."""
        tensor_mean = self._to_tensor(self._prognostic_mean_np, data.device)
        tensor_std = self._to_tensor(self._prognostic_std_np, data.device)

        if data.ndim == 4:  # (B, C, H, W)
            assert data.shape[1] == self._prognostic_mean_np.shape[0]
            tensor_mean = tensor_mean.reshape([1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, -1, 1, 1])
        elif data.ndim == 5:  # (B, T, C, H, W)
            assert data.shape[2] == self._prognostic_mean_np.shape[0]
            tensor_mean = tensor_mean.reshape([1, 1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, 1, -1, 1, 1])
        else:
            raise ValueError(f"Invalid data shape: {data.shape}")

        unnorm = data * tensor_std + tensor_mean
        unnorm = unnorm * self.wet_mask.to(data.device)
        return unnorm

    def normalize_numpy_prognostics(
        self, data: np.ndarray, fill_nan=True, fill_value=0.0
    ) -> np.ndarray:
        """Normalize numpy array."""
        if data.ndim == 3:  # (C, H, W)
            norm = (data - self._prognostic_mean_np) / self._prognostic_std_np
        elif data.ndim == 4:  # (B, C, H, W)
            norm = (
                data - self._prognostic_mean_np.reshape(1, -1, 1, 1)
            ) / self._prognostic_std_np.reshape(1, -1, 1, 1)
        if fill_nan:
            norm = np.nan_to_num(norm, nan=fill_value)
        return norm

    def unnormalize_numpy_prognostics(self, data: np.ndarray) -> np.ndarray:
        """Unnormalize numpy array."""
        data_unnorm = data * self._prognostic_std_np + self._prognostic_mean_np
        data_unnorm = data_unnorm * self._wet_mask_np
        return data_unnorm


class Ocean_MultiStep_Batcher(torch.utils.data.Dataset):
    """
    Ocean dataset that handles both single-step and multi-step autoregressive training.
    Returns tensors with shape (batch, channels, time, lat, lon).
    """

    def __init__(
        self,
        conf,
        seed=42,
        rank=0,
        world_size=1,
        batch_size=1,
        shuffle=True
    ):
        """
        Parameters:
        
        """
        
        self.input_length = conf["data"]["input_length"]
        self.output_length = conf["data"]["output_length"]
        self.forecast_len = conf["data"]["forecast_len"]
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.batch_size = batch_size 

        # set random seed
        self.rng = np.random.default_rng(seed=seed)

        # Construct full paths
        data_path = conf["data"]["data_path"]
        data_means_path = conf["data"]["mean_path"]
        data_stds_path = conf["data"]["std_path"]

        # Open the zarr/dataset files
        data_xr = xr.open_zarr(data_path, chunks={})
        data_mean_xr = xr.open_zarr(data_means_path, chunks={})
        data_std_xr = xr.open_zarr(data_stds_path, chunks={})

        try:
            # whatever call initializes TensorMap
            TensorMap.init_instance(
                conf["data"]["prognostic_vars_key"], conf["data"]["dynamic_forcing_vars_key"]
            )
        except ValueError as e:
            if "TensorMap already initialized" in str(e):
                TensorMap.get_instance()
            else:
                raise

        #   Validate data (assuming this function is available from your utils)
        data, data_mean, data_std = validate_data(data_xr, data_mean_xr, data_std_xr)

        # Subset by time range if provided
        # Convert config strings to cftime.DatetimeJulian
        if shuffle:
            start, end = conf["train_years"]
        else:
            start, end = conf["valid_years"]

        start_cf = cftime.DatetimeJulian(*pd.to_datetime(start).timetuple()[:6])
        end_cf   = cftime.DatetimeJulian(*pd.to_datetime(end).timetuple()[:6])

        # Subset data using cftime-compatible times
        data = data.sel(time=slice(start_cf, end_cf))
 
        # Get prognostic and boundary variables from your constants file
        prognostic_vars = PROG_VARS_MAP[conf["data"]["prognostic_vars_key"]]
        boundary_vars = BOUND_VARS_MAP[conf["data"]["dynamic_forcing_vars_key"]]

        # Store ocean data
        self._prognostic_data = data[prognostic_vars]
        self._boundary_data = data[boundary_vars]
        self.num_prognostic_vars = len(prognostic_vars)
        self.num_boundary_vars = len(boundary_vars)

        # Extract wet masks
        ## Set original Samudra history = 0 as we do things step-by-step
        wet, wet_surface = extract_wet_mask(data_xr, prognostic_vars, 0)
        
        self.wet = wet.bool()
        self.wet_surface = wet_surface.bool()

        # Normalize data
        self.normalize = StandardScaler(
            data_mean,
            data_std,
            prognostic_vars,
            boundary_vars,
            wet,
        )
        self._prognostic_data = self.normalize.normalize_prognostics(self._prognostic_data)
        self._boundary_data = self.normalize.normalize_boundary(self._boundary_data)

        # Dataset size - ensure we have enough timesteps for input + output
        self.size = data.time.size - self.input_length - self.output_length + 1

        # Use DistributedSampler for index management
        self.sampler = DistributedSampler(
            self, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=True
        )

        # Initialize state variables for batch management
        self.batch_indices = None
        self.time_steps = None
        self.forecast_step_counts = None
        self.current_epoch = None

        # Initialize batch indices
        self.sampler.set_epoch(0)
        self.batch_indices = list(self.sampler)
        if len(self.batch_indices) < batch_size:
            self.batch_size = len(self.batch_indices)

    def initialize_batch(self):
        """Initialize batch indices using DistributedSampler's indices."""
        if not hasattr(self, "batch_call_count"):
            self.batch_call_count = 0

        if self.current_epoch is None:
            logging.warning("You must first set the epoch number using set_epoch method.")

        total_indices = len(self.batch_indices)
        start = self.batch_call_count * self.batch_size
        end = start + self.batch_size

        if not self.shuffle:
            if end > total_indices:
                start = start % total_indices
                end = min(start + self.batch_size, total_indices)
            indices = self.batch_indices[start:end]
        else:
            if end > total_indices:
                indices = (
                    self.batch_indices[start:] + self.batch_indices[:(end % total_indices)]
                )
            else:
                indices = self.batch_indices[start:end]

        self.batch_call_count += 1
        if start + self.batch_size >= total_indices:
            self.batch_call_count = 0

        self.current_batch_indices = list(indices)
        self.time_steps = [0 for _ in self.current_batch_indices]
        self.forecast_step_counts = [0 for _ in self.current_batch_indices]

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        """Set epoch for distributed training."""
        self.current_epoch = epoch
        self.sampler.set_epoch(epoch)
        self.batch_indices = list(self.sampler)
        self.batch_call_count = 0
        self.initialize_batch()

    def batches_per_epoch(self):
        return math.ceil(len(list(self.batch_indices)) / self.batch_size)

    def _get_batch_samples(self, sample_indices):
        """
        Get multiple ocean samples efficiently by batching xarray operations.
        Returns batch dict with stacked tensors.
        """    
        # Pre-allocate lists for stacking
        input_list = []
        target_list = []
        input_datetime_list = []
        target_datetime_list = []
        
        # Get all time slices we need
        input_slices = []
        target_slices = []
        boundary_indices = []
        
        for idx in sample_indices:
            # Input time range
            input_start = idx
            input_end = idx + self.input_length
            input_slices.append(slice(input_start, input_end))
            
            # Target time range  
            target_start = idx + self.input_length
            target_end = target_start + self.output_length
            target_slices.append(slice(target_start, target_end))
            
            # Boundary time index
            boundary_indices.append(idx + self.input_length - 1)
        
        # Batch load all required time indices at once
        all_time_indices = set()
        for s_idx in sample_indices:
            for t in range(s_idx, s_idx + self.input_length + self.output_length):
                all_time_indices.add(t)
        
        time_indices = sorted(all_time_indices)
        
        # Load all needed data in one go
        prog_data_batch = self._prognostic_data.isel(time=time_indices)
        boundary_data_batch = self._boundary_data.isel(time=time_indices)
        
        # Create time index mapping
        time_map = {t_idx: i for i, t_idx in enumerate(time_indices)}
        
        # Process each sample
        for _, sample_idx in enumerate(sample_indices):
            # Map sample indices to batch indices
            input_start_batch = time_map[sample_idx]
            input_end_batch = time_map[sample_idx + self.input_length - 1] + 1
            target_start_batch = time_map[sample_idx + self.input_length]
            target_end_batch = time_map[sample_idx + self.input_length + self.output_length - 1] + 1
            boundary_batch_idx = time_map[sample_idx + self.input_length - 1]
            
            # Get prognostic input
            prog_input = prog_data_batch.isel(time=slice(input_start_batch, input_end_batch))
            prog_input = prog_input.to_array().transpose("variable", "time", "lat", "lon").to_numpy()
            prog_input = torch.from_numpy(prog_input).float()
            
            # Apply wet mask
            for var_idx in range(self.num_prognostic_vars):
                prog_input[var_idx] = torch.where(self.wet[var_idx], prog_input[var_idx], 0.0)
            
            # Get boundary input
            boundary_input = boundary_data_batch.isel(time=boundary_batch_idx)
            boundary_input = boundary_input.to_array().transpose("variable", "lat", "lon").to_numpy()
            boundary_input = torch.from_numpy(boundary_input).float()
            boundary_input = torch.where(self.wet_surface, boundary_input, 0.0)
            boundary_input = boundary_input.unsqueeze(1).expand(-1, self.input_length, -1, -1)
            
            # Combine input
            full_input = torch.cat([prog_input, boundary_input], dim=0)
            input_list.append(full_input.numpy())
            
            # Get target
            target = prog_data_batch.isel(time=slice(target_start_batch, target_end_batch))
            target = target.to_array().transpose("variable", "time", "lat", "lon").to_numpy()
            target = torch.from_numpy(target).float()
            
            # Apply wet mask to target
            for var_idx in range(self.num_prognostic_vars):
                target[var_idx] = torch.where(self.wet[var_idx], target[var_idx], 0.0)
            target_list.append(target.numpy())
            
            # Get datetimes
            input_times = self._prognostic_data.time.isel(time=slice(sample_idx, sample_idx + self.input_length)).values
            target_times = self._prognostic_data.time.isel(time=slice(sample_idx + self.input_length, sample_idx + self.input_length + self.output_length)).values
            
            input_datetime_list.append(input_times.astype("datetime64[ns]").astype(np.int64))
            target_datetime_list.append(target_times.astype("datetime64[ns]").astype(np.int64))
        
        # Stack everything
        batch = {
            'input': torch.from_numpy(np.stack(input_list, axis=0)).float(),
            'target': torch.from_numpy(np.stack(target_list, axis=0)).float(),
            'input_datetime': torch.from_numpy(np.stack(input_datetime_list, axis=0)).long(),
            'target_datetime': torch.from_numpy(np.stack(target_datetime_list, axis=0)).long(),
            'index': torch.tensor(sample_indices, dtype=torch.int64).unsqueeze(-1)
        }
        
        return batch

    def __getitem__(self, _):
        """
        Returns batch with tensors shaped (batch, channels, time, lat, lon).
        Optimized for speed.
        """
        # Reset batch if forecast length exceeded
        if self.forecast_step_counts[0] == self.forecast_len:
            self.initialize_batch()

        # Collect all indices first
        sample_indices = []
        for k, idx in enumerate(self.current_batch_indices):
            current_t = self.time_steps[k]
            sample_idx = idx + current_t
            sample_indices.append(sample_idx)
        
        # Batch load all data at once
        batch = self._get_batch_samples(sample_indices)

        # Update counters
        for k in range(len(self.current_batch_indices)):
            self.time_steps[k] += self.output_length
            self.forecast_step_counts[k] += 1

        # Add rollout control flags
        batch["forecast_step"] = torch.tensor([self.forecast_step_counts[0]])
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len

        return batch

    def _get_ocean_sample(self, idx):
        """
        Get single ocean sample with proper time dimensions.
        Returns tensors with shape (channels, time, lat, lon).
        """
        # Get input timesteps: [idx, idx+1, ..., idx+input_length-1]
        input_start = idx
        input_end = idx + self.input_length
        
        # Get prognostic input data
        prog_input = self._prognostic_data.isel(time=slice(input_start, input_end))
        prog_input = prog_input.to_array().transpose("variable", "time", "lat", "lon").to_numpy()
        prog_input = torch.from_numpy(prog_input).float()
        
        # Apply wet mask to prognostic variables
        for var_idx in range(self.num_prognostic_vars):
            prog_input[var_idx] = torch.where(self.wet[var_idx], prog_input[var_idx], 0.0)
        
        # Get boundary input (use last timestep of input sequence)
        boundary_time = idx + self.input_length - 1
        boundary_input = self._boundary_data.isel(time=boundary_time)
        boundary_input = boundary_input.to_array().transpose("variable", "lat", "lon").to_numpy()
        boundary_input = torch.from_numpy(boundary_input).float()
        
        # Apply wet surface mask to boundary variables
        boundary_input = torch.where(self.wet_surface, boundary_input, 0.0)
        
        # Expand boundary to match time dimension of input
        boundary_input = boundary_input.unsqueeze(1).expand(-1, self.input_length, -1, -1)
        
        # Combine prognostic and boundary: (prog_channels + boundary_channels, time, lat, lon)
        full_input = torch.cat([prog_input, boundary_input], dim=0)
        
        # Get target timesteps: [idx+input_length, ..., idx+input_length+output_length-1]
        target_start = idx + self.input_length
        target_end = target_start + self.output_length
        
        # Get prognostic target data (only prognostic variables for target)
        target = self._prognostic_data.isel(time=slice(target_start, target_end))
        target = target.to_array().transpose("variable", "time", "lat", "lon").to_numpy()
        target = torch.from_numpy(target).float()
        
        # Apply wet mask to target
        for var_idx in range(self.num_prognostic_vars):
            target[var_idx] = torch.where(self.wet[var_idx], target[var_idx], 0.0)

        # Input datetimes
        input_datetimes = self._prognostic_data.time.isel(time=slice(input_start, input_end)).values
        input_datetimes = input_datetimes.astype("datetime64[ns]").astype(np.int64)

        # Target datetimes
        target_datetimes = self._prognostic_data.time.isel(time=slice(target_start, target_end)).values
        target_datetimes = target_datetimes.astype("datetime64[ns]").astype(np.int64)
        
        sample = {
            "input": full_input.numpy(),    # (prog_channels + boundary_channels, time, lat, lon)
            "target": target.numpy(),       # (prog_channels, time, lat, lon)
            "input_datetime": input_datetimes,
            "target_datetime": target_datetimes
        }
        
        return sample


class Ocean_Tensor_Batcher(torch.utils.data.Dataset):
    """
    Ocean dataset that handles both single-step and multi-step autoregressive training.
    Loads cached .pt samples while preserving autoregressive logic.
    """

    def __init__(
        self,
        conf,
        seed=42,
        rank=0,
        world_size=1,
        batch_size=1,
        shuffle=True,
    ):

        # Construct full paths
        data_path = conf["data"]["data_path"]
        data_means_path = conf["data"]["mean_path"]
        data_stds_path = conf["data"]["std_path"]

        # Open the zarr/dataset files
        data_xr = xr.open_zarr(data_path, chunks={})
        data_mean_xr = xr.open_zarr(data_means_path, chunks={})
        data_std_xr = xr.open_zarr(data_stds_path, chunks={})

        try:
            # whatever call initializes TensorMap
            TensorMap.init_instance(
                conf["data"]["prognostic_vars_key"], conf["data"]["dynamic_forcing_vars_key"]
            )
        except ValueError as e:
            if "TensorMap already initialized" in str(e):
                TensorMap.get_instance()
            else:
                raise

        #   Validate data (assuming this function is available from your utils)
        data, _, _ = validate_data(data_xr, data_mean_xr, data_std_xr)

        # Get prognostic and boundary variables from your constants file
        prognostic_vars = PROG_VARS_MAP[conf["data"]["prognostic_vars_key"]]

        # Extract wet masks
        ## Set original Samudra history = 0 as we do things step-by-step
        wet, wet_surface = extract_wet_mask(data_xr, prognostic_vars, 0)

        self.wet = wet.bool()
        self.wet_surface = wet_surface.bool()

        # Regular arguments
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.batch_size = batch_size

        # Add these missing lines:
        self.input_length = conf["data"]["input_length"]
        self.output_length = conf["data"]["output_length"]
        self.forecast_len = conf["data"]["forecast_len"]

        # Cached samples directory
        self.samples_dir = Path(conf["data"]["data_path"]).parent / "cached" / "samples"
        all_files = sorted(list(self.samples_dir.glob("sample_*.pt")))

        train_years = conf["data"]["train_years"]
        valid_years = conf["data"]["valid_years"]

        # Convert config strings to cftime.DatetimeJulian
        train_start = cftime.DatetimeJulian(*pd.to_datetime(train_years[0]).timetuple()[:6])
        train_end   = cftime.DatetimeJulian(*pd.to_datetime(train_years[1]).timetuple()[:6])
        valid_start = cftime.DatetimeJulian(*pd.to_datetime(valid_years[0]).timetuple()[:6])
        valid_end   = cftime.DatetimeJulian(*pd.to_datetime(valid_years[1]).timetuple()[:6])

        # Map timesteps to indices
        time_index = data.time.values  # still cftime.DatetimeJulian
        train_mask = (time_index >= train_start) & (time_index <= train_end)
        valid_mask = (time_index > valid_start) & (time_index <= valid_end)

        train_indices = np.where(train_mask)[0]
        valid_indices = np.where(valid_mask)[0]

        all_files = sorted(list(self.samples_dir.glob("sample_*.pt")))

        if shuffle:
            self.sample_files = [all_files[i] for i in train_indices]
        else:
            self.sample_files = [all_files[i] for i in valid_indices]

        self.size = len(self.sample_files)

        # Use DistributedSampler for index management
        self.sampler = DistributedSampler(
            self, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=True
        )

        # Initialize batch tracking
        self.batch_indices = None
        self.time_steps = None
        self.forecast_step_counts = None
        self.current_epoch = None
        self.sampler.set_epoch(0)
        self.batch_indices = list(self.sampler)
        if len(self.batch_indices) < batch_size:
            self.batch_size = len(self.batch_indices)

    def initialize_batch(self):
        if not hasattr(self, "batch_call_count"):
            self.batch_call_count = 0

        if self.current_epoch is None:
            logging.warning("You must first set the epoch number using set_epoch method.")

        total_indices = len(self.batch_indices)
        start = self.batch_call_count * self.batch_size
        end = start + self.batch_size

        if not self.shuffle:
            if end > total_indices:
                start = start % total_indices
                end = min(start + self.batch_size, total_indices)
            indices = self.batch_indices[start:end]
        else:
            if end > total_indices:
                indices = self.batch_indices[start:] + self.batch_indices[:(end % total_indices)]
            else:
                indices = self.batch_indices[start:end]

        self.batch_call_count += 1
        if start + self.batch_size >= total_indices:
            self.batch_call_count = 0

        self.current_batch_indices = list(indices)
        self.time_steps = [0 for _ in self.current_batch_indices]
        self.forecast_step_counts = [0 for _ in self.current_batch_indices]

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.sampler.set_epoch(epoch)
        self.batch_indices = list(self.sampler)
        self.batch_call_count = 0
        self.initialize_batch()

    def batches_per_epoch(self):
        return math.ceil(len(list(self.batch_indices)) / self.batch_size)

    def _get_batch_samples(self, sample_indices):
        """
        Load multiple cached .pt samples at once.
        """
        input_list = []
        target_list = []
        input_datetime_list = []
        target_datetime_list = []

        for idx in sample_indices:
            sample = self._get_ocean_sample(idx)
            input_list.append(sample["input"])
            target_list.append(sample["target"])
            input_datetime_list.append(sample["input_datetime"])
            target_datetime_list.append(sample["target_datetime"])

        batch = {
            "input": torch.stack(input_list, dim=0).float(),
            "target": torch.stack(target_list, dim=0).float(),
            "input_datetime": torch.stack(input_datetime_list, dim=0).long(),
            "target_datetime": torch.stack(target_datetime_list, dim=0).long(),
            "index": torch.tensor(sample_indices, dtype=torch.int64).unsqueeze(-1),
        }
        return batch

    def __getitem__(self, _):
        """
        Returns batch with tensors shaped (batch, channels, time, lat, lon)
        using autoregressive logic.
        """
        if self.forecast_step_counts[0] == self.forecast_len:
            self.initialize_batch()

        sample_indices = []
        for k, idx in enumerate(self.current_batch_indices):
            sample_idx = idx + self.time_steps[k]
            sample_indices.append(sample_idx)

        batch = self._get_batch_samples(sample_indices)

        for k in range(len(self.current_batch_indices)):
            self.time_steps[k] += 1
            self.forecast_step_counts[k] += 1

        batch["forecast_step"] = torch.tensor([self.forecast_step_counts[0]])
        batch["stop_forecast"] = batch["forecast_step"] == self.forecast_len

        return batch

    def _get_ocean_sample(self, idx):
        """
        Load a cached sample from disk.
        """
        sample_file = self.samples_dir / f"sample_{idx:06d}.pt"
        sample = torch.load(sample_file, map_location="cpu")
        return {
            "input": sample["input"],
            "target": sample["target"],
            "input_datetime": sample["input_datetime"],
            "target_datetime": sample["target_datetime"],
        }


if __name__ == "__main__":
    import yaml
    import pandas as pd

    with open("/glade/derecho/scratch/schreck/samudra/mom.yml", "r") as f:
        conf = yaml.safe_load(f)

    # Construct full paths
    # data_path = conf["data"]["data_path"]
    # data_means_path = conf["data"]["mean_path"]
    # data_stds_path = conf["data"]["std_path"]

    # # Open the zarr/dataset files
    # data_xr = xr.open_zarr(data_path, chunks={})
    # data_mean_xr = xr.open_zarr(data_means_path, chunks={})
    # data_std_xr = xr.open_zarr(data_stds_path, chunks={})

    # tensor_map = TensorMap.init_instance(
    #     conf["data"]["prognostic_vars_key"], conf["data"]["dynamic_forcing_vars_key"]
    # )
    # #   Validate data (assuming this function is available from your utils)
    # data, data_mean, data_std = validate_data(data_xr, data_mean_xr, data_std_xr)

    # # Get prognostic and boundary variables from your constants file
    # prognostic_vars = PROG_VARS_MAP[conf["data"]["prognostic_vars_key"]]
    # boundary_vars = BOUND_VARS_MAP[conf["data"]["dynamic_forcing_vars_key"]]

    # # Extract wet masks
    # ## Set original Samudra history = 0 as we do things step-by-step
    # wet, wet_surface = extract_wet_mask(data_xr, prognostic_vars, 0)
    # wet_without_hist, _ = extract_wet_mask(data_xr, prognostic_vars, 0)

    dataset = Ocean_MultiStep_Batcher(conf=conf, batch_size=1)
    dataset.set_epoch(0)

    # Get first batch
    batch = dataset[0]
    print("Step 1:")
    print(f"Input shape: {batch['input'].shape}")         # (2, 158, 180, 360) - batch_size=2, hist=1 gives (1+1)*79=158 channels
    print(f"Target shape: {batch['target'].shape}")       # (2, 79, 180, 360) - batch_size=2, 79 prognostic channels
    print(f"Forecast step: {batch['forecast_step']}")     # tensor([1])
    print(f"Input datetime: {pd.to_datetime(batch['input_datetime'].numpy().flatten(), unit='ns')}")     # tensor([1])
    print(f"Target datetime: {pd.to_datetime(batch['target_datetime'].numpy().flatten(), unit='ns')}")     # tensor([1])
    print(f"Stop forecast: {batch['stop_forecast']}")     # tensor([False])

    # For subsequent calls during rollout:
    batch2 = dataset[0]  
    print("Step 2:")
    print(f"Input shape: {batch2['input'].shape}")        # (2, 158, 180, 360) - same shape
    print(f"Target shape: {batch2['target'].shape}")      # (2, 79, 180, 360) - same shape
    print(f"Forecast step: {batch2['forecast_step']}")    # tensor([2])
    print(f"Input datetime: {pd.to_datetime(batch2['input_datetime'].numpy().flatten(), unit='ns')}")     # tensor([1])
    print(f"Target datetime: {pd.to_datetime(batch2['target_datetime'].numpy().flatten(), unit='ns')}")     # tensor([1])
    print(f"Stop forecast: {batch2['stop_forecast']}")    # tensor([False])

    batch3 = dataset[0]  
    print("Step 3:")
    print(f"Input shape: {batch3['input'].shape}")        # (2, 158, 180, 360) - same shape
    print(f"Target shape: {batch3['target'].shape}")      # (2, 79, 180, 360) - same shape
    print(f"Forecast step: {batch3['forecast_step']}")    # tensor([3])
    print(f"Input datetime: {pd.to_datetime(batch3['input_datetime'].numpy().flatten(), unit='ns')}")     # tensor([1])
    print(f"Target datetime: {pd.to_datetime(batch3['target_datetime'].numpy().flatten(), unit='ns')}")     # tensor([1])
    print(f"Stop forecast: {batch3['stop_forecast']}")    # tensor([False])

    batch4 = dataset[0]  
    print("Step 4:")
    print(f"Input shape: {batch4['input'].shape}")        # (2, 158, 180, 360) - same shape
    print(f"Target shape: {batch4['target'].shape}")      # (2, 79, 180, 360) - same shape
    print(f"Forecast step: {batch4['forecast_step']}")    # tensor([4])
    print(f"Input datetime: {pd.to_datetime(batch4['input_datetime'].numpy().flatten(), unit='ns')}")     # tensor([1])
    print(f"Target datetime: {pd.to_datetime(batch4['target_datetime'].numpy().flatten(), unit='ns')}")     # tensor([1])
    print(f"Stop forecast: {batch4['stop_forecast']}")    # tensor([False])