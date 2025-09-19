import logging
from typing import Dict, Optional, Tuple

import cftime
import numpy as np
import torch
import xarray as xr
from einops import rearrange

from credit.ocean.samudra_constants import DEPTH_I_LEVELS, DEPTH_LEVELS, MASK_VARS, TensorMap


def extract_wet_mask(data, outputs, hist):
    wet_mask = data[MASK_VARS]
    if "time" in wet_mask.dims:
        wet_mask_np = wet_mask.isel(time=0).to_array().to_numpy()
        wet_surface_mask_np = wet_mask[MASK_VARS[0]].isel(time=0).to_numpy()
    else:
        wet_mask_np = wet_mask.to_array().to_numpy()
        wet_surface_mask_np = wet_mask[MASK_VARS[0]].to_numpy()

    depth_ind = []
    for var_depth_i in outputs:
        var_split = var_depth_i.split("_")
        if len(var_split) == 1:
            depth_ind.append(0)
        else:
            depth_ind.append(int(var_split[-1]))

    wet_inp = torch.from_numpy(wet_mask_np[depth_ind])
    wet_surface = torch.from_numpy(wet_surface_mask_np)
    wet_inp = torch.concat([wet_inp] * (hist + 1), dim=0)
    return wet_inp, wet_surface


def get_inference_steps(time_config, time_delta=5, hist=1):
    """
    Get the number of inference/rollout steps for the given time configuration.

    Args:
        time_config: Time configuration
        time_delta: Time delta in days
        hist: Number of rollout steps

    Returns:
        num_steps: Number of rollout steps
    """
    start_time_str = time_config.start_time
    start_year, start_month, start_day = start_time_str.split("-")
    start_time = cftime.DatetimeNoLeap(
        int(start_year), int(start_month), int(start_day), 0, 0, 0
    )

    end_time_str = time_config.end_time
    end_year, end_month, end_day = end_time_str.split("-")
    end_time = cftime.DatetimeNoLeap(
        int(end_year), int(end_month), int(end_day), 0, 0, 0
    )
    num_steps = (end_time - start_time).days // time_delta + 1
    # Might have extra remaining days, so we remove them
    mod = num_steps % (hist + 1)
    num_steps = num_steps - mod
    return num_steps


def convert_tensor_out_to_dict(tensor_out: torch.Tensor) -> Dict[str, torch.Tensor]:
    tensor_map = TensorMap.get_instance()
    assert tensor_out.ndim == 5
    assert tensor_out.shape[2] == len(tensor_map.outputs)
    out_dict = {}
    for i, var in enumerate(tensor_map.outputs):
        out_dict[var] = tensor_out[:, :, i]
    return out_dict


def get_norm_unnorm_dicts(
    data: torch.Tensor,
    input_type: str = "target",
    prognostic_channels: int = 0,
    hist: int = 1,
):
    normalize = Normalize.get_instance()
    # Remove boundary data if input
    if input_type == "input":
        data = data[:, :prognostic_channels]

    # Separate history from channels
    data_reshaped = rearrange(data, "n (hi c) h w -> n hi c h w", hi=hist + 1)
    # Get normalized dict
    data_dict = convert_tensor_out_to_dict(data_reshaped)
    # Unnormalize
    data_unnorm = normalize.unnormalize_tensor_prognostics(data_reshaped)
    # Get unnormalized dict
    data_unnorm_dict = convert_tensor_out_to_dict(data_unnorm)
    return data_dict, data_unnorm_dict


def compute_anomalies(data: xr.Dataset, var: str) -> xr.Dataset:
    """
    Compute the anomalies of a data variable.
    """
    climatology = data[var].groupby("time.dayofyear").mean("time").compute()
    # Remove the seasonal cycle (climatology) from the detrended data
    day_of_year = data[var]["time"].dt.dayofyear
    data[var + "_anomalies"] = (
        data[var] - climatology.sel(dayofyear=day_of_year)
    ).compute()
    return data


def rename_vars(data: xr.Dataset) -> xr.Dataset:
    """
    Rename variables if required.
    """
    for var_str in data.variables:
        # OM4 data format has variables in the form: var_lev_depthlevel
        # ex. so_lev_1040_0. We need to convert into var_depthlevelidx
        if "_lev_" in var_str:
            var_split = var_str.split("_lev_")
            var = var_split[0]
            lev_in_depth = float(var_split[1].replace("_", "."))
            lev_in_depth_idx = DEPTH_LEVELS.index(lev_in_depth)
            data = data.rename({var_str: var + "_" + str(lev_in_depth_idx)})
    return data


def validate_data(
    data: xr.Dataset,
    data_mean: xr.Dataset,
    data_std: xr.Dataset,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Validate the data such that we have the correct format for training/evaluation.
    """
    # Check if Mask variables exist
    if MASK_VARS[0] not in data.variables:
        assert "wetmask" in data.variables, "Wet mask cannot be constructed without "
        "either the wetmask variable or the level-wise masks"

        # Construct the mask variables
        wet_mask = data["wetmask"]
        for i, lev in enumerate(DEPTH_I_LEVELS):
            assert int(lev) == i, "Level indices must match the order of DEPTH_I_LEVELS"
            data[f"mask_{lev}"] = wet_mask.isel(lev=i)

        data = data.drop_vars("wetmask")

    # Check if data variables are in the right format
    # This check is to ensure we convert data to the correct format
    data = rename_vars(data)
    data_mean = rename_vars(data_mean)
    data_std = rename_vars(data_std)

    # OM4 data has coordinates we don't need
    # We drop them and rename x, y dimensions to lon, lat
    if "lat" not in data.dims:
        # Drop unnecessary coordinates and rename dimensions
        data = data.drop_vars(
            ["lat", "lon", "lat_b", "lon_b", "dayofyear"], errors="ignore"
        ).rename({"x": "lon", "y": "lat"})

    # Check if anomalies are needed to be computed
    tensor_map = TensorMap.get_instance()
    for var in tensor_map.boundary_vars:
        if var.endswith("_anomalies"):
            base_var = var.replace("_anomalies", "")
            if var not in data.variables and base_var in data.variables:
                logging.info(f"Computing anomalies for {base_var}")
                data = compute_anomalies(data, base_var)

    return data, data_mean, data_std


class Normalize:
    _instance: Optional["Normalize"] = None

    def __new__(cls, *args, **kwargs) -> "Normalize":
        # Prevent direct instantiation
        raise TypeError(
            "Normalize cannot be instantiated directly. Use init_instance() instead."
        )

    @classmethod
    def get_instance(cls) -> "Normalize":
        if cls._instance is None:
            raise ValueError("Normalize not initialized")
        return cls._instance

    @classmethod
    def init_instance(
        cls,
        data_mean: xr.Dataset,
        data_std: xr.Dataset,
        prognostic_vars: str,
        boundary_vars: str,
        wet_mask: torch.Tensor,
    ) -> "Normalize":
        """Initialize the singleton instance with normalization parameters."""
        if cls._instance is not None:
            raise ValueError("Normalize already initialized")

        instance = super().__new__(cls)
        instance._initialize(
            data_mean, data_std, prognostic_vars, boundary_vars, wet_mask
        )
        cls._instance = instance
        return cls._instance

    def _initialize(
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
        """Normalize output tensor."""
        tensor_mean = self._to_tensor(self._prognostic_mean_np, data.device)
        tensor_std = self._to_tensor(self._prognostic_std_np, data.device)
        if data.ndim == 4:
            tensor_mean = tensor_mean.reshape([1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, -1, 1, 1])
        elif data.ndim == 5:
            tensor_mean = tensor_mean.reshape([1, 1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, 1, -1, 1, 1])

        norm = (data - tensor_mean) / tensor_std
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        return norm

    def unnormalize_tensor_prognostics(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize output tensor."""
        tensor_mean = self._to_tensor(self._prognostic_mean_np, data.device)
        tensor_std = self._to_tensor(self._prognostic_std_np, data.device)

        if data.ndim == 4:
            assert data.shape[1] == self._prognostic_mean_np.shape[0]
            tensor_mean = tensor_mean.reshape([1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, -1, 1, 1])
        elif data.ndim == 5:
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
        """Normalize output numpy array."""
        if data.ndim == 3:
            norm = (data - self._prognostic_mean_np) / self._prognostic_std_np
        elif data.ndim == 4:
            norm = (
                data - self._prognostic_mean_np.reshape(1, -1, 1, 1)
            ) / self._prognostic_std_np.reshape(1, -1, 1, 1)
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def unnormalize_numpy_prognostics(self, data: np.ndarray) -> np.ndarray:
        """Unnormalize output numpy array."""
        data_unnorm = data * self._prognostic_std_np + self._prognostic_mean_np
        data_unnorm = data_unnorm * self._wet_mask_np
        return data_unnorm
