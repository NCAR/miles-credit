"""
The old transforms.py; it is deprecated
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

import torch
from torchvision import transforms as tforms

from credit.data import Sample
from bridgescaler import read_scaler

logger = logging.getLogger(__name__)




class NormalizeState_Quantile:
    """Class to use the Quantile scaler functionality."""

    def __init__(self, conf):
        """Normalize via quantile transform.

        Normalize via provided scaler file/s.

        Args:
            conf (str): path to config file.

        Attributes:
            scaler_file (str): path to scaler file.
            variables (list): list of upper air variables.
            surface_variables (list): list of surface variables.
            levels (int): number of upper-air variable levels.
            scaler_df (pd.df): scaler df.
            scaler_3ds (xr.ds): 3d scaler dataset.
            scaler_surfs (xr.ds): surface scaler dataset.

        """
        self.scaler_file = conf["data"]["quant_path"]
        self.variables = conf["data"]["variables"]
        self.surface_variables = conf["data"]["surface_variables"]
        self.levels = int(conf["model"]["levels"])
        self.scaler_df = pd.read_parquet(self.scaler_file)
        self.scaler_3ds = self.scaler_df["scaler_3d"].apply(read_scaler)
        self.scaler_surfs = self.scaler_df["scaler_surface"].apply(read_scaler)
        self.scaler_3d = self.scaler_3ds.sum()
        self.scaler_surf = self.scaler_surfs.sum()

        self.scaler_surf.channels_last = False
        self.scaler_3d.channels_last = False

    def __call__(self, sample: Sample, inverse: bool = False) -> Sample:
        """Normalize via quantile transform.

        Normalize via provided scaler file/s.

        Args:
            sample: batch.
            inverse: if true, will inverse the transform.

        Returns:
            torch.tensor: transformed type.

        """
        if inverse:
            return self.inverse_transform(sample)
        else:
            return self.transform(sample)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform.

        Inverse transform.

        Args:
            x: batch.

        Returns:
            inverse transformed x.

        """
        device = x.device
        tensor = x[:, : (len(self.variables) * self.levels), :, :]  # B, Var, H, W
        surface_tensor = x[
            :, (len(self.variables) * self.levels) :, :, :
        ]  # B, Var, H, W
        # beep
        # Reverse quantile transform using bridge scaler:
        transformed_tensor = tensor.clone()
        transformed_surface_tensor = surface_tensor.clone()
        # 3dvars
        rscal_3d = np.array(x[:, : (len(self.variables) * self.levels), :, :])
        transformed_tensor[:, :, :, :] = device_compatible_to(torch.tensor(
            (self.scaler_3d.inverse_transform(rscal_3d))
        ),device)
        # surf
        rscal_surf = np.array(x[:, (len(self.variables) * self.levels) :, :, :])
        transformed_surface_tensor[:, :, :, :] = device_compatible_to(torch.tensor(
            (self.scaler_surf.inverse_transform(rscal_surf))
        ),device)
        # cat them
        transformed_x = torch.cat(
            (transformed_tensor, transformed_surface_tensor), dim=1
        )
        # return
        return device_compatible_to(transformed_x,device)

    def transform(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform.

        Transform.

        Args:
            sample: batch.

        Returns:
            transformed batch.

        """
        normalized_sample = {}
        for key, value in sample.items():
            if isinstance(value, xr.Dataset):
                var_levels = []
                for var in self.variables:
                    levels = value.level.values
                    for level in levels:
                        var_levels.append(f"{var}_{level:d}")
                ds_times = value["time"].values
                for time in ds_times:
                    var_slices = []
                    for var in self.variables:
                        for level in levels:
                            var_slices.append(value[var].sel(time=time, level=level))

                    e3d = xr.concat(var_slices, pd.Index(var_levels, name="variable"))
                    e3d = e3d.expand_dims(dim="time", axis=0)
                    TTtrans = self.scaler_3d.transform(np.array(e3d))
                    # this is bad and should be fixed:
                    value["U"].sel(time=time)[:, :, :] = TTtrans[
                        :, : self.levels, :, :
                    ].squeeze()
                    value["V"].sel(time=time)[:, :, :] = TTtrans[
                        :, self.levels : (self.levels * 2), :, :
                    ].squeeze()
                    value["T"].sel(time=time)[:, :, :] = TTtrans[
                        :, (self.levels * 2) : (self.levels * 3), :, :
                    ].squeeze()
                    value["Q"].sel(time=time)[:, :, :] = TTtrans[
                        :, (self.levels * 3) : (self.levels * 4), :, :
                    ].squeeze()
                    e_surf = xr.concat(
                        [value[v].sel(time=time) for v in self.surface_variables],
                        pd.Index(self.surface_variables, name="variable"),
                    )
                    e_surf = e_surf.expand_dims(dim="time", axis=0)
                    TTtrans = self.scaler_surf.transform(e_surf)

                    for ee, varvar in enumerate(self.surface_variables):
                        value[varvar].sel(time=time)[:, :] = TTtrans[
                            0, ee, :, :
                        ].squeeze()
            normalized_sample[key] = value
        return normalized_sample

class NormalizeState:
    """Class to normalize state."""

    def __init__(self, conf):
        """Normalize state.

        Normalize the state via provided scaler file/s.

        Args:
            conf (str): path to config file.

        Attributes:
            mean_ds (str): path to mean.
            std_ds (str): path to std.
            variables (list): list of upper air variables.
            surface_variables (list): list of surface variables.
            levels (int): number of upper-air variable levels.

        """
        self.mean_ds = xr.open_dataset(conf["data"]["mean_path"])
        self.std_ds = xr.open_dataset(conf["data"]["std_path"])
        self.variables = conf["data"]["variables"]
        self.surface_variables = conf["data"]["surface_variables"]
        self.levels = conf["model"]["levels"]

        logger.info(
            "Loading preprocessing object for transform/inverse transform states into z-scores"
        )

    def __call__(self, sample: Sample, inverse: bool = False) -> Sample:
        """Normalize via quantile transform.

        Normalize via provided scaler file/s.

        Args:
            sample: batch.
            inverse: if true, will inverse the transform.

        Returns:
            torch.tensor: transformed type.

        """
        if inverse:
            return self.inverse_transform(sample)
        else:
            return self.transform(sample)

    def transform_dataset(self, DS: xr.Dataset) -> xr.Dataset:
        DS = (DS - self.mean_ds)/self.std_ds
        return DS

    def transform_array(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from unscaled to scaled values.

        Transform.

        Args:
            x: batch.

        Returns:
            transformed x.

        """
        device = x.device
        tensor = x[:, : (len(self.variables) * self.levels), :, :]
        surface_tensor = x[:, (len(self.variables) * self.levels) :, :, :]

        # Reverse z-score normalization using the pre-loaded mean and std
        transformed_tensor = tensor.clone()
        k = 0
        for name in self.variables:
            for level in range(self.levels):
                mean = self.mean_ds[name].values[level]
                std = self.std_ds[name].values[level]
                transformed_tensor[:, k] = (tensor[:, k] - mean) / std
                k += 1

        transformed_surface_tensor = surface_tensor.clone()
        for k, name in enumerate(self.surface_variables):
            mean = self.mean_ds[name].values
            std = self.std_ds[name].values
            transformed_surface_tensor[:, k] = (surface_tensor[:, k] - mean) / std

        transformed_x = torch.cat(
            (transformed_tensor, transformed_surface_tensor), dim=1
        )

        return device_compatible_to(transformed_x,device)

    def transform(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform from unscaled to scaled values.

        Transform.

        Args:
            sample: batch.

        Returns:
            transformed sample.

        """
        normalized_sample = {}
        for key, value in sample.items():
            if isinstance(value, xr.Dataset):
                normalized_sample[key] = (value - self.mean_ds) / self.std_ds
        return normalized_sample

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform between tensor forms.

        Inverse transform.

        Args:
            x: batch.

        Returns:
            inverse transformed x.

        """
        device = x.device
        tensor = x[:, : (len(self.variables) * self.levels), :, :]
        surface_tensor = x[:, (len(self.variables) * self.levels) :, :, :]

        # Reverse z-score normalization using the pre-loaded mean and std
        transformed_tensor = tensor.clone()
        k = 0
        for name in self.variables:
            for level in range(self.levels):
                mean = self.mean_ds[name].values[level]
                std = self.std_ds[name].values[level]
                transformed_tensor[:, k] = tensor[:, k] * std + mean
                k += 1

        transformed_surface_tensor = surface_tensor.clone()
        for k, name in enumerate(self.surface_variables):
            mean = self.mean_ds[name].values
            std = self.std_ds[name].values
            transformed_surface_tensor[:, k] = surface_tensor[:, k] * std + mean

        transformed_x = torch.cat(
            (transformed_tensor, transformed_surface_tensor), dim=1
        )

        return device_compatible_to(transformed_x,device)

class NormalizeTendency:
    """Normalize tendency."""

    def __init__(self, variables, surface_variables, base_path):
        """Normalize tendency.

        Normalize tendency.

        Args:
            variables (list of strings): list of upper air variables.
            surface_variables (list): list of surface variables.
            base_path (str): base_path.

        """
        self.variables = variables
        self.surface_variables = surface_variables
        self.base_path = base_path

        # Load the NetCDF files and store the data
        self.mean = {}
        self.std = {}
        for name in self.variables + self.surface_variables:
            mean_dataset = nc.Dataset(
                f"{self.base_path}/All_NORMtend_{name}_2010_staged.mean.nc"
            )
            std_dataset = nc.Dataset(
                f"{self.base_path}/All_NORMtend_{name}_2010_staged.STD.nc"
            )
            self.mean[name] = torch.from_numpy(mean_dataset.variables[name][:])
            self.std[name] = torch.from_numpy(std_dataset.variables[name][:])

        logger.info(
            "Loading preprocessing object for transform/inverse transform tendencies into z-scores"
        )

    def transform(self, tensor, surface_tensor):
        """Transform.

        Transform input tensor/s.

        Args:
            tensor (torch tensor): batch.
            surface_tensor (torch tensor): surface batch.

        Returns:
            torch.Tensor: transformed torch tensors.

        """
        device = tensor.device

        # Apply z-score normalization using the pre-loaded mean and std
        for name in self.variables:
            mean = device_compatible_to(self.mean[name].view(1, 1, self.mean[name].size(0), 1, 1),device)
            std = device_compatible_to(self.std[name].view(1, 1, self.std[name].size(0), 1, 1),device)
            transformed_tensor = (tensor - mean) / std

        for name in self.surface_variables:
            mean = device_compatible_to(self.mean[name].view(1, 1, 1, 1),device)
            std = device_compatible_to(self.std[name].view(1, 1, 1, 1),device)
            transformed_surface_tensor = (surface_tensor - mean) / std

        return transformed_tensor, transformed_surface_tensor

    def inverse_transform(self, tensor, surface_tensor):
        """Inverse transform.

        Inverse transform input tensor/s.

        Args:
            tensor (torch tensor): batch.
            surface_tensor (torch tensor): surface batch.

        Returns:
            torch.Tensor: inverse transformed torch tensors.

        """
        device = tensor.device

        # Reverse z-score normalization using the pre-loaded mean and std
        for name in self.variables:
            mean = device_compatible_to(self.mean[name].view(1, 1, self.mean[name].size(0), 1, 1),device)
            std = device_compatible_to(self.std[name].view(1, 1, self.std[name].size(0), 1, 1),device)
            transformed_tensor = tensor * std + mean

        for name in self.surface_variables:
            mean = device_compatible_to(self.mean[name].view(1, 1, 1, 1),device)
            std = device_compatible_to(self.std[name].view(1, 1, 1, 1),device)
            transformed_surface_tensor = surface_tensor * std + mean

        return transformed_tensor, transformed_surface_tensor


class ToTensor:
    """Convert variables from xr.Datasets to Pytorch Tensors."""

    def __init__(self, conf):
        """Convert variables to rescaled tensor.

        Rescale and convert to torch tensor.

        Args:
            conf (str): path to config file.

        Attributes:
            hist_len (int): Length of
            for_len (int): state-in-state-out.
            variables (list): list of upper air variables.
            surface_variables (list): list of surface variables.
            static_variables (list): list of static variables.

        """
        self.conf = conf
        self.hist_len = int(conf["data"]["history_len"])
        self.for_len = int(conf["data"]["forecast_len"])
        self.variables = conf["data"]["variables"]
        self.surface_variables = conf["data"]["surface_variables"]
        self.allvars = self.variables + self.surface_variables
        self.static_variables = conf["data"]["static_variables"]

    def __call__(self, sample: Sample) -> Sample:
        """Convert to reshaped tensor.

        Reshape and convert to torch tensor.

        Args:
            sample (interator): batch.

        Returns:
            torch.tensor: reshaped torch tensor.

        """
        return_dict = {}

        for key, value in sample.items():
            if key == "historical_ERA5_images" or key == "x":
                self.datetime = value["time"]
                self.doy = value["time.dayofyear"]
                self.hod = value["time.hour"]

            if isinstance(value, xr.DataArray):
                value_var = value.values

            elif isinstance(value, xr.Dataset):
                surface_vars = []
                concatenated_vars = []
                for vv in self.allvars:
                    value_var = value[vv].values
                    if vv in self.surface_variables:
                        surface_vars_temp = value_var
                        surface_vars.append(surface_vars_temp)
                    else:
                        concatenated_vars.append(value_var)
                surface_vars = np.array(
                    surface_vars
                )  # [num_surf_vars, hist_len, lat, lon]
                concatenated_vars = np.array(
                    concatenated_vars
                )  # [num_vars, hist_len, num_levels, lat, lon]

            else:
                value_var = value

            if key == "historical_ERA5_images" or key == "x":
                x_surf = torch.as_tensor(surface_vars).squeeze()
                return_dict["x_surf"] = (
                    x_surf.permute(1, 0, 2, 3)
                    if len(x_surf.shape) == 4
                    else x_surf.unsqueeze(0)
                )
                # !!! there are two cases: time_frame=1 and num_variable=1, unsqueeze(0) is not always right
                # see ToTensor_ERA5_and_Forcing
                return_dict["x"] = torch.as_tensor(
                    np.hstack([np.expand_dims(x, axis=1) for x in concatenated_vars])
                )
                # [hist_len, num_vars, level, lat, lon]

            elif key == "target_ERA5_images" or key == "y":
                y_surf = torch.as_tensor(surface_vars)
                y = torch.as_tensor(
                    np.hstack([np.expand_dims(x, axis=1) for x in concatenated_vars])
                )
                return_dict["y_surf"] = y_surf.permute(1, 0, 2, 3)
                return_dict["y"] = y

        if self.static_variables:
            DSD = xr.open_dataset(self.conf["loss"]["latitude_weights"])
            arrs = []
            for sv in self.static_variables:
                if sv == "tsi":
                    TOA = xr.open_dataset(self.conf["data"]["TOA_forcing_path"])
                    times_b = pd.to_datetime(TOA.time.values)
                    mask_toa = [
                        any(
                            i == time.dayofyear and j == time.hour
                            for i, j in zip(self.doy, self.hod)
                        )
                        for time in times_b
                    ]
                    return_dict["TOA"] = torch.tensor(
                        ((TOA[sv].sel(time=mask_toa)) / 2540585.74).to_numpy()
                    ).float()  # [time, lat, lon]
                    # Need the datetime at time t(i) (which is the last element) to do multi-step training
                    return_dict["datetime"] = (
                        pd.to_datetime(self.datetime).astype(int).values[-1]
                    )

                if sv == "Z_GDS4_SFC":
                    arr = 2 * torch.tensor(
                        np.array(
                            (
                                (DSD[sv] - DSD[sv].min())
                                / (DSD[sv].max() - DSD[sv].min())
                            )
                        )
                    )
                else:
                    try:
                        arr = DSD[sv].squeeze()
                    except KeyError:
                        continue
                arrs.append(arr)

            return_dict["static"] = np.stack(arrs, axis=0)  # [num_stat_vars, lat, lon]

        return return_dict
