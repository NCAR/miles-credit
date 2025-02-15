"""transforms.py provides transforms.

-------------------------------------------------------
Content:
    - load_transforms
    - NormalizeState
    - Normalize_ERA5_and_Forcing
    - BridgescalerScaleState
    - NormalizeState_Quantile
    - NormalizeTendency
    - ToTensor
    - ToTensor_ERA5_and_Forcing
    - NormalizeState_Quantile_Bridgescalar
    - ToTensor_BridgeScaler
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


def load_transforms(conf, scaler_only=False):
    """Load transforms.

    Args:
        conf (str): path to config
        scaler_only (bool): True --> retrun scaler; False --> return scaler and ToTensor

    Returns:
        tf.tensor: transform

    """
    # ------------------------------------------------------------------- #
    # transform class
    if conf["data"]["scaler_type"] == "quantile":
        transform_scaler = NormalizeState_Quantile(conf)
    elif conf["data"]["scaler_type"] == "quantile-cached":
        transform_scaler = NormalizeState_Quantile_Bridgescalar(conf)
    elif conf["data"]["scaler_type"] == "bridgescaler":
        transform_scaler = BridgescalerScaleState(conf)
    elif conf["data"]["scaler_type"] == "std":
        transform_scaler = NormalizeState(conf)
    elif conf["data"]["scaler_type"] == "std_new":
        transform_scaler = Normalize_ERA5_and_Forcing(conf)
    elif conf["data"]["scaler_type"] == "std_cached":
        transform_scaler = None
    else:
        logger.log("scaler type not supported check data: scaler_type in config file")
        raise

    if scaler_only:
        return transform_scaler
    # ------------------------------------------------------------------- #
    # ToTensor class
    if conf["data"]["scaler_type"] == "quantile-cached":
        # beidge scaler ToTensor
        to_tensor_scaler = ToTensor_BridgeScaler(conf)

    elif conf["data"]["scaler_type"] == "std_new" or "std_cached":
        # std_new and std_cached ToTensor
        to_tensor_scaler = ToTensor_ERA5_and_Forcing(conf)
    else:
        # the old ToTensor
        to_tensor_scaler = ToTensor(conf=conf)

    # ------------------------------------------------------------------- #
    # combine transform and ToTensor
    if transform_scaler is not None:
        transforms = [transform_scaler, to_tensor_scaler]
    else:
        transforms = [to_tensor_scaler]

    return tforms.Compose(transforms)


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

        return transformed_x.to(device)

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

        return transformed_x.to(device)


class Normalize_ERA5_and_Forcing:
    """Class to normalize ERA5 and Forcing Datasets."""

    def __init__(self, conf):
        """Normalize ERA5 and Forcing Datasets.

        Transform and normalize model inputs.

        Args:
            conf (str): path to config file.

        Attributes:
            mean_ds (xr.Dataset): xarray Dataset containing mean values for all variables.
            std_ds (xr.Dataset): xarray Dataset containing standard deviation values for all variables.
            varnames_all (list of str): list of all variables.
            levels (int): number of upper-air variable levels.
            varname_upper_air (list): list of upper air variables.
            varname_surface (list): list of surface variables.
            varname_dyn_forcing (list): list of dynamic forcing variables.
            varname_diagnostic (list): list of diagnostic variables.
            varname_forcing (list): list of forcing variables.
            varname_static (list): list of static variables.
            static_first (bool): if True, static listed before forcing variables.

        """
        self.mean_ds = xr.open_dataset(conf["data"]["mean_path"]).load()
        self.std_ds = xr.open_dataset(conf["data"]["std_path"]).load()

        varnames_all = conf["data"]["all_varnames"]
        self.mean_tensors = {}
        self.std_tensors = {}

        for var in varnames_all:
            mean_array = self.mean_ds[var].values
            std_array = self.std_ds[var].values
            # convert to tensor
            self.mean_tensors[var] = torch.tensor(mean_array)  # .float()
            self.std_tensors[var] = torch.tensor(std_array)  # .float()

        # Get levels and upper air variables
        self.levels = conf["data"]["levels"]  # It was conf['model']['levels']
        self.varname_upper_air = conf["data"]["variables"]
        self.num_upper_air = len(self.varname_upper_air) * self.levels

        # Identify the existence of other variables
        self.flag_surface = ("surface_variables" in conf["data"]) and (
            len(conf["data"]["surface_variables"]) > 0
        )
        self.flag_dyn_forcing = ("dynamic_forcing_variables" in conf["data"]) and (
            len(conf["data"]["dynamic_forcing_variables"]) > 0
        )
        self.flag_diagnostic = ("diagnostic_variables" in conf["data"]) and (
            len(conf["data"]["diagnostic_variables"]) > 0
        )
        self.flag_forcing = ("forcing_variables" in conf["data"]) and (
            len(conf["data"]["forcing_variables"]) > 0
        )
        self.flag_static = ("static_variables" in conf["data"]) and (
            len(conf["data"]["static_variables"]) > 0
        )

        # Get surface varnames
        if self.flag_surface:
            self.varname_surface = conf["data"]["surface_variables"]
            self.num_surface = len(self.varname_surface)

        # Get dynamic forcing varnames
        if self.flag_dyn_forcing:
            self.varname_dyn_forcing = conf["data"]["dynamic_forcing_variables"]
            self.num_dyn_forcing = len(self.varname_dyn_forcing)

        # Get diagnostic varnames
        if self.flag_diagnostic:
            self.varname_diagnostic = conf["data"]["diagnostic_variables"]
            self.num_diagnostic = len(self.varname_diagnostic)

        # Get forcing varnames
        if self.flag_forcing:
            self.varname_forcing = conf["data"]["forcing_variables"]
        else:
            self.varname_forcing = []

        # Get static varnames:
        if self.flag_static:
            self.varname_static = conf["data"]["static_variables"]
        else:
            self.varname_static = []

        if self.flag_forcing or self.flag_static:
            self.has_forcing_static = True
            self.num_static = len(self.varname_static)
            self.num_forcing = len(self.varname_forcing)
            self.num_forcing_static = self.num_static + self.num_forcing
            self.varname_forcing_static = self.varname_forcing + self.varname_static
            self.static_first = conf["data"]["static_first"]
        else:
            self.has_forcing_static = False

        logger.info(
            "Loading stored mean and std data for z-score-based transform and inverse transform"
        )

    def __call__(self, sample: Sample, inverse: bool = False) -> Sample:
        """Normalize ERA5 and Forcing.

        Args:
            sample: batch.
            inverse: whether to transform or inverse transform the sample.

        Returns:
            torch.tensor: transformed and normalized sample.

        """
        if inverse:
            # Inverse transformation
            return self.inverse_transform(sample)
        else:
            # Transformation
            return self.transform(sample)

    def transform_dataset(self, DS: xr.Dataset) -> xr.Dataset:
        DS = (DS - self.mean_ds)/self.std_ds
        return DS

    def transform_array(self, x: torch.Tensor) -> torch.Tensor:
        """Transform of y_pred.

        Transform via provided scaler file/s of the prediction variable.
        Dynamic forcing, forcing, and static vars not transformed.

        Args:
            x: batch.

        Returns:
            transformed x.

        """
        # Get the current device
        device = x.device

        # Subset upper air
        tensor_upper_air = x[:, : self.num_upper_air, :, :]
        transformed_upper_air = tensor_upper_air.clone()

        # Surface variables
        if self.flag_surface:
            tensor_surface = x[
                :, self.num_upper_air : (self.num_upper_air + self.num_surface), :, :
            ]
            transformed_surface = tensor_surface.clone()

        # y_pred does not have dynamic_forcing, skip this var type

        # Diagnostic variables (the very last of the stack)
        if self.flag_diagnostic:
            tensor_diagnostic = x[:, -self.num_diagnostic :, :, :]
            transformed_diagnostic = tensor_diagnostic.clone()

        # Standardize upper air variables
        # Upper air variable structure: var 1 [all levels] --> var 2 [all levels]
        k = 0
        for name in self.varname_upper_air:
            mean_tensor = self.mean_tensors[name].to(device)
            std_tensor = self.std_tensors[name].to(device)
            for level in range(self.levels):
                var_mean = mean_tensor[level]
                var_std = std_tensor[level]
                transformed_upper_air[:, k] = (
                    tensor_upper_air[:, k] - var_mean
                ) / var_std
                k += 1

        # Standardize surface variables
        if self.flag_surface:
            for k, name in enumerate(self.varname_surface):
                var_mean = self.mean_tensors[name].to(device)
                var_std = self.std_tensors[name].to(device)
                transformed_surface[:, k] = (tensor_surface[:, k] - var_mean) / var_std

        # Standardize diagnostic variables
        if self.flag_diagnostic:
            for k, name in enumerate(self.varname_diagnostic):
                var_mean = self.mean_tensors[name].to(device)
                var_std = self.std_tensors[name].to(device)
                transformed_diagnostic[:, k] = (
                    transformed_diagnostic[:, k] - var_mean
                ) / var_std

        # Concatenate everything
        if self.flag_surface:
            if self.flag_diagnostic:
                transformed_x = torch.cat(
                    (
                        transformed_upper_air,
                        transformed_surface,
                        transformed_diagnostic,
                    ),
                    dim=1,
                )
            else:
                transformed_x = torch.cat(
                    (transformed_upper_air, transformed_surface), dim=1
                )
        else:
            if self.flag_diagnostic:
                transformed_x = torch.cat(
                    (transformed_upper_air, transformed_diagnostic), dim=1
                )
            else:
                transformed_x = transformed_upper_air

        return transformed_x.to(device)

    def transform(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform training batches.

        Transform handles forcing & static as follows:
        - forcing & static don't need to be transformed; users should transform them and save them to the file
        - other variables (upper-air, surface, dynamic forcing, diagnostics) need to be transformed

        Args:
            sample: batch.

        Returns:
            transformed sample.

        """
        normalized_sample = {}
        if self.has_forcing_static:
            for key, value in sample.items():
                # key: 'historical_ERA5_images', 'target_ERA5_images'
                # value: the xarray datasets
                if isinstance(value, xr.Dataset):
                    # training input
                    if key == "historical_ERA5_images":
                        # get all the input vars
                        varname_inputs = value.keys()

                        # loop through dataset variables, handle forcing and static differently
                        for varname in varname_inputs:
                            # if forcing and static skip it, otherwise do z-score
                            if (varname in self.varname_forcing_static) is False:
                                value[varname] = (
                                    value[varname] - self.mean_ds[varname]
                                ) / self.std_ds[varname]

                        # put transformed xr.Dataset to the output dictionary
                        normalized_sample[key] = value

                    # target fields do not contain forcing and static, normalize everything
                    else:
                        normalized_sample[key] = (value - self.mean_ds) / self.std_ds

        # if there's no forcing / static, normalize everything
        else:
            for key, value in sample.items():
                if isinstance(value, xr.Dataset):
                    normalized_sample[key] = (value - self.mean_ds) / self.std_ds

        return normalized_sample

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform of y_pred.

        Inverse transform of prediction variable. Dynamic forcing, forcing,
        and static vars not transformed.

        Args:
            x: batch.

        Returns:
            inverse transformed x.

        """
        # Get the current device
        device = x.device

        # Subset upper air
        tensor_upper_air = x[:, : self.num_upper_air, :, :]
        transformed_upper_air = tensor_upper_air.clone()

        # Surface variables
        if self.flag_surface:
            tensor_surface = x[
                :, self.num_upper_air : (self.num_upper_air + self.num_surface), :, :
            ]
            transformed_surface = tensor_surface.clone()

        # Diagnostic variables (the very last of the stack)
        if self.flag_diagnostic:
            tensor_diagnostic = x[:, -self.num_diagnostic :, :, :]
            transformed_diagnostic = tensor_diagnostic.clone()

        # Reverse upper air variables
        k = 0
        for name in self.varname_upper_air:
            mean_tensor = self.mean_tensors[name].to(device)
            std_tensor = self.std_tensors[name].to(device)
            for level in range(self.levels):
                mean = mean_tensor[level]
                std = std_tensor[level]
                transformed_upper_air[:, k] = tensor_upper_air[:, k] * std + mean
                k += 1

        # Reverse surface variables
        if self.flag_surface:
            for k, name in enumerate(self.varname_surface):
                mean = self.mean_tensors[name].to(device)
                std = self.std_tensors[name].to(device)
                transformed_surface[:, k] = tensor_surface[:, k] * std + mean

        # Reverse diagnostic variables
        if self.flag_diagnostic:
            for k, name in enumerate(self.varname_diagnostic):
                mean = self.mean_tensors[name].to(device)
                std = self.std_tensors[name].to(device)
                transformed_diagnostic[:, k] = transformed_diagnostic[:, k] * std + mean

        # Concatenate everything
        if self.flag_surface:
            if self.flag_diagnostic:
                transformed_x = torch.cat(
                    (
                        transformed_upper_air,
                        transformed_surface,
                        transformed_diagnostic,
                    ),
                    dim=1,
                )
            else:
                transformed_x = torch.cat(
                    (transformed_upper_air, transformed_surface), dim=1
                )
        else:
            if self.flag_diagnostic:
                transformed_x = torch.cat(
                    (transformed_upper_air, transformed_diagnostic), dim=1
                )
            else:
                transformed_x = transformed_upper_air

        return transformed_x.to(device)

    def inverse_transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform for input x.

        Forcing and static variables are not transformed
        (they were not transformed in the transform function).

        Args:
            x: batch.

        Returns:
            transformed x.

        """
        # Get the current device
        device = x.device

        # Subset upper air variables
        tensor_upper_air = x[:, : self.num_upper_air, :, :]
        transformed_upper_air = tensor_upper_air.clone()

        idx = self.num_upper_air

        # Surface variables
        if self.flag_surface:
            tensor_surface = x[:, idx : idx + self.num_surface, :, :]
            transformed_surface = tensor_surface.clone()
            idx += self.num_surface

        # Dynamic forcing variables
        if self.flag_dyn_forcing:
            if self.static_first:
                tensor_dyn_forcing = x[
                    :,
                    idx + self.num_static : idx
                    + self.num_static
                    + self.num_dyn_forcing,
                    :,
                    :,
                ]
            else:
                tensor_dyn_forcing = x[:, idx : idx + self.num_dyn_forcing, :, :]
                idx += self.num_dyn_forcing

            transformed_dyn_forcing = tensor_dyn_forcing.clone()

        # Forcing and static variables (not transformed)
        if self.has_forcing_static:
            if self.static_first:
                tensor_static = x[:, idx : idx + self.num_static, :, :]
                tensor_forcing = x[:, -self.num_forcing :, :, :]
            else:
                tensor_forcing = x[:, idx : idx + self.num_forcing, :, :]
                tensor_static = x[:, idx : idx + self.num_static, :, :]

        # Inverse transform upper air variables
        k = 0
        for name in self.varname_upper_air:
            mean_tensor = self.mean_tensors[name].to(device)
            std_tensor = self.std_tensors[name].to(device)
            for level in range(self.levels):
                mean = mean_tensor[level]
                std = std_tensor[level]
                transformed_upper_air[:, k] = tensor_upper_air[:, k] * std + mean
                k += 1

        # Inverse transform surface variables
        if self.flag_surface:
            for k, name in enumerate(self.varname_surface):
                mean = self.mean_tensors[name].to(device)
                std = self.std_tensors[name].to(device)
                transformed_surface[:, k] = tensor_surface[:, k] * std + mean

        # Inverse transform dynamic forcing variables
        if self.flag_dyn_forcing:
            for k, name in enumerate(self.varname_dyn_forcing):
                mean = self.mean_tensors[name].to(device)
                std = self.std_tensors[name].to(device)
                transformed_dyn_forcing[:, k] = tensor_dyn_forcing[:, k] * std + mean

        # Reconstruct input tensor
        tensors = [transformed_upper_air]

        if self.flag_surface:
            tensors.append(transformed_surface)

        if self.flag_dyn_forcing and self.has_forcing_static:
            if self.static_first:
                tensors.append(tensor_static)
                tensors.append(transformed_dyn_forcing)
                tensors.append(tensor_forcing)
            else:
                tensors.append(transformed_dyn_forcing)
                tensors.append(tensor_forcing)
                tensors.append(tensor_static)

        elif self.has_forcing_static:
            if self.static_first:
                tensors.append(tensor_static)
                tensors.append(tensor_forcing)
            else:
                tensors.append(tensor_forcing)
                tensors.append(tensor_static)

        elif self.flag_dyn_forcing:
            tensors.append(transformed_dyn_forcing)

        transformed_x = torch.cat(tensors, dim=1)

        return transformed_x.to(device)


class BridgescalerScaleState(object):
    """Convert to rescaled tensor using Bridgescaler."""

    def __init__(self, conf):
        """Convert to rescaled tensor.

        Rescale and convert to torch tensor.

        Args:
            conf (str): path to config file.

        Attributes:
            scaler_file (str): path to scaler file.
            variables (list): list of upper air variables.
            surface_variables (list): list of surface variables.
            level_ids (list of ints): level ids.
            n_levels (int): number of upper-air variable levels.

        """
        self.scaler_file = conf["data"]["quant_path"]
        self.variables = conf["data"]["variables"]
        self.surface_variables = conf["data"]["surface_variables"]
        if "level_ids" in conf["data"].keys():
            self.level_ids = conf["data"]["level_ids"]
        else:
            self.level_ids = np.array(
                [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137],
                dtype=np.int64,
            )
        self.n_levels = int(conf["model"]["levels"])
        self.var_levels = []
        for variable in self.variables:
            for level in self.level_ids:
                self.var_levels.append(f"{variable}_{level:d}")
        self.n_surface_variables = len(self.surface_variables)
        self.n_3dvar_levels = len(self.variables) * self.n_levels
        self.scaler_df = pd.read_parquet(self.scaler_file)
        self.scaler_3d = np.sum(self.scaler_df["scaler_3d"].apply(read_scaler))
        self.scaler_surf = np.sum(self.scaler_df["scaler_surface"].apply(read_scaler))

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform.

        Inverse transform.

        Args:
            x: batch.

        Returns:
            inverse transformed batch.

        """
        device = x.device
        x_3d = x[:, : self.n_3dvar_levels].cpu()
        x_surface = x[:, self.n_3dvar_levels :].cpu()
        x_3d_transformed = x_3d.clone()
        x_surface_transformed = x_surface.clone()
        x_3d_da = xr.DataArray(
            x_3d.numpy(),
            dims=("time", "variable", "latitude", "longitude"),
            coords=dict(variable=self.var_levels),
        )
        x_3d_transformed.numpy()[:] = self.scaler_3d.inverse_transform(
            x_3d_da, channels_last=False
        ).values
        x_surface_da = xr.DataArray(
            x_surface.numpy(),
            dims=("time", "variable", "latitude", "longitude"),
            coords=dict(variable=self.surface_variables),
        )
        x_surface_transformed.numpy()[:] = self.scaler_surf.inverse_transform(
            x_surface_da, channels_last=False
        ).values
        x_transformed = torch.cat((x_3d_transformed, x_surface_transformed), dim=1)
        return x_transformed.to(device)

    def transform_array(self, x: torch.Tensor) -> torch.Tensor:
        """Transform.

        Transform.

        Args:
            x: batch.

        Returns:
            transformed batch.

        """
        device = x.device
        x_3d = x[:, : self.n_3dvar_levels].cpu()
        x_surface = x[:, self.n_3dvar_levels :].cpu()
        x_3d_transformed = x_3d.clone()
        x_surface_transformed = x_surface.clone()
        x_3d_da = xr.DataArray(
            x_3d.numpy(),
            dims=("time", "variable", "latitude", "longitude"),
            coords=dict(variable=self.var_levels),
        )
        x_3d_transformed.numpy()[:] = self.scaler_3d.transform(
            x_3d_da, channels_last=False
        ).values
        x_surface_da = xr.DataArray(
            x_surface.numpy(),
            dims=("time", "variable", "latitude", "longitude"),
            coords=dict(variable=self.surface_variables),
        )
        x_surface_transformed.numpy()[:] = self.scaler_surf.transform(
            x_surface_da, channels_last=False
        ).values
        x_transformed = torch.cat((x_3d_transformed, x_surface_transformed), dim=1)
        return x_transformed.to(device)

    def transform(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform.

        Transform.

        Args:
            sample: batch.

        Returns:
            transformed batch.

        """
        normalized_sample = {}
        for data_id, ds in sample.items():
            if isinstance(ds, xr.Dataset):
                normalized_sample[data_id] = xr.Dataset()
                for variable in self.variables:
                    single_var = ds[variable]
                    single_var["level"] = [f"{variable}_{lev:d}" for lev in ds["level"]]
                    transformed_var = self.scaler_3d.transform(
                        single_var, channels_last=False
                    )
                    transformed_var["level"] = ds["level"]
                    normalized_sample[data_id][variable] = transformed_var
                surface_ds = (
                    ds[self.surface_variables]
                    .to_dataarray()
                    .transpose("time", "variable", "latitude", "longitude")
                )
                surface_ds_transformed = self.scaler_surf.transform(
                    surface_ds, channels_last=False
                )
                normalized_sample[data_id] = normalized_sample[data_id].merge(
                    surface_ds_transformed.to_dataset(dim="variable")
                )
        return normalized_sample


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
        transformed_tensor[:, :, :, :] = torch.tensor(
            (self.scaler_3d.inverse_transform(rscal_3d))
        ).to(device)
        # surf
        rscal_surf = np.array(x[:, (len(self.variables) * self.levels) :, :, :])
        transformed_surface_tensor[:, :, :, :] = torch.tensor(
            (self.scaler_surf.inverse_transform(rscal_surf))
        ).to(device)
        # cat them
        transformed_x = torch.cat(
            (transformed_tensor, transformed_surface_tensor), dim=1
        )
        # return
        return transformed_x.to(device)

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
            mean = self.mean[name].view(1, 1, self.mean[name].size(0), 1, 1).to(device)
            std = self.std[name].view(1, 1, self.std[name].size(0), 1, 1).to(device)
            transformed_tensor = (tensor - mean) / std

        for name in self.surface_variables:
            mean = self.mean[name].view(1, 1, 1, 1).to(device)
            std = self.std[name].view(1, 1, 1, 1).to(device)
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
            mean = self.mean[name].view(1, 1, self.mean[name].size(0), 1, 1).to(device)
            std = self.std[name].view(1, 1, self.std[name].size(0), 1, 1).to(device)
            transformed_tensor = tensor * std + mean

        for name in self.surface_variables:
            mean = self.mean[name].view(1, 1, 1, 1).to(device)
            std = self.std[name].view(1, 1, 1, 1).to(device)
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


class ToTensor_ERA5_and_Forcing:
    """Class to convert ERA5 and Forcing Datasets to torch tensor."""

    def __init__(self, conf):
        """Convert variables to input/output torch tensors.

        Convert variables from config file to proper model inputs.

        Args:
            conf (str): path to config file.

        Attributes:
            hist_len (bool): state-in-state-out.
            for_len (bool): state-in-state-out.
            varname_upper_air (str): list of upper air variables.
            varname_surface (list): list of surface variables.
            varname_dyn_forcing (list): list of dynamic forcing variables.
            varname_diagnostic (list): list of diagnostic variables.
            varname_forcing (list): list of forcing variables.
            varname_static (list): list of static variables.
            flag_static_first (bool): if True, static listed before forcing variables.

        """
        self.conf = conf

        # =============================================== #
        self.output_dtype = torch.float32
        # ============================================== #

        self.hist_len = int(conf["data"]["history_len"])
        self.for_len = int(conf["data"]["forecast_len"])

        # identify the existence of other variables
        self.flag_surface = ("surface_variables" in conf["data"]) and (
            len(conf["data"]["surface_variables"]) > 0
        )
        self.flag_dyn_forcing = ("dynamic_forcing_variables" in conf["data"]) and (
            len(conf["data"]["dynamic_forcing_variables"]) > 0
        )
        self.flag_diagnostic = ("diagnostic_variables" in conf["data"]) and (
            len(conf["data"]["diagnostic_variables"]) > 0
        )
        self.flag_forcing = ("forcing_variables" in conf["data"]) and (
            len(conf["data"]["forcing_variables"]) > 0
        )
        self.flag_static = ("static_variables" in conf["data"]) and (
            len(conf["data"]["static_variables"]) > 0
        )

        self.varname_upper_air = conf["data"]["variables"]
        self.flag_upper_air = True

        # get surface varnames
        if self.flag_surface:
            self.varname_surface = conf["data"]["surface_variables"]

        # get dynamic forcing varnames
        self.num_forcing_static = 0

        if self.flag_dyn_forcing:
            self.varname_dyn_forcing = conf["data"]["dynamic_forcing_variables"]
            self.num_forcing_static += len(self.varname_dyn_forcing)
        else:
            self.varname_dyn_forcing = []

        # get diagnostic varnames
        if self.flag_diagnostic:
            self.varname_diagnostic = conf["data"]["diagnostic_variables"]

        # get forcing varnames
        if self.flag_forcing:
            self.varname_forcing = conf["data"]["forcing_variables"]
            self.num_forcing_static += len(self.varname_forcing)
        else:
            self.varname_forcing = []

        # get static varnames:
        if self.flag_static:
            self.varname_static = conf["data"]["static_variables"]
            self.num_forcing_static += len(self.varname_static)
        else:
            self.varname_static = []

        if self.flag_forcing or self.flag_static:
            self.has_forcing_static = True
            # ======================================================================================== #
            # forcing variable first (new models) vs. static variable first (some old models)
            # this flag makes sure that the class is compatible with some old CREDIT models
            self.flag_static_first = ("static_first" in conf["data"]) and (
                conf["data"]["static_first"]
            )
            # ======================================================================================== #
        else:
            self.has_forcing_static = False

    def __call__(self, sample: Sample) -> Sample:
        """Convert variables to input/output torch tensors.

        Args:
            sample (interator): batch.

        Returns:
            torch.tensor: converted torch tensor.

        """
        return_dict = {}

        for key, value in sample.items():
            ## if DataArray
            if isinstance(value, xr.DataArray):
                var_value = value.values

            ## if Dataset
            elif isinstance(value, xr.Dataset):
                # organize upper-air vars
                list_vars_upper_air = []
                self.flag_upper_air = True
                # check if upper air in dataset
                dataset_vars = list(value.data_vars)

                self.flag_upper_air = all(
                    [varname in dataset_vars for varname in self.varname_upper_air]
                )
                if self.flag_upper_air:
                    for var_name in self.varname_upper_air:
                        var_value = value[var_name].values
                        list_vars_upper_air.append(var_value)
                    numpy_vars_upper_air = np.array(
                        list_vars_upper_air
                    )  # [num_vars, hist_len, num_levels, lat, lon]

                self.flag_surface = all(
                    [varname in dataset_vars for varname in self.varname_surface]
                )

                # organize surface vars
                if self.flag_surface:
                    list_vars_surface = []

                    for var_name in self.varname_surface:
                        var_value = value[var_name].values
                        list_vars_surface.append(var_value)

                    numpy_vars_surface = np.array(
                        list_vars_surface
                    )  # [num_surf_vars, hist_len, lat, lon]

                # !!! DO NOT DELETE !!!
                # this is the space if we plan to create an independent key for dynamic forcing
                # right now it is part of the varname_forcing_static so this part is comment-out
                # ------------------------------------------------------------------------------ #
                # # organize dynamic forcing vars (input only)
                # if self.flag_dyn_forcing:
                #     if key == 'historical_ERA5_images' or key == 'x':
                #         list_vars_dyn_forcing = []

                #         for var_name in self.varname_dyn_forcing:
                #             var_value = value[var_name].values
                #             list_vars_dyn_forcing.append(var_value)

                #         numpy_vars_dyn_forcing = np.array(list_vars_dyn_forcing)

                # organize forcing and static (input only)
                if self.has_forcing_static or self.flag_dyn_forcing:
                    # enter this scope if one of the (dyn_forcing, folrcing, static) exists
                    if self.flag_static_first:
                        varname_forcing_static = (
                            self.varname_static
                            + self.varname_dyn_forcing
                            + self.varname_forcing
                        )
                    else:
                        varname_forcing_static = (
                            self.varname_dyn_forcing
                            + self.varname_forcing
                            + self.varname_static
                        )

                    if key == "historical_ERA5_images" or key == "x":
                        list_vars_forcing_static = []
                        for var_name in varname_forcing_static:
                            var_value = value[var_name].values
                            list_vars_forcing_static.append(var_value)

                        numpy_vars_forcing_static = np.array(list_vars_forcing_static)

                # organize diagnostic vars (target only)
                if self.flag_diagnostic:
                    if key == "target_ERA5_images" or key == "y":
                        list_vars_diagnostic = []
                        for var_name in self.varname_diagnostic:
                            var_value = value[var_name].values
                            list_vars_diagnostic.append(var_value)

                        numpy_vars_diagnostic = np.array(list_vars_diagnostic)

            ## if numpy
            else:
                var_value = value

            # ---------------------------------------------------------------------- #
            # ToTensor: upper-air varialbes
            ## produces [time, upper_var, level, lat, lon]
            ## np.hstack concatenates the second dim (axis=1)
            if self.flag_upper_air:
                x_upper_air = np.hstack(
                    [
                        np.expand_dims(var_upper_air, axis=1)
                        for var_upper_air in numpy_vars_upper_air
                    ]
                )
                x_upper_air = torch.as_tensor(x_upper_air)

            # ---------------------------------------------------------------------- #
            # ToTensor: surface variables
            if self.flag_surface:
                # this line produces [surface_var, time, lat, lon]
                x_surf = torch.as_tensor(numpy_vars_surface).squeeze()

                if len(x_surf.shape) == 4:
                    # permute: [surface_var, time, lat, lon] --> [time, surface_var, lat, lon]
                    x_surf = x_surf.permute(1, 0, 2, 3)

                # =============================================== #
                # separate single variable vs. single history_len
                elif len(x_surf.shape) == 3:
                    if len(self.varname_surface) > 1:
                        # single time, multi-vars
                        x_surf = x_surf.unsqueeze(0)
                    else:
                        # multi-time, single vars
                        x_surf = x_surf.unsqueeze(1)
                # =============================================== #

                else:
                    # num_var=1, time=1, only has lat, lon
                    x_surf = x_surf.unsqueeze(0).unsqueeze(0)

            if key == "historical_ERA5_images" or key == "x":
                # !!! DO NOT DELETE !!!
                # this is the space if we plan to create an independent key for dynamic forcing
                # right now it is part of the 'x_forcing_static' so this part is comment-out
                # # ---------------------------------------------------------------------- #
                # # ToTensor: dynamic forcing
                # if self.flag_dyn_forcing:
                #     x_dyn_forcing = torch.as_tensor(dyn_forcing).squeeze()

                #     if len(x_dyn_forcing.shape) == 4:
                #         # [dyn_forcing_var, time, lat, lon] --> [time, dyn_forcing_var, lat, lon]

                #     # =============================================== #
                #     # separate single variable vs. single history_len
                #     elif len(x_dyn_forcing.shape) == 3:
                #         if len(self.varname_dyn_forcing) > 1:
                #             # single time, multi-vars
                #             x_dyn_forcing = x_dyn_forcing.unsqueeze(0)
                #         else:
                #             # multi-time, single vars
                #             x_dyn_forcing = x_dyn_forcing.unsqueeze(1)
                #     # =============================================== #

                #     else:
                #         # single var and single time, unsqueeze both
                #         x_dyn_forcing = x_dyn_forcing.unsqueeze(0).unsqueeze(0)

                # ---------------------------------------------------------------------- #
                # ToTensor: forcing and static
                if self.has_forcing_static:
                    # this line produces [forcing_var, time, lat, lon]
                    x_static = torch.as_tensor(numpy_vars_forcing_static).squeeze()

                    if len(x_static.shape) == 4:
                        # permute: [forcing_var, time, lat, lon] --> [time, forcing_var, lat, lon]
                        x_static = x_static.permute(1, 0, 2, 3)

                    elif len(x_static.shape) == 3:
                        if self.num_forcing_static > 1:
                            # single time, multi-vars
                            x_static = x_static.unsqueeze(0)
                        else:
                            # multi-time, single vars
                            x_static = x_static.unsqueeze(1)
                    else:
                        # num_var=1, time=1, only has lat, lon
                        x_static = x_static.unsqueeze(0).unsqueeze(0)
                        # x_static = x_static.unsqueeze(1)

                    return_dict["x_forcing_static"] = x_static.type(self.output_dtype)

                if self.flag_surface:
                    return_dict["x_surf"] = x_surf.type(self.output_dtype)
                if self.flag_upper_air:
                    return_dict["x"] = x_upper_air.type(self.output_dtype)

            elif key == "target_ERA5_images" or key == "y":
                # ---------------------------------------------------------------------- #
                # ToTensor: diagnostic
                if self.flag_diagnostic:
                    # this line produces [forcing_var, time, lat, lon]
                    y_diag = torch.as_tensor(numpy_vars_diagnostic).squeeze()

                    if len(y_diag.shape) == 4:
                        # permute: [diag_var, time, lat, lon] --> [time, diag_var, lat, lon]
                        y_diag = y_diag.permute(1, 0, 2, 3)

                    # =============================================== #
                    # separate single variable vs. single history_len
                    elif len(y_diag.shape) == 3:
                        if len(self.varname_diagnostic) > 1:
                            # single time, multi-vars
                            y_diag = y_diag.unsqueeze(0)
                        else:
                            # multi-time, single vars
                            y_diag = y_diag.unsqueeze(1)
                    # =============================================== #

                    else:
                        # num_var=1, time=1, only has lat, lon
                        y_diag = y_diag.unsqueeze(0).unsqueeze(0)

                    return_dict["y_diag"] = y_diag.type(self.output_dtype)

                if self.flag_surface:
                    return_dict["y_surf"] = x_surf.type(self.output_dtype)
                if self.flag_upper_air:
                    return_dict["y"] = x_upper_air.type(self.output_dtype)
        return return_dict


class NormalizeState_Quantile_Bridgescalar:
    """Class to use the bridgescaler Quantile functionality.

    Some hoops have to be jumped thorugh, and the efficiency could be
    improved if we were to retrain the bridgescaler.
    """

    def __init__(self, conf):
        """Normalize via quantile bridgescaler.

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
        """Normalize via quantile transform with bridgescaler.

        Normalize via provided scaler file/s.

        Args:
            sample (iterator): batch.

        Returns:
            torch.tensor: transformed torch tensor.

        """
        if inverse:
            return self.inverse_transform(sample)
        else:
            return self.transform(sample)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform.

        Inverse transform via provided scaler file/s.

        Args:
            x: batch.

        Returns:
            inverse transformed torch tensor.

        """
        device = x.device
        tensor = x[:, : (len(self.variables) * self.levels), :, :]  # B, Var, H, W
        surface_tensor = x[
            :, (len(self.variables) * self.levels) :, :, :
        ]  # B, Var, H, W
        # Reverse quantile transform using bridge scaler:
        transformed_tensor = tensor.clone()
        transformed_surface_tensor = surface_tensor.clone()
        # 3dvars
        rscal_3d = np.array(x[:, : (len(self.variables) * self.levels), :, :])

        transformed_tensor[:, :, :, :] = torch.tensor(
            (self.scaler_3d.inverse_transform(rscal_3d))
        ).to(device)
        # surf
        rscal_surf = np.array(x[:, (len(self.variables) * self.levels) :, :, :])
        transformed_surface_tensor[:, :, :, :] = torch.tensor(
            (self.scaler_surf.inverse_transform(rscal_surf))
        ).to(device)
        # cat them
        transformed_x = torch.cat(
            (transformed_tensor, transformed_surface_tensor), dim=1
        )
        # return
        return transformed_x.to(device)

    def transform(self, sample):
        """Transform.

        Transform via provided scaler file/s.

        Args:
            sample (iterator): batch.

        Returns:
            torch.Tensor: transformed torch tensor.

        """
        normalized_sample = {}
        for key, value in sample.items():
            normalized_sample[key] = value
        return normalized_sample


class ToTensor_BridgeScaler:
    """Convert to reshaped tensor."""

    def __init__(self, conf):
        """Convert to reshaped tensor.

        Reshape and convert to torch tensor.

        Args:
            conf (str): path to config file.

        Attributes:
            hist_len (bool): state-in-state-out.
            for_len (bool): state-in-state-out.
            variables (list): list of upper air variables.
            surface_variables (list): list of surface variables.
            static_variables (list): list of static variables.
            latN (int): number of latitude grids (default: 640).
            lonN (int): number of longitude grids (default: 1280).
            levels (int): number of upper-air variable levels.
            one_shot (bool): one shot.

        """
        self.conf = conf
        self.hist_len = int(conf["data"]["history_len"])
        self.for_len = int(conf["data"]["forecast_len"])
        self.variables = conf["data"]["variables"]
        self.surface_variables = conf["data"]["surface_variables"]
        self.allvars = self.variables + self.surface_variables
        self.static_variables = conf["data"]["static_variables"]
        self.latN = int(conf["model"]["image_height"])
        self.lonN = int(conf["model"]["image_width"])
        self.levels = int(conf["model"]["levels"])
        self.one_shot = conf["data"]["one_shot"]

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
            if key == "historical_ERA5_images":
                self.datetime = value["time"]
                self.doy = value["time.dayofyear"]
                self.hod = value["time.hour"]

            if key == "historical_ERA5_images" or key == "x":
                x_surf = torch.tensor(np.array(value["surface"])).squeeze()
                return_dict["x_surf"] = (
                    x_surf if len(x_surf.shape) == 4 else x_surf.unsqueeze(0)
                )
                len_vars = len(self.variables)
                return_dict["x"] = torch.tensor(
                    np.reshape(
                        np.array(value["levels"]),
                        [self.hist_len, len_vars, self.levels, self.latN, self.lonN],
                    )
                )

            elif key == "target_ERA5_images" or key == "y":
                y_surf = torch.tensor(np.array(value["surface"])).squeeze()
                return_dict["y_surf"] = (
                    y_surf if len(y_surf.shape) == 4 else y_surf.unsqueeze(0)
                )
                len_vars = len(self.variables)
                if self.one_shot:
                    return_dict["y"] = torch.tensor(
                        np.reshape(
                            np.array(value["levels"]),
                            [1, len_vars, self.levels, self.latN, self.lonN],
                        )
                    )
                else:
                    return_dict["y"] = torch.tensor(
                        np.reshape(
                            np.array(value["levels"]),
                            [
                                self.for_len + 1,
                                len_vars,
                                self.levels,
                                self.latN,
                                self.lonN,
                            ],
                        )
                    )

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
                    )
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

            return_dict["static"] = np.stack(arrs, axis=0)

        return return_dict
