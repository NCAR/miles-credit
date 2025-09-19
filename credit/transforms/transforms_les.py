"""
normalize_les.py
-------------------------------------------------------
Content
    - Normalize_LES
    - ToTensor_LES
"""

import logging
from typing import Dict

import numpy as np
import xarray as xr

import torch

logger = logging.getLogger(__name__)

class Normalize_LES:
    def __init__(self, conf):
        self.mean_ds = xr.open_dataset(conf["data"]["mean_path"]).load()
        self.std_ds = xr.open_dataset(conf["data"]["std_path"]).load()

        varnames_all = conf["data"]["all_varnames"]
        self.mean_tensors = {}
        self.std_tensors = {}

        for var in varnames_all:
            mean_array = self.mean_ds[var].values
            std_array = self.std_ds[var].values
            # convert to tensor
            self.mean_tensors[var] = torch.tensor(mean_array)
            self.std_tensors[var] = torch.tensor(std_array)

        # Get levels and upper air variables
        self.levels = conf["data"]["levels"]
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
            self.num_diagnostic = len(self.varname_diagnostic) * self.levels

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

    def __call__(self, sample, inverse: bool = False):
        if inverse:
            # Inverse transformation
            return self.inverse_transform(sample)
        else:
            # Transformation
            return self.transform(sample)

    def transform_array(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function applies to y_pred, so there won't be dynamic forcing, forcing, and static variables.
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
        # if self.flag_diagnostic:
        #     for k, name in enumerate(self.varname_diagnostic):
        #         var_mean = self.mean_tensors[name].to(device)
        #         var_std = self.std_tensors[name].to(device)
        #         transformed_diagnostic[:, k] = (
        #             transformed_diagnostic[:, k] - var_mean
        #         ) / var_std

        if self.flag_diagnostic:
            k = 0
            for name in self.varname_diagnostic:
                mean_tensor = self.mean_tensors[name].to(device)
                std_tensor = self.std_tensors[name].to(device)
                for level in range(self.levels):
                    var_mean = mean_tensor[level]
                    var_std = std_tensor[level]
                    transformed_diagnostic[:, k] = (
                        transformed_diagnostic[:, k] - var_mean
                    ) / var_std
                    k += 1
        
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
        """
        This function transforms training batches, it handles forcing & static as follows:
            - forcing & static don't need to be transformed; users should transform them and save them to the file
            - other variables (upper-air, surface, dynamic forcing, diagnostics) need to be transformed
        """
        normalized_sample = {}
        if self.has_forcing_static:
            for key, value in sample.items():
                # key: 'historical_ERA5_images', 'target_ERA5_images'
                # value: the xarray datasets
                if isinstance(value, xr.Dataset):
                    # training input
                    if key == "historical_ERA5_images" or ('input' in key):
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
        """
        This function applies to y_pred, so there won't be dynamic forcing, forcing, and static vars
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
            k = 0
            for name in self.varname_diagnostic:
                mean_tensor = self.mean_tensors[name].to(device)
                std_tensor = self.std_tensors[name].to(device)
                for level in range(self.levels):
                    mean = mean_tensor[level]
                    std = std_tensor[level]
                    transformed_diagnostic[:, k] = transformed_diagnostic[:, k] * std + mean
                    k += 1

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
        """
        Inverse transform for input x.
        Forcing and static variables are not transformed
        (they were not transformed in the transform function).
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



class ToTensor_LES:
    def __init__(self, conf):
        self.conf = conf

        # =============================================== #
        self.output_dtype = torch.float32
        # ============================================== #

        self.hist_len = int(conf["data"]["history_len"])
        self.for_len = int(conf["data"]["forecast_len"])

        # identify the existence of other variables
        self.flag_surface = conf["data"]["flag_surface"]
        self.flag_dyn_forcing = conf["data"]["flag_dyn_forcing"]
        self.flag_diagnostic = conf["data"]["flag_diagnostic"]
        self.flag_forcing = conf["data"]["flag_forcing"]
        self.flag_static = conf["data"]["flag_static"]
        self.varname_upper_air = conf["data"]["variables"]

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
            self.num_diagnostic = len(self.varname_diagnostic)

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
            self.flag_static_first = ("static_first" in conf["data"]) and (conf["data"]["static_first"])
            # ======================================================================================== #
        else:
            self.has_forcing_static = False

    def __call__(self, sample):
        return_dict = {}

        for key, value in sample.items():
            ## if DataArray
            if isinstance(value, xr.DataArray):
                var_value = value.values

            ## if Dataset
            elif isinstance(value, xr.Dataset):
                # organize upper-air vars
                list_vars_upper_air = []

                for var_name in self.varname_upper_air:
                    var_value = value[var_name].values
                    list_vars_upper_air.append(var_value)

                # [num_vars, hist_len, num_levels, lat, lon]
                numpy_vars_upper_air = np.array(list_vars_upper_air)  

                # organize surface vars
                if self.flag_surface:
                    list_vars_surface = []

                    for var_name in self.varname_surface:
                        var_value = value[var_name].values
                        list_vars_surface.append(var_value)
                        
                    # [num_surf_vars, hist_len, lat, lon]
                    numpy_vars_surface = np.array(list_vars_surface) 
                    
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

                    if key == "LES_input":
                        list_vars_forcing_static = []
                        for var_name in varname_forcing_static:
                            var_value = value[var_name].values
                            list_vars_forcing_static.append(var_value)

                        numpy_vars_forcing_static = np.array(list_vars_forcing_static)

                # organize diagnostic vars (target only)
                if self.flag_diagnostic:
                    if key == "LES_target":
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

            if key == "LES_input":
                
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

                return_dict["x"] = x_upper_air.type(self.output_dtype)

            elif key == "LES_target":
                # ---------------------------------------------------------------------- #
                # ToTensor: diagnostic
                if self.flag_diagnostic:
                    
                    # single diagnositc var
                    if self.num_diagnostic == 1:
                        y_diag = torch.as_tensor(numpy_vars_diagnostic)
                        
                    # multiple diagnositc var
                    else:
                        list_diag_collect = []
                        for var_diag in numpy_vars_diagnostic:
                            
                            # (time, level, lat, lon)
                            if len(var_diag.shape) == 4:
                                list_diag_collect.append(np.expand_dims(var_diag, axis=1))
                                
                            # (time, lat, lon)
                            elif len(var_diag.shape) == 3: 
                                list_diag_collect.append(np.expand_dims(var_diag, axis=(1, 2)))
                                
                        y_diag = torch.as_tensor(np.hstack(list_diag_collect))
                            
                    if len(y_diag.shape) == 5:
                        # (time, var, level, lat, lon)
                        pass
                    elif len(y_diag.shape) == 4:
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

                return_dict["y"] = x_upper_air.type(self.output_dtype)
        return return_dict
