"""
normalize_quantile.py
-------------------------------------------------------
Content
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
from credit.data import Sample
from bridgescaler import read_scaler

logger = logging.getLogger(__name__)

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
