"""
normalize_quantile.py
-------------------------------------------------------
Content
    - BridgescalerScaleState
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
from bridgescaler import read_scaler
from credit.data import Sample, device_compatible_to

logger = logging.getLogger(__name__)

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
        return device_compatible_to(x_transformed,device)

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
        return device_compatible_to(x_transformed,device)

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
