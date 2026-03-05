import multiprocessing as mp
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "torch"
import xarray as xr
import os
from bridgescaler import load_scaler
import pandas as pd
from keras.models import load_model
from mlguess.keras.losses import evidential_cat_loss
from metpy.calc import dewpoint_from_specific_humidity
from metpy.units import units
import logging

mp.set_start_method("spawn", force=True)


class CreditPostProcessor:
    def __init__(self):
        self.save_vars = [
            "ML_u",
            "ML_rain_ale",
            "ML_rain_epi",
            "ML_snow_ale",
            "ML_snow_epi",
            "ML_icep_ale",
            "ML_icep_epi",
            "ML_frzr_ale",
            "ML_frzr_epi",
            "ML_rain",
            "ML_crain",
            "ML_snow",
            "ML_csnow",
            "ML_icep",
            "ML_cicep",
            "ML_frzr",
            "ML_cfrzr",
        ]

    def dewpoint_temp(self, dataset: xr.Dataset):
        dpt_shape = dataset.Q_HEIGHT.shape
        dpt_values = np.empty(dpt_shape, dtype=np.float32)
        time_values = dataset.time.values

        for t_idx in range(len(time_values)):
            q_3d = dataset.Q_HEIGHT.isel(time=t_idx).values
            pressure_3d = dataset.P_HEIGHT.isel(time=t_idx).values

            dew_point_3d = dewpoint_from_specific_humidity(pressure_3d * units.Pa, q_3d * units.kg / units.kg).magnitude

            dpt_values[t_idx, ...] = dew_point_3d

        dataset["DPT_HEIGHT"] = xr.DataArray(
            dpt_values,
            dims=dataset.Q_HEIGHT.dims,
            attrs={
                "long_name": "Dew point temperature",
                "units": "C",
                "description": "Calculated from specific humidity",
            },
        )

        dataset["T_HEIGHT"] -= 273.15
        dataset["T_HEIGHT"].attrs["units"] = "C"
        return dataset

    def convert_longitude(self, lon):
        """Convert longitude from -180-180 to 0-360"""
        return lon % 360

    def subset_extent(self, nwp_data, extent, data_proj=None):
        """
        Subset data by given extent in projection coordinates.

        Args:
            nwp_data: Xr.dataset of NWP data
            extent: List of coordinates for subsetting (lon_min, lon_max, lat_min, lat_max)
            transformer: Pyproj Projection transformer object

        Returns:
            Subsetted Xr.Dataset
        """
        lon_min, lon_max, lat_min, lat_max = extent
        if data_proj is not None:
            x_coords, y_coords = data_proj(
                np.array([lon_min, lon_max], dtype=np.float64),
                np.array([lat_min, lat_max], dtype=np.float64),
            )
            subset = (
                nwp_data.swap_dims({"y": "y_projection_coordinate", "x": "x_projection_coordinate"})
                .sel(
                    y_projection_coordinate=slice(y_coords[0], y_coords[1]),
                    x_projection_coordinate=slice(x_coords[0], x_coords[1]),
                )
                .swap_dims({"y_projection_coordinate": "y", "x_projection_coordinate": "x"})
            )
        else:
            subset = nwp_data.sel(
                longitude=slice(self.convert_longitude(lon_min), self.convert_longitude(lon_max)),
                latitude=slice(lat_max, lat_min),
            )
        return subset

    def extract_variable_levels(self, dataset: xr.Dataset) -> np.ndarray:
        """
        Extracts data from an xarray dataset into a NumPy array of shape (84, lat, lon),
        where each height level is treated as a separate variable.

        Parameters:
            dataset (xr.Dataset): Input dataset with dimensions (time, height, lat, lon)
                             and variables (t, dpt, u, v).

        Returns:
            np.ndarray: Extracted data of shape (lat * long, 84).
        """
        vars = ["T_HEIGHT", "DPT_HEIGHT", "U_HEIGHT", "V_HEIGHT"]
        data = dataset[vars]

        n_vars = len(data.data_vars) * 21  # Total variables (4 vars * 21 levels)
        lat_size = data.sizes["latitude"]
        lon_size = data.sizes["longitude"]

        data_array = np.empty((n_vars, lat_size, lon_size), dtype=np.float64)

        k = 0
        for var_name in data.data_vars:
            data_array[k : k + 21, ...] = data[var_name].isel(time=0).values
            k += 21
        data_array.reshape(n_vars, -1).T
        return data_array.reshape(n_vars, -1).T

    def load_scaler(self, scaler_path):
        """
        Load bridgescaler object.

        Args:
            scaler_path: Path to scalar object.

        Returns:
            Loaded bridgescaler object
        """
        scaler = load_scaler(scaler_path)
        groups = scaler.groups_
        input_features = [x for y in groups for x in y]

        return scaler, input_features

    def load_model(self, model_path):
        return load_model(model_path, custom_objects={"loss": evidential_cat_loss})

    def transform_data(self, input_data, transformer, input_features):
        """
        Transform data for input into ML model.

        Args:
            input_data: Pandas Dataframe of input data
            transformer: Bridgescaler object used to fit data.

        Returns:
            Pandas dataframe of transformed input.
        """
        transformed_data = transformer.transform(pd.DataFrame(input_data, columns=input_features))

        return transformed_data

    def grid_predictions(self, data, predictions, output_uncertainties=False):
        """
        Populate gridded xarray dataset with ML probabilities and categorical predictions as separate variables.

        Args:
            data: Xarray dataset of input data.
            predictions: Pandas Dataframe of ML predictions.
            output_uncertainties: Boolean, whether to output uncertainties.

        Returns:
            Xarray dataset of ML predictions and surface variables on model grid.
        """
        if output_uncertainties:
            probabilities = predictions[0].cpu().numpy()  # predictions[0].numpy()
            ptype = probabilities.argmax(axis=1).reshape(-1, 1)
            u = predictions[1].cpu().numpy().reshape(data["latitude"].size, data["longitude"].size)
            data["ML_u"] = (["latitude", "longitude"], u.astype("float64"))
            data["ML_u"].attrs = {"Description": "Evidential Uncertainty (Dempster-Shafer Theory)"}
            ale = (
                predictions[2]
                .cpu()
                .numpy()
                .reshape(
                    data["latitude"].size,
                    data["longitude"].size,
                    probabilities.shape[-1],
                )
            )
            epi = (
                predictions[3]
                .cpu()
                .numpy()
                .reshape(
                    data["latitude"].size,
                    data["longitude"].size,
                    probabilities.shape[-1],
                )
            )
            for i, (var, v) in enumerate(
                zip(
                    ["Rain", "Snow", "Ice Pellets", "Freezing Rain"],
                    ["rain", "snow", "icep", "frzr"],
                )
            ):
                for uncertainty_type, long_name, short_name in zip(
                    [ale, epi], ["aleatoric", "epistemic"], ["ale", "epi"]
                ):
                    data[f"ML_{v}_{short_name}"] = (
                        ["latitude", "longitude"],
                        uncertainty_type[:, :, i].astype("float64"),
                    )
                    data[f"ML_{v}_{short_name}"].attrs = {
                        "Description": f"Machine Learned {long_name}u ncertainty of {var}"
                    }
        else:
            ptype = predictions.cpu().argmax(axis=1).reshape(-1, 1)
            probabilities = predictions.cpu()

        preds = np.hstack([probabilities, ptype])
        reshaped_preds = preds.reshape(data["latitude"].size, data["longitude"].size, preds.shape[-1])
        for i, (long_v, v) in enumerate(
            zip(
                ["rain", "snow", "ice pellets", "freezing rain"],
                ["rain", "snow", "icep", "frzr"],
            )
        ):
            data[f"ML_{v}"] = (
                ["latitude", "longitude"],
                reshaped_preds[:, :, i].astype("float64"),
            )  # ML probability
            data[f"ML_{v}"].attrs = {"Description": f"Machine Learned Probability of {long_v}"}
            data[f"ML_c{v}"] = (
                ["latitude", "longitude"],
                np.where(reshaped_preds[:, :, -1] == i, 1, 0).astype("uint8"),
            )  # ML categorical
            data[f"ML_c{v}"].attrs = {"Description": f"Machine Learned Categorical {long_v}"}

        for var in ["crain", "csnow", "cicep", "cfrzr"]:
            if var in list(data.data_vars):
                data[var] = data[var].astype("uint8")

        for v in data.coords:
            if data[v].dtype == "float32":
                data[v] = data[v].astype("float64")

        return data

    def ptype_classification(self, dataset):
        return dataset[self.save_vars].expand_dims({"time": dataset.time.values})

    def write_to_netcdf(self, dataset, nc_filename, forecast_hour, conf):
        """Saves the processed data to a NetCDF file."""

        logging.info(f"Trying to save forecast hour {forecast_hour} to {nc_filename}")

        save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
        os.makedirs(save_location, exist_ok=True)

        unique_filename = os.path.join(save_location, f"pred_{nc_filename}_{forecast_hour:03d}.nc")
        dataset.to_netcdf(unique_filename)
