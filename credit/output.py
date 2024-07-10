import xarray as xr
import yaml
from os.path import join
import os
import traceback
import logging
logger = logging.getLogger(__name__)


def load_metadata(dataset="era5"):
    """
    Load metadata attributes from yaml file in credit/metadata directory

    Args:
        dataset (str): `era5` or `conus404` (TBD)

    Returns:

    """
    meta_file = join(__file__, "metadata", dataset + ".yaml")
    with open(meta_file) as f:
        meta_data = yaml.load(f, Loader=yaml.SafeLoader)
    return meta_data


def split_and_reshape(tensor, conf):
    """
    Split the output tensor of the model to upper air variables and diagnostics/surface variables.

    Upperair level arrangement: top-of-atmosphere--> near-surface --> single layer
    An example: U (top-of-atmosphere) --> U (near-surface) --> V (top-of-atmosphere) --> V (near-surface)
    The shape of the output tensor is (variables, latitude, longitude)

    Args:
        tensor: PyTorch Tensor containing output of the AI NWP model
        conf: config file for the model

    """

    # get the number of levels
    levels = conf["model"]["levels"]

    # get number of channels
    channels = len(conf["data"]["variables"])
    single_level_channels = len(conf["data"]["surface_variables"])

    # subset upper air variables
    tensor_upper_air = tensor[:, : int(channels * levels), :, :]

    shape_upper_air = tensor_upper_air.shape
    tensor_upper_air = tensor_upper_air.view(
        shape_upper_air[0], channels, levels, shape_upper_air[-2], shape_upper_air[-1]
    )

    # subset surface variables
    tensor_single_level = tensor[:, -int(single_level_channels):, :, :]

    # return x, surf for B, c, lat, lon output
    return tensor_upper_air, tensor_single_level


def make_xarray(pred, forecast_datetime, lat, lon, conf):

    # subset upper air and surface variables
    tensor_upper_air, tensor_single_level = split_and_reshape(pred, conf)

    # save upper air variables
    darray_upper_air = xr.DataArray(
        tensor_upper_air,
        dims=["time", "vars", "level", "lat", "lon"],
        coords=dict(
            vars=conf["data"]["variables"],
            time=[forecast_datetime],
            level=range(conf["model"]["levels"]),
            lat=lat,
            lon=lon,
        ),
    )

    # save diagnostics and surface variables
    darray_single_level = xr.DataArray(
        tensor_single_level.squeeze(2),
        dims=["time", "vars", "lat", "lon"],
        coords=dict(
            vars=conf["data"]["surface_variables"],
            time=[forecast_datetime],
            lat=lat,
            lon=lon,
        ),
    )
    # return x-arrays as outputs
    return darray_upper_air, darray_single_level


def save_netcdf_increment(darray_upper_air, darray_single_level, nc_filename, forecast_hour, meta_data, conf):
    try:
        """
        Save increment to a unique NetCDF file using Dask for parallel processing.
        """
        # Convert DataArrays to Datasets
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_single = darray_single_level.to_dataset(dim="vars")

        # Merge datasets
        ds_merged = xr.merge([ds_upper, ds_single])

        # Add metadata attributes to every model variable if available
        for var in ds_merged.variables:
            if var in meta_data.keys():
                ds_merged[var].attrs.update(meta_data[var])

        # Add forecast_hour coordinate
        ds_merged['forecast_hour'] = forecast_hour

        # Add CF convention version
        ds_merged.attrs["Conventions"] = "CF-1.11"

        # Add model config file parameters
        ds_merged.attrs.update(conf)

        logger.info(f"Trying to save forecast hour {forecast_hour} to {nc_filename}")

        save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
        os.makedirs(save_location, exist_ok=True)
        unique_filename = os.path.join(save_location, f"pred_{nc_filename}_{forecast_hour:03d}.nc")

        # Convert to Dask array if not already
        ds_merged = ds_merged.chunk({'datetime': 1})

        # Use Dask to write the dataset in parallel
        ds_merged.to_netcdf(unique_filename, mode='w')

        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")
    except Exception:
        print(traceback.format_exc())