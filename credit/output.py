'''
output.py 
-------------------------------------------------------
Content:
    - load_metadata()
    - make_xarray()
    - save_netcdf_increment()
'''

import os

import yaml
import logging
import traceback
import xarray as xr

logger = logging.getLogger(__name__)

from credit.data import drop_var_from_dataset

def load_metadata(conf):
    """
    Load metadata attributes from yaml file in credit/metadata directory
    """
    # set priorities for user-specified metadata
    if conf['predict']['metadata']:
        meta_file = conf['predict']['metadata']
        with open(meta_file) as f:
            meta_data = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        print("conf['predict']['metadata'] not given. Skip.")
        meta_data = False
        
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

    # save surface variables
    # !!! need to add diagnostic vars !!!
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
        
        # Add forecast_hour coordinate
        ds_merged['forecast_hour'] = forecast_hour

        # Add CF convention version
        ds_merged.attrs["Conventions"] = "CF-1.11"

        # Add model config file parameters (x)
        #ds_merged.attrs.update(conf)
        
        logger.info(f"Trying to save forecast hour {forecast_hour} to {nc_filename}")

        save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
        os.makedirs(save_location, exist_ok=True)
        
        unique_filename = os.path.join(save_location, f"pred_{nc_filename}_{forecast_hour:03d}.nc")

        # ---------------------------------------------------- #
        # If conf['predict']['save_vars'] provided --> drop useless vars
        if 'save_vars' in conf['predict']:
            if len(conf['predict']['save_vars']) > 0:
                ds_merged = drop_var_from_dataset(ds_merged, conf['predict']['save_vars'])

        # when there's no metafile --> meta_data = False
        if meta_data is not False:
            # Add metadata attributes to every model variable if available
            for var in ds_merged.variables:
                if var in meta_data.keys():
                    if var != 'time':
                        # use attrs.update for non-datetime variables
                        ds_merged[var].attrs.update(meta_data[var])
                    else:
                        # use time.encoding for datetime variables/coords
                        for metadata_time in meta_data['time']:
                            ds_merged.time.encoding[metadata_time] = meta_data['time'][metadata_time]
        
        # Convert to Dask array if not already
        ds_merged = ds_merged.chunk({'time': 1})
        
        # Use Dask to write the dataset in parallel
        ds_merged.to_netcdf(unique_filename, mode='w')

        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")
    except Exception:
        print(traceback.format_exc())