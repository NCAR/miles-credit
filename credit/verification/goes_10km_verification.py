from functools import partial
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from credit.verification.standard import radial_fft_spectrum
from scores.spatial import fss_2d, fss_2d_binary



logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

latitude_slices = {
    "global": slice(-91, 91),
    "s_extratropics": slice(-91, -24.5),
    "tropics": slice(-24.5, 24.5),
    "n_extratropics": slice(24.5, 91),
}

def verification(forecast_save_loc, p, conf, model_conf, dataset, climo, **kwargs):
    """
    Verification for goes cloud prediction. returns a list of dicts
    
    :param forecast_save_loc: Description
    :param conf: Description
    :param model_conf: Description
    :param p: Description
    :param kwargs: Description

    """

    forecast_files = sorted([f for f in Path(forecast_save_loc).iterdir() if '.nc' in f.name])
    step_file_tuples = enumerate(forecast_files, start=1)

    f = partial(verification_per_timestep, dataset, climo, conf)
    result = p.map(f, step_file_tuples)

    return result

def verification_per_timestep(dataset, climo, eval_conf, step_file_tuple):
    """
    Compute verification metrics for a single forecast hour.
    
    Calculates standard forecast verification scores comparing predictions to 
    observations, including error metrics, spectral analysis, and skill scores
    relative to climatology.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Ground truth/observation dataset containing the "y" variable with 
        dimensions (time, ..., latitude, longitude). Must be indexed by time 
        and contain variable "y".
    climo : xarray.DataArray
        Climatological reference forecast with dimensions (channel, latitude, 
        longitude). Used to compute skill scores.
    step_file_tuple : tuple of (int, str)
        Tuple containing:
        - step : int
            Forecast step/lead time
        - file : str
            Path to forecast file containing "BT_or_R" variable with dimensions
            (t, channel, latitude, longitude)
    
    Returns
    -------
    result_dict : dict
        Dictionary containing verification metrics with keys:
        - "forecast_hour" : int
            The forecast lead time
        - "ME_{channel}" : float
            Mean Error for each channel (bias)
        - "MAE_{channel}" : float
            Mean Absolute Error for each channel
        - "MSE_{channel}" : float
            Root Mean Squared Error (RMSE) for each channel
        - "FFT_{channel}_{wavenumber}" : float
            Radially averaged 2D power spectrum of predictions
        - "FFT_true_{channel}_{wavenumber}" : float
            Radially averaged 2D power spectrum of observations
        - "MAESS_{channel}" : float
            Mean Absolute Error Skill Score relative to climatology
            (positive values indicate skill above climatology)
    
    Notes
    -----
    - All spatial means are computed over latitude and longitude dimensions
    - FFT spectra are computed using 10.0 km grid spacing
    - MAESS is calculated as: (MAE_climo - MAE_forecast) / MAE_climo
      where positive values indicate improvement over climatology
    - Metrics are unpacked per channel using the unpack_da_to_dict helper function
    
    Examples
    --------
    >>> results = verification_per_timestep(
    ...     dataset=obs_dataset,
    ...     climo=climatology,
    ...     fh_file_tuple=(24, "forecast_24h.nc")
    ... )
    >>> print(results["forecast_hour"])
    24
    >>> print(results["MAE_C04"])
    0.45
    """

    step, file = step_file_tuple
    result_dict = {"forecast_step": step}
    
    pred_da = xr.open_dataset(file)["BT_or_R"] # BT_or_R with t channel lat lon
    # if "ensemble_member_label" in pred_da.coordinates:
    #     # do ensemble verification
    #     # spread


    #     pred_da = pred_da.mean("ensemble_member_labels")


    true_da = dataset[pred_da.t[0], "y"]["y"]

    w_lat = np.cos(np.deg2rad(pred_da.latitude))

    diff = true_da - pred_da

    # mean error
    me = diff.mean(dim=["t", "latitude", "longitude"])
    result_dict = result_dict | unpack_da_to_dict(me, "ME")

    # MAE
    mae = np.abs(diff).mean(dim=["t", "latitude", "longitude"])
    result_dict = result_dict | unpack_da_to_dict(mae, "MAE")

    # RMSE (mean then sqrt)
    mse = np.sqrt((diff ** 2).mean(dim=["t", "latitude", "longitude"]))
    result_dict = result_dict | unpack_da_to_dict(mse, "RMSE")

    # 2D FFT
    fft_pred = radial_fft_spectrum(pred_da, 10.0)
    result_dict = result_dict | unpack_da_to_dict(fft_pred, "FFT")
    fft_true = radial_fft_spectrum(true_da, 10.0)
    result_dict = result_dict | unpack_da_to_dict(fft_true, "FFT_true")

    # MAESS compared to climo
    # S = (x - x_r) / (x_p - x_r)
    mae_climo = np.abs(climo - true_da).mean(dim=["latitude", "longitude"])
    maess = (mae - mae_climo) / (-1 * mae_climo)
    result_dict = result_dict | unpack_da_to_dict(maess, "MAESS")

    # FSS for channel 13
    if eval_conf["compute_fss"]:
        channel = 13
        pred_da = pred_da.isel(t=0).sel(channel=channel) 
        target_da = true_da.sel(channel=channel)

        def is_in_bin(da, bin_edges):
            """
            outputs a masked dataarray of where the values are within the bin_edges
            """
            return (bin_edges[0] < da) & (da <= bin_edges[1])

        for window_size in eval_conf["fss_window_sizes"]:
            # binary threshold fss
            for threshold in eval_conf["C13_thresholds"]:
                fss = fss_2d(pred_da, target_da,
                            event_threshold=threshold,
                            window_size=(window_size, window_size),
                            spatial_dims=("latitude", "longitude"),
                            threshold_operator=np.less
                            )
                result_dict = result_dict | {f"FSS_WS{window_size}_C13_T{threshold}": float(fss.values)}

            # categorical fss
            for category_name, bin in eval_conf["sky_categories"].items():
                pred_binary = is_in_bin(pred_da, bin)
                target_binary = is_in_bin(target_da, bin)
                
                fss = fss_2d_binary(pred_binary, target_binary,
                        window_size=(window_size, window_size),
                        spatial_dims=("latitude", "longitude"),
                        )
                result_dict = result_dict | {f"FSS_WS{window_size}_C13_{category_name}": float(fss.values)}

    return result_dict


def unpack_da_to_dict(da, metric_prefix):

    results_dict = {}
    for channel in da.channel:
        results_dict[f"{metric_prefix}_C{channel:02}"] = da.sel(channel=[channel]).values.flatten()
    
    return results_dict


