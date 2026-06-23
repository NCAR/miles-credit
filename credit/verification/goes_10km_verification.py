from functools import partial
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from credit.verification.standard import radial_fft_spectrum
from scores.spatial import fss_2d, fss_2d_binary
from sklearn.metrics import confusion_matrix


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

    step, f = step_file_tuple
    result_dict = {"forecast_step": step}
    
    pred_da = xr.open_dataset(f)["BT_or_R"] # BT_or_R with t channel lat lon 
    true_da = dataset[pred_da.t[0].values, "y"]["y"]
    w_lat = np.cos(np.deg2rad(pred_da.latitude))

    if eval_conf.get("compute_bulk", True):
        diff = pred_da - true_da

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
    
    # confusion matrix
    pred_da_all = pred_da.isel(t=0)
    target_da_all = true_da
    for channel, fss_conf in eval_conf["FSS"].items():
        pred_da = pred_da_all.sel(channel=channel)
        target_da = target_da_all.sel(channel=channel) 
        if "pct" in eval_conf.get("compute_FSS_types", ["pct", "pct_tropics"]):
            result_dict = result_dict | confusion_matrix_percentile_categories(pred_da, target_da, fss_conf, channel)

        if "pct_tropics" in eval_conf.get("compute_FSS_types", ["pct", "pct_tropics"]):
            # percentile FSS tropics
            lat_bounds = [-21, 21]
            target_da = target_da.sel(latitude = slice(*lat_bounds))
            pred_da = pred_da.sel(latitude = slice(*lat_bounds))
            
            tropics_results = confusion_matrix_percentile_categories(pred_da, target_da, fss_conf, channel)
            tropics_results = {f"{k}_tropics": v for k,v in tropics_results.items()}
            result_dict = result_dict | tropics_results
    # FSS
    # for channel, fss_conf in eval_conf["FSS"].items():
    #     pred_da = pred_da_all.sel(channel=channel)
    #     target_da = target_da_all.sel(channel=channel) 
        
    #     if "raw" in eval_conf.get("compute_FSS_types", ["raw", "pct", "pct_tropics"]):
    #         result_dict = result_dict | FSS_raw_threshold(pred_da, target_da, eval_conf, fss_conf, channel)
    #     if "pct" in eval_conf.get("compute_FSS_types", ["raw", "pct", "pct_tropics"]):
    #         result_dict = result_dict | FSS_percentile_threshold(pred_da, target_da, eval_conf, fss_conf, channel)

    #     if "pct_tropics" in eval_conf.get("compute_FSS_types", ["raw", "pct", "pct_tropics"]):
    #         # percentile FSS tropics
    #         lat_bounds = [-21, 21]
    #         target_da = target_da.sel(latitude = slice(*lat_bounds))
    #         pred_da = pred_da.sel(latitude = slice(*lat_bounds))
            
    #         tropics_results = FSS_percentile_threshold(pred_da, target_da, eval_conf, fss_conf, channel)
    #         tropics_results = {f"{k}_tropics": v for k,v in tropics_results.items()}
    #         result_dict = result_dict | tropics_results

    return result_dict


def is_in_bin(da, bin_edges):
    """
    outputs a masked dataarray of where the values are within the bin_edges
    """
    return (bin_edges[0] < da) & (da <= bin_edges[1])

def FSS_percentile_threshold(pred_da, target_da, eval_conf, fss_conf, channel):
    result_dict = {}

    target_sorted = np.sort(target_da.values.flatten())
    num_idxs = len(target_sorted)

    def find_quantile_target(thresholds):
        idxs = np.searchsorted(target_sorted, thresholds)
        return idxs / num_idxs
    
    def find_pred_pct_thresholds(thresholds):
        quantiles = find_quantile_target(thresholds)
        pred_pct_thresholds = pred_da.quantile(quantiles)
        return pred_pct_thresholds

    # percentile FSS
    for threshold in fss_conf["thresholds"]:
        
        f_o = is_in_bin(target_da, [0., threshold]).mean() # use original threshold for target obs
        result_dict = result_dict | {f"obs_freq_C{channel:02}_PCT{threshold}": float(f_o.values)}
        
        threshold_pred = find_pred_pct_thresholds(threshold) # use corresponding value at percentile
        # result_dict = result_dict | {f"obs_freq_C{channel:02}_PCT{threshold}": float(f_o.values)}

        for window_size in eval_conf["fss_window_sizes"]:
        # binary threshold fss
            fss = fss_2d(pred_da, target_da,
                        event_threshold=threshold_pred.values,
                        window_size=(window_size, window_size),
                        spatial_dims=("latitude", "longitude"),
                        threshold_operator=np.less
                        )
            result_dict = result_dict | {f"FSS_WS{window_size}_C{channel:02}_PCT{threshold}": float(fss.values)}

    # category percentile
    for category_name, bin_edges in fss_conf["sky_categories"].items():
        bin_edges_pred = find_pred_pct_thresholds(bin_edges) # get corresponding value at equiv percentile
        pred_binary = is_in_bin(pred_da, bin_edges_pred.values)
        result_dict = result_dict | {f"bin_edges_C{channel:02}_PCT_{category_name}": bin_edges_pred.values.flatten()}

        target_binary = is_in_bin(target_da, bin_edges)
        f_o = target_binary.mean()
        result_dict = result_dict | {f"obs_freq_C{channel:02}_PCT_{category_name}": float(f_o.values)}

        for window_size in eval_conf["fss_window_sizes"]:
            fss = fss_2d_binary(pred_binary, target_binary,
                    window_size=(window_size, window_size),
                    spatial_dims=("latitude", "longitude"),
                    )
            result_dict = result_dict | {f"FSS_WS{window_size}_C{channel:02}_PCT_{category_name}": float(fss.values)}
    return result_dict
    
def FSS_raw_threshold(pred_da, target_da, eval_conf, fss_conf, channel):
    result_dict = {}

    for threshold in fss_conf["thresholds"]:
        f_o = is_in_bin(target_da, [0., threshold]).mean()
        result_dict = result_dict | {f"obs_freq_C{channel:02}_T{threshold}": float(f_o.values)}

        for window_size in eval_conf["fss_window_sizes"]:
        # binary threshold fss
            fss = fss_2d(pred_da, target_da,
                        event_threshold=threshold,
                        window_size=(window_size, window_size),
                        spatial_dims=("latitude", "longitude"),
                        threshold_operator=np.less
                        )
            result_dict = result_dict | {f"FSS_WS{window_size}_C{channel:02}_T{threshold}": float(fss.values)}

    # categorical fss
    for category_name, bin_edges in fss_conf["sky_categories"].items():
        pred_binary = is_in_bin(pred_da, bin_edges)
        target_binary = is_in_bin(target_da, bin_edges)

        f_o = target_binary.mean()
        result_dict = result_dict | {f"obs_freq_C{channel:02}_{category_name}": float(f_o.values)}

        for window_size in eval_conf["fss_window_sizes"]:
            fss = fss_2d_binary(pred_binary, target_binary,
                    window_size=(window_size, window_size),
                    spatial_dims=("latitude", "longitude"),
                    )
            result_dict = result_dict | {f"FSS_WS{window_size}_C{channel:02}_{category_name}": float(fss.values)}
    return result_dict

def confusion_matrix_percentile_categories(pred_da, target_da, fss_conf, channel):
    result_dict = {}

    target_sorted = np.sort(target_da.values.flatten())
    num_idxs = len(target_sorted)

    def find_quantile_target(thresholds):
        idxs = np.searchsorted(target_sorted, thresholds)
        return idxs / num_idxs

    def find_pred_pct_thresholds(thresholds):
        quantiles = find_quantile_target(thresholds)
        pred_pct_thresholds = pred_da.quantile(quantiles)
        return pred_pct_thresholds

    bins = list(fss_conf["sky_categories"].items()) # list of lists of bins
    bins = sorted(bins, key=lambda x: x[1][0]) # sort by lower bound

    target_bins = [bin[1] for bin in bins] # grab the bounds
    target_bin_bounds = [x[0] for x in target_bins][1:] # get bin bounds, remove first lower bound
    target_categorical = np.digitize(target_da, target_bin_bounds, right=True)

    pred_bins = [find_pred_pct_thresholds(thresh) for thresh in target_bins]
    pred_bin_bounds = [x[0] for x in pred_bins][1:]
    pred_categorical = np.digitize(pred_da, pred_bin_bounds, right=True)

    # use sklearn confusion matrix
    conf_matrix = confusion_matrix(target_categorical.flatten(), pred_categorical.flatten(), normalize="true")

    bin_names = [bin[0] for bin in bins]
    for obs_cat, row in zip(bin_names, conf_matrix):
        result_dict = result_dict | {f"confusion_C{channel:02}_obs_{obs_cat}": row}

    return result_dict


    # for category_name, bin_edges in fss_conf["sky_categories"].items():
    #     bin_edges_pred = find_pred_pct_thresholds(bin_edges) # get corresponding value at equiv percentile
    #     pred_binary = is_in_bin(pred_da, bin_edges_pred.values)
    #     pred_binarized[category_name] = pred_binary
    
    # # now compute confusion matrix obs cat as outer loop
    # # sklearn def:
    # # Confusion matrix whose i-th row and j-th column entry indicates the number of samples
    # # with true label being i-th class and predicted label being j-th class.
    # for category_name, bin_edges in fss_conf["sky_categories"].items():
    #     target_binary = is_in_bin(target_da, bin_edges)
        
    #     for pred_cat, pred_binary in pred_binarized.items():

    #         result_dict | {f"confusion_obs_{category_name}_pred_{pred_cat}"}
    
def unpack_da_to_dict(da, metric_prefix):

    results_dict = {}
    for channel in da.channel:
        results_dict[f"{metric_prefix}_C{channel:02}"] = da.sel(channel=[channel]).values.flatten()
    
    return results_dict


