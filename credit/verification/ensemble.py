import logging

import numpy as np
import xarray as xr
from pysteps.verification.ensscores import rankhist
from properscoring import crps_ensemble

logger = logging.getLogger(__name__)

latitude_slices = {
    "global": slice(-91, 91),
    "s_extratropics": slice(-91, -24.5),
    "tropics": slice(-24.5, 24.5),
    "n_extratropics": slice(24.5, 91),
}


def spread_error(da_pred, da_true, w_lat=None):
    """
    computes the latitude weighted ensemble standard deviation of da_pred and ensemble rmse with respect to da_true

    input: da_pred, da_true with matching time, lat, lon dimensions
    output: result_dict with std and rmse for regions defined by latitude partition (see above)
    """
    if w_lat is None:
        w_lat = np.cos(np.deg2rad(da_pred.latitude))

    ensemble_size = len(da_pred.ensemble_member_label)
    result_dict = {}

    std_raw = da_pred.std(dim="ensemble_member_label").mean(dim=["time", "longitude"])
    rmse_raw = np.sqrt((da_pred.mean(dim="ensemble_member_label") - da_true) ** 2).mean(dim=["time", "longitude"])

    for slice_name, s in latitude_slices.items():
        w_lat_slice = w_lat.sel(latitude=s)
        sum_wts = np.sum(w_lat_slice)

        # area weighted mean
        std = (std_raw.sel(latitude=s) * w_lat_slice).sum() / sum_wts
        rmse = (rmse_raw.sel(latitude=s) * w_lat_slice).sum() / sum_wts

        # add to dict and apply correction factor
        result_dict[f"std_{slice_name}"] = (ensemble_size + 1) / (ensemble_size - 1) * std.values
        result_dict[f"rmse_{slice_name}"] = float(rmse.values)

    return result_dict


def binned_spread_skill(da_pred, da_true, num_bins, w_lat=None):
    """
    computes the binned spread-skill

    input: da_pred, da_true with matching time, lat, lon dimensions
    output: result_dict
    """
    spread = da_pred.std(dim="ensemble_member_label").values.flatten()
    rmse = np.sqrt((da_pred.mean(dim="ensemble_member_label") - da_true) ** 2).values.flatten()

    bins = np.linspace(spread.min(), spread.max(), num_bins + 1)
    bin_indices = np.digitize(spread, bins)  # Assign bins

    bin_centers = [(bins[i] + bins[i - 1]) / 2 for i in range(1, len(bins))]
    spread_means = [
        spread[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else np.nan for i in range(1, len(bins))
    ]
    rmse_means = [
        rmse[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else np.nan for i in range(1, len(bins))
    ]
    counts = [(bin_indices == i).sum() for i in range(1, len(bins))]

    return {
        "bin_centers": bin_centers,
        "spread_means": spread_means,
        "rmse_means": rmse_means,
        "counts": counts,
    }


def crps(da_pred, da_true, w_lat=None):
    """Latitude-weighted mean CRPS per forecast region.

    Args:
        da_pred: ensemble predictions with an 'ensemble_member_label' dimension,
                 shape (time, ensemble_member_label, latitude, longitude).
        da_true: truth DataArray, shape (time, latitude, longitude).
        w_lat:   latitude weights (defaults to cos(lat)).

    Returns:
        dict with keys 'crps_global', 'crps_tropics', 'crps_n_extratropics',
        'crps_s_extratropics' — each a latitude-weighted float.
    """
    if w_lat is None:
        w_lat = np.cos(np.deg2rad(da_pred.latitude))

    # properscoring expects (n_obs, n_members); flatten spatial dims together.
    # Transpose to (time, lat, lon, ensemble) first.
    fcst = da_pred.transpose("time", "latitude", "longitude", "ensemble_member_label").values
    obs = da_true.values  # (time, lat, lon)

    crps_map = crps_ensemble(obs, fcst)  # (time, lat, lon)

    # Average over time and longitude → (lat,)
    crps_lat = crps_map.mean(axis=(0, 2))
    crps_da = xr.DataArray(crps_lat, coords={"latitude": da_pred.latitude.values}, dims=["latitude"])

    result_dict = {}
    for slice_name, s in latitude_slices.items():
        w = w_lat.sel(latitude=s)
        c = crps_da.sel(latitude=s)
        result_dict[f"crps_{slice_name}"] = float((c * w).sum() / w.sum())

    return result_dict


def rank_histogram_apply(da_pred, da_true, w_lat=None):
    """
    computes the rank histogram

    input: da_pred, da_true with matching time, lat, lon dimensions
    output: result_dict
    """

    ensemble_size = len(da_pred.ensemble_member_label)

    # Vectorize: reshape (ensemble, time, lat, lon) → (ensemble, time*lat*lon)
    # and (time, lat, lon) → (time*lat*lon,) so rankhist processes all grid points at once.
    pred_arr = da_pred.transpose("ensemble_member_label", "time", "latitude", "longitude").values
    true_arr = da_true.transpose("time", "latitude", "longitude").values
    pred_flat = pred_arr.reshape(ensemble_size, -1)
    true_flat = true_arr.reshape(-1)

    return rankhist(pred_flat, true_flat, normalize=False)


def crps_spatial_avg(pred, truth, w_lat):
    """Spatially-weighted CRPS and ensemble spread for a numpy ensemble.

    Parameters
    ----------
    pred : np.ndarray, shape (n_members, lat, lon)
        Ensemble forecast.
    truth : np.ndarray, shape (lat, lon)
        Verification field.
    w_lat : np.ndarray, shape (lat,)
        Latitude weights (should be normalized; e.g. cos(lat) / mean(cos(lat))).

    Returns
    -------
    crps_val : float
        Spatially-weighted mean CRPS.
    spread_val : float
        Spatially-weighted mean ensemble standard deviation.
    """
    n_members, lat, lon = pred.shape

    obs_flat = truth.reshape(-1)
    fcst_flat = pred.reshape(n_members, -1).T  # (lat*lon, n_members)
    crps_map = crps_ensemble(obs_flat, fcst_flat).reshape(lat, lon)

    crps_val = float((crps_map * w_lat[:, None]).sum() / (lat * lon))

    spread_map = pred.std(axis=0)
    spread_val = float((spread_map * w_lat[:, None]).sum() / (lat * lon))

    return crps_val, spread_val
