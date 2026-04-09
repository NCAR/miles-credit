import logging

import numpy as np
from pysteps.verification.ensscores import rankhist

logger = logging.getLogger(__name__)


def crps_spatial_avg(pred_ens, truth, w_lat):
    """
    Spatially-averaged CRPS using the sorted-ensemble formula.

    Uses the energy decomposition CRPS = E[|X - y|] - 0.5 * E[|X - X'|] with
    E[|X - X'|] computed via the O(n log n) sorted-ensemble identity:

        E[|X - X'|] = (2/n²) * sum_i (2i - n + 1) * x_{(i)}   (i: 0-indexed sorted)

    Parameters
    ----------
    pred_ens : np.ndarray  shape (n_members, lat, lon), float64
    truth    : np.ndarray  shape (lat, lon),             float64
    w_lat    : np.ndarray  shape (lat,)  normalized cos-lat weights
                           (w = cos(lat) / mean(cos(lat)))

    Returns
    -------
    crps   : float  spatially-averaged CRPS
    spread : float  spatially-averaged ensemble std (ddof=1)
    """
    n, n_lat, n_lon = pred_ens.shape
    w2d = w_lat[:, None]  # (lat, 1)

    def sp_avg(arr2d):
        return (arr2d * w2d).sum() / (n_lat * n_lon)

    # E[|X - y|]: mean over members, spatial avg
    abs_err = np.abs(pred_ens - truth[None]).mean(axis=0)
    term1 = sp_avg(abs_err)

    # E[|X - X'|] via sorted-ensemble formula
    sorted_ens = np.sort(pred_ens, axis=0)
    wk = (2 * np.arange(n) - n + 1).reshape(n, 1, 1).astype(np.float64)
    energy_map = (2.0 / (n * n)) * (sorted_ens * wk).sum(axis=0)
    term2 = sp_avg(energy_map)

    crps = float(term1 - 0.5 * term2)
    spread = float(sp_avg(pred_ens.std(axis=0, ddof=1)))
    return crps, spread


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


def rank_histogram_apply(da_pred, da_true, w_lat=None):
    """
    computes the rank histogram

    input: da_pred, da_true with matching time, lat, lon dimensions
    output: result_dict
    """

    ensemble_size = len(da_pred.ensemble_member_label)
    rank_hist = np.zeros(ensemble_size + 1)

    da_pred = da_pred.transpose("ensemble_member_label", ...)

    # TODO: vectorize this computation
    for time in da_pred.time:
        rank_hist += rankhist(
            da_pred.sel(time=time).values,  # requires ensemble_member_label to be first dim after removing time
            da_true.sel(time=time).values,
            normalize=False,
        )

    return rank_hist
