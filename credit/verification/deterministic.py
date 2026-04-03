import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Standard WeatherBench2 latitude regions
latitude_slices = {
    "global": slice(-90, 90),
    "s_extratropics": slice(-90, -20),
    "tropics": slice(-20, 20),
    "n_extratropics": slice(20, 90),
}

# Standard WB2 pressure levels (hPa)
WB2_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Standard WB2 upper-air variables (CREDIT name → WB2 name)
WB2_UPPER_AIR_VARS = {
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "T": "temperature",
    "Q": "specific_humidity",
    "Z": "geopotential",
}

# Standard WB2 surface variables (CREDIT name → WB2 name)
WB2_SURFACE_VARS = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "SP": "mean_sea_level_pressure",
    "Z500": "geopotential_500",
    "T500": "temperature_500",
    "U500": "u_component_of_wind_500",
    "V500": "v_component_of_wind_500",
    "Q500": "specific_humidity_500",
}


def latitude_weights(da):
    """Cosine latitude weights normalized to sum to 1 over the latitude dimension."""
    w = np.cos(np.deg2rad(da.latitude))
    return w / w.mean()


def weighted_mean(da, w_lat=None, dims=("latitude", "longitude")):
    """Area-weighted mean over spatial dimensions."""
    if w_lat is None:
        w_lat = latitude_weights(da)
    return (da * w_lat).mean(dim=list(dims)) / w_lat.mean()


def rmse(da_pred, da_true, w_lat=None, time_mean=True):
    """
    Latitude-weighted RMSE matching WeatherBench2 convention.

    Parameters
    ----------
    da_pred : xr.DataArray
        Forecast. Dims: (time, latitude, longitude[, level])
    da_true : xr.DataArray
        Verification (ERA5). Same dims as da_pred.
    w_lat : xr.DataArray, optional
        Latitude weights. Defaults to cos(lat).
    time_mean : bool
        If True, average over time. If False, return per-timestep values.

    Returns
    -------
    float or xr.DataArray
    """
    if w_lat is None:
        w_lat = np.cos(np.deg2rad(da_pred.latitude))

    sq_err = (da_pred - da_true) ** 2
    # Average longitude first (uniform weight), then latitude-weighted mean.
    # This matches WB2 convention: MSE = sum_lat(w * mean_lon(sq_err)) / sum_lat(w)
    spatial_mse = (sq_err.mean(dim="longitude") * w_lat).sum(dim="latitude") / w_lat.sum()

    result = np.sqrt(spatial_mse)
    if time_mean:
        return float(result.mean(dim="time").values)
    return result


def bias(da_pred, da_true, w_lat=None, time_mean=True):
    """
    Latitude-weighted mean bias (forecast - truth).

    Returns
    -------
    float or xr.DataArray
    """
    if w_lat is None:
        w_lat = np.cos(np.deg2rad(da_pred.latitude))

    diff = da_pred - da_true
    spatial_bias = (diff.mean(dim="longitude") * w_lat).sum(dim="latitude") / w_lat.sum()

    if time_mean:
        return float(spatial_bias.mean(dim="time").values)
    return spatial_bias


def acc(da_pred, da_true, da_clim, w_lat=None, time_mean=True):
    """
    Anomaly Correlation Coefficient (ACC) following WeatherBench2 convention.

    ACC = sum(w * pred_anom * true_anom) / sqrt(sum(w * pred_anom^2) * sum(w * true_anom^2))

    Parameters
    ----------
    da_pred : xr.DataArray
        Forecast. Dims: (time, latitude, longitude[, level])
    da_true : xr.DataArray
        Verification (ERA5). Same dims as da_pred.
    da_clim : xr.DataArray
        Climatology matched to da_true time axis (e.g., daily or monthly climatology).
        Must be broadcastable to da_true.
    w_lat : xr.DataArray, optional
        Latitude weights. Defaults to cos(lat).
    time_mean : bool
        If True, average over time.

    Returns
    -------
    float or xr.DataArray
    """
    if w_lat is None:
        w_lat = np.cos(np.deg2rad(da_pred.latitude))

    pred_anom = da_pred - da_clim
    true_anom = da_true - da_clim

    num = (w_lat * pred_anom * true_anom).sum(dim=["latitude", "longitude"])
    denom = np.sqrt(
        (w_lat * pred_anom**2).sum(dim=["latitude", "longitude"])
        * (w_lat * true_anom**2).sum(dim=["latitude", "longitude"])
    )
    result = num / (denom + 1e-12)

    if time_mean:
        return float(result.mean(dim="time").values)
    return result


def skill_score(metric_forecast, metric_baseline):
    """
    Generic skill score: SS = 1 - metric_forecast / metric_baseline.
    Positive = better than baseline, 0 = same, negative = worse.
    """
    return 1.0 - metric_forecast / metric_baseline


def rmse_by_region(da_pred, da_true, w_lat=None):
    """
    Compute latitude-weighted RMSE for each standard WB2 region.

    Returns
    -------
    dict with keys like 'rmse_global', 'rmse_tropics', etc.
    """
    if w_lat is None:
        w_lat = np.cos(np.deg2rad(da_pred.latitude))

    result = {}
    for region, s in latitude_slices.items():
        pred_r = da_pred.sel(latitude=s)
        true_r = da_true.sel(latitude=s)
        w_r = w_lat.sel(latitude=s)
        result[f"rmse_{region}"] = rmse(pred_r, true_r, w_lat=w_r)
    return result


def acc_by_region(da_pred, da_true, da_clim, w_lat=None):
    """
    Compute ACC for each standard WB2 region.

    Returns
    -------
    dict with keys like 'acc_global', 'acc_tropics', etc.
    """
    if w_lat is None:
        w_lat = np.cos(np.deg2rad(da_pred.latitude))

    result = {}
    for region, s in latitude_slices.items():
        pred_r = da_pred.sel(latitude=s)
        true_r = da_true.sel(latitude=s)
        clim_r = da_clim.sel(latitude=s)
        w_r = w_lat.sel(latitude=s)
        result[f"acc_{region}"] = acc(pred_r, true_r, clim_r, w_lat=w_r)
    return result


def deterministic_scores(da_pred, da_true, da_clim=None, w_lat=None):
    """
    Compute full suite of deterministic WB2 metrics for a single variable.

    Parameters
    ----------
    da_pred : xr.DataArray
        Forecast. Dims: (time, latitude, longitude[, level])
    da_true : xr.DataArray
        ERA5 verification. Same dims.
    da_clim : xr.DataArray, optional
        Climatology for ACC. If None, ACC is skipped.
    w_lat : xr.DataArray, optional
        Latitude weights.

    Returns
    -------
    dict
        Keys: rmse_{region}, bias_{region}[, acc_{region}]
    """
    if w_lat is None:
        w_lat = np.cos(np.deg2rad(da_pred.latitude))

    result = {}
    result.update(rmse_by_region(da_pred, da_true, w_lat=w_lat))

    # bias by region
    sq_err = da_pred - da_true
    for region, s in latitude_slices.items():
        diff_r = sq_err.sel(latitude=s)
        w_r = w_lat.sel(latitude=s)
        spatial_bias = (diff_r.mean(dim="longitude") * w_r).sum(dim="latitude") / w_r.sum()
        result[f"bias_{region}"] = float(spatial_bias.mean(dim="time").values)

    if da_clim is not None:
        result.update(acc_by_region(da_pred, da_true, da_clim, w_lat=w_lat))

    return result


def load_wb2_climatology(clim_path, variable, lead_time=None):
    """
    Load ERA5 climatology for ACC computation.

    Expects a netCDF/zarr file with dims (dayofyear, latitude, longitude[, level])
    or (month, latitude, longitude[, level]).

    Parameters
    ----------
    clim_path : str or Path
        Path to climatology file.
    variable : str
        Variable name to select.
    lead_time : xr.DataArray, optional
        Forecast times; used to align climatology to forecast time axis.

    Returns
    -------
    xr.DataArray
        Climatology aligned to lead_time if provided.
    """
    ds = xr.open_dataset(clim_path)
    da = ds[variable]

    if lead_time is not None:
        if "dayofyear" in da.dims:
            doy = lead_time.dt.dayofyear
            da = da.sel(dayofyear=doy).drop_vars("dayofyear")
            da["time"] = lead_time
        elif "month" in da.dims:
            month = lead_time.dt.month
            da = da.sel(month=month).drop_vars("month")
            da["time"] = lead_time

    return da
