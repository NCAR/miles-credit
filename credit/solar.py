import numpy as np
import pandas as pd
import pvlib.solarposition
from pvlib.solarposition import get_solarposition
from scipy.integrate import trapezoid
import xarray as xr
from tqdm import tqdm
from collections.abc import Sequence


def era5_tsi_data() -> xr.DataArray:
    """A TsiDataProvider that returns ERA5 compatible TSI data. From [Graphcast](https://github.com/google-deepmind/graphcast/blob/main/graphcast/solar_radiation.py)."""
    # ECMWF provided the data used for ERA5, which was hardcoded in the IFS (cycle
    # 41r2). The values were scaled down to agree better with more recent
    # observations of the sun.
    time = np.arange(1951.5, 2035.5, 1.0)
    tsi = 0.9965 * np.array(
        [
            # fmt: off
            # 1951-1995 (non-repeating sequence)
            1365.7765,
            1365.7676,
            1365.6284,
            1365.6564,
            1365.7773,
            1366.3109,
            1366.6681,
            1366.6328,
            1366.3828,
            1366.2767,
            1365.9199,
            1365.7484,
            1365.6963,
            1365.6976,
            1365.7341,
            1365.9178,
            1366.1143,
            1366.1644,
            1366.2476,
            1366.2426,
            1365.9580,
            1366.0525,
            1365.7991,
            1365.7271,
            1365.5345,
            1365.6453,
            1365.8331,
            1366.2747,
            1366.6348,
            1366.6482,
            1366.6951,
            1366.2859,
            1366.1992,
            1365.8103,
            1365.6416,
            1365.6379,
            1365.7899,
            1366.0826,
            1366.6479,
            1366.5533,
            1366.4457,
            1366.3021,
            1366.0286,
            1365.7971,
            1365.6996,
            # 1996-2008 (13 year cycle, repeated below)
            1365.6121,
            1365.7399,
            1366.1021,
            1366.3851,
            1366.6836,
            1366.6022,
            1366.6807,
            1366.2300,
            1366.0480,
            1365.8545,
            1365.8107,
            1365.7240,
            1365.6918,
            # 2009-2021
            1365.6121,
            1365.7399,
            1366.1021,
            1366.3851,
            1366.6836,
            1366.6022,
            1366.6807,
            1366.2300,
            1366.0480,
            1365.8545,
            1365.8107,
            1365.7240,
            1365.6918,
            # 2022-2034
            1365.6121,
            1365.7399,
            1366.1021,
            1366.3851,
            1366.6836,
            1366.6022,
            1366.6807,
            1366.2300,
            1366.0480,
            1365.8545,
            1365.8107,
            1365.7240,
            1365.6918,
            # fmt: on
        ]
    )
    return xr.DataArray(tsi, dims=["time"], coords={"time": time})


tsi_data = era5_tsi_data()


def get_tsi(timestamps: Sequence, tsi_data: xr.DataArray) -> np.array:
    """Returns TSI values for the given timestamps.

    TSI values are interpolated from the provided yearly TSI data.

    Args:
        timestamps: Timestamps for which to compute TSI values.
        tsi_data: A DataArray with a single dimension `time` that has coordinates in
        units of years since 0000-1-1. E.g. 2023.5 corresponds to the middle of
        the year 2023.

    Returns:
        An Array containing interpolated TSI data.
    """
    timestamps = pd.DatetimeIndex(timestamps, tz="utc")
    timestamps_date = pd.DatetimeIndex(timestamps.date, tz="utc")
    day_fraction = (timestamps - timestamps_date) / pd.Timedelta(days=1)
    year_length = 365 + timestamps.is_leap_year
    year_fraction = (timestamps.dayofyear - 1 + day_fraction) / year_length
    fractional_year = timestamps.year + year_fraction
    return np.interp(fractional_year, tsi_data.coords["time"].data, tsi_data.data)


def get_toa_radiation(start_date: str, end_date: str, step_freq: str = "1h", sub_freq: str = "10Min"):
    """
    Calculate top of atmosphere solar irradiance

    Args:
        start_date (str): Start date of time series
        end_date (str): End date of time series (inclusive).
        step_freq (str): How much time between steps in pandas time string format (e.g., 1h, 10Min)
        sub_freq (str): How much time between substeps that are integrated forward (e.g., 10Min)

    Returns:
        top of atmosphere radiation in W m**-2.
    """
    start_date_ts = pd.Timestamp(start_date)
    end_date_ts = pd.Timestamp(end_date)
    dates = pd.date_range(
        start=start_date_ts - pd.Timedelta(step_freq) + pd.Timedelta(sub_freq),
        end=end_date_ts,
        freq=sub_freq,
        tz="utc",
    )
    total_rad = get_tsi(dates, tsi_data)
    solar_distance = pvlib.solarposition.nrel_earthsun_distance(dates, how="numba")
    solar_factor = (1.0 / solar_distance) ** 2
    return total_rad * solar_factor


def get_solar_radiation_loc(
    toa_radiation: pd.Series,
    lon: float,
    lat: float,
    altitude: float,
    start_date: str,
    end_date: str,
    step_freq: str = "1h",
    sub_freq: str = "10Min",
) -> xr.Dataset:
    """
    Calculate total solar irradiance at a single location over a range of times. Solar irradiance is integrated
    over the step frequency at specified substeps.

    Args:
        toa_radiation (pd.Series): Top of atmosphere solar radiation in W m**-2.
        lon (float): longitude.
        lat (float): latitude.
        altitude (float): altitude in meters.
        start_date (str): date str for the beginning of the period (inclusive).
        end_date (str):  date str for the end of the period (inclusive).
        step_freq (str): period over which irradiance is integrated. Defaults to "1h".
        sub_freq (str): sub step frequency over the step period. Defaults to "5Min".

    Returns:
        xarray.Dataset: total solar irradiance and cosine of solar zenith angle time series with metadata.
    """
    start_date_ts = pd.Timestamp(start_date)
    end_date_ts = pd.Timestamp(end_date)
    step_sec = pd.Timedelta(step_freq).total_seconds()
    sub_sec = pd.Timedelta(sub_freq).total_seconds()
    step_len = int(step_sec // sub_sec)
    dates = pd.date_range(
        start=start_date_ts - pd.Timedelta(step_freq) + pd.Timedelta(sub_freq),
        end=end_date_ts,
        freq=sub_freq,
        tz="utc",
    )
    solar_pos = get_solarposition(dates, lat, lon, altitude, method="nrel_numba")
    cos_zenith = np.maximum(0, np.cos(np.radians(solar_pos["zenith"].values)))
    solar_rad = toa_radiation * cos_zenith

    step_rad = trapezoid(
        np.reshape(solar_rad, (int(solar_rad.size // step_len), step_len)),
        dx=sub_sec,
        axis=1,
    )
    step_dates = pd.date_range(start=start_date_ts, end=end_date_ts, freq=step_freq, tz="utc")
    step_cos_zenith = pd.Series(cos_zenith, index=dates)[step_dates]

    out_rad_da = xr.DataArray(
        step_rad.reshape(-1, 1, 1),
        coords={"time": step_dates, "latitude": [lat], "longitude": [lon]},
        dims=("time", "latitude", "longitude"),
        name="tsi",
        attrs={
            "standard_name": "solar_irradiance",
            "long_name": "total solar irradiance",
            "units": "J m-2",
        },
    )

    zenith_da = xr.DataArray(
        step_cos_zenith.values.reshape(-1, 1, 1),
        coords={"time": step_dates, "latitude": [lat], "longitude": [lon]},
        dims=("time", "latitude", "longitude"),
        name="solar_zenith_angle",
        attrs={
            "standard_name": "cos_solar_zenith_angle",
            "long_name": "cosine of the solar zenith angle",
            "units": "",
        },
    )
    out_rad_ds = xr.Dataset({"tsi": out_rad_da, "coszen": zenith_da})
    return out_rad_ds


def get_solar_index(curr_date, ref_date="2000-01-01"):
    curr_date_ts = pd.to_datetime(curr_date)
    year_start = pd.Timestamp(f"{curr_date_ts.year:d}-01-01")
    curr_diff = curr_date_ts - year_start
    return int(curr_diff.total_seconds() / 3600)


if __name__ == "__main__":
    lons = np.arange(-100.0, -89.5, 0.5)
    lats = np.arange(30.0, 35.0, 0.5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    solar_ts = []
    toa_radiation = get_toa_radiation("2016-01-01", "2016-12-31 23:00")
    for lon_val, lat_val in tqdm(zip(lon_grid.ravel(), lat_grid.ravel())):
        out = get_solar_radiation_loc(toa_radiation, lon_val, lat_val, 0.0, "2016-01-01", "2016-12-31 23:00")
        solar_ts.append(out)
    combined = xr.combine_by_coords(solar_ts)
    print(combined)
