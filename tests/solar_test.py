from credit.solar import get_solar_radiation_loc
import numpy as np
import pandas as pd
import xarray as xr


def test_get_solar_radiation_loc():
    start_date = "1999-01-01"
    end_date = "2001-12-31"
    lon = -95.2
    lat = 49.2
    altitude = 1000.0
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    tsi_ds = get_solar_radiation_loc(lon, lat, altitude, start_date, end_date)
    assert np.all(tsi_ds["tsi"].values) >= 0, "Negative solar values"
    assert np.all(tsi_ds["coszen"].min() >= 0), "Negative cos zenith values"
    assert np.all(dates == pd.DatetimeIndex(tsi_ds.time)), "Dates do not match"


def test_get_solar_radiation_loc_6h():
    start_date = "1999-01-01"
    end_date = "2001-12-31 18:00"
    lon = -95.2
    lat = 49.2
    altitude = 1000.0
    dates = pd.date_range(start=start_date, end=end_date, freq='6h')
    tsi_ds = get_solar_radiation_loc(lon, lat, altitude, start_date, end_date, step_freq='6h')
    assert np.all(tsi_ds["tsi"].values) >= 0, "Negative solar values"
    assert np.all(tsi_ds["coszen"].min() >= 0), "Negative cos zenith values"
    assert np.all(dates == pd.DatetimeIndex(tsi_ds.time)), "Dates do not match"


def test_solar_gridding():
    lons = np.arange(-100.0, -89.5, 0.5)
    lats = np.arange(30.0, 35.0, 0.5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    solar_ts = []
    start_date = "2016-01-01"
    end_date = "2016-12-31 23:00"
    for lon_val, lat_val in zip(lon_grid.ravel(), lat_grid.ravel()):
        out = get_solar_radiation_loc(lon_val, lat_val, 0.0,
                                      start_date, end_date, sub_freq="15Min")
        solar_ts.append(out)
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    combined = xr.combine_by_coords(solar_ts)
    assert np.all(combined["longitude"] == lons), "longitudes do not match"
    assert np.all(combined["latitude"] == lats), "latitudes do not match"
    assert combined["tsi"].shape == (dates.size, lats.size, lons.size), "shape ordering is not correct"
