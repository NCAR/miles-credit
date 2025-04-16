import os

import numpy as np
import xarray as xr

from credit.verification.ensemble import spread_error

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join(
    "/".join(os.path.abspath(__file__).split("/")[:-2]), "config"
)


def test_spread_error():
    ensemble_size = 1000000
    latitude = 2
    longitude = 3
    times = 2

    mu, sigma = 0, 1. # mean and standard deviation
    rng = np.random.default_rng()

    data = rng.normal(mu, sigma, (ensemble_size, times, latitude, longitude))

    da_true = xr.DataArray(np.zeros((times, latitude, longitude)),
                           coords={
                               "time": np.arange(times),
                               "latitude": np.linspace(-90, 90, latitude),
                               "longitude": np.linspace(0, 360, longitude)
                           }
                          )   
    da_pred = xr.DataArray(data,
                           coords={
                               "ensemble_member_label": np.arange(ensemble_size),
                               "time": np.arange(times),
                               "latitude": np.linspace(-90, 90, latitude),
                               "longitude": np.linspace(0, 360, longitude)
                           }
                          )
    
    result_dict = spread_error(da_pred, da_true)
    
    assert(np.isclose(result_dict["rmse_global"], 0.0, atol=1e-2) )
    assert(np.isclose(result_dict["std_global"], 1.0, atol=1e-2) )


if __name__ == "__main__":
    test_spread_error()