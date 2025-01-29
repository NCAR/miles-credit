from credit.transforms import BridgescalerScaleState
from credit.data import Sample
import numpy as np
import xarray as xr
import os
import torch


def test_BridgescalerScaleState():
    test_file_dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    conf = {
        "data": {
            "quant_path": os.path.join(
                test_file_dir, "data/era5_standard_scalers_2024-07-27_1030.parquet"
            ),
            "variables": ["U", "V"],
            "surface_variables": ["U500", "V500"],
        },
        "model": {"levels": 15},
    }
    data = xr.Dataset()
    d_shape = (1, 15, 16, 32)
    d2_shape = (1, 16, 32)

    data["U"] = (
        ("time", "level", "latitude", "longitude"),
        np.random.normal(1, 12, size=d_shape),
    )
    data["V"] = (
        ("time", "level", "latitude", "longitude"),
        np.random.normal(-2, 20, size=d_shape),
    )
    data["U500"] = (
        ("time", "latitude", "longitude"),
        np.random.normal(1, 13, size=d2_shape),
    )
    data["V500"] = (
        ("time", "latitude", "longitude"),
        np.random.normal(-2, 20, size=d2_shape),
    )
    data.coords["level"] = np.array(
        [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136],
        dtype=np.int64,
    )
    samp = Sample()
    samp["historical_ERA5_images"] = data
    transform = BridgescalerScaleState(conf)
    transformed = transform.transform(samp)
    tdvars = list(transformed["historical_ERA5_images"].data_vars.keys())
    assert tdvars == ["U", "V", "U500", "V500"]
    test_tensor = torch.normal(2, 5, size=(1, 67, 8, 16))
    test_trans_tensor = torch.normal(0, 1, size=(1, 67, 8, 16))
    conf = {
        "data": {
            "quant_path": os.path.join(
                test_file_dir, "data/era5_standard_scalers_2024-07-27_1030.parquet"
            ),
            "variables": ["U", "V", "T", "Q"],
            "surface_variables": ["SP", "t2m", "Z500", "T500", "U500", "V500", "Q500"],
            "level_ids": data.coords["level"].values,
        },
        "model": {"levels": 15},
    }
    transform = BridgescalerScaleState(conf)
    trans_tensor = transform.transform_array(test_tensor)
    reverse_tensor = transform.inverse_transform(test_trans_tensor)
    assert reverse_tensor.shape == (1, 67, 8, 16)
    assert np.abs((trans_tensor - test_tensor).numpy()).max() > 0
    assert np.abs((reverse_tensor - test_trans_tensor).numpy()).max() > 0

    return
