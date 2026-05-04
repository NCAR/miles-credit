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
            "quant_path": os.path.join(test_file_dir, "data/era5_standard_scalers_2024-07-27_1030.parquet"),
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
            "quant_path": os.path.join(test_file_dir, "data/era5_standard_scalers_2024-07-27_1030.parquet"),
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


# ---------------------------------------------------------------------------
# ToTensor_BridgeScaler — __init__ reads conf, no file I/O
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PARQUET_PATH = os.path.join(_TEST_DIR, "data/era5_standard_scalers_2024-07-27_1030.parquet")


def _to_tensor_conf():
    return {
        "data": {
            "history_len": 1,
            "forecast_len": 2,
            "variables": ["U", "V"],
            "surface_variables": ["SP", "t2m"],
            "static_variables": [],
            "one_shot": False,
        },
        "model": {
            "image_height": 64,
            "image_width": 128,
            "levels": 15,
        },
    }


class TestToTensorBridgeScaler:
    def test_init_reads_conf_fields(self):
        from credit.transforms.transforms_quantile import ToTensor_BridgeScaler

        t = ToTensor_BridgeScaler(_to_tensor_conf())
        assert t.hist_len == 1
        assert t.for_len == 2
        assert t.variables == ["U", "V"]
        assert t.surface_variables == ["SP", "t2m"]
        assert t.latN == 64
        assert t.lonN == 128
        assert t.levels == 15
        assert t.one_shot is False

    def test_allvars_concatenates_variables_and_surface(self):
        from credit.transforms.transforms_quantile import ToTensor_BridgeScaler

        t = ToTensor_BridgeScaler(_to_tensor_conf())
        assert t.allvars == ["U", "V", "SP", "t2m"]

    def test_static_variables_stored(self):
        from credit.transforms.transforms_quantile import ToTensor_BridgeScaler

        conf = _to_tensor_conf()
        conf["data"]["static_variables"] = ["orography"]
        t = ToTensor_BridgeScaler(conf)
        assert t.static_variables == ["orography"]
