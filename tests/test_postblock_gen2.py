from credit.postblock.geopotential import GeopotentialDiagnostic
from credit.trainers.utils import load_dataloader
from credit.preblock import ConcatToTensor
from credit.postblock import Reconstruct
import yaml
from copy import deepcopy
from torch import isnan, all
import torch
import pandas as pd
from torch.utils.data import Dataset


conf_str = """
data:
  source:
    ARCO_ERA5:
      dataset_type: "arco_era5"
      level_coord: "hybrid"
      variables:
        prognostic:
          vars_3D: ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"]
          vars_2D: ["surface_pressure"]
        dynamic_forcing:
          vars_2D: ["toa_incident_solar_radiation"]
        static:
          vars_2D: ["geopotential_at_surface"]
        diagnostic:
          vars_2D: ["total_precipitation"]

  start_datetime: "2017-01-01"
  end_datetime: "2018-12-31"
  timestep: "6h"
  forecast_len: 1
trainer:
  train_batch_size: 2
  valid_batch_size: 1
  thread_workers: 2
  valid_thread_workers: 2
"""
conf = yaml.safe_load(conf_str)


class MockARCOERA5MultiSourceDataset(Dataset):
    """Synthetic stand-in for MultiSourceDataset / ARCOERA5Dataset.

    Returns ERA5-like tensors on a tiny (N_LAT x N_LON) domain so the test
    runs without any network access.  Values are physically plausible enough
    for GeopotentialDiagnostic to produce finite output: temperature increases
    from 200 K at model top to 290 K near surface, specific humidity increases
    from near-zero at the top to ~0.012 kg/kg near the surface, and surface
    pressure is ~101 000 Pa.

    The nested dict structure mirrors what MultiSourceDataset.__getitem__ and
    the default DataLoader collation produce:

        batch["input"]["ARCO_ERA5"]["ARCO_ERA5/prognostic/3d/temperature"]
            -> Tensor (B, N_LEVELS, 1, N_LAT, N_LON)
    """

    SOURCE = "ARCO_ERA5"
    N_LEVELS = 137
    N_LAT = 8
    N_LON = 16

    def __init__(self, data_config: dict, return_target: bool = False) -> None:
        super().__init__()
        dt = pd.Timedelta(data_config["timestep"])
        start = pd.Timestamp(data_config["start_datetime"])
        end = pd.Timestamp(data_config["end_datetime"])
        forecast_len = int(data_config.get("forecast_len", 1))
        self.dt = dt
        self.datetimes = pd.date_range(start, end - forecast_len * dt, freq=dt)
        self.return_target = return_target
        self.static_metadata = {self.SOURCE: {"levels": list(range(1, 138)), "datetime_fmt": "unix_ns"}}

    def __len__(self) -> int:
        return len(self.datetimes)

    def _prognostic_vars(self) -> dict:
        """Return one set of ERA5-like prognostic variable tensors."""
        N, H, W, src = self.N_LEVELS, self.N_LAT, self.N_LON, self.SOURCE
        # Temperature: 200 K at model top (level 1) -> 290 K near surface (level 137)
        temp = torch.linspace(200.0, 290.0, N).view(N, 1, 1, 1).expand(N, 1, H, W).clone()
        # Specific humidity: near-zero at top, ~0.012 kg/kg near surface
        q = torch.linspace(1e-6, 0.012, N).view(N, 1, 1, 1).expand(N, 1, H, W).clone()
        return {
            f"{src}/prognostic/3d/temperature": temp,
            f"{src}/prognostic/3d/u_component_of_wind": torch.randn(N, 1, H, W) * 15.0,
            f"{src}/prognostic/3d/v_component_of_wind": torch.randn(N, 1, H, W) * 10.0,
            f"{src}/prognostic/3d/specific_humidity": q,
            f"{src}/prognostic/2d/surface_pressure": torch.full((1, 1, H, W), 101_000.0),
        }

    def __getitem__(self, args: tuple) -> dict:
        """Return a nested input/target sample dict.

        Mirrors the structure of MultiSourceDataset.__getitem__:
            {"input": {source: {var_key: tensor}}, "metadata": {source: {...}}}

        Args:
            args: (t, i) where t is a timestamp (nanoseconds) and i is the
                within-sequence step index from the sampler.
        """
        t, i = args
        t = pd.Timestamp(t)
        src, H, W = self.SOURCE, self.N_LAT, self.N_LON

        # Dynamic forcing is present at every step
        input_sample: dict = {
            f"{src}/dynamic_forcing/2d/toa_incident_solar_radiation": torch.full((1, 1, H, W), 400.0),
        }
        # Prognostic and static are only loaded at the initial step (i == 0)
        if i == 0:
            input_sample[f"{src}/static/2d/geopotential_at_surface"] = torch.full((1, 1, H, W), 9_810.0)
            input_sample.update(self._prognostic_vars())

        result: dict = {
            "input": {src: input_sample},
            "metadata": {src: {"input_datetime": int(t.value)}},
        }

        if self.return_target:
            target_sample = self._prognostic_vars()
            target_sample[f"{src}/diagnostic/2d/total_precipitation"] = torch.zeros(1, 1, H, W)
            result["target"] = {src: target_sample}
            result["metadata"][src]["target_datetime"] = int((t + self.dt).value)

        return result


def test_geopotential():
    msd = MockARCOERA5MultiSourceDataset(conf["data"])
    mdl = load_dataloader(conf, msd, 0, 1, True)
    batch = next(iter(mdl))
    ct = ConcatToTensor()
    batch_tensor, meta = ct(batch)
    meta_2 = deepcopy(meta)
    meta_2["target"]["_channel_map"] = meta_2["input"]["_channel_map"]
    recon = Reconstruct()
    output = recon({"prediction": batch_tensor, "metadata": meta_2})
    output_var_name = "ARCO_ERA5/derived_diagnostic/3d/geopotential"
    geopotential_layer = GeopotentialDiagnostic(
        output_name=output_var_name,
        data_keys=["prediction"],
        surface_geopotential_var="ARCO_ERA5/static/2d/geopotential_at_surface",
        surface_pressure_var="ARCO_ERA5/prognostic/2d/surface_pressure",
        temperature_var="ARCO_ERA5/prognostic/3d/temperature",
        specific_humidity_var="ARCO_ERA5/prognostic/3d/specific_humidity",
        level_info_file="ERA5_Lev_Info.nc",
        model_a_half_var="a_half",
        model_b_half_var="b_half",
    )
    diagnosed = geopotential_layer(output)
    pred = diagnosed["prediction"]
    geo = pred["ARCO_ERA5"]["ARCO_ERA5/derived_diagnostic/3d/geopotential"]
    temp = pred["ARCO_ERA5"]["ARCO_ERA5/prognostic/3d/temperature"]
    assert geo.shape == temp.shape
    assert all(~isnan(geo))
