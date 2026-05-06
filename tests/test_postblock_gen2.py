from credit.postblock.geopotential import GeopotentialDiagnostic
from credit.datasets.multi_source import MultiSourceDataset
from credit.trainers.utils import load_dataloader
import yaml


conf_str = """
data:
  source:
    ARCO_ERA5:
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


def test_geopotential():
    msd = MultiSourceDataset(conf["data"])
    mdl = load_dataloader(conf, msd, 0, 1, True)
    geopotential_layer = GeopotentialDiagnostic()

    batch = next(iter(mdl))
    batch["prediction"] = batch["arco_era5"]["input"]
    updated_batch = geopotential_layer(batch)
    print(updated_batch)
    assert geopotential_layer.output_name in updated_batch["prediction"]
