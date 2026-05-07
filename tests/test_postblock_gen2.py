from credit.postblock.geopotential import GeopotentialDiagnostic
from credit.datasets.multi_source import MultiSourceDataset
from credit.trainers.utils import load_dataloader
from credit.preblock import ConcatToTensor
from credit.postblock import Reconstruct
import yaml
from copy import deepcopy


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

    batch = next(iter(mdl))
    ct = ConcatToTensor()
    batch_tensor, meta = ct(batch)
    meta_2 = deepcopy(meta)
    meta_2["_channel_map"]["output"] = meta_2["_channel_map"]["input"]
    recon = Reconstruct()
    output = recon(batch_tensor, meta_2)
    geopotential_layer = GeopotentialDiagnostic(output_name=batch.keys()[0] + "/derived_diagnostic/3d/geopotential")
    diagnosed = geopotential_layer(output)
