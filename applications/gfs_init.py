from credit.nwp import build_GFS_init, format_datetime
from credit import metadata
import yaml
import argparse
import xarray as xr
import os
from os.path import join
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as config_file:
    config = yaml.safe_load(config_file)

os.makedirs(config['predict']['save_forecast'], exist_ok=True)
base_path = os.path.abspath(os.path.dirname(__file__))
credit_grid = xr.open_dataset(os.path.join(base_path, "credit/metadata/ERA5_Lev_Info.nc"))
model_levels = pd.read_csv(join(base_path, "credit/metadata/L137_model_level_indices.csv"))
model_level_indices = model_levels["model_level_indices"].values
date = format_datetime(config["predict"]["forecasts"])
variables = config['data']['variables'] + config['data']['surface_variables']

gfs_init = build_GFS_init(output_grid=credit_grid,
                          date=date,
                          variables=variables,
                          model_level_indices=model_level_indices)

gfs_init.to_zarr(join(base_out_path, f"gfs_init_{date.strftime('%Y%m%d_%H00')}.zarr"))





