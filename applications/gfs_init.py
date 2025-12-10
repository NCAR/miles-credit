from credit.nwp import build_GFS_init
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

os.makedirs(config["predict"]["initial_condition_path"], exist_ok=True)
base_path = os.path.abspath(os.path.dirname(__file__))
credit_grid = xr.open_dataset(
    os.path.join(base_path, "../credit/metadata/ERA5_Lev_Info.nc")
)
model_levels = pd.read_csv(
    join(base_path, "../credit/metadata/L137_model_level_indices.csv")
)
model_level_indices = model_levels["model_level_indices"].values
variables = config["data"]["variables"] + config["data"]["surface_variables"]
date = pd.Timestamp(config["predict"]["realtime"]["forecast_start_time"], tz="UTC")
now_date = pd.Timestamp.utcnow()
if now_date - date >= pd.Timedelta(days=10):
    gdas_base_path = "gs://global-forecast-system/"
else:
    gdas_base_path = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"

gfs_init = build_GFS_init(
    output_grid=credit_grid,
    date=date,
    variables=variables,
    model_level_indices=model_level_indices,
    gdas_base_path=gdas_base_path
)

gfs_init.to_zarr(
    join(
        config["predict"]["initial_condition_path"],
        f"gfs_init_{date.strftime('%Y%m%d_%H00')}.zarr",
    )
)
