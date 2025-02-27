from credit.nwp import build_GFS_init
from credit import metadata
import yaml
import argparse
import xarray as xr
import importlib.resources


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file")
args = parser.parse_args()
with open(args.config) as config_file:
    config = yaml.safe_load(config_file)

with importlib.resources.path(metadta, "ERA5_Lev_Info.nc") as nc_path:
    credit_grid = xr.open_dataset(nc_path)

date = "202502200600"
model_levels = [ 10,  30,  40,  50,  60,  70,  80,  90,  95, 100, 105, 110, 120, 130, 136, 137]

variables = config['data']['variables'] + config['data']['surface_variables']
gfs_init = build_GFS_init(output_grid=credit_grid,
                          variables=variables,
                          model_level_indices=model_levels)

gfs_init.to_zarr("/gladE/derecho/scratch/cbecker/gfs_init.zarr")
