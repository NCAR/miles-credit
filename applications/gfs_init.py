from credit.nwp import build_GFS_init
import yaml
import argparse
import xarray as xr
import os
from os.path import join, expandvars
from importlib.resources import files
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-p", "--proc", default=1, type=int, help="Number of processors.")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    n_procs = args.proc
    initial_condition_path = expandvars(config["predict"]["initial_condition_path"])
    os.makedirs(initial_condition_path, exist_ok=True)
    metadata_path = str(files("credit.metadata"))
    credit_grid = xr.open_dataset(os.path.join(metadata_path, "ERA5_Lev_Info.nc"))
    model_levels = pd.read_csv(os.path.join(metadata_path, "L137_model_level_indices.csv"))
    model_level_indices = model_levels["model_level_indices"].values
    variables = config["data"]["variables"] + config["data"]["surface_variables"]
    date = pd.Timestamp(config["predict"]["realtime"]["forecast_start_time"], tz="UTC")
    gdas_base_path = "gs://global-forecast-system/"
    if "variable_mapping" in config["predict"]:
        variable_mapping = config["predict"]["variable_mapping"]
    else:
        # Default is wchapmanera5 to preserve backwards compatibility with original CREDIT models.
        variable_mapping = "wchapmanera5"
    gfs_init = build_GFS_init(
        output_grid=credit_grid,
        date=date,
        variables=variables,
        model_level_indices=model_level_indices,
        gdas_base_path=gdas_base_path,
        variable_mapping=variable_mapping,
        n_procs=n_procs,
    )
    out_file = join(initial_condition_path, f"gfs_init_{date.strftime('%Y%m%d_%H00')}.zarr")
    gfs_init.to_zarr(out_file, mode="w")
    config["data"]["save_loc"] = out_file
    config["data"]["save_loc_surface"] = out_file
    config["data"]["save_loc_diagnostic"] = out_file
    real_config = args.config.replace(".yml", "_realtime.yml")
    print(
        f"Saving realtime config to {real_config}. Update data:save_loc_dynamic_forcing to point to appropriate solar netcdf files."
    )
    print(
        "See /glade/campaign/cisl/aiml/credit/ for pre-generated files, or use credit_calc_global_solar to produce your own."
    )
    with open(real_config, "w") as out_config_file:
        yaml.dump(config, out_config_file, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
