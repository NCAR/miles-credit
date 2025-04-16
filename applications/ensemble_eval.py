# ensemble evaluation suite for rollouts generated with rollout_to_netcdf.py
# parallelizes across CPUs
# WARNING: DOES NOT USE model config file
# see config/example_ensemble_eval.yml for an example config for this rollout
#
# WARNING: currently only works for CAM data, since the XRSamplerByYear can only handle one data file for all variables
# 

from argparse import ArgumentParser
from functools import partial
from glob import glob
import logging
import os
from os.path import join
from pathlib import Path
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr

import yaml

from credit.pbs import launch_script, launch_script_mpi, get_num_cpus
from credit.verification.ensemble import binned_spread_skill, rank_histogram_apply, spread_error
from credit.verification.standard import average_div_rot_spectrum, average_zonal_spectrum
from credit.xr_sampler import XRSamplerByYear

def evaluate(num_files, forecast_save_loc, conf, model_conf, p):
    # break up computations by forecast hour6
    # computations: spread-error, zonal spectrum, binned hist, rank hist, div/vorticity spectrum
    # TODO: CRPS

    model_timestep = model_conf["data"]["lead_time_periods"]
    forecast_hours = model_timestep * (np.arange(num_files) + 1)
    
    detailed_eval_hours = conf["detailed_eval_hours"]
    standard_eval_hours = [fh for fh in forecast_hours if fh not in detailed_eval_hours]
    
    # parallelize across forecast hours
    # start detailed eval first
    f = partial(do_eval, forecast_save_loc, conf, model_conf)
    result = p.map(f, detailed_eval_hours + standard_eval_hours)

    # result will be list of dicts eg. {"spread_U10": num, "spectrum_U_24": num, "ranks_": vec, "freq_": vec}
    # pack result into dataframe and save
    result.sort(key= lambda x: x["forecast_hour"]) # sort of forecast hours are in order
    
    # future: use multi-indexing
    df = pd.DataFrame(result)
    eval_save_loc = join(forecast_save_loc, conf["save_filename"])

    df.to_parquet(eval_save_loc) # parquet keeps all the dtypes, don't have to split up np arrays in the entries
    logging.info(f"saved ensemble eval to {eval_save_loc}")


def do_eval(forecast_save_loc, conf, model_conf, fh):
    """compute ensemble verification per forecast hour across all ICs
    returns None values for special metrics when fh not in detailed_eval_hours"""
    result_dict = {"forecast_hour": fh}
    rollout_files = glob(join(forecast_save_loc, f"*/*_{fh:03}.nc"))

    sampler = XRSamplerByYear(None, conf=model_conf) #load the truth sampler
    # TODO: make this sampler work for ERA5

    # get lat wts
    ds = xr.open_dataset(rollout_files[0])
    w_lat = np.cos(np.deg2rad(ds.latitude))

    #xarray does lazy loading so we just iterate through variables/levels
    for variable in conf["variables"]:
        for level in conf["levels"]:
            #get ensemble and truth
            da_pred, da_true = get_data(sampler, rollout_files, variable, level)
            #compute and merge dicts
            result_dict = result_dict | _do_standard_eval_on_variable(w_lat, da_pred, da_true, variable, level)
            result_dict = result_dict | _do_special_eval_on_variable(w_lat, conf, fh, da_pred, da_true, variable, level) # returns dict of None if not computed

     
    # eval of U and V combined
    if "U" in conf["variables"] and "V" in conf["variables"]:
        variable = "wind_norm"
        for level in conf["levels"]:
            #get ensemble and truth
            da_pred_u, da_true_u = get_data(sampler, rollout_files, "U", level)
            da_pred_v, da_true_v = get_data(sampler, rollout_files, "V", level)

            # do wind norm
            da_pred = np.sqrt(da_pred_u ** 2 + da_pred_v ** 2)
            da_true = np.sqrt(da_true_u ** 2 + da_true_v ** 2)
            result_dict = result_dict | _do_standard_eval_on_variable(w_lat, da_pred, da_true, variable, level)
            result_dict = result_dict | _do_special_eval_on_variable(w_lat, conf, fh, da_pred, da_true, variable, level) # returns dict of None if not computed

            # do vrt, div
            ds_pred = xr.Dataset({"U": da_pred_u, "V": da_pred_v})
            ds_true = xr.Dataset({"U": da_true_u, "V": da_true_v})

            vrt_pred, div_pred = average_div_rot_spectrum(ds_pred, conf["grid"], wave_spec="n")
            vrt_true, div_true = average_div_rot_spectrum(ds_true, conf["grid"], wave_spec="n")

            result_dict = result_dict | {f"vrt_spectrum_{level}": vrt_pred, f"div_spectrum_{level}": div_pred}
            result_dict = result_dict | {f"vrt_spectrum_{level}_truth": vrt_true, f"div_spectrum_{level}_truth": div_true}

    for variable in conf["single_level_variables"]:
        da_pred, da_true = get_data(sampler, rollout_files, variable, None)
        #compute and merge dicts
        result_dict = result_dict | _do_standard_eval_on_variable(w_lat, da_pred, da_true, variable, None)
        result_dict = result_dict | _do_special_eval_on_variable(w_lat, conf, fh, da_pred, da_true, variable, None)


    return result_dict


def _do_standard_eval_on_variable(w_lat, da_pred, da_true, variable, level):
    # ensemble spread, ensemble RMSE
    variable_name = f"{variable}"
    if level:
        variable_name += f"_{level}"
    
    # compute spread error
    result_dict = spread_error(da_pred, da_true, w_lat)

    # append variable name to keys
    result_dict = {f"{k}_{variable_name}": value for k, value in result_dict.items()}

    return result_dict

    
def _do_special_eval_on_variable(w_lat, conf, fh, da_pred, da_true, variable, level):
    # spectrum, binned spread-error hist, rank hist
    if fh not in conf["detailed_eval_hours"]:
        return {}
    
    variable_name = f"{variable}"
    if level:
        variable_name += f"_{level}"
        
    result_dict = {}
    #### compute ####

    # zonal spectrum
    spectrum = average_zonal_spectrum(da_pred, conf["grid"])
    spectrum_true = average_zonal_spectrum(da_true, conf["grid"])
    result_dict[f"zonal_spectrum_{variable_name}"] = spectrum
    result_dict[f"zonal_spectrum_{variable_name}_truth"] = spectrum_true

    # binned spread-skill
    binned_dict = binned_spread_skill(da_pred, da_true, conf["num_bins"])
    binned_dict = {f"{k}_{variable_name}": value for k, value in binned_dict.items()}
    result_dict = result_dict | binned_dict

    # rank histogram
    rank_hist = rank_histogram_apply(da_pred, da_true)
    result_dict[f"rank_hist_{variable_name}"] = rank_hist

    #insert other computations below
        
    return result_dict



def get_data(sampler, rollout_files, variable, level):
    """ uses a XRSamplerByYear object to sample the true data"""
    def select_darray(ds_given):
        ds_sel = ds_given[variable]
        if variable in "UVTQtot":
            return ds_sel.isel(level=level)
        return ds_sel
    
    # get pred and true data
    pred_da_list = []
    true_da_list = []
    for file in rollout_files:
        ds = xr.open_dataset(file)
        da = select_darray(ds)
        pred_da_list.append(da)
        
        da_true = select_darray(sampler(da.time.values[0]))
        true_da_list.append(da_true)

    # there will never be duplicate times
    # if rollout_files are all the same forecast hour (e.g. 3, 6, 24)
    da_pred = xr.concat(pred_da_list, dim='time')

    da_true = xr.concat(true_da_list, dim='time')
    return da_pred, da_true

def check_rollout_files(forecast_save_loc):
    """
    checks that all subfolders in forecast_save_loc has the same number of files
    """
    rollout_dirs = [dir for dir in os.listdir(forecast_save_loc) if os.path.isdir(join(forecast_save_loc, dir))]
    num_files = []
    for dir in rollout_dirs:
        dir = join(forecast_save_loc, dir)
        files = [f for f in os.listdir(dir) if os.path.isfile(join(dir, f))]
        num_files.append(len(files))
    assert all(num == num_files[0] for num in num_files), "not all rollouts have the same number of files"
    return num_files[0]

if __name__ == "__main__":
    description = "evaluate ensemble rollouts"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l, -w
    parser.add_argument(
        "-c",
        dest="eval_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )
    parser.add_argument(
        "-cpus",
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPU workers to use per GPU",
    )

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("eval_config")
    launch = int(args_dict.pop("launch"))
    num_cpus = int(args_dict.pop("num_cpus"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    with open(conf["config"]) as cf:
        model_conf = yaml.load(cf, Loader=yaml.FullLoader)

    # get save location for rollout
    if "save_forecast" not in conf or conf["save_forecast"] is None:
        forecast_save_loc = model_conf["predict"]["save_forecast"]
    else:
        forecast_save_loc = conf["save_forecast"]

    logging.info(f"evaluating forecast at {forecast_save_loc}")

    conf["save_filename"] = conf.get("save_filename", "ensemble_eval.parquet")
    if conf["save_filename"] is None:
        conf["save_filename"] = "ensemble_eval.parquet"
    # check that we are not overwriting an existing eval file
    eval_save_loc = join(forecast_save_loc, conf["save_filename"])
    assert not os.path.isfile(eval_save_loc), (
            f'''ensemble_eval results already exists at {eval_save_loc}, aborting. 
            Move or rename the existing file to run this script''')
    # rollout file check
    # checks that all ICs have the same forecast hours
    num_files = check_rollout_files(forecast_save_loc)

    # check all detailed_eval_hours exist:
    model_timestep = model_conf["data"]["lead_time_periods"]
    if max(conf["detailed_eval_hours"]) > num_files * model_timestep:
        raise RuntimeError(f"You specified {conf['detailed_eval_hours']} detailed hours"
                           + f"but only up to hour {num_files * model_timestep} exists")


    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    #local_rank, world_rank, world_size = get_rank_info(conf["predict"]["mode"])

    num_process = get_num_cpus() - 1 # count the num_cpus
    if "num_process" in conf:
        num_process = conf["num_process"]

    with mp.Pool(num_process) as p:
        evaluate(num_files, forecast_save_loc, conf, model_conf, p)
    # Ensure all processes are finished
    p.close()
    p.join()
