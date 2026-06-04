import os
import sys
import time
import yaml
import logging
import warnings
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
import traceback
from os.path import join

import pandas as pd
# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

# ---------- #
import torch

# ---------- #
# credit
from credit.models import load_model
from credit.seed import seed_everything
from credit.distributed import get_rank_info
from credit.datasets.goes_load_dataset_and_dataloader import load_predict_dataset, load_dataloader
from credit.pbs import launch_script, launch_script_mpi
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import subprocess

def get_num_cpus():
    if "glade" in os.getcwd():
        num_cpus = subprocess.run(
            "qstat -f $PBS_JOBID | grep Resource_List.ncpus",
            shell=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout.split()[-1]
    else:
        num_cpus = os.cpu_count()
    
    return int(num_cpus)



def get_rollout_init_times(conf):

    forecast_conf = conf["predict"]["forecasts"]
    
    if forecast_conf["type"] == "debugger":
        init_times_str = [
                            "2022-05-21T17:55:06", # NA spring case
                            "2022-06-30T23:55:06", 
                            "2022-07-01T14:55:06", # NA convective case
                            "2022-12-02T11:55:06", 
                            "2022-12-15T23:55:05", #SA convective case
                            "2022-12-16T11:55:06", #SA convective case
                         ]
        rollout_init_times = ([pd.Timestamp("2022-07-01") + pd.Timedelta("30m")] # check 30min offset case
                              + [pd.Timestamp(time) for time in init_times_str])
        return rollout_init_times
    
    elif forecast_conf["type"] == "standard":
        start_dt = pd.Timestamp(year=2022, month=7, day=1, hour=6)
        ic_interval = pd.Timedelta(2, "w")
        num_inits = 13
        rollout_init_times = [start_dt + k * ic_interval for k in range(num_inits)]
        rollout_init_times += [t + pd.Timedelta("12h") for t in rollout_init_times] # 18z inits
        # rollout_init_times = ([pd.Timestamp("2022-01-01")] 
        #                       + [pd.Timestamp("2022-07-01") + pd.Timedelta("30m")] # check 30m offset case
        #                       + [pd.Timestamp("2022-07-01") +  pd.Timedelta("15h")] # NA convection case
                            #   + rollout_init_times)
        return rollout_init_times
    
    elif forecast_conf["type"] == "custom":
        rollout_init_times = [pd.Timestamp(time) for time in forecast_conf["init_times"]]
        return rollout_init_times
    elif forecast_conf["type"] == "2025_test_rollout":
        rollout_init_times = [
            "2025-06-12T23:55:00",
            "2025-06-13T05:55:00",
            "2025-06-13T23:55:00",
            "2025-06-14T05:55:00",
            "2025-06-14T11:55:00",
            "2025-06-14T17:55:00",
            "2025-06-14T23:55:00",
            "2025-06-15T05:55:00",
            "2025-06-15T11:55:00",
            "2025-06-15T17:55:00",
            "2025-06-15T23:55:00",
            "2025-06-16T05:55:00",
            "2025-06-16T11:55:00",
            "2025-06-16T17:55:00",
            "2025-06-16T23:55:00",
            "2025-06-17T05:55:00",
            "2025-06-17T11:55:00",
            "2025-06-17T17:55:00",
            "2025-06-17T23:55:00",
            "2025-06-18T05:55:00",
            "2025-06-18T11:55:00",
            "2025-06-18T17:55:00",
            "2025-06-18T23:55:00",
            "2025-06-19T05:55:00",
            "2025-06-19T11:55:00",
            "2025-06-19T17:55:00",
            "2025-06-19T23:55:00",
            "2025-06-20T05:55:00",
            "2025-06-20T11:55:00",
            "2025-06-20T17:55:00",
            "2025-06-20T23:55:00",
            "2025-06-21T05:55:00",
            "2025-06-21T11:55:00",
            "2025-06-21T17:55:00",
            "2025-06-21T23:55:00",
            "2025-06-22T05:55:00",
            "2025-06-22T11:55:00",
            "2025-06-22T17:55:00",
            "2025-06-22T23:55:00",
            "2025-06-23T05:55:00",
            "2025-06-23T23:55:00",
            "2025-06-24T05:55:00",
            "2025-06-24T11:55:00",
            "2025-06-24T17:55:00",
            "2025-06-24T23:55:00",
            "2025-06-25T05:55:00",
            "2025-06-25T11:55:00",
            "2025-06-25T17:55:00",
            "2025-06-25T23:55:00",
            "2025-06-26T05:55:00",
            "2025-06-26T11:55:00",
            "2025-06-26T17:55:00",
            "2025-06-26T23:55:00",
            "2025-06-27T05:55:00",
            "2025-06-27T11:55:00",
            "2025-06-27T17:55:00",
            "2025-06-27T23:55:00",
            "2025-06-28T05:55:00",
            "2025-06-28T11:55:00",
            "2025-06-28T17:55:00",
            "2025-06-28T23:55:00",
            "2025-06-29T05:55:00",
            "2025-06-29T11:55:00",
            "2025-06-29T17:55:00",
            "2025-06-29T23:55:00",
            "2025-06-30T05:55:00",
            "2025-06-30T11:55:00",
            "2025-06-30T17:55:00",
            "2025-06-30T23:55:00",
            "2025-07-01T05:55:00",
            "2025-07-01T11:55:00",
            "2025-07-01T17:55:00",
            "2025-07-01T23:55:00",
            "2025-07-02T05:55:00",
            "2025-07-02T11:55:00",
            "2025-07-02T17:55:00",
            "2025-07-02T23:55:00",
            "2025-07-03T05:55:00",
            "2025-07-03T11:55:00",
            "2025-07-03T17:55:00",
            "2025-07-03T23:55:00",
            "2025-07-04T05:55:00",
            "2025-07-04T11:55:00",
            "2025-07-04T17:55:00",
            "2025-07-04T23:55:00",
            "2025-07-05T05:55:00",
            "2025-07-05T11:55:00",
            "2025-07-05T17:55:00",
            "2025-07-05T23:55:00",
            "2025-07-06T05:55:00",
            "2025-07-06T11:55:00",
            "2025-07-06T17:55:00",
            "2025-07-06T23:55:00",
            "2025-07-07T05:55:00",
            "2025-07-07T23:55:00",
            "2025-07-08T05:55:00",
            "2025-07-08T11:55:00",
            "2025-07-08T17:55:00",
            "2025-07-08T23:55:00",
        ]

        return [pd.Timestamp(t) for t in rollout_init_times]
    raise ValueError(f"{forecast_conf['type']} is not a valid rollout type")

def main():
    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l, -w
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    parser.add_argument(
        "-r",
        "--rollout-config",
        dest="rollout_config",
        type=str,
        default=None,
        help="rollout config file to override model rollout config",
    )

    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )

    parser.add_argument(
        "-w",
        "--world-size",
        type=int,
        default=4,
        help="Number of processes (world size) for multiprocessing",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=0,
        help="Update the config to use none, DDP, or FSDP",
    )

    parser.add_argument(
        "-nd",
        "--no-data",
        type=str,
        default=0,
        help="If set to True, only pandas CSV files will we saved for each forecast",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=int,
        default=False,
        help="Predict on subset X of forecasts",
    )
    parser.add_argument(
        "-ns",
        "--no_subset",
        type=int,
        default=False,
        help="Break the forecasts list into X subsets to be processed by X GPUs",
    )
    parser.add_argument(
        "-cpus",
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPU workers to use per GPU",
    )

    parser.add_argument(
        "-debug",
        "--debug",
        type=int,
        default=-1, # -1 does nothing
        help="debug mode",
    )

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    rollout_config = args_dict.pop("rollout_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    subset = int(args_dict.pop("subset"))
    number_of_subsets = int(args_dict.pop("no_subset"))
    num_cpus = int(args_dict.pop("num_cpus"))
    debug = int(args_dict.pop("debug"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()


    ch.setFormatter(formatter)
    root.addHandler(ch)
    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    if rollout_config:
        with open(rollout_config) as cf:
            rollout_conf = yaml.load(cf, Loader=yaml.FullLoader)
        conf["predict"] = rollout_conf["predict"]
    
    if debug == 1 or conf["model"]["type"] == "debugger" or conf["predict"]["forecasts"]["type"] == "debugger" or conf.get("debug", False):
        print("setting logging to debug")
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    if debug == 0:
        ch.setLevel(logging.INFO) 

    conf["predict"]["thread_workers"] = 0
    del conf["data"]["source"]["ERA5"]
    # Load the forecasts we wish to compute
    rollout_init_times = get_rollout_init_times(conf)
    if rollout_init_times[0].year == 2025:
        conf["data"]["save_loc"] = conf["data"]["save_loc"][:-5] + "_2025.zarr"
        logging.warning(f"loading 2025 zarr from {conf['save_loc']}")
    logger.info(f"rolling out with init times {rollout_init_times}")

    timesteps = ["10m", "20m", "30m", "1h"]
    for tstep in timesteps:
        print(tstep)
        conf["data"]["timestep"] = tstep

        conf["predict"]["forecasts"]["num_forecast_hours"] = 12
        _ = load_predict_dataset(conf, 0, 0, rollout_init_times, device=torch.device("cpu"))

        del conf["predict"]["forecasts"]["num_forecast_hours"]
        conf["predict"]["forecasts"]["num_forecast_steps"] = 1

        _ = load_predict_dataset(conf, 0, 0, rollout_init_times, device=torch.device("cpu"))

if __name__ == "__main__":
    main()