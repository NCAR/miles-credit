# ensemble evaluation suite for rollouts generated with rollout_to_netcdf.py
# parallelizes across CPUs
# WARNING: DOES NOT USE model config file
# see config/example_ensemble_eval.yml for an example config for this rollout

from argparse import ArgumentParser
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
from credit.verification import load_verification
from credit.datasets.goes_load_dataset_and_dataloader import load_verification_dataset

from dateutil.parser import parse, ParserError

def is_timestamp(s):
    try:
        parse(s)
        return True
    except (ParserError, TypeError):
        return False

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
    # parser.add_argument(
    #     "-cpus",
    #     "--num_cpus",
    #     type=int,
    #     default=8,
    #     help="Number of CPU workers to use per GPU",
    # )

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("eval_config")
    launch = int(args_dict.pop("launch"))
    # num_cpus = int(args_dict.pop("num_cpus"))

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
    # get the model config
    with open(conf["config"]) as cf:
        model_conf = yaml.load(cf, Loader=yaml.FullLoader)

    # get save location for rollout
    forecast_save_loc = conf.get("save_forecast", None)
    if not forecast_save_loc: # save_loc not specified by config
        logging.warning("save_forecast not specified in eval config, using forecast_save_loc in model config")
        forecast_save_loc = model_conf["predict"].get("save_forecast", None)
        if not forecast_save_loc: # save_loc not specified by model config, use default
            forecast_save_loc = os.path.expandvars(join(conf["save_loc"], "forecasts"))
            logging.warning("save_forecast not specified in model config, using default location- defined by model save_loc")
    logging.info(f"evaluating forecast at {forecast_save_loc}")
    
    conf["save_filename"] = conf.get("save_filename", "verif.parquet")
    # check that we are not overwriting an existing eval file
    eval_save_loc = join(forecast_save_loc, conf["save_filename"])
    if not conf.get("overwrite", False):
        assert not os.path.isfile(eval_save_loc), (
                f'''{conf["save_filename"]} results already exists at {eval_save_loc}, aborting. 
                Move or rename the existing file to run this script''')
    else:
        logging.warning(f"potentially overwriting existing file {conf['save_filename']}")

    # get forecast dirs
    dirs = [d for d in Path(forecast_save_loc).iterdir() if d.is_dir() and is_timestamp(d.name)]
    logging.info(f"in these dirs: {[d.name for d in dirs]}")

    # load the verification
    verification = load_verification(conf)

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

    num_process = conf.get("num_process", max(1, get_num_cpus() - 1))

    dataset = load_verification_dataset(model_conf)
    climo = xr.open_dataset(conf["climo_file"])["mean"]


    with mp.Pool(num_process) as p:
        # process each forecast separately:
        df_dict = {}
        for dir in dirs:
            # run evaluation, needs to output a result dict. all entries need to be a type with addition
            result = verification(dir, p, conf, model_conf, dataset, climo)
 
            # reformat, save in memory, and save to parquet
            result.sort(key= lambda x: x["forecast_step"]) # sort of forecast steps are in order
            df = pd.DataFrame(result)
            df.attrs["init_times"] = dir.name
            df_dict[dir.name] = df

            # save to forecast_save_loc with a timestamp in the name
            intermediate_eval_save_loc = join(forecast_save_loc, f"verif_{dir.name}.parquet")
            df.to_parquet(intermediate_eval_save_loc) # parquet keeps all the dtypes, don't have to split up np arrays in the entries
            logging.info(f"saved verification of {dir.name} to {intermediate_eval_save_loc}")

    # take nanmean of all verifications and save
    def nanmean_cell(*values):
        arrays = [np.atleast_1d(np.array(v, dtype=float)) for v in values]
        result = np.nanmean(arrays, axis=0)
        return result[0] if result.size == 1 else result
    
    dfs = list(df_dict.values())
    df = pd.DataFrame(
        {col: [nanmean_cell(*values) for values in zip(*[df[col] for df in dfs])]
        for col in dfs[0].columns},
        index=dfs[0].index
    )
    df.attrs["init_times"] = list(df_dict.keys())

    # df.sort(key= lambda x: x["forecast_step"]) # sort of forecast hours are in order
    df.to_parquet(eval_save_loc) # parquet keeps all the dtypes, don't have to split up np arrays in the entries
    logging.info(f"saved verification to {eval_save_loc}")

    # Ensure all processes are finished
    p.close()
    p.join()
