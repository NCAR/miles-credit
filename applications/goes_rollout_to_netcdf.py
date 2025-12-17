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

    if forecast_conf["type"] in ["standard", "debugger"]:
        # start_dt = pd.Timestamp(year=forecast_conf["start_year"], month=1, day=1, hour=0)
        start_dt = pd.Timestamp(year=2022, month=7, day=1, hour=0)
        ic_interval = pd.Timedelta(2, "w")
        num_inits = 13
        rollout_init_times = [start_dt + k * ic_interval for k in range(num_inits)]
        rollout_init_times += [t + pd.Timedelta("12h") for t in rollout_init_times[-2:]] # 12h to last two SA convection
        rollout_init_times = ([pd.Timestamp("2022-01-01")] 
                              + [pd.Timestamp("2022-07-01") + pd.Timedelta("30m")]
                              + [pd.Timestamp("2022-07-01") +  pd.Timedelta("15h")] # NA convection case
                              + rollout_init_times)
        if forecast_conf["type"] == "debugger":
            return rollout_init_times[:5] + rollout_init_times[-4:]
        return rollout_init_times
    
    elif forecast_conf["type"] == "custom":
        start_dts = [pd.Timestamp(year=forecast_conf["start_year"],
                                month=forecast_conf["start_month"],
                                day=forecast_conf["start_day"],
                                hour=hour)
                                for hour in forecast_conf["start_hours"]
        ]
        ic_interval_days = pd.Timedelta(forecast_conf["ic_interval_days"], "d")
        num_inits = forecast_conf["num_inits"]

        rollout_init_times = []
        for k in range(num_inits):
            rollout_init_times += [start + k * ic_interval_days for start in start_dts]
        
        return rollout_init_times

    raise ValueError(f"{forecast_conf['type']} is not a valid rollout type")


class ForecastProcessor:
    def __init__(self, conf, device):
        self.conf = conf
        self.device = device
        padding_conf = conf["model"].get("padding_conf", {})
        self.pad_output = True if not padding_conf else not padding_conf.get("activate", False)

        self.save_forecast_dir = conf["predict"].get("save_forecast", None)
        if not self.save_forecast_dir:
            self.save_forecast_dir = os.path.expandvars(join(conf["save_loc"], "forecasts"))

        self.batch_size = conf["predict"].get("batch_size", 1)
        self.ensemble_size = conf["predict"].get("ensemble_size", 1)
        self.lead_time_periods = conf["data"]["lead_time_periods"]

        if self.ensemble_size > 1:
            self.ensemble_index = xr.DataArray(
                np.arange(self.ensemble_size), dims="ensemble_member_label"
            )

        # setup xarray saving
        self.ref_array = self._ref_array()

        #setup scaler
        scaler_ds_path = "/glade/derecho/scratch/dkimpara/goes-cloud-dataset/data_stats.nc"
        self.log_normal_scaling = conf["data"].get("log_normal_scaling", False)
        if self.log_normal_scaling:
            scaler_ds_path = "/glade/derecho/scratch/dkimpara/goes-cloud-dataset/data_stats_logC04.nc"
            logger.info("inverse log normalizing visible channels")
        self.scaler_ds = xr.open_dataset(scaler_ds_path)

    def _ref_array(self):
        ds = xr.open_dataset(self.conf["data"]["save_loc"])
        ref_array = ds.isel(t=0)["BT_or_R"].copy()

        self.channel = ref_array.channel

        self.padded_lat = ref_array.latitude
        self.padded_lon = ref_array.longitude

        if self.pad_output:
            self.padded_lat = np.pad(ref_array.latitude, 
                                (11,10), 
                                mode="linear_ramp", 
                                end_values=(-50.1 - 11* 0.1, 50.1 + 10*.1))
            self.padded_lon = np.pad(ref_array.longitude, 
                                (19,18), 
                                mode="linear_ramp", 
                                end_values=(-121.1 - 19 * 0.1, -28.9 + 18*.1))
        
    def inverse_transform_ABI(self, da):
        unscaled = da * self.scaler_ds["std"] + self.scaler_ds["mean"]

        if self.log_normal_scaling: #log normal for ch4
            unscaled[:,0] = np.exp(unscaled[:,0])
        
        return unscaled
    
    def make_dataarray(self, y_pred_member, save_datetime):
        y_pred_member = y_pred_member.squeeze(2) # 1,c,t,lat,lon -> 1, c, lat lon

        da = xr.DataArray(y_pred_member, 
                        dims=[ "t", "channel", "latitude", "longitude"],
                                coords={
                                    "t": [pd.Timestamp(save_datetime)],
                                    "channel": self.channel,
                                    "latitude": self.padded_lat,
                                    "longitude": self.padded_lon,
                                },
                        )
        da = self.inverse_transform_ABI(da)
        return da
    
    def save_dataarray(self, pred_da, init_datetime, save_datetime):
        
        pred_ds = xr.Dataset({"BT_or_R": pred_da})
        pred_ds.to_netcdf(join(self.save_forecast_dir,
                               init_datetime,
                               f"{save_datetime}.nc"),
                               mode="w")

    def process(self, y_pred, init_datetimes, save_datetimes):
        try:
            for dt in init_datetimes:
                os.makedirs(join(self.save_forecast_dir, dt), exist_ok=True)

            # Convert to xarray and handle results
            for j, times in enumerate(zip(init_datetimes, save_datetimes)):
                init_datetime, save_datetime = times
                y_pred_ensemble = []
                for i in range(self.ensemble_size):  
                    # ensemble_size default is 1, will run with i=0 retaining behavior of non-ensemble loop
                    y_pred_member = self.make_dataarray(y_pred[j + i : j + i + 1], save_datetime)
                    y_pred_ensemble.append(y_pred_member)

                if self.ensemble_size > 1:
                    pred_da = xr.concat(y_pred_ensemble, self.ensemble_index)
                else:
                    pred_da = y_pred_member

                # Save the current forecast hour data
                self.save_dataarray(pred_da, init_datetime, save_datetime)

                print_str = f"processed {save_datetime} initialized at {init_datetime}"
                print(print_str)
        except Exception as e:
            print(traceback.format_exc())
            raise e


def predict(rank, world_size, conf, p):
    # setup rank and world size for GPU-based rollout
    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["predict"]["mode"])


    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # config settings
    seed_everything(conf["seed"])


    # batch size
    batch_size = conf["predict"].get("batch_size", 1)
    ensemble_size = conf["predict"].get("ensemble_size", 1)
    if ensemble_size > 1:
        logger.info(f"Rolling out with ensemble size {ensemble_size}")

    # clamp to remove outliers
    if conf["data"]["data_clamp"] is None:
        flag_clamp = False
    else:
        flag_clamp = True
        clamp_min = float(conf["data"]["data_clamp"][0])
        clamp_max = float(conf["data"]["data_clamp"][1])

    # Load the forecasts we wish to compute
    rollout_init_times = get_rollout_init_times(conf)

    logger.info(f"rolling out with init times {rollout_init_times}")

    if len(rollout_init_times) < batch_size:
        logger.warning(
            f"number of forecast init times {len(rollout_init_times)} is less than batch_size {batch_size}, will result in under-utilization"
        )

    dataset = load_predict_dataset(conf, 0, 0, rollout_init_times, device)
    dataloader = load_dataloader(conf, dataset, 0, 1, is_predict=True)

    # Class for saving in parallel
    result_processor = ForecastProcessor(conf, device)

    # Warning -- see next line
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]

    # Load the model
    if conf["predict"]["mode"] == "none":
        model = load_model(conf, load_weights=True).to(device)
    elif conf["predict"]["mode"] == "ddp":
        model = load_model(conf).to(device)
        model = distributed_model_wrapper(conf, model, device)
        save_loc = os.path.expandvars(conf["save_loc"])
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)
        load_msg = model.module.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        load_state_dict_error_handler(load_msg)
    elif conf["predict"]["mode"] == "fsdp":
        model = load_model(conf, load_weights=True).to(device)
        model = distributed_model_wrapper(conf, model, device)
        # Load model weights (if any), an optimizer, scheduler, and gradient scaler
        model = load_model_state(conf, model, device)
    else:
        raise ValueError(f'''{conf["predict"]["mode"]} is not a valid prediction mode''')

    # Put model in inference mode
    model.eval()

    mode = None

    num_batches = len(dataloader)

    dl = iter(dataloader)

    # forecast count = a constant for each run
    forecast_count = 0
    
    # y_pred allocation and results tracking
    results = []
    
    start_time = time.time()
    # Rollout
    with torch.no_grad():
        # model inference loop
        for _ in range(num_batches):

            # init
            batch = next(dl)
            if "era5" in batch.keys():
                batch_era5 = batch["era5"]
            init_datetimes = batch["datetime"]
            batch_size = len(init_datetimes)

            mode = batch["mode"][0]

            x = batch["x"].to(device).float()
            # create ensemble:
            if ensemble_size > 1:
                x = torch.repeat_interleave(x, ensemble_size, 0)
            if "era5" in batch.keys():
                era5_static = batch_era5["static"].to(device)

            while mode != "stop":
                if flag_clamp:
                    x = torch.clamp(x, min=clamp_min, max=clamp_max)

                if "era5" in batch.keys():
                    x_era5 = torch.concat([batch_era5["prognostic"].to(device),
                                           era5_static.detach().clone(),
                                           batch_era5["dynamic_forcing"].to(device)],
                                           dim=1).float()
                    forcing_t_delta = batch_era5["timedelta_seconds"].to(device).float()

                    if ensemble_size > 1:
                        x_era5 = torch.repeat_interleave(x_era5, ensemble_size, 0)
                    if flag_clamp:
                        x_era5 = torch.clamp(x_era5, min=clamp_min, max=clamp_max)
                ###############################
                
                # Model inference on the entire batch
                y_pred = model(x, x_era5, forcing_t_delta) if "era5" in batch.keys() else model(x)

                batch = next(dl) # load in y timestep for metric computation and forecast timestamp
                
                save_datetimes = batch["datetime"]
                # process according to y timestep

                result = p.apply_async(
                    result_processor.process,
                    (y_pred.cpu(),
                     init_datetimes,
                     save_datetimes,
                    ),
                )
                results.append(result)

                # result_processor.process(
                #         y_pred.cpu(),
                #         init_datetimes,
                #         save_datetimes,
                #     )

                # reset for next iter
                mode = batch["mode"][0]
                x = y_pred.detach()

                if mode == "stop":
                    # Wait for processes to finish
                    for result in results:
                        result.get()

                    y_pred = None

                    if distributed:
                        torch.distributed.barrier()

                    forecast_count += batch_size


        if distributed:
            torch.distributed.barrier()

    end_time = time.time()
    logger.info(f"total rollout time: {end_time - start_time:02}s for {len(rollout_init_times) * dataset.num_forecast_steps} forecasts")
    return 1


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

    ch.setFormatter(formatter)
    root.addHandler(ch)

    # create a save location for rollout
    forecast_save_loc = conf["predict"].get("save_forecast", None)
    if not forecast_save_loc:
        forecast_save_loc = os.path.expandvars(join(conf["save_loc"], "forecasts"))
        logger.warning("forecast_save_loc not specified, saving to default location-")
    os.makedirs(forecast_save_loc, exist_ok=True)

    logging.info("Save roll-outs to {}".format(forecast_save_loc))

    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logger.info(f"Setting the running mode to {mode}")
        conf["predict"]["mode"] = mode

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

    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="Derecho parallelism",
    #         name=f"Worker {os.environ["RANK"]} {os.environ["WORLD_SIZE"]}"
    #         # track hyperparameters and run metadata
    #         config=conf
    #     )

    if number_of_subsets > 0:
        forecasts = load_forecasts(conf)
        if number_of_subsets > 0 and subset >= 0:
            subsets = np.array_split(forecasts, number_of_subsets)
            forecasts = subsets[subset - 1]  # Select the subset based on subset_size
            conf["predict"]["forecasts"] = forecasts

    seed = conf["seed"]
    seed_everything(seed)

    local_rank, world_rank, world_size = get_rank_info(conf["predict"]["mode"])

    logger.info(f"saving forecasts with {num_cpus} cpus")
    with mp.Pool(num_cpus) as p:
        if conf["predict"]["mode"] in ["fsdp", "ddp"]:  # multi-gpu inference
            _ = predict(world_rank, world_size, conf, p=p)
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

        # Ensure all processes are finished
        p.close()
        p.join()
    
if __name__ == "__main__":
    main()
