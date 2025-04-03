import os
import gc
import sys
import yaml
import logging
import warnings
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from collections import defaultdict

# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import cftime

# ---------- #
import torch

# ---------- #
# credit
from credit.models import load_model
from credit.seed import seed_everything
from credit.distributed import get_rank_info

from credit.data import (
    concat_and_reshape,
    reshape_only,
    Predict_Dataset,
)

from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.metrics import LatWeightedMetrics
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.parser import credit_main_parser, predict_data_check
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


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
    seed = conf["seed"]
    seed_everything(seed)

    # number of input time frames
    history_len = conf["data"]["history_len"]

    # length of forecast steps
    lead_time_periods = conf["data"]["lead_time_periods"]

    # transform and ToTensor class
    transform = load_transforms(conf)
    if conf["data"]["scaler_type"] == "std_new":
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        print("Scaler type {} not supported".format(conf["data"]["scaler_type"]))
        raise
    # ----------------------------------------------------------------- #
    # parse varnames and save_locs from config

    # upper air variables
    all_ERA_files = sorted(glob(conf["data"]["save_loc"]))
    varname_upper_air = conf["data"]["variables"]

    # surface variables
    varname_surface = conf["data"]["surface_variables"]

    if conf["data"]["flag_surface"]:
        surface_files = sorted(glob(conf["data"]["save_loc_surface"]))
    else:
        surface_files = None

    # diagnostic variables
    varname_diagnostic = conf["data"]["diagnostic_variables"]

    if conf["data"]["flag_diagnostic"]:
        diagnostic_files = sorted(glob(conf["data"]["save_loc_diagnostic"]))
    else:
        diagnostic_files = None

    # dynamic forcing variables
    varname_dyn_forcing = conf["data"]["dynamic_forcing_variables"]

    if conf["data"]["flag_dyn_forcing"]:
        dyn_forcing_files = sorted(glob(conf["data"]["save_loc_dynamic_forcing"]))
    else:
        dyn_forcing_files = None

    # forcing variables
    forcing_files = conf["data"]["save_loc_forcing"]
    varname_forcing = conf["data"]["forcing_variables"]

    # static variables
    static_files = conf["data"]["save_loc_static"]
    varname_static = conf["data"]["static_variables"]

    # number of diagnostic variables
    varnum_diag = len(conf["data"]["diagnostic_variables"])

    # number of dynamic forcing + forcing + static
    static_dim_size = (
        len(conf["data"]["dynamic_forcing_variables"])
        + len(conf["data"]["forcing_variables"])
        + len(conf["data"]["static_variables"])
    )

    # ------------------------------------------------------- #
    # clamp to remove outliers
    if conf["data"]["data_clamp"] is None:
        flag_clamp = False
    else:
        flag_clamp = True
        clamp_min = float(conf["data"]["data_clamp"][0])
        clamp_max = float(conf["data"]["data_clamp"][1])

    # ====================================================== #
    # postblock opts outside of model
    post_conf = conf["model"]["post_conf"]
    flag_mass_conserve = False
    flag_water_conserve = False
    flag_energy_conserve = False

    if post_conf["activate"]:
        if post_conf["global_mass_fixer"]["activate"]:
            if post_conf["global_mass_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalMassFixer outside of model")
                flag_mass_conserve = True
                opt_mass = GlobalMassFixer(post_conf)

        if post_conf["global_water_fixer"]["activate"]:
            if post_conf["global_water_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalWaterFixer outside of model")
                flag_water_conserve = True
                opt_water = GlobalWaterFixer(post_conf)

        if post_conf["global_energy_fixer"]["activate"]:
            if post_conf["global_energy_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalEnergyFixer outside of model")
                flag_energy_conserve = True
                opt_energy = GlobalEnergyFixer(post_conf)
    # ====================================================== #

    # ----------------------------------------------------------------- #\
    # get dataset
    dataset = Predict_Dataset(
        conf,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames=all_ERA_files,
        filename_surface=surface_files,
        filename_dyn_forcing=dyn_forcing_files,
        filename_forcing=forcing_files,
        filename_static=static_files,
        filename_diagnostic=diagnostic_files,
        fcst_datetime=load_forecasts(conf),
        history_len=history_len,
        rank=rank,
        world_size=world_size,
        transform=transform,
        rollout_p=0.0,
        which_forecast=None,
    )

    # setup the dataloder
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # flag for distributed inference
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    # ================================================================================ #
    if conf["predict"]["mode"] == "none":
        model = load_model(conf, load_weights=True).to(device)

    elif conf["predict"]["mode"] == "ddp":
        model = load_model(conf).to(device)
        # if conf["trainer"].get("compile", False):
        #     model = torch.compile(model)
        model = distributed_model_wrapper(conf, model, device)
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
    # ================================================================================ #

    model.eval()

    # get lat/lons from x-array
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])

    meta_data = load_metadata(conf)

    # Set up metrics and containers
    metrics = LatWeightedMetrics(conf)
    metrics_results = defaultdict(list)

    # Set up the diffusion and pole filters
    if (
        "use_laplace_filter" in conf["predict"]
        and conf["predict"]["use_laplace_filter"]
    ):
        dpf = Diffusion_and_Pole_Filter(
            nlat=conf["model"]["image_height"],
            nlon=conf["model"]["image_width"],
            device=device,
        )

    # Rollout
    with torch.no_grad():
        # forecast count = a constant for each run
        forecast_count = 0

        # y_pred allocation
        results = []

        # model inference loop
        for k, batch in enumerate(data_loader):
            # get the datetime and forecasted hours
            date_time = batch["datetime"].item()
            forecast_hour = batch["forecast_hour"].item()
            # initialization on the first forecast hour
            if forecast_hour == 1:
                # Initialize x and x_surf with the first time step
                if "x_surf" in batch:
                    # combine x and x_surf
                    # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                    # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                    x = (
                        concat_and_reshape(batch["x"], batch["x_surf"])
                        .to(device)
                        .float()
                    )
                else:
                    # no x_surf
                    x = reshape_only(batch["x"]).to(device).float()

                init_datetime = datetime.utcfromtimestamp(date_time)
                init_datetime_str = init_datetime.strftime("%Y-%m-%dT%HZ")

            # -------------------------------------------------------------------------------------- #
            # add forcing and static variables (regardless of fcst hours)
            if "x_forcing_static" in batch:
                # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                x_forcing_batch = (
                    batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                )

                # concat on var dimension
                x = torch.cat((x, x_forcing_batch), dim=1)

            # -------------------------------------------------------------------------------------- #
            # Load y-truth
            if "y_surf" in batch:
                # combine y and y_surf
                y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device).float()
            else:
                # no y_surf
                y = reshape_only(batch["y"]).to(device).float()

            # adding diagnostic vars to y
            if "y_diag" in batch:
                y_diag_batch = batch["y_diag"].to(device).permute(0, 2, 1, 3, 4)
                y = torch.cat((y, y_diag_batch), dim=1).to(device).float()

            save_location = os.path.join(
                    os.path.expandvars(conf["save_loc"]), "init_times"
                )
            os.makedirs(
                    save_location, exist_ok=True
                )
            torch.save(x, f'{save_location}/init_condition_tensor_{init_datetime_str}.pth')
            print('init cond saved to:' f'{save_location}/init_condition_tensor_{init_datetime_str}.pth') 
            break
    return 1


if __name__ == "__main__":
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

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    no_data = 0 if "no-data" not in args_dict else int(args_dict.pop("no-data"))
    subset = int(args_dict.pop("subset"))
    number_of_subsets = int(args_dict.pop("no_subset"))
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

    # ======================================================== #
    # handling config args
    conf = credit_main_parser(
        conf, parse_training=False, parse_predict=True, print_summary=False
    )
    predict_data_check(conf, print_summary=False)
    # ======================================================== #

    # create a save location for rollout
    # ---------------------------------------------------- #
    assert (
        "save_forecast" in conf["predict"]
    ), "Please specify the output dir through conf['predict']['save_forecast']"

    forecast_save_loc = conf["predict"]["save_forecast"]
    os.makedirs(forecast_save_loc, exist_ok=True)

    print("Save roll-outs to {}".format(forecast_save_loc))

    # Create a project directory (to save launch.sh and model.yml) if they do not exist
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)

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


    if number_of_subsets > 0:
        forecasts = load_forecasts(conf)
        if number_of_subsets > 0 and subset >= 0:
            subsets = np.array_split(forecasts, number_of_subsets)
            forecasts = subsets[subset - 1]  # Select the subset based on subset_size
            conf["predict"]["forecasts"] = forecasts

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])

    with mp.Pool(num_cpus) as p:
        if conf["predict"]["mode"] in ["fsdp", "ddp"]:  # multi-gpu inference
            _ = predict(world_rank, world_size, conf, p=p)
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

    # Ensure all processes are finished
    p.close()
    p.join()
