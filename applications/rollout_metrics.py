# ---------- #
# System
import logging
import multiprocessing as mp
import os
import sys
import warnings
from argparse import ArgumentParser
from collections import defaultdict

# ---------- #
# Numerics
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- #
import torch
import xarray as xr
import yaml

# ---------- #
# credit
from credit.data import concat_and_reshape, reshape_only
from credit.datasets import setup_data_loading
from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
from credit.distributed import distributed_model_wrapper, get_rank_info, setup
from credit.forecast import load_forecasts
from credit.metrics import LatWeightedMetrics, LatWeightedMetricsClimatology

from credit.models import load_model
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.parser import credit_main_parser, predict_data_check
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.postblock import GlobalEnergyFixer, GlobalMassFixer, GlobalWaterFixer
from credit.seed import seed_everything
from credit.transforms import Normalize_ERA5_and_Forcing, load_transforms

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def compute_metrics(metrics, y_pred, y, date_time, forecast_step, utc_datetime):
    """Compute metrics and update metrics_results."""
    metrics_results = {}
    metrics_dict = metrics(y_pred.float(), y.float(), forecast_datetime=date_time)
    for k, m in metrics_dict.items():
        metrics_results[k] = m.item()
    metrics_results["forecast_step"] = forecast_step
    metrics_results["datetime"] = utc_datetime
    return metrics_results


def predict(rank, world_size, conf, backend=None, p=None):
    # setup rank and world size for GPU-based rollout
    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["predict"]["mode"], backend)

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

    # batch and ensemble size
    batch_size = conf["predict"].get("batch_size", 1)
    ensemble_size = conf["predict"].get("ensemble_size", 1)
    if ensemble_size > 1:
        logger.info(f"Rolling out with ensemble size {ensemble_size}")

    # transform and ToTensor class
    logger.info("Loading z-score transforms")
    if conf["data"]["scaler_type"] == "std_new":
        logger.info("Loading Normalize_ERA5_and_Forcing transforms")
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        logger.warning(
            "Scaler type {} not supported".format(conf["data"]["scaler_type"])
        )
        raise

    # number of diagnostic variables
    varnum_diag = len(conf["data"]["diagnostic_variables"])

    # number of dynamic forcing + forcing + static
    static_dim_size = (
        len(conf["data"]["dynamic_forcing_variables"])
        + len(conf["data"]["forcing_variables"])
        + len(conf["data"]["static_variables"])
    )

    # clamp to remove outliers
    if conf["data"]["data_clamp"] is None:
        flag_clamp = False
    else:
        flag_clamp = True
        clamp_min = float(conf["data"]["data_clamp"][0])
        clamp_max = float(conf["data"]["data_clamp"][1])

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

    # Load the forecasts we wish to compute
    forecasts = load_forecasts(conf)
    if len(forecasts) % world_size != 0:
        raise ValueError(
            f'Number of forecast inits ({len(forecasts)}) given by conf["predict"]["duration"] x len(conf["predict"]["start_hours"]) should be divisible by number of processes/GPUs ({world_size})'
        )

    dataset = Predict_Dataset_Batcher(
        varname_upper_air=data_config["varname_upper_air"],
        varname_surface=data_config["varname_surface"],
        varname_dyn_forcing=data_config["varname_dyn_forcing"],
        varname_forcing=data_config["varname_forcing"],
        varname_static=data_config["varname_static"],
        varname_diagnostic=data_config["varname_diagnostic"],
        filenames=data_config["all_ERA_files"],
        filename_surface=data_config["surface_files"],
        filename_dyn_forcing=data_config["dyn_forcing_files"],
        filename_forcing=data_config["forcing_files"],
        filename_static=data_config["static_files"],
        filename_diagnostic=data_config["diagnostic_files"],
        fcst_datetime=forecasts,
        lead_time_periods=lead_time_periods,
        history_len=data_config["history_len"],
        skip_periods=data_config["skip_periods"],
        transform=load_transforms(conf),
        sst_forcing=data_config["sst_forcing"],
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )

    # Use a custom DataLoader so we get the len correct
    data_loader = BatchForecastLenDataLoader(dataset)

    # Set distributed mode
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]

    # Load the model
    if conf["predict"]["mode"] == "none":
        model = load_model(conf, load_weights=True).to(device)

    elif conf["predict"]["mode"] == "ddp":
        model = load_model(conf).to(device)
        # if conf["trainer"].get("compile", False):
        #     model = torch.compile(model)
        model = distributed_model_wrapper(conf, model, device)
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)
        if conf["predict"]["mode"] in ["ddp", "fsdp"]:
            load_msg = model.module.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
        else:
            load_msg = model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
        load_state_dict_error_handler(load_msg)

    elif conf["predict"]["mode"] == "fsdp":
        model = load_model(conf, load_weights=True).to(device)
        model = distributed_model_wrapper(conf, model, device)
        # Load model weights (if any), an optimizer, scheduler, and gradient scaler
        model = load_model_state(conf, model, device)

    model.eval()

    # Set up metrics and containers
    if "climatology" in conf["predict"]:
        metrics = LatWeightedMetricsClimatology(
            conf, climatology=xr.open_dataset(conf["predict"]["climatology"])
        )
    else:
        metrics = LatWeightedMetrics(conf, training_mode=False)
    metrics_results = defaultdict(list)
    dpf = None

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

        # y_pred allocation and results tracking
        results = []
        # model inference loop
        for k, batch in enumerate(data_loader):
            batch_size = batch["datetime"].shape[0]
            forecast_step = batch["forecast_step"].item()

            # Initial input processing
            if forecast_step == 1:
                # Set up dictionaries for metrics results
                metrics_results = [defaultdict(list) for _ in range(batch_size)]

                # Process the entire batch at once
                init_datetimes = [
                    datetime.utcfromtimestamp(batch["datetime"][i].item()).strftime(
                        "%Y-%m-%dT%HZ"
                    )
                    for i in range(batch_size)
                ]
                if "x_surf" in batch:
                    x = (
                        concat_and_reshape(batch["x"], batch["x_surf"])
                        .to(device)
                        .float()
                    )
                else:
                    x = reshape_only(batch["x"]).to(device).float()
                # create ensemble:
                if ensemble_size > 1:
                    x = torch.repeat_interleave(x, ensemble_size, 0)

            # Add forcing and static variables
            if "x_forcing_static" in batch:
                x_forcing_batch = (
                    batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                )
                if ensemble_size > 1:
                    x_forcing_batch = torch.repeat_interleave(
                        x_forcing_batch, ensemble_size, 0
                    )
                x = torch.cat((x, x_forcing_batch), dim=1)

            # Load y-truth
            if "y_surf" in batch:
                y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device).float()
            else:
                y = reshape_only(batch["y"]).to(device).float()

            if "y_diag" in batch:
                y_diag_batch = batch["y_diag"].to(device).permute(0, 2, 1, 3, 4)
                y = torch.cat((y, y_diag_batch), dim=1).to(device).float()

            # Clamp if needed
            if flag_clamp:
                x = torch.clamp(x, min=clamp_min, max=clamp_max)

            y_pred = model(x.float())

            # Post-processing blocks
            if flag_mass_conserve:
                if forecast_step == 1:
                    x_init = x.clone()
                input_dict = {"y_pred": y_pred, "x": x_init}
                input_dict = opt_mass(input_dict)
                y_pred = input_dict["y_pred"]

            if flag_water_conserve:
                input_dict = {"y_pred": y_pred, "x": x}
                input_dict = opt_water(input_dict)
                y_pred = input_dict["y_pred"]

            if flag_energy_conserve:
                input_dict = {"y_pred": y_pred, "x": x}
                input_dict = opt_energy(input_dict)
                y_pred = input_dict["y_pred"]

            # Transform predictions
            y_pred = state_transformer.inverse_transform(y_pred.cpu())
            y = state_transformer.inverse_transform(y.cpu())

            if (
                "use_laplace_filter" in conf["predict"]
                and conf["predict"]["use_laplace_filter"]
            ):
                y_pred = (
                    dpf.diff_lap2d_filt(y_pred.to(device).squeeze())
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .cpu()
                )

            # Calculate correct datetime for current forecast
            init_datetime = [datetime.utcfromtimestamp(t) for t in batch["datetime"]]
            utc_datetime = [
                t + timedelta(hours=lead_time_periods) for t in init_datetime
            ]
            _y_pred = y_pred.clone()

            # Prepare for next iteration
            y_pred = state_transformer.transform_array(y_pred).to(device)

            if ensemble_size > 1:
                _y_pred = _y_pred.view(batch_size, ensemble_size, *_y_pred.shape[1:])
            # Process each item in the batch (batch idx corresponds to init time)
            for j in range(batch_size):
                # Compute the metrics in parallel
                # if ensemble_size > 1, latWeightedMetrics also computes some ensemble metrics
                result = p.apply_async(
                    compute_metrics,
                    (
                        metrics,
                        _y_pred[j].unsqueeze(0),
                        y[j].unsqueeze(0),
                        batch["datetime"][j].item(),
                        forecast_step,
                        utc_datetime[j],
                    ),
                )
                results.append((j, result))  # Store the batch index with the result

                # Print to screen
                print_str = f"{rank=:} Forecast: {forecast_count + 1 + j} "
                print_str += f"Date: {utc_datetime[j].strftime('%Y-%m-%d %H:%M:%S')} "
                print_str += f"Hour: {forecast_step * lead_time_periods} "
                print(print_str)

            if history_len == 1:
                if "y_diag" in batch:
                    x = y_pred[:, :-varnum_diag, ...].detach()
                else:
                    x = y_pred.detach()
            else:
                if static_dim_size == 0:
                    x_detach = x[:, :, 1:, ...].detach()
                else:
                    x_detach = x[:, :-static_dim_size, 1:, ...].detach()

                if "y_diag" in batch:
                    x = torch.cat(
                        [x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2
                    )
                else:
                    x = torch.cat([x_detach, y_pred.detach()], dim=2)

            if batch["stop_forecast"].item():
                # Wait for processes to finish and collect metrics
                for batch_idx, result in results:
                    metric_dict = result.get()
                    for h, v in metric_dict.items():
                        metrics_results[batch_idx][h].append(v)

                # Save metrics files
                save_location = os.path.join(
                    os.path.expandvars(conf["save_loc"]), "metrics"
                )
                os.makedirs(save_location, exist_ok=True)

                for j in range(batch_size):
                    df = pd.DataFrame(metrics_results[j])
                    df.to_csv(os.path.join(save_location, f"{init_datetimes[j]}.csv"))

                # Clear everything
                results = []
                y_pred = None

                forecast_count += batch_size

        if distributed:
            torch.distributed.destroy_process_group()

    return 1


if __name__ == "__main__":
    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
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
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend for distribted training.",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
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
    backend = args_dict.pop("backend")

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

    conf = credit_main_parser(
        conf, parse_training=False, parse_predict=True, print_summary=False
    )
    predict_data_check(conf, print_summary=False)
    data_config = setup_data_loading(conf)

    # create a save location for rollout
    assert "save_forecast" in conf["predict"], (
        "Please specify the output dir through conf['predict']['save_forecast']"
    )

    forecast_save_loc = conf["predict"]["save_forecast"]
    os.makedirs(forecast_save_loc, exist_ok=True)

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

    with mp.Pool(num_cpus) as p:
        if conf["predict"]["mode"] in ["fsdp", "ddp"]:  # multi-gpu inference
            local_rank, world_rank, world_size = get_rank_info(conf["predict"]["mode"])
            _ = predict(world_rank, world_size, conf, p=p)
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

    # Ensure all processes are finished
    p.close()
    p.join()
