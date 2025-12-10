import os
import sys
import yaml
import logging
import warnings
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
import traceback

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
from credit.datasets import setup_data_loading
from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
from credit.data import concat_and_reshape, reshape_only
from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
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


class ForecastProcessor:
    def __init__(self, conf, device):
        self.conf = conf
        self.device = device

        self.batch_size = conf["predict"].get("batch_size", 1)
        self.ensemble_size = conf["predict"].get("ensemble_size", 1)
        self.lead_time_periods = conf["data"]["lead_time_periods"]

        # transform and ToTensor class
        if conf["data"]["scaler_type"] == "std_new":
            self.state_transformer = Normalize_ERA5_and_Forcing(conf)
        else:
            print("Scaler type {} not supported".format(conf["data"]["scaler_type"]))
            raise

        # get lat/lons from x-array
        self.latlons = xr.open_dataset(conf["loss"]["latitude_weights"]).load()
        # grab ERA5 (etc) metadata
        self.meta_data = load_metadata(conf)

        # Set up the diffusion and pole filters
        if (
            "use_laplace_filter" in conf["predict"]
            and conf["predict"]["use_laplace_filter"]
        ):
            self.dpf = Diffusion_and_Pole_Filter(
                nlat=conf["model"]["image_height"],
                nlon=conf["model"]["image_width"],
                device=device,
            )

    def process(self, y_pred, forecast_step, forecast_count, datetimes, save_datetimes):
        try:
            # Transform predictions
            conf = self.conf
            y_pred = self.state_transformer.inverse_transform(y_pred)

            # This will fail if not using torch multiprocessing AND using a GPU
            if (
                "use_laplace_filter" in conf["predict"]
                and conf["predict"]["use_laplace_filter"]
            ):
                y_pred = (
                    self.dpf.diff_lap2d_filt(y_pred.to(self.device).squeeze())
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .cpu()
                )

            # Calculate correct datetime for current forecast
            utc_datetimes = [
                datetime.utcfromtimestamp(datetimes[i].item())
                + timedelta(hours=self.lead_time_periods)
                for i in range(self.batch_size)
            ]

            # Convert to xarray and handle results
            for j in range(self.batch_size):
                upper_air_list, single_level_list = [], []
                for i in range(
                    self.ensemble_size
                ):  # ensemble_size default is 1, will run with i=0 retaining behavior of non-ensemble loop
                    darray_upper_air, darray_single_level = make_xarray(
                        y_pred[j + i : j + i + 1],  # Process each ensemble member
                        utc_datetimes[j],
                        self.latlons.latitude.values,
                        self.latlons.longitude.values,
                        conf,
                    )
                    upper_air_list.append(darray_upper_air)
                    single_level_list.append(darray_single_level)

                if self.ensemble_size > 1:
                    ensemble_index = xr.DataArray(
                        np.arange(self.ensemble_size), dims="ensemble_member_label"
                    )
                    all_upper_air = xr.concat(
                        upper_air_list, ensemble_index
                    )  # .transpose("time", ...)
                    all_single_level = xr.concat(
                        single_level_list, ensemble_index
                    )  # .transpose("time", ...)
                else:
                    all_upper_air = darray_upper_air
                    all_single_level = darray_single_level

                # Save the current forecast hour data in parallel
                save_netcdf_increment(
                    all_upper_air,
                    all_single_level,
                    save_datetimes[
                        forecast_count + j
                    ],  # Use correct index for current batch item
                    self.lead_time_periods * forecast_step,
                    self.meta_data,
                    conf,
                )

                print_str = f"Forecast: {forecast_count + 1 + j} "
                print_str += f"Date: {utc_datetimes[j].strftime('%Y-%m-%d %H:%M:%S')} "
                print_str += f"Hour: {forecast_step * self.lead_time_periods} "
                print(print_str)
        except Exception as e:
            print(traceback.format_exc())
            raise e


def predict(rank, world_size, conf, p):
    # setup rank and world size for GPU-based rollout
    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["predict"]["mode"])

    # Set up dataloading
    data_config = setup_data_loading(conf)

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

    # number of input time frames
    history_len = conf["data"]["history_len"]

    # length of forecast steps
    lead_time_periods = conf["data"]["lead_time_periods"]

    # batch size
    batch_size = conf["predict"].get("batch_size", 1)
    ensemble_size = conf["predict"].get("ensemble_size", 1)
    if ensemble_size > 1:
        logger.info(f"Rolling out with ensemble size {ensemble_size}")

    # number of diagnostic variables
    varnum_diag = len(conf["data"]["diagnostic_variables"])

    # number of dynamic forcing + forcing + static
    static_dim_size = (
        len(conf["data"]["dynamic_forcing_variables"])
        + len(conf["data"]["forcing_variables"])
        + len(conf["data"]["static_variables"])
    )

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

    # clamp to remove outliers
    if conf["data"]["data_clamp"] is None:
        flag_clamp = False
    else:
        flag_clamp = True
        clamp_min = float(conf["data"]["data_clamp"][0])
        clamp_max = float(conf["data"]["data_clamp"][1])

    # Load the forecasts we wish to compute
    forecasts = load_forecasts(conf)
    if len(forecasts) < batch_size:
        logger.warning(
            f"number of forecast init times {len(forecasts)} is less than batch_size {batch_size}, will result in under-utilization"
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
        skip_target=True,
    )

    # Use a custom DataLoader so we get the len correct
    data_loader = BatchForecastLenDataLoader(dataset)

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
        model = None

    # Put model in inference mode
    model.eval()

    # Rollout
    with torch.no_grad():
        # forecast count = a constant for each run
        forecast_count = 0

        # y_pred allocation and results tracking
        results = []
        save_datetimes = [0] * len(forecasts)

        # model inference loop
        for batch in data_loader:
            batch_size = batch["datetime"].shape[0]
            forecast_step = batch["forecast_step"].item()

            # Initial input processing
            if forecast_step == 1:
                # Process the entire batch at once
                init_datetimes = [
                    datetime.utcfromtimestamp(batch["datetime"][i].item()).strftime(
                        "%Y-%m-%dT%HZ"
                    )
                    for i in range(batch_size)
                ]
                save_datetimes[forecast_count : forecast_count + batch_size] = (
                    init_datetimes
                )

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

            # Add forcing and static variables for the entire batch
            if "x_forcing_static" in batch:
                x_forcing_batch = (
                    batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                )
                if ensemble_size > 1:
                    x_forcing_batch = torch.repeat_interleave(
                        x_forcing_batch, ensemble_size, 0
                    )
                x = torch.cat((x, x_forcing_batch), dim=1)

            # Clamp if needed
            if flag_clamp:
                x = torch.clamp(x, min=clamp_min, max=clamp_max)

            # Model inference on the entire batch
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

            result = p.apply_async(
                result_processor.process,
                (
                    y_pred.cpu(),
                    forecast_step,
                    forecast_count,
                    batch["datetime"],
                    save_datetimes,
                ),
            )
            results.append(result)

            # use `varnum_diag > 0` to check if diagnostic vars exists
            # use previous step y_pred as the next step input
            if history_len == 1:
                # cut diagnostic vars from y_pred, they are not inputs
                if varnum_diag > 0:
                    x = y_pred[:, :-varnum_diag, ...].detach()
                else:
                    x = y_pred.detach()

            # multi-step in
            else:
                if static_dim_size == 0:
                    x_detach = x[:, :, 1:, ...].detach()
                else:
                    x_detach = x[:, :-static_dim_size, 1:, ...].detach()

                # cut diagnostic vars from y_pred, they are not inputs
                if varnum_diag > 0:
                    x = torch.cat(
                        [x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2
                    )
                else:
                    x = torch.cat([x_detach, y_pred.detach()], dim=2)

            if batch["stop_forecast"][0]:
                # Wait for processes to finish
                for result in results:
                    result.get()

                y_pred = None

                if distributed:
                    torch.distributed.barrier()

                forecast_count += batch_size

        if distributed:
            torch.distributed.barrier()

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

    # handling config args
    conf = credit_main_parser(
        conf, parse_training=False, parse_predict=True, print_summary=False
    )
    predict_data_check(conf, print_summary=False)

    # create a save location for rollout
    assert (
        "save_forecast" in conf["predict"]
    ), "Please specify the output dir through conf['predict']['save_forecast']"

    forecast_save_loc = conf["predict"]["save_forecast"]
    os.makedirs(forecast_save_loc, exist_ok=True)

    logging.info("Save roll-outs to {}".format(forecast_save_loc))

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

    local_rank, world_rank, world_size = get_rank_info(conf["predict"]["mode"])

    with mp.Pool(num_cpus) as p:
        if conf["predict"]["mode"] in ["fsdp", "ddp"]:  # multi-gpu inference
            _ = predict(world_rank, world_size, conf, p=p)
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

    # Ensure all processes are finished
    p.close()
    p.join()
