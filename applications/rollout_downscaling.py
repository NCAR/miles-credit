import os
import sys
import copy
import yaml
import logging
import warnings

from pathlib import Path
from argparse import ArgumentParser

# import multiprocessing as mp
# import traceback

# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

# ---------- #
import torch

# ---------- #
# credit

from credit.datasets.load_dataset_and_dataloader import load_dataset, load_dataloader
from credit.distributed import distributed_model_wrapper, setup
from credit.models import load_model
from credit.output_downscaling import OutputWrangler
from credit.parser import credit_main_parser, predict_data_check
from credit.pbs import launch_script, launch_script_mpi
from credit.seed import seed_everything
from credit.distributed import get_rank_info

# from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
# from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
# from credit.datasets import setup_data_loading
# from credit.data import concat_and_reshape, reshape_only
# from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
# from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
# from credit.forecast import load_forecasts
# from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
# from credit.output import load_metadata, make_xarray, save_netcdf_increment
# from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# class ForecastProcessor:
#     def __init__(self, conf, device):
#         self.conf = conf
#         self.device = device
# 
#         self.batch_size = conf["predict"].get("batch_size", 1)
#         self.ensemble_size = conf["predict"].get("ensemble_size", 1)
#         self.lead_time_periods = conf["data"]["lead_time_periods"]
# 
#         # transform and ToTensor class
#         if conf["data"]["scaler_type"] == "std_new":
#             self.state_transformer = Normalize_ERA5_and_Forcing(conf)
#         else:
#             print("Scaler type {} not supported".format(conf["data"]["scaler_type"]))
#             raise
# 
# 
#     def process(self, y_pred, forecast_step, forecast_count, datetimes, save_datetimes):
#         try:
#             # Transform predictions
#             conf = self.conf
#             y_pred = self.state_transformer.inverse_transform(y_pred)
# 
#             # # This will fail if not using torch multiprocessing AND using a GPU
#             # if (
#             #     "use_laplace_filter" in conf["predict"]
#             #     and conf["predict"]["use_laplace_filter"]
#             # ):
#             #     y_pred = (
#             #         self.dpf.diff_lap2d_filt(y_pred.to(self.device).squeeze())
#             #         .unsqueeze(0)
#             #         .unsqueeze(2)
#             #         .cpu()
#             #     )
# 
#             # Calculate correct datetime for current forecast
#             utc_datetimes = [
#                 datetime.utcfromtimestamp(datetimes[i].item())
#                 + timedelta(hours=self.lead_time_periods)
#                 for i in range(self.batch_size)
#             ]
# 
#             # Convert to xarray and handle results
#             for j in range(self.batch_size):
#                 upper_air_list, single_level_list = [], []
#                 for i in range(
#                     self.ensemble_size
#                 ):  # ensemble_size default is 1, will run with i=0 retaining behavior of non-ensemble loop
#                     darray_upper_air, darray_single_level = make_xarray(
#                         y_pred[j + i : j + i + 1],  # Process each ensemble member
#                         utc_datetimes[j],
#                         self.latlons.latitude.values,
#                         self.latlons.longitude.values,
#                         conf,
#                     )
#                     upper_air_list.append(darray_upper_air)
#                     single_level_list.append(darray_single_level)
# 
#                 if self.ensemble_size > 1:
#                     ensemble_index = xr.DataArray(
#                         np.arange(self.ensemble_size), dims="ensemble_member_label"
#                     )
#                     all_upper_air = xr.concat(
#                         upper_air_list, ensemble_index
#                     )  # .transpose("time", ...)
#                     all_single_level = xr.concat(
#                         single_level_list, ensemble_index
#                     )  # .transpose("time", ...)
#                 else:
#                     all_upper_air = darray_upper_air
#                     all_single_level = darray_single_level
# 
#                 # Save the current forecast hour data in parallel
#                 save_netcdf_increment(
#                     all_upper_air,
#                     all_single_level,
#                     save_datetimes[
#                         forecast_count + j
#                     ],  # Use correct index for current batch item
#                     self.lead_time_periods * forecast_step,
#                     self.meta_data,
#                     conf,
#                 )
# 
#         except Exception as e:
#             print(traceback.format_exc())
#             raise e


def predict(rank, world_size, conf, p):

    # setup rank and world size for GPU-based rollout
    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["predict"]["mode"])

    # # Set up dataloading
    # data_config = setup_data_loading(conf)

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # this should already have been called
    # # config settings
    #seed_everything(conf["seed"])

    # # this should already have been called
    # conf = credit_main_parser(
    #     conf, parse_training=False, parse_predict=True, print_summary=True
    # )


    # Warning -- see next line   # ? ? ?
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

    # post-processing setup goes here

    dataset = load_dataset(conf)

    # prognostic data initialization
    if conf['predict']['downscaling']['driver'] == 'test':
        x0 = dataset[0]['x']
    else:
        raise(ValueError("non-testing prog data init not implemented yet"))
        # when downscaling other datasets, we don't have high-res
        # truth to initialize with.  Options:
        #
        # * intialize prog vars from climatology
        #       pro: easy to generate
        #       con: overly smooth, non-seasonal
        #            (could use seasclim -> extra steps)
        # * initialize w/ interpolated GCM data
        #       pro: results consistant w/ boundaries
        #       con: require wrangling an extra dataset
        #            GCM may not have all needed vars
        #            blocky / overly smooth data
        # * initialize w/ noise
        #       pro: very easy & very fast
        #       con: need to know normalization range
        #            very noisy -> spinup period needed
        #
        # in any case, we could add a small perturbation
        # afterward to get an initial-condition ensemble

        pass

    dataset.mode = "infer"
    data_loader = load_dataloader(conf, dataset, is_train=False)

    output_wrangler = OutputWrangler(dataset, **conf['predict']['output'])

    # number of channels by variable type
    nbound = conf['model']['channels']['boundary']
    nprog  = conf['model']['channels']['prognostic']
    ndiag  = conf['model']['channels']['diagnostic']
    
    # Rollout
    first_loop = True
    with torch.no_grad():

        # need to collect all the asynchronous output writing promises
        # so we can make sure they all complete before we exit.
        results = []

        # model inference loop

        for timestep in data_loader:

            if first_loop:
                x = x0
                prognostic = x[:,nbound:, ...]
                first_loop = False
                del x0
            else:
                # recycle prognostic outputs from last step as inputs for this step
                # tensors are dimensioned [batch, var, time, x, y]

                old_prog = prognostic[:, :, 1:, ...]
                new_prog = y_pred[:, :-ndiag, ...].detach()
                prognostic = torch.cat([old_prog, new_prog], dim=2)

                x = torch.cat([timestep['x'], prognostic], dim=1)
                
            y_pred = model(x.float())

           # (post-processing blocks go here)
           
           result = p.apply_async(
               output_wrangler.process,
               (y_pred.cpu(), timestep['dates'])
           )
                 
            results.append(result)

            # end of inference loop

        # wait for asynchronous write processes to finish
        for result in results:
            result.get()

        if distributed:
            torch.distributed.barrier()

    return 1


if __name__ == "__main__":
    # description = "Rollout AI-NWP forecasts"
    description = "Rollout ML downscaling"
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

    # parser.add_argument(
    #     "-nd",
    #     "--no-data",
    #     type=str,
    #     default=0,
    #     help="If set to True, only pandas CSV files will we saved for each forecast",
    # )
    
    parser.add_argument(
        "-cpus",
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPU workers to use per GPU",
    )

    # parse arguments
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    # no_data = 0 if "no-data" not in args_dict else int(args_dict.pop("no-data"))
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
    ## todo: data check for downscaling mode
    # predict_data_check(conf, print_summary=False)

    # for downscaling rollout, we need to write data incrementally.
    # Default pattern is to write 2d & 3d data to separate files, one
    # file per timestep.  This can later be extended to collect
    # multiple timesteps into a file and write larger chunks.

    # ancillary data & metadata come from a template file.  Time
    # coordinate comes from the driving model.

    # for now, we just write write everything into one directory
    # todo: come up with a good way to specify organization of output

    # filename is {dataset}.{date}.nc

    outdir = os.path.expandvars(conf['predict']['output']['output_dir'])
    os.makedirs(outdir, exist_ok=True)
    if not os.access(outdir, os.W_OK):
        raise PermissionError(f"Output directory not writeable: {outdir}")
    logging.info("Saving downscaled data to {outdir}")


    # Run directory for launch.sh, model.yml, logs, etc.
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    if not os.access(outdir, os.W_OK):
        raise PermissionError(f"Run directory not writeable: {save_loc}")


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
