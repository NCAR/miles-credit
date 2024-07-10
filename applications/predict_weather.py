# ---------- #
# System
import gc
import os
import sys
import yaml
import glob
import copy
import logging
import warnings
import functools
import subprocess
import time
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
from collections import defaultdict
from argparse import ArgumentParser


# ---------- #
# Numerics
import datetime
import numpy as np
import pandas as pd
import xarray as xr

# ---------- #
# AI libs
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
# import wandb

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)

# ---------- #
# credit
from credit.data import PredictForecastRollout
from credit.loss import VariableTotalLoss2D
from credit.models import load_model
from credit.models.crossformer_may1 import CrossFormer
from credit.metrics import LatWeightedMetrics
from credit.transforms import ToTensor, NormalizeState, NormalizeState_Quantile
from credit.seed import seed_everything
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.forecast import load_forecasts
# from credit.trainer import TOADataLoader

from credit.models.checkpoint import (
    TorchFSDPModel,
    TorchFSDPCheckpointIO
)
from credit.mixed_precision import parse_dtype

# ---------- #
from credit.visualization_tools import shared_mem_draw_wrapper
# from visualization_tools import shared_mem_draw_wrapper

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class TOADataLoader:
    def __init__(self, conf):
        self.TOA = xr.open_dataset(conf["data"]["TOA_forcing_path"]).load()
        self.times_b = pd.to_datetime(self.TOA.time.values)

        # Precompute day of year and hour arrays
        self.days_of_year = self.times_b.dayofyear
        self.hours_of_day = self.times_b.hour

    def __call__(self, datetime_input):
        doy = datetime_input.dayofyear
        hod = datetime_input.hour

        # Use vectorized comparison for masking
        mask_toa = (self.days_of_year == doy) & (self.hours_of_day == hod)
        selected_tsi = self.TOA['tsi'].sel(time=mask_toa) / 2540585.74

        # Convert to tensor and add dimension
        return torch.tensor(selected_tsi.to_numpy()).unsqueeze(0).float()

def get_num_cpus():
    num_cpus = len(os.sched_getaffinity(0))
    return int(num_cpus)


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def split_and_reshape(tensor, conf):
    """
    Split the output tensor of the model to upper air variables and diagnostics/surface variables.

    tensor size: (variables, latitude, longitude)
    Upperair level arrangement: top-of-atmosphere--> near-surface --> single layer
    An example: U (top-of-atmosphere) --> U (near-surface) --> V (top-of-atmosphere) --> V (near-surface)
    """

    # get the number of levels
    levels = conf["model"]["levels"]

    # get number of channels
    channels = len(conf["data"]["variables"])
    single_level_channels = len(conf["data"]["surface_variables"])

    # subset upper air variables
    tensor_upper_air = tensor[:, : int(channels * levels), :, :]

    shape_upper_air = tensor_upper_air.shape
    tensor_upper_air = tensor_upper_air.view(
        shape_upper_air[0], channels, levels, shape_upper_air[-2], shape_upper_air[-1]
    )

    # subset surface variables
    tensor_single_level = tensor[:, -int(single_level_channels):, :, :]

    # return x, surf for B, c, lat, lon output
    return tensor_upper_air, tensor_single_level


def make_xarray(pred, forecast_datetime, lat, lon, conf):

    # subset upper air and surface variables
    tensor_upper_air, tensor_single_level = split_and_reshape(pred, conf)

    # save upper air variables
    darray_upper_air = xr.DataArray(
        tensor_upper_air,
        dims=["datetime", "vars", "level", "lat", "lon"],
        coords=dict(
            vars=conf["data"]["variables"],
            datetime=[forecast_datetime],
            level=range(conf["model"]["levels"]),
            lat=lat,
            lon=lon,
        ),
    )

    # save diagnostics and surface variables
    darray_single_level = xr.DataArray(
        tensor_single_level.squeeze(2),
        dims=["datetime", "vars", "lat", "lon"],
        coords=dict(
            vars=conf["data"]["surface_variables"],
            datetime=[forecast_datetime],
            lat=lat,
            lon=lon,
        ),
    )

    # return x-arrays as outputs
    return darray_upper_air, darray_single_level


def save_netcdf(list_darray_upper_air, list_darray_single_level, conf):
    """
    Save netCDF files from x-array inputs
    """
    # concat full upper air variables from a list of x-arrays
    darray_upper_air_merge = xr.concat(list_darray_upper_air, dim="datetime")

    # concat full single level variables from a list of x-arrays
    darray_single_level_merge = xr.concat(list_darray_single_level, dim="datetime")

    # produce datetime string
    init_datetime_str = np.datetime_as_string(
        darray_upper_air_merge.datetime[0], unit="h", timezone="UTC"
    )

    # create save directory for xarrays
    save_location = os.path.join(os.path.expandvars(conf["save_loc"]), "weather", "netcdf")
    os.makedirs(save_location, exist_ok=True)

    nc_filename_all = os.path.join(
        save_location, f"pred_{init_datetime_str}.nc"
    )
    ds_x = darray_upper_air_merge.to_dataset(dim="vars")
    ds_surf = darray_single_level_merge.to_dataset(dim="vars")
    ds = xr.merge([ds_x, ds_surf])
    ds.to_netcdf(
        path=nc_filename_all,
        format="NETCDF4",
        engine="netcdf4",
        encoding={variable: {"zlib": True, "complevel": 1} for variable in ds.data_vars}
    )
    logger.info(
        f"wrote .nc file for prediction: \n{nc_filename_all}"
    )

    # return saved file names
    return nc_filename_all


def make_video(video_name_prefix, save_location, image_file_names, format="gif"):
    """
    make videos based on images. MP4 format requires ffmpeg.
    """
    output_name = "{}.{}".format(video_name_prefix, format)

    # send all png files to the gif maker
    if format == "gif":
        command_str = f'convert -delay 20 -loop 0 {" ".join(image_file_names)} {save_location}/{output_name}'
        out = subprocess.Popen(
            command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()
    elif format == "mp4":
        # write "input.txt" to summarize input images and frame settings
        input_txt = os.path.join(save_location, f"input_{video_name_prefix}.txt")
        f = open(input_txt, "w")
        for i_file, filename in enumerate(image_file_names):
            print("file {}\nduration 1".format(os.path.basename(filename)), file=f)
        f.close()

        # cd to the save_location and run ffmpeg
        cmd_cd = "cd {}; ".format(save_location)
        cmd_ffmpeg = f'ffmpeg -y -f concat -i input_{video_name_prefix}.txt -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 1 -pix_fmt yuv420p {output_name}'
        command_str = cmd_cd + cmd_ffmpeg
        out, err = subprocess.Popen(
            command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()
        if err:
            logger.info(f"making movie with\n{command_str}\n")
            logger.info(f"The process raised an error:{err.decode()}")
        else:
            logger.info(f"--No errors--\n{out.decode()}")
    else:
        logger.info("Video format not supported")
        raise


def create_shared_mem(da, smm):
    da_bytes = da.to_netcdf()
    da_mem = memoryview(da_bytes)
    shm = smm.SharedMemory(da_mem.nbytes)
    shm.buf[:] = da_mem
    return shm


def distributed_model_wrapper(conf, neural_network, device):

    if conf["trainer"]["mode"] == "fsdp":

        # Define the sharding policies

        if "crossformer" in conf["model"]["type"]:
            from credit.models.crossformer import Attention as Attend
        elif "fuxi" in conf["model"]["type"]:
            from credit.models.fuxi import UTransformer as Attend
        else:
            raise OSError("You asked for FSDP but only crossformer and fuxi are currently supported.")

        auto_wrap_policy1 = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Attend}
        )

        auto_wrap_policy2 = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1_000
        )

        def combined_auto_wrap_policy(module, recurse, nonwrapped_numel):
            # Define a new policy that combines policies
            p1 = auto_wrap_policy1(module, recurse, nonwrapped_numel)
            p2 = auto_wrap_policy2(module, recurse, nonwrapped_numel)
            return p1 or p2

        # Mixed precision

        use_mixed_precision = conf["trainer"]["use_mixed_precision"] if "use_mixed_precision" in conf["trainer"] else False

        logging.info(f"Using mixed_precision: {use_mixed_precision}")

        if use_mixed_precision:
            for key, val in conf["trainer"]["mixed_precision"].items():
                conf["trainer"]["mixed_precision"][key] = parse_dtype(val)
            mixed_precision_policy = MixedPrecision(**conf["trainer"]["mixed_precision"])
        else:
            mixed_precision_policy = None

        # CPU offloading

        cpu_offload = conf["trainer"]["cpu_offload"] if "cpu_offload" in conf["trainer"] else False

        logging.info(f"Using CPU offloading: {cpu_offload}")

        # FSDP module

        model = TorchFSDPModel(
            neural_network,
            use_orig_params=True,
            auto_wrap_policy=combined_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=cpu_offload)
        )

        # activation checkpointing on the transformer blocks

        activation_checkpoint = conf["trainer"]["activation_checkpoint"] if "activation_checkpoint" in conf["trainer"] else False

        logging.info(f"Activation checkpointing: {activation_checkpoint}")

        if activation_checkpoint:

            # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/

            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

            check_fn = lambda submodule: isinstance(submodule, Attend)

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn
            )

    elif conf["trainer"]["mode"] == "ddp":
        model = DDP(neural_network, device_ids=[device])
    else:
        model = neural_network

    return model


def load_model_state(conf, model, device):
    save_loc = os.path.expandvars(conf['save_loc'])
    #  Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    ckpt = os.path.join(save_loc, "checkpoint.pt")
    checkpoint = torch.load(ckpt, map_location=device)
    if conf["trainer"]["mode"] == "fsdp":
        logging.info(f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
        checkpoint_io = TorchFSDPCheckpointIO()
        checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
    else:
        if conf["trainer"]["mode"] == "ddp":
            logging.info(f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            logging.info(f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict(rank, world_size, conf, pool, smm):

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    # Config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    history_len = conf["data"]["history_len"]
    forecast_len = conf["data"]["forecast_len"]
    skip_periods = conf["data"]["forecast_len"]
    time_step = conf["data"]["time_step"] if "time_step" in conf["data"] else None

    # Load paths to all ERA5 data available
    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))

    # Preprocessing transformations
    if conf["data"]["scaler_type"] == "std":
        state_transformer = NormalizeState(conf)
    else:
        state_transformer = NormalizeState_Quantile(conf)
    transform = transforms.Compose(
        [
            state_transformer,
            ToTensor(conf),
        ]
    )

    dataset = PredictForecastRollout(
        filenames=all_ERA_files,
        forecasts=load_forecasts(conf),
        history_len=history_len,
        forecast_len=forecast_len,
        skip_periods=time_step,
        transform=transform,
        rank=rank,
        world_size=world_size,
        shuffle=False,
    )

    # setup the dataloder for this process
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # load model
    model = load_model(conf, load_weights=True).to(device)

    # Warning -- see next line
    distributed = conf["trainer"]["mode"] in ["ddp", "fsdp"]
    if distributed:  # A new field needs to be added to predict
        model = distributed_model_wrapper(conf, model, device)
        if conf["trainer"]["mode"] == "fsdp":
            # Load model weights (if any), an optimizer, scheduler, and gradient scaler
            model = load_model_state(conf, model, device)

    model.eval()

    # Set up metrics and containers
    metrics = LatWeightedMetrics(conf)
    metrics_results = defaultdict(list)
    loss_fn = VariableTotalLoss2D(conf, validation=True)

    # get lat/lons from x-array
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])

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

        # lists to collect x-arrays
        list_darray_upper_air = []
        list_darray_single_level = []

        # a list that collects image file names
        job_info = []
        filenames_upper_air = []
        filenames_diagnostics = []
        filenames_surface = []

        # Total # of figures
        N_vars = len(
            conf["visualization"]["sigma_level_visualize"]["variable_keys"]
        )
        N_vars += len(
            conf["visualization"]["diagnostic_variable_visualize"]["variable_keys"]
        )
        N_vars += len(
            conf["visualization"]["surface_visualize"]["variable_keys"]
        )

        # y_pred allocation
        y_pred = None
        static = None
        loop_time = time.time()
        
        # model inference loops
        k = 0
        time_step = 1 # this will have to change (hr)
        start_date = conf["predict"]["forecasts"][0][0]
        end_date =conf["predict"]["forecasts"][0][1]
        next_fore = 0
        print('Forecasts!!!!:',conf["predict"]["forecasts"])

        # First loop is context setting
        for batch in data_loader:
            print('next fore is:', next_fore)
            end_date =conf["predict"]["forecasts"][next_fore][1]
            end_date = pd.to_datetime(end_date)
            start_time = time.time()

            # get the datetime and forecasted hours
            date_time = batch["datetime"].item()
            print('first datetime:', date_time)
            forecast_hour = batch["forecast_hour"].item()

            # initialization on the first forecast hour
            if forecast_hour == 1:
                # Initialize x and x_surf with the first time step
                x = model.concat_and_reshape(batch["x"], batch["x_surf"]).to(device)

                # setup save directory for images
                init_time = datetime.datetime.utcfromtimestamp(date_time).strftime(
                    "%Y-%m-%dT%HZ"
                )
                print('init time!:', init_time)
                img_save_loc = os.path.join(
                    os.path.expandvars(conf["save_loc"]),
                    f"forecasts/images_{init_time}",
                )
                if N_vars > 0:
                    os.makedirs(img_save_loc, exist_ok=True)

            # Add statics
            if "static" in batch:
                if static is None:
                    static = batch["static"].to(device).unsqueeze(2).expand(-1, -1, x.shape[2], -1, -1).float()
                x = torch.cat((x, static.clone()), dim=1)

            # Add solar "statics"
            if "static_variables" in conf["data"] and "tsi" in conf["data"]["static_variables"]:
                if k==0:
                    toaDL = TOADataLoader(conf)
                elapsed_time = pd.Timedelta(hours=k)
                tnow = pd.to_datetime(datetime.datetime.utcfromtimestamp(batch["datetime"]))
                tnow = tnow + elapsed_time
                if history_len == 1:
                    current_times = [pd.to_datetime(datetime.datetime.utcfromtimestamp(_t)) + elapsed_time for _t in tnow]
                else:
                    current_times = [tnow if hl == 0 else tnow - pd.Timedelta(hours=hl) for hl in range(history_len)]
                
                toa = torch.cat([toaDL(_t) for _t in current_times], dim=0).to(device)
                toa = toa.squeeze().unsqueeze(0)
                #print(f"toa shape 1: {toa.shape} || ")
                print_str2 = f"toa shape 2: {toa.unsqueeze(1).shape} || "
                x = torch.cat([x, toa.unsqueeze(1).to(device).float()], dim=1)
                k += 1
            
            y_pred = model(x)

            end_time = time.time()
            duration = end_time - start_time
            print_str2 += f"time to make pred: {duration:.4f} ||"

            # convert to real space for laplace filter and metrics
            y_pred = state_transformer.inverse_transform(y_pred.cpu())

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

            utc_datetime = datetime.datetime.utcfromtimestamp(date_time)
            print_str = f"Forecast: {forecast_count} "
            print_str += f"Date: {utc_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
            print_str += f"Hour: {batch['forecast_hour'].item()} "

            print_str += f"ypred shape: {y_pred.shape} "
            print_str += f"ypred dtype: {y_pred.dtype} "
            logger.info(print_str)
            # convert the current step result as x-array
            darray_upper_air, darray_single_level = make_xarray(
                y_pred,
                utc_datetime,
                latlons.latitude.values,
                latlons.longitude.values,
                conf,
            )

            # collect x-arrays for upper air and surface variables
            list_darray_upper_air.append(darray_upper_air)
            list_darray_single_level.append(darray_single_level)

            end_time = time.time()
            duration = end_time - start_time
            print_str2 += f"time to make xarray: {duration:.4f} ||"

            # ---------------------------------------------------------------------------------- #
            # Draw upper air variables

            # get the number of variables to draw
            N_vars = len(
                conf["visualization"]["sigma_level_visualize"]["variable_keys"]
            )

            if N_vars > 0:
                # get the required model levels to plot
                sigma_levels = conf["visualization"]["sigma_level_visualize"][
                    "visualize_levels"
                ]

                f = partial(
                    shared_mem_draw_wrapper,
                    visualization_key="sigma_level_visualize",
                    step=forecast_hour,
                    conf=conf,
                    save_location=img_save_loc,
                )

                # slice x-array on its time dimension to get rid of time dim
                darray_upper_air_slice = darray_upper_air.isel(datetime=0)
                shm_upper_air = create_shared_mem(darray_upper_air_slice, smm)
                # produce images
                job_result = pool.starmap_async(
                    f, [(shm_upper_air, lvl) for lvl in sigma_levels]
                )
                job_info.append(job_result)
                filenames_upper_air.append(
                    job_result
                )  # .get() blocks computation. need to get after the pool closes

            # ---------------------------------------------------------------------------------- #
            # Draw diagnostics

            # get the number of variables to draw
            N_vars = len(
                conf["visualization"]["diagnostic_variable_visualize"]["variable_keys"]
            )
            # slice x-array on its time dimension to get rid of time dim
            darray_single_level_slice = darray_single_level.isel(datetime=0)
            shm_single_level = create_shared_mem(darray_single_level_slice, smm)
            if N_vars > 0:
                f = partial(
                    shared_mem_draw_wrapper,
                    level=-1,
                    visualization_key="diagnostic_variable_visualize",
                    step=forecast_hour,
                    conf=conf,
                    save_location=img_save_loc,
                )
                # produce images
                job_result = pool.map_async(
                    f,
                    [
                        shm_single_level,
                    ],
                )
                job_info.append(job_result)
                filenames_diagnostics.append(job_result)
            # ---------------------------------------------------------------------------------- #
            # Draw surface variables
            N_vars = len(conf["visualization"]["surface_visualize"]["variable_keys"])
            if N_vars > 0:
                f = partial(
                    shared_mem_draw_wrapper,
                    level=-1,
                    visualization_key="surface_visualize",
                    step=forecast_hour,
                    conf=conf,
                    save_location=img_save_loc,
                )

                # produce images
                job_result = pool.map_async(
                    f,
                    [
                        shm_single_level,
                    ],
                )
                job_info.append(job_result)
                filenames_surface.append(job_result)

            # Update the input
            # setup for next iteration, transform to z-space and send to device

            y_pred = state_transformer.transform_array(y_pred).to(device)

            if history_len == 1:
                x = y_pred.detach()
            else:
                # use multiple past forecast steps as inputs
                static_dim_size = abs(x.shape[1] - y_pred.shape[1])  # static channels will get updated on next pass
                x_detach = x[:, :-static_dim_size, 1:].detach()
                x = torch.cat([x_detach, y_pred.detach()], dim=2)

            end_time = time.time()
            duration = end_time - start_time
            print_str2 += f"time to cycle to next x input: {duration:.4f} ||"

            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            end_time = time.time()
            duration = end_time - start_time
            print_str2 += f"TOT time: {duration:.4f} ||"


            end_time = time.time()
            duration = end_time - loop_time
            print_str2 += f"LOOP time: {duration:.4f} ||"

            logger.info(print_str2)        

            #rollout loop, which is much faster. 
            end_condition = 0
            # end_date = pd.to_datetime(end_date)
            print('we are looking for end_date:', end_date)
            while end_condition < 1:
                print('k is :', k)
                
                #deal with the time step:
                elapsed_time = pd.Timedelta(hours=k)
                tnow = pd.to_datetime(datetime.datetime.utcfromtimestamp(batch["datetime"]))
                tnow = tnow + elapsed_time
                # setup save directory for images
                file_save_name_time = tnow.strftime(
                        "%Y-%m-%dT%HZ"
                    )
                print('time now is:', tnow, file_save_name_time)
                start_time = time.time()
    
                # Add statics
                if "static" in batch:
                    if static is None:
                        static = batch["static"].to(device).unsqueeze(2).expand(-1, -1, x.shape[2], -1, -1).float()
                    x = torch.cat((x, static.clone()), dim=1)
    
                # Add solar "statics"
                if "static_variables" in conf["data"] and "tsi" in conf["data"]["static_variables"]:
                    if k==0:
                        toaDL = TOADataLoader(conf)
                    elapsed_time = pd.Timedelta(hours=k)
                    tnow = pd.to_datetime(datetime.datetime.utcfromtimestamp(batch["datetime"]))
                    tnow = tnow + elapsed_time
                    if history_len == 1:
                        current_times = [pd.to_datetime(datetime.datetime.utcfromtimestamp(_t)) + elapsed_time for _t in tnow]
                    else:
                        current_times = [tnow if hl == 0 else tnow - pd.Timedelta(hours=hl) for hl in range(history_len)]
                    
                    toa = torch.cat([toaDL(_t) for _t in current_times], dim=0).to(device)
                    toa = toa.squeeze().unsqueeze(0)
                    #print(f"toa shape 1: {toa.shape} || ")
                    print_str2 = f"toa shape 2: {toa.unsqueeze(1).shape} || "
                    x = torch.cat([x, toa.unsqueeze(1).to(device).float()], dim=1)
                    k += 1
                
                y_pred = model(x)
    
                end_time = time.time()
                duration = end_time - start_time
                print_str2 += f"time to make pred: {duration:.4f} ||"
    
                # convert to real space for laplace filter and metrics
                y_pred = state_transformer.inverse_transform(y_pred.cpu())
    
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
    
                utc_datetime = tnow
                print_str = f"Forecast: {forecast_count} "
                print_str += f"Date: {utc_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
                print_str += f"Hour: {batch['forecast_hour'].item()} "
                logger.info(print_str)
                # convert the current step result as x-array
                darray_upper_air, darray_single_level = make_xarray(
                    y_pred,
                    utc_datetime,
                    latlons.latitude.values,
                    latlons.longitude.values,
                    conf,
                )
    
                # collect x-arrays for upper air and surface variables
                list_darray_upper_air.append(darray_upper_air)
                list_darray_single_level.append(darray_single_level)
    
                end_time = time.time()
                duration = end_time - start_time
                print_str2 += f"time to make xarray: {duration:.4f} ||"
    
                # ---------------------------------------------------------------------------------- #
                # Draw upper air variables
    
                # get the number of variables to draw
                N_vars = len(
                    conf["visualization"]["sigma_level_visualize"]["variable_keys"]
                )
    
                if N_vars > 0:
                    # get the required model levels to plot
                    sigma_levels = conf["visualization"]["sigma_level_visualize"][
                        "visualize_levels"
                    ]
    
                    f = partial(
                        shared_mem_draw_wrapper,
                        visualization_key="sigma_level_visualize",
                        step=forecast_hour,
                        conf=conf,
                        save_location=img_save_loc,
                    )
    
                    # slice x-array on its time dimension to get rid of time dim
                    darray_upper_air_slice = darray_upper_air.isel(datetime=0)
                    shm_upper_air = create_shared_mem(darray_upper_air_slice, smm)
                    # produce images
                    job_result = pool.starmap_async(
                        f, [(shm_upper_air, lvl) for lvl in sigma_levels]
                    )
                    job_info.append(job_result)
                    filenames_upper_air.append(
                        job_result
                    )  # .get() blocks computation. need to get after the pool closes
    
                # ---------------------------------------------------------------------------------- #
                # Draw diagnostics
    
                # get the number of variables to draw
                N_vars = len(
                    conf["visualization"]["diagnostic_variable_visualize"]["variable_keys"]
                )
                # slice x-array on its time dimension to get rid of time dim
                darray_single_level_slice = darray_single_level.isel(datetime=0)
                shm_single_level = create_shared_mem(darray_single_level_slice, smm)
                if N_vars > 0:
                    f = partial(
                        shared_mem_draw_wrapper,
                        level=-1,
                        visualization_key="diagnostic_variable_visualize",
                        step=forecast_hour,
                        conf=conf,
                        save_location=img_save_loc,
                    )
                    # produce images
                    job_result = pool.map_async(
                        f,
                        [
                            shm_single_level,
                        ],
                    )
                    job_info.append(job_result)
                    filenames_diagnostics.append(job_result)
                # ---------------------------------------------------------------------------------- #
                # Draw surface variables
                N_vars = len(conf["visualization"]["surface_visualize"]["variable_keys"])
                if N_vars > 0:
                    f = partial(
                        shared_mem_draw_wrapper,
                        level=-1,
                        visualization_key="surface_visualize",
                        step=forecast_hour,
                        conf=conf,
                        save_location=img_save_loc,
                    )
    
                    # produce images
                    job_result = pool.map_async(
                        f,
                        [
                            shm_single_level,
                        ],
                    )
                    job_info.append(job_result)
                    filenames_surface.append(job_result)
    
                # Update the input
                # setup for next iteration, transform to z-space and send to device
    
                y_pred = state_transformer.transform_array(y_pred).to(device)
    
                if history_len == 1:
                    x = y_pred.detach()
                else:
                    # use multiple past forecast steps as inputs
                    static_dim_size = abs(x.shape[1] - y_pred.shape[1])  # static channels will get updated on next pass
                    x_detach = x[:, :-static_dim_size, 1:].detach()
                    x = torch.cat([x_detach, y_pred.detach()], dim=2)
    
                end_time = time.time()
                duration = end_time - start_time
                print_str2 += f"time to cycle to next x input: {duration:.4f} ||"
    
                # Explicitly release GPU memory
                torch.cuda.empty_cache()
                gc.collect()
    
                end_time = time.time()
                duration = end_time - start_time
                print_str2 += f"TOT time: {duration:.4f} ||"
    
    
                end_time = time.time()
                duration = end_time - loop_time
                print_str2 += f"LOOP time: {duration:.4f} ||"
    
                #logger.info(print_str2)
               
                if tnow > end_date:
                    end_condition = 9999
    
                if end_condition  == 9999:
                    # convert to datasets and save out
                    # save forecast results to file
                    if "save_format" in conf["predict"] and conf["predict"]["save_format"] == "nc":
                        logger.info("Save forecasts as netCDF format")
    
                        logger.info("Save forecasts as netCDF format")
                        filename_netcdf = save_netcdf(
                            list_darray_upper_air, list_darray_single_level, conf
                        )
                        
                        # xr.merge(list_datasets).to_netcdf(
                        #             path=os.path.join(dataset_save_loc, f"pred_{init_time}-{utc_datetime.strftime('%Y-%m-%d %H:%M:%S')}.nc"),
                        #             format="NETCDF4",
                        #             engine="netcdf4",
                        #             encoding={variable: {"zlib": True, "complevel": 1} for variable in pred_ds.data_vars}
                        #     )
                    else:
                        logger.info("Warning: forecast results will not be saved")
    
                    # forecast count = a constant for each run
                    forecast_count += 1
    
                    # lists to collect x-arrays
                    list_darray_upper_air = []
                    list_darray_single_level = []
    
                    # a list that collects image file names
                    job_info = []
                    filenames_upper_air = []
                    filenames_diagnostics = []
                    filenames_surface = []
    
                    # y_pred allocation
                    y_pred = None
    
                    # Set up metrics and containers
                    metrics = LatWeightedMetrics(conf)
                    metrics_results = defaultdict(list)
    
                    gc.collect()
    
                    if distributed:
                        torch.distributed.barrier()

            batch["stop_forecast"][0] = True
            forecast_count += 1
            next_fore += 1
            k = 0
            continue
            
    
    # y_pred allocation
    y_pred = None
            
    gc.collect()
    if distributed:
        torch.distributed.barrier()
            
    # collect all image file names for making videos
    filename_bundle = {}
    filename_bundle["sigma_level_visualize"] = filenames_upper_air
    filename_bundle["diagnostic_variable_visualize"] = filenames_diagnostics
    filename_bundle["surface_visualize"] = filenames_surface

    if distributed:
        torch.distributed.barrier()

    return (
        list_darray_upper_air,
        list_darray_single_level,
        job_info,
        img_save_loc,
        filename_bundle,
    )


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

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    no_data = 0 if "no-data" not in args_dict else int(args_dict.pop("no-data"))

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

    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logger.info(f"Setting the running mode to {mode}")
        conf["trainer"]["mode"] = mode

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

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    num_cpus = get_num_cpus()
    logger.info(f"using {num_cpus} cpus for image generation")
    
    
    
    num_forecasts = conf
    
    
    
    with Pool(processes=num_cpus - 1) as pool, SharedMemoryManager() as smm:
        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            (
                list_darray_upper_air,
                list_darray_single_level,
                job_info,
                img_save_loc,
                filename_bundle,
            ) = predict(
                int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf, pool, smm
            )
        else:
            (
                list_darray_upper_air,
                list_darray_single_level,
                job_info,
                img_save_loc,
                filename_bundle,
            ) = predict(0, 1, conf, pool, smm)

        pool.close()
        pool.join()
    # exit the context before making videos

    if no_data:
        # set the fields in the config file to prevent any movies from being made
        for movie_option in ["sigma_level_visualize", "diagnostic_variable_visualize", "surface_visualize"]:
            conf["visualization"][movie_option] = []

    # ---------------------------------------------------------------------------------- #
    # Making videos need to get() after pool closes otherwise .get blocks computation
    filenames_upper_air = [
        res.get() for res in filename_bundle["sigma_level_visualize"]
    ]
    filenames_diagnostics = [
        res.get()[0] for res in filename_bundle["diagnostic_variable_visualize"]
    ]
    filenames_surface = [res.get()[0] for res in filename_bundle["surface_visualize"]]

    video_format = conf["visualization"]["video_format"]

    # more than one image --> making video for upper air variables
    if len(filenames_upper_air) > 1 and video_format in ["gif", "mp4"]:
        logger.info("Making video for upper air variables")

        # get the required model levels to plot
        sigma_levels = conf["visualization"]["sigma_level_visualize"][
            "visualize_levels"
        ]
        N_levels = len(sigma_levels)

        for i_level, level in enumerate(sigma_levels):
            # add level info into the video file name
            video_name_prefix = conf["visualization"]["sigma_level_visualize"][
                "file_name_prefix"
            ]
            video_name_prefix += "_level{:02d}".format(level)

            # get current level files
            filename_current_level = [
                files_t[i_level] for files_t in filenames_upper_air
            ]

            # make video
            make_video(
                video_name_prefix,
                img_save_loc,
                filename_current_level,
                format=video_format,
            )
    else:
        logger.info("Skipping video production for upper air variables")

    # more than one image --> making video for diagnostics
    if len(filenames_diagnostics) > 1 and video_format in ["gif", "mp4"]:
        logger.info("Making video for diagnostic variables")

        # get file names
        video_name_prefix = conf["visualization"]["diagnostic_variable_visualize"][
            "file_name_prefix"
        ]

        # make video
        make_video(
            video_name_prefix, img_save_loc, filenames_diagnostics, format=video_format
        )
    else:
        logger.info("SKipping video production for diagnostic variables")

    # more than one image --> making video for surface variables
    if len(filenames_surface) > 1 and video_format in ["gif", "mp4"]:
        logger.info("Making video for surface variables")

        # get file names
        video_name_prefix = conf["visualization"]["surface_visualize"][
            "file_name_prefix"
        ]

        # make video
        make_video(
            video_name_prefix, img_save_loc, filenames_surface, format=video_format
        )
    else:
        logger.info("SKipping video production for surface variables")


# # ------------------------------------------------------------------------------------------ #
# # Debugging function
# def make_images_from_xarray(nc_filename_upper_air, nc_filename_single_level, conf):
#     '''
#     Produce images from x-array inputs
#     '''
#     # import upper air variables
#     darray_upper_air = xr.load_dataarray(nc_filename_upper_air)

#     # import surface variables
#     darray_single_level = xr.load_dataarray(nc_filename_single_level)


#     # Create directories to save images, overwrite files if already exists,
#     # filenames have uniquely id

#     ## create image folder based on the first forecasted time
#     init_time = np.datetime_as_string(darray_upper_air.datetime[0], unit='h', timezone='UTC')
#     save_loc = os.path.join(os.path.expandvars(conf["save_loc"]), f'forecasts/images_{init_time}')
#     os.makedirs(save_loc, exist_ok=True)

#     # get the required model levels to plot
#     sigma_levels = conf['visualization']['sigma_level_visualize']['visualize_levels']

#     # todo: parallelize over times
#     for level in sigma_levels:
#         datetimes = darray_upper_air.datetime.to_numpy()
#         with Pool(processes=8) as pool:
#             f = partial(draw_sigma_level, conf=conf, save_location=save_loc)
#             da_level = darray_upper_air.sel(level=level)
#             pool.map(f, [da_level.sel(datetime=dt) for dt in datetimes])

#     return save_loc
# # ------------------------------------------------------------------------------------------ #

# def make_movie(filenames, conf, save_location): #level, datetime
#     '''
#     Make movies based on produced images
#     '''
#     # get the required model levels to plot
#     sigma_levels = conf['visualization']['sigma_level_visualize']['visualize_levels']

#     # produce videos on each required upper air level
#     for level_idx, sigma_level in enumerate(sigma_levels):
#         level_image_filenames = [filename_list[level_idx] for filename_list in filenames]

#         ## send all png files to the gif maker
#         gif_name = '{}_level{:02d}.gif'.format(gif_name_prefix, sigma_level)
#         command_str = f'convert -delay 20 -loop 0 {" ".join(level_image_filenames)} {save_location}/{gif_name}'
#         out = subprocess.Popen(command_str, shell=True,
#                                 stdout=subprocess.PIPE,
#                                 stderr=subprocess.PIPE).communicate()
#         print(out)
