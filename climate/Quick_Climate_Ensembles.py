import os
import gc
import sys
from sklearn import ensemble
import yaml
import time
import logging
import warnings
import copy
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from collections import defaultdict
import cftime
from cftime import DatetimeNoLeap
import json
import pickle
import argparse


# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# ---------- #
import torch
from torch.utils.data import get_worker_info
from torch.profiler import profile, record_function, ProfilerActivity


# ---------- #
# credit
from credit.models import load_model
from credit.seed import seed_everything

from credit.data import (
    concat_and_reshape,
    reshape_only,
    drop_var_from_dataset,
    generate_datetime,
    nanoseconds_to_year,
    hour_to_nanoseconds,
    get_forward_data,
    extract_month_day_hour,
    find_common_indices,
)

from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.metrics import LatWeightedMetrics
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state
from credit.parser import credit_main_parser, predict_data_check
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def save_task(data, meta_data, conf):
    """Wrapper function for saving data in parallel."""
    darray_upper_air, darray_single_level, init_datetime_str, lead_time, forecast_hour = data
    save_netcdf_increment(
        darray_upper_air,
        darray_single_level,
        init_datetime_str,
        lead_time * forecast_hour,
        meta_data,
        conf,
    )


def run_year_rmse(p, config, input_shape, forcing_shape, output_shape, device, model_name=None, init_noise=None):
    """
    Function to compute RMSE for a year-long climate model prediction.

    Parameters:
    - config: str
        Path to the YAML configuration file.
    - input_shape: tuple
        Shape of the input tensor to the model.
    - forcing_shape: tuple
        Shape of the forcing tensor for the model.
    - output_shape: tuple
        Shape of the output tensor from the model.
    - device: torch.device
        Device to run the model on (CPU/GPU).
    - model_name: str, optional
        Name of the model to load, if specified.

    The function handles:
    - Loading configurations, model, and transforms
    - Setting up data pre-processing steps
    - Handling dynamic/static variables and conservation constraints
    - Preparing input/output tensors for prediction and RMSE computation
    - Tracing the model for optimized execution
    - running for a specified amount of time and saving mean
    """
    # Load configuration from the YAML file
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Parse and preprocess the configuration for prediction
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    conf["predict"]["mode"] = None

    # Extract the history length for the data
    history_len = conf["data"]["history_len"]

    # Load transformation utilities and scalers
    transform = load_transforms(conf)
    if conf["data"]["scaler_type"] == "std_new":
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        print("Scaler type {} not supported".format(conf["data"]["scaler_type"]))
        raise

    # Load the model (optionally a custom model) and configure distributed mode
    if model_name is not None:
        print("loading custom: ", model_name)
        model = load_model(conf, model_name=model_name, load_weights=True).to(device)
    else:
        model = load_model(conf, load_weights=True).to(device)

    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        model = distributed_model_wrapper(conf, model, device)
        if conf["predict"]["mode"] == "fsdp":
            model = load_model_state(conf, model, device)

    model.eval()
    post_conf = conf["model"]["post_conf"]

    # number of dynamic forcing + forcing + static
    static_dim_size = (
        len(conf["data"]["dynamic_forcing_variables"])
        + len(conf["data"]["forcing_variables"])
        + len(conf["data"]["static_variables"])
    )

    # Extract conservation flags from the configuration
    flag_mass_conserve, flag_water_conserve, flag_energy_conserve = False, False, False
    if post_conf["activate"]:
        if post_conf["global_mass_fixer"]["activate"] and post_conf["global_mass_fixer"]["activate_outside_model"]:
            flag_mass_conserve = True
            opt_mass = GlobalMassFixer(post_conf)
        if post_conf["global_water_fixer"]["activate"] and post_conf["global_water_fixer"]["activate_outside_model"]:
            flag_water_conserve = True
            opt_water = GlobalWaterFixer(post_conf)
        if post_conf["global_energy_fixer"]["activate"] and post_conf["global_energy_fixer"]["activate_outside_model"]:
            flag_energy_conserve = True
            opt_energy = GlobalEnergyFixer(post_conf)

    # Extract variable names from configuration
    df_variables = conf["data"]["dynamic_forcing_variables"]
    sf_variables = conf["data"]["static_variables"]
    varnum_diag = len(conf["data"]["diagnostic_variables"])
    lead_time_periods = conf["data"]["lead_time_periods"]

    # Load initial condition and forcing dataset
    x = torch.load(conf["predict"]["init_cond_fast_climate"], map_location=torch.device(device)).to(device)
    DSforc = xr.open_dataset(conf["predict"]["forcing_file"])

    ensemble_size = conf["predict"].get("ensemble_size", 1)
    if ensemble_size > 1:
        logger.info(f"rolling out with ensemble size {ensemble_size}")
        x = torch.repeat_interleave(x, ensemble_size, 0)
        input_shape[0] = ensemble_size
        forcing_shape[0] = ensemble_size  # dont do for forcing shape
        output_shape[0] = ensemble_size

    # Set up metrics and transformations
    metrics = LatWeightedMetrics(conf)
    DSforc_norm = state_transformer.transform_dataset(DSforc)

    # Extract indices for RMSE computation and load the truth field
    inds_to_rmse = conf["predict"]["inds_rmse_fast_climate"]
    truth_field = torch.load(conf["predict"]["seasonal_mean_fast_climate"], map_location=torch.device(device))
    print(f'torch loaded truth: {conf["predict"]["seasonal_mean_fast_climate"]}')

    # Load metadata and set up static and dynamic forcing variables
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])
    meta_data = load_metadata(conf)
    num_ts = conf["predict"]["timesteps_fast_climate"]

    if conf["data"]["static_first"]:
        df_sf_variables = conf["data"]["static_variables"] + conf["data"]["dynamic_forcing_variables"]
    else:
        df_sf_variables = conf["data"]["dynamic_forcing_variables"] + conf["data"]["static_variables"]

    y_diag_present = len(conf["data"]["diagnostic_variables"]) > 0

    DS_forcx_static = DSforc[sf_variables].load()

    # Initialize prediction start date
    conf_pred = conf["predict"]
    conf_pred_for = conf_pred["forecasts"]
    year_ = conf_pred_for.get("start_year", 1970)  # Default to 1970 if missing
    month_ = conf_pred_for.get("start_month", 1)  # Default to January
    day_ = conf_pred_for.get("start_day", 1)  # Default to 1st
    hours_ = conf_pred_for.get("start_hours", [0])  # Default to [0]

    # Ensure hours is a list and take the first value
    hour_ = hours_[0] if isinstance(hours_, list) and hours_ else 0
    init_date_obj = datetime(year_, month_, day_, hour_)

    init_date_obj = DatetimeNoLeap(year_, month_, day_, hour_)
    print("init date obj: ", init_date_obj)

    # Align forcing dataset to the initial date
    nearest_time = DSforc_norm.sel(time=init_date_obj, method="nearest")

    index = DSforc_norm.indexes["time"].get_loc(init_date_obj)
    indx_start = index
    print("index start: ", indx_start)
    init_datetime_str = init_date_obj.strftime("%Y-%m-%d %H:%M:%S")
    print("init_datetime_str: ", init_datetime_str)
    print(f"Index of start time: {indx_start}")

    # Select dynamic forcing data for prediction window
    DS_forcx_dynamic = DSforc_norm[df_variables].isel(time=slice(indx_start, indx_start + num_ts + 10)).load()
    DS_forcx_dynamic_time = DS_forcx_dynamic["time"].values

    if init_noise is not None:
        print("adding forecast noise")
        # Define the standard deviation for the noise (e.g., 0.01)
        noise_std = 0.05

        # Generate random noise tensor with the same shape as `x`
        # Use `torch.randn` for a normal distribution with mean=0 and std=1
        noise = torch.randn_like(x) * noise_std

        # Add the noise to `x`
        x = x + noise.to(device)

    # Initialize RMSE computation parameters
    if inds_to_rmse is None:
        inds_to_rmse = np.arange(0, output_shape[1])

    test_tensor_rmse = torch.zeros(output_shape).to(device)
    x_forcing_batch = torch.zeros(forcing_shape).to(device)

    # Prepare forcing dictionary for input tensors
    forcing_dict = {sfv: torch.tensor(DS_forcx_static[sfv].values).to(device) for sfv in sf_variables}
    forcing_dict.update({dfv: torch.tensor(DS_forcx_dynamic[dfv].values).to(device) for dfv in df_variables})

    forecast_hour = 1
    add_it = 0
    num_steps = 0

    # Trace the model for optimized execution
    # print('jit trace model start')
    # model = torch.jit.trace(model, x)
    # print('jit trace model done')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"Model size: {model_size:.2f} MB")

    for k in range(num_ts):
        loop_start = time.time()
        if (k + 1) % 20 == 0:
            print(f"model step: {k:05}")

        start_forcing = time.time()
        if k != 0:
            if conf["data"]["static_first"]:
                for bb, sfv in enumerate(sf_variables):
                    x_forcing_batch[:, bb, :, :, :] = forcing_dict[sfv]
                for bb, dfv in enumerate(df_variables):
                    x_forcing_batch[:, len(sf_variables) + bb, :, :, :] = forcing_dict[dfv][k, :, :]
            else:
                for bb, dfv in enumerate(df_variables):
                    x_forcing_batch[:, bb, :, :, :] = forcing_dict[dfv][k, :, :]
                for bb, sfv in enumerate(sf_variables):
                    x_forcing_batch[:, len(df_variables) + bb, :, :, :] = forcing_dict[sfv]

            x = torch.cat((x, x_forcing_batch), dim=1)

        if k == 0:
            cftime_obj = DS_forcx_dynamic_time[k]
            init_datetime_str = cftime_obj.strftime("%Y-%m-%dT%HZ")

        cftime_obj = DS_forcx_dynamic_time[k + 1]
        # Convert to string directly using str()
        cftime_str = str(cftime_obj)  # This will look like 'YYYY-MM-DD HH:SS:MM'
        # Parse the string into a standard datetime object
        utc_datetime = datetime.strptime(cftime_str, "%Y-%m-%d %H:%M:%S")

        # print('Forcing time:', time.time() - start_forcing)

        start_model = time.time()
        x = x.contiguous()
        with torch.no_grad():
            y_pred = model(x.float())
        # print('Model inference time:', time.time() - start_model)

        start_postprocess = time.time()
        # test_tensor_rmse.add_(y_pred)

        if flag_mass_conserve:
            if forecast_hour == 1:
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
        # print('Postprocess time:', time.time() - start_postprocess)
        test_tensor_rmse.add_(y_pred)
        upper_air_list, single_level_list = [], []
        for i in range(
            ensemble_size
        ):  # ensemble_size default is 1, will run with i=0 retaining behavior of non-ensemble loop
            darray_upper_air, darray_single_level = make_xarray(
                y_pred[i : i + 1].cpu(),
                utc_datetime,
                latlons.latitude.values,
                latlons.longitude.values,
                conf,
            )
            upper_air_list.append(darray_upper_air)
            single_level_list.append(darray_single_level)

        if ensemble_size > 1:
            ensemble_index = xr.DataArray(np.arange(ensemble_size), dims="ensemble_member_label")
            all_upper_air = xr.concat(upper_air_list, ensemble_index)  # .transpose("time", ...)
            all_single_level = xr.concat(single_level_list, ensemble_index)  # .transpose("time", ...)
        else:
            all_upper_air = darray_upper_air
            all_single_level = darray_single_level

        result = p.apply_async(
            save_netcdf_increment,
            (
                all_upper_air,
                all_single_level,
                init_datetime_str,
                lead_time_periods * forecast_hour,
                meta_data,
                conf,
            ),
        )

        # cuda_empty_start = time.time()
        # torch.cuda.empty_cache()
        # gc.collect()
        add_it += 1
        num_steps += 1
        forecast_hour += 1
        # print('CUDA empty time:', time.time() - cuda_empty_start)
        start_switch = time.time()
        # ============================================================ #
        # use previous step y_pred as the next step input
        if history_len == 1:
            # cut diagnostic vars from y_pred, they are not inputs
            if y_diag_present:
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
            if y_diag_present:
                x = torch.cat([x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2)
            else:
                x = torch.cat([x_detach, y_pred.detach()], dim=2)

        # print('Total loop time:', time.time() - loop_start)
        # ============================================================ #
        # print('Switch time:', time.time() - start_switch)
        loop_time = time.time() - loop_start
        logger.info(f"Total loop time: {loop_time:.2f}")
        # print('add_it: ', add_it)
        # if add_it == 365:
        #     break

    if model_name is not None:
        fsout = f"{os.path.basename(config)}_{model_name}"
    else:
        fsout = os.path.basename(config)

    METS = metrics(test_tensor_rmse.cpu() / add_it, truth_field.cpu())

    # save out: test_tensor_rmse/num_ts, MET, plots
    with open(f'{conf["save_loc"]}/{fsout}_quick_climate_METS.pkl', "wb") as f:
        pickle.dump(METS, f)

    torch.save(test_tensor_rmse.cpu() / add_it, f'{conf["save_loc"]}/{fsout}_quick_climate_avg_tensor.pt')

    fig, axes = plt.subplots(2, 3, figsize=(25, 15))

    p1 = axes[0, 0].pcolor(test_tensor_rmse.squeeze()[-16, :, :].cpu() / add_it, vmin=-2, vmax=2, cmap="RdBu_r")
    fig.colorbar(p1, ax=axes[0, 0])
    axes[0, 0].set_title("Test Tensor RMSE")

    p2 = axes[0, 1].pcolor(truth_field.squeeze()[-16, :, :].cpu(), vmin=-2, vmax=2, cmap="RdBu_r")
    fig.colorbar(p2, ax=axes[0, 1])
    axes[0, 1].set_title("Truth Field")

    p3 = axes[0, 2].pcolor(
        (test_tensor_rmse.squeeze()[-16, :, :].cpu() / add_it) - truth_field.squeeze()[-16, :, :].cpu(),
        vmin=-0.1,
        vmax=0.1,
        cmap="RdBu_r",
    )
    fig.colorbar(p3, ax=axes[0, 2])
    axes[0, 2].set_title("Difference (RMSE - Truth)")

    p1 = axes[1, 0].pcolor(test_tensor_rmse.squeeze()[-15, :, :].cpu() / add_it, vmin=-2, vmax=2, cmap="RdBu_r")
    fig.colorbar(p1, ax=axes[1, 0])
    axes[1, 0].set_title("Test Tensor RMSE")

    p2 = axes[1, 1].pcolor(truth_field.squeeze()[-15, :, :].cpu(), vmin=-2, vmax=2, cmap="RdBu_r")
    fig.colorbar(p2, ax=axes[1, 1])
    axes[1, 1].set_title("Truth Field")

    p3 = axes[1, 2].pcolor(
        (test_tensor_rmse.squeeze()[-15, :, :].cpu() / add_it) - truth_field.squeeze()[-15, :, :].cpu(),
        vmin=-0.5,
        vmax=0.5,
        cmap="RdBu_r",
    )
    fig.colorbar(p3, ax=axes[1, 2])
    axes[1, 2].set_title("Difference (RMSE - Truth)")

    plt.tight_layout()
    plt.savefig(f'{conf["save_loc"]}/{fsout}_quick_climate_plot_slow.png', bbox_inches="tight")
    plt.show()

    return test_tensor_rmse, truth_field, inds_to_rmse, metrics, conf, METS


def main():
    parser = argparse.ArgumentParser(description="Run year RMSE for WxFormer model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    parser.add_argument("--input_shape", type=int, nargs="+", required=True, help="Input shape as a list of integers.")
    parser.add_argument(
        "--forcing_shape", type=int, nargs="+", required=True, help="Forcing shape as a list of integers."
    )
    parser.add_argument(
        "--output_shape", type=int, nargs="+", required=True, help="Output shape as a list of integers."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda or cpu).")
    parser.add_argument("--model_name", type=str, default=None, help="Optional model checkpoint name.")
    parser.add_argument("--init_noise", type=int, default=None, help="init model noise")

    args = parser.parse_args()

    start_time = time.time()

    # # Call the run_year_rmse function with parsed arguments
    # test_tensor_rmse, truth_field, inds_to_rmse, metrics, conf, METS = run_year_rmse(
    #     config=args.config,
    #     input_shape=args.input_shape,
    #     forcing_shape=args.forcing_shape,
    #     output_shape=args.output_shape,
    #     device=args.device,
    #     model_name=args.model_name
    # )

    num_cpus = 12
    with mp.Pool(num_cpus) as p:
        run_year_rmse(
            p,
            config=args.config,
            input_shape=args.input_shape,
            forcing_shape=args.forcing_shape,
            output_shape=args.output_shape,
            device=args.device,
            model_name=args.model_name,
            init_noise=args.init_noise,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    # How to run:
    # python Quick_Climate_Year.py --config /path/to/config.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt
    print(f"Run completed. Results saved to configured location. Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Run completed. Results saved to configured location. Elapsed time: {elapsed_time/60:.2f} minutes")


if __name__ == "__main__":
    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    main()
