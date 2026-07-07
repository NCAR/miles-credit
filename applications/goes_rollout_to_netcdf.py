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
from dateutil.parser import parse

# ---------- #
# Numerics
import xarray as xr
import numpy as np

# ---------- #
import torch

# ---------- #
# credit
from credit.models import load_model
from credit.seed import seed_everything
from credit.distributed import get_rank_info
from credit.datasets.goes_load_dataset_and_dataloader import (
    load_predict_dataset,
    load_dataloader,
)
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


def is_datetime_str(s):
    try:
        parse(s)
        return True
    except:
        return False


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
        return rollout_init_times

    elif forecast_conf["type"] == "custom":
        rollout_init_times = [pd.Timestamp(time) for time in forecast_conf["init_times"]]
        return rollout_init_times

    elif forecast_conf["type"] == "2025_test_rollout":
        rollout_init_times = [
            "2025-06-24T00:00:00",
            "2025-07-06T06:00:00",
            "2025-06-13T00:00:00",
            "2025-06-18T12:00:00",
            "2025-06-14T18:00:00",
            "2025-07-07T00:00:00",
            "2025-06-28T12:00:00",
            "2025-07-02T00:00:00",
            "2025-06-13T06:00:00",
            "2025-06-22T18:00:00",
            "2025-06-27T06:00:00",
            "2025-07-05T18:00:00",
            "2025-07-07T06:00:00",
            "2025-07-04T18:00:00",
            "2025-06-26T06:00:00",
            "2025-06-18T06:00:00",
            "2025-07-05T06:00:00",
            "2025-07-01T06:00:00",
            "2025-06-25T18:00:00",
            "2025-06-27T12:00:00",
            "2025-06-14T06:00:00",
            "2025-06-16T12:00:00",
            "2025-06-19T18:00:00",
            "2025-06-25T12:00:00",
            "2025-06-18T00:00:00",
            "2025-06-22T12:00:00",
            "2025-06-21T18:00:00",
            "2025-06-29T06:00:00",
            "2025-06-20T12:00:00",
            "2025-06-19T12:00:00",
            "2025-06-25T06:00:00",
            "2025-07-01T12:00:00",
            "2025-06-22T06:00:00",
            "2025-07-04T00:00:00",
            "2025-06-16T00:00:00",
            "2025-06-15T12:00:00",
            "2025-06-28T06:00:00",
            "2025-07-02T12:00:00",
            "2025-07-06T18:00:00",
            "2025-07-03T12:00:00",
            "2025-07-08T18:00:00",
            "2025-06-24T18:00:00",
            "2025-06-14T12:00:00",
            "2025-06-21T12:00:00",
            "2025-07-08T12:00:00",
            "2025-07-08T06:00:00",
            "2025-06-27T18:00:00",
            "2025-06-27T00:00:00",
            "2025-06-23T00:00:00",
            "2025-07-06T00:00:00",
            "2025-06-15T06:00:00",
            "2025-07-06T12:00:00",
            "2025-07-09T00:00:00",
            "2025-06-24T06:00:00",
            "2025-07-08T00:00:00",
            "2025-06-16T06:00:00",
            "2025-06-17T06:00:00",
            "2025-06-16T18:00:00",
            "2025-07-04T06:00:00",
            "2025-06-28T18:00:00",
            "2025-06-29T18:00:00",
            "2025-07-03T18:00:00",
            "2025-07-03T00:00:00",
            "2025-06-22T00:00:00",
            "2025-06-17T18:00:00",
            "2025-06-20T00:00:00",
            "2025-06-25T00:00:00",
            "2025-06-19T00:00:00",
            "2025-06-17T00:00:00",
            "2025-07-02T06:00:00",
            "2025-06-26T00:00:00",
            "2025-07-03T06:00:00",
            "2025-06-29T00:00:00",
            "2025-06-15T18:00:00",
            "2025-06-21T06:00:00",
            "2025-06-20T12:00:00",
            "2025-06-30T18:00:00",
            "2025-07-01T00:00:00",
            "2025-06-28T00:00:00",
            "2025-06-24T12:00:00",
        ]

        return [pd.Timestamp(t) for t in sorted(rollout_init_times)]

    raise ValueError(f"{forecast_conf['type']} is not a valid rollout type")

class ForecastProcessor:
    def __init__(self, conf, device):
        self.conf = conf
        self.device = device
        padding_conf = conf["model"].get("padding_conf", {})
        self.pad_output = (
            True if not padding_conf else not padding_conf.get("activate", False)
        )

        self.save_forecast_dir = conf["predict"].get("save_forecast", None)
        if not self.save_forecast_dir:
            self.save_forecast_dir = os.path.expandvars(
                join(conf["save_loc"], "forecasts")
            )

        self.batch_size = conf["predict"].get("batch_size", 1)
        self.ensemble_size = conf["predict"].get("ensemble_size", 1)
        self.lead_time_periods = conf["data"]["lead_time_periods"]

        if self.ensemble_size > 1:
            self.ensemble_index = xr.DataArray(
                np.arange(self.ensemble_size), dims="ensemble_member_label"
            )

        # FIX: Initialize active_channels BEFORE calling _ref_array(),
        # which needs self.active_channels to select channels from the Zarr store.
        self.surface_variables = conf["data"].get(
            "surface_variables",
            [f"C{str(i).zfill(2)}" for i in range(1, 17)]
        )
        self.active_channels = [
            int(v.replace("C", "")) for v in self.surface_variables if v.startswith("C")
        ]

        # Now safe to call _ref_array() because self.active_channels exists
        self.ref_array = self._ref_array()

        # setup scaler
        base_scaler_path = conf["data"].get(
            "goes_scaler_path",
            "/glade/derecho/scratch/bagherio/cloud.dir/datasets/goes_cloud_dataset/goes_stats_linear.nc"
        )
        base_scaler = xr.open_dataset(base_scaler_path).load()

        # Build Active Scaler precise to requested channels
        self.active_scaler = base_scaler.sel(channel=self.active_channels).copy()

        self.log_normal_scaling = conf["data"].get("log_normal_scaling", False)
        if self.log_normal_scaling:
            log_scaler_path = conf["data"].get(
                "goes_log_scaler_path",
                "/glade/derecho/scratch/bagherio/cloud.dir/datasets/goes_cloud_dataset/goes_stats_vis_log.nc"
            )
            logger.info(f"Inverse log normalizing visible channels using: {log_scaler_path}")
            log_scaler = xr.open_dataset(log_scaler_path).load()

            # Dynamically override the linear stats with log stats ONLY for visible channels
            for ch in self.active_channels:
                if ch <= 6:
                    self.active_scaler["mean"].loc[dict(channel=ch)] = log_scaler["mean"].sel(channel=ch)
                    self.active_scaler["std"].loc[dict(channel=ch)] = log_scaler["std"].sel(channel=ch)

    def _ref_array(self):
        # FIX 1: The data source is a Zarr store, not a classic NetCDF/HDF5 file.
        # xr.open_dataset does not know how to read Zarr; use xr.open_zarr instead.
        # chunks=None keeps reads synchronous (no Dask), matching the pattern used
        # everywhere else in the codebase for metadata-only opens.
        ds = xr.open_zarr(self.conf["data"]["save_loc"], chunks=None)
        ref_array = ds.isel(t=0)["BT_or_R"].sel(channel=self.active_channels).load().copy()

        self.channel = ref_array.channel
        self.padded_lat = ref_array.latitude
        self.padded_lon = ref_array.longitude

        if self.pad_output:
            self.padded_lat = np.pad(
                ref_array.latitude, (11, 10), mode="linear_ramp",
                end_values=(-50.1 - 11 * 0.1, 50.1 + 10 * 0.1)
            )
            self.padded_lon = np.pad(
                ref_array.longitude, (19, 18), mode="linear_ramp",
                end_values=(-121.1 - 19 * 0.1, -28.9 + 18 * 0.1)
            )

    def inverse_transform_ABI(self, da):
        scaler = self.active_scaler.assign_coords(channel=da.channel)
        unscaled = da * scaler["std"] + scaler["mean"]

        if self.log_normal_scaling:
            for idx, ch in enumerate(da.channel.values):
                if ch <= 6:
                    unscaled[:, idx] = np.exp(unscaled[:, idx])
        return unscaled

    def make_dataarray(self, y_pred_member, save_datetime):
        y_pred_member = y_pred_member.squeeze(2)  # 1,c,t,lat,lon -> 1, c, lat, lon
        da = xr.DataArray(
            y_pred_member,
            dims=["t", "channel", "latitude", "longitude"],
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
        pred_ds.to_netcdf(
            join(self.save_forecast_dir, init_datetime, f"{save_datetime}.nc"), mode="w"
        )

    def process(self, y_pred, init_datetimes, save_datetimes):
        try:
            for dt in init_datetimes:
                os.makedirs(join(self.save_forecast_dir, dt), exist_ok=True)
            for j, times in enumerate(zip(init_datetimes, save_datetimes)):
                init_datetime, save_datetime = times
                y_pred_ensemble = []
                for i in range(self.ensemble_size):
                    y_pred_member = self.make_dataarray(y_pred[j + i : j + i + 1], save_datetime)
                    y_pred_ensemble.append(y_pred_member)
                if self.ensemble_size > 1:
                    pred_da = xr.concat(y_pred_ensemble, self.ensemble_index)
                else:
                    pred_da = y_pred_member
                self.save_dataarray(pred_da, init_datetime, save_datetime)
                print(f"processed {save_datetime} initialized at {init_datetime}")
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

    # Number of GOES output channels — model predicts only these, not static/temporal.
    # This must match the trainer, which uses y_pred[:, :n_goes] everywhere.
    n_goes = len(conf["data"]["surface_variables"])

    # Read padding crop amounts once from config so they are never out of sync
    # with what the trainer does.  These are used to crop the ERA5-grid static
    # tensor back to the native GOES grid before concatenation (FIX 2 below).
    pad_conf = conf["model"].get("padding_conf", {})
    pad_lat = pad_conf.get("pad_lat", [0, 0])   # e.g. [11, 10]
    pad_lon = pad_conf.get("pad_lon", [0, 0])   # e.g. [19, 18]

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
        model = load_model_state(conf, model, device)
    else:
        raise ValueError(
            f"""{conf["predict"]["mode"]} is not a valid prediction mode"""
        )

    # Check for noise calibration mode
    noise_calibration = conf["predict"].get("noise_calibration", False)

    if noise_calibration:
        logger.info("Noise calibration mode enabled: Setting model.train() and overriding noise factors.")
        model.train()
        if hasattr(model, 'bottleneck_noise_factor'):
            model.bottleneck_noise_factor = conf["model"]["bottleneck_noise_factor"]
        if hasattr(model, 'encoder_noise_factor'):
            model.encoder_noise_factor = conf["model"]["encoder_noise_factor"]
        if hasattr(model, 'decoder_noise_factor'):
            model.decoder_noise_factor = conf["model"]["decoder_noise_factor"]
    else:
        logger.info("Standard evaluation mode: Setting model.eval().")
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

            # ------------------------------------------------------------------ #
            # Load the INIT batch.
            # ------------------------------------------------------------------ #
            batch = next(dl)
            if "era5" in batch.keys():
                batch_era5 = batch["era5"]
            init_datetimes = batch["datetime"]
            batch_size = len(init_datetimes)
            mode = batch["mode"][0]

            x = batch["x"].to(device).float()

            # Create ensemble copies
            if ensemble_size > 1:
                x = torch.repeat_interleave(x, ensemble_size, 0)

            # Grab the static ERA5 fields from the init batch.
            # Static fields do not change over the rollout, so we hold onto
            # this reference and reuse it at every subsequent rollout step.
            if "era5" in batch.keys():
                era5_static = batch_era5.get("static", None)
                if era5_static is not None and era5_static.numel() > 0:
                    era5_static = era5_static.to(device)
                else:
                    era5_static = None

            # ------------------------------------------------------------------ #
            # Rollout while loop — runs until the "stop" batch is reached.
            # ------------------------------------------------------------------ #
            while mode != "stop":

                if flag_clamp:
                    x = torch.clamp(x, min=clamp_min, max=clamp_max)

                if "era5" in batch.keys():
                    # ---------------------------------------------------------- #
                    # FIX 2: Crop era5_static from the ERA5 grid (1024×960) back
                    # to the native GOES grid (1003×923) before concatenation.
                    #
                    # The trainer does this crop in every forward step:
                    #   era5_static_cropped = era5_static[
                    #       :, :, :, pad_lat[0]:h_end, pad_lon[0]:w_end
                    #   ]
                    # The old rollout code skipped this, so the static tensor had
                    # the wrong spatial size, causing a shape mismatch inside the
                    # model's padding layer.
                    # ---------------------------------------------------------- #
                    concat_x_list = [x]

                    if era5_static is not None:
                        h_end = era5_static.shape[-2] - pad_lat[1]  # 1024 - 10 = 1014
                        w_end = era5_static.shape[-1] - pad_lon[1]  #  960 - 18 = 942
                        era5_static_cropped = era5_static[
                            :, :, :, pad_lat[0]:h_end, pad_lon[0]:w_end
                        ]  # → [..., 1003, 923]
                        concat_x_list.append(era5_static_cropped)

                    # Temporal forcing is already on the native GOES grid — no crop
                    if "temporal_forcing_10m" in batch_era5:
                        concat_x_list.append(batch_era5["temporal_forcing_10m"].to(device))

                    # x is now 35 channels at native GOES resolution (1003×923).
                    # The model will pad it to 1024×960 via padding_conf internally.
                    x = torch.concat(concat_x_list, dim=1).float()

                    # Build FiLM input: prognostic (3D+2D) + dynamic_forcing
                    concat_era5_list = []
                    prog = batch_era5.get("prognostic", None)
                    if prog is not None and prog.numel() > 0:
                        concat_era5_list.append(prog.to(device))
                    dyn = batch_era5.get("dynamic_forcing", None)
                    if dyn is not None and dyn.numel() > 0:
                        concat_era5_list.append(dyn.to(device))

                    if concat_era5_list:
                        x_era5 = torch.concat(concat_era5_list, dim=1).float()
                        forcing_t_delta = batch_era5["timedelta_seconds"].to(device).float()

                        if ensemble_size > 1:
                            x_era5 = torch.repeat_interleave(x_era5, ensemble_size, 0)
                            forcing_t_delta = torch.repeat_interleave(forcing_t_delta, ensemble_size, 0)
                        if flag_clamp:
                            x_era5 = torch.clamp(x_era5, min=clamp_min, max=clamp_max)
                    else:
                        x_era5 = None

                # ------------------------------------------------------------------ #
                # FIX 3: Squash NaNs (e.g. ocean fill values) before the forward
                # pass.  The trainer does this unconditionally in every step; without
                # it, a single NaN propagates through all attention operations and
                # corrupts the entire output field.
                # ------------------------------------------------------------------ #
                x = torch.nan_to_num(x, nan=0.0)
                if "era5" in batch.keys() and x_era5 is not None:
                    x_era5 = torch.nan_to_num(x_era5, nan=0.0)

                # Model inference
                y_pred = (
                    model(x, x_era5=x_era5, forcing_t_delta=forcing_t_delta)
                    if "era5" in batch.keys() and x_era5 is not None
                    else model(x)
                )

                # Load the next batch (a "forcing" or "stop" batch)
                batch = next(dl)
                save_datetimes = batch["datetime"]

                # ------------------------------------------------------------------ #
                # FIX 5: Update batch_era5 from the newly loaded batch so that the
                # next iteration of the while loop gets the correct temporal forcing
                # and dynamic ERA5 fields for that forecast step.
                #
                # GOES10kmDataset.get_era5() always calls the ERA5Dataset with mode
                # "init", so "forcing" batches still contain prognostic + dynamic +
                # temporal_forcing_10m (but NOT the "stop" batch, which has no era5
                # key at all).  We update batch_era5 only when the key is present;
                # for the "stop" batch we are about to break anyway so it does not
                # matter.
                # ------------------------------------------------------------------ #
                if "era5" in batch.keys():
                    batch_era5 = batch["era5"]

                result = p.apply_async(
                    result_processor.process,
                    (
                        y_pred[:, :n_goes].cpu(),
                        init_datetimes,
                        save_datetimes,
                    ),
                )

                results.append(result)

                mode = batch["mode"][0]

                # ------------------------------------------------------------------ #
                # FIX 4: Strip the output back to the GOES channels only before
                # feeding it as x into the next rollout step.
                #
                # The model output tensor y_pred has shape (B, C_out, T, H, W).
                # C_out == n_goes (16) for this model because surface_channels=35
                # are inputs and only the GOES portion is predicted.  Using the
                # full y_pred without slicing would work for single-step (the shapes
                # happen to match), but is semantically wrong and will break for any
                # multi-step rollout where the trainer explicitly does:
                #   x = y_pred[:, :n_goes]
                # ------------------------------------------------------------------ #
                x = y_pred[:, :n_goes].detach()

                if mode == "stop":
                    # Wait for all async save processes to finish
                    for result in results:
                        result.get()

                    y_pred = None

                    if distributed:
                        torch.distributed.barrier()

                    forecast_count += batch_size

    end_time = time.time()
    logger.info(
        f"total rollout time: {end_time - start_time:02}s for {len(rollout_init_times) * dataset.num_forecast_steps} forecasts"
    )
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
        default=-1,  # -1 does nothing
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

    if (
        debug == 1
        or conf["model"]["type"] == "debugger"
        or conf["predict"]["forecasts"]["type"] == "debugger"
        or conf.get("debug", False)
    ):
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
        logger.warning("forecast_save_loc not specified, saving to default location")
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
        if (
            conf["pbs"]["queue"] == "casper"
            or conf["pbs"]["queue"] == "develop"
            and conf["pbs"]["ngpus"] == 1
        ):
            logging.info("Launching to PBS on Casper or Derecho develop")
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

    # post check save_dir
    for p in Path(forecast_save_loc).iterdir():
        if is_datetime_str(p.name):
            print(f"{len(list(p.iterdir()))} {p.name}")


if __name__ == "__main__":
    main()