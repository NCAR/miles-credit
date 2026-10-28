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

from credit.grid_align import align_spatial, padded_hw


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# [SAVE-PATCH] avoid HDF5 file-lock stalls on GLADE
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def quantize(x, digits):
    """[SAVE-PATCH] Round to a power-of-two step <= 10**-digits (bit rounding).
    Zeroes the noisy low mantissa bits so zlib compresses well and fast.
    digits=3 -> step 1/1024, max abs error ~0.0005 (K or reflectance units)."""
    bits = int(np.ceil(np.log2(10.0 ** digits)))
    scale = np.float32(2.0 ** bits)
    return np.round(x * scale) / scale

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

def snap_to_goes_times(target_times, zarr_path, how="previous", max_gap="20m"):
    """Map requested (clock-round) init times onto real GOES scan times.
 
    how = "previous" -> last scan at or before the target  (no look-ahead; fair vs HRRR)
          "next"     -> first scan at or after the target
          "nearest"  -> closest scan either side (current implicit behaviour)
    Targets with no scan inside max_gap are dropped and reported.
    """
    with xr.open_zarr(zarr_path, chunks=None) as ds:
        grid = pd.DatetimeIndex(np.asarray(ds["t"].values)).sort_values()
 
    targets = pd.DatetimeIndex(pd.to_datetime(list(target_times)))
    method = {"previous": "ffill", "next": "bfill", "nearest": "nearest"}[how]
    idx = grid.get_indexer(targets, method=method, tolerance=pd.Timedelta(max_gap))
 
    ok = idx >= 0
    snapped = grid[idx[ok]]
    offsets = (snapped - targets[ok]).to_series().dt.total_seconds()
 
    if (~ok).any():
        logger.warning(
            "%d/%d requested init times had no GOES scan within %s (e.g. %s)",
            (~ok).sum(), len(targets), max_gap, list(targets[~ok][:5]),
        )
    logger.info(
        "snap=%s | offset vs requested: min %.0fs max %.0fs median %.0fs",
        how, offsets.min(), offsets.max(), offsets.median(),
    )
    if offsets.round().nunique() > 1:  # [SAVE-PATCH] ignore sub-second jitter
        logger.warning(
            "init-time offset is NOT constant across the period "
            "(%d distinct values) -- check the GOES-16/GOES-19 transition.",
            offsets.nunique(),
        )
 
    # requested hour -> actual scan used, for pairing with HRRR later
    mapping = pd.DataFrame(
        {"requested": targets[ok], "init_used": snapped, "offset_s": offsets.values}
    )
    return list(snapped), mapping


def forecast_dir(conf):
    """[SAVE-PATCH] where forecasts go: predict.save_forecast, else <save_loc>/forecasts."""
    return conf["predict"].get("save_forecast") or os.path.expandvars(
        join(conf["save_loc"], "forecasts"))


def filter_completed_inits(conf, init_times):
    """[SAVE-PATCH] resume support: drop inits whose zarr store has a .complete
    marker. Stores are named with the init time truncated to whole seconds."""
    pconf = conf["predict"]
    if pconf.get("save_format", "netcdf").lower() != "zarr" or not pconf.get("skip_completed", True):
        return list(init_times)
    out_dir = forecast_dir(conf)
    done = set()
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.endswith(".zarr.complete"):
                try:
                    done.add(pd.Timestamp(name[: -len(".zarr.complete")]))
                except ValueError:
                    pass
    keep = [t for t in init_times
            if pd.Timestamp(t).floor("s") not in done and pd.Timestamp(t).round("s") not in done]
    logger.info(f"resume: {len(init_times) - len(keep)} of {len(init_times)} inits already "
                f"complete in {out_dir}; {len(keep)} to run")
    return keep


def get_rollout_init_times(conf):
    forecast_conf = conf["predict"]["forecasts"]

    if forecast_conf["type"] == "debugger":
        init_times_str = [
                            "2022-05-21T17:55:06", # NA spring case
                            "2022-06-30T23:55:06", 
                            "2022-07-01T14:55:06", # NA convective case
                            "2022-12-02T11:55:06", 
                            "2022-12-15T23:55:05", # SA convective case
                            "2022-12-16T11:55:06", # SA convective case
                            "2025-06-14T18:25:05", # the one with cyclone
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

    elif forecast_conf["type"] == "2025_eval":
        start = pd.Timestamp(forecast_conf.get("start_datetime", "2025-05-01"))
        end = pd.Timestamp(forecast_conf.get("end_datetime", "2025-05-15"))
        freq = forecast_conf.get("ic_interval", "1h")
        targets = pd.date_range(start=start, end=end, freq=freq, inclusive="left")

        snapped, mapping = snap_to_goes_times(
            targets,
            conf["data"]["save_loc"],
            how=forecast_conf.get("snap", "previous"),
        )
        # [SAVE-PATCH] one mapping CSV per date range, next to the forecasts, so
        # parallel chunk jobs never overwrite each other (was skipped entirely
        # when save_forecast was unset)
        save_dir = forecast_dir(conf)
        os.makedirs(save_dir, exist_ok=True)
        mapping.to_csv(os.path.join(
            save_dir, f"init_time_map_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.csv"), index=False)
        return snapped
    
    # elif forecast_conf["type"] == "2025_eval":
    #     rollout_init_times = pd.date_range(
    #         start="2025-01-01",
    #         end="2026-01-01",
    #         freq="h",
    #         inclusive="left",
    #     )
    #     return [pd.Timestamp(t) for t in sorted(rollout_init_times)]

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
            "2025-06-20T18:00:00",
            "2025-06-30T00:00:00",
            "2025-06-30T12:00:00",
            "2025-06-26T12:00:00",
            "2025-06-26T18:00:00",
            "2025-06-23T06:00:00",
            "2025-07-05T00:00:00",
            "2025-07-02T18:00:00",
            "2025-06-30T06:00:00",
            "2025-06-21T00:00:00",
            "2025-06-14T00:00:00",
            "2025-06-15T00:00:00",
            "2025-07-01T18:00:00",
            "2025-07-04T12:00:00",
            "2025-06-19T06:00:00",
            "2025-06-17T12:00:00",
            "2025-06-29T12:00:00",
            "2025-07-05T12:00:00",
            "2025-06-18T18:00:00",
            "2025-06-20T06:00:00",
            "2025-06-30T18:00:00",
            "2025-07-01T00:00:00",
            "2025-06-28T00:00:00",
            "2025-06-24T12:00:00",
        ]

        return [pd.Timestamp(t) for t in sorted(rollout_init_times)]

    raise ValueError(f"{forecast_conf['type']} is not a valid rollout type")

class ForecastProcessor:
    def __init__(self, conf, device, num_steps=None):
        self.conf = conf
        self.num_steps = num_steps  # [SAVE-PATCH] forecast steps per init (zarr store length)
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

        # [SAVE-PATCH] keep everything float32 (the float64 scaler was silently
        # promoting every forecast to float64 -> 2x the bytes to move and compress)
        self.active_scaler = self.active_scaler.astype(np.float32)

        # [SAVE-PATCH] ---------------- output options (predict.* in the YAML) ----------------
        pconf = conf["predict"]
        self.save_format = pconf.get("save_format", "netcdf").lower()   # "zarr" | "netcdf"
        if self.save_format not in ("zarr", "netcdf"):
            raise ValueError(f"predict.save_format must be 'zarr' or 'netcdf', got {self.save_format!r}")

        save_conf = pconf.get("save_encoding", {}) or {}
        self.quantize_digits = save_conf.get("quantize_digits", 3)   # null -> lossless float32
        default_level = 3 if self.save_format == "zarr" else 1
        self.complevel = int(save_conf.get("complevel", default_level))  # zstd (zarr) / zlib (netcdf); 0 -> none

        # which predicted channels to write; null/omitted -> all predicted channels
        predicted = [int(c) for c in self.channel.values]
        requested = pconf.get("save_channels") or predicted
        requested = [int(c) for c in requested]
        missing = [c for c in requested if c not in predicted]
        if missing:
            raise ValueError(f"save_channels {missing} are not predicted by this model "
                             f"(predicted channels: {predicted})")
        self.save_channels = requested
        self.save_channel_idx = [predicted.index(c) for c in requested]  # positions in y_pred

        # inverse-transform constants for the saved channels only, as float32 numpy
        sc = self.active_scaler.sel(channel=self.save_channels)
        self._mean = sc["mean"].values.astype(np.float32)[:, None, None]
        self._std = sc["std"].values.astype(np.float32)[:, None, None]
        self._log_mask = np.array([self.log_normal_scaling and c <= 6 for c in self.save_channels])

        self.timestep = pd.Timedelta(conf["data"]["timestep"])
        self.shard_per_step = bool(save_conf.get("shard_per_step", True))  # fewer files
        self.channels_in_one_chunk = False
        # how far a real scan time may sit from the nominal step (default 2 min)
        self.step_tolerance = pd.Timedelta(pconf.get("save_step_tolerance", "2min"))
        logger.info(f"saving {self.save_format}: channels {self.save_channels}, "
                    f"quantize_digits={self.quantize_digits}, complevel={self.complevel}")

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

    # ------------------------------------------------------------------ #
    # [SAVE-PATCH] writer: numpy inverse transform, then zarr or netcdf
    # ------------------------------------------------------------------ #
    def inverse_transform(self, block):
        """block: (members, saved_channels, lat, lon) in model space -> physical units."""
        out = block * self._std + self._mean
        if self._log_mask.any():
            out[:, self._log_mask] = np.exp(out[:, self._log_mask])
        return out

    def _layout(self):
        """dims / shape / chunks of one init's store: one chunk = one step x all
        members x one channel x full domain."""
        nlat, nlon = len(self.padded_lat), len(self.padded_lon)
        nch = len(self.save_channels)
        if self.ensemble_size > 1:
            E = self.ensemble_size
            return (("t", "ensemble_member_label", "channel", "latitude", "longitude"),
                    (self.num_steps, E, nch, nlat, nlon), (1, E, 1, nlat, nlon))
        return (("t", "channel", "latitude", "longitude"),
                (self.num_steps, nch, nlat, nlon), (1, 1, nlat, nlon))

    def _zarr_encoding(self, chunks):
        enc = {"chunks": chunks, "dtype": "float32"}
        if self.complevel <= 0:
            enc["compressors"] = None
            return enc
        import zarr
        if int(zarr.__version__.split(".")[0]) >= 3:
            from zarr.codecs import BloscCodec
            enc["compressors"] = [BloscCodec(cname="zstd", clevel=self.complevel, shuffle="bitshuffle")]
        else:
            from numcodecs import Blosc
            enc["compressor"] = Blosc(cname="zstd", clevel=self.complevel, shuffle=Blosc.BITSHUFFLE)
        return enc

    def store_path(self, init_datetime):
        return join(self.save_forecast_dir, f"{init_datetime}.zarr")

    def create_store(self, init_datetime):
        """Called in the MAIN process when an init starts: writes metadata and
        coordinates only. Workers then fill one t-slice each (region writes)."""
        import dask.array as dsa
        if self.num_steps is None:
            raise ValueError("ForecastProcessor needs num_steps for zarr output")
        dims, shape, chunks = self._layout()
        init = pd.Timestamp(init_datetime)
        coords = {
            "t": init + self.timestep * np.arange(1, self.num_steps + 1),
            "channel": np.array(self.save_channels, dtype=np.int64),
            "latitude": np.asarray(self.padded_lat),
            "longitude": np.asarray(self.padded_lon),
        }
        if self.ensemble_size > 1:
            coords["ensemble_member_label"] = np.arange(self.ensemble_size)
        # dask chunk = one whole step (never written: compute=False). It must cover whole
        # zarr chunks/shards, so it can't be the per-channel chunk when sharding.
        step_shape = tuple(chunks[:-3]) + (len(self.save_channels),) + tuple(chunks[-2:])
        data = dsa.full(shape, np.nan, dtype=np.float32, chunks=step_shape)
        # t is the NOMINAL valid time (init + k*timestep). The rollout's real valid
        # times are GOES scan stamps that jitter by ~1 s, so each worker records the
        # actual one in valid_time[k]. One chunk per step: workers never share a chunk.
        valid_time = dsa.from_array(np.asarray(coords["t"], dtype="datetime64[ns]"),
                                    chunks=self.num_steps)
        ds = xr.Dataset({"BT_or_R": (dims, data), "valid_time": (("t",), valid_time)},
                        coords=coords)
        ds.attrs.update(init_time=str(init_datetime),
                        quantize_digits=str(self.quantize_digits),
                        note="t = nominal valid time; valid_time = actual GOES scan time. "
                             "Steps not yet written read as NaN; "
                             "<store>.complete marks a fully written init")
        path = self.store_path(init_datetime)
        if os.path.exists(path + ".complete"):
            os.remove(path + ".complete")
        vt_enc = {"chunks": (self.num_steps,), "dtype": "int64", "units": "seconds since 1970-01-01"}
        bt_enc = self._zarr_encoding(chunks)
        if self.shard_per_step:
            # one FILE per step (all members + channels), one chunk per channel inside it:
            # ~4x fewer files, reading a single channel still decodes only its chunk
            shard = step_shape
            try:
                ds.to_zarr(path, mode="w", compute=False,
                           encoding={"BT_or_R": {**bt_enc, "shards": shard}, "valid_time": vt_enc})
                return
            except Exception as e:  # zarr v2 / older xarray: no sharding
                logger.warning(f"zarr sharding unavailable ({e!r}); falling back to one chunk "
                               f"(= one file) per step holding all saved channels")
                self.shard_per_step = False
                self.channels_in_one_chunk = True
        if self.channels_in_one_chunk:
            bt_enc["chunks"] = step_shape
        ds.to_zarr(path, mode="w", compute=False,
                   encoding={"BT_or_R": bt_enc, "valid_time": vt_enc})

    def mark_complete(self, init_datetime):
        if self.save_format == "zarr":
            open(self.store_path(init_datetime) + ".complete", "w").close()

    def _write_zarr_step(self, block, init_datetime, save_datetime, step):
        """Write rollout step `step` (1-based) into t-slot step-1. Placement uses the
        rollout's own step counter: real GOES scan stamps can sit minutes off the
        10-min grid (e.g. 11:02:43), so time-matching is only a logged sanity check.
        The actual stamps go into valid_time, written once per init by the main process."""
        if not 1 <= step <= self.num_steps:
            raise ValueError(f"step {step} outside 1..{self.num_steps} for init {init_datetime}")
        miss = abs(pd.Timestamp(save_datetime) - pd.Timestamp(init_datetime) - step * self.timestep)
        if miss > self.step_tolerance:
            print(f"WARNING: step {step} of {init_datetime} has valid time {save_datetime}, "
                  f"{miss} off the nominal grid (kept; recorded in valid_time)")
        dims, _, _ = self._layout()
        data = block[None] if self.ensemble_size > 1 else block  # add t axis (E=1: member axis is t)
        xr.Dataset({"BT_or_R": (dims, data)}).to_zarr(
            self.store_path(init_datetime), mode="r+", region={"t": slice(step - 1, step)})

    def write_valid_times(self, init_datetimes, step_times):
        """Main process, after all steps of a batch are written. step_times[k][j] is
        the valid time of step k+1 for init j."""
        if self.save_format != "zarr":
            return
        for j, init_datetime in enumerate(init_datetimes):
            vt = np.array([pd.Timestamp(st[j]) for st in step_times], dtype="datetime64[ns]")
            if len(vt) != self.num_steps:
                raise ValueError(f"{init_datetime}: got {len(vt)} steps, store has {self.num_steps}")
            xr.Dataset({"valid_time": (("t",), vt)}).to_zarr(
                self.store_path(init_datetime), mode="r+", region={"t": slice(0, self.num_steps)})

    def _write_netcdf_step(self, block, init_datetime, save_datetime):
        """Old layout: <init>/<valid>.nc, one file per step. Written to .tmp and renamed."""
        os.makedirs(join(self.save_forecast_dir, init_datetime), exist_ok=True)
        coords = {"t": [pd.Timestamp(save_datetime)],
                  "channel": np.array(self.save_channels, dtype=np.int64),
                  "latitude": np.asarray(self.padded_lat),
                  "longitude": np.asarray(self.padded_lon)}
        if self.ensemble_size > 1:
            dims = ("ensemble_member_label", "t", "channel", "latitude", "longitude")
            data = block[:, None]
            coords["ensemble_member_label"] = np.arange(self.ensemble_size)
        else:
            dims = ("t", "channel", "latitude", "longitude")
            data = block
        ds = xr.Dataset({"BT_or_R": (dims, data)}, coords=coords)
        enc = {"dtype": "float32"}
        if self.complevel > 0:
            chunks = tuple(ds.sizes[d] if d in ("latitude", "longitude") else 1 for d in dims)
            enc.update(zlib=True, complevel=self.complevel, shuffle=True, chunksizes=chunks)
        out = join(self.save_forecast_dir, init_datetime, f"{save_datetime}.nc")
        ds.to_netcdf(out + ".tmp", mode="w", engine="netcdf4", encoding={"BT_or_R": enc})
        os.replace(out + ".tmp", out)

    def process(self, y_pred, init_datetimes, save_datetimes, step=None):
        """y_pred: numpy (batch*members, saved_channels, 1, lat, lon), channels
        already subset on the GPU. Members of init j are rows j*E .. (j+1)*E-1
        (repeat_interleave order)."""
        try:
            E = self.ensemble_size
            for j, (init_datetime, save_datetime) in enumerate(zip(init_datetimes, save_datetimes)):
                t0 = time.time()
                block = self.inverse_transform(y_pred[j * E:(j + 1) * E, :, 0])  # (E, C, lat, lon)
                if self.quantize_digits is not None:
                    block = quantize(block, self.quantize_digits)
                block = block.astype(np.float32, copy=False)
                if self.save_format == "zarr":
                    self._write_zarr_step(block, init_datetime, save_datetime, step)
                else:
                    self._write_netcdf_step(block, init_datetime, save_datetime)
                print(f"processed {save_datetime} initialized at {init_datetime} "
                      f"(write {time.time() - t0:.1f}s)")
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
    rollout_init_times = filter_completed_inits(conf, rollout_init_times)  # [SAVE-PATCH]
    if not rollout_init_times:
        logger.info("all requested inits are already complete; nothing to do")
        return 1

    logger.info(f"rolling out with init times {rollout_init_times}")

    if len(rollout_init_times) < batch_size:
        logger.warning(
            f"number of forecast init times {len(rollout_init_times)} is less than batch_size {batch_size}, will result in under-utilization"
        )

    dataset = load_predict_dataset(conf, 0, 0, rollout_init_times, device)
    dataloader = load_dataloader(conf, dataset, 0, 1, is_predict=True)

    # Class for saving in parallel
    result_processor = ForecastProcessor(conf, device, num_steps=dataset.num_forecast_steps)

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
    # [SAVE-PATCH] max steps waiting to be written (default 2x pool size)
    max_inflight = conf["predict"].get("max_inflight_writes", 2 * getattr(p, "_processes", 8))

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

            # [SAVE-PATCH] one zarr store per init, created before any worker writes to it
            if result_processor.save_format == "zarr":
                for init_dt in init_datetimes:
                    result_processor.create_store(init_dt)
            step = 0          # [SAVE-PATCH] rollout step counter (1-based once incremented)
            step_times = []   # [SAVE-PATCH] valid times per step, for valid_time
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
                    if ensemble_size > 1:
                        era5_static = torch.repeat_interleave(era5_static, ensemble_size, 0)
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
                    pad_conf = conf["model"].get("padding_conf", {})
                    pad_lat = tuple(pad_conf.get("pad_lat", [0, 0]))
                    pad_lon = tuple(pad_conf.get("pad_lon", [0, 0]))

                    # Forcing grids differ by resolution: at 10 km ERA5/static were on the padded
                    # grid and static needed cropping; at 3 km both are native and x_era5 needs
                    # padding instead. Derive from shapes rather than assuming either.
                    native = x.shape[-2:]
                    concat_x_list = [x]
                    if era5_static is not None:
                        concat_x_list.append(align_spatial(era5_static, native, pad_lat, pad_lon))
                    if "temporal_forcing_10m" in batch_era5:
                        tf = batch_era5["temporal_forcing_10m"].to(device)
                        if ensemble_size > 1:
                            tf = torch.repeat_interleave(tf, ensemble_size, 0)
                        concat_x_list.append(align_spatial(tf, native, pad_lat, pad_lon))
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

                # [SAVE-PATCH] backpressure: cap unwritten steps held in RAM.
                # Total runtime is unchanged (GPU waited at "stop" anyway),
                # but memory stays bounded and a write bottleneck shows up immediately.
                while len(results) >= max_inflight:
                    results.pop(0).get()
                step += 1
                step_times.append(list(save_datetimes))

                result = p.apply_async(
                    result_processor.process,
                    (
                        # [SAVE-PATCH] subset channels on the GPU: fewer bytes to copy and pickle
                        y_pred[:, :n_goes][:, result_processor.save_channel_idx].float().cpu().numpy(),
                        init_datetimes,
                        save_datetimes,
                        step,
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
                    results = []  # [SAVE-PATCH] don't keep finished handles forever
                    result_processor.write_valid_times(init_datetimes, step_times)
                    for init_dt in init_datetimes:
                        result_processor.mark_complete(init_dt)

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

    # [SAVE-PATCH] date window for this job (2025_eval); overrides the YAML so
    # one config can drive many chunk jobs. End is exclusive.
    parser.add_argument("--start", dest="start", type=str, default=None,
                        help="override predict.forecasts.start_datetime (2025_eval)")
    parser.add_argument("--end", dest="end", type=str, default=None,
                        help="override predict.forecasts.end_datetime (2025_eval, exclusive)")

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
    start_override = args_dict.pop("start")
    end_override = args_dict.pop("end")

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

    if start_override or end_override:  # [SAVE-PATCH]
        fc = conf["predict"]["forecasts"]
        if fc.get("type") != "2025_eval":
            raise ValueError("--start/--end only apply to forecasts.type '2025_eval'")
        if start_override:
            fc["start_datetime"] = start_override
        if end_override:
            fc["end_datetime"] = end_override

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