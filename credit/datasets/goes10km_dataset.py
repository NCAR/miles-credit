from typing import Dict

import torch
from torch.utils.data import Dataset

from os.path import join
import os
import time            # DIAG: wall-clock timing throughout __init__
import numpy as np
import pandas as pd
import xarray as xr
import logging

# import xesmf as xe
logger = logging.getLogger(__name__)


def _diag(msg: str) -> None:
    """
    Single helper used for all diagnostic prints in this file.

    Prepends wall-clock time, PID, and DataLoader worker ID (if inside a
    worker process) to every message so the output can be sorted/grepped:

        [HH:MM:SS pid=12345 worker=3] msg
        [HH:MM:SS pid=12345 worker=MAIN] msg

    Using print(..., flush=True) rather than logging so the lines appear
    immediately even when stdout is line-buffered across multiple nodes.
    """
    try:
        wi = torch.utils.data.get_worker_info()
        worker_tag = f"worker={wi.id}" if wi else "worker=MAIN"
    except Exception:
        worker_tag = "worker=?"
    ts = time.strftime("%H:%M:%S")
    print(f"[GOES_DIAG {ts} pid={os.getpid()} {worker_tag}] {msg}", flush=True)


class GOES10kmDataset(Dataset):
    def __init__(
        self,
        ds_path: str,
        data_conf: Dict,
        time_config: Dict = None,
        padding: bool = True,
        era5dataset: Dataset = None,
        evaluate: bool = False,
        valid_init_dir="/glade/derecho/scratch/bagherio/cloud.dir/datasets/goes_cloud_dataset/valid_init_times",
        scaler_ds_path="/glade/derecho/scratch/bagherio/cloud.dir/datasets/goes_cloud_dataset/data_stats.nc",
    ):
        """
        Dataset class for GOES-10km data with optional static and ERA5 forcing.
        Implements on-the-fly data loading inside workers to avoid Dask multiprocessing issues.

        The dataset is NOT opened in __init__. Instead, self.ds is set to None here
        and opened lazily inside each DataLoader worker on the first __getitem__ call.
        This prevents PyTorch from forking a Dask-backed xr.Dataset into worker
        processes, which caused catastrophic thread contention on Lustre (the
        "PyTorch + Dask Multiprocessing Collision" problem).

        A lightweight metadata-only dataset (_meta_ds) is opened once in the main
        process with chunks=None to perform all setup that requires the time coordinate
        (init_times, grid cache). It is explicitly deleted before workers are forked.

        Args:
            ds_path: path to the Zarr store (replaces the previously passed xr.Dataset).
                     Stored on self so each worker can open its own independent handle.
            data_conf: dict with dataset configuration.
            time_config: dict with time parameters.
            padding: bool, whether to pad inputs.
            era5dataset: optional ERA5Dataset for static/forcing fields.
            evaluate: bool, if True disables scaling and uses all times.
            valid_init_dir: path to store/load valid init time .nc files.
                            Also used to cache static grid files (lon_vals.npy,
                            lat_size.npy) so DataLoader workers never read the
                            Zarr store during __init__.
            scaler_ds_path: path to the ABI channel statistics netCDF.

        Example time config:
            {
                "timestep": pd.Timedelta(1, "h"),
                "num_forecast_steps": 1
            }
        """
        # ------------------------------------------------------------------ #
        # DIAG: record the wall-clock time at the very start of __init__ so
        # every subsequent checkpoint below shows elapsed seconds from here.
        # Also log whether we are inside a DataLoader worker process — if
        # worker=N appears here it means workers are re-running __init__,
        # which multiplies all I/O by num_workers×num_ranks simultaneously.
        # ------------------------------------------------------------------ #
        _t_init_start = time.time()
        _diag("__init__ START")

        self.evaluate = evaluate  # turns off scaling

        # ------------------------------------------------------------------ #
        # LAZY DATASET INIT
        #
        # self.ds_path stores the Zarr path so each DataLoader worker can open
        # its own independent file handle in __getitem__ (see the lazy-open
        # block there). self.ds is intentionally None here — it will be
        # populated in the worker process on the first __getitem__ call.
        #
        # Previously this class accepted a pre-opened xr.Dataset and stored it
        # as self.ds. That caused PyTorch's fork-based multiprocessing to clone
        # the Dask graph and its internal thread pool into every worker, leading
        # to massive thread contention and Lustre metadata locking on Derecho.
        # ------------------------------------------------------------------ #
        self.ds_path = ds_path
        self.ds = None  # intentionally None — opened per worker in __getitem__

        # ------------------------------------------------------------------ #
        # METADATA-ONLY OPEN (main process only)
        #
        # Open the Zarr store once with chunks=None (no Dask) purely to access
        # the time coordinate for init_times generation and the lon/lat grid for
        # the temporal forcing cache. This is safe in the main process because:
        #   - chunks=None means no Dask thread pool is created
        #   - _meta_ds is explicitly deleted below before PyTorch forks workers
        #
        # All methods that previously relied on self.ds during __init__
        # (_timestamps, _generate_valid_init_times, grid cache) now receive
        # _meta_ds as an explicit argument so they never touch self.ds.
        #
        # DIAG: sortby on a large time coordinate can be slow if not cached.
        # ------------------------------------------------------------------ #
        _t = time.time()
        _diag(f"opening metadata-only dataset (chunks=None) from {ds_path}")
        _meta_ds = xr.open_zarr(ds_path, chunks=None)
        _meta_ds = _meta_ds.drop_duplicates(dim="t").sortby("t")
        _diag(f"metadata dataset opened and sorted in {time.time()-_t:.2f}s")

        self.era5dataset = era5dataset
        self.padding = padding
        logger.info(f"{'  ' if self.padding else 'NOT '}padding GOES input")
        logger.info(
            f"GOES10km Dataset with{'  ' if self.era5dataset else 'out'} ERA5 forcing"
        )
        self.valid_sampling_modes = ["init", "forcing", "y", "stop"]  # class attribute

        # setup init times
        if not evaluate:  # dont need to specify time_config for evaluate mode
            self.timestep = time_config["timestep"]
            self.num_forecast_steps = time_config["num_forecast_steps"]
            self.start_datetime = time_config["start_datetime"]
            self.end_datetime = time_config["end_datetime"]

        self.valid_init_dir = valid_init_dir
        if not self.evaluate:
            # DIAG: _timestamps() either loads a cached .nc file (fast) or generates
            # valid init times from scratch (~5 min). Timing here confirms which.
            _t = time.time()
            _diag("_timestamps() START — loading or generating valid init times")
            # Pass _meta_ds explicitly so _timestamps and _generate_valid_init_times
            # can access the time coordinate without touching self.ds.
            self.init_times = self._timestamps(_meta_ds)
            _diag(f"_timestamps() done in {time.time()-_t:.2f}s  n_times={len(self.init_times)}")
            logger.info(f"initializing GOES10kmDataset with timestep [{self.timestep}] with {self.num_forecast_steps} steps")
        else:  # don't need to handle init times, this is handled by rollout already
            # Load only the time coordinate from the metadata dataset — this is
            # a lightweight operation (just the 1D 't' array, no spatial data).
            self.init_times = _meta_ds.t.load()
            logger.warning("loading dataset for with 0 forecast steps! only for eval")

        # --- DYNAMIC CHANNEL SELECTION ---
        # Default to all 16 ABI channels if not specified in config
        self.surface_variables = data_conf.get(
            "surface_variables",
            [f"C{str(i).zfill(2)}" for i in range(1, 17)]
        )
        # Extract integer channel IDs (e.g., 'C04' -> 4)
        self.active_channels = [int(v.replace("C", "")) for v in self.surface_variables if v.startswith("C")]

        # Load ABI channel-wise mean/std scaler
        # DIAG: .load() pulls the entire file into memory — if this is called
        # in every DataLoader worker (256 workers) simultaneously it saturates
        # Lustre metadata bandwidth. Timing here will expose that.
        base_scaler_path = data_conf.get("goes_scaler_path", scaler_ds_path)
        _t = time.time()
        _diag(f"loading scaler_ds from {base_scaler_path}")
        self.scaler_ds = xr.open_dataset(base_scaler_path).load()
        _diag(f"scaler_ds loaded in {time.time()-_t:.2f}s")

        # Optionally apply log-normal scaling to visible/near-IR channels (C01-C06)
        self.log_normal_scaling = data_conf.get("log_normal_scaling", False)
        if self.log_normal_scaling:
            log_scaler_path = data_conf.get(
                "goes_log_scaler_path",
                "/glade/derecho/scratch/bagherio/cloud.dir/datasets/goes_cloud_dataset/goes_stats_vis_log.nc"
            )
            logger.info("log normalizing visible channels (C01-C06)")
            # DIAG: second netCDF load — same Lustre contention concern as above
            _t = time.time()
            _diag(f"loading log_scaler_ds from {log_scaler_path}")
            self.log_scaler_ds = xr.open_dataset(log_scaler_path).load()
            _diag(f"log_scaler_ds loaded in {time.time()-_t:.2f}s")

        # ------------------------------------------------------------------ #
        # OPTION 1: On-the-fly temporal forcing
        #
        # Previous approach: pre-computed zarr store on disk (temporal_forcing_10m.zarr)
        #   - Required t=1 chunking → millions of files → exceeded inode quota
        #   - Required careful time alignment to handle GOES data gaps
        #
        # New approach: compute sin/cos julian day + sin/cos local time on-the-fly
        #   - Pure numpy math, ~microseconds per call, zero disk I/O
        #   - Automatically gap-safe: only called for valid timestamps
        #   - Toggled via `use_temporal_forcing: True/False` in the YAML config
        #     so ablation runs (with vs. without temporal forcing) require only
        #     a config change, not a code change
        #
        # NOTE: when use_temporal_forcing is False, surface_channels in model.yml
        #       must be reduced from 35 to 31 (remove the 4 temporal channels)
        # ------------------------------------------------------------------ #
        self.use_temporal_forcing = data_conf.get("use_temporal_forcing", False)
        if self.use_temporal_forcing:
            # ------------------------------------------------------------------ #
            # GRID CACHE: lon_vals and lat_size are loaded from pre-saved .npy files
            # stored in valid_init_dir rather than read directly from the Zarr store.
            #
            # WHY: calling ds["longitude"].values in __init__ forces an actual Zarr
            # read on every DataLoader worker at startup. With 32 ranks × 8 workers
            # = 256 processes all hitting the same Zarr store simultaneously on
            # Lustre, this caused initialization to balloon from seconds to 60+ minutes.
            #
            # The .npy files are plain flat binary — Lustre handles many concurrent
            # reads of small static files orders of magnitude faster than Zarr metadata
            # + chunk reads. The files are generated once on first run (rank 0 / first
            # worker) and reused by every subsequent worker and run.
            #
            # With the lazy-open refactor, _meta_ds (opened with chunks=None in the
            # main process) is used as the data source if the cache files are missing,
            # rather than reading from a forked Zarr handle inside a worker process.
            # ------------------------------------------------------------------ #
            lon_cache_path = join(self.valid_init_dir, "lon_vals.npy")
            lat_cache_path = join(self.valid_init_dir, "lat_size.npy")

            _t = time.time()
            _diag(f"grid cache — checking {lon_cache_path}")

            if not os.path.exists(lon_cache_path) or not os.path.exists(lat_cache_path):
                # First time only: read from _meta_ds (already open, chunks=None, in
                # the main process) and save as lightweight .npy files. This will happen
                # at most once per cluster environment — all subsequent workers skip this
                # branch entirely and go straight to np.load below.
                logger.info(
                    "grid cache not found — reading lon/lat from metadata dataset and "
                    f"saving to {self.valid_init_dir}. This only happens on first run."
                )
                _diag("grid cache MISS — reading lon/lat from _meta_ds (slow path, main process only)")
                os.makedirs(self.valid_init_dir, exist_ok=True)
                lon_vals = _meta_ds["longitude"].values.astype(np.float32)
                lat_size = np.array([len(_meta_ds["latitude"])], dtype=np.int64)
                np.save(lon_cache_path, lon_vals)
                np.save(lat_cache_path, lat_size)
                logger.info(f"saved lon_vals.npy ({lon_vals.shape}) and lat_size.npy to {self.valid_init_dir}")
            else:
                _diag("grid cache HIT — loading from .npy (fast path)")

            # Fast path: load from flat .npy — no Zarr reads, no metadata overhead.
            # All 256 workers hit this branch on every run after the first.
            self.lon_vals = np.load(lon_cache_path)        # shape: (lon,), float32
            self.lat_size = int(np.load(lat_cache_path)[0])  # scalar int
            _diag(f"grid cache loaded in {time.time()-_t:.2f}s  lon_shape={self.lon_vals.shape} lat_size={self.lat_size}")
            logger.info(
                "temporal forcing ENABLED — computing sin/cos julian day + "
                "sin/cos local time on-the-fly (no disk reads)"
            )
        else:
            self.lon_vals = None
            self.lat_size = None
            logger.info("temporal forcing DISABLED")
        # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
        # DISCARD METADATA DATASET
        #
        # _meta_ds has served its purpose (init_times, grid cache). Delete it
        # explicitly so it is not present in the object when PyTorch forks
        # DataLoader workers. Even though it was opened with chunks=None (no
        # Dask), discarding it avoids carrying an unnecessary open file handle
        # across the fork boundary.
        # ------------------------------------------------------------------ #
        del _meta_ds
        _diag("_meta_ds deleted — workers will open their own Zarr handle in __getitem__")

        # setup rollout mode if configured
        if not evaluate and "rollout_init_times" in time_config.keys():
            logger.info("setting up GOES 10km dataset for rollout mode by subsetting times")
            self._rollout_mode(
                time_config["rollout_init_times"],
                time_config.get("time_tol", (1, "D")),
            )

        # ------------------------------------------------------------------ #
        # DIAG: total __init__ wall time. If this appears with worker=N it means
        # DataLoader workers are re-running __init__, multiplying all I/O above
        # by the worker count. In that case the fix is persistent_workers=True
        # or reducing num_workers to limit concurrent Lustre load.
        # ------------------------------------------------------------------ #
        _diag(f"__init__ COMPLETE — total wall time {time.time()-_t_init_start:.2f}s")

    # ------------------------------------------------------------------ #
    # NEW METHOD: _compute_temporal_forcing
    # ------------------------------------------------------------------ #
    def _compute_temporal_forcing(self, ts: pd.Timestamp) -> np.ndarray:
        """
        Compute the four cyclic temporal forcing channels on-the-fly from a
        single pd.Timestamp. No file I/O is performed.

        Channel definitions (matching the original pre-computed zarr variables):
            0: sin_julian_day  — encodes seasonal cycle (day-of-year)
            1: cos_julian_day  — encodes seasonal cycle (day-of-year)
            2: sin_local_time  — encodes diurnal cycle (UTC + longitude offset)
            3: cos_local_time  — encodes diurnal cycle (UTC + longitude offset)

        Spatial broadcasting strategy:
            - sin/cos_julian_day depend on date only → scalar, broadcast to full (lat, lon)
            - sin/cos_local_time depend on time + longitude only → shape (lon,),
              then tiled across latitude to (lat, lon)

        Gap safety:
            This method is only ever called from get_era5(), which is only called
            from __getitem__(), which is only called for timestamps that have
            already passed the valid_init_times filter. GOES data gaps are
            therefore handled automatically — there is no index to misalign.

        Args:
            ts: pd.Timestamp for the current sample (exact GOES time, not rounded)

        Returns:
            np.ndarray of shape (4, 1, lat, lon), dtype float32
            The time dimension of size 1 matches the (C, T, H, W) convention
            used by the rest of the dataset and trainer.
        """
        n_lon = len(self.lon_vals)
        n_lat = self.lat_size

        # --- A. Julian Day Forcings (Seasonal) ----------------------------
        # Map day-of-year to [0, 2π] and compute sin/cos.
        # Using 365.25 (not 365) to account for leap years consistently.
        doy_angle = (ts.day_of_year / 365.25) * 2.0 * np.pi

        # sin/cos_julian_day are spatially uniform — broadcast scalar to full grid
        sin_jd = np.full((n_lat, n_lon), np.float32(np.sin(doy_angle)), dtype=np.float32)
        cos_jd = np.full((n_lat, n_lon), np.float32(np.cos(doy_angle)), dtype=np.float32)

        # --- B. Local Time Forcings (Diurnal) -----------------------------
        # Compute UTC fractional hour from the exact 10-minute timestamp
        utc_hour = ts.hour + ts.minute / 60.0

        # Convert each longitude to a local solar time offset (hours).
        # lon / 15.0 gives hours offset from UTC (15° per hour).
        local_time = (utc_hour + self.lon_vals / 15.0) % 24.0   # shape: (lon,)

        # Map local time to [0, 2π] and compute sin/cos
        local_angle = (local_time / 24.0) * 2.0 * np.pi
        sin_lt = np.sin(local_angle).astype(np.float32)           # shape: (lon,)
        cos_lt = np.cos(local_angle).astype(np.float32)           # shape: (lon,)

        # Tile along latitude axis to produce (lat, lon).
        # np.tile creates a contiguous copy — required for torch.from_numpy.
        # (np.broadcast_to would be read-only and cause a torch error.)
        sin_lt_2d = np.tile(sin_lt[np.newaxis, :], (n_lat, 1))   # shape: (lat, lon)
        cos_lt_2d = np.tile(cos_lt[np.newaxis, :], (n_lat, 1))   # shape: (lat, lon)

        # --- C. Stack and add time dimension ------------------------------
        # Stack to (4, lat, lon), then insert time dim to get (4, 1, lat, lon).
        # This matches the (C, T, H, W) shape convention used by the trainer
        # when it concatenates temporal forcing with the main GOES input.
        tf = np.stack([sin_jd, cos_jd, sin_lt_2d, cos_lt_2d], axis=0)
        return tf[:, np.newaxis, :, :]  # shape: (4, 1, lat, lon)
    # ------------------------------------------------------------------ #

    def _rollout_mode(self, rollout_init_times, time_tol):
        logger.info(f"selecting rollout times with time tolerance {time_tol}")
        self.init_times = self.init_times.sel(t=rollout_init_times, method="nearest",
                                              tolerance=pd.Timedelta(*time_tol))

    def _generate_valid_init_times(self, valid_init_filepath, ds: xr.Dataset):
        # due to missing data, need to have a different list of valid init times
        # ds is passed explicitly (from _meta_ds in __init__) so this method
        # never reads from self.ds, which is None until worker-side lazy open.
        def check_valid_forecast_times(t_init, timestep, num_forecast_steps):
            time_tolerance = pd.Timedelta(3, "m")
            target_times = [t_init.values + timestep * step for step in range(1, num_forecast_steps + 1)]
            zarr_times = ds.t.sel(t=target_times, method="nearest")
            within_tol = np.abs(zarr_times.values - np.array(target_times).astype(zarr_times.dtype)) < time_tolerance
            return all(within_tol)

        logger.info(f"generating valid init times and saving to {valid_init_filepath}. will take around 5 min")
        # DIAG: log progress every 10 000 timestamps so we can tell this loop is alive
        _t_gen = time.time()
        _diag("_generate_valid_init_times loop START")

        valid_times = []
        for i, t in enumerate(ds.t):
            if i % 10000 == 0 and i > 0:
                _diag(f"_generate_valid_init_times progress: {i} timestamps checked, {len(valid_times)} valid so far, elapsed {time.time()-_t_gen:.0f}s")
            if check_valid_forecast_times(t, self.timestep, self.num_forecast_steps):
                valid_times.append(t.values)

        _diag(f"_generate_valid_init_times loop done in {time.time()-_t_gen:.2f}s  n_valid={len(valid_times)}")

        valid_times_da = xr.DataArray(valid_times, coords={"t": valid_times})

        # Ensure your nan_times.nc has been generated for the new dataset
        nan_file = join(self.valid_init_dir, "nan_times.nc")
        if os.path.exists(nan_file):
            times_to_drop = xr.open_dataarray(nan_file)
            valid_times_da = valid_times_da[~valid_times_da.t.isin(times_to_drop)]

        os.makedirs(self.valid_init_dir, exist_ok=True)
        valid_times_da.to_netcdf(valid_init_filepath)
        logger.info(f"wrote valid init times to {valid_init_filepath}")

        return valid_times_da

    def _timestamps(self, ds: xr.Dataset):
        # grab or compute valid init times across whole dataset, due to missing data
        # this removes need to mess with the samplers.
        # ds is passed explicitly (from _meta_ds in __init__) so this method
        # never reads from self.ds, which is None until worker-side lazy open.
        if self.timestep.seconds >= 3600:
            filename = f"{self.timestep.seconds // 3600:02d}h_{self.num_forecast_steps:02}step.nc"
        else:
            filename = f"{self.timestep.seconds // 60:02d}m_{self.num_forecast_steps:02}step.nc"
        valid_init_filepath = join(self.valid_init_dir, filename)

        # DIAG: distinguishes fast cache-hit path from slow generation path
        if os.path.exists(valid_init_filepath):
            _diag(f"_timestamps: loading cached file {valid_init_filepath}")
            _t = time.time()
            timestamps = xr.open_dataarray(valid_init_filepath)
            _diag(f"_timestamps: cached file loaded in {time.time()-_t:.2f}s")
        else:
            _diag(f"_timestamps: cache MISS — generating {valid_init_filepath} (slow ~5 min)")
            timestamps = self._generate_valid_init_times(valid_init_filepath, ds)

        logger.debug(f"loading date range {self.start_datetime} to {self.end_datetime}")
        timestamps = timestamps.sel(
            t=(timestamps.t >= self.start_datetime) & (timestamps.t < self.end_datetime)
        )

        return timestamps

    def __len__(self):
        # total number of valid start times
        return len(self.init_times)

    def inverse_transform_ABI(self, da):
        # da must be in the same order as the source data
        # channels must be the first dimension

        # 1. Build a dynamic scaler based on the incoming channels
        active_mean = self.scaler_ds["mean"].sel(channel=da.channel).copy()
        active_std = self.scaler_ds["std"].sel(channel=da.channel).copy()

        # FIX: Log-normal scaling now applies dynamically
        if self.log_normal_scaling:
            for ch in da.channel.values:
                if ch <= 6:
                    active_mean.loc[dict(channel=ch)] = self.log_scaler_ds["mean"].sel(channel=ch)
                    active_std.loc[dict(channel=ch)] = self.log_scaler_ds["std"].sel(channel=ch)

        # FIX: Reshape the 1D stats to broadcast over (C, H, W)
        mean_vals = active_mean.values[:, None, None]
        std_vals = active_std.values[:, None, None]

        unscaled = da * std_vals + mean_vals

        # FIX: Apply log norm dynamically
        if self.log_normal_scaling:
            for i, ch in enumerate(da.channel.values):
                if ch <= 6:
                    unscaled[i] = np.exp(unscaled[i])

        return unscaled

    def _normalize_ABI(self, da):
        """
        Normalizes the ABI data using the scaler dataset.
        da: xarray.DataArray of shape (channel, latitude, longitude)
        """
        # 1. Align the stats to the specific channels present in 'da'
        # This uses Xarray's label-based indexing to match channels by name/ID
        active_mean = self.scaler_ds["mean"].sel(channel=da.channel).copy()
        active_std = self.scaler_ds["std"].sel(channel=da.channel).copy()

        if self.log_normal_scaling:
            for ch in da.channel.values:
                if ch <= 6:
                    active_mean.loc[dict(channel=ch)] = self.log_scaler_ds["mean"].sel(channel=ch)
                    active_std.loc[dict(channel=ch)] = self.log_scaler_ds["std"].sel(channel=ch)

        # 2. Reshape for broadcasting
        # The values are pulled from the aligned Xarray objects as numpy arrays
        mean_vals = active_mean.values[:, None, None]
        std_vals = active_std.values[:, None, None]

        # 3. Normalize
        return (da - mean_vals) / std_vals

    def _nanfill_ABI(self, da):
        if self.evaluate:
            nan_fraction = da.isnull().mean()
            return da.fillna(0.0) if nan_fraction <= 0.1 else da
        else:
            return da.fillna(0.0)

    def __getitem__(self, args):
        # ------------------------------------------------------------------ #
        # LAZY ZARR OPEN (worker-side initialization)
        #
        # self.ds is None until the first __getitem__ call in this worker.
        # Opening here rather than in __init__ ensures each DataLoader worker
        # gets its own independent Zarr file handle without inheriting a
        # Dask-backed dataset across the fork boundary.
        #
        # chunks=None: opens the store with synchronous NumPy-backed I/O —
        # no Dask thread pool, no risk of thread contention on Lustre.
        # drop_duplicates + sortby mirrors the transformation applied to
        # _meta_ds in __init__ so time-based selection behaves identically.
        #
        # With persistent_workers=True in the DataLoader this open happens
        # exactly once per worker for the entire training run, making the
        # per-sample overhead effectively zero after the first call.
        # ------------------------------------------------------------------ #
        if self.ds is None:
            _diag(f"lazy open: opening {self.ds_path} with chunks=None in this worker")
            _t = time.time()
            self.ds = xr.open_zarr(self.ds_path, chunks=None)
            self.ds = self.ds.drop_duplicates(dim="t").sortby("t")
            _diag(f"lazy open: done in {time.time()-_t:.2f}s")

        ts, mode = args

        # TODO: rollout needs to pick from valid_init_times
        ds = self.ds.sel(t=ts, method="nearest")
        # no need to check time tolerance, should be taken care of by init time generation

        time_str = pd.Timestamp(ds.t.values).strftime("%Y-%m-%dT%H:%M:%S")

        return_data = {
            "mode": mode,
            "stop_forecast": mode == "stop",
            "datetime": time_str,
        }

        if mode != "forcing":  # draw goes if mode is not forcing
            # DYNAMIC FIX: Slice ONLY the channels requested in config
            da = ds["BT_or_R"].sel(channel=self.active_channels).copy()

            # FIX: Apply log norm to the visible reflective channels present
            if self.log_normal_scaling:  # channels is the first axis
                for i, ch in enumerate(da.channel.values):
                    if ch <= 6:
                        da[i] = np.log(da[i] + 1e-6)  # Added tiny epsilon to prevent log(0) errors

            da = self._normalize_ABI(da)
            da = self._nanfill_ABI(da)

            if not self.evaluate:
                # da.shape = c, 1003, 923
                data = torch.tensor(da.values).unsqueeze(1)
            else:  # return a dataset after nanfilling
                data = self.inverse_transform_ABI(da)

            if self.padding:
                data = torch.nn.functional.pad(data, (19, 18, 11, 10), "constant", 0.0)
            # coerce to c, t, 1024, 960 to work with wxformer

        if self.era5dataset and mode != "stop":  # always draw era5 if not stopping
            return_data["era5"] = self.get_era5(ds)

        if mode == "init":
            return_data["x"] = data
            return return_data
        elif mode == "forcing":
            return return_data
        elif mode == "y" or mode == "stop":
            return_data["y"] = data
            return return_data
        else:
            raise ValueError(f"{mode} is not a valid sampling mode")

    def get_era5(self, ds):
        ts = pd.Timestamp(ds.t.values)
        era5_ts = ts.round("h")  # round to the nearest hour

        # Load standard hourly ERA5 fields
        era5_data = self.era5dataset[(era5_ts, "init")]
        era5_data["timedelta_seconds"] = int((era5_ts - ts).total_seconds())

        # ------------------------------------------------------------------ #
        # OPTION 1: On-the-fly temporal forcing
        #
        # Replaces the old block that read from the pre-computed zarr store:
        #   tf_slice = self.temporal_ds.sel(t=ts, method="nearest")
        #   tf_vars  = [tf_slice[var].values for var in self.temporal_vars]
        #   tf_tensor = torch.tensor(np.stack(tf_vars, axis=0)).unsqueeze(1)
        #
        # The new call is equivalent but:
        #   - Uses no disk I/O (pure numpy trig math)
        #   - Is gap-safe (ts is always a valid GOES time at this point)
        #   - Is controlled by use_temporal_forcing in the YAML config
        #
        # Output key and tensor shape are IDENTICAL to the old implementation:
        #   era5_data["temporal_forcing_10m"] — torch.Tensor (4, 1, lat, lon)
        # The trainer and dataloader are completely unaware of this change.
        # ------------------------------------------------------------------ #
        if self.use_temporal_forcing:
            tf_array = self._compute_temporal_forcing(ts)           # (4, 1, lat, lon) float32
            era5_data["temporal_forcing_10m"] = torch.from_numpy(tf_array)
        # ------------------------------------------------------------------ #

        return era5_data


# class ERA5Interpolator:
#     def __init__(self,
#                  data_conf: Dict,
#                  era5dataset: Dataset = None,):
#         """
#         taking advantage of DistributedSampler class code with this dataset
#
#         Args:
#             ds: xr dataset with a time attribute
#
#             example time config:
#             {"timestep": pd.Timedelta(1, "h"),
#                 "num_forecast_steps": 1
#                  },
#         """
#         self.era5dataset = era5dataset
#         # setup regridder
#         self.regridder = None
#         if era5dataset and data_conf.get("regrid_loc", None):
#             regrid_loc = data_conf["regrid_loc"]
#             da_outgrid = xr.open_dataset(data_conf["outgrid_loc"], engine="h5netcdf")
#             ds_ingrid = xr.open_dataset(data_conf["ingrid_loc"], engine="h5netcdf")
#             self.regridder = xe.Regridder(ds_ingrid, da_outgrid, 'bilinear', unmapped_to_nan=True, weights=regrid_loc)
#
#     def __getitem__(self, args):
#         # default: load target state
#         ts, mode = args
#
#         # run interpolation
#         era5_ds_dict = self.era5dataset[(ts, "y_xarray")] #draw an xarray
#         field_types = ["prognostic", "dynamic_forcing"]
#         combined_ds = xr.merge([era5_ds_dict[field] for field in field_types])
#         regridded = self.regridder(combined_ds, skipna=True, na_thres=1.0)
#
#         return regridded
#         # ts = pd.Timestamp(combined_ds.time.values)
#         # save_dir = os.path.join("/glade/derecho/scratch/dkimpara/goes-cloud-dataset/era5_regrid/",
#         #                          str(ts.year))
#         # os.makedirs(save_dir, exist_ok=True)
#         # regridded.to_netcdf(os.path.join(save_dir, ts.strftime("%Y-%m-%dT%H:%M:%S")), engine="h5netcdf")