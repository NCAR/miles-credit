"""
camulator_server.py
-------------------
Python inference server that couples CAMulator (AI atmosphere) to CESM2/POP
via a file-based flag protocol.

Protocol (every 6-hour coupling step):
  CESM/DATM writes:  camulator_sst_in.nc  + camulator_go.flag
  Server reads:      sst_in.nc  → remaps T62→CAM grid → injects SST → runs model
  Server writes:     camulator_cam_out.nc + camulator_done.flag
  CESM/DATM reads:   cam_out.nc → populates a2x fields → sends to POP/CICE

Grids:
  T62  (CESM DATM): 94 lat × 192 lon = 18,048 points (Gaussian, flat 1-D in NC)
  CAMulator:       192 lat × 288 lon = 55,296 points (1° regular lat/lon)

Usage:
  conda activate /glade/work/wchapman/conda-envs/credit-coupling
  cd /glade/work/wchapman/Roman_Coupling/credit_feb182026/climate
  python camulator_server.py \\
      --config     ./camulator_config.yml \\
      --model_name checkpoint.pt00091.pt \\
      --rundir     /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v01/run/ \\
      --init_cond  /path/to/init_tensor.pth


qsub -I -A NAML0001 -q casper -l job_priority=premium -l walltime=12:00:00 -l select=1:ncpus=32:mpiprocs=1:ompthreads=1:mem=100gb:ngpus=1:gpu_type=a100_80gb

python camulator_server.py --config ./camulator_config.yml --model_name checkpoint.pt00091.pt --rundir /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_SBV01/run/ --save_atm_nc camulator_out --daily_mean
See climate/README_Coupling.md for full documentation.
"""

import os
import sys
import time
import shutil
import argparse
import logging
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import netCDF4 as nc
import yaml
from scipy.special import roots_legendre

from credit.output import make_xarray
from Model_State import initialize_camulator, StateVariableAccessor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

GO_FLAG      = "camulator_go.flag"
DONE_FLAG    = "camulator_done.flag"
SST_FILE     = "camulator_sst_in.nc"
CAM_FILE     = "camulator_cam_out.nc"
ATM_RESTART  = "camulator_atm_restart.pth"   # atmosphere restart: state + timestep
READY_FLAG   = "camulator_server_ready.flag"  # written when server is fully initialised

POLL_SLEEP = 0.1                        # seconds between go.flag polls (50 ms — was 500 ms)
POLL_MAX   = int(3 * 3600 / POLL_SLEEP)  # 3-hour timeout = 216,000 polls at 50 ms each
POLL_LOG   = int(60   / POLL_SLEEP)      # log a waiting message every ~60 s = 1200 polls

# T62 Gaussian grid dimensions (standard CESM DATM T62 forcing grid)
T62_NLAT  = 94
T62_NLON  = 192
T62_NGRID = T62_NLAT * T62_NLON   # 18,048

# CAMulator native grid dimensions (from camulator_config.yml)
CAM_NLAT = 192
CAM_NLON = 288

# 6-hour timestep in seconds (CAMulator model step)
DT_SEC = 21600.0

# SST fill value used by CAMulator over land grid cells in training data.
# T62 land points carry So_t = 0 K (no ocean data from POP).  Sea ice in POP
# keeps SST ≥ ~271 K (saltwater freezing point), so anything below OCEAN_MIN_K
# is either a land point or an initialisation step where POP has not yet
# produced output.  We replace those with LAND_SST_FILL before normalisation
# so the model always sees values within its training distribution.
OCEAN_MIN_K   = 270.0   # K  — safe lower bound for open ocean / sea-ice SST
LAND_SST_FILL = 283.0   # K  — training-data fill value for continental points


# =============================================================================
# Grid helpers
# =============================================================================

def t62_latlons():
    """
    Return T62 Gaussian grid latitude and longitude arrays.

    Latitudes are the 94 Gauss-Legendre quadrature points, ascending S→N.
    Longitudes are 192 evenly-spaced points starting at 0°.
    """
    mu, _ = roots_legendre(T62_NLAT)
    lats = np.degrees(np.arcsin(mu))          # shape (94,), ascending S→N
    lons = np.linspace(0.0, 360.0, T62_NLON, endpoint=False)  # shape (192,)
    return lats, lons


class BilinearRemap:
    """
    Fast bilinear remap between two regular lat/lon grids.

    Precomputes the 4-corner indices and weights at construction time from the
    fixed grid geometry.  Each subsequent __call__ or batch() is a handful of
    pure-NumPy array lookups — roughly 100–200× faster than constructing a
    new RegularGridInterpolator on every step.

    Both src_lats / dst_lats must be ascending (S→N).
    Handles 360°/0° longitude wrap-around.
    """

    def __init__(self, src_lats, src_lons, dst_lats, dst_lons):
        nlat_src = len(src_lats)
        nlon_src = len(src_lons)

        # Build flat destination mesh ----------------------------------------
        lat_g, lon_g = np.meshgrid(dst_lats, dst_lons, indexing="ij")  # (nlat_dst, nlon_dst)
        flat_lats = lat_g.ravel()   # (ndst,)
        flat_lons = lon_g.ravel() % 360.0   # normalise to [0, 360)

        # ---- row indices (latitude direction) --------------------------------
        i0 = np.clip(
            np.searchsorted(src_lats, flat_lats, side="right") - 1,
            0, nlat_src - 2
        )
        i1 = i0 + 1

        # ---- col indices (longitude direction, with wrap-around) ------------
        j0 = np.clip(
            np.searchsorted(src_lons, flat_lons, side="right") - 1,
            0, nlon_src - 1
        )
        j1 = (j0 + 1) % nlon_src   # wraps last column back to 0

        # right-edge longitude: handle the wrap (j0 == nlon-1 → j1=0 means +360)
        lon_right = np.where(j0 < nlon_src - 1, src_lons[j1], src_lons[0] + 360.0)

        # ---- bilinear fractions ---------------------------------------------
        dlat = src_lats[i1] - src_lats[i0]
        dlon = lon_right - src_lons[j0]
        a = np.clip((flat_lats - src_lats[i0]) / np.where(dlat == 0, 1.0, dlat), 0.0, 1.0)
        b = np.clip((flat_lons  - src_lons[j0]) / np.where(dlon == 0, 1.0, dlon), 0.0, 1.0)

        # Store as float32 to halve memory; broadcast with input data fine
        self.i0 = i0;  self.i1 = i1
        self.j0 = j0;  self.j1 = j1
        self.w00 = ((1 - a) * (1 - b)).astype(np.float32)   # (ndst,)
        self.w01 = ((1 - a) * b      ).astype(np.float32)
        self.w10 = (a       * (1 - b)).astype(np.float32)
        self.w11 = (a       * b      ).astype(np.float32)
        self.shape_out = (len(dst_lats), len(dst_lons))
        self.ndst      = len(flat_lats)

    def __call__(self, field):
        """Remap a single 2-D field (nlat_src, nlon_src) → (nlat_dst, nlon_dst)."""
        return (
            self.w00 * field[self.i0, self.j0]
          + self.w01 * field[self.i0, self.j1]
          + self.w10 * field[self.i1, self.j0]
          + self.w11 * field[self.i1, self.j1]
        ).reshape(self.shape_out)

    def batch(self, fields):
        """
        Remap N fields in one vectorised call.

        Args:
            fields : np.ndarray, shape (N, nlat_src, nlon_src)
        Returns:
            np.ndarray, shape (N, ndst)  — flat destination layout
        """
        # fields[:, i, j] with 1-D fancy indices i, j gives shape (N, ndst)
        return (
            self.w00 * fields[:, self.i0, self.j0]
          + self.w01 * fields[:, self.i0, self.j1]
          + self.w10 * fields[:, self.i1, self.j0]
          + self.w11 * fields[:, self.i1, self.j1]
        )


# =============================================================================
# NetCDF I/O
# =============================================================================

def read_sst_nc(path):
    """
    Read camulator_sst_in.nc written by the Fortran DATM.

    Returns:
        sst   : np.float64 array, shape (T62_NGRID,)  [K]
        ifrac : np.float64 array, shape (T62_NGRID,)  [0-1]
        ymd   : int  YYYYMMDD
        tod   : int  seconds since midnight
    """
    with nc.Dataset(str(path), "r") as ds:
        sst   = ds["sst"][:].data.astype(np.float64)
        ifrac = ds["ifrac"][:].data.astype(np.float64)
        ymd   = int(ds["ymd"][:])
        tod   = int(ds["tod"][:])
    return sst, ifrac, ymd, tod


def write_cam_nc(path, u10, v10, tbot, zbot, tref, qbot, pbot, fsds, flnsd, prect):
    """
    Write camulator_cam_out.nc for the Fortran DATM to read.

    All fields are 1-D float64 arrays of length T62_NGRID (18,048).
    Variable names and units must match what datm_datamode_camulator.F90 expects.

    tbot  : temperature at the CAM6 L32 bottom model level (Sa_tbot input to bulk formula)
    zbot  : height of the CAM6 L32 bottom model level midpoint (Sa_z, dynamic ~50-67 m)
    tref  : TREFHT 2 m diagnostic temperature (stored for verification; not wired to any
            DATM a2x field yet — use tbot/zbot for the actual bulk formula inputs)
    fsds  : downwelling SW at surface [W m-2], reconstructed from FSNS via
            FSDS = FSNS / (1 - alpha_sfc) where alpha_sfc uses CICE ice fraction.
            This is the quantity the CPL7 coupler expects for Faxa_sw* — it applies
            ocean/ice albedo internally (seq_flux_mct.F90) to derive absorbed SW.
    """
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("ngrid", T62_NGRID)

        def mkvar(name, data, units, long_name):
            v = ds.createVariable(name, "f8", ("ngrid",))
            v[:] = data
            v.units = units
            v.long_name = long_name

        mkvar("u10",   u10,   "m s-1",  "Zonal wind at CAMulator bottom model level (~60 m)")
        mkvar("v10",   v10,   "m s-1",  "Meridional wind at CAMulator bottom model level (~60 m)")
        mkvar("tbot",  tbot,  "K",      "Temperature at CAMulator bottom model level (Sa_tbot)")
        mkvar("zbot",  zbot,  "m",      "Height of bottom model level midpoint (Sa_z, dynamic ~50-67 m)")
        mkvar("tref",  tref,  "K",      "TREFHT 2 m reference temperature (diagnostic only)")
        mkvar("qbot",  qbot,  "kg kg-1","Specific humidity at CAMulator bottom model level (Sa_shum)")
        mkvar("pbot",  pbot,  "Pa",     "Surface pressure (PS → Sa_pbot)")
        mkvar("fsds",  fsds,  "W m-2",  "Downwelling SW at surface (reconstructed from FSNS via ice-fraction albedo)")
        mkvar("flnsd", flnsd, "W m-2",  "Downward LW at surface")
        mkvar("prect", prect, "m s-1",  "Total precip, liquid-water equivalent")


def cesm_ymd_tod_to_dt(ymd, tod):
    """CESM ymd (YYYYMMDD, model year) + tod (seconds) → datetime."""
    y, m, d = ymd // 10000, (ymd % 10000) // 100, ymd % 100
    return datetime(y, m, d) + timedelta(seconds=int(tod))


def write_flag(path):
    Path(path).touch()


def delete_flag(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# =============================================================================
# Atmospheric NetCDF output helpers  (top-level so mp.Pool can pickle them)
# =============================================================================

def save_camulator_step_nc(upper_air, single_level, filepath):
    """
    Write one 6-hourly prediction step to NetCDF.  Runs in a worker process
    via pool.apply_async so disk I/O is hidden behind CESM's inter-step compute.

    Args:
        upper_air    : xr.DataArray  (time, vars, level, lat, lon) — physical units
        single_level : xr.DataArray  (time, vars, lat, lon)        — physical units
        filepath     : str  destination path (parent dirs created if needed)
    """
    import os
    import xarray as xr
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ds = xr.merge([upper_air.to_dataset(dim="vars"),
                   single_level.to_dataset(dim="vars")])
    ds.to_netcdf(filepath, mode="w")


def save_camulator_daily_nc(buffer, filepath):
    """
    Average N 6-hourly step datasets and write one daily-mean NetCDF.
    Runs in a worker process via pool.apply_async.

    Args:
        buffer   : list of (upper_air, single_level) xr.DataArray tuples
        filepath : str  destination path (parent dirs created if needed)
    """
    import os
    import xarray as xr
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    datasets = [xr.merge([ua.to_dataset(dim="vars"), sl.to_dataset(dim="vars")])
                for ua, sl in buffer]
    ds_daily = xr.concat(datasets, dim="time").mean("time", keep_attrs=True)
    ds_daily.to_netcdf(filepath, mode="w")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CAMulator inference server for CESM2 air-sea coupling"
    )
    p.add_argument("--config",     required=True,
                   help="Path to camulator_config.yml")
    p.add_argument("--model_name", required=True,
                   help="Checkpoint filename, e.g. checkpoint.pt00091.pt")
    p.add_argument("--rundir",     required=True,
                   help="CESM run directory (where go/done flag files live)")
    p.add_argument("--init_cond",  default=None,
                   help="Override init condition .pth path (else uses config value)")
    p.add_argument("--device",     default="cuda",
                   help="Torch device: cuda or cpu")
    p.add_argument("--save_atm_nc", default=None, metavar="SUBDIR",
                   help="Subdirectory under --rundir for atmospheric NetCDF output. "
                        "The value is appended to rundir, e.g. --save_atm_nc my_run_001 "
                        "→ <rundir>/my_run_001/. Omit to disable NC output.")
    p.add_argument("--daily_mean", action="store_true",
                   help="Also save daily-mean NetCDF files alongside the 6-hourly output "
                        "(<DIR>/<YYYY>/camulator.h1d.<YYYY-MM-DD>.nc). "
                        "Requires --save_atm_nc.")
    p.add_argument("--flnsd_diag", action="store_true",
                   help="Log area-weighted FLNSD decomposition each step (TS / TREFHT / Tbot "
                        "SB-term comparison). Useful for diagnostics but adds ~2ms per step.")
    p.add_argument("--save_vars", nargs="+", default=None, metavar="VAR",
                   help="Subset of variables to write to --save_atm_nc output. "
                        "Names must match entries in conf[data][variables], "
                        "surface_variables, or diagnostic_variables. "
                        "Example: --save_vars U V T PS PRECT TREFHT TS "
                        "Default: all variables.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    rundir = Path(args.rundir)
    if not rundir.is_dir():
        logger.error(f"rundir does not exist: {rundir}")
        sys.exit(1)

    go_flag   = rundir / GO_FLAG
    done_flag = rundir / DONE_FLAG
    sst_file  = rundir / SST_FILE
    cam_file  = rundir / CAM_FILE

    # Annual archive directory: one restart per model year saved here.
    # Named camulator_atm_restart.year000N.pth — useful for ./xmlchange STOP_OPTION=nyears runs.
    _atm_restart_archive_dir = rundir / "atm_restarts"
    _atm_restart_archive_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Initialize CAMulator
    #    If --init_cond is given, patch the config temporarily so
    #    initialize_camulator picks up the right initial condition file.
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("CAMulator coupling server starting")
    logger.info(f"  rundir     : {rundir}")
    logger.info(f"  model_name : {args.model_name}")
    logger.info(f"  device     : {args.device}")
    logger.info("=" * 65)

    config_path = args.config
    _tmp_config = None

    if args.init_cond:
        logger.info(f"Overriding init_cond_fast_climate → {args.init_cond}")
        with open(args.config) as f:
            raw_conf = yaml.safe_load(f)
        raw_conf["predict"]["init_cond_fast_climate"] = args.init_cond
        # Write to a temp file alongside the original config
        _tmp_config = str(Path(args.config).parent / "_tmp_server_config.yml")
        with open(_tmp_config, "w") as f:
            yaml.dump(raw_conf, f)
        config_path = _tmp_config

    ctx = initialize_camulator(config_path, model_name=args.model_name, device=args.device)

    if _tmp_config and os.path.exists(_tmp_config):
        os.remove(_tmp_config)

    stepper           = ctx["stepper"]
    state             = ctx["initial_state"]
    state_transformer = ctx["state_transformer"]
    forcing_ds_norm   = ctx["forcing_dataset"]
    static_forcing    = ctx["static_forcing"]
    conf              = ctx["conf"]
    device            = ctx["device"]
    latlons           = ctx["latlons"]

    accessor_input  = StateVariableAccessor(conf, tensor_type="input")
    accessor_output = StateVariableAccessor(conf, tensor_type="output")

    # ------------------------------------------------------------------
    # 1b. Check for a CAMulator atmosphere restart file.
    #     On a CESM restart, POP/CICE resume from their own restart files
    #     while this server would otherwise restart from the original IC,
    #     creating an atmosphere/ocean inconsistency.  If a restart file
    #     exists in rundir, load the saved state and timestep instead.
    # ------------------------------------------------------------------
    atm_restart_file = rundir / ATM_RESTART
    timestep_init = 0
    _expected_first_ymd = -1   # set below if restart found; checked on first go.flag
    _expected_first_tod = -1
    _restart_last_ymd   = -1   # last ymd saved in restart file (for CONTINUE_RUN detection)
    _restart_last_tod   = -1
    _restart_cam_out    = None  # saved cam_out from restart file (re-served on CONTINUE_RUN)
    if atm_restart_file.exists():
        logger.info(f"ATM restart found: {atm_restart_file}")
        _ckpt         = torch.load(atm_restart_file, map_location=device)
        state         = _ckpt["state"].to(device)
        timestep_init = int(_ckpt["timestep"])
        logger.info(f"  Resuming atmosphere from step {timestep_init} "
                    f"(state shape: {list(state.shape)})")
        # Compute the expected CESM date of the FIRST go.flag after restart.
        # We saved the ymd/tod of the LAST processed step; CESM will send the
        # NEXT step (+DT_SEC).  If CESM's date doesn't match, the server and
        # CESM are out of sync (e.g. stale restart file from a prior run).
        _last_ymd = int(_ckpt.get("last_ymd", -1))
        _last_tod = int(_ckpt.get("last_tod", -1))
        if _last_ymd > 0:
            _restart_last_ymd   = _last_ymd
            _restart_last_tod   = _last_tod
            _next_dt            = cesm_ymd_tod_to_dt(_last_ymd, _last_tod) + timedelta(seconds=DT_SEC)
            _expected_first_ymd = _next_dt.year * 10000 + _next_dt.month * 100 + _next_dt.day
            _expected_first_tod = _next_dt.hour * 3600 + _next_dt.minute * 60 + _next_dt.second
            logger.info(f"  Expecting first go.flag: ymd={_expected_first_ymd} tod={_expected_first_tod}s "
                        f"(last saved: ymd={_last_ymd} tod={_last_tod}s)")
        else:
            logger.warning("  *** Restart file has no date metadata (created before date-check was added).")
            logger.warning("  *** Cannot verify CESM/atmosphere date alignment automatically.")
            logger.warning("  *** If this is a FRESH CESM run, delete the restart file and relaunch:")
            logger.warning(f"  ***   rm {atm_restart_file}")
        # Load saved cam_out arrays if present (written by server versions ≥ Feb 2026).
        # Used to re-serve the last step cleanly on CONTINUE_RUN restarts (CESM resends
        # the last processed step once before continuing forward).
        _restart_cam_out = _ckpt.get("cam_out", None)
        if _restart_cam_out is not None:
            logger.info("  Saved cam_out found — CONTINUE_RUN re-send will be handled cleanly")
        else:
            logger.info("  No saved cam_out (old restart) — CONTINUE_RUN would need manual patch")
    else:
        logger.info("No ATM restart file found — starting from IC")
        timestep_init = 0

    # ------------------------------------------------------------------
    # 2. Grid setup
    # ------------------------------------------------------------------
    t62_lats, t62_lons = t62_latlons()

    # CAMulator grid — ensure ascending lat order for RegularGridInterpolator
    cam_lats_raw = latlons.latitude.values.copy()   # may be N→S
    cam_lons     = latlons.longitude.values.copy()  # 0→360
    cam_lats_asc = np.sort(cam_lats_raw)            # ensure S→N
    cam_flip     = cam_lats_raw[0] > cam_lats_raw[-1]  # True if original is N→S

    logger.info(f"T62 grid      : {T62_NLAT} lat × {T62_NLON} lon = {T62_NGRID} pts")
    logger.info(f"CAMulator grid: {len(cam_lats_asc)} lat × {len(cam_lons)} lon")
    logger.info(f"CAM lat order : {'N→S (will flip)' if cam_flip else 'S→N (OK)'}")

    # Precompute bilinear remap weights (done once; each step is then ~1 ms) --
    # Precompute cos(lat) weights for area-weighted global diagnostics (shape: nlat, 1)
    _cam_lat_w = np.cos(np.radians(cam_lats_asc))[:, None]  # (192, 1)
    _cam_lat_w = _cam_lat_w / _cam_lat_w.mean()              # normalise so mean weight = 1

    logger.info("Precomputing bilinear remap weights ...")
    _t_remap_init = time.time()
    t62_to_cam_remap = BilinearRemap(t62_lats, t62_lons, cam_lats_asc, cam_lons)
    cam_to_t62_remap = BilinearRemap(cam_lats_asc, cam_lons, t62_lats, t62_lons)
    logger.info(f"  Remap weights ready in {time.time()-_t_remap_init:.2f}s  "
                f"(T62→CAM and CAM→T62 both precomputed)")

    # ------------------------------------------------------------------
    # 3. SST / ICEFRAC normalization scalars
    #    Both are dynamic_forcing_variables → scalar mean and std from state_transformer
    # ------------------------------------------------------------------
    sst_mean     = float(state_transformer.mean_tensors["SST"])
    sst_std      = float(state_transformer.std_tensors["SST"])
    icefrac_mean = float(state_transformer.mean_tensors["ICEFRAC"])
    icefrac_std  = float(state_transformer.std_tensors["ICEFRAC"])
    logger.info(f"SST scaler    : mean={sst_mean:.3f} K, std={sst_std:.3f} K")
    logger.info(f"ICEFRAC scaler: mean={icefrac_mean:.4f}, std={icefrac_std:.4f}")

    # ------------------------------------------------------------------
    # 3b. Bottom model level height scale factor
    #     CAM6 L32 bottom level is pure sigma (hyam[-1]=0, hybm[-1]≈0.9926).
    #     Hypsometric formula with virtual temperature correction:
    #       p_mid = hyam[-1]*P0 + hybm[-1]*PS  =  hybm[-1]*PS   (a=0)
    #       Tv    = T * (1 + 0.608*q)
    #       z_bot = (Rd/g) * Tv * ln(PS / p_mid)
    #             = (Rd/g) * (-ln(hybm[-1])) * Tv
    #             = Z_BOT_SCALE * Tv
    #     Z_BOT_SCALE is PS-independent because hyam[-1]=0 (pure sigma at bottom).
    #     Dry-air range: ~50 m (T=230K) to ~67 m (T=305K); Tv adds ~1-2% in tropics.
    # ------------------------------------------------------------------
    _statics_path = conf["data"]["save_loc_static"]
    with nc.Dataset(_statics_path, "r") as _ds:
        _hybm_bot = float(_ds.variables["hybm"][-1])   # midpoint b-coeff, bottom level
        _hyam_bot = float(_ds.variables["hyam"][-1])   # midpoint a-coeff (nondimensional)
    # Verify pure-sigma assumption: hyam[-1] should be 0 or negligible
    if abs(_hyam_bot) > 1e-6:
        logger.warning(f"Bottom level hyam[-1]={_hyam_bot:.6f} != 0 — "
                       f"z_bot scale is PS-dependent; using hybm[-1] only")
    Z_BOT_SCALE = (287.058 / 9.80616) * (-np.log(_hybm_bot))  # m / K (dry)
    logger.info(f"Bottom level  : hybi[-2]=0.98511219, Z_BOT_SCALE={Z_BOT_SCALE:.6f} m/K")
    logger.info(f"              z_bot range: "
                f"{Z_BOT_SCALE*230:.1f} m (T=230K) to "
                f"{Z_BOT_SCALE*305:.1f} m (T=305K)")

    # ------------------------------------------------------------------
    # 4. Forcing dataset navigation
    #    forcing_ix is derived from the CESM date (ymd/tod) in each go.flag,
    #    NOT from a raw timestep counter.  This makes solar forcing immune to
    #    CESM initialization steps that repeat the same date multiple times
    #    (e.g. 2-3 steps at "20101 tod=0" at every annual restart transition).
    # ------------------------------------------------------------------
    df_vars    = conf["data"]["dynamic_forcing_variables"]
    dynamic_ds = forcing_ds_norm[df_vars]
    start_raw  = conf["predict"]["start_datetime"]
    loc        = dynamic_ds.indexes["time"].get_loc(start_raw)
    start_ix   = loc.start if isinstance(loc, slice) else loc
    model_start_dt  = datetime.strptime(start_raw, "%Y-%m-%d %H:%M:%S")
    # Grab the cftime type used by the forcing dataset so get_loc works correctly.
    # ERA5 / CESM forcing files often use cftime (e.g. DatetimeGregorian, DatetimeNoLeap)
    # rather than standard Python datetime — passing the wrong type raises KeyError.
    _cftime_type = type(dynamic_ds.indexes["time"][start_ix])
    logger.info(f"Forcing dataset start index: {start_ix}  ({start_raw})")
    logger.info(f"Model start year: {model_start_dt.year}  "
                f"(CESM year 1 = {model_start_dt.year}, year 2 = {model_start_dt.year+1}, ...)  "
                f"cftime type: {_cftime_type.__name__}")

    # Detect single-year climatological forcing file (e.g. from make_cyclic_forcing.py).
    # If the file only covers one calendar year, every model year wraps back to that year
    # so the run can cycle indefinitely without running off the end of the forcing record.
    _forcing_years = sorted({t.year for t in dynamic_ds.indexes["time"]})
    _cyclic_forcing_year = _forcing_years[0] if len(_forcing_years) == 1 else None
    _n_forcing_steps     = len(dynamic_ds.indexes["time"])
    if _cyclic_forcing_year is not None:
        logger.info(f"Cyclic forcing detected: single-year file (year {_cyclic_forcing_year}). "
                    f"All model years will wrap to {_cyclic_forcing_year}.")
    else:
        logger.info(f"Multi-year forcing: years {_forcing_years[0]}–{_forcing_years[-1]}.")

    def _next_forcing_ix(ix):
        """Return the prefetch index one step ahead of ix.

        For single-year cyclic files wraps modulo the file length so Dec 31 18:00
        (index 1459) rolls over to Jan 1 00:00 (index 0) instead of raising
        IndexError.  Multi-year files use plain +1 — unchanged behaviour.
        """
        if _cyclic_forcing_year is not None:
            return (ix + 1) % _n_forcing_steps
        return ix + 1

    def cesm_to_forcing_ix(ymd, tod):
        """
        Convert CESM model date (ymd=YYYYMMDD, tod=seconds) to forcing dataset index.

        CESM model year 1 maps to model_start_dt.year.
        Uses the dataset's own cftime type to avoid calendar-mismatch KeyErrors.
        Handles repeated dates during init steps: same ymd/tod → same index.

        Cyclic forcing: if the forcing file spans only one calendar year (detected at
        startup), every model year is wrapped back to that year so the run can cycle
        indefinitely (e.g. a 25-year run with a 1-year climatological forcing file).
        """
        real_year = (
            _cyclic_forcing_year                           # always use the climatology year
            if _cyclic_forcing_year is not None
            else model_start_dt.year + (ymd // 10000) - 1  # map model year → real year
        )
        real_dt = _cftime_type(
            real_year,
            (ymd % 10000) // 100,
            ymd % 100,
            tod // 3600,
        )
        try:
            idx = dynamic_ds.indexes["time"].get_loc(real_dt)
            return idx.start if isinstance(idx, slice) else int(idx)
        except KeyError:
            logger.error(f"Forcing date {real_dt} not found in dataset "
                         f"(CESM ymd={ymd} tod={tod}). Run may have exceeded forcing coverage.")
            raise

    # Log the resume point for verification (still useful for human cross-check)
    if timestep_init > 0:
        if _cyclic_forcing_year is not None:
            # Cyclic file: raw timestep index is meaningless (wraps every 1460 steps).
            # Derive resume index from the saved CESM date instead.
            if _expected_first_ymd > 0:
                _resume_ix = cesm_to_forcing_ix(_expected_first_ymd, _expected_first_tod)
            else:
                _resume_ix = start_ix  # fallback: old restart with no date info
        else:
            _resume_ix = start_ix + timestep_init
        _resume_time = dynamic_ds.indexes["time"][_resume_ix]
        logger.info(f"Forcing resume  : approx index={_resume_ix}  date={_resume_time}")
        logger.info(f"  *** RESTART: forcing_ix will be set from CESM date on first go.flag ***")

    # Background thread pool for prefetching forcing data.
    # While CESM runs its ocean/ice/coupler step (~seconds), we pre-load the
    # next forcing chunk so disk I/O cost is hidden behind CESM's compute time.
    _forcing_executor = ThreadPoolExecutor(max_workers=1)

    def _load_forcing_slice(ix):
        """Load and stack forcing variables for timestep ix (runs in background)."""
        ds_slice = dynamic_ds.isel(time=ix).load()
        arr = np.stack([ds_slice[v].values for v in df_vars], axis=0)
        # Return CPU tensor — never call .to(device) from a background thread
        return torch.from_numpy(arr.copy()).float().unsqueeze(0).unsqueeze(2)

    # Kick off prefetch for step 0 immediately so it overlaps with JIT tracing.
    # _prefetch_ix tracks what index was submitted so we can detect mismatches.
    # For cyclic forcing on restart: derive from CESM date, not raw timestep counter.
    if _cyclic_forcing_year is not None and timestep_init > 0 and _expected_first_ymd > 0:
        _prefetch_ix = cesm_to_forcing_ix(_expected_first_ymd, _expected_first_tod)
    else:
        _prefetch_ix = start_ix + timestep_init   # best guess before first go.flag
    _prefetch_future = _forcing_executor.submit(_load_forcing_slice, _prefetch_ix)

    # ------------------------------------------------------------------
    # 5. Compile model for performance
    #    torch.compile (PyTorch 2.x) performs kernel fusion and shape-specific
    #    tuning beyond what torch.jit.trace can do.  'reduce-overhead' targets
    #    repeated fixed-shape inference (minimises Python/CUDA launch overhead).
    #    Expect the first 2-3 steps to be slow while Triton kernels compile;
    #    subsequent steps should be faster than the JIT-traced baseline.
    #    Falls back to torch.jit.trace if torch.compile is unavailable.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 5. Trace model for performance
    #    JIT trace requires an input with the correct full channel count
    #    (136 = prognostic + dynamic forcing + static).
    #    - Fresh start:  ctx["initial_state"] already has 136 channels embedded.
    #    - Restart:      state has only 130 prognostic channels; must build the
    #                    full model input via build_input_with_forcing so the
    #                    trace sees the correct shape.
    # ------------------------------------------------------------------
    logger.info("Tracing model with torch.jit.trace ...")
    if timestep_init == 0:
        trace_input = state.float()
    else:
        # Build full 136-channel input at the resume timestep for tracing.
        # For cyclic forcing, derive index from CESM date not raw timestep counter.
        if _cyclic_forcing_year is not None and _expected_first_ymd > 0:
            _trace_ix = cesm_to_forcing_ix(_expected_first_ymd, _expected_first_tod)
        else:
            _trace_ix = start_ix + timestep_init
        _trace_ds  = dynamic_ds.isel(time=_trace_ix).load()
        _trace_arr = np.stack([_trace_ds[v].values for v in df_vars], axis=0)
        _trace_forcing = (
            torch.from_numpy(_trace_arr.copy())
            .float().unsqueeze(0).unsqueeze(2).to(device)
        )
        trace_input = stepper.state_manager.build_input_with_forcing(
            state, _trace_forcing, static_forcing
        ).float()
    stepper.model = torch.jit.trace(stepper.model, trace_input)
    logger.info(f"Model traced  (input shape: {list(trace_input.shape)})")

    # ------------------------------------------------------------------
    # 5b. Extract initial SST from the IC tensor for early-step fallback.
    #     On coupling steps 0-1, POP has not yet produced output so the
    #     coupler delivers So_t = 0 K everywhere.  Rather than feeding the
    #     model a blank (or uniform 283 K) ocean, we seed sst_cam_persistent
    #     from the initial condition — which carries realistic SST for the
    #     run start date.  This persistent field is updated every step once
    #     POP starts delivering real SST values.
    # ------------------------------------------------------------------
    if timestep_init == 0:
        # Fresh start: IC state has 136 channels (forcing embedded), SST is present.
        with torch.no_grad():
            sst_ic_norm = accessor_input.get_state_var(state, "SST")[0, 0, 0].cpu().numpy()
        sst_cam_persistent = sst_ic_norm * sst_std + sst_mean
        sst_cam_persistent = np.where(sst_cam_persistent < OCEAN_MIN_K,
                                      LAND_SST_FILL, sst_cam_persistent)
        _ocn = sst_cam_persistent[sst_cam_persistent >= OCEAN_MIN_K]
        logger.info(f"IC SST (from init .pth): ocn_mean={_ocn.mean():.1f} K  "
                    f"ocn_min={_ocn.min():.1f} K  ocn_max={_ocn.max():.1f} K")
    else:
        # Restart: state has only 130 prognostic channels; SST (dynamic forcing) is
        # not present.  Initialise the persistent-SST fallback to a uniform fill —
        # POP restarts with valid SST and will update sst_cam_persistent on step 1,
        # so this default is used at most for one step if So_t happens to be 0.
        sst_cam_persistent = np.full((CAM_NLAT, CAM_NLON), LAND_SST_FILL, dtype=np.float64)
        logger.info("IC SST (restart): sst_cam_persistent initialised to LAND_SST_FILL; "
                    "POP will supply valid SST on first coupling step")

    # ------------------------------------------------------------------
    # 6. Clean up any stale flag files from a previous run
    # ------------------------------------------------------------------
    delete_flag(go_flag)
    delete_flag(done_flag)
    delete_flag(rundir / READY_FLAG)

    write_flag(rundir / READY_FLAG)
    logger.info("")
    logger.info("Server ready — waiting for CESM go.flag  ...")
    logger.info(f"  (ready flag written: {rundir / READY_FLAG})")
    logger.info("")

    # ------------------------------------------------------------------
    # 7b. Atmospheric NetCDF output pool (optional)
    #     Only active when --save_atm_nc is specified.  The provided subdir
    #     is appended to rundir.  4-worker mp.Pool mirrors Quick_Climate.py's
    #     pool.apply_async pattern so disk I/O runs during CESM's inter-step compute.
    # ------------------------------------------------------------------
    _save_nc     = args.save_atm_nc is not None
    _save_nc_dir = rundir / args.save_atm_nc if _save_nc else None
    _do_daily    = _save_nc and args.daily_mean
    _cam_lat     = latlons.latitude.values
    _cam_lon     = latlons.longitude.values
    pool         = None
    if _save_nc:
        _save_nc_dir.mkdir(parents=True, exist_ok=True)
        pool = mp.Pool(4)
        logger.info(f"Atm NC output      : {_save_nc_dir}")
        if _do_daily:
            logger.info(f"  Mode             : daily-mean only  (camulator.h1d.<YYYY-MM-DD>.nc)")
        else:
            logger.info(f"  Mode             : 6-hourly  (camulator.h1.<YYYY-MM-DD-SSSSS>.nc)")

    # ==================================================================
    # 8. Main coupling loop
    # ==================================================================
    timestep          = timestep_init
    _prev_ymd         = -1    # CESM date of the previous step (for repeated-date detection)
    _prev_tod         = -1
    _prev_step_year   = -1    # CESM model year of the last completed non-repeated step
    _cached_cam_out   = None  # last inference output: reused for repeated-date init steps
    _daily_buffer     = []    # (upper_air, single_level) tuples for daily averaging
    _daily_buffer_ymd = -1    # ymd of the day currently being accumulated

    while True:

        # --- (a) wait for go.flag ---
        poll = 0
        while not go_flag.exists():
            time.sleep(POLL_SLEEP)
            poll += 1
            if poll > POLL_MAX:
                logger.error("Timed out waiting for camulator_go.flag. "
                             "Is CESM still running? Exiting.")
                if _do_daily and _daily_buffer:
                    _d_str = (f"{_daily_buffer_ymd // 10000:04d}-"
                              f"{(_daily_buffer_ymd % 10000) // 100:02d}-"
                              f"{_daily_buffer_ymd % 100:02d}")
                    _d_dir = _save_nc_dir / f"{_daily_buffer_ymd // 10000:04d}"
                    pool.apply_async(save_camulator_daily_nc,
                                     (_daily_buffer,
                                      str(_d_dir / f"camulator.h1d.{_d_str}.nc")))
                    logger.info(
                        f"  Flushing partial daily buffer "
                        f"({len(_daily_buffer)} steps) → {_d_str}"
                    )
                pool.close()
                pool.join()
                sys.exit(1)
            if poll % POLL_LOG == 0:
                logger.info(f"  Waiting for go.flag  (poll {poll} / {POLL_MAX}) ...")

        t_step_start = time.time()
        logger.info(f"Step {timestep:04d}: received go.flag")
        delete_flag(go_flag)

        # --- (b) read SST from CESM ---
        sst_flat, ifrac_flat, ymd, tod = read_sst_nc(sst_file)
        sst_ocn = sst_flat[sst_flat >= OCEAN_MIN_K]   # ocean-only points for diagnostics
        ocn_mean = sst_ocn.mean() if len(sst_ocn) > 0 else 0.0
        logger.info(f"  SST  min={sst_flat.min():.1f} K  max={sst_flat.max():.1f} K  "
                    f"ocn_mean={ocn_mean:.1f} K  (land pts: {(sst_flat < OCEAN_MIN_K).sum()})  "
                    f"date={ymd}  tod={tod}s")

        # --- CONTINUE_RUN re-send detection (first step only) ---
        # When CESM is restarted with CONTINUE_RUN=TRUE it resends the last step the
        # server already processed (CESM's checkpoint == server's last step).  The
        # atmosphere state is already one step ahead.  Re-serve the saved cam_out
        # without re-running inference so the run continues cleanly with no overlap.
        # This is distinct from the year-boundary case (2-3× repeated dates) — here
        # CESM sends the resent step exactly once, then advances normally.
        if (timestep == timestep_init and _restart_last_ymd > 0
                and ymd == _restart_last_ymd and tod == _restart_last_tod):
            if _restart_cam_out is not None:
                logger.info(f"  CONTINUE_RUN re-send: ymd={ymd} tod={tod}s — "
                            f"re-serving saved cam_out, skipping inference")
                write_cam_nc(cam_file,
                             _restart_cam_out["u10"], _restart_cam_out["v10"],
                             _restart_cam_out["tbot"], _restart_cam_out["zbot"],
                             _restart_cam_out["tref"], _restart_cam_out["qbot"],
                             _restart_cam_out["pbot"], _restart_cam_out["fsds"],
                             _restart_cam_out["flnsd"], _restart_cam_out["prect"])
                write_flag(done_flag)
                forcing_ix   = cesm_to_forcing_ix(ymd, tod)
                _prefetch_ix = _next_forcing_ix(forcing_ix)
                _prefetch_future = _forcing_executor.submit(_load_forcing_slice, _prefetch_ix)
                torch.save({"state": state.cpu(), "timestep": timestep,
                            "last_ymd": ymd, "last_tod": tod,
                            "cam_out": _restart_cam_out}, atm_restart_file)
                _prev_ymd = ymd
                _prev_tod = tod
                continue
            else:
                # Old restart file with no saved cam_out.  Accept the step and re-run
                # inference on the already-advanced state (one 6-hr overlap — fine for climate).
                logger.warning(f"  CONTINUE_RUN re-send: ymd={ymd} tod={tod}s but no saved cam_out.")
                logger.warning(f"  Re-running inference on already-advanced state (1-step discrepancy).")
                _expected_first_ymd = ymd   # disarm mismatch guard so run can continue
                _expected_first_tod = tod

        # --- repeated-date detection (CESM init steps at restart boundaries) ---
        # CESM sends the same ymd/tod 2-3 times during restart initialization.
        # Running inference + shift_state_forward on each would advance the
        # atmosphere's internal state extra steps, drifting ahead of the CESM clock.
        # Instead: reuse the previous step's output and skip state advance entirely.
        _date_repeated = (ymd == _prev_ymd and tod == _prev_tod)
        _prev_ymd = ymd
        _prev_tod = tod

        if _date_repeated and _cached_cam_out is not None:
            _u10, _v10, _tbot, _zbot, _tref, _qbot, _pbot, _fsds, _flnsd, _prect = _cached_cam_out
            logger.info(f"  Repeated date — reusing last output, skipping inference + state advance")
            write_cam_nc(cam_file, _u10, _v10, _tbot, _zbot, _tref, _qbot, _pbot, _fsds, _flnsd, _prect)
            write_flag(done_flag)
            _prefetch_ix     = _next_forcing_ix(forcing_ix)
            _prefetch_future = _forcing_executor.submit(_load_forcing_slice, _prefetch_ix)
            torch.save({"state": state.cpu(), "timestep": timestep,
                        "last_ymd": ymd, "last_tod": tod,
                        "cam_out": {"u10": _u10, "v10": _v10, "tbot": _tbot, "zbot": _zbot,
                                    "tref": _tref, "qbot": _qbot, "pbot": _pbot, "fsds": _fsds,
                                    "flnsd": _flnsd, "prect": _prect}}, atm_restart_file)
            continue   # do NOT shift_state_forward, do NOT increment timestep

        # --- restart date guard (first step only) ---
        # Abort if CESM's date doesn't match what the restart file expects.
        # This catches the case where the user starts a fresh CESM run without
        # deleting the stale camulator_atm_restart.pth from a prior run.
        if timestep == timestep_init and _expected_first_ymd > 0:
            if ymd != _expected_first_ymd or tod != _expected_first_tod:
                logger.error("=" * 65)
                logger.error("RESTART DATE MISMATCH — atmosphere and CESM are out of sync!")
                logger.error(f"  Expected first go.flag : ymd={_expected_first_ymd}  tod={_expected_first_tod}s")
                logger.error(f"  CESM sent              : ymd={ymd}  tod={tod}s")
                logger.error(f"  If this is a fresh CESM run, delete {atm_restart_file} and restart the server.")
                logger.error("=" * 65)
                sys.exit(1)
            logger.info(f"  Restart date check PASSED (CESM ymd={ymd} matches expected)")

        # Reshape flat T62 array to 2D (94 lat × 192 lon), lat ascending S→N
        sst_t62 = sst_flat.reshape(T62_NLAT, T62_NLON)

        # --- (c) remap SST  T62 → CAMulator (ascending lat) ---
        sst_cam = t62_to_cam_remap(sst_t62)   # (cam_nlat_asc, cam_nlon)

        # If the CAMulator grid is stored N→S, flip now
        if cam_flip:
            sst_cam = sst_cam[::-1, :]

        # --- (c2) land mask + POP-readiness check ---
        # If POP has not yet produced output (steps 0-1, So_t = 0 everywhere),
        # fall back to the persisted IC SST so the model sees a realistic ocean
        # pattern rather than a blank field.  Once POP starts delivering real
        # SST, replace land points with 283 K (training fill) and save the
        # updated field as the new persistent fallback.
        ocean_pts = sst_cam >= OCEAN_MIN_K
        if not ocean_pts.any():
            logger.info("  SST: POP not ready — using persisted IC SST")
            sst_cam = sst_cam_persistent.copy()
        else:
            sst_cam = np.where(ocean_pts, sst_cam, LAND_SST_FILL)
            sst_cam_persistent = sst_cam.copy()   # update fallback for future steps

        # --- (d) normalize SST for model ---
        sst_norm = (sst_cam - sst_mean) / sst_std

        # --- (e) collect forcing for this CESM date ---
        t_e = time.time()
        forcing_ix = cesm_to_forcing_ix(ymd, tod)   # derived from CESM date, not timestep counter

        if forcing_ix == _prefetch_ix:
            # Happy path: prefetch matches — return is instant
            dynamic_forcing_t = _prefetch_future.result().to(device)
        else:
            # Mismatch: CESM repeated a date (init steps) or date-counter drifted.
            # Load synchronously and drain the orphaned prefetch.
            logger.info(f"  Forcing idx mismatch: prefetched={_prefetch_ix} need={forcing_ix} "
                        f"(CESM date={ymd} tod={tod}s) — sync load")
            dynamic_forcing_t = _load_forcing_slice(forcing_ix).to(device)
            _prefetch_future.result()   # drain orphaned thread before we submit next

        # shape: (1, n_forcing_vars, 1, 192, 288)
        t_forcing_ms = (time.time() - t_e) * 1000
        # NOTE: next prefetch is submitted AFTER write_flag so disk I/O runs
        #       during CESM's ocean/ice compute, not during our active GPU work.

        # --- (f) build model input ---
        t_f = time.time()
        if timestep == 0:
            # First step: the initial state already contains forcing embedded
            model_input = state
        else:
            model_input = stepper.state_manager.build_input_with_forcing(
                state, dynamic_forcing_t, static_forcing
            )
        t_build_ms = (time.time() - t_f) * 1000

        # --- (g) inject ocean SST (replaces CAMulator's climatological SST) ---
        # Expected shape for set_state_var: (batch=1, ch=1, time=1, lat=192, lon=288)
        sst_tensor = (
            torch.from_numpy(sst_norm.copy())
            .float()
            .to(device)
            .unsqueeze(0)   # batch
            .unsqueeze(0)   # channel
            .unsqueeze(0)   # time
        )
        accessor_input.set_state_var(model_input, "SST", sst_tensor)

        # --- (g2) inject CICE ice fraction (replaces CAMulator's climatological ICEFRAC) ---
        # Only inject when POP/CICE is ready (same guard as SST).  On steps 0-1 before
        # POP delivers data, the IC state already carries a realistic ICEFRAC so we leave
        # it untouched.  ifrac_flat comes from camulator_sst_in.nc alongside sst_flat.
        if ocean_pts.any():
            ifrac_t62  = ifrac_flat.reshape(T62_NLAT, T62_NLON)
            ifrac_cam  = t62_to_cam_remap(ifrac_t62)
            if cam_flip:
                ifrac_cam = ifrac_cam[::-1, :]
            ifrac_norm = (ifrac_cam - icefrac_mean) / icefrac_std
            ifrac_tensor = (
                torch.from_numpy(ifrac_norm.copy())
                .float()
                .to(device)
                .unsqueeze(0)   # batch
                .unsqueeze(0)   # channel
                .unsqueeze(0)   # time
            )
            accessor_input.set_state_var(model_input, "ICEFRAC", ifrac_tensor)

        # --- (h) run model inference + post-processing ---
        t_inf = time.time()
        with torch.no_grad():
            prediction = stepper.model(model_input.float())
        prediction = stepper._apply_postprocessing(prediction, model_input)
        t_inf_ms = (time.time() - t_inf) * 1000
        logger.info(f"  Inference: {t_inf_ms:.0f}ms")

        # --- (i) inverse-transform to physical units ---
        t_i = time.time()
        prediction_out = state_transformer.inverse_transform(prediction)
        t_itrans_ms = (time.time() - t_i) * 1000
        # prediction_out shape: (1, n_out_channels, 1, 192, 288)

        # --- (j) extract coupling variables ---
        # Move to CPU once; all subsequent indexing stays on CPU (avoids 9 GPU→CPU syncs)
        t_j = time.time()
        prediction_cpu = prediction_out.cpu()
        # 3D vars: shape (1, levels, 1, lat, lon) — take lowest model level (index −1 = surface)
        U_cam    = accessor_output.get_state_var(prediction_cpu, "U"   )[0, -1, 0].numpy()
        V_cam    = accessor_output.get_state_var(prediction_cpu, "V"   )[0, -1, 0].numpy()
        T_bot_cam= accessor_output.get_state_var(prediction_cpu, "T"   )[0, -1, 0].numpy()
        Qtot_cam = accessor_output.get_state_var(prediction_cpu, "Qtot")[0, -1, 0].numpy()

        # Dynamic bottom-level height with virtual temperature correction.
        # Tv = T * (1 + 0.608*q) accounts for moisture effect on air density.
        # Hypsometric: z = (Rd/g)*(-ln(hybm[-1]))*Tv  (PS-independent, pure sigma bottom)
        # Range: ~50 m (polar, T=230K) to ~68 m (moist tropics)
        Tv_bot_cam = T_bot_cam * (1.0 + 0.608 * np.clip(Qtot_cam, 0.0, 0.04))
        z_bot_cam  = Z_BOT_SCALE * Tv_bot_cam   # shape (192, 288)

        # 2D surface vars: shape (1, 1, 1, lat, lon)
        TREFHT_cam = accessor_output.get_state_var(prediction_cpu, "TREFHT")[0, 0, 0].numpy()
        PS_cam     = accessor_output.get_state_var(prediction_cpu, "PS"    )[0, 0, 0].numpy()

        # 2D diagnostics — model output is cumulative over 6 h (J/m²), convert to W/m²
        FSNS_cam  = accessor_output.get_state_var(prediction_cpu, "FSNS" )[0, 0, 0].numpy()
        FSNS_cam  = FSNS_cam / DT_SEC              # → W/m², positive downward

        FLNS_cam  = accessor_output.get_state_var(prediction_cpu, "FLNS" )[0, 0, 0].numpy()
        TS_cam    = accessor_output.get_state_var(prediction_cpu, "TS"  )[0, 0, 0].numpy()
        # FLNSD = ε σ T⁴ + FLNS/DT  (FLNS accumulated is net outward, negative for typical conditions)
        _sb_term   = 0.99 * 5.670374419e-8 * TS_cam**4
        _flns_term = FLNS_cam / DT_SEC
        FLNSD_cam  = _sb_term + _flns_term
        # Optional FLNSD diagnostic: area-weighted SB-term comparison across TS/TREFHT/Tbot.
        # Enable with --flnsd_diag. Validated 2026-03: TS gives ~331 W/m² vs CAM6 ref 336.9 W/m².
        if args.flnsd_diag:
            _cam_w     = _cam_lat_w if not cam_flip else _cam_lat_w[::-1]
            _wmean     = lambda f: np.average(f, weights=np.broadcast_to(_cam_w, f.shape))
            _sb_trefht = 0.99 * 5.670374419e-8 * TREFHT_cam**4
            _sb_tbot   = 0.99 * 5.670374419e-8 * T_bot_cam**4
            logger.info(
                f"  FLNSD decomp [area-wtd]: FLNS/DT={_wmean(_flns_term):.1f}  "
                f"using TS: εσTS⁴={_wmean(_sb_term):.1f} (TS={_wmean(TS_cam):.2f}K → FLNSD={_wmean(_sb_term+_flns_term):.1f})  "
                f"using TREFHT: εσ⁴={_wmean(_sb_trefht):.1f} (T={_wmean(TREFHT_cam):.2f}K → FLNSD={_wmean(_sb_trefht+_flns_term):.1f})  "
                f"using Tbot: εσ⁴={_wmean(_sb_tbot):.1f} (T={_wmean(T_bot_cam):.2f}K → FLNSD={_wmean(_sb_tbot+_flns_term):.1f})"
            )

        PRECT_cam = accessor_output.get_state_var(prediction_cpu, "PRECT")[0, 0, 0].numpy()
        PRECT_cam = PRECT_cam / DT_SEC              # accumulated mm → m/s liquid-water equivalent
        t_extract_ms = (time.time() - t_j) * 1000

        # --- (k) remap all fields CAMulator → T62  (single batched call) ---
        # Stack all 11 output fields, flip to ascending lat if needed, then
        # call cam_to_t62_remap.batch() once instead of 11 separate remap_field calls.
        _cam_stack = np.stack([
            U_cam, V_cam, T_bot_cam, z_bot_cam, TREFHT_cam,
            Qtot_cam, PS_cam, FSNS_cam, FLNSD_cam, PRECT_cam,
            TS_cam,
        ])  # (11, cam_nlat, cam_nlon)
        if cam_flip:
            _cam_stack = _cam_stack[:, ::-1, :]   # flip all fields to ascending lat

        t_remap_start = time.time()
        _t62 = cam_to_t62_remap.batch(_cam_stack)  # (11, T62_NGRID)
        t_remap_ms = (time.time() - t_remap_start) * 1000

        u10, v10, tbot, zbot, tref, qbot, pbot, fsns, flnsd, prect, ts_t62 = (
            _t62[i].astype(np.float64) for i in range(11)
        )

        # --- (k2) reconstruct downwelling SW (FSDS) from net SW (FSNS) ---
        # Problem: CAMulator provides FSNS = FSDS*(1-alpha_sfc), but CPL7's
        #   seq_flux_mct.F90 treats Faxa_sw* as DOWNWELLING SW and applies the
        #   ocean/ice albedo itself (swupc = sum(Faxa_sw* * albedo_ocn)).
        #   Passing FSNS directly would double-count the surface albedo:
        #     SW_absorbed_wrong = FSNS*(1-alpha_ocn)  instead of  FSNS.
        #   The fix: undo CAMulator's surface albedo so that after the coupler
        #   re-applies its own (consistent) albedo the net result is correct.
        # Method: use the CICE ice fraction + surface temperature to estimate alpha_sfc.
        #   alpha_ocean ≈ 0.06 (CAM6 / CICE5 open-water shortwave albedo)
        #   alpha_ice   = f(TS): temperature-dependent sea-ice albedo.
        #     TS ≤ -1 °C (T_crit): alpha_dry = 0.80 (snow-covered ice)
        #     TS ≥  0 °C         : alpha_wet = 0.50 (ponded/melting ice)
        #     Linear interpolation between T_crit and 0 °C.
        #   The 1°C transition window reflects the sharp onset of surface melting;
        #   ice stays snow-covered (high albedo) until very close to the melting
        #   point, then rapidly develops melt ponds.
        #   Physical rationale: a fixed alpha_ice = 0.60 severely underestimates
        #   FSDS in spring (real dry-snow alpha ~0.80), leaving CICE with too
        #   little SW and delaying the Arctic melt-season onset.
        #   FSDS = FSNS / max(1 - alpha_sfc, 0.10)   [only where FSNS > 0]
        _ALPHA_OCN   = 0.06
        _ALPHA_DRY   = 0.80   # snow-covered ice, TS ≤ T_CRIT
        _ALPHA_WET   = 0.50   # ponded/melting ice, TS ≥ 0 °C
        _T_CRIT_K    = 272.15  # -1 °C: onset of surface melting
        _T_MELT_K    = 273.15  #  0 °C: fully wet/ponded
        # alpha_dry for TS ≤ T_crit; alpha_wet for TS ≥ 0; linear ramp between
        _frac_melt = np.clip(
            (ts_t62 - _T_CRIT_K) / (_T_MELT_K - _T_CRIT_K), 0.0, 1.0
        )
        _alpha_ice   = _ALPHA_DRY + _frac_melt * (_ALPHA_WET - _ALPHA_DRY)
        _alpha_sfc   = (1.0 - ifrac_flat) * _ALPHA_OCN + ifrac_flat * _alpha_ice
        _one_minus_a = np.maximum(1.0 - _alpha_sfc, 0.10)   # floor prevents wild values
        fsds = np.where(fsns > 0.0, fsns / _one_minus_a, 0.0).astype(np.float64)
        fsds = np.minimum(fsds, 1500.0)    # cap at plausible maximum TSI at surface

        logger.info(f"  FSNS  mean={FSNS_cam.mean():7.1f} W/m²  "
                    f"FSDS  mean={fsds.mean():7.1f} W/m²  "
                    f"FLNSD mean={FLNSD_cam.mean():7.1f} W/m²  "
                    f"|U_bot| mean={np.sqrt(U_cam**2+V_cam**2).mean():5.2f} m/s  "
                    f"zbot mean={z_bot_cam.mean():.1f} m  "
                    f"remap={t_remap_ms:.1f}ms")

        # --- (l) safety clamps (Fortran also clamps, but belt-and-suspenders) ---
        qbot  = np.maximum(qbot,  1.0e-9)
        zbot  = np.clip(zbot, 20.0, 200.0)   # physical bounds: 20–200 m
        fsds  = np.maximum(fsds,  0.0)
        flnsd = np.maximum(flnsd, 0.0)
        prect = np.maximum(prect, 0.0)

        # --- (m) write cam_out.nc ---
        t_m = time.time()
        write_cam_nc(cam_file, u10, v10, tbot, zbot, tref, qbot, pbot, fsds, flnsd, prect)
        t_write_ms = (time.time() - t_m) * 1000

        # --- (n) signal CESM + kick off next forcing prefetch + async NC save ---
        # All three background operations run during CESM's ocean/ice/coupler compute
        # (several seconds), fully hidden behind CESM's work.
        write_flag(done_flag)
        _prefetch_ix     = _next_forcing_ix(forcing_ix)   # best guess for next step (+6h)
        _prefetch_future = _forcing_executor.submit(_load_forcing_slice, _prefetch_ix)

        # --- (n2) atmospheric NetCDF output (async worker process, optional) ---
        if _save_nc:
            _real_year = model_start_dt.year + (ymd // 10000) - 1
            _real_dt   = datetime(_real_year, (ymd % 10000) // 100, ymd % 100, tod // 3600)
            upper_air, single_level = make_xarray(
                prediction_cpu, _real_dt, _cam_lat, _cam_lon, conf
            )
            if args.save_vars is not None:
                _keep_ua = [v for v in args.save_vars if v in upper_air.coords["vars"].values]
                _keep_sl = [v for v in args.save_vars if v in single_level.coords["vars"].values]
                if _keep_ua:
                    upper_air   = upper_air.sel(vars=_keep_ua)
                if _keep_sl:
                    single_level = single_level.sel(vars=_keep_sl)
            _cesm_yr  = ymd // 10000
            _year_dir = _save_nc_dir / f"{_cesm_yr:04d}"
            _date_str = (f"{_cesm_yr:04d}-{(ymd % 10000) // 100:02d}"
                         f"-{ymd % 100:02d}-{tod:05d}")
            # 6-hourly: skipped when --daily_mean is active (saves storage)
            if not _do_daily:
                pool.apply_async(
                    save_camulator_step_nc,
                    (upper_air, single_level,
                     str(_year_dir / f"camulator.h1.{_date_str}.nc")),
                )
            # Daily mean buffer: accumulate steps; submit when day rolls over
            if _do_daily:
                if _daily_buffer_ymd > 0 and ymd != _daily_buffer_ymd:
                    _prev_d_str = (f"{_daily_buffer_ymd // 10000:04d}-"
                                   f"{(_daily_buffer_ymd % 10000) // 100:02d}-"
                                   f"{_daily_buffer_ymd % 100:02d}")
                    _prev_yr = _save_nc_dir / f"{_daily_buffer_ymd // 10000:04d}"
                    pool.apply_async(
                        save_camulator_daily_nc,
                        (_daily_buffer,
                         str(_prev_yr / f"camulator.h1d.{_prev_d_str}.nc")),
                    )
                    _daily_buffer = []
                _daily_buffer.append((upper_air, single_level))
                _daily_buffer_ymd = ymd
        t_total_s = time.time() - t_step_start
        logger.info(f"  Wrote done.flag   "
                    f"forc={t_forcing_ms:.0f}ms  build={t_build_ms:.0f}ms  "
                    f"inf={t_inf_ms:.0f}ms  itrans={t_itrans_ms:.0f}ms  "
                    f"extract={t_extract_ms:.0f}ms  remap={t_remap_ms:.1f}ms  "
                    f"write={t_write_ms:.0f}ms  total={t_total_s:.2f}s")

        # --- (o) advance CAMulator state ---
        # Cache output for potential reuse if CESM sends a repeated date next step
        _cached_cam_out = (u10, v10, tbot, zbot, tref, qbot, pbot, fsds, flnsd, prect)
        state = stepper.state_manager.shift_state_forward(state, prediction)
        timestep += 1

        # --- (p) save atmosphere restart file ---
        # Written every step so CESM restarts at any walltime boundary can
        # resume the atmosphere from the correct state (not the original IC).
        # ~30 MB write; happens during CESM's inter-step compute time.

        # Annual archive: when the CESM model year rolls over, the current
        # camulator_atm_restart.pth holds the end-of-year-N state (written at
        # the last step of year N).  Copy it before overwriting with year-N+1.
        # e.g. atm_restarts/camulator_atm_restart.year0001.pth
        _current_year = ymd // 10000
        if _prev_step_year > 0 and _current_year != _prev_step_year:
            _archive_name = f"camulator_atm_restart.year{_prev_step_year:04d}.pth"
            _archive_path = _atm_restart_archive_dir / _archive_name
            if atm_restart_file.exists():
                shutil.copy2(atm_restart_file, _archive_path)
                logger.info(f"  Year {_prev_step_year:04d}→{_current_year:04d}: "
                            f"archived ATM restart → atm_restarts/{_archive_name}")
        _prev_step_year = _current_year

        torch.save({
            "state":    state.cpu(),
            "timestep": timestep,
            "last_ymd": ymd,    # CESM date of the go.flag just processed
            "last_tod": tod,    # used to verify date alignment on next restart
            "cam_out":  {       # re-served cleanly on CONTINUE_RUN without re-running inference
                "u10": u10, "v10": v10, "tbot": tbot, "zbot": zbot,
                "tref": tref, "qbot": qbot, "pbot": pbot, "fsds": fsds,
                "flnsd": flnsd, "prect": prect,
            },
        }, atm_restart_file)


if __name__ == "__main__":
    main()
