"""
compute_wb2_stats.py
--------------------
Compute per-variable global mean, std, and xi from the WB2 ERA5 GCS zarr store.

Follows Watt-Meyer et al. (2023) / Schreck et al. (2025) residual normalization.

Two-phase workflow for base stats + xi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 1 — submit one PBS job per year range (parallel across years):

    python scripts/compute_wb2_stats.py compute \
        --start 2000-01-01 --end 2000-12-31 \
        --out_dir /path/to/stats --tag 2000_v2 --partial_only

Phase 2 — merge partial stats into final mean/std:

    python scripts/compute_wb2_stats.py merge \
        --partials /path/to/stats/partial_*_v2.pkl \
        --out_dir /path/to/stats --tag 2000_2018_v2

Phase 3 — compute xi from tendency stats (single PBS job):

    python scripts/compute_wb2_stats.py tendency \
        --std_path /path/to/wb2_era5_1440x721_2000_2018_v2_std.nc \
        --start 2000-01-01 --end 2018-12-31 \
        --out_dir /path/to/stats --tag 2000_2018_v2

Output: wb2_era5_1440x721_{tag}_{mean,std,xi,tend_std}.nc
"""

import argparse
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_WB2_STORES = {
    "1440x721": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    "240x121": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
}
_WB2_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

_VARS_3D = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
    "vertical_velocity",
    "relative_humidity",
    "vorticity",
    "divergence",
    "potential_vorticity",
]
_VARS_2D = [
    "surface_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "sea_ice_cover",
    "snow_depth",
    "total_cloud_cover",
    "boundary_layer_height",
    "total_column_water_vapour",
    "total_column_water",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "mean_top_downward_short_wave_radiation_flux",
    "mean_surface_latent_heat_flux",
    "mean_surface_sensible_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux",
    "mean_vertically_integrated_moisture_divergence",
]

# Variables included in the xi geometric mean (prognostic/dynamical vars only).
# Static and diagnostic vars (radiation fluxes, slow soil/ice vars) are excluded
# from the gmean so they don't skew the normalization scale.
_DYNAMIC_3D = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
    "vertical_velocity",
]
_DYNAMIC_2D = [
    "surface_pressure",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]


def open_store(resolution: str) -> xr.Dataset:
    url = _WB2_STORES[resolution]
    logger.info("Opening WB2 zarr: %s", url)
    fs = gcsfs.GCSFileSystem(token="anon")
    store = zarr.storage.FsspecStore(fs=fs, path=url.split("://", 1)[-1])
    ds = xr.open_zarr(store, consolidated=True)
    time_idx = pd.DatetimeIndex(ds.time.values)
    duplicated = time_idx.duplicated()
    if duplicated.any():
        logger.warning("Dropping %d duplicate timestamps", duplicated.sum())
        ds = ds.isel(time=~duplicated)
    return ds


# ---------------------------------------------------------------------------
# Sum accumulator: tracks n, sum, sum-of-squares over all valid pixels.
# Gives global spatial+temporal mean and std, not just temporal variance of
# the spatial mean.  Merges trivially across PBS jobs (just add components).
# ---------------------------------------------------------------------------


class SumAccumulator:
    """Per-variable accumulator for global mean/std over all pixels."""

    def __init__(self, vars_3d: list[str], vars_2d: list[str], n_levels: int):
        self.n: dict[str, np.ndarray] = {}
        self.s: dict[str, np.ndarray] = {}
        self.s2: dict[str, np.ndarray] = {}
        for v in vars_3d:
            self.n[v] = np.zeros(n_levels, dtype=np.int64)
            self.s[v] = np.zeros(n_levels, dtype=np.float64)
            self.s2[v] = np.zeros(n_levels, dtype=np.float64)
        for v in vars_2d:
            self.n[v] = np.int64(0)
            self.s[v] = np.float64(0.0)
            self.s2[v] = np.float64(0.0)

    @staticmethod
    def merge(a: "SumAccumulator", b: "SumAccumulator") -> "SumAccumulator":
        result = SumAccumulator([], [], 0)
        for v in a.n:
            result.n[v] = a.n[v] + b.n[v]
            result.s[v] = a.s[v] + b.s[v]
            result.s2[v] = a.s2[v] + b.s2[v]
        return result

    def finalize(self) -> tuple[dict, dict]:
        means, stds = {}, {}
        for v in self.n:
            n = self.n[v]
            safe_n = np.where(n > 0, n, 1)
            mean = self.s[v] / safe_n
            # Var = E[x^2] - E[x]^2; clamp to 0 for float rounding noise
            var = np.maximum(self.s2[v] / safe_n - mean**2, 0.0)
            means[v] = mean.astype(np.float32)
            stds[v] = np.sqrt(var).astype(np.float32)
        return means, stds


# ---------------------------------------------------------------------------
# Chunk processor for base stats (runs in a thread)
# ---------------------------------------------------------------------------


def process_chunk(
    ds: xr.Dataset,
    time_positions: np.ndarray,
    level_positions: np.ndarray,
    vars_3d: list[str],
    vars_2d: list[str],
    chunk_idx: int,
    n_chunks: int,
) -> SumAccumulator:
    t0 = pd.Timestamp(ds.time.values[time_positions[0]]).date()
    t1 = pd.Timestamp(ds.time.values[time_positions[-1]]).date()
    logger.info("Chunk %d/%d  %s → %s", chunk_idx + 1, n_chunks, t0, t1)

    # isel with integer positions is thread-safe (no pandas label lookup)
    ds_t = ds.isel(time=time_positions).compute()
    acc = SumAccumulator(vars_3d, vars_2d, len(level_positions))

    for v in vars_3d:
        arr = ds_t[v].isel(level=level_positions).values.astype(np.float64)  # (T, L, lat, lon)
        for li in range(len(level_positions)):
            a = arr[:, li]  # (T, lat, lon)
            valid = ~np.isnan(a)
            acc.n[v][li] += np.int64(valid.sum())
            acc.s[v][li] += float(np.where(valid, a, 0.0).sum())
            acc.s2[v][li] += float(np.where(valid, a**2, 0.0).sum())

    for v in vars_2d:
        arr = ds_t[v].values.astype(np.float64)  # (T, lat, lon)
        valid = ~np.isnan(arr)
        acc.n[v] += np.int64(valid.sum())
        acc.s[v] += np.float64(np.where(valid, arr, 0.0).sum())
        acc.s2[v] += np.float64(np.where(valid, arr**2, 0.0).sum())

    return acc


# ---------------------------------------------------------------------------
# Chunk processor for tendency stats (Phase 3)
# ---------------------------------------------------------------------------


def process_tendency_chunk(
    ds: xr.Dataset,
    time_positions: np.ndarray,  # chunk_size+1 positions → chunk_size pairs
    level_positions: np.ndarray,
    vars_3d: list[str],
    vars_2d: list[str],
    stds: dict,  # {varname: np.ndarray (per-level) or scalar}
    chunk_idx: int,
    n_chunks: int,
) -> SumAccumulator:
    """Accumulate tendency stats for one chunk of consecutive time pairs.

    For each consecutive pair (t, t+1): diff = (arr[t+1] - arr[t]) / sigma(arr).
    Accumulates sum/sum_sq of diff over all valid pixels.
    """
    t0_date = pd.Timestamp(ds.time.values[time_positions[0]]).date()
    t1_date = pd.Timestamp(ds.time.values[time_positions[-1]]).date()
    n_pairs = len(time_positions) - 1
    logger.info("Tendency chunk %d/%d  %s → %s  (%d pairs)", chunk_idx + 1, n_chunks, t0_date, t1_date, n_pairs)

    ds_t = ds.isel(time=time_positions).compute()
    acc = SumAccumulator(vars_3d, vars_2d, len(level_positions))

    for v in vars_3d:
        arr = ds_t[v].isel(level=level_positions).values.astype(np.float64)  # (T, L, lat, lon)
        for li in range(len(level_positions)):
            sigma = max(float(stds[v][li]), 1e-10)
            diff = (arr[1:, li] - arr[:-1, li]) / sigma  # (n_pairs, lat, lon)
            valid = ~np.isnan(diff)
            acc.n[v][li] += np.int64(valid.sum())
            acc.s[v][li] += float(np.where(valid, diff, 0.0).sum())
            acc.s2[v][li] += float(np.where(valid, diff**2, 0.0).sum())

    for v in vars_2d:
        arr = ds_t[v].values.astype(np.float64)  # (T, lat, lon)
        sigma = max(float(stds[v]), 1e-10)
        diff = (arr[1:] - arr[:-1]) / sigma  # (n_pairs, lat, lon)
        valid = ~np.isnan(diff)
        acc.n[v] += np.int64(valid.sum())
        acc.s[v] += np.float64(np.where(valid, diff, 0.0).sum())
        acc.s2[v] += np.float64(np.where(valid, diff**2, 0.0).sum())

    return acc


# ---------------------------------------------------------------------------
# NetCDF I/O
# ---------------------------------------------------------------------------


def save_stats(means: dict, stds: dict, levels: list[int], out_dir: Path, tag: str):
    vars_3d = [v for v in _VARS_3D if v in means]
    vars_2d = [v for v in _VARS_2D if v in means]
    lev_da = xr.DataArray(levels, dims=["level"], attrs={"units": "hPa"})

    def _build(d):
        dvars = {}
        for v in vars_3d:
            dvars[v] = xr.DataArray(d[v], dims=["level"], coords={"level": lev_da})
        for v in vars_2d:
            dvars[v] = xr.DataArray(float(d[v]))
        return xr.Dataset(dvars)

    mean_path = out_dir / f"wb2_era5_1440x721_{tag}_mean.nc"
    std_path = out_dir / f"wb2_era5_1440x721_{tag}_std.nc"
    _build(means).to_netcdf(mean_path)
    _build(stds).to_netcdf(std_path)
    logger.info("Saved %s", mean_path)
    logger.info("Saved %s", std_path)


def save_xi(
    xi: dict,
    tend_stds: dict,
    levels: list[int],
    vars_3d: list[str],
    vars_2d: list[str],
    out_dir: Path,
    tag: str,
) -> None:
    """Save xi and tend_std NetCDF files (same variable structure as mean/std)."""
    lev_da = xr.DataArray(levels, dims=["level"], attrs={"units": "hPa"})

    def _build(d, as_float32=True):
        dvars = {}
        for v in vars_3d:
            if v in d:
                arr = np.array(d[v], dtype=np.float32 if as_float32 else np.float64)
                dvars[v] = xr.DataArray(arr, dims=["level"], coords={"level": lev_da})
        for v in vars_2d:
            if v in d:
                dvars[v] = xr.DataArray(np.float32(float(d[v])))
        return xr.Dataset(dvars)

    xi_path = out_dir / f"wb2_era5_1440x721_{tag}_xi.nc"
    ts_path = out_dir / f"wb2_era5_1440x721_{tag}_tend_std.nc"
    _build(xi).to_netcdf(xi_path)
    _build(tend_stds).to_netcdf(ts_path)
    logger.info("Saved xi       → %s", xi_path)
    logger.info("Saved tend_std → %s", ts_path)


def save_partial(acc: SumAccumulator, out_dir: Path, tag: str) -> Path:
    p = out_dir / f"partial_{tag}.pkl"
    with open(p, "wb") as f:
        pickle.dump(acc, f)
    logger.info("Saved partial state %s", p)
    return p


def merge_partials(partial_paths: list[Path], out_dir: Path, tag: str, levels: list[int]):
    logger.info("Merging %d partial files", len(partial_paths))
    combined = None
    for p in partial_paths:
        with open(p, "rb") as f:
            acc = pickle.load(f)
        combined = acc if combined is None else SumAccumulator.merge(combined, acc)
    means, stds = combined.finalize()
    for v in sorted(means):
        print(f"  {v:<55}  mean={float(np.mean(means[v])):+12.4g}  std={float(np.mean(stds[v])):10.4g}")
    save_stats(means, stds, levels, out_dir, tag)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # ------------------------------------------------------------------
    # Phase 1: compute base stats for one year range
    # ------------------------------------------------------------------
    run = sub.add_parser("compute", help="Compute base stats for a time range")
    run.add_argument("--start", default="2000-01-01")
    run.add_argument("--end", default="2000-12-31")
    run.add_argument("--resolution", default="1440x721", choices=list(_WB2_STORES))
    run.add_argument("--levels", nargs="+", type=int, default=_WB2_LEVELS)
    run.add_argument("--chunk_size", type=int, default=20)
    run.add_argument("--n_workers", type=int, default=4)
    run.add_argument("--out_dir", type=Path, default=Path("/glade/campaign/cisl/aiml/credit/static_scalers"))
    run.add_argument("--tag", default=None)
    run.add_argument("--partial_only", action="store_true")

    # ------------------------------------------------------------------
    # Phase 2: merge partial pkl files into final mean/std NetCDF
    # ------------------------------------------------------------------
    mg = sub.add_parser("merge", help="Merge partial pkl files into final NetCDF")
    mg.add_argument("--partials", nargs="+", type=Path, required=True)
    mg.add_argument("--out_dir", type=Path, default=Path("/glade/campaign/cisl/aiml/credit/static_scalers"))
    mg.add_argument("--tag", default="merged")
    mg.add_argument("--levels", nargs="+", type=int, default=_WB2_LEVELS)

    # ------------------------------------------------------------------
    # Phase 3: compute xi from tendency stats (single job over full range)
    # ------------------------------------------------------------------
    td = sub.add_parser("tendency", help="Compute xi from tendency stats (Phase 3)")
    td.add_argument(
        "--std_path", type=Path, required=True, help="Base std NetCDF from Phase 2 (e.g. wb2_era5_..._std.nc)"
    )
    td.add_argument("--start", default="2000-01-01")
    td.add_argument("--end", default="2018-12-31")
    td.add_argument("--resolution", default="1440x721", choices=list(_WB2_STORES))
    td.add_argument("--levels", nargs="+", type=int, default=_WB2_LEVELS)
    td.add_argument(
        "--chunk_size", type=int, default=20, help="Consecutive pairs per chunk (chunk loads chunk_size+1 timesteps)"
    )
    td.add_argument("--n_workers", type=int, default=4)
    td.add_argument("--out_dir", type=Path, default=Path("/glade/campaign/cisl/aiml/credit/static_scalers"))
    td.add_argument("--tag", default="2000_2018_v2")

    args = p.parse_args()

    # ------------------------------------------------------------------
    # Phase 2: merge
    # ------------------------------------------------------------------
    if args.cmd == "merge":
        merge_partials(args.partials, args.out_dir, args.tag, args.levels)
        return

    # ------------------------------------------------------------------
    # Phase 3: tendency / xi
    # ------------------------------------------------------------------
    if args.cmd == "tendency":
        logger.info("Phase 3 — tendency stats and xi")
        logger.info("std_path  = %s", args.std_path)
        logger.info("range     = %s → %s", args.start, args.end)

        # Load base stds for normalization
        ds_std = xr.open_dataset(args.std_path)
        stds: dict[str, np.ndarray] = {}
        for v in ds_std.data_vars:
            stds[v] = np.array(ds_std[v].values)

        ds = open_store(args.resolution)
        available = set(ds.data_vars)
        vars_3d = [v for v in _VARS_3D if v in available and v in stds]
        vars_2d = [v for v in _VARS_2D if v in available and v in stds]
        skip = [v for v in _VARS_3D + _VARS_2D if v not in available or v not in stds]
        if skip:
            logger.warning("Skipping (not in zarr or missing std): %s", skip)

        ds_times = pd.DatetimeIndex(ds.time.values)
        times = pd.date_range(args.start, args.end, freq="6h")
        time_pos = np.where(ds_times.isin(times))[0]
        level_pos = np.where(np.isin(ds.level.values, args.levels))[0]
        n_pairs = len(time_pos) - 1
        logger.info("%d timesteps → %d consecutive pairs", len(time_pos), n_pairs)

        # Build overlapping chunks: [0:B+1], [B:2B+1], [2B:3B+1], ...
        # Each chunk loads B+1 timesteps and yields B pairs.
        B = args.chunk_size
        chunks = []
        for i in range(0, len(time_pos) - 1, B):
            end = min(i + B + 1, len(time_pos))
            chunks.append(time_pos[i:end])
        n = len(chunks)
        logger.info("%d chunks × chunk_size=%d × %d workers", n, B, args.n_workers)

        acc = SumAccumulator(vars_3d, vars_2d, len(level_pos))
        with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {
                pool.submit(process_tendency_chunk, ds, ch, level_pos, vars_3d, vars_2d, stds, i, n): i
                for i, ch in enumerate(chunks)
            }
            for fut in as_completed(futures):
                acc = SumAccumulator.merge(acc, fut.result())

        _, tend_stds = acc.finalize()

        # Compute xi: gmean over prognostic/dynamical variable-levels
        ts_for_gmean = []
        for v in _DYNAMIC_3D:
            if v in tend_stds:
                for val in np.array(tend_stds[v]).ravel():
                    if float(val) > 1e-6:
                        ts_for_gmean.append(float(val))
        for v in _DYNAMIC_2D:
            if v in tend_stds:
                val = float(tend_stds[v])
                if val > 1e-6:
                    ts_for_gmean.append(val)

        if ts_for_gmean:
            gmean = float(np.exp(np.mean(np.log(ts_for_gmean))))
            logger.info("Tendency gmean = %.4g over %d dynamic variable-levels", gmean, len(ts_for_gmean))
        else:
            gmean = 1.0
            logger.warning("No valid tendency stds — xi=1 everywhere")

        xi: dict[str, np.ndarray | np.floating] = {}
        for v in vars_3d:
            arr = np.array(tend_stds[v], dtype=np.float64)
            xi[v] = (arr / gmean).astype(np.float32)
        for v in vars_2d:
            xi[v] = np.float32(float(tend_stds[v]) / gmean)

        # Summary table
        print(f"\n  {'variable':<55}  {'tend_std':>10}  {'xi':>8}")
        print(f"  {'-' * 55}  {'-' * 10}  {'-' * 8}")
        for v in sorted(vars_3d + vars_2d):
            if v in tend_stds:
                ts_val = float(np.mean(tend_stds[v]))
                xi_val = float(np.mean(xi[v]))
                dyn_tag = " *" if (v in _DYNAMIC_3D or v in _DYNAMIC_2D) else ""
                print(f"  {v:<55}  {ts_val:10.4g}  {xi_val:8.4f}{dyn_tag}")
        print(f"\n  (* = included in xi gmean;  gmean = {gmean:.4g})")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        save_xi(xi, tend_stds, args.levels, vars_3d, vars_2d, args.out_dir, args.tag)
        return

    # ------------------------------------------------------------------
    # Phase 1: compute base stats for one year range
    # ------------------------------------------------------------------
    if args.cmd != "compute":
        p.print_help()
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.start}_{args.end}"

    ds = open_store(args.resolution)
    available = set(ds.data_vars)
    vars_3d = [v for v in _VARS_3D if v in available]
    vars_2d = [v for v in _VARS_2D if v in available]
    skip = [v for v in _VARS_3D + _VARS_2D if v not in available]
    if skip:
        logger.warning("Not in zarr, skipping: %s", skip)

    ds_times = pd.DatetimeIndex(ds.time.values)
    times = pd.date_range(args.start, args.end, freq="6h")
    time_pos = np.where(ds_times.isin(times))[0]
    level_pos = np.where(np.isin(ds.level.values, args.levels))[0]
    logger.info("%d time steps selected", len(time_pos))

    acc = SumAccumulator(vars_3d, vars_2d, len(level_pos))
    pos_batches = [time_pos[i : i + args.chunk_size] for i in range(0, len(time_pos), args.chunk_size)]
    n = len(pos_batches)
    logger.info("%d chunks × %d workers", n, args.n_workers)

    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(process_chunk, ds, b, level_pos, vars_3d, vars_2d, i, n): i for i, b in enumerate(pos_batches)
        }
        for fut in as_completed(futures):
            chunk_acc = fut.result()
            acc = SumAccumulator.merge(acc, chunk_acc)

    if args.partial_only:
        save_partial(acc, args.out_dir, tag)
    else:
        means, stds = acc.finalize()
        for v in sorted(means):
            print(f"  {v:<55}  mean={float(np.mean(means[v])):+12.4g}  std={float(np.mean(stds[v])):10.4g}")
        save_stats(means, stds, args.levels, args.out_dir, tag)


if __name__ == "__main__":
    main()
