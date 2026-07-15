import argparse
import asyncio
import hashlib
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml

import obstore as obs
from credit.datasets._utils import _start_s3_obstore
from credit.datasets.hrrr import (
    VAR_REGISTRY,
    _build_nat_entry_map,
    _build_prs_entry_map,
    _fetch_obstore_idx,
    _hrrr_s3_entry_name,
    _resolve_nat_levels,
    _resolve_pressure_levels,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_hrrr_source(config: dict) -> tuple[str, dict]:
    """Find the first source with hrrr dataset_type in the configuration."""
    sources = config.get("data", {}).get("source", {})
    for name, source_cfg in sources.items():
        if source_cfg.get("dataset_type", "").lower() == "hrrr":
            return name, source_cfg

    raise ValueError(
        "No HRRR source found in the config under data.source. Please provide a configuration file containing an HRRR dataset source."
    )


def generate_profiling_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    timestep_str: str,
    count: int,
    sample_random: bool = False,
    seed: int = 42,
) -> list[pd.Timestamp]:
    """Generate list of datetimes starting from start_date, incremented by timestep or sampled randomly."""
    freq = timestep_str
    if freq == "15min":
        freq = "15T"
    try:
        # Generate the full range of datetimes
        full_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    except Exception as e:
        print(
            f"[NOTIFICATION] Fallback logic: Failed to generate date range with frequency '{timestep_str}'. Defaulting to '1H'. Error: {e}"
        )
        full_range = pd.date_range(start=start_date, end=end_date, freq="1H")

    if sample_random:
        if len(full_range) < count:
            print(
                f"[NOTIFICATION] Fallback logic: Requested count {count} is larger than the total available dates in the range ({len(full_range)}). Defaulting to returning all available dates sequentially."
            )
            return list(full_range)
        # Set seed for reproducibility
        random.seed(seed)
        # sampled = sorted(random.sample(list(full_range), count))
        sampled = random.sample(list(full_range), count)
        return sampled
    else:
        # Sequential from start_date
        return list(full_range[:count])


def get_byte_ranges_for_source(
    store: obs.store.S3Store, s3_entry_name: str, source_cfg: dict, verbose: bool = False
) -> list[tuple[int, int]]:
    """Retrieve the idx sidecar, parse the variable configuration, and return exclusive byte ranges."""
    if verbose:
        print(f"Fetching and parsing idx sidecar for S3 entry: {s3_entry_name}")
    idx_entries = _fetch_obstore_idx(store, s3_entry_name)

    levels = source_cfg.get("levels")
    if levels is None:
        if verbose:
            print(
                "[NOTIFICATION] Default value employed: source config does not specify 'levels'. Defaulting to all available levels."
            )

    product = source_cfg.get("product", "wrfprs")
    if "product" not in source_cfg:
        if verbose:
            print(
                "[NOTIFICATION] Default value employed: source config does not specify 'product'. Defaulting to 'wrfprs'."
            )

    variables_block = source_cfg.get("variables", {})

    fetch_plan = []

    # Iterate over all field types
    for field_type, field_config in variables_block.items():
        if not isinstance(field_config, dict):
            continue

        vars_3d = field_config.get("vars_3D") or []
        vars_2d = field_config.get("vars_2D") or []

        # 3D Variables
        for vname in vars_3d:
            if vname not in VAR_REGISTRY:
                if verbose:
                    print(f"[NOTIFICATION] Warning: Variable {vname} not found in VAR_REGISTRY.")
                continue
            reg = VAR_REGISTRY[vname]
            if product == "wrfnat":
                nat_map = _build_nat_entry_map(idx_entries, reg["idx_name"])
                resolved_levels = _resolve_nat_levels(levels, nat_map, vname)
                for lv in resolved_levels:
                    fetch_plan.append(nat_map[lv])
            else:
                prs_map = _build_prs_entry_map(idx_entries, reg["idx_name"])
                resolved_levels = _resolve_pressure_levels(levels, prs_map, vname)
                for lv in resolved_levels:
                    fetch_plan.append(prs_map[lv])

        # 2D Variables
        for vname in vars_2d:
            if vname not in VAR_REGISTRY:
                if verbose:
                    print(f"[NOTIFICATION] Warning: Variable {vname} not found in VAR_REGISTRY.")
                continue
            reg = VAR_REGISTRY[vname]
            if product == "wrfsubh":
                if verbose:
                    print("[NOTIFICATION] Default value employed: step_min for wrfsubh defaults to 15.")
                step_min = 15
                from credit.datasets.hrrr import _find_subhf_entry

                entry = _find_subhf_entry(idx_entries, reg["idx_name"], reg["idx_level"], step_min)
                fetch_plan.append(entry)
            else:
                matching = [e for e in idx_entries if e["var"] == reg["idx_name"] and e["level"] == reg["idx_level"]]
                if not matching:
                    raise KeyError(f"No idx entry found for 2D variable {vname}")
                fetch_plan.append(matching[0])

    # Convert to exclusive byte ranges (start, end)
    byte_ranges = []
    has_none_end = False
    for entry in fetch_plan:
        start = entry["byte_start"]
        end = entry["byte_end"]
        if end is None:
            has_none_end = True
            # Query object size using obstore.head to resolve EOF range
            meta = obs.head(store, s3_entry_name)
            end = meta.size
        else:
            end = end + 1  # end parameter in obstore is exclusive
        byte_ranges.append((start, end))

    if has_none_end and verbose:
        print(
            "[NOTIFICATION] Fallback logic: One or more GRIB messages had byte_end as None (EOF). Used obstore.head to resolve file size."
        )

    return byte_ranges


# ------------------------------------------------------------------
# Retrieval Implementations
# ------------------------------------------------------------------


def profile_get_range_sequential(
    store: obs.store.S3Store, path: str, byte_ranges: list[tuple[int, int]]
) -> tuple[float, list[bytes]]:
    """Synchronous sequential get_range requests."""
    t0 = time.perf_counter()
    results = []
    for start, end in byte_ranges:
        res = obs.get_range(store, path, start=start, end=end)
        results.append(res.to_bytes())
    t1 = time.perf_counter()
    return t1 - t0, results


def profile_get_range_parallel(
    store: obs.store.S3Store,
    path: str,
    byte_ranges: list[tuple[int, int]],
    max_workers: int = 8,
) -> tuple[float, list[bytes]]:
    """Synchronous parallel get_range requests using a ThreadPoolExecutor."""
    t0 = time.perf_counter()

    def fetch_one(r: tuple[int, int]) -> bytes:
        res = obs.get_range(store, path, start=r[0], end=r[1])
        return res.to_bytes()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_one, byte_ranges))

    t1 = time.perf_counter()
    return t1 - t0, results


async def fetch_range_async(store, path, start, end) -> bytes:
    res = await obs.get_range_async(store, path, start=start, end=end)
    return res.to_bytes()


async def get_range_async_gather(store, path, byte_ranges) -> list[bytes]:
    tasks = [fetch_range_async(store, path, start, end) for start, end in byte_ranges]
    return await asyncio.gather(*tasks)


def profile_get_range_async(
    store: obs.store.S3Store, path: str, byte_ranges: list[tuple[int, int]]
) -> tuple[float, list[bytes]]:
    """Asynchronous parallel get_range_async requests using asyncio.gather."""
    t0 = time.perf_counter()
    results = asyncio.run(get_range_async_gather(store, path, byte_ranges))
    t1 = time.perf_counter()
    return t1 - t0, results


def profile_get_ranges(
    store: obs.store.S3Store, path: str, byte_ranges: list[tuple[int, int]]
) -> tuple[float, list[bytes]]:
    """Synchronous multi-range request via get_ranges."""
    starts = [r[0] for r in byte_ranges]
    ends = [r[1] for r in byte_ranges]
    t0 = time.perf_counter()
    results = obs.get_ranges(store, path, starts=starts, ends=ends)
    bytes_results = [res.to_bytes() for res in results]
    t1 = time.perf_counter()
    return t1 - t0, bytes_results


async def get_ranges_async_call(store, path, starts, ends) -> list[bytes]:
    results = await obs.get_ranges_async(store, path, starts=starts, ends=ends)
    return [res.to_bytes() for res in results]


def profile_get_ranges_async(
    store: obs.store.S3Store, path: str, byte_ranges: list[tuple[int, int]]
) -> tuple[float, list[bytes]]:
    """Asynchronous multi-range request via get_ranges_async."""
    starts = [r[0] for r in byte_ranges]
    ends = [r[1] for r in byte_ranges]
    t0 = time.perf_counter()
    results = asyncio.run(get_ranges_async_call(store, path, starts, ends))
    t1 = time.perf_counter()
    return t1 - t0, results


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------


def run_benchmarks(
    store: obs.store.S3Store,
    dates: list[pd.Timestamp],
    source_cfg: dict,
    forecast_hour: int,
    product: str,
    trials: int,
    workers: int,
) -> tuple[dict[str, list[float]], float]:
    """Run warm-up followed by timed trials over unique dates."""
    strategies = {
        "get_range (Sequential)": lambda br, path: profile_get_range_sequential(store, path, br),
        "get_range (Parallel, ThreadPool)": lambda br, path: profile_get_range_parallel(
            store, path, br, max_workers=workers
        ),
        "get_range_async (asyncio.gather)": lambda br, path: profile_get_range_async(store, path, br),
        "get_ranges (Sync)": lambda br, path: profile_get_ranges(store, path, br),
        "get_ranges_async (Async)": lambda br, path: profile_get_ranges_async(store, path, br),
    }

    # 1. Warm-up Trial
    warmup_date = dates[0]
    print(f"Running warm-up on date: {warmup_date.strftime('%Y-%m-%d %H:%M:%SZ')}")
    s3_entry_name = _hrrr_s3_entry_name(warmup_date, forecast_hour, product)
    byte_ranges = get_byte_ranges_for_source(store, s3_entry_name, source_cfg, verbose=True)

    total_bytes = sum(end - start for start, end in byte_ranges)
    total_mb = total_bytes / (1024 * 1024)

    print("Executing warm-up for all strategies...")
    warmup_hashes = {}
    for name, func in strategies.items():
        try:
            t_el, res = func(byte_ranges, s3_entry_name)
            warmup_hashes[name] = hashlib.sha256(b"".join(res)).hexdigest()
            print(f"  - {name}: {t_el:.4f}s")
        except Exception as e:
            print(f"  - {name} FAILED: {e}")
            raise e

    # Verify identical bytes across all strategies
    if len(set(warmup_hashes.values())) > 1:
        raise ValueError(f"Hash mismatch during warm-up! Retrieval is inconsistent: {warmup_hashes}")
    print("Warm-up completed successfully. All strategies verified identical bytes.\n")

    # 2. Timed Trials
    print(f"Starting {trials} timed trials over unique datetimes...")
    timings = {name: [] for name in strategies}

    for i in range(1, trials + 1):
        date = dates[i]
        s3_path = _hrrr_s3_entry_name(date, forecast_hour, product)
        br = get_byte_ranges_for_source(store, s3_path, source_cfg, verbose=False)

        trial_results = {}
        trial_hashes = {}

        for name, func in strategies.items():
            t_el, res = func(br, s3_path)
            timings[name].append(t_el)
            trial_results[name] = t_el
            trial_hashes[name] = hashlib.sha256(b"".join(res)).hexdigest()

        # Check for consistency within trial
        if len(set(trial_hashes.values())) > 1:
            print(f"[WARNING] Trial {i} hash mismatch! Hashes: {trial_hashes}")

        # Compact print format: one line per trial
        log_line = (
            f"Trial {i:2d}/{trials} (Date: {date.strftime('%Y-%m-%d %H:%M:%SZ')}) - {len(br)} ranges: "
            f"seq={trial_results['get_range (Sequential)']:6.3f}s | "
            f"pool={trial_results['get_range (Parallel, ThreadPool)']:6.3f}s | "
            f"gather={trial_results['get_range_async (asyncio.gather)']:6.3f}s | "
            f"sync_ranges={trial_results['get_ranges (Sync)']:6.3f}s | "
            f"async_ranges={trial_results['get_ranges_async (Async)']:6.3f}s"
        )
        print(log_line)

    return timings, total_mb


def main():
    parser = argparse.ArgumentParser(description="Profile obstore S3 retrieval strategies for HRRR GRIB2 datasets.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the model/data configuration YAML file.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Initialization date for profiling (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End datetime limit for random sampling (defaults to end_datetime in config).",
    )
    parser.add_argument(
        "--sample-random",
        action="store_true",
        help="Sample random datetimes between start and end date instead of sequential.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of timed trials to run (default: 3).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of threads/workers for parallel get_range (default: 8).",
    )
    args = parser.parse_args()

    # 1. Config loading
    if args.config is None:
        default_config = "config/hrrr_emulator/sf_hrrr_emulator_test_small_workers.yml"
        print(f"[NOTIFICATION] Fallback logic: No config path provided. Defaulting to: {default_config}")
        config_path = default_config
    else:
        config_path = args.config

    config = load_config(config_path)

    # 2. Date loading
    if args.date is None:
        default_date = config["data"].get("start_datetime", "2024-01-01")
        print(
            f"[NOTIFICATION] Fallback logic: No profiling date provided. Defaulting to start_datetime from config: {default_date}"
        )
        date_str = default_date
    else:
        date_str = args.date

    start_date = pd.Timestamp(date_str)
    timestep = config["data"].get("timestep", "1h")

    if args.end_date is None:
        end_date_val = config["data"].get("end_datetime")
        if end_date_val is None:
            default_end = (start_date + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            print(
                f"[NOTIFICATION] Fallback logic: No end datetime found in config or args. Defaulting to 30 days after start date: {default_end}"
            )
            end_date = start_date + pd.Timedelta(days=30)
        else:
            end_date = pd.Timestamp(end_date_val)
    else:
        end_date = pd.Timestamp(args.end_date)

    # Generate dates list: 1 for warm-up + trials
    dates = generate_profiling_dates(
        start_date, end_date, timestep, count=args.trials + 1, sample_random=args.sample_random
    )

    # 3. Source config and S3 initialization
    source_name, source_cfg = get_hrrr_source(config)
    print(f"Using source: {source_name}")

    forecast_hour = source_cfg.get("forecast_hour")
    if forecast_hour is None:
        print(
            "[NOTIFICATION] Default value employed: forecast_hour is not specified in source config. Defaulting to 0."
        )
        forecast_hour = 0
    else:
        forecast_hour = int(forecast_hour)

    product = source_cfg.get("product", "wrfprs")

    # Extract target variables for print summary
    vars_3d = source_cfg.get("variables", {}).get("prognostic", {}).get("vars_3D") or []
    vars_2d = source_cfg.get("variables", {}).get("prognostic", {}).get("vars_2D") or []
    levels = source_cfg.get("levels") or []

    print("\nProfiling Target Setup:")
    print(f"  3D Variables: {vars_3d} (Levels: {levels})")
    print(f"  2D Variables: {vars_2d}")
    print(f"  Timestep:     {timestep}")
    print(f"  Sampling:     {'Random' if args.sample_random else 'Sequential'}")
    print(f"  Total Dates:  {len(dates)} (1 warm-up + {args.trials} trials)")

    # Construct file path on S3 and initialize
    s3_bucket = "noaa-hrrr-bdp-pds"
    print(f"Initializing obstore connection to S3 bucket: {s3_bucket}")
    store = _start_s3_obstore(s3_bucket)

    # 4. Run benchmarks
    timings, total_mb = run_benchmarks(store, dates, source_cfg, forecast_hour, product, args.trials, args.workers)

    # 5. Report results
    print("\n" + "=" * 90)
    print(f"{'PROFILING CONFIGURATION SUMMARY':^90}")
    print("=" * 90)
    print(f"  Configuration File:   {config_path}")
    print(f"  HRRR Product:         {product}")
    print(f"  3D Variables:         {vars_3d}")
    print(f"  2D Variables:         {vars_2d}")
    print(f"  Levels:               {levels}")
    print(f"  Total Timed Trials:   {args.trials}")
    print(f"  Sampling Strategy:    {'Random' if args.sample_random else 'Sequential'}")
    print(f"  ThreadPool Workers:   {args.workers}")
    print(f"  Avg Data Size/Trial:  {total_mb:.4f} MB")
    print("=" * 90)
    print(f"{'Method Comparison (Timed Trials)':^90}")
    print("=" * 90)
    header = f"{'Method':<35} | {'Mean Time':<10} | {'Std Dev':<10} | {'Throughput':<14} | {'Speedup':<8}"
    print(header)
    print("-" * 90)

    # Baseline for speedup comparison is "get_range (Sequential)"
    seq_mean_time = np.mean(timings.get("get_range (Sequential)", [1.0]))

    for name, times in timings.items():
        if not times:
            continue
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = total_mb / mean_time
        speedup = seq_mean_time / mean_time
        print(f"{name:<35} | {mean_time:8.4f} s | {std_time:8.4f} s | {throughput:9.4f} MB/s | {speedup:7.2f}x")
    print("=" * 90)


if __name__ == "__main__":
    print()
    print("Starting obstore profiling...")
    main()
    print()
