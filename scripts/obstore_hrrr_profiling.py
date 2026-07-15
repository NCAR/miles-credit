import argparse
import asyncio
import hashlib
import logging
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


def get_byte_ranges_for_source(store: obs.store.S3Store, s3_entry_name: str, source_cfg: dict) -> list[tuple[int, int]]:
    """Retrieve the idx sidecar, parse the variable configuration, and return exclusive byte ranges."""
    print(f"Fetching and parsing idx sidecar for S3 entry: {s3_entry_name}")
    idx_entries = _fetch_obstore_idx(store, s3_entry_name)

    levels = source_cfg.get("levels")
    if levels is None:
        print(
            "[NOTIFICATION] Default value employed: source config does not specify 'levels'. Defaulting to all available levels."
        )

    product = source_cfg.get("product", "wrfprs")
    if "product" not in source_cfg:
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
                print(f"[NOTIFICATION] Warning: Variable {vname} not found in VAR_REGISTRY.")
                continue
            reg = VAR_REGISTRY[vname]
            if product == "wrfsubh":
                # wrfsubh requires step_min
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

    if has_none_end:
        print(
            "[NOTIFICATION] Fallback logic: One or more GRIB messages had byte_end as None (EOF). Used obstore.head metadata to resolve the file size."
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
    path: str,
    byte_ranges: list[tuple[int, int]],
    trials: int,
    workers: int,
) -> dict[str, list[float]]:
    """Run warm-ups followed by multiple trials for all retrieval strategies."""
    strategies = {
        "get_range (Sequential)": lambda: profile_get_range_sequential(store, path, byte_ranges),
        "get_range (Parallel, ThreadPool)": lambda: profile_get_range_parallel(
            store, path, byte_ranges, max_workers=workers
        ),
        "get_range_async (asyncio.gather)": lambda: profile_get_range_async(store, path, byte_ranges),
        "get_ranges (Sync)": lambda: profile_get_ranges(store, path, byte_ranges),
        "get_ranges_async (Async)": lambda: profile_get_ranges_async(store, path, byte_ranges),
    }

    raw_data = b"".join(obs.get_range(store, path, start=r[0], end=r[1]).to_bytes() for r in byte_ranges)
    expected_hash = hashlib.sha256(raw_data).hexdigest()

    timings = {name: [] for name in strategies}

    for name, func in strategies.items():
        print(f"\n--- Method: {name} ---")
        print("Running warm-up trial...")
        try:
            warm_time, warm_res = func()
            warm_hash = hashlib.sha256(b"".join(warm_res)).hexdigest()
            if warm_hash != expected_hash:
                print(f"Warning: Hash mismatch in warm-up for {name}!")
            else:
                print(f"Warm-up trial took {warm_time:.4f} seconds (verified)")
        except Exception as e:
            print(f"Error during warm-up for {name}: {e}")
            continue

        for i in range(trials):
            print(f"Running timed trial {i + 1}/{trials}...")
            trial_time, trial_res = func()
            trial_hash = hashlib.sha256(b"".join(trial_res)).hexdigest()
            if trial_hash != expected_hash:
                print(f"Warning: Hash mismatch in trial {i + 1}!")
            timings[name].append(trial_time)

    return timings


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

    # 1. Config loading with fallbacks
    if args.config is None:
        default_config = "config/hrrr_emulator/sf_hrrr_emulator_test_small_workers.yml"
        print(f"[NOTIFICATION] Fallback logic: No config path provided. Defaulting to: {default_config}")
        config_path = default_config
    else:
        config_path = args.config

    config = load_config(config_path)

    # 2. Date loading with fallbacks
    if args.date is None:
        default_date = config["data"].get("start_datetime", "2024-01-01")
        print(
            f"[NOTIFICATION] Fallback logic: No profiling date provided. Defaulting to start_datetime from config: {default_date}"
        )
        date_str = default_date
    else:
        date_str = args.date

    t = pd.Timestamp(date_str)

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

    # Construct file path on S3
    s3_bucket = "noaa-hrrr-bdp-pds"
    s3_entry_name = _hrrr_s3_entry_name(t, forecast_hour, product)

    print(f"Initializing obstore connection to S3 bucket: {s3_bucket}")
    store = _start_s3_obstore(s3_bucket)

    # 4. Extract byte ranges
    byte_ranges = get_byte_ranges_for_source(store, s3_entry_name, source_cfg)
    total_bytes = sum(end - start for start, end in byte_ranges)
    total_mb = total_bytes / (1024 * 1024)

    print("\nConfiguration summary:")
    print(f"  GRIB2 Key:   {s3_entry_name}")
    print(f"  Messages:    {len(byte_ranges)}")
    print(f"  Total Size:  {total_mb:.4f} MB")

    # 5. Run benchmarks
    timings = run_benchmarks(store, s3_entry_name, byte_ranges, args.trials, args.workers)

    # 6. Report results
    print("\n" + "=" * 80)
    print(f"{'Method Comparison (Timed Trials)':^80}")
    print("=" * 80)
    header = f"{'Method':<35} | {'Mean Time':<11} | {'Std Dev':<10} | {'Throughput':<15}"
    print(header)
    print("-" * 80)

    for name, times in timings.items():
        if not times:
            continue
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = total_mb / mean_time
        print(f"{name:<35} | {mean_time:8.4f} s | {std_time:8.4f} s | {throughput:8.4f} MB/s")
    print("=" * 80)


if __name__ == "__main__":
    print()
    main()
    print()
