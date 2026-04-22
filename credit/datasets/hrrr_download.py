"""
hrrr_download.py
----------------
Standalone utility for downloading HRRR prs GRIB2 data from AWS S3 to local disk.

Downloads are embarrassingly parallel: each timestamp is an independent task
dispatched to a ``multiprocessing.Pool``.  Both the grib2 file and its ``.idx``
sidecar are downloaded so that ``HRRRDataset`` in local mode can use byte-range
reads rather than scanning the full file.

Downloaded files follow the native HRRR directory layout used by ``HRRRDataset``
in local mode, so they are immediately usable without any renaming::

    v3/v4 (2018-07-12+): {base_path}/hrrr.{YYYYMMDD}/conus/hrrr.t{HH}z.wrfprsf{FF:02d}.grib2
    v1/v2 (before):      {base_path}/hrrr.{YYYYMMDD}/hrrr.t{HH}z.wrfprsf{FF:02d}.grib2

After downloading, switch ``mode`` to ``"local"`` in the config.

Usage::

    python -m credit.datasets.hrrr_download -c config/my_conf.yaml --num-workers 8

Or programmatically::

    from credit.datasets.hrrr_download import download_hrrr
    download_hrrr(config['data'], num_workers=8, overwrite=False)

Config section used (``data.source.HRRR``)::

    data:
      source:
        HRRR:
          mode: "local"          # mode to use after download
          base_path: "/data/hrrr"
          forecast_hour: 0
      start_datetime: "2022-01-01"
      end_datetime:   "2022-01-31"
      timestep:       "1h"
      forecast_len:   0
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import NamedTuple

import pandas as pd

from credit.datasets.hrrr import _hrrr_local_path, _hrrr_s3_uri

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-task helper (must be a module-level function for multiprocessing.Pool)
# ---------------------------------------------------------------------------


class _DownloadTask(NamedTuple):
    s3_uri: str
    local_path: str
    overwrite: bool


def _download_one(task: _DownloadTask) -> str:
    """Download one grib2 + .idx pair.  Returns a status string for logging.

    Runs in a worker process — imports s3fs locally so the pool workers don't
    need to inherit an open filesystem object from the parent.
    """
    import s3fs  # noqa: PLC0415

    if os.path.exists(task.local_path) and not task.overwrite:
        return f"skip  {task.local_path}"

    os.makedirs(os.path.dirname(task.local_path), exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    s3_key = task.s3_uri[5:]  # strip "s3://"
    try:
        fs.get(s3_key, task.local_path)
        fs.get(s3_key + ".idx", task.local_path + ".idx")
        return f"ok    {task.local_path}"
    except FileNotFoundError:
        return f"miss  {task.s3_uri}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_hrrr(
    config: dict,
    product: str = "wrfprsf",
    num_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """Download HRRR grib2 + .idx files from AWS S3 to local disk.

    Each timestamp is downloaded in parallel using a ``multiprocessing.Pool``.
    Both the grib2 file and its ``.idx`` sidecar are fetched so that
    ``HRRRDataset`` in ``mode: "local"`` can use fast byte-range reads.

    Args:
        config: Top-level ``data`` config dict (same object passed to
            ``HRRRDataset``).
        product: HRRR product to download — ``"wrfprsf"`` (pressure-level,
            default), ``"wrfnatf"`` (native/hybrid-sigma), or ``"wrfsubhf"``
            (15-min sub-hourly surface).  For ``"wrfsubhf"`` the full set of
            FF files implied by the date-range and timestep is computed
            automatically (one file per 60-min block).
        num_workers: Number of parallel download workers.  Each worker opens
            its own ``s3fs`` connection.  Default ``4``.
        overwrite: Re-download files that already exist on disk. Default
            ``False`` (skip existing files).

    Raises:
        ImportError: If ``s3fs`` is not installed.
        KeyError: If the config is missing required fields.
        ValueError: If *product* is not a recognised HRRR product.
    """
    from credit.datasets.hrrr import VALID_PRODUCTS  # noqa: PLC0415

    if product not in VALID_PRODUCTS:
        raise ValueError(f"Unknown product '{product}'. Valid: {sorted(VALID_PRODUCTS)}")

    try:
        import s3fs  # noqa: PLC0415

        del s3fs
    except ImportError as exc:
        raise ImportError("s3fs is required for downloading: pip install s3fs") from exc

    # Determine which config block to read (HRRR / HRRR_NAT / HRRR_SUBH)
    _config_key_map = {"wrfprsf": "HRRR", "wrfnatf": "HRRR_NAT", "wrfsubhf": "HRRR_SUBH"}
    config_key = _config_key_map[product]
    source_cfg = config["source"][config_key]
    base_path: str = source_cfg["base_path"]
    forecast_hour: int = int(source_cfg.get("forecast_hour", 0))

    dt = pd.Timedelta(config["timestep"])
    num_steps: int = config.get("forecast_len", 0)
    timestamps = pd.date_range(
        pd.Timestamp(config["start_datetime"]),
        pd.Timestamp(config["end_datetime"]) - num_steps * dt,
        freq=dt,
    )

    if product == "wrfsubhf":
        # For sub-hourly, derive the unique set of (init_hour, ff) file pairs
        # implied by the requested timestamps rather than using forecast_hour directly.
        seen: set[tuple] = set()
        file_pairs: list[tuple] = []
        for t in timestamps:
            init = t.floor("1h")
            mins = int((t - init).total_seconds() / 60)
            if mins == 0:
                init = init - pd.Timedelta("1h")
                mins = 60
            ff = (mins + 59) // 60
            key = (init, ff)
            if key not in seen:
                seen.add(key)
                file_pairs.append(key)
        tasks = [
            _DownloadTask(
                s3_uri=_hrrr_s3_uri(init_t, ff, product),
                local_path=_hrrr_local_path(base_path, init_t, ff, product),
                overwrite=overwrite,
            )
            for init_t, ff in file_pairs
        ]
    else:
        tasks = [
            _DownloadTask(
                s3_uri=_hrrr_s3_uri(t, forecast_hour, product),
                local_path=_hrrr_local_path(base_path, t, forecast_hour, product),
                overwrite=overwrite,
            )
            for t in timestamps
        ]

    n_total = len(tasks)
    logger.info("Starting download: %d files, %d workers.", n_total, num_workers)

    n_ok = n_skip = n_miss = 0
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_download_one, tasks):
            status = result[:4]
            logger.info(result)
            if status == "ok  ":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_miss += 1

    logger.info(
        "Done — downloaded: %d, skipped: %d, not found: %d / %d total.",
        n_ok,
        n_skip,
        n_miss,
        n_total,
    )


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Download HRRR grib2 data from AWS S3.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--product",
        default="wrfprsf",
        choices=["wrfprsf", "wrfnatf", "wrfsubhf"],
        help="HRRR product to download (default: wrfprsf).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-download files that already exist on disk.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_hrrr(cfg["data"], product=args.product, num_workers=args.num_workers, overwrite=args.overwrite)
