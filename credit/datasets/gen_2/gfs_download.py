"""
gfs_download.py
---------------
Download GFS or GDAS NetCDF files for ``GFSDataset`` local mode.

The downloader reads the same single-source data configuration used by
``GFSDataset``, checks which configured timestamps are available in the public
Google Cloud bucket, and downloads the atmospheric/surface file pair for each
available run. Files are written without renaming, so a local ``GFSDataset``
can use the resulting directory directly.

The output layout is::

    {base_path}/{system}.YYYYMMDD/HH/atmos/{system}.tHHz.atmanl.nc
    {base_path}/{system}.YYYYMMDD/HH/atmos/{system}.tHHz.sfcanl.nc

For forecast output, ``forecast_hour`` changes the final names to forms such
as ``atmf003.nc`` and ``sfcf003.nc``. Downloads use obstore and a thread pool;
``num_workers`` controls the number of concurrent file transfers.

Command-line usage::

    python -m credit.datasets.gen_2.gfs_download -c config/gfs.yml
    python -m credit.datasets.gen_2.gfs_download -c config/gfs.yml --num-workers 8 --overwrite
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import pandas as pd

from credit.datasets.gen_2.gfs import (
    _GCS_BUCKET,
    GFSDataset,
    _file_paths,
)
from credit.datasets.gen_2.multi_source import make_single_source_subconfig

logger = logging.getLogger(__name__)


class _DownloadTask(NamedTuple):
    remote_path: str
    local_path: str
    overwrite: bool


def _download_one(task: _DownloadTask, store: Any) -> str:
    if os.path.exists(task.local_path) and not task.overwrite:
        return f"skip  {task.local_path}"
    os.makedirs(os.path.dirname(task.local_path), exist_ok=True)
    try:
        result = store.get(task.remote_path)
        with open(task.local_path, "wb") as output:
            output.write(result.bytes())
        return f"ok    {task.local_path}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not download %s: %s", task.remote_path, exc)
        return f"miss  {task.remote_path}"


def download_gfs(data_config: dict[str, Any], num_workers: int = 4, overwrite: bool = False) -> None:
    """Download configured GFS/GDAS files for local ``GFSDataset`` use.

    Args:
        data_config: The top-level ``data`` configuration passed to
            ``GFSDataset``. It must contain exactly one source with
            ``dataset_type: "gfs"`` and a local ``base_path``.
        num_workers: Number of concurrent file downloads. Defaults to ``4``.
        overwrite: If ``True``, download files even when the corresponding
            local file already exists. Defaults to ``False``.

    Raises:
        KeyError: If the configuration does not contain the required ``source``
            block.
        ValueError: If multiple sources, a non-GFS source, or no ``base_path``
            is supplied.
        ImportError: If obstore is not installed when the download begins.

    Example:
        >>> from credit.datasets.gen_2.gfs_download import download_gfs
        >>> download_gfs(config["data"], num_workers=8)

    After downloading, set the source ``mode`` to ``"local"`` and point
    ``base_path`` at the download directory before constructing
    ``GFSDataset``.
    """
    source_name = next(iter(data_config["source"]))
    if len(data_config["source"]) > 1:
        raise ValueError("Provide a config containing exactly one GFS source.")
    source_cfg = data_config["source"][source_name]
    if source_cfg.get("dataset_type") != "gfs":
        raise ValueError(f"Expected dataset_type 'gfs', got {source_cfg.get('dataset_type')!r}.")
    if "base_path" not in source_cfg:
        raise ValueError("base_path is required to download GFS/GDAS files.")

    source_data = make_single_source_subconfig(data_config, source_name)
    source_data["source"][source_name] = {
        **source_data["source"][source_name],
        "mode": "remote",
        "check_availability": True,
    }
    dataset = GFSDataset(source_data)
    timestamps = dataset.datetimes
    tasks: list[_DownloadTask] = []
    seen: set[str] = set()
    for timestamp in timestamps:
        t = pd.Timestamp(timestamp)
        for remote_path, local_path in zip(
            _file_paths(dataset.system, t, dataset.forecast_hour),
            _file_paths(dataset.system, t, dataset.forecast_hour, dataset.base_path),
        ):
            if remote_path not in seen:
                seen.add(remote_path)
                tasks.append(_DownloadTask(remote_path, local_path, overwrite))

    from obstore.store import GCSStore

    store = GCSStore(bucket=_GCS_BUCKET, config={"skip_signature": True})
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(lambda task: _download_one(task, store), tasks):
            logger.info(result)


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Download GFS/GDAS NetCDF files from Google Cloud Storage.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--source-name", default=None, help="Name of the GFS source in the data config.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    if args.source_name is not None:
        config["data"] = make_single_source_subconfig(config["data"], args.source_name)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_gfs(config["data"], num_workers=args.num_workers, overwrite=args.overwrite)
