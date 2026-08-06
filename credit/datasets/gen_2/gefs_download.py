"""
gefs_download.py
----------------
Download raw GEFS cube-sphere initialization files for local Gen2 loading.

The downloader uses the same source configuration as ``GEFSDataset`` and
preserves the public GEFS layout::

    {base_path}/gefs.YYYYMMDD/HH/atmos/init/{member}/gfs_ctrl.nc
    {base_path}/gefs.YYYYMMDD/HH/atmos/init/{member}/gfs_data.tile1.nc
    {base_path}/gefs.YYYYMMDD/HH/atmos/init/{member}/sfc_data.tile1.nc

All six atmospheric and six surface tiles are downloaded for every selected
member and initialization time. Forecast lead products are not supported;

Command-line usage::

    python -m credit.datasets.gen_2.gefs_download -c config/gefs.yml
    python -m credit.datasets.gen_2.gefs_download -c config/gefs.yml --num-workers 8 --overwrite
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import pandas as pd

from credit.datasets.gen_2.gefs import (
    _GEFS_BUCKET,
    GEFSDataset,
    _member_file_paths,
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


def download_gefs(data_config: dict[str, Any], num_workers: int = 4, overwrite: bool = False) -> None:
    """Download selected GEFS initialization files for local ``GEFSDataset`` use.

    Args:
        data_config: Top-level ``data`` configuration with exactly one
            ``dataset_type: "gefs"`` source. The source must define
            ``base_path``. Omitted ``members`` downloads ``c00``; an empty
            member list discovers all members in the first requested run.
        num_workers: Number of concurrent file downloads. Defaults to ``4``.
        overwrite: Redownload existing files when ``True``. Defaults to
            ``False``.

    Raises:
        KeyError: If the configuration does not contain a ``source`` block.
        ValueError: If the configuration has multiple sources, the wrong
            dataset type, no base path, or a forecast lead setting.
        FileNotFoundError: If any selected member or required tile is missing
            for a requested initialization.

    Example:
        >>> from credit.datasets.gen_2.gefs_download import download_gefs
        >>> download_gefs(config["data"], num_workers=8)
    """
    source_name = next(iter(data_config["source"]))
    if len(data_config["source"]) > 1:
        raise ValueError("Provide a config containing exactly one GEFS source.")
    source_cfg = data_config["source"][source_name]
    if source_cfg.get("dataset_type") != "gefs":
        raise ValueError(f"Expected dataset_type 'gefs', got {source_cfg.get('dataset_type')!r}.")
    if "base_path" not in source_cfg:
        raise ValueError("base_path is required to download GEFS files.")
    if "forecast_hour" in source_cfg:
        raise ValueError("GEFS cube-sphere downloads support initialization times only; remove forecast_hour.")

    source_data = make_single_source_subconfig(data_config, source_name)
    source_data["source"][source_name] = {
        **source_data["source"][source_name],
        "mode": "remote",
    }
    dataset = GEFSDataset(source_data)
    tasks: list[_DownloadTask] = []
    seen: set[str] = set()
    for timestamp in dataset.datetimes:
        t = pd.Timestamp(timestamp)
        for member in dataset.members:
            control, atmospheric, surface = _member_file_paths(t, member)
            local_control, local_atmospheric, local_surface = _member_file_paths(t, member, dataset.base_path)
            for remote_path, local_path in zip(
                [control, *atmospheric, *surface],
                [local_control, *local_atmospheric, *local_surface],
            ):
                if remote_path not in seen:
                    seen.add(remote_path)
                    tasks.append(_DownloadTask(remote_path, local_path, overwrite))

    import obstore
    from obstore.store import GCSStore

    del obstore
    store = GCSStore(bucket=_GEFS_BUCKET, config={"skip_signature": True})
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(lambda task: _download_one(task, store), tasks):
            logger.info(result)


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Download GEFS cube-sphere initialization NetCDF files.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--source-name", default=None, help="Name of the GEFS source in the data config.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    if args.source_name is not None:
        config["data"] = make_single_source_subconfig(config["data"], args.source_name)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_gefs(config["data"], num_workers=args.num_workers, overwrite=args.overwrite)
