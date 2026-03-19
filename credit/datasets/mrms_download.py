"""
mrms_download.py
----------------
Standalone utility for downloading MRMS data from AWS S3 to local disk.

Downloaded files follow the same ``filename_time_format`` strftime convention
used by ``MRMSDataset`` in local mode, so files are immediately usable without
any renaming.

Usage::

    python -m credit.datasets.mrms_download --config config/my_conf.yaml

Or programmatically::

    from credit.datasets.mrms_download import download_mrms
    download_mrms(config, output_format="netcdf")

Config section used (``data.source.MRMS``)::

    data:
      source:
        MRMS:
          mode: "local"             # output mode after download
          region: "CONUS"
          variables:
            prognostic:
              vars_2D:
                - "MultiSensor_QPE_01H_Pass2_00.00"
                - "MultiSensor_QPE_06H_Pass2_00.00"
              path: "/data/mrms_*.nc"          # determines base_path + format
              filename_time_format: "%Y%m%d-%H%M%S"
          extent: [-130, -60, 20, 55]          # optional

      start_datetime: "2024-06-01"
      end_datetime:   "2024-07-01"
      timestep:       "6h"
      forecast_len:   0

Output files are written to the directory of ``path``, with names derived
from the ``filename_time_format``.  For example, with
``filename_time_format: "%Y%m%d-%H%M%S"`` and ``path: "/data/mrms_*.nc"``
the file ``/data/mrms_20240601-060000.nc`` is created for the 06:00 UTC step.
"""

from __future__ import annotations

import gzip
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from credit.datasets.MRMS import _S3_URI, _apply_extent

logger = logging.getLogger(__name__)

_GRIB_TABLE_URL = (
    "https://raw.githubusercontent.com/NOAA-National-Severe-Storms-Laboratory"
    "/mrms-support/refs/heads/main/GRIB2_TABLES/UserTable_MRMS_v12.2.csv"
)


def download_mrms(
    config: dict,
    output_format: str = "netcdf",
    grib_table_url: str = _GRIB_TABLE_URL,
) -> None:
    """Download MRMS data from AWS S3 to local disk.

    Iterates over all timestamps defined by the top-level data config and
    writes one output file per timestamp.  Files are named using
    ``filename_time_format`` so they are immediately loadable by
    ``MRMSDataset`` in ``mode: "local"``.

    Args:
        config: Top-level ``data`` config dict (same object passed to
            ``MRMSDataset``).
        output_format: ``"netcdf"`` (default) or ``"zarr"``.
        grib_table_url: URL to the MRMS GRIB2 parameter table CSV used for
            attaching variable metadata to output files.

    Raises:
        ValueError: If *output_format* is not ``"netcdf"`` or ``"zarr"``.
        KeyError: If the config is missing required fields.
    """
    import pygrib
    import s3fs

    if output_format not in ("netcdf", "zarr"):
        raise ValueError(f"output_format must be 'netcdf' or 'zarr', got '{output_format}'")

    grib_table = pd.read_csv(grib_table_url)

    source_cfg = config["source"]["MRMS"]
    region = source_cfg.get("region", "CONUS")
    extent = source_cfg.get("extent", None)

    # Collect all vars_2D across all configured field types.  The field type
    # distinction (prognostic/dynamic_forcing/etc.) is a training concept; the
    # download utility writes all variables into a single file per timestamp.
    variables: list[str] = []
    time_fmt: str = "%Y%m%d-%H%M%S"
    path_pattern: str = ""
    for field_cfg in source_cfg["variables"].values():
        if isinstance(field_cfg, dict):
            variables.extend(field_cfg.get("vars_2D", []))
            if not path_pattern:
                path_pattern = field_cfg.get("path", "")
                time_fmt = field_cfg.get("filename_time_format", time_fmt)
    variables = list(dict.fromkeys(variables))  # deduplicate, preserve order

    # Derive output directory from the path glob pattern
    base_dir = os.path.dirname(path_pattern) or "."

    dt = pd.Timedelta(config["timestep"])
    datetimes = pd.date_range(
        config["start_datetime"],
        config["end_datetime"],
        freq=dt,
    )

    fs = s3fs.S3FileSystem(anon=True)
    n_total = len(datetimes)

    for idx, t in enumerate(datetimes):
        date_str = t.strftime("%Y%m%d")
        datetime_str = t.strftime("%Y%m%d-%H%M%S")
        file_stem = f"MRMS_{t.strftime(time_fmt)}"

        all_vars: list[xr.DataArray] = []

        for vname in variables:
            s3_path = _S3_URI.format(
                region=region,
                varname=vname,
                date_str=date_str,
                datetime_str=datetime_str,
            )
            try:
                with fs.open(s3_path, "rb") as f:
                    raw = pygrib.fromstring(gzip.decompress(f.read()))
            except FileNotFoundError:
                logger.warning("S3 file not found, skipping: %s", s3_path)
                continue

            # Attach GRIB metadata as variable attributes
            grib_attrs = (
                grib_table.loc[
                    (grib_table["Category"] == raw["parameterCategory"])
                    & (grib_table["Parameter"] == raw["parameterNumber"])
                ]
                .T.iloc[:, 0]
                .dropna()
                .to_dict()
            )

            lats = raw.latitudes[:: raw.Nx][::-1]  # ascending south -> north
            lons = raw.longitudes[: raw.Nx]  # 0-360

            da = xr.DataArray(
                np.expand_dims(raw.values, axis=0),  # (time, lat, lon)
                coords={"time": [t], "lat": lats, "lon": lons},
                dims=["time", "lat", "lon"],
                name=vname,
                attrs=grib_attrs,
            )

            da = _apply_extent(da, extent)

            all_vars.append(da)

        if not all_vars:
            logger.warning("No data retrieved for %s, skipping.", t)
            continue

        ds = xr.merge([da.to_dataset() for da in all_vars])

        if output_format == "zarr":
            out_path = os.path.join(base_dir, f"{file_stem}.zarr")
            ds.to_zarr(out_path, mode="w")
        else:
            out_path = os.path.join(base_dir, f"{file_stem}.nc")
            ds.to_netcdf(out_path)

        logger.info("[%d/%d] Saved %s", idx + 1, n_total, out_path)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Download MRMS data from AWS S3.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--format",
        default="netcdf",
        choices=["netcdf", "zarr"],
        help="Output file format (default: netcdf).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_mrms(cfg["data"], output_format=args.format)
