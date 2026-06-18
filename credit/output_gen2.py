"""
output_gen2.py
--------------
ForecastWriter: stateful output writer for Gen2 rollout.

Accepts one forecast step at a time via __call__, buffers steps according to
group_by, and flushes NetCDF files asynchronously via a multiprocessing pool.

Supports:
  - output_freq:  temporal subsampling (write only every Nh)
  - group_by:     file chunking (one file per step | per Nh window | per forecast)
  - variables:    variable + level subsetting
  - encoding:     per-variable NetCDF compression
  - metadata:     CF attribute injection from a YAML file
"""

import logging
import os
import traceback
from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async write worker (must be module-level to be picklable)
# ---------------------------------------------------------------------------


def _write_netcdf_worker(ds: xr.Dataset, path: str, encoding: dict) -> None:
    """Write a Dataset to NetCDF; intended for pool.apply_async."""
    try:
        ds.to_netcdf(path, mode="w", encoding=encoding)
        logger.info("Saved: %s", path)
    except Exception:
        logger.error("Failed to write %s:\n%s", path, traceback.format_exc())


# ---------------------------------------------------------------------------
# ForecastWriter
# ---------------------------------------------------------------------------


def _fmt_ts(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d_%H%MZ")


class ForecastWriter:
    """Stateful writer that buffers forecast steps and flushes to disk.

    Instantiated once before the rollout loop. Pass as ``save_output_fn`` to
    ``run_forecast`` — the call signature matches the protocol exactly so no
    changes to ``run_forecast`` are needed.

    Args:
        output_conf: the ``inference.output`` config block.
        conf: full model config dict (used to read grid dimensions and levels).
        n_steps: total forecast steps — needed so ``group_by="full"`` knows
            when to flush without requiring an external flush() call.
    """

    def __init__(self, output_conf: dict, conf: dict, n_steps: int) -> None:
        self._n_steps = n_steps
        self._conf = conf

        self._fmt = output_conf.get("format", "netcdf")

        # output_freq: only write steps whose fhr is a multiple of this value
        ofreq = output_conf.get("output_freq")
        self._output_freq_hrs: Optional[int] = int(pd.Timedelta(ofreq).total_seconds() / 3600) if ofreq else None

        # group_by: controls file chunking
        gb = output_conf.get("group_by")
        if gb is None:
            self._group_mode = "step"  # one file per step
            self._group_hrs: Optional[int] = None
        elif gb == "full":
            self._group_mode = "full"  # one file per forecast
            self._group_hrs = None
        else:
            self._group_mode = "period"  # one file per Nh window
            self._group_hrs = int(pd.Timedelta(gb).total_seconds() / 3600)

        # Variable filter: None = save everything
        self._var_filter: Optional[dict] = self._parse_var_filter(output_conf.get("variables"))

        # CF metadata
        self._meta: dict = self._load_metadata(output_conf.get("metadata"))

        # NetCDF encoding config
        self._encoding_conf: dict = output_conf.get("encoding") or {}

        # Coordinates: lazy-initialized from first y_processed received
        self._coords: Optional[dict] = None

        # Step buffer: list of (valid_time, xr.Dataset)
        self._buffer: list = []

    # ------------------------------------------------------------------
    # Public interface (matches save_output_fn protocol)
    # ------------------------------------------------------------------

    def __call__(
        self,
        y_processed: dict,
        init_time: pd.Timestamp,
        step: int,
        fhr_per_step: int,
        save_dir: str,
        pool,
    ) -> None:
        """Accept one forecast step. Buffers or writes depending on group_by.

        Steps excluded by output_freq are silently dropped. A group is flushed
        when its time boundary is crossed or when the final step is reached.
        """
        fhr = step * fhr_per_step

        # -- output_freq filter --
        if self._output_freq_hrs is not None and fhr % self._output_freq_hrs != 0:
            return

        # -- lazy coordinate init --
        if self._coords is None:
            self._coords = self._init_coords(y_processed)

        valid_time = init_time + pd.Timedelta(hours=fhr)
        ds = self._to_dataset(y_processed, valid_time)
        is_last = step == self._n_steps

        if self._group_mode == "step":
            self._submit(ds, init_time, valid_time, valid_time, save_dir, pool)

        elif self._group_mode == "full":
            self._buffer.append((valid_time, ds))
            if is_last:
                self._flush(init_time, save_dir, pool)

        else:  # "period"
            self._buffer.append((valid_time, ds))
            at_boundary = fhr % self._group_hrs == 0
            if at_boundary or is_last:
                self._flush(init_time, save_dir, pool)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush(self, init_time: pd.Timestamp, save_dir: str, pool) -> None:
        if not self._buffer:
            return
        times, datasets = zip(*self._buffer)
        combined = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
        self._submit(combined, init_time, times[0], times[-1], save_dir, pool)
        self._buffer.clear()

    def _submit(
        self,
        ds: xr.Dataset,
        init_time: pd.Timestamp,
        valid_start: pd.Timestamp,
        valid_end: pd.Timestamp,
        save_dir: str,
        pool,
    ) -> None:
        path = self._make_path(init_time, valid_start, valid_end, save_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._apply_metadata(ds)
        encoding = self._make_encoding(ds)
        if pool is not None:
            pool.apply_async(_write_netcdf_worker, args=(ds, path, encoding))
        else:
            _write_netcdf_worker(ds, path, encoding)

    def _to_dataset(self, y_processed: dict, valid_time: pd.Timestamp) -> xr.Dataset:
        """Convert one step of y_processed to an xr.Dataset."""
        coords = self._coords
        data_vars = {}

        for source_name, source_dict in y_processed.items():
            source_levels = coords["levels"].get(source_name, np.array([]))

            for var_key, tensor in source_dict.items():
                if self._var_filter is not None and var_key not in self._var_filter:
                    continue

                # tensor shape: (B, n_levels, n_time, H, W)
                # slice batch=0, time=0 → (n_levels, H, W)
                t = tensor[0, :, 0, :, :] if tensor.ndim == 5 else tensor[0]
                arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)

                parts = var_key.split("/")
                dim_type = parts[2] if len(parts) > 2 else ""
                var_name = parts[-1]

                if dim_type == "3d":
                    requested = self._var_filter[var_key] if self._var_filter else None
                    if requested is not None and len(source_levels):
                        idx = [i for i, lv in enumerate(source_levels) if lv in requested]
                        arr = arr[idx]
                        level_coord = source_levels[idx]
                    else:
                        level_coord = source_levels if len(source_levels) else np.arange(arr.shape[0])

                    data_vars[var_name] = xr.DataArray(
                        arr[np.newaxis],  # (1, L, H, W)
                        dims=["time", "level", "latitude", "longitude"],
                        coords={
                            "time": [valid_time],
                            "level": level_coord,
                            "latitude": coords["latitude"],
                            "longitude": coords["longitude"],
                        },
                    )

                else:  # 2D — arr shape (1, H, W); squeeze level dim
                    data_vars[var_name] = xr.DataArray(
                        arr[[0]][np.newaxis],  # (1, 1, H, W) → squeeze level below
                        dims=["time", "latitude", "longitude"],
                        coords={
                            "time": [valid_time],
                            "latitude": coords["latitude"],
                            "longitude": coords["longitude"],
                        },
                    )

        ds = xr.Dataset(data_vars)
        ds.attrs["Conventions"] = "CF-1.11"
        return ds

    def _init_coords(self, y_processed: dict) -> dict:
        """Build coordinate arrays from model config + first tensor shape.

        Placeholder until static_metadata carries actual grid coordinates.
        Assumes a uniform lat/lon grid (standard for ERA5).
        """
        model_conf = self._conf.get("model", {})
        H = model_conf.get("image_height")
        W = model_conf.get("image_width")

        if H is None or W is None:
            for source_dict in y_processed.values():
                for tensor in source_dict.values():
                    H, W = tensor.shape[-2], tensor.shape[-1]
                    break
                if H is not None:
                    break

        # ERA5 1-deg convention: lat S→N, lon 0→359
        lat = np.linspace(-90.0, 90.0, H) if H else np.array([])
        lon = np.linspace(0.0, 360.0 - 360.0 / W, W) if W else np.array([])

        source_levels = {}
        for src_name, src_conf in self._conf["data"]["source"].items():
            lvs = src_conf.get("levels")
            source_levels[src_name] = np.array(lvs) if lvs else np.array([])

        return {"latitude": lat, "longitude": lon, "levels": source_levels}

    def _apply_metadata(self, ds: xr.Dataset) -> None:
        for var in ds.data_vars:
            if var in self._meta:
                ds[var].attrs.update(self._meta[var])
        time_enc = self._meta.get("time") or {"units": "hours since 1900-01-01 00:00:00", "calendar": "gregorian"}
        for k, v in time_enc.items():
            ds["time"].encoding[k] = v

    def _make_encoding(self, ds: xr.Dataset) -> dict:
        default = dict(self._encoding_conf.get("default") or {})
        encoding = {}
        for var in ds.data_vars:
            enc = dict(default)
            # Check per-variable overrides keyed by full path or short name
            for key, override in self._encoding_conf.items():
                if key == "default":
                    continue
                if key == var or key.split("/")[-1] == var:
                    enc.update(override)
            encoding[var] = enc
        return encoding

    def _make_path(
        self,
        init_time: pd.Timestamp,
        valid_start: pd.Timestamp,
        valid_end: pd.Timestamp,
        save_dir: str,
    ) -> str:
        init_str = _fmt_ts(init_time)
        start_str = _fmt_ts(valid_start)
        end_str = _fmt_ts(valid_end)
        if valid_start == valid_end:
            fname = f"forecast_{init_str}_f{start_str}.nc"
        else:
            fname = f"forecast_{init_str}_f{start_str}-{end_str}.nc"
        return os.path.join(save_dir, init_str, fname)

    @staticmethod
    def _load_metadata(path: Optional[str]) -> dict:
        if not path:
            return {}
        resolved = os.path.expandvars(path)
        if not os.path.dirname(resolved):
            resolved = str(files("credit.metadata").joinpath(resolved))
        if not os.path.isfile(resolved):
            logger.warning("ForecastWriter: metadata file not found: %s", resolved)
            return {}
        with open(resolved) as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    @staticmethod
    def _parse_var_filter(variables_conf) -> Optional[dict]:
        """Return {var_key: levels_or_None} or None to save everything."""
        if variables_conf is None:
            return None
        result = {}
        for entry in variables_conf:
            if isinstance(entry, dict):
                result[entry["name"]] = entry.get("levels")
            else:
                result[entry] = None
        return result
