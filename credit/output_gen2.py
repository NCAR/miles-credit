"""
output_gen2.py
--------------
ForecastWriter: stateful output writer for Gen2 rollout.

Accepts one forecast step at a time via __call__, groups output by calendar
period, and writes to NetCDF (buffered) or Zarr (appended live).

group_by values (case-insensitive, several aliases accepted):
  null              one file per forecast step
  "1d" / "day"     one file per calendar day     → YYYY-MM-DD.nc/.zarr
  "1m" / "month"   one file per calendar month   → YYYY-MM.nc/.zarr
  "1y" / "year"    one file per calendar year    → YYYY.nc/.zarr
  "full"            one file per forecast         → full.nc/.zarr

Output always lands under  save_dir/YYYYMMDD_HHMMZ/  (init-time directory).

Zarr stores are appended step-by-step (no in-memory buffer required).
NetCDF is buffered per period and written in bulk at each calendar boundary.
"""

import logging
import os
import traceback
from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Group-by constants and alias table
# ---------------------------------------------------------------------------

_GROUP_STEP = "step"
_GROUP_DAILY = "daily"
_GROUP_MONTHLY = "monthly"
_GROUP_YEARLY = "yearly"
_GROUP_FULL = "full"

_VALID_GROUP_BY: dict[str, str] = {
    "1d": _GROUP_DAILY,
    "d": _GROUP_DAILY,
    "day": _GROUP_DAILY,
    "daily": _GROUP_DAILY,
    "1m": _GROUP_MONTHLY,
    "m": _GROUP_MONTHLY,
    "month": _GROUP_MONTHLY,
    "monthly": _GROUP_MONTHLY,
    "1y": _GROUP_YEARLY,
    "y": _GROUP_YEARLY,
    "year": _GROUP_YEARLY,
    "yearly": _GROUP_YEARLY,
    "full": _GROUP_FULL,
}


# ---------------------------------------------------------------------------
# Module-level workers (must be picklable for multiprocessing pool)
# ---------------------------------------------------------------------------


def _write_netcdf_worker(ds: xr.Dataset, path: str, encoding: dict) -> None:
    try:
        ds.to_netcdf(path, mode="w", encoding=encoding)
    except Exception:
        logger.error("Failed to write %s:\n%s", path, traceback.format_exc())


def _write_zarr_worker(ds: xr.Dataset, path: str, append: bool) -> None:
    try:
        if append:
            ds.to_zarr(path, mode="a", append_dim="time")
        else:
            ds.to_zarr(path, mode="w")
    except Exception:
        logger.error("Failed to write zarr %s:\n%s", path, traceback.format_exc())


# ---------------------------------------------------------------------------
# Timestamp formatters
# ---------------------------------------------------------------------------


def _fmt_init(dt: pd.Timestamp) -> str:
    """Init-time directory name: YYYYMMDD_HHMMZ"""
    return dt.strftime("%Y%m%d_%H%MZ")


def _fmt_step(dt: pd.Timestamp) -> str:
    """Sub-daily / per-step filename stem: YYYY-MM-DD_THHMMZ"""
    return dt.strftime("%Y-%m-%d_T%H%MZ")


# ---------------------------------------------------------------------------
# ForecastWriter
# ---------------------------------------------------------------------------


class ForecastWriter:
    """Stateful writer that groups and writes forecast steps to NetCDF or Zarr.

    Args:
        output_conf: the ``inference.output`` config block.
        conf: full model config (used to read grid dimensions and levels).
        n_steps: total forecast steps — needed so ``group_by="full"`` knows
            when to flush without an external flush() call.
    """

    def __init__(self, output_conf: dict, conf: dict, n_steps: int, verbose: bool = True) -> None:
        self._n_steps = n_steps
        self._conf = conf

        # output format
        fmt = (output_conf.get("format") or "netcdf").lower().strip()
        if fmt not in ("netcdf", "zarr"):
            raise ValueError(f"Invalid format: {fmt!r}. Must be 'netcdf' or 'zarr'.")
        self._fmt = fmt
        self._ext = ".nc" if fmt == "netcdf" else ".zarr"

        # output_interval: write only steps where fhr is a multiple of this
        ofreq = output_conf.get("output_interval")
        try:
            self._output_freq_hrs: Optional[int] = int(pd.Timedelta(ofreq).total_seconds() / 3600) if ofreq else None
        except Exception:
            raise ValueError(f"Cannot parse output_interval: {ofreq!r}. Use a string like '6h', '24h', etc.")

        # group_by
        self._group_mode: str = self._parse_group_by(output_conf.get("group_by"))

        # variable + level filter
        self._var_filter: Optional[dict] = self._parse_var_filter(output_conf.get("variables"))

        # CF attribute metadata
        self._meta: dict = self._load_metadata(output_conf.get("metadata"))

        # NetCDF encoding config block (ignored for zarr data variables)
        self._encoding_conf: dict = output_conf.get("encoding") or {}

        # Coordinates: lazy-initialised on first __call__
        self._coords: Optional[dict] = None

        # NetCDF buffer: list of (valid_time, xr.Dataset)
        self._buffer: list = []
        self._buffer_period: object = None  # period key of current buffer

        # Zarr: remember which store paths are already initialised (mode w vs a)
        self._zarr_initialized: set = set()

        # One-time validation runs on first __call__ when fhr_per_step is known
        self._validated: bool = False

        self._verbose = verbose

        # Pending async netcdf writes: list of (AsyncResult, path) in submission order
        self._pending: list = []

    # ------------------------------------------------------------------
    # Public interface
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
        """Accept one forecast step; buffer or write depending on group_by.

        Steps excluded by output_freq are silently dropped.  Calendar-period
        groups flush automatically when the period rolls over or the forecast ends.
        """
        fhr = step * fhr_per_step

        # output_freq filter
        if self._output_freq_hrs is not None and fhr % self._output_freq_hrs != 0:
            return

        # lazy coordinate init + one-time cross-parameter validation
        if self._coords is None:
            self._coords = self._init_coords(y_processed)
        if not self._validated:
            self._validate(fhr_per_step)
            self._validated = True

        valid_time = init_time + pd.Timedelta(hours=fhr)
        ds = self._to_dataset(y_processed, valid_time)
        is_last = step == self._n_steps

        if self._fmt == "zarr":
            self._submit_zarr(ds, init_time, valid_time, save_dir)
        else:
            self._submit_netcdf(ds, init_time, valid_time, save_dir, pool, is_last)

    # ------------------------------------------------------------------
    # NetCDF: buffer by calendar period, flush at boundary or forecast end
    # ------------------------------------------------------------------

    def _submit_netcdf(self, ds, init_time, valid_time, save_dir, pool, is_last):
        period = self._period_key(valid_time)

        if self._group_mode == _GROUP_STEP:
            path = self._make_path(init_time, valid_time, save_dir)
            self._write_nc(ds, path, pool)
            return

        if self._group_mode == _GROUP_FULL:
            self._buffer.append((valid_time, ds))
            if is_last:
                self._flush_nc(init_time, save_dir, pool)
            return

        # calendar modes (daily / monthly / yearly)
        if self._buffer and period != self._buffer_period:
            # period rolled over — flush the completed group first
            self._flush_nc(init_time, save_dir, pool)

        if not self._buffer:
            self._buffer_period = period
        self._buffer.append((valid_time, ds))

        if is_last:
            self._flush_nc(init_time, save_dir, pool)

    def _flush_nc(self, init_time, save_dir, pool):
        if not self._buffer:
            return
        times, datasets = zip(*self._buffer)
        combined = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
        path = self._make_path(init_time, times[0], save_dir)
        self._write_nc(combined, path, pool)
        self._buffer.clear()
        self._buffer_period = None

    def _write_nc(self, ds, path, pool):
        self._apply_metadata(ds)
        encoding = self._make_encoding(ds)
        self._submit(ds, path, pool=pool, encoding=encoding, append=False)

    # ------------------------------------------------------------------
    # Zarr: append each step directly — no in-memory buffer needed.
    # Zarr writes must be sequential (time ordering), so pool is not used.
    # ------------------------------------------------------------------

    def _submit_zarr(self, ds, init_time, valid_time, save_dir):
        path = self._make_path(init_time, valid_time, save_dir)
        self._apply_metadata(ds)
        append = path in self._zarr_initialized
        if not append:
            self._zarr_initialized.add(path)
        self._submit(ds, path, pool=None, encoding={}, append=append)

    def _submit(
        self,
        ds: xr.Dataset,
        path: str,
        pool=None,
        encoding: dict = None,
        append: bool = False,
    ) -> None:
        """Dispatch to the writer backend.  Mockable hook for unit tests."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self._fmt == "zarr":
            _write_zarr_worker(ds, path, append)
            if self._verbose:
                tqdm.tqdm.write(f"Zarr {'appended' if append else 'created'}: {path}")
        else:
            enc = encoding or {}
            if pool is not None:
                result = pool.apply_async(_write_netcdf_worker, args=(ds, path, enc))
                self._pending.append((result, path))
            else:
                _write_netcdf_worker(ds, path, enc)
                if self._verbose:
                    tqdm.tqdm.write(f"Saved: {path}")

    def flush(self) -> None:
        """Wait for all pending async writes in submission order and print each path."""
        for result, path in self._pending:
            try:
                result.get()
                if self._verbose:
                    tqdm.tqdm.write(f"Saved: {path}")
            except Exception as exc:
                tqdm.tqdm.write(f"Failed: {path}: {exc}")
        self._pending.clear()

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def _make_path(self, init_time: pd.Timestamp, valid_time: pd.Timestamp, save_dir: str) -> str:
        """Build the output path for a step.

        Directory: save_dir/YYYYMMDD_HHMMZ/
        Filename:  determined by group_mode and valid_time.
        """
        return os.path.join(save_dir, _fmt_init(init_time), self._period_fname(valid_time))

    def _period_fname(self, dt: pd.Timestamp) -> str:
        """Filename (including extension) derived from the calendar period."""
        if self._group_mode == _GROUP_YEARLY:
            return f"{dt.year:04d}{self._ext}"
        if self._group_mode == _GROUP_MONTHLY:
            return f"{dt.year:04d}-{dt.month:02d}{self._ext}"
        if self._group_mode == _GROUP_DAILY:
            return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}{self._ext}"
        if self._group_mode == _GROUP_FULL:
            return f"full{self._ext}"
        # step / per-step
        return f"{_fmt_step(dt)}{self._ext}"

    def _period_key(self, dt: pd.Timestamp) -> object:
        """Return the grouping bucket for a timestamp (used to detect boundary crossings)."""
        if self._group_mode == _GROUP_YEARLY:
            return dt.year
        if self._group_mode == _GROUP_MONTHLY:
            return (dt.year, dt.month)
        if self._group_mode == _GROUP_DAILY:
            return dt.date()
        if self._group_mode == _GROUP_FULL:
            return 0  # constant — all steps share one bucket
        return dt  # step mode — each timestamp is its own bucket

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _to_dataset(self, y_processed: dict, valid_time: pd.Timestamp) -> xr.Dataset:
        """Convert one step of y_processed to an xr.Dataset."""
        coords = self._coords
        data_vars = {}

        for source_name, source_dict in y_processed.items():
            source_levels = coords["levels"].get(source_name, np.array([]))

            for var_key, tensor in source_dict.items():
                if self._var_filter is not None and var_key not in self._var_filter:
                    continue

                # tensor: (B, n_levels, n_time, H, W) → slice batch=0, time=0
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

                else:  # 2D — arr shape (1, H, W); squeeze the level dim
                    data_vars[var_name] = xr.DataArray(
                        arr[0][np.newaxis],  # (H, W) → (1, H, W)
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
        """Build lat/lon/level arrays from model config, falling back to tensor shape."""
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

        lat = np.linspace(-90.0, 90.0, H) if H else np.array([])
        lon = np.linspace(0.0, 360.0 - 360.0 / W, W) if W else np.array([])

        source_levels = {}
        for src_name, src_conf in self._conf["data"]["source"].items():
            lvs = src_conf.get("levels")
            source_levels[src_name] = np.array(lvs) if lvs else np.array([])

        return {"latitude": lat, "longitude": lon, "levels": source_levels}

    # ------------------------------------------------------------------
    # Encoding and metadata
    # ------------------------------------------------------------------

    def _apply_metadata(self, ds: xr.Dataset) -> None:
        """Inject CF variable attributes and set time encoding."""
        for var in ds.data_vars:
            if var in self._meta:
                ds[var].attrs.update(self._meta[var])
                # _FillValue must live in .encoding, not .attrs — xarray's zarr
                # encoder raises ValueError if it finds it in attrs.
                fill = ds[var].attrs.pop("_FillValue", None)
                if fill is not None:
                    ds[var].encoding["_FillValue"] = fill

        if self._fmt == "netcdf":
            # Exact datetime64 round-trip: seconds since epoch, proleptic calendar
            time_meta = self._meta.get("time") or {}
            ds["time"].encoding.update(
                {
                    "units": time_meta.get("units", "seconds since 1970-01-01"),
                    "calendar": time_meta.get("calendar", "proleptic_gregorian"),
                    "dtype": "int64",
                }
            )
        else:
            # Zarr appends re-derive CF time units per write. If left to xarray's
            # autodetection, the first (mode="w") step pins units relative to the
            # first timestamp (e.g. "hours since <t0>") while each appended step
            # re-derives its own units ("days since ...") — the integers written no
            # longer match the store's units and the time coordinate is silently
            # corrupted by a days/hours (x24) factor on read-back. Pin an explicit
            # fixed-epoch encoding (as in the netcdf branch) so every append encodes
            # consistently and round-trips exactly.
            time_meta = self._meta.get("time") or {}
            ds["time"].encoding.update(
                {
                    "units": time_meta.get("units", "seconds since 1970-01-01"),
                    "calendar": time_meta.get("calendar", "proleptic_gregorian"),
                    "dtype": "int64",
                    "chunks": 1,
                }
            )

    def _make_encoding(self, ds: xr.Dataset) -> dict:
        """Build per-variable NetCDF encoding (not used for zarr data variables)."""
        _SCALAR_KEYS = {"dtype", "zlib", "complevel", "shuffle", "chunksizes", "fletcher32", "contiguous"}
        default = {k: v for k, v in self._encoding_conf.items() if k in _SCALAR_KEYS}
        overrides = {k: v for k, v in self._encoding_conf.items() if k not in _SCALAR_KEYS}
        encoding = {}
        for var in ds.data_vars:
            enc = dict(default)
            for key, override in overrides.items():
                if key == var or key.split("/")[-1] == var:
                    enc.update(override)
            encoding[var] = enc
        return encoding

    # ------------------------------------------------------------------
    # Parsing, validation, and static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_group_by(gb) -> str:
        if gb is None:
            return _GROUP_STEP
        key = str(gb).lower().strip()
        if key not in _VALID_GROUP_BY:
            valid_str = ", ".join(f"'{k}'" for k in sorted(_VALID_GROUP_BY))
            raise ValueError(f"Invalid group_by: {gb!r}. Must be null or one of: {valid_str}")
        return _VALID_GROUP_BY[key]

    def _validate(self, fhr_per_step: int) -> None:
        """Cross-parameter checks run once on the first forecast step."""
        freq = self._output_freq_hrs
        total_fhr = self._n_steps * fhr_per_step

        if freq is not None and freq > total_fhr:
            raise ValueError(
                f"output_interval ({freq}h) exceeds total forecast length ({total_fhr}h) — no steps would ever be written."
            )

        if freq is not None and freq < fhr_per_step:
            logger.warning(
                "output_interval (%dh) is less than the model timestep (%dh) — "
                "the filter has no effect and every step will be written.",
                freq,
                fhr_per_step,
            )

        if freq is not None and self._group_mode not in (_GROUP_STEP, _GROUP_FULL):
            # Conservative lower bound on steps per period
            min_steps_per_period = {
                _GROUP_DAILY: 24 // fhr_per_step,
                _GROUP_MONTHLY: (28 * 24) // fhr_per_step,
                _GROUP_YEARLY: (365 * 24) // fhr_per_step,
            }[self._group_mode]
            written_per_period = max(1, (min_steps_per_period * fhr_per_step) // freq)
            if written_per_period <= 1:
                logger.warning(
                    "output_interval (%dh) with group_by=%r will produce files with "
                    "≤1 timestep each. Consider a coarser group_by or finer output_interval.",
                    freq,
                    self._group_mode,
                )

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
        """Return {var_key: levels_or_None} dict, or None to save everything."""
        if variables_conf is None:
            return None
        result = {}
        for entry in variables_conf:
            if isinstance(entry, dict):
                result[entry["name"]] = entry.get("levels")
            else:
                result[entry] = None
        return result
