"""
hrrr.py
-------------------------------------------------------
HRRRDataset: PyTorch Dataset for HRRR GRIB2 data.

Supports three HRRR products (``VALID_PRODUCTS``):

* ``wrfprsf`` — pressure-level output (default, ~200 MB/file)
* ``wrfnatf`` — native/hybrid-sigma level output (~200 MB/file, ~65 levels)
* ``wrfsubhf`` — 15-minute sub-hourly surface output (surface vars only)

Tensor keys follow the pattern ``{prefix}/{field_type}/{dim}/{varname}``
where *prefix* is product-specific:

    wrfprsf  → ``hrrr/…``
    wrfnatf  → ``hrrr_nat/…``
    wrfsubhf → ``hrrr_subh/…``

*dim* is ``"3d"`` for multi-level variables and ``"2d"`` for surface variables.

Tensor shapes (before DataLoader batching):
    3D variables: ``(n_levels, 1, y, x)``
    2D variables: ``(1, 1, y, x)``

The ``y`` / ``x`` spatial dimensions correspond to HRRR's native Lambert
Conformal Conic grid; if ``extent`` is specified they reflect the cropped
sub-domain rather than the full CONUS grid (~1059 × 1799).

Two S3 path layouts are handled automatically:

    v1/v2  (before 2018-07-12):
        s3://noaa-hrrr-bdp-pds/hrrr.{YYYYMMDD}/hrrr.t{HH}z.{product}{FF:02d}.grib2
    v3/v4  (2018-07-12 onward):
        s3://noaa-hrrr-bdp-pds/hrrr.{YYYYMMDD}/conus/hrrr.t{HH}z.{product}{FF:02d}.grib2

GRIB2 reading
-------------
Both local and remote modes use the same ``.idx`` + byte-range pipeline:

*Remote mode*:

1. Fetch the sidecar ``.idx`` inventory (~100 KB) via HTTPS to get exact byte
   offsets for every GRIB message.
2. Issue one HTTP Range GET per required message (~50–200 KB each) via
   ``requests``, with all messages fetched in parallel using
   :class:`concurrent.futures.ThreadPoolExecutor`.

*Local mode*: reads the ``.idx`` sidecar from disk, then uses
``file.seek()`` + ``file.read()`` — identical byte-range approach, no
full-file scan.  The ``.idx`` sidecar must be present alongside the grib2;
download it with ``hrrr_download.py``.

For a typical training sample (5 vars × 6 levels ≈ 30 messages) remote mode
transfers ~3 MB instead of ~200 MB (~60–100× reduction).

Variable lookup is driven by :data:`VAR_REGISTRY`.  Extend it at import
time to add variables without subclassing::

    from credit.datasets.hrrr import VAR_REGISTRY
    VAR_REGISTRY["MYVAR"] = {
        "shortName": "myvar", "typeOfLevel": "isobaricInhPa",
        "idx_name": "MYVAR", "idx_level": None,
    }

Example YAML (wrfprsf, local mode)::

    data:
      source:
        HRRR:
          mode: "local"
          base_path: "/data/hrrr"
          forecast_hour: 0
          levels: [250, 500, 700, 850, 925, 1000]
          variables:
            prognostic:
              vars_3D: [T, U, V, Q, GH]
              vars_2D: [t2m]
          extent: [-130, -60, 20, 55]

      start_datetime: "2021-06-01"
      end_datetime:   "2021-06-05"
      timestep:       "1h"
      forecast_len:   0

Example YAML (wrfnatf, remote mode)::

    data:
      source:
        HRRR_NAT:
          mode: "remote"
          forecast_hour: 0
          levels: [10, 20, 30, 40, 50]   # hybrid level indices 1–65
          variables:
            prognostic:
              vars_3D: [T, U, V, Q]

      start_datetime: "2022-01-01"
      end_datetime:   "2022-01-31"
      timestep:       "1h"
      forecast_len:   0

Example YAML (wrfsubhf, remote mode — 15-min output)::

    data:
      source:
        HRRR_SUBH:
          mode: "remote"
          variables:
            prognostic:
              vars_2D: [t2m, sp, refc]

      start_datetime: "2022-01-01 00:15"
      end_datetime:   "2022-01-31 00:00"
      timestep:       "15min"
      forecast_len:   0
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

VALID_FIELD_TYPES = {"prognostic", "diagnostic", "dynamic_forcing"}

# V3+ S3 path includes a 'conus/' subdirectory; v1/v2 does not
_HRRR_V3_CUTOFF = pd.Timestamp("2018-07-12")
_S3_BUCKET = "noaa-hrrr-bdp-pds"
# Public HTTPS base — used for Range requests (faster than s3fs seek+read)
_HRRR_HTTPS_BASE = f"https://{_S3_BUCKET}.s3.amazonaws.com"

#: Variable registry mapping user-facing names to HRRR ``.idx`` lookup keys.
#:
#: Each entry contains exactly two keys:
#:
#:   ``idx_name``   — variable abbreviation as it appears in the ``.idx`` file
#:                    (e.g. ``"TMP"``, ``"UGRD"``)
#:   ``idx_level``  — level string in the ``.idx`` file; ``None`` for
#:                    pressure-level variables (matched dynamically as ``"{N} mb"``)
#:
#: Extend at import time to add variables without subclassing::
#:
#:     from credit.datasets.hrrr import VAR_REGISTRY
#:     VAR_REGISTRY["MYVAR"] = {"idx_name": "MYVAR", "idx_level": "surface"}
VAR_REGISTRY: dict[str, dict] = {
    # -------------------------------------------------------------------------
    # Pressure-level variables  (idx_level=None → matched as "{N} mb")
    # -------------------------------------------------------------------------
    # Dynamics / thermodynamics
    "T": {"idx_name": "TMP", "idx_level": None},  # temperature (K)
    "U": {"idx_name": "UGRD", "idx_level": None},  # u-component of wind (m/s)
    "V": {"idx_name": "VGRD", "idx_level": None},  # v-component of wind (m/s)
    "W": {"idx_name": "VVEL", "idx_level": None},  # vertical velocity (Pa/s)
    "GH": {"idx_name": "HGT", "idx_level": None},  # geopotential height (gpm)
    "ABSV": {"idx_name": "ABSV", "idx_level": None},  # absolute vorticity (1/s)
    # Moisture
    "Q": {"idx_name": "SPFH", "idx_level": None},  # specific humidity (kg/kg)
    "RH": {"idx_name": "RH", "idx_level": None},  # relative humidity (%)
    "DPT": {"idx_name": "DPT", "idx_level": None},  # dew point temperature (K)
    # Microphysics (not always present at all levels — verify against your files)
    "CLWMR": {"idx_name": "CLWMR", "idx_level": None},  # cloud liquid water mixing ratio (kg/kg)
    "ICMR": {"idx_name": "ICMR", "idx_level": None},  # ice crystal mixing ratio (kg/kg)
    "RWMR": {"idx_name": "RWMR", "idx_level": None},  # rain water mixing ratio (kg/kg)
    "SNMR": {"idx_name": "SNMR", "idx_level": None},  # snow mixing ratio (kg/kg)
    "GRLE": {"idx_name": "GRLE", "idx_level": None},  # graupel mixing ratio (kg/kg)
    # -------------------------------------------------------------------------
    # Surface / near-surface variables
    # -------------------------------------------------------------------------
    # 2 m
    "t2m": {"idx_name": "TMP", "idx_level": "2 m above ground"},  # 2-m temperature (K)
    "d2m": {"idx_name": "DPT", "idx_level": "2 m above ground"},  # 2-m dew point temperature (K)
    "rh2m": {"idx_name": "RH", "idx_level": "2 m above ground"},  # 2-m relative humidity (%)
    # 10 m
    "u10m": {"idx_name": "UGRD", "idx_level": "10 m above ground"},  # 10-m u-wind (m/s)
    "v10m": {"idx_name": "VGRD", "idx_level": "10 m above ground"},  # 10-m v-wind (m/s)
    # 80 m (wind turbine hub height)
    "u80m": {"idx_name": "UGRD", "idx_level": "80 m above ground"},  # 80-m u-wind (m/s)
    "v80m": {"idx_name": "VGRD", "idx_level": "80 m above ground"},  # 80-m v-wind (m/s)
    # Pressure / mass
    "sp": {"idx_name": "PRES", "idx_level": "surface"},  # surface pressure (Pa)
    "mslp": {"idx_name": "MSLMA", "idx_level": "mean sea level"},  # mean sea-level pressure (Pa)
    "orog": {"idx_name": "HGT", "idx_level": "surface"},  # orography / model terrain height (m)
    # Wind
    "gust": {"idx_name": "GUST", "idx_level": "surface"},  # surface wind gust speed (m/s)
    "fricv": {"idx_name": "FRICV", "idx_level": "surface"},  # friction velocity (m/s)
    # Precipitation
    "prate": {"idx_name": "PRATE", "idx_level": "surface"},  # precipitation rate (kg/m²/s)
    "tp": {"idx_name": "APCP", "idx_level": "surface"},  # accumulated total precipitation (kg/m²)
    # Reflectivity
    "refc": {"idx_name": "REFC", "idx_level": "entire atmosphere"},  # composite reflectivity (dBZ)
    # Convection
    "cape": {"idx_name": "CAPE", "idx_level": "surface"},  # convective available potential energy (J/kg)
    "cin": {"idx_name": "CIN", "idx_level": "surface"},  # convective inhibition (J/kg)
    # Boundary layer
    "hpbl": {"idx_name": "HPBL", "idx_level": "surface"},  # planetary boundary layer height (m)
    "vis": {"idx_name": "VIS", "idx_level": "surface"},  # surface visibility (m)
    # Radiation (instantaneous fluxes at the surface, W/m²)
    "dswrf": {"idx_name": "DSWRF", "idx_level": "surface"},  # downward shortwave radiation flux
    "uswrf": {"idx_name": "USWRF", "idx_level": "surface"},  # upward shortwave radiation flux
    "dlwrf": {"idx_name": "DLWRF", "idx_level": "surface"},  # downward longwave radiation flux
    "ulwrf": {"idx_name": "ULWRF", "idx_level": "surface"},  # upward longwave radiation flux
    # Surface energy / heat fluxes (W/m²)
    "shtfl": {"idx_name": "SHTFL", "idx_level": "surface"},  # sensible heat flux
    "lhtfl": {"idx_name": "LHTFL", "idx_level": "surface"},  # latent heat flux
    # Snow / land surface
    "snowd": {"idx_name": "SNOD", "idx_level": "surface"},  # snow depth (m)
    "weasd": {"idx_name": "WEASD", "idx_level": "surface"},  # water equivalent of snow depth (kg/m²)
    "snowc": {"idx_name": "SNOWC", "idx_level": "surface"},  # snow cover (%)
}

# Maximum parallel workers for remote fetching
_MAX_REMOTE_WORKERS = 8

# Timeout (seconds) for HTTPS requests to AWS S3.
# Passed as (connect_timeout, read_timeout) — requests treats them independently.
# The read timeout covers waiting for S3 to begin streaming the response body;
# large GRIB messages (~10 MB) over a slow or loaded connection can exceed 30 s.
_HTTP_TIMEOUT: tuple[int, int] = (10, 120)  # (connect, read)

#: Supported HRRR GRIB2 products.
VALID_PRODUCTS = {"HRRR": "wrfprsf", "HRRR_NAT": "wrfnatf", "HRRR_SUBH": "wrfsubhf"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _hrrr_s3_uri(t: pd.Timestamp, forecast_hour: int, product: str = "wrfprsf") -> str:
    """Construct the S3 URI for a HRRR grib2 file.

    Args:
        t: Initialisation timestamp (UTC).
        forecast_hour: Forecast lead hour (FF), e.g. ``0`` for analysis.
        product: HRRR product name — one of ``VALID_PRODUCTS``.
    """
    date_str = t.strftime("%Y%m%d")
    hour_str = t.strftime("%H")
    fname = f"hrrr.t{hour_str}z.{product}{forecast_hour:02d}.grib2"
    subdir = "conus/" if t >= _HRRR_V3_CUTOFF else ""
    return f"s3://{_S3_BUCKET}/hrrr.{date_str}/{subdir}{fname}"


def _hrrr_local_path(base_path: str, t: pd.Timestamp, forecast_hour: int, product: str = "wrfprsf") -> str:
    """Construct the local filesystem path for a HRRR grib2 file.

    Args:
        base_path: Root directory containing HRRR data.
        t: Initialisation timestamp (UTC).
        forecast_hour: Forecast lead hour (FF).
        product: HRRR product name — one of ``VALID_PRODUCTS``.
    """
    date_str = t.strftime("%Y%m%d")
    hour_str = t.strftime("%H")
    fname = f"hrrr.t{hour_str}z.{product}{forecast_hour:02d}.grib2"
    if t >= _HRRR_V3_CUTOFF:
        return os.path.join(base_path, f"hrrr.{date_str}", "conus", fname)
    return os.path.join(base_path, f"hrrr.{date_str}", fname)


def _s3_uri_to_https(s3_uri: str) -> str:
    """Convert an ``s3://noaa-hrrr-bdp-pds/...`` URI to a public HTTPS URL."""
    key = s3_uri[len(f"s3://{_S3_BUCKET}/") :]
    return f"{_HRRR_HTTPS_BASE}/{key}"


# ---------------------------------------------------------------------------
# Remote reading: .idx parsing + parallel HTTPS range fetching
# ---------------------------------------------------------------------------


def _parse_idx(text: str) -> list[dict]:
    """Parse a HRRR ``.idx`` inventory file into a list of message entries.

    Each entry dict has keys: ``var``, ``level``, ``byte_start``, ``byte_end``
    (``None`` for the last entry, meaning read to EOF).
    """
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 6:
            continue
        entries.append(
            {
                "var": parts[3].strip(),
                "level": parts[4].strip(),
                "step": parts[5].strip() if len(parts) > 5 else "",
                "byte_start": int(parts[1]),
                "byte_end": None,
            }
        )
    for i in range(len(entries) - 1):
        entries[i]["byte_end"] = entries[i + 1]["byte_start"] - 1
    return entries


def _fetch_idx(s3_uri: str) -> list[dict]:
    """Fetch and parse the ``.idx`` sidecar for a HRRR grib2 file via HTTPS.

    Raises:
        FileNotFoundError: If the ``.idx`` file is not found (older v1/v2 files
            may lack sidecars; pre-download with ``hrrr_download.py`` and use
            local mode instead).
    """
    import requests  # noqa: PLC0415

    url = _s3_uri_to_https(s3_uri) + ".idx"
    resp = requests.get(url, timeout=_HTTP_TIMEOUT)
    if resp.status_code == 404:
        raise FileNotFoundError(
            f"HRRR .idx file not found: {url}\n"
            "Older HRRR files (v1/v2) may lack .idx files. "
            "Pre-download with hrrr_download.py and use local mode."
        )
    resp.raise_for_status()
    return _parse_idx(resp.text)


def _fetch_message(
    https_url: str,
    byte_start: int,
    byte_end: int | None,
    session=None,
) -> bytes:
    """Fetch a single GRIB message via an HTTP Range request.

    Args:
        https_url: Public HTTPS URL of the grib2 file.
        byte_start: First byte of the message (inclusive).
        byte_end: Last byte of the message (inclusive), or ``None`` for EOF.
        session: Optional ``requests.Session`` for connection reuse.  Falls
            back to module-level ``requests.get`` if ``None``.
    """
    import requests  # noqa: PLC0415

    range_header = f"bytes={byte_start}-{byte_end}" if byte_end is not None else f"bytes={byte_start}-"
    getter = session.get if session is not None else requests.get
    try:
        resp = getter(https_url, headers={"Range": range_header}, timeout=_HTTP_TIMEOUT)
    except requests.exceptions.ConnectionError:
        # AWS S3 closes idle keep-alive connections after ~20 s.  On the next
        # request the session tries to reuse the stale socket and gets
        # RemoteDisconnected.  One retry is enough — the second attempt opens
        # a fresh connection.
        resp = getter(https_url, headers={"Range": range_header}, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def _build_prs_entry_map(idx_entries: list[dict], idx_name: str) -> dict[float, dict]:
    """Return a ``{pressure_level_hPa: idx_entry}`` dict for a pressure-level variable."""
    result: dict[float, dict] = {}
    for e in idx_entries:
        if e["var"] == idx_name and e["level"].endswith(" mb"):
            try:
                lv_f = float(e["level"].replace(" mb", ""))
            except ValueError:
                continue
            result[lv_f] = e
    return result


def _resolve_pressure_levels(
    requested: list[int] | None,
    prs_map: dict[float, dict],
    var_name: str,
) -> list[float]:
    """Return the float pressure levels to fetch, validating against available."""
    if requested is None:
        return sorted(prs_map.keys(), reverse=True)

    avail = sorted(prs_map.keys())
    resolved, missing = [], []
    for lv in requested:
        match = next((k for k in avail if abs(k - lv) < 0.5), None)
        if match is None:
            missing.append(lv)
        else:
            resolved.append(match)
    if missing:
        raise ValueError(
            f"Pressure levels {missing} not found for '{var_name}' in .idx. "
            f"Available: {[int(k) if k == int(k) else k for k in avail]}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Native (hybrid-sigma) level helpers — wrfnatf
# ---------------------------------------------------------------------------


def _build_nat_entry_map(idx_entries: list[dict], idx_name: str) -> dict[int, dict]:
    """Return ``{hybrid_level_index: idx_entry}`` for a wrfnatf variable.

    HRRR native-level ``.idx`` entries look like::

        TMP:10 hybrid level:anl:

    i.e. ``level`` ends with ``" hybrid level"`` and the prefix is the integer
    level index (1–65, bottom-up).
    """
    result: dict[int, dict] = {}
    for e in idx_entries:
        if e["var"] == idx_name and e["level"].endswith(" hybrid level"):
            try:
                lv = int(e["level"].replace(" hybrid level", ""))
            except ValueError:
                continue
            result[lv] = e
    return result


def _resolve_nat_levels(
    requested: list[int] | None,
    nat_map: dict[int, dict],
    var_name: str,
) -> list[int]:
    """Return native level indices to fetch, validating against available."""
    if requested is None:
        return sorted(nat_map.keys())
    avail = sorted(nat_map.keys())
    resolved, missing = [], []
    for lv in requested:
        if lv in avail:
            resolved.append(lv)
        else:
            missing.append(lv)
    if missing:
        raise ValueError(f"Native levels {missing} not found for '{var_name}' in .idx. Available: {avail}")
    return resolved


# ---------------------------------------------------------------------------
# Sub-hourly helpers — wrfsubhf
# ---------------------------------------------------------------------------


def _find_subhf_entry(
    idx_entries: list[dict],
    idx_name: str,
    idx_level: str,
    step_min: int,
) -> dict:
    """Return the idx entry for a wrfsubhf variable at a specific sub-step.

    Sub-hourly ``.idx`` entries have a ``step`` field like ``"15 min fcst"``,
    ``"30 min fcst"``, ``"45 min fcst"``, ``"60 min fcst"``.

    Args:
        idx_entries: Parsed ``.idx`` entries for the wrfsubhf file.
        idx_name: Variable name as it appears in the ``.idx``.
        idx_level: Level string (e.g. ``"2 m above ground"``).
        step_min: Sub-step in minutes (15, 30, 45, 60, …).

    Raises:
        KeyError: If no matching entry is found.
    """
    step_str = f"{step_min} min fcst"
    for e in idx_entries:
        if e["var"] == idx_name and e["level"] == idx_level and e.get("step", "") == step_str:
            return e
    raise KeyError(
        f"No .idx entry for '{idx_name}' at level='{idx_level}', step='{step_str}'. "
        "Verify that the wrfsubhf .idx step strings match the expected format."
    )


# ---------------------------------------------------------------------------
# Local byte-range reading (mirrors the remote HTTPS approach)
# ---------------------------------------------------------------------------


def _fetch_bytes_local(path: str, byte_start: int, byte_end: int | None) -> bytes:
    """Read a byte range directly from a local GRIB2 file.

    Args:
        path: Absolute path to the local grib2 file.
        byte_start: First byte (inclusive).
        byte_end: Last byte (inclusive), or ``None`` to read to EOF.

    Returns:
        Raw bytes for that message.
    """
    with open(path, "rb") as f:
        f.seek(byte_start)
        if byte_end is not None:
            return f.read(byte_end - byte_start + 1)
        return f.read()


def _load_idx_local(grib2_path: str) -> list[dict]:
    """Read and parse the ``.idx`` sidecar from local disk.

    Expects the index at ``{grib2_path}.idx``.  Download it alongside the
    grib2 with ``hrrr_download.py``.

    Raises:
        FileNotFoundError: If the ``.idx`` file is absent.
    """
    idx_path = grib2_path + ".idx"
    try:
        with open(idx_path) as f:
            return _parse_idx(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Local .idx file not found: {idx_path}\nRe-run hrrr_download.py — it fetches the .idx alongside the grib2."
        ) from None


# ---------------------------------------------------------------------------
# DataArray builders
# ---------------------------------------------------------------------------


def _to_float32(values: np.ndarray) -> np.ndarray:
    """Return float32, replacing masked values with NaN."""
    if hasattr(values, "filled"):
        values = values.filled(np.nan)
    return values.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HRRRDataset(Dataset):
    """PyTorch Dataset for HRRR GRIB2 data (wrfprsf / wrfnatf / wrfsubhf).

    Implements the same field-type semantics as MRMSDataset:

    * ``prognostic``      — input at step 0 and target (autoregressive rollout)
    * ``dynamic_forcing`` — input at every step; never a target
    * ``diagnostic``      — target only

    Both modes use ``pygrib`` for GRIB2 decoding.  Remote mode fetches the
    ``.idx`` sidecar and issues parallel HTTP Range requests — no full file
    download required.

    See module docstring for full output format, tensor shapes, and YAML
    configuration examples.

    Attributes:
        source_name: Tensor key prefix — ``"hrrr"``, ``"hrrr_nat"``, or
            ``"hrrr_subh"`` depending on *product*.
        product: Active HRRR product (``"wrfprsf"``, ``"wrfnatf"``, or
            ``"wrfsubhf"``).
        datetimes: DatetimeIndex of valid initialisation timestamps.
        static_metadata: Dataset-level metadata for MultiSourceDataset.
    """

    def __init__(
        self,
        config: dict,
        return_target: bool = False,
        # product: str = "wrfprsf",
        # config_key: str = "HRRR",
    ) -> None:
        """Initialise HRRRDataset.

        Args:
            config: Top-level ``data`` config dict.
            return_target: Whether to include a ``"target"`` key in each sample.
            product: HRRR product to load.  One of ``VALID_PRODUCTS``:
                ``"wrfprsf"`` (pressure-level, default), ``"wrfnatf"``
                (native/hybrid-sigma levels), or ``"wrfsubhf"`` (15-min
                sub-hourly surface).
            config_key: Key under ``config["source"]`` containing this
                product's settings.  Defaults to ``"HRRR"`` for the
                pressure-level product; pass ``"HRRR_NAT"`` or
                ``"HRRR_SUBH"`` for the other products.
        """
        # Get the configuration key from the source config:
        if "source" not in config:
            raise ValueError(f"Missing 'source' key in config: {config}")

        if len(config["source"]) != 1:
            raise ValueError("Expected exactly one source in config['source'], " + f"got: {config['source'].keys()}")
        config_key = list(config["source"].keys())[0]  # Probably a more pythonic way to do this

        if config_key not in VALID_PRODUCTS:
            raise ValueError(
                f"Unknown HRRR product '{config_key}' in config['source']."
                + f"Valid products mapped as: {VALID_PRODUCTS}"
            )
        product = VALID_PRODUCTS[config_key]
        source_cfg = config["source"][config_key]

        self.product: str = product
        self.source_name: str = config_key.lower()
        self.return_target: bool = return_target
        self.mode: str = source_cfg.get("mode", "local")
        self.base_path: str | None = source_cfg.get("base_path", None)
        self.forecast_hour: int = int(source_cfg.get("forecast_hour", 0))
        self.extent: list[float] | None = source_cfg.get("extent", None)
        self.global_levels: list[int] | None = source_cfg.get("levels", None)
        self.num_fetch_workers: int = int(source_cfg.get("num_fetch_workers", _MAX_REMOTE_WORKERS))

        self.dt = pd.Timedelta(config["timestep"])
        self.num_forecast_steps: int = config["forecast_len"]
        self.start_datetime = pd.Timestamp(config["start_datetime"])
        self.end_datetime = pd.Timestamp(config["end_datetime"])
        self.datetimes: pd.DatetimeIndex = self._build_timestamps()

        if self.mode == "local" and self.base_path is None:
            raise ValueError(f"config['source']['{config_key}']['base_path'] is required for local mode")

        if "variables" not in source_cfg:
            raise KeyError(
                f"Missing 'variables' key in config['source']['{config_key}']" + f"Current keys: {source_cfg.keys()}"
            )
        if len(source_cfg["variables"]) == 0:
            raise ValueError(
                f"No variables specified under config['source']['{config_key}']['variables']"
                + f"Current dictionary: {source_cfg['variables']}"
            )

        self.var_dict: dict[str, dict] = {}
        for field_type, d in source_cfg.get("variables", {}).items():
            self._register_field(field_type, d)

        self.static_metadata: dict = {
            "levels": self.global_levels,
            "forecast_hour": self.forecast_hour,
            "datetime_fmt": "unix_ns",
        }

        # Caches — all created lazily so they are fork-safe when DataLoader
        # spins up worker processes after __init__.
        self._idx_cache: dict[str, list[dict]] = {}
        self._http_session = None  # requests.Session; built on first remote call
        self._spatial_slice: tuple[slice, slice] | None = None  # extent → (row, col) slices

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def _get_session(self):
        """Return the shared ``requests.Session``, creating it on first call.

        Created lazily so the session is never open before a DataLoader worker
        forks — each worker ends up with its own independent connection pool.
        """
        import requests  # noqa: PLC0415

        if self._http_session is None:
            self._http_session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=_MAX_REMOTE_WORKERS,
                pool_maxsize=_MAX_REMOTE_WORKERS,
            )
            self._http_session.mount("https://", adapter)
        return self._http_session

    def _get_spatial_slice(self, lats: np.ndarray, lons: np.ndarray) -> tuple[slice, slice]:
        """Return ``(row_slice, col_slice)`` for ``self.extent``, computed once.

        The HRRR grid is fixed (Lambert Conformal Conic, ~1059 × 1799), so the
        bounding-box row/col indices for a given ``extent`` are identical for
        every message and every timestep.  The result is cached after the first
        call so subsequent samples pay no recomputation cost.

        Args:
            lats: 2-D latitude array from a decoded pygrib message.
            lons: 2-D longitude array from a decoded pygrib message.

        Returns:
            ``(row_slice, col_slice)`` ready for direct numpy indexing.
            Both slices are ``slice(None)`` when ``self.extent`` is ``None``.

        Raises:
            ValueError: If ``self.extent`` does not intersect the HRRR domain.
        """
        if self._spatial_slice is not None:
            return self._spatial_slice

        if self.extent is None:
            self._spatial_slice = (slice(None), slice(None))
            return self._spatial_slice

        min_lon, max_lon, min_lat, max_lat = self.extent
        min_lon = (min_lon + 180.0) % 360.0 - 180.0
        max_lon = (max_lon + 180.0) % 360.0 - 180.0
        lon_norm = (lons + 180.0) % 360.0 - 180.0

        mask = (lats >= min_lat) & (lats <= max_lat) & (lon_norm >= min_lon) & (lon_norm <= max_lon)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        if rows.size == 0 or cols.size == 0:
            raise ValueError(f"extent {self.extent} does not intersect the HRRR CONUS domain.")

        self._spatial_slice = (
            slice(int(rows[0]), int(rows[-1]) + 1),
            slice(int(cols[0]), int(cols[-1]) + 1),
        )
        return self._spatial_slice

    def __len__(self) -> int:
        return len(self.datetimes)

    def __getitem__(self, args: tuple) -> dict:
        """Return a nested input/target sample dict.

        Args:
            args: ``(t, i)`` where *t* is the init timestamp (nanoseconds or
                ``pd.Timestamp``) and *i* is the within-sequence step index.

        Returns:
            Dict with ``"input"``, ``"metadata"``, and optionally ``"target"``.
        """
        t, i = args
        t = pd.Timestamp(t)
        t_target = t + self.dt

        input_data: dict = {}
        self._extract_field("dynamic_forcing", t, input_data)
        if i == 0:
            self._extract_field("prognostic", t, input_data)

        sample: dict = {
            "input": input_data,
            "metadata": {"input_datetime": int(t.value)},
        }

        if self.return_target:
            target_data: dict = {}
            for ft in ("prognostic", "diagnostic"):
                if ft in self.var_dict:
                    self._extract_field(ft, t_target, target_data)
            sample["target"] = target_data
            sample["metadata"]["target_datetime"] = int(t_target.value)

        return sample

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_timestamps(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.start_datetime,
            self.end_datetime - self.num_forecast_steps * self.dt,
            freq=self.dt,
        )

    def _register_field(self, field_type: str, d: dict | None) -> None:
        if field_type not in VALID_FIELD_TYPES:
            raise KeyError(f"Unknown field_type '{field_type}'. Valid options: {sorted(VALID_FIELD_TYPES)}")
        if not isinstance(d, dict):
            return

        vars_3d: list[str] = d.get("vars_3D") or []
        vars_2d: list[str] = d.get("vars_2D") or []
        if not vars_3d and not vars_2d:
            raise ValueError(f"Field '{field_type}' must define vars_3D and/or vars_2D")

        for vname in vars_3d + vars_2d:
            if vname not in VAR_REGISTRY:
                raise KeyError(f"Variable '{vname}' is not in VAR_REGISTRY. Available: {sorted(VAR_REGISTRY)}")

        self.var_dict[field_type] = {
            "vars_3D": vars_3d,
            "vars_2D": vars_2d,
            "levels": d.get("levels", self.global_levels),
        }

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict,
    ) -> None:
        """Load all variables for *field_type* at time *t* into *sample*.

        Resolves the file path / URI, loads the ``.idx`` (cached), then
        delegates to :meth:`_extract_from_idx` with the appropriate byte
        fetcher for the current mode.

        For ``wrfsubhf``, *t* is a 15-min-resolution timestamp.  This method
        derives the HRRR init time and FF file number automatically:

        * ``init_hour = t.floor("1h")``
        * ``step_min  = minutes since init`` (15, 30, 45, 60, …)
        * ``ff        = ceil(step_min / 60)`` (file number within the run)
        * If *t* is exactly on the hour, it is treated as the 60-min step of
          the previous hour's run (``init_hour -= 1h``, ``step_min = 60``).
        """
        vd = self.var_dict.get(field_type)
        if not vd:
            return

        # ------------------------------------------------------------------
        # Compute effective init time, FF file number, and sub-step for subhf
        # ------------------------------------------------------------------
        if self.product == "wrfsubhf":
            init_hour = t.floor("1h")
            step_min = int((t - init_hour).total_seconds() / 60)
            if step_min == 0:
                # t is on the hour → 60-min step of the previous run
                init_hour = init_hour - pd.Timedelta("1h")
                step_min = 60
            ff = (step_min + 59) // 60  # ceil: 1-60 → 1, 61-120 → 2, …
            file_t = init_hour
        else:
            ff = self.forecast_hour
            step_min = None
            file_t = t

        if self.mode == "remote":
            s3_uri = _hrrr_s3_uri(file_t, ff, self.product)
            if s3_uri not in self._idx_cache:
                self._idx_cache[s3_uri] = _fetch_idx(s3_uri)
            idx_entries = self._idx_cache[s3_uri]
            https_url = _s3_uri_to_https(s3_uri)
            session = self._get_session()

            def _fetcher(entry: dict) -> bytes:
                return _fetch_message(https_url, entry["byte_start"], entry["byte_end"], session)
        else:
            path = _hrrr_local_path(self.base_path, file_t, ff, self.product)
            if path not in self._idx_cache:
                self._idx_cache[path] = _load_idx_local(path)
            idx_entries = self._idx_cache[path]

            def _fetcher(entry: dict) -> bytes:
                return _fetch_bytes_local(path, entry["byte_start"], entry["byte_end"])

        self._extract_from_idx(field_type, idx_entries, _fetcher, vd, sample, step_min=step_min)

    def _extract_from_idx(
        self,
        field_type: str,
        idx_entries: list[dict],
        fetcher,
        vd: dict,
        sample: dict,
        step_min: int | None = None,
    ) -> None:
        """Shared fetch-plan → parallel byte fetch → decode → tensor pipeline.

        Used by both local and remote modes.  The only difference between modes
        is the *fetcher* callable that maps an idx entry to raw GRIB bytes.
        Product-specific level dispatch (pressure vs hybrid-sigma vs sub-hourly)
        is handled here based on ``self.product``.

        Args:
            field_type: e.g. ``"prognostic"``.
            idx_entries: Parsed ``.idx`` entries for the target file.
            fetcher: Callable ``(entry: dict) -> bytes`` that fetches the raw
                GRIB message for a given idx entry.
            vd: Variable dict (``vars_3D``, ``vars_2D``, ``levels``).
            sample: Output dict to populate in-place.
            step_min: Sub-hourly step in minutes (15, 30, 45, 60, …).  Only
                used when ``self.product == "wrfsubhf"``.
        """
        try:
            import pygrib  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pygrib is required: pip install pygrib") from exc

        levels = vd["levels"]

        # ------------------------------------------------------------------
        # wrfsubhf — surface-only product, 3D vars not supported
        # ------------------------------------------------------------------
        if self.product == "wrfsubhf":
            if vd["vars_3D"]:
                raise ValueError(f"wrfsubhf is a surface-only product; vars_3D is not supported. Got: {vd['vars_3D']}")
            if step_min is None:
                raise ValueError("step_min is required for wrfsubhf extraction")

        # ------------------------------------------------------------------
        # Build fetch plan: list of (var_name, is_3d, level_value|None, entry)
        # ------------------------------------------------------------------
        fetch_plan: list[tuple] = []

        for vname in vd["vars_3D"]:
            reg = VAR_REGISTRY[vname]
            if self.product == "wrfnatf":
                nat_map = _build_nat_entry_map(idx_entries, reg["idx_name"])
                for lv in _resolve_nat_levels(levels, nat_map, vname):
                    fetch_plan.append((vname, True, lv, nat_map[lv]))
            else:
                # wrfprsf (default pressure-level path)
                prs_map = _build_prs_entry_map(idx_entries, reg["idx_name"])
                for lv in _resolve_pressure_levels(levels, prs_map, vname):
                    fetch_plan.append((vname, True, lv, prs_map[lv]))

        for vname in vd["vars_2D"]:
            reg = VAR_REGISTRY[vname]
            if self.product == "wrfsubhf":
                entry = _find_subhf_entry(
                    idx_entries,
                    reg["idx_name"],
                    reg["idx_level"],
                    step_min,  # type: ignore[arg-type]
                )
            else:
                matching = [e for e in idx_entries if e["var"] == reg["idx_name"] and e["level"] == reg["idx_level"]]
                if not matching:
                    raise KeyError(
                        f"No .idx entry for '{vname}' (idx_name='{reg['idx_name']}', idx_level='{reg['idx_level']}')"
                    )
                entry = matching[0]
            fetch_plan.append((vname, False, None, entry))

        # ------------------------------------------------------------------
        # Fetch all messages, then decode.
        #
        # Remote mode: ThreadPoolExecutor issues HTTP Range requests in
        #   parallel.  Each request has ~200 ms network latency, so parallelism
        #   provides a large speedup (30 sequential = ~6 s vs ~200 ms parallel).
        #
        # Local mode: disk seeks + reads are cheap and largely sequential at
        #   the OS level.  Thread overhead outweighs any gain, so we read
        #   sequentially.  Note that DataLoader num_workers already provides
        #   process-level parallelism across samples in local mode.
        # ------------------------------------------------------------------
        if self.mode == "remote":
            n_workers = min(len(fetch_plan), self.num_fetch_workers)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                raw_messages = list(pool.map(lambda task: fetcher(task[3]), fetch_plan))
        else:
            raw_messages = [fetcher(task[3]) for task in fetch_plan]

        decoded = [pygrib.fromstring(raw) for raw in raw_messages]

        # ------------------------------------------------------------------
        # Compute the spatial slice once from the first message's lat/lon grid.
        # The HRRR grid is fixed, so this result is cached for subsequent calls.
        # ------------------------------------------------------------------
        lats, lons = decoded[0].latlons()
        row_sl, col_sl = self._get_spatial_slice(lats, lons)

        # ------------------------------------------------------------------
        # Group decoded arrays by variable name and build tensors
        # ------------------------------------------------------------------
        arrs_3d: dict[str, list[np.ndarray]] = defaultdict(list)
        lvls_3d: dict[str, list] = defaultdict(list)
        arr_2d: dict[str, np.ndarray] = {}

        for (vname, is_3d, lv, _), msg in zip(fetch_plan, decoded):
            arr = _to_float32(msg.values)[row_sl, col_sl]
            if is_3d:
                arrs_3d[vname].append(arr)
                lvls_3d[vname].append(lv)
            else:
                arr_2d[vname] = arr

        for vname in vd["vars_3D"]:
            stacked = np.stack(arrs_3d[vname])  # (n_levels, y, x)
            sample[f"{self.source_name}/{field_type}/3d/{vname}"] = torch.tensor(
                stacked, dtype=torch.float32
            ).unsqueeze(1)

        for vname in vd["vars_2D"]:
            sample[f"{self.source_name}/{field_type}/2d/{vname}"] = (
                torch.tensor(arr_2d[vname], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
