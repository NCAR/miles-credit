"""
tisr.py
-------------------------------------------------------
TISRDataset: PyTorch Dataset for Total Incident Solar Radiation (TISR) at the top of the atmosphere (TOA).

Sample structure returned by __getitem__::

    {
        "input":    {<user_provided_name>: {"<user_provided_name>/dynamic_forcing/2d/tisr": tensor}},
        "target":   {<user_provided_name>: {}},  # empty since dynamic forcing is only input
        "metadata": {<user_provided_name>: {"input_datetime": int, "target_datetime": int}},
    }

TISR only has a single variable and is 2D. Tensor shape (no batch dimension)::

    (1, 1, lat, lon)   — singleton level dim, consistent with CREDIT Gen2 2D convention

After DataLoader collation the batch dimension is prepended::

    (batch, 1, 1, lat, lon)

Note that Total Incident Solar Radiation (TISR) and Total Solar Irradiance (TSI) are different physical quantities.

- TSI is the total solar power per unit area measured on a plane perpendicular (at a 90 degree angle) to the sun's rays.
  It is measured at TOA and at the mean Sun-Earth distance (1 AU), and it fluctuates slightly with the Sun's 11-year solar cycle.
- TISR is the actual amount of solar energy that hits a specific surface with any orientation. It can be measured at TOA or surface
  level, and it varies with time and location.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

import pandas as pd
import torch
import xarray as xr

from credit.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_TORCH_DTYPE = torch.float32


# ------------------------------------------------------------------
# Total Solar Irradiance (TSI) Data and Interpolation
# ------------------------------------------------------------------
def _era5_tsi_data(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """ERA5-compatible Total Solar Irradiance (TSI) time series.

    Sourced from
    `Graphcast <https://github.com/google-deepmind/graphcast/blob/main/graphcast/solar_radiation.py>`_.

    ECMWF provided the data used for ERA5, which was hardcoded in the IFS
    (cycle 41r2, 2016). Values from 2009 onwards repeat the 1996–2008 period
    (the last completed 13-year solar cycle available when the code was
    written). All values are scaled by 0.9965 to agree better with more
    recent solar observations.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - **times** – 1-D tensor of fractional years, one entry per year
              from 1951.5 to 2034.5 (mid-year sampling).
            - **tsi_values** – 1-D tensor of TSI values (W m⁻²) corresponding
              to each entry in *times*.
    """
    # Mid-year fractional years from 1951 through 2034 (one value per year)
    times = torch.arange(1951.5, 2035.5, 1.0, dtype=_TORCH_DTYPE, device=device)
    tsi_values = 0.9965 * torch.tensor(
        [
            # 1951–1995: non-repeating observational sequence (45 values)
            1365.7765,
            1365.7676,
            1365.6284,
            1365.6564,
            1365.7773,
            1366.3109,
            1366.6681,
            1366.6328,
            1366.3828,
            1366.2767,
            1365.9199,
            1365.7484,
            1365.6963,
            1365.6976,
            1365.7341,
            1365.9178,
            1366.1143,
            1366.1644,
            1366.2476,
            1366.2426,
            1365.9580,
            1366.0525,
            1365.7991,
            1365.7271,
            1365.5345,
            1365.6453,
            1365.8331,
            1366.2747,
            1366.6348,
            1366.6482,
            1366.6951,
            1366.2859,
            1366.1992,
            1365.8103,
            1365.6416,
            1365.6379,
            1365.7899,
            1366.0826,
            1366.6479,
            1366.5533,
            1366.4457,
            1366.3021,
            1366.0286,
            1365.7971,
            1365.6996,
            # 1996–2008: one complete 13-year solar cycle (template for repeats below)
            1365.6121,
            1365.7399,
            1366.1021,
            1366.3851,
            1366.6836,
            1366.6022,
            1366.6807,
            1366.2300,
            1366.0480,
            1365.8545,
            1365.8107,
            1365.7240,
            1365.6918,
            # 2009–2021: repeat of the 1996–2008 cycle
            1365.6121,
            1365.7399,
            1366.1021,
            1366.3851,
            1366.6836,
            1366.6022,
            1366.6807,
            1366.2300,
            1366.0480,
            1365.8545,
            1365.8107,
            1365.7240,
            1365.6918,
            # 2022–2034: repeat of the 1996–2008 cycle
            1365.6121,
            1365.7399,
            1366.1021,
            1366.3851,
            1366.6836,
            1366.6022,
            1366.6807,
            1366.2300,
            1366.0480,
            1365.8545,
            1365.8107,
            1365.7240,
            1365.6918,
        ],
        dtype=_TORCH_DTYPE,
        device=device,
    )
    return times, tsi_values


def _get_tsi(
    timestamps: Sequence[str | pd.Timestamp],
    tsi_times: torch.Tensor,
    tsi_values: torch.Tensor,
) -> torch.Tensor:
    """Interpolate Total Solar Irradiance (TSI) at the given timestamps.

    Converts each timestamp to a fractional year and performs piecewise
    linear interpolation against the provided annual TSI time series.

    Args:
        timestamps:  Sequence of timestamps (strings or ``pd.Timestamp``
            objects) at which to evaluate TSI.
        tsi_times:   1-D tensor of fractional years (e.g. ``2003.5``),
            sorted in ascending order.
        tsi_values:  1-D tensor of TSI values (W m⁻²) corresponding
            element-wise to *tsi_times*.

    Returns:
        torch.Tensor: 1-D tensor of interpolated TSI values (W m⁻²),
        one per input timestamp.

    Raises:
        ValueError: If any timestamp's **year** falls outside the integer
            range spanned by *tsi_times*.  Note that timestamps in the
            first half of the first year or the second half of the last
            year are still accepted and will be lightly extrapolated.
    """
    # Normalise input to a DatetimeIndex for vectorised calendar operations
    ts = pd.DatetimeIndex([timestamps] if isinstance(timestamps, pd.Timestamp) else timestamps)

    # Extract the date component (time-of-day stripped) for intra-day fraction calculation
    ts_date = pd.DatetimeIndex(ts.date)

    # Guard: reject timestamps whose *year* is entirely outside the TSI dataset.
    # Timestamps in the first/last partial year are still passed through and
    # handled gracefully by the linear interpolation below.
    t_min, t_max = tsi_times[0].item(), tsi_times[-1].item()
    out_mask = (ts.year < int(t_min)) | (ts.year > int(t_max))

    if out_mask.any():
        bad = ts[out_mask].tolist()
        raise ValueError(
            f"{len(bad)} timestamp(s) fall outside the TSI data range "
            f"[{t_min:.4f}, {t_max:.4f}]: {bad[:5]}" + (" ..." if len(bad) > 5 else "")
        )

    # Compute sub-day fraction: 0.0 at midnight, approaching 1.0 at the next midnight
    day_frac = (ts - ts_date) / pd.Timedelta(days=1)

    # Days in year (accounts for leap years)
    year_len = 365 + ts.is_leap_year

    # Fractional year: e.g. 1 Jan noon ≈ year + 0.001; 31 Dec midnight ≈ year + 0.999
    year_frac = (ts.dayofyear - 1 + day_frac) / year_len
    frac_year = torch.tensor((ts.year + year_frac).to_numpy(), dtype=tsi_times.dtype, device=tsi_times.device)

    # Interpolate TSI values at the given fractional years using linear interpolation
    idx = torch.searchsorted(
        tsi_times.contiguous(), frac_year.contiguous()
    )  # searchsorted requires tensors are contiguous in memory

    # Clamp indices so that boundary timestamps don't go out of bounds
    idx = idx.clamp(1, len(tsi_times) - 1)

    # Indices of the surrounding annual samples
    lo, hi = idx - 1, idx

    # Linear interpolation weight between the two surrounding annual samples
    t = (frac_year - tsi_times[lo]) / (tsi_times[hi] - tsi_times[lo])
    return tsi_values[lo] + t * (tsi_values[hi] - tsi_values[lo])


# ------------------------------------------------------------------
# Lat/Lon Grid Loading and Construction
# ------------------------------------------------------------------
def _get_latlon_grid(
    path: str | None = None,
    lat_spec: Sequence[float] | None = None,
    lon_spec: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Obtain a lat/lon grid, either by reading a NetCDF file or building one from specs.

    Exactly one of the two modes must be supplied:

    * **From file** (``path``) – read latitude/longitude from a NetCDF file.
      Handles two representations:

      - *Curvilinear grids* – latitude/longitude stored as 2-D variables of
        shape ``(ny, nx)``, read directly.
      - *Rectangular grids* – latitude/longitude stored as 1-D coordinate
        arrays of lengths ``ny`` and ``nx``; a 2-D meshgrid is constructed.

      Common field name aliases (``latitude``/``lat``/``XLAT``/``nav_lat`` and
      ``longitude``/``lon``/``XLONG``/``nav_lon``) are tried in order so files
      from different models/tools are accepted without preprocessing.

    * **From specs** (``lat_spec`` and ``lon_spec``) – synthesize a rectangular
      grid in-memory without any file I/O. Each spec is ``[start, end, num_points]``
      with both endpoints inclusive (so ``[90, -90, 721]`` yields the ERA5 0.25°
      latitude axis). Both specs must be supplied together; callers are expected
      to validate the pair (see :class:`TISRDataset.__init__`).

    Args:
        path (str | None): Path to a NetCDF file containing latitude/longitude
            information. Mutually exclusive with ``lat_spec``/``lon_spec``.
        lat_spec (Sequence[float] | None): Latitude axis as ``[start, end, num_points]``
            (e.g. ``[90, -90, 721]``). Mutually exclusive with ``path``.
        lon_spec (Sequence[float] | None): Longitude axis as ``[start, end, num_points]``
            (e.g. ``[0, 359.75, 1440]``). Mutually exclusive with ``path``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - **lat_tensor** – Latitude grid in degrees, shape ``(1, ny, nx)``.
            - **lon_tensor** – Longitude grid in degrees, shape ``(1, ny, nx)``.

    Raises:
        ValueError: If neither or both modes are supplied; if only one of
            ``lat_spec``/``lon_spec`` is given; if a spec is not length 3, or its
            ``num_points`` is not an int >= 1; if the file cannot be opened; if no
            recognised latitude/longitude field is found; or if the lat/lon arrays
            are neither 1-D nor 2-D after squeezing.
    """
    # Determine which mode is active. Spec mode counts as active if *either* axis
    # is given, so supplying only a path vs. only spec(s) routes correctly here;
    # a one-sided spec pair is validated upstream in __init__.
    from_file = path is not None
    from_spec = lat_spec is not None or lon_spec is not None

    # True in exactly the two bad cases: neither source, or both sources.
    if from_file == from_spec:
        raise ValueError(
            "Provide exactly one grid source: either 'path' or ('lat_spec' and 'lon_spec'), not both or neither."
        )

    # ----------------------------------------------------------------
    # Mode 1: build a rectangular grid from [start, end, num_points] specs
    # ----------------------------------------------------------------
    if from_spec:
        # Both axes are required together; a one-sided pair has no sensible
        # default. Guards direct callers (TISRDataset.__init__ also checks this).
        if lat_spec is None or lon_spec is None:
            raise ValueError(
                "Both 'lat_spec' and 'lon_spec' are required when building from specs. "
                f"Got lat_spec={lat_spec!r}, lon_spec={lon_spec!r}."
            )

        def _axis(name: str, spec: Sequence[float]) -> torch.Tensor:
            if len(spec) != 3:
                raise ValueError(
                    f"{name} spec must be [start, end, num_points], got {list(spec)!r} "
                    f"({len(spec)} element(s) instead of 3)"
                )
            start, end = float(spec[0]), float(spec[1])
            n = spec[2]
            # num_points must be a true int (reject bool and float like 2.5).
            if not isinstance(n, int) or isinstance(n, bool):
                raise ValueError(f"{name} spec {list(spec)!r}: num_points must be an int, got {type(n).__name__}")
            if n < 1:
                raise ValueError(f"{name} spec {list(spec)!r}: num_points must be >= 1, got {n}")
            # linspace over n points, both endpoints inclusive (n == 1 -> just start).
            return torch.linspace(start, end, n, dtype=_TORCH_DTYPE, device=device)

        lat_1d = _axis("lat", lat_spec)
        lon_1d = _axis("lon", lon_spec)
        # ij indexing -> shape (ny, nx), matching the file-read curvilinear case.
        lat_tensor, lon_tensor = torch.meshgrid(lat_1d, lon_1d, indexing="ij")
        # Prepend singleton dim -> (1, ny, nx), matching mode 1.
        return lat_tensor.unsqueeze(0), lon_tensor.unsqueeze(0)

    # ----------------------------------------------------------------
    # Mode 2: read the grid from a NetCDF file
    # ----------------------------------------------------------------
    # Alias names tried in priority order; first present wins for each axis.
    _LAT_NAMES = ("latitude", "lat", "XLAT", "nav_lat")
    _LON_NAMES = ("longitude", "lon", "XLONG", "nav_lon")

    # Wrap I/O errors as a descriptive, path-tagged ValueError.
    try:
        ds = xr.open_dataset(path)
    except Exception as exc:
        raise ValueError(f"Could not open NetCDF file at '{path}': {exc}") from exc

    # Search both coordinates and data variables so files with non-coordinate lat/lon are handled
    all_fields = set(ds.coords) | set(ds.data_vars)
    lat_key = next((n for n in _LAT_NAMES if n in all_fields), None)
    lon_key = next((n for n in _LON_NAMES if n in all_fields), None)

    # Validate that we found both latitude and longitude keys; if not, raise an error with details about what was found
    if lat_key is None or lon_key is None:
        ds.close()
        raise ValueError(
            f"No latitude/longitude found in '{path}'. "
            f"Searched lat={_LAT_NAMES}, lon={_LON_NAMES}. "
            f"Available fields: {sorted(all_fields)}"
        )

    # Drops any size-1 dimensions. This handles the common case where a file stores lat/lon as (1, ny, nx)
    lat = ds[lat_key].squeeze()
    lon = ds[lon_key].squeeze()

    # Check the dimensionality of the latitude and longitude arrays after squeezing. We expect either:
    #   - 2-D arrays of shape (ny, nx) for curvilinear grids, or
    #   - 1-D arrays of shape (ny,) and (nx,) for rectangular grids. Any other shape is unexpected and will raise an error.
    if lat.ndim == 2 and lon.ndim == 2:
        # curvilinear grid case: lat and lon are already 2-D arrays of shape (ny, nx)
        lat_tensor = torch.tensor(lat.values, dtype=_TORCH_DTYPE, device=device)
        lon_tensor = torch.tensor(lon.values, dtype=_TORCH_DTYPE, device=device)
    elif lat.ndim == 1 and lon.ndim == 1:
        # rectangular grid case: lat and lon are 1-D arrays; create a 2-D meshgrid
        lat_1d = torch.tensor(lat.values, dtype=_TORCH_DTYPE, device=device)
        lon_1d = torch.tensor(lon.values, dtype=_TORCH_DTYPE, device=device)
        lat_tensor, lon_tensor = torch.meshgrid(lat_1d, lon_1d, indexing="ij")
    else:
        ds.close()
        raise ValueError(
            f"Expected lat/lon to be 1-D or 2-D after squeezing, "
            f"got lat.ndim={lat.ndim}, lon.ndim={lon.ndim} in '{path}'"
        )

    ds.close()
    # Add a leading time dimension of size 1: (ny, nx) → (1, ny, nx)
    return lat_tensor.unsqueeze(0), lon_tensor.unsqueeze(0)


# ------------------------------------------------------------------
# Julian Date and Orbital Parameter Calculations
# ------------------------------------------------------------------
def _get_j2000_days(
    timestamp: pd.Timestamp | pd.DatetimeIndex,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert UTC timestamp(s) to fractional days since the J2000.0 epoch.

    Args:
        timestamp (pd.Timestamp | pd.DatetimeIndex): UTC timestamp or collection
            of timestamps to convert.

    Returns:
        torch.Tensor: Fractional days elapsed since J2000.0 (2000-01-01 12:00:00 UTC).
            Shape matches the input dimensions, matching `_TORCH_DTYPE`.
    References:
        - https://en.wikipedia.org/wiki/Epoch_(astronomy)#Julian_years_and_J2000
    """
    _J2000_EPOCH = 2451545.0
    # Convert the input timestamp(s) to a DatetimeIndex if it's a single Timestamp.
    dti = pd.DatetimeIndex([timestamp] if isinstance(timestamp, pd.Timestamp) else timestamp)

    # Convert to Julian date and then to days since J2000 epoch, returning as a PyTorch tensor
    return torch.tensor(dti.to_julian_date().to_numpy() - _J2000_EPOCH, dtype=_TORCH_DTYPE, device=device)


def _get_orbital_parameters(
    j2000_days: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute solar orbital parameters from J2000 day count.

    Derives the key quantities needed for TISR calculation — solar declination,
    equation of time, and Earth-Sun distance — using low-order trigonometric
    approximations sourced from the ERA5 / IFS parameterization.

    This function is a PyTorch port of ``_get_orbital_parameters`` from Graphcast's
    ``graphcast/solar_radiation.py`` (Google DeepMind).  The logic, variable names,
    and numerical constants are kept identical to the original; the only changes are
    replacing JAX/NumPy array operations (``jnp.stack``, ``jnp.dot``, ``jnp.sin``,
    etc.) with their PyTorch equivalents (``torch.stack``, ``@``, ``torch.sin``,
    etc.), and passing explicit ``dtype`` and ``device`` arguments to tensor
    constructors to ensure compatibility with the calling context.

    Args:
        j2000_days (torch.Tensor): Days elapsed since the J2000 epoch
            (2000-01-01 12:00 TT), shape ``(T,)``.

    Returns:
        dict[str, torch.Tensor]: Dictionary with the following keys, all
        shape ``(T,)`` unless noted:

        - ``theta``: fractional Julian years since J2000.
        - ``rotational_phase``: UTC time-of-day as a day-fraction
          (0.0 = UTC noon, 0.5 = UTC midnight).
        - ``sin_declination``, ``cos_declination``: sine and cosine of the
          solar declination angle (dimensionless).
        - ``eq_of_time_seconds``: equation of time in seconds.
        - ``solar_distance_au``: Earth-Sun distance in Astronomical Units.

    References:
        - https://github.com/google-deepmind/graphcast/blob/08cf73625c9d12bd9aaa038868bcb2fe488f2a22/graphcast/solar_radiation.py#L293

    """
    _JULIAN_YEAR_LENGTH_IN_DAYS = 365.25

    theta = j2000_days / _JULIAN_YEAR_LENGTH_IN_DAYS
    rotational_phase = j2000_days % 1.0

    # intermediate angles in radians (ERA5/IFS variable names preserved from Graphcast)
    rel = 1.7535 + 6.283076 * theta  # mean ecliptic longitude of the Sun
    rem = 6.240041 + 6.283020 * theta  # mean anomaly of Earth's orbit
    rlls = 4.8951 + 6.283076 * theta  # mean ecliptic longitude (used for EoT / declination)

    # precompute trig terms reused across multiple outputs
    one = torch.ones_like(theta)
    sin_rel = torch.sin(rel)
    cos_rel = torch.cos(rel)
    sin_two_rel = torch.sin(2.0 * rel)
    cos_two_rel = torch.cos(2.0 * rel)
    sin_two_rlls = torch.sin(2.0 * rlls)
    cos_two_rlls = torch.cos(2.0 * rlls)
    sin_four_rlls = torch.sin(4.0 * rlls)
    sin_rem = torch.sin(rem)
    sin_two_rem = torch.sin(2.0 * rem)

    # true ecliptic longitude of the Sun
    rllls = torch.stack([one, theta, sin_rel, cos_rel, sin_two_rel, cos_two_rel], dim=-1) @ torch.tensor(
        [4.8952, 6.283320, -0.0075, -0.0326, -0.0003, 0.0002], dtype=j2000_days.dtype, device=j2000_days.device
    )

    # solar declination
    repsm = torch.tensor(0.409093, dtype=j2000_days.dtype, device=j2000_days.device)  # obliquity ≈ 23.44°
    sin_declination = torch.sin(repsm) * torch.sin(rllls)
    cos_declination = torch.sqrt(1.0 - sin_declination**2)

    # equation of time (seconds)
    eq_of_time_seconds = torch.stack(
        [sin_two_rlls, sin_rem, sin_rem * cos_two_rlls, sin_four_rlls, sin_two_rem],
        dim=-1,
    ) @ torch.tensor([591.8, -459.4, 39.5, -12.7, -4.8], dtype=j2000_days.dtype, device=j2000_days.device)

    # Earth-Sun distance in AU (low-order approximation; 1 AU at mean distance)
    solar_distance_au = torch.stack([one, sin_rel, cos_rel], dim=-1) @ torch.tensor(
        [1.0001, -0.0163, 0.0037], dtype=j2000_days.dtype, device=j2000_days.device
    )

    return {
        "theta": theta,
        "rotational_phase": rotational_phase,
        "sin_declination": sin_declination,
        "cos_declination": cos_declination,
        "eq_of_time_seconds": eq_of_time_seconds,
        "solar_distance_au": solar_distance_au,
    }


# ------------------------------------------------------------------
# Solar Geometry: Hour Angle and Zenith Angle
# ------------------------------------------------------------------
def _get_solar_time(
    rotational_phase: torch.Tensor,
    eq_of_time_seconds: torch.Tensor,
) -> torch.Tensor:
    """Compute local apparent solar time as a fraction of a day.

    Adjusts the fractional UTC day (rotational phase) by the Equation of Time
    to account for solar variance due to Earth's orbital eccentricity and axial tilt.

    Args:
        rotational_phase (torch.Tensor): Fractional part of the J2000 day count,
            representing UTC time-of-day, shape ``(T,)``. Because the J2000 epoch
            starts at noon, ``0.0`` represents UTC Noon (12:00) and ``0.5``
            represents UTC Midnight (00:00).
        eq_of_time_seconds (torch.Tensor): Equation of time in seconds,
            shape ``(T,)``. Positive values indicate apparent solar noon occurs
            before mean solar noon.

    Returns:
        torch.Tensor: Apparent solar time at the prime meridian as a fraction
            of a day in ``[0, 1)``, shape ``(T,)``.

    References:
        - https://en.wikipedia.org/wiki/Equation_of_time
    """
    _SECONDS_PER_DAY = 60 * 60 * 24
    # apparent = mean + EOT  (Wikipedia: EOT = apparent − mean)
    # result is referenced to the prime meridian; pass to _get_hour_angle to localize
    solar_time = rotational_phase + eq_of_time_seconds / _SECONDS_PER_DAY  # day-fraction
    return solar_time


def _get_hour_angle(
    solar_time: torch.Tensor,
    longitude: torch.Tensor,
) -> torch.Tensor:
    """Compute the solar hour angle from apparent solar time and longitude.

    The hour angle measures how far the Sun has moved across the sky relative
    to the local meridian: 0° at solar noon, increasing 15° per hour (360° per day).
    Longitude shifts the prime-meridian-referenced solar time to each grid point's
    local meridian.

    Args:
        solar_time (torch.Tensor): Apparent solar time as a fraction of a day,
            with 0.0 corresponding to UTC noon at the prime meridian (J2000 origin).
            Shape broadcastable to ``(T,)``.
        longitude (torch.Tensor): Geographic longitude in degrees (positive east),
            shape broadcastable to ``(ny, nx)``.

    Returns:
        torch.Tensor: Solar hour angle in degrees, shape ``(T, ny, nx)`` after
            broadcasting.  0° at solar noon; ±180° at solar midnight.  Not
            wrapped to ``[−180°, 180°]`` — pass directly to ``torch.deg2rad``
            before taking the cosine.

    References:
        - https://en.wikipedia.org/wiki/Hour_angle#Solar_hour_angle
          (conceptual reference; the page gives no explicit formula, only the
          prose rule "15° per hour before/after solar noon". The formula here
          uses a J2000 rotational phase origin at UTC noon, so the −180° offset
          present in midnight-origin derivations is absent.)
    """
    # 360° per day; solar_time=0 → UTC noon at prime meridian (HA=0° there)
    # adding longitude shifts from prime-meridian reference to local meridian
    hour_angle = 360.0 * solar_time + longitude
    return hour_angle


def _get_cosine_zenith_angle(
    cos_declination: torch.Tensor,
    sin_declination: torch.Tensor,
    latitude: torch.Tensor,
    hour_angle: torch.Tensor,
) -> torch.Tensor:
    """Compute the cosine of the solar zenith angle at each grid point and time.

    Uses the standard spherical-trigonometry identity::

        cos(θ_z) = cos(φ)·cos(δ)·cos(H) + sin(φ)·sin(δ)

    where ``φ`` is geographic latitude, ``δ`` is solar declination, and
    ``H`` is the hour angle.  Negative values (Sun below the horizon) are
    floored to zero; no upper clamp is applied since values above 1.0 are
    physically impossible and should surface as bugs rather than be silently
    masked.

    Args:
        cos_declination (torch.Tensor): Cosine of solar declination, shape
            ``(T,)`` – one value per timestamp.
        sin_declination (torch.Tensor): Sine of solar declination, shape
            ``(T,)``.
        latitude (torch.Tensor): Geographic latitude in degrees, shape
            ``(1, ny, nx)`` or broadcastable equivalent.
        hour_angle (torch.Tensor): Solar hour angle in degrees, shape
            broadcastable to ``(T, ny, nx)``.

    Returns:
        torch.Tensor: Cosine of the solar zenith angle floored at 0,
        shape ``(T, ny, nx)``.  A value of 1.0 means the Sun is directly
        overhead; 0.0 means the Sun is on or below the horizon.

    References:
        https://en.wikipedia.org/wiki/Solar_zenith_angle#Formula
    """
    # cos(zenith) = cos(lat) * cos(declination) * cos(hour_angle) + sin(lat) * sin(declination)
    cos_zenith = (
        torch.cos(torch.deg2rad(latitude)) * cos_declination * torch.cos(torch.deg2rad(hour_angle))
        + torch.sin(torch.deg2rad(latitude)) * sin_declination
    )

    return cos_zenith.clamp(min=0)


# ------------------------------------------------------------------
# TISR Computation
# ------------------------------------------------------------------
def _get_instantaneous_toa_tisr(
    tsi: torch.Tensor,
    solar_factor: torch.Tensor,
    cos_zenith: torch.Tensor,
) -> torch.Tensor:
    """Compute the instantaneous total incident solar radiation at the top of the atmosphere.

    Applies the standard TISR formula::

        tisr = tsi * solar_factor * cos_zenith

    Args:
        tsi (torch.Tensor): Total solar irradiance, in W/m². Broadcast-compatible
            with ``solar_factor`` and ``cos_zenith``.
        solar_factor (torch.Tensor): Earth-Sun distance correction factor, defined
            as the inverse square of the Earth-Sun distance in Astronomical Units
            (AU). Also referred to as the eccentricity correction factor.
        cos_zenith (torch.Tensor): Cosine of the solar zenith angle

    Returns:
        torch.Tensor: Instantaneous total incident solar radiation at the top of
            the atmosphere, in W/m².
    """
    return tsi * solar_factor * cos_zenith.clamp(min=0)


def _get_integrated_toa_tisr(
    instantaneous_toa_tisr: torch.Tensor,
    integration_period: pd.Timedelta = pd.Timedelta(hours=1),
    num_integration_steps: int = 360,
) -> torch.Tensor:
    """Compute the integrated total incident solar radiation at the top of the atmosphere.

    Uses the trapezoidal rule to integrate instantaneous TOA TISR over a given
    period, following ERA5's convention of labeling accumulated fields at the end
    of the accumulation window: ``(target_time - integration_period, target_time]``.
    For example, with the default 1-hour period, the integrated TOA TISR at
    2021-06-01 00:00:00 covers 2021-05-31 23:00:00 to 2021-06-01 00:00:00.

    Args:
        instantaneous_toa_tisr (torch.Tensor): Instantaneous total incident solar
            radiation at the top of the atmosphere. Shape: ``(time_steps, lat, lon)``
            where ``time_steps`` must equal ``num_integration_steps + 1``
            (both endpoints are required for trapezoidal integration).
        integration_period (pd.Timedelta, optional): Duration over which to integrate
            ``instantaneous_toa_tisr``. Defaults to 1 hour (compatible with ERA5).
        num_integration_steps (int, optional): Number of equally-spaced bins over
            the integration period. Defaults to 360.

    Raises:
        ValueError: If ``num_integration_steps`` is not a positive integer.
        ValueError: If ``instantaneous_toa_tisr.shape[0] != num_integration_steps + 1``.

    Returns:
        torch.Tensor: Integrated total incident solar radiation at the top of the
            atmosphere, in J/m². Shape: ``(lat, lon)``.
    """
    # Check that num_integration_steps is a positive integer
    if not isinstance(num_integration_steps, int) or num_integration_steps <= 0:
        raise ValueError(f"num_integration_steps must be a positive integer, but got {num_integration_steps}")

    # Check that instantaneous_toa_tisr.shape[0] equals num_integration_steps + 1
    expected_steps = num_integration_steps + 1
    if instantaneous_toa_tisr.shape[0] != expected_steps:
        raise ValueError(
            f"instantaneous_toa_tisr.shape[0] must equal num_integration_steps + 1 "
            f"({expected_steps}), but got {instantaneous_toa_tisr.shape[0]}"
        )

    # Convert integration period to seconds for physical units (J/m²)
    dt = integration_period / num_integration_steps / pd.Timedelta(seconds=1)

    # Integrate along the time dimension using the trapezoidal rule
    integrated_toa_tisr = torch.trapz(instantaneous_toa_tisr, dx=dt, dim=0)

    return integrated_toa_tisr


def _compute_tisr(
    t: pd.Timestamp,
    integration_period: pd.Timedelta,
    num_integration_steps: int,
    latitude: torch.Tensor,
    longitude: torch.Tensor,
) -> torch.Tensor:
    """Full pipeline for integrated top-of-atmosphere TISR at a target timestamp.

    Vectorized over both space and time — all grid points and timesteps are
    processed in a single pass without any Python-level loops: builds a
    time grid covering the accumulation window, loads the lat/lon grid, retrieves
    total solar irradiance and orbital parameters, computes per-grid-point cosine
    zenith angles, and finally integrates instantaneous TOA TISR over the
    accumulation window using the trapezoidal rule.

    Following ERA5 convention, the accumulation window is the half-open interval
    ``(t - integration_period, t]``. For example, with the default 1-hour period,
    a target time of 2021-06-01 01:00:00 covers 2021-06-01 00:00:00 to
    2021-06-01 01:00:00.

    The ERA5 dataset contains one hourly, 31 km high resolution realisation
    (referred to as "reanalysis" or "HRES") and a reduced resolution ten member
    ensemble (referred to as "ensemble" or "EDA"). For more details, see the
    `ERA5 data documentation <https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation>`_.

    Note:
        This function always returns integrated TISR in J/m². To obtain
        instantaneous TISR in W/m², use :func:`_get_instantaneous_toa_tisr`
        directly.

    Args:
        t (pd.Timestamp): Target timestamp at the end of the accumulation window.
        integration_period (pd.Timedelta): Length of the accumulation window.
            Defaults to 1 hour in callers, consistent with ERA5 hourly accumulations.
        num_integration_steps (int): Number of equally-spaced sub-intervals used
            by the trapezoidal integrator. Higher values increase accuracy.
            Must be a positive integer.
        latitude (torch.Tensor): Latitude grid in degrees, shape ``(1, ny, nx)``.
        longitude (torch.Tensor): Longitude grid in degrees, shape ``(1, ny, nx)``.

    Returns:
        torch.Tensor: Integrated TOA TISR over the accumulation window, in J/m².
            Shape: ``(lat, lon)``.
    """
    # Build a uniform time grid of (num_integration_steps + 1) timestamps ending
    # at t, spanning exactly one integration_period. Both endpoints are included
    # because the trapezoidal rule requires values at every interval boundary.
    ts = pd.date_range(
        end=t,
        periods=num_integration_steps + 1,
        freq=integration_period / num_integration_steps,
    )

    # Reuse the cached grid's device so callers don't pass it separately.
    device = latitude.device

    # Retrieve the total solar irradiance (TSI) time series and interpolate it
    # onto ts. Unsqueeze to add singleton spatial dims for broadcasting: (time, 1, 1).
    times, tsi_values = _era5_tsi_data(device=device)
    tsi = _get_tsi(ts, times, tsi_values).unsqueeze(-1).unsqueeze(-1)

    # Compute orbital parameters (declination, equation of time, Earth-Sun distance,
    # etc.) for each timestep. J2000 days are unsqueezed to (time, 1, 1) so the
    # result broadcasts cleanly against the spatial grid.
    orbital = _get_orbital_parameters(_get_j2000_days(ts, device=device).unsqueeze(-1).unsqueeze(-1))

    # Derive the cosine of the solar zenith angle at every grid point and timestep.
    cos_zenith = _get_cosine_zenith_angle(
        cos_declination=orbital["cos_declination"],
        sin_declination=orbital["sin_declination"],
        latitude=latitude,
        hour_angle=_get_hour_angle(
            solar_time=_get_solar_time(
                orbital["rotational_phase"],
                orbital["eq_of_time_seconds"],
            ),
            longitude=longitude,
        ),
    )

    # Combine TSI, the inverse-square Earth-Sun distance correction (eccentricity
    # factor), and cos_zenith to get instantaneous TOA TISR at every grid point
    # and timestep. Shape: (num_integration_steps + 1, lat, lon).
    instantaneous_toa_tisr = _get_instantaneous_toa_tisr(
        tsi=tsi,
        solar_factor=1.0 / orbital["solar_distance_au"] ** 2,
        cos_zenith=cos_zenith,
    )

    # Integrate over the time dimension using the trapezoidal rule and return the
    # accumulated TISR in J/m² for each grid point. Shape: (lat, lon).
    return _get_integrated_toa_tisr(
        instantaneous_toa_tisr=instantaneous_toa_tisr,
        integration_period=integration_period,
        num_integration_steps=num_integration_steps,
    )


# ------------------------------------------------------------------
# TISR PyTorch Dataset Class
# ------------------------------------------------------------------
class TISRDataset(BaseDataset):
    """PyTorch Dataset for Total Incident Solar Radiation (TISR) at the top of the atmosphere (TOA).

    Computations in this class are designed to mimic ERA5's ``toa_incident_solar_radiation`` (``tisr``)
    variable (units: J/m2, see https://codes.ecmwf.int/grib/param-db/212) by interpolating ERA5-compatible
    Total Solar Irradiance (TSI) values to the requested timestamps, then integrating the product of the TSI,
    a solar scaling factor, and the cosine of the solar zenith angle over the specified period. Defaults to
    ERA5-compatible settings: an integration period of one hour with 360 integration bins.

    While the default configuration targets ERA5 compatibility, both the integration period and bin
    count are configurable for other use cases. Input timestamps must fall within the TSI data
    range (1951-2034).

    Note that the TISR dataset is typically used as a dynamic forcing/input variable rather than a
    target, so the ``return_target`` parameter is set to False by default. TISR dataset is not loading
    any data from local or remote files, but rather performing the computation on-the-fly (no need to
    specify loading mode like most other datasets). Because computation happens inside ``__getitem__``,
    the dataset emits CPU tensors by default. When used with a multi-worker ``DataLoader``
    (``num_workers > 0``), keep ``device="cpu"`` (the default) and let the training loop move each
    collated batch to the GPU; constructing CUDA tensors in worker subprocesses is unsupported by
    PyTorch. The ``device`` config key is provided mainly for single-process (``num_workers=0``) use.

    Exactly one grid source must be configured: either ``latlon_grid_path`` (read from a NetCDF
    file) or both ``lat_spec`` and ``lon_spec`` (build a rectangular grid in-memory, no file read).
    Each spec is a ``[start, end, num_points]`` list with both endpoints inclusive.

    See module docstring for full description of output format and file naming.

    Example YAML configuration (grid read from file)::

        data:
            source:
                Example_TISR:  # User-provided name (arbitrary key)
                    dataset_type: "tisr"
                    variables:
                        prognostic: null
                        diagnostic: null
                        dynamic_forcing:
                            var_2d: ['tisr']  # only accept 'tisr'
                    num_integration_steps: 2160  # 360 steps per hour → 6h integration with 1h accumulation windows
                    latlon_grid_path: "/glade/derecho/scratch/cbecker/test_CREDIT_data/era5_local_testing_data_onedeg_2021.nc"

            start_datetime: "2021-06-01"
            end_datetime: "2021-06-04"
            timestep: "6h"
            forecast_len: 0

    Example YAML configuration (grid built in-memory from specs)::

        data:
            source:
                Example_TISR:
                    dataset_type: "tisr"
                    variables:
                        prognostic: null
                        diagnostic: null
                        dynamic_forcing:
                            var_2d: ['tisr']
                    num_integration_steps: 2160
                    lat_spec: [90, -90, 721]      # [start, end, num_points], endpoints inclusive
                    lon_spec: [0, 359.75, 1440]   # 0.25° grid; excludes the 360° wrap

            start_datetime: "2021-06-01"
            end_datetime: "2021-06-04"
            timestep: "6h"
            forecast_len: 0
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """Initialize TISRDataset with config parsing.

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return the target variable. Defaults to False since "tisr"
                is typically used as a dynamic forcing/input variable rather than a target.
        """
        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "tisr", (
            f"Expected dataset_type 'tisr' in config for TISRDataset, got '{self.curr_source_cfg['dataset_type']}'"
        )

        # Set TISR-specific attributes
        self.dataset_type = "tisr"
        self.static_metadata: dict[str, Any] = {"datetime_fmt": "unix_ns"}
        self.num_integration_steps: int = self.curr_source_cfg.get("num_integration_steps", 360)

        # Initialize the field registration based on the provided config
        self.init_register_all_fields()

        # Device for on-the-fly TISR computation. Defaults to CPU and should
        # almost always stay there: this dataset produces tensors inside
        # __getitem__, which run in DataLoader worker subprocesses when
        # num_workers > 0. CUDA tensors cannot be created in forked workers
        # ("Cannot re-initialize CUDA in forked subprocess") and don't survive
        # the default collate + IPC, so the trainer should move collated CPU
        # batches to the GPU instead. Only set device="cuda" here if you load
        # this dataset single-process (num_workers=0).
        self.device: torch.device = torch.device(self.curr_source_cfg.get("device", "cpu"))
        if self.device.type != "cpu":
            logger.warning(
                "TISRDataset configured with device=%s. Computing tensors on a "
                "non-CPU device inside the dataset is incompatible with "
                "multi-worker DataLoaders (num_workers > 0). Prefer device='cpu' "
                "and move batches to the GPU in the training loop.",
                self.device,
            )

        # Grid source from config — exactly one of: 'latlon_grid_path' (file)
        # or both 'lat_spec'/'lon_spec' (in-memory). All keys optional individually.
        path = self.curr_source_cfg.get("latlon_grid_path")
        lat_spec = self.curr_source_cfg.get("lat_spec")
        lon_spec = self.curr_source_cfg.get("lon_spec")

        # Spec axes must be supplied as a pair; catch a one-sided pair here with a
        # clear message before it reaches _get_latlon_grid.
        if (lat_spec is None) != (lon_spec is None):
            raise ValueError(
                "TISR config: 'lat_spec' and 'lon_spec' must be supplied together. "
                f"Got lat_spec={lat_spec!r}, lon_spec={lon_spec!r}."
            )

        # True in the two bad cases (neither source, or both); raise with the actual keys.
        has_path = path is not None
        has_spec = lat_spec is not None  # lon_spec is guaranteed to match by the check above
        if has_path == has_spec:
            raise ValueError(
                "TISR config must specify exactly one grid source: either "
                "'latlon_grid_path' (read from file) or both 'lat_spec' and "
                "'lon_spec' (build in-memory). "
                f"Got latlon_grid_path={path!r}, lat_spec={lat_spec!r}, lon_spec={lon_spec!r}."
            )

        # Build the static grid once and cache it, keeping I/O off the per-sample path.
        self.latlon_grid_path: str | None = path
        self.latitude, self.longitude = _get_latlon_grid(
            path=path,
            lat_spec=lat_spec,
            lon_spec=lon_spec,
            device=self.device,
        )

    def _get_file_source(self, field_config: dict[str, Any]) -> None:
        """Returns None since TISR dataset is not loading any data from local or remote files."""
        return None

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict,
    ) -> None:
        """Load the TISR 2-D variable for a field type (dynamic_forcing only) at time ``t`` into ``sample``.

        Computes the top-of-atmosphere solar radiation integrated over ``dt``
        ending at ``t``, and stores it as a ``torch.Tensor`` of shape
        ``(1, 1, ny, nx)`` under the key ``"{source_name}/{field_type}/2d/tisr"``
        in ``sample``. Does nothing if the field type has no registered variables.

        Args:
            field_type: ``"dynamic_forcing"`` only, and others set to null.
            t: Timestamp for which to load data.
            sample: Output dictionary that is updated in-place.
        """
        # Check if the var_dict exists for the field type
        vd = self.var_dict.get(field_type)
        if not vd:
            logger.debug("No var_dict entry for field_type '%s', skipping.", field_type)
            return

        # Check if the field type has any 2-D variables and if it is "tisr"
        vars_2D = vd.get("vars_2D", [])
        if not vars_2D:
            logger.debug("No vars_2D in var_dict for field_type '%s', skipping.", field_type)
            return
        if vars_2D != ["tisr"]:
            raise ValueError(f"TISRDataset only supports vars_2D=['tisr'], got {vars_2D}")

        # Compute the top-of-atmosphere solar radiation, expand to be (1, 1, lat, lon),
        # and store it in the sample dictionary
        tisr = _compute_tisr(t, self.dt, self.num_integration_steps, self.latitude, self.longitude)
        key = self._get_field_name(field_type, "2d", "tisr")
        sample[key] = tisr.unsqueeze(0).unsqueeze(0)
