"""
tisr.py
-------------------------------------------------------
TISRDataset: PyTorch Dataset for Total Incident Solar Radiation (TISR) at the top of the atmosphere (TOA).

Sample structure returned by __getitem__:

    {
        "input":    {<user_provided_name>: {"<user_provided_name>/dynamic_forcing/2d/tisr": tensor}},
        "target":   {<user_provided_name>: {}},  # empty since dynamic forcing is only input
        "metadata": {<user_provided_name>: {"input_datetime": int, "target_datetime": int}},
    }

TISR only has a single variable and is 2D. Tensor shape (no batch dimension):
    (1, 1, lat, lon)   — singleton level dim, consistent with CREDIT Gen2 2D convention

After DataLoader collation the batch dimension is prepended:
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

_TORCH_DTYPE = torch.float64

# ------------------------------------------------------------------
# Total Solar Irradiance (TSI) Data and Interpolation
# ------------------------------------------------------------------
def _era5_tsi_data() -> tuple[torch.Tensor, torch.Tensor]:
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
    times = torch.arange(1951.5, 2035.5, 1.0, dtype=_TORCH_DTYPE)
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
    frac_year = torch.tensor((ts.year + year_frac).to_numpy(), dtype=tsi_times.dtype)

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
# Lat/Lon Grid Loading
# ------------------------------------------------------------------
def _load_latlon_grid(path: str) -> torch.Tensor:
    """Read a lat/lon grid from a NetCDF file and return it as a (1, ny, nx) tensor pair.
 
    Handles two common grid representations:
 
    * **Curvilinear grids** – latitude and longitude are stored as 2-D
      variables of shape ``(ny, nx)`` and are read directly.
    * **Rectangular grids** – latitude and longitude are stored as 1-D
      coordinate arrays of lengths ``ny`` and ``nx`` respectively; a 2-D
      meshgrid is constructed from them.
 
    Common field name aliases (``latitude``/``lat``/``XLAT``/``nav_lat``
    and ``longitude``/``lon``/``XLONG``/``nav_lon``) are tried in order of
    preference so that files from different models/tools are accepted
    without preprocessing.
 
    Args:
        path (str): Path to a NetCDF file containing latitude/longitude
            information.
 
    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - **lat_tensor** – Latitude grid in degrees, shape ``(1, ny, nx)``.
            - **lon_tensor** – Longitude grid in degrees, shape ``(1, ny, nx)``.
 
    Raises:
        ValueError: If the file cannot be opened by xarray.
        ValueError: If no recognised latitude or longitude field is found.
        ValueError: If the lat/lon arrays are neither 1-D nor 2-D after
            squeezing size-1 leading dimensions.
    """
    # Preferred alias order; first match wins for each coordinate
    _LAT_NAMES = ("latitude", "lat", "XLAT", "nav_lat")
    _LON_NAMES = ("longitude", "lon", "XLONG", "nav_lon")

    # Open the NetCDF file; propagate any I/O error as a descriptive ValueError
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
        lat_tensor = torch.tensor(lat.values, dtype=_TORCH_DTYPE)
        lon_tensor = torch.tensor(lon.values, dtype=_TORCH_DTYPE)
    elif lat.ndim == 1 and lon.ndim == 1:
        # rectangular grid case: lat and lon are 1-D arrays; create a 2-D meshgrid
        lat_1d = torch.tensor(lat.values, dtype=_TORCH_DTYPE)
        lon_1d = torch.tensor(lon.values, dtype=_TORCH_DTYPE)
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
) -> torch.Tensor:
    """

    Args:
        timestamp (pd.Timestamp | pd.DatetimeIndex): _description_

    Returns:
        torch.Tensor: _description_
    """
    _J2000_EPOCH = 2451545.0
    # Convert the input timestamp(s) to a DatetimeIndex if it's a single Timestamp.
    dti = pd.DatetimeIndex([timestamp] if isinstance(timestamp, pd.Timestamp) else timestamp)

    # Convert to Julian date and then to days since J2000 epoch, returning as a PyTorch tensor
    return torch.tensor(dti.to_julian_date().to_numpy() - _J2000_EPOCH, dtype=_TORCH_DTYPE)


def _get_orbital_parameters(
    j2000_days: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """_summary_

    Args:
        j2000_days (torch.Tensor): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        dict[str, torch.Tensor]: _description_
    """
    _JULIAN_YEAR_LENGTH_IN_DAYS = 365.25

    theta = j2000_days / _JULIAN_YEAR_LENGTH_IN_DAYS
    rotational_phase = j2000_days % 1.0

    rel = 1.7535 + 6.283076 * theta
    rem = 6.240041 + 6.283020 * theta
    rlls = 4.8951 + 6.283076 * theta

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

    rllls = torch.stack([one, theta, sin_rel, cos_rel, sin_two_rel, cos_two_rel], dim=-1) @ torch.tensor(
        [4.8952, 6.283320, -0.0075, -0.0326, -0.0003, 0.0002], dtype=j2000_days.dtype, device=j2000_days.device
    )

    repsm = torch.tensor(0.409093, dtype=j2000_days.dtype, device=j2000_days.device)
    sin_declination = torch.sin(repsm) * torch.sin(rllls)
    cos_declination = torch.sqrt(1.0 - sin_declination**2)

    eq_of_time_seconds = torch.stack(
        [sin_two_rlls, sin_rem, sin_rem * cos_two_rlls, sin_four_rlls, sin_two_rem],
        dim=-1,
    ) @ torch.tensor([591.8, -459.4, 39.5, -12.7, -4.8], dtype=j2000_days.dtype, device=j2000_days.device)

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
    """_summary_

    Returns:
        _type_: _description_
    """
    _SECONDS_PER_DAY = 60 * 60 * 24
    solar_time = rotational_phase + eq_of_time_seconds / _SECONDS_PER_DAY
    return solar_time


def _get_hour_angle(
    solar_time: torch.Tensor,
    longitude: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Returns:
        torch.Tensor: _description_

    References:
        - https://en.wikipedia.org/wiki/Hour_angle#Solar_hour_angle
    """
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

    return torch.maximum(cos_zenith, torch.tensor([0.0], dtype=cos_zenith.dtype))


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
    #return tsi * solar_factor * cos_zenith
    return tsi * solar_factor * torch.maximum(cos_zenith, torch.zeros_like(cos_zenith))

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
        raise ValueError(
            f"num_integration_steps must be a positive integer, but got {num_integration_steps}"
        )

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
    latlon_grid_path: str,
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
        latlon_grid_path (str): Path to the file containing the latitude/longitude
            grid over which TISR is computed.

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

    # Load the spatial grid; latitude and longitude are broadcast-compatible
    # tensors used for zenith angle and solar time calculations below.
    latitude, longitude = _load_latlon_grid(latlon_grid_path)

    # Retrieve the total solar irradiance (TSI) time series and interpolate it
    # onto ts. Unsqueeze to add singleton spatial dims for broadcasting: (time, 1, 1).
    times, tsi_values = _era5_tsi_data()
    tsi = _get_tsi(ts, times, tsi_values).unsqueeze(-1).unsqueeze(-1)

    # Compute orbital parameters (declination, equation of time, Earth-Sun distance,
    # etc.) for each timestep. J2000 days are unsqueezed to (time, 1, 1) so the
    # result broadcasts cleanly against the spatial grid.
    orbital = _get_orbital_parameters(_get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1))
    
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

    Note that the TISR dataset is typically used as a dynamic forcing/input variable rather than a target,
    so the ``return_target`` parameter is set to False by default. TISR dataset is not loading any data
    from local or remote files, but rather performing the computation on-the-fly (no need to specify
    loading mode like most other datasets).

    See module docstring for full description of output format and file naming.

    Example YAML configuration (local mode):

        data:
            source:
                Example_TISR:  # User-provided name (arbitrary key)
                    dataset_type: "tisr"
                    variables:
                        prognostic: null
                        diagnostic: null
                        dynamic_forcing:
                            var_2d: ['tisr']  # only accept 'tisr'
                    num_integration_steps: 360
                    latlon_grid_path: "/glade/derecho/scratch/cbecker/test_CREDIT_data/era5_local_testing_data_onedeg_2021.nc"

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

        # Load the latitude-longitude grid file path from the config; this is needed to compute 
        # the cosine of the solar zenith angle for each grid point
        self.latlon_grid_path: str = self.curr_source_cfg.get("latlon_grid_path")

    def _get_file_source(self, 
                         field_config: dict[str, Any]) -> None:
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
            return

        # Check if the field type has any 2-D variables and if it is "tisr"
        vars_2D = vd.get("vars_2D", [])
        if not vars_2D:
            return
        if vars_2D != ["tisr"]:
            raise ValueError(
                f"TISRDataset only supports vars_2D=['tisr'], got {vars_2D}"
            )

        # Compute the top-of-atmosphere solar radiation, expand to be (1, 1, lat, lon), 
        # and store it in the sample dictionary
        tisr = _compute_tisr(t, self.dt, self.num_integration_steps, self.latlon_grid_path)
        key = self._get_field_name(field_type, "2d", "tisr")
        sample[key] = tisr.unsqueeze(0).unsqueeze(0)
