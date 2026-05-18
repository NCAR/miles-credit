"""_summary_"""

from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
import xarray as xr

from credit.datasets.base_dataset import BaseDataset


_TORCH_DTYPE = torch.float32

_DEFAULT_INTEGRATION_PERIOD = pd.Timedelta(hours=1)

_DEFAULT_NUM_INTEGRATION_BINS = 360


class TISRDataset(BaseDataset):
    """PyTorch Dataset for Total Incident Solar Radiation (TISR) at the top of the atmosphere (TOA).

    Methods in this class are designed to mimic ERA5's ``toa_incident_solar_radiation`` (``tisr``)
    variable (units: J/m2, see https://codes.ecmwf.int/grib/param-db/212) by interpolating ERA5-compatible
    Total Solar Irradiance (TSI) values to the requested timestamps, then integrating the product of the TSI,
    a solar scaling factor, and the cosine of the solar zenith angle over the specified period. Defaults to
    ERA5-compatible settings: an integration period of one hour with 360 integration bins.

    While the default configuration targets ERA5 compatibility, both the integration period and bin
    count are configurable for other use cases. Input timestamps must fall within the TSI data
    range (1951-2034).
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return the target variable. Defaults to False since tisr
                is typically used as a dynamic forcing/input variable rather than a target.

        Raises:


        """
        # Super constructor to inherit common config parsing and timestamp generation logic
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "tisr", (
            f"Expected dataset_type 'tisr' in config for TISRDataset, got '{self.curr_source_cfg['dataset_type']}'"
        )

        # Set TISR-specific attributes
        self.dataset_type = "tisr"
        self.static_metadata: dict[str, Any] = {"datetime_fmt": "unix_ns"}
        self.mode = "remote"  # TISR is computed on-the-fly, so we use "remote" mode to indicate no local files are read

        # Initialize the field registration based on the provided config and populate
        #   dictionary of variables and file paths for each field type
        self.init_register_all_fields()

        # Load the latitude-longitude grid file path from the config; this is needed to compute
        #   the cosine of the solar zenith angle for each grid point
        self.latlon_grid_path: str = self.curr_source_cfg.get("latlon_grid_path")

    def _get_file_source(
        self,
    ) -> None:
        """Return the file source for a field. Override in subclasses for different modes/backends.

        Args:
            field_config (dict[str, Any]): Validated field-type config dict.

        Raises:
            ValueError: If ``self.mode`` is not a recognised mode.

        Returns:
            list[tuple[pd.Timestamp, pd.Timestamp, str]] | bool | None: Depending on the mode and field type,
                this method may return a list of (start_time, end_time, file_path) tuples produced by _map_files,
                a boolean indicating the presence of the field (e.g., for remote data), or None if the field is disabled.
                The expected return type should be consistent within a dataset class.
        """
        return None

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict,
    ) -> None:
        """Load all 2-D variables for a field type at time ``t`` into ``sample``.

        Dispatches to ``_load_local_var`` or ``_load_remote_var`` depending on
        ``mode``, then stores each variable as a ``torch.Tensor`` of shape
        ``(1, 1, ny, nx)`` under the key
        ``"{source_name}/{field_type}/2d/{vname}"`` in ``sample``. Does nothing if
        the field type has no registered variables.

        Args:
            field_type: One of ``"prognostic"``, ``"diagnostic"``, or
                ``"dynamic_forcing"``.
            t: Timestamp for which to load data.
            sample: Output dictionary that is updated in-place.
        """
        vd = self.var_dict.get(field_type)
        if not vd:
            return

        vnames = vd.get("vars_2D", [])
        if not vnames:
            return

        #
        latitude, longitude = _load_latlon_grid(self.latlon_grid_path)

        #
        times, tsi_values = _era5_tsi_data()
        tsi_at_t = _get_tsi(t, times, tsi_values)

        #
        j2000_days = _get_j2000_days(t)

        orbital_params = _get_orbital_parameters(j2000_days)
        rotational_phase = orbital_params["rotational_phase"]
        eq_of_time_seconds = orbital_params["eq_of_time_seconds"]
        solar_distance_au = orbital_params["solar_distance_au"]
        cos_declination = orbital_params["cos_declination"]
        sin_declination = orbital_params["sin_declination"]

        solar_factor = 1.0 / solar_distance_au**2
        solar_time = _get_solar_time(rotational_phase, eq_of_time_seconds)

        #
        hour_angle = _get_hour_angle(
            solar_time=solar_time,
            latitude=latitude,
            longitude=longitude,
        )

        cos_zenith = _get_cosine_zenith_angle(
            cos_declination=cos_declination,
            sin_declination=sin_declination,
            latitude=latitude,
            longitude=longitude,
            hour_angle=hour_angle,
        )

        instantaneous_radiation = _get_instantaneous_toa_solar_radiation(
            tsi=tsi_at_t, solar_factor=solar_factor, cos_zenith=cos_zenith
        )

        arrays = instantaneous_radiation

        for vname, arr in arrays.items():
            tensor = torch.tensor(arr, dtype=_TORCH_DTYPE).unsqueeze(0).unsqueeze(0)
            key = self._get_field_name(field_type, "2d", vname)
            sample[key] = tensor


# ------------------------------------------------------------------
#
# ------------------------------------------------------------------
def _era5_tsi_data() -> tuple[torch.Tensor, torch.Tensor]:
    """ERA5 compatible Total Solar Irradiance (TSI) from
    [Graphcast](https://github.com/google-deepmind/graphcast/blob/main/graphcast/solar_radiation.py).

    ECMWF provided the data used for ERA5, which was hardcoded in the IFS (cycle 41r2, 2016). 2009 onwards
    are repeated values from 1996-2008 (last completed 13-year cycle available when the code was written).
    The values were scaled down to agree better with more recent observations of the sun.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the time and TSI values as PyTorch tensors.
    """
    times = torch.arange(1951.5, 2035.5, 1.0, dtype=_TORCH_DTYPE)
    tsi_values = 0.9965 * torch.tensor(
        [
            # 1951-1995 (non-repeating sequence)
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
            # 1996-2008 (13 year cycle, repeated below)
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
            # 2009-2021
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
            # 2022-2034
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
        dtype=torch.float64,
    )
    return times, tsi_values


def _get_tsi(
    timestamps: Sequence[str | pd.Timestamp],
    tsi_times: torch.Tensor,
    tsi_values: torch.Tensor,
) -> torch.Tensor:
    """Interpolate Total Solar Irradiance (TSI) at the given timestamps.

    Converts each timestamp to a fractional year and performs linear
    interpolation against the provided TSI time series.

    Args:
        timestamps: Sequence of timestamps (strings or pd.Timestamps).
        tsi_times:  1-D tensor of fractional years (e.g. 2003.45), sorted ascending.
        tsi_values: 1-D tensor of TSI values corresponding to tsi_times.

    Returns:
        1-D tensor of interpolated TSI values, one per input timestamp.

    Raises:
        ValueError: If any timestamp falls outside [tsi_times[0], tsi_times[-1]].
    """
    ts = pd.DatetimeIndex([timestamps] if isinstance(timestamps, pd.Timestamp) else timestamps)
    ts_date = pd.DatetimeIndex(ts.date)

    # Check that all timestamps are within the range of the TSI data; first half of 1951 and second
    # half of 2034 will pass and be extrapolated, but anything outside that will raise an error.
    t_min, t_max = tsi_times[0].item(), tsi_times[-1].item()
    out_mask = (ts.year < int(t_min)) | (ts.year > int(t_max))

    if out_mask.any():
        bad = ts[out_mask].tolist()
        raise ValueError(
            f"{len(bad)} timestamp(s) fall outside the TSI data range "
            f"[{t_min:.4f}, {t_max:.4f}]: {bad[:5]}" + (" ..." if len(bad) > 5 else "")
        )

    # Compute the fractional year for each timestamp
    day_frac = (ts - ts_date) / pd.Timedelta(days=1)
    year_len = 365 + ts.is_leap_year
    year_frac = (ts.dayofyear - 1 + day_frac) / year_len
    frac_year = torch.tensor((ts.year + year_frac).to_numpy(), dtype=tsi_times.dtype)

    # Interpolate TSI values at the given fractional years using linear interpolation
    idx = torch.searchsorted(
        tsi_times.contiguous(), frac_year.contiguous()
    )  # searchsorted requires tensors are contiguous in memory
    idx = idx.clamp(1, len(tsi_times) - 1)  # guard against out-of-bounds indices
    lo, hi = idx - 1, idx
    t = (frac_year - tsi_times[lo]) / (tsi_times[hi] - tsi_times[lo])
    return tsi_values[lo] + t * (tsi_values[hi] - tsi_values[lo])


# ------------------------------------------------------------------
#
# ------------------------------------------------------------------
def _load_latlon_grid(path: str) -> torch.Tensor:
    """Read a lat/lon grid from a NetCDF file.

    If ``latitude`` and ``longitude`` are stored as 2-D *variables* (curvilinear
    grid), read them directly. If they are provided as 1-D *coordinates*
    (rectangular grid), build a 2-D meshgrid. Common aliases (``lat``/``lon``,
    ``XLAT``/``XLONG``, ``nav_lat``/``nav_lon``) are tried in order.

    Args:
        path (str): Path to a NetCDF file containing latitude/longitude information.

    Raises:
        ValueError: If the file cannot be opened.
        ValueError: If no recognised latitude or longitude field is found.
        ValueError: If lat/lon arrays have unexpected dimensionality (not 1-D or 2-D).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of latitude and longitude tensors,
            both in degrees and shape ``(ny, nx)``.
    """
    # Alias names to search in order of preference
    _LAT_NAMES = ("latitude", "lat", "XLAT", "nav_lat")
    _LON_NAMES = ("longitude", "lon", "XLONG", "nav_lon")

    # Open the NetCDF file using xarray; this will raise an error if the file cannot be read
    try:
        ds = xr.open_dataset(path)
    except Exception as exc:
        raise ValueError(f"Could not open NetCDF file at '{path}': {exc}") from exc

    # Gather all field names from both coordinates and data variables to search for lat/lon keys
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
    return lat_tensor, lon_tensor


# ------------------------------------------------------------------
#
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

    repsm = 0.409093
    sin_declination = torch.sin(torch.tensor(repsm)) * torch.sin(rllls)
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
#
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
    latitude: torch.Tensor,
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
    longitude: torch.Tensor,
    hour_angle: torch.Tensor,
) -> torch.Tensor:
    """Compute the cosine of the solar zenith angle.

    Args:
        declination (torch.Tensor): Solar declination angle in degrees.
        latitude (torch.Tensor): Latitude in degrees.
        longitude (torch.Tensor): Longitude in degrees.
        hour_angle (torch.Tensor): Hour angle in degrees.

    Returns:
        torch.Tensor: Cosine of the solar zenith angle, values in [0, 1].

    References:
        - https://en.wikipedia.org/wiki/Solar_zenith_angle#Formula
    """
    # cos(zenith) = cos(lat) * cos(declination) * cos(hour_angle) + sin(lat) * sin(declination)
    cos_zenith = (
        torch.cos(torch.deg2rad(latitude)) * cos_declination * torch.cos(torch.deg2rad(hour_angle))
        + torch.sin(torch.deg2rad(latitude)) * sin_declination
    )

    # Validate that cos_zenith values are within the physical range [0, 1]
    if not torch.all((cos_zenith >= 0) & (cos_zenith <= 1)):
        mask = (cos_zenith < 0) | (cos_zenith > 1)
        invalid = cos_zenith[mask]
        raise ValueError(
            f"cos_zenith must be in [0, 1], but found {mask.sum().item()} invalid values "
            f"out of {cos_zenith.numel()} total. "
            f"Range of invalid values: [{invalid.min().item():.4f}, {invalid.max().item():.4f}]"
        )
    return cos_zenith.clamp(0.0, 1.0)


# ------------------------------------------------------------------
#
# ------------------------------------------------------------------
def _get_instantaneous_toa_solar_radiation(
    tsi: torch.Tensor, solar_factor: float, cos_zenith: torch.Tensor
) -> torch.Tensor:
    """_summary_

    Args:
        tsi (torch.Tensor): _description_
        solar_factor (float): _description_
        cos_zenith (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    instantaneous_toa_solar_radiation = tsi * solar_factor * cos_zenith
    return instantaneous_toa_solar_radiation


def _get_integrated_toa_solar_radiation(
    instantaneous_radiation: torch.Tensor,
    integration_period: pd.Timedelta | pd.Timedelta(hours=1),
    num_integration_steps: int | 360,
) -> torch.Tensor:
    """_summary_

    Args:
        instantaneous_radiation (torch.Tensor): _description_
        integration_period (pd.Timedelta): _description_
        num_integration_steps (int | 360): _description_

    Returns:
        torch.Tensor: _description_
    """
    # Validate that num_integration_steps is a positive integer
    if not isinstance(num_integration_steps, int) or num_integration_steps <= 0:
        raise ValueError(f"num_integration_steps must be a positive integer, but got {num_integration_steps}")

    # Compute the time step for integration
    dt = integration_period / num_integration_steps / pd.Timedelta(seconds=1)  # convert to seconds for integration

    # Use the trapezoidal rule to integrate the instantaneous radiation over the period
    # Assuming instantaneous_radiation is sampled at regular intervals of dt
    integrated_radiation = torch.trapz(
        instantaneous_radiation, dx=dt, dim=0
    )  # integrate along the time dimension (dim=0)

    return integrated_radiation
