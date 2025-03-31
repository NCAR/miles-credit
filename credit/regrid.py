'''
This scripts contains functions that performs nearest, bilinear, and conservative interpolation
on xarray.Datasets. The original version of this script is available at WeatherBench2.

Note: only rectalinear grids (one dimensional lat/lon coordinates) are supported.

Reference
 - WeatherBench2 regridding: 
     https://github.com/google-research/weatherbench2/blob/main/weatherbench2/regridding.py

Example usage
# ================================================================================== #
import credit.regrid as regrid

# --------------------- #
# prepare grids

# target grid
lon_1deg = np.arange(0, 360, 1)
lat_1deg = np.arange(-90, 91, 1)
target_grid = regrid.Grid.from_degrees(lon_1deg, lat_1deg)

# input grid (flip 90 --> -90 to -90 --> 90)
lon_025deg = ds_static['longitude'].values
lat_025deg = ds_static['latitude'].values[::-1]
source_grid = regrid.Grid.from_degrees(lon_025deg, lat_025deg)

# --------------------- #
# define regridder
regridder = regrid.ConservativeRegridder(source=source_grid, target=target_grid)

# --------------------- #
# clear old chunking and interpolate data
ds_static = ds_static.chunk({'longitude': -1, 'latitude': -1})
ds_static_1deg = regridder.regrid_dataset(ds_static)

# --------------------- #
# ... some xarray operations to preserve the order of dims ... #

# assign coordinates
lon_1deg = np.arange(0, 360, 1)
lat_1deg = np.arange(-90, 91, 1)
ds_static_1deg = ds_static_1deg.assign_coords({
    'latitude': lat_1deg,
    'longitude': lon_1deg
})

# flip latitude from -90 --> 90 to 90 --> -90
ds_static_1deg = ds_static_1deg.isel(latitude=slice(None, None, -1))

     
'''
from __future__ import annotations

import xarray
import numpy as np
from sklearn import neighbors

import functools
import dataclasses
from typing import Union

Array = Union[np.ndarray]

@dataclasses.dataclass(frozen=True)
class Grid:
    """Representation of a rectilinear grid."""

    lon: np.ndarray
    lat: np.ndarray

    @classmethod
    def from_degrees(cls, lon: np.ndarray, lat: np.ndarray) -> Grid:
        return cls(np.deg2rad(lon), np.deg2rad(lat))

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.lon), len(self.lat))

    def _to_tuple(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return tuple(self.lon.tolist()), tuple(self.lat.tolist())

    def __eq__(self, other):  # needed for hashability
        return isinstance(other, Grid) and self._to_tuple() == other._to_tuple()

    def __hash__(self):
        return hash(self._to_tuple())


@dataclasses.dataclass(frozen=True)
class Regridder:
    """Base class for regridding."""

    source: Grid
    target: Grid

    def regrid_array(self, field: Array) -> np.ndarray:
        """Regrid an array with dimensions (..., lon, lat) from source to target."""
        raise NotImplementedError

    def regrid_dataset(self, dataset: xarray.Dataset) -> xarray.Dataset:
        """Regrid an xarray.Dataset from source to target."""
        if not (dataset['latitude'].diff('latitude') > 0).all():
            dataset = dataset.isel(latitude=slice(None, None, -1))  # Reverse
        assert (dataset['latitude'].diff('latitude') > 0).all()
        
        dataset = xarray.apply_ufunc(
            self.regrid_array,
            dataset,
            input_core_dims=[['longitude', 'latitude']],
            output_core_dims=[['longitude', 'latitude']],
            exclude_dims={'longitude', 'latitude'},
            vectorize=True,
            dask='allowed',  # Allow Dask arrays to be passed to the function
            output_dtypes=[dataset[list(dataset.data_vars)[0]].dtype],
        )
        return dataset


def nearest_neighbor_indices(
    source_grid: Grid, target_grid: Grid
) -> np.ndarray:
    """Returns Haversine nearest neighbor indices from source_grid to target_grid."""
    # Construct a BallTree to find nearest neighbors on the sphere
    source_mesh = np.meshgrid(source_grid.lat, source_grid.lon, indexing='ij')
    target_mesh = np.meshgrid(target_grid.lat, target_grid.lon, indexing='ij')
    index_coords = np.stack([x.ravel() for x in source_mesh], axis=-1)
    query_coords = np.stack([x.ravel() for x in target_mesh], axis=-1)
    tree = neighbors.BallTree(index_coords, metric='haversine')
    indices = tree.query(query_coords, return_distance=False).squeeze(axis=-1)
    return indices


class NearestRegridder(Regridder):
    """Regrid with nearest neighbor interpolation."""

    @functools.cached_property
    def indices(self):
        """The interpolation indices associated with source_grid."""
        return nearest_neighbor_indices(self.source, self.target)

    def _nearest_neighbor_2d(self, array: Array) -> np.ndarray:
        """2D nearest neighbor interpolation using BallTree."""
        if array.shape != self.source.shape:
            raise ValueError(f'Expected array.shape={array.shape} to match source.shape={self.source.shape}')
        array_flat = array.ravel()
        interpolated = array_flat[self.indices]
        return interpolated.reshape(self.target.shape)

    def regrid_array(self, field: Array) -> np.ndarray:
        interp = np.vectorize(self._nearest_neighbor_2d, signature='(a,b)->(c,d)')
        return interp(field)


class BilinearRegridder(Regridder):
    """Regrid with bilinear interpolation."""

    def regrid_array(self, field: Array) -> np.ndarray:
        lat_source = self.source.lat
        lat_target = self.target.lat
        lon_source = self.source.lon
        lon_target = self.target.lon

        # Interpolate over latitude
        lat_interp = np.array([
            np.interp(lat_target, lat_source, f) for f in field.transpose(1, 0, *range(2, field.ndim))
        ]).transpose(1, 0, *range(2, field.ndim))

        # Interpolate over longitude
        lon_interp = np.array([
            np.interp(lon_target, lon_source, f) for f in lat_interp.transpose(0, *range(2, field.ndim), 1)
        ]).transpose(0, *range(2, field.ndim), 1)

        return lon_interp


def _assert_increasing(x: np.ndarray) -> None:
    if not (np.diff(x) > 0).all():
        raise ValueError(f'Array is not increasing: {x}')


def _latitude_cell_bounds(x: Array) -> np.ndarray:
    pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
    return np.concatenate((-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2))


def _latitude_overlap(
    source_points: Array,
    target_points: Array,
) -> np.ndarray:
    """Calculate the area overlap as a function of latitude."""
    source_bounds = _latitude_cell_bounds(source_points)
    target_bounds = _latitude_cell_bounds(target_points)
    upper = np.minimum(
        target_bounds[1:, np.newaxis], source_bounds[np.newaxis, 1:]
    )
    lower = np.maximum(
        target_bounds[:-1, np.newaxis], source_bounds[np.newaxis, :-1]
    )
    # Normalized cell area: integral from lower to upper of cos(latitude)
    overlap = (upper > lower) * (np.sin(upper) - np.sin(lower))
    return overlap


def _conservative_latitude_weights(
    source_points: Array, target_points: Array
) -> np.ndarray:
    """Create a weight matrix for conservative regridding along latitude.

    Args:
        source_points: 1D latitude coordinates in radians for centers of source cells.
        target_points: 1D latitude coordinates in radians for centers of target cells.

    Returns:
        NumPy array with shape (target_size, source_size). Rows sum to 1.
    """
    _assert_increasing(source_points)
    _assert_increasing(target_points)
    weights = _latitude_overlap(source_points, target_points)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


def _align_phase_with(x, target, period):
    """Align the phase of a periodic number to match another."""
    shift_down = x > target + period / 2
    shift_up = x < target - period / 2
    return x + period * shift_up - period * shift_down


def _periodic_upper_bounds(x, period):
    x_plus = _align_phase_with(np.roll(x, -1), x, period)
    return (x + x_plus) / 2


def _periodic_lower_bounds(x, period):
    x_minus = _align_phase_with(np.roll(x, +1), x, period)
    return (x_minus + x) / 2


def _periodic_overlap(x0, x1, y0, y1, period):
    """Calculate the overlap between two intervals considering periodicity."""
    y0 = _align_phase_with(y0, x0, period)
    y1 = _align_phase_with(y1, x0, period)
    upper = np.minimum(x1, y1)
    lower = np.maximum(x0, y0)
    return np.maximum(upper - lower, 0)


def _longitude_overlap(
    first_points: Array,
    second_points: Array,
    period: float = 2 * np.pi,
) -> np.ndarray:
    """Calculate the area overlap as a function of longitude."""
    first_points = first_points % period
    first_upper = _periodic_upper_bounds(first_points, period)
    first_lower = _periodic_lower_bounds(first_points, period)

    second_points = second_points % period
    second_upper = _periodic_upper_bounds(second_points, period)
    second_lower = _periodic_lower_bounds(second_points, period)

    x0 = first_lower[:, np.newaxis]
    x1 = first_upper[:, np.newaxis]
    y0 = second_lower[np.newaxis, :]
    y1 = second_upper[np.newaxis, :]

    overlap_func = np.vectorize(_periodic_overlap, excluded=['period'])
    overlap = overlap_func(x0, x1, y0, y1, period=period)
    return overlap


def _conservative_longitude_weights(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Create a weight matrix for conservative regridding along longitude.

    Args:
        source_points: 1D longitude coordinates in radians for centers of source cells.
        target_points: 1D longitude coordinates in radians for centers of target cells.

    Returns:
        NumPy array with shape (target_size, source_size). Rows sum to 1.
    """
    _assert_increasing(source_points)
    _assert_increasing(target_points)
    weights = _longitude_overlap(target_points, source_points)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


class ConservativeRegridder(Regridder):
    """Regrid with linear conservative regridding."""

    def _mean(self, field: Array) -> np.ndarray:
        """Computes cell-averages of field on the target grid."""
        lon_weights = _conservative_longitude_weights(
            self.source.lon, self.target.lon
        )
        lat_weights = _conservative_latitude_weights(
            self.source.lat, self.target.lat
        )
        result = np.einsum(
            'ac,bd,...cd->...ab',
            lon_weights,
            lat_weights,
            field,
            optimize=True,
        )
        return result

    def _nanmean(self, field: Array) -> np.ndarray:
        """Compute cell-averages skipping NaNs like np.nanmean."""
        nulls = np.isnan(field)
        total = self._mean(np.where(nulls, 0, field))
        count = self._mean(~nulls)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(total, count)
            result[count == 0] = np.nan  # Set divisions by zero to NaN
        return result

    def regrid_array(self, field: Array) -> np.ndarray:
        return self._nanmean(field)