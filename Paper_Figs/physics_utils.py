import xarray as xr
import numpy as np

R = 6371000  # m
GRAVITY = 9.80665
RHO_WATER = 1000.0 # kg/m^3
RAD_EARTH = 6371000 # m
LH_WATER = 2.26e6  # J/kg
CP_DRY = 1005 # J/kg K
CP_VAPOR = 1846 # J/kg K



# Function to compute grid cell areas
def grid_area(lat, lon):
    '''
    Compute grid cell areas on a sphere for a regular latitude-longitude grid.

    Args:
        lat: xarray.DataArray of latitudes in degrees, dimension 'latitude'
        lon: xarray.DataArray of longitudes in degrees, dimension 'longitude'

    Returns:
        area: xarray.DataArray of grid cell areas in square meters, dimensions ('latitude', 'longitude')
    '''
    
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Compute sine of latitude
    sin_lat_rad = np.sin(lat_rad)
    
    # Compute gradient of sine of latitude (d_phi)
    d_phi = np.gradient(sin_lat_rad, axis=0, edge_order=2)
    
    # Compute gradient of longitude (d_lambda)
    d_lambda = np.gradient(lon_rad, axis=1, edge_order=2)
    
    # Adjust d_lambda to be within -π and π
    d_lambda = (d_lambda + np.pi) % (2 * np.pi) - np.pi
    
    # Compute grid cell area
    area = np.abs(RAD_EARTH**2 * d_phi * d_lambda)

    # Create DataArray for area with dimensions ('latitude', 'longitude')
    area = xr.DataArray(
        area,
        coords={'latitude': lat[:, 0], 'longitude': lon[0, :]},
        dims=['latitude', 'longitude']
    )

    return area


def pressure_integral_midpoint(q_mid, surface_pressure, coef_a, coef_b)
    '''
    Compute the pressure level integral of a given quantity; assuming its mid-point
    values are pre-computed.

    Args:
        q_mid: The quantity with dims of (batch, level-1, time, latitude, longitude)
        surface_pressure: Surface pressure in Pa (batch, time, latitude, longitude).

    Returns:
        Pressure level integrals of q
    '''
    # (batch, 1, lat, lon)
    surface_pressure = surface_pressure.unsqueeze(1)

    # (batch, level, lat, lon)
    pressure = coef_a + coef_b * surface_pressure

    # (batch, level-1, lat, lon)
    delta_p = pressure.diff(dim=1).to(q_mid.device)

    # Element-wise multiplication
    q_area = q_mid * delta_p

    # Sum over level dimension
    q_integral = torch.sum(q_area, dim=1)
    
    return q_integral


def weighted_sum(data, weights, dims):
    '''
    Compute the weighted sum of a given quantity using xarray.

    Args:
        data: xarray.DataArray to be summed
        weights: xarray.DataArray of weights, broadcastable to data
        dims: dimensions over which to sum

    Returns:
        xarray.DataArray: weighted sum
    '''
    # Perform weighted sum
    return (data * weights).sum(dim=dims)