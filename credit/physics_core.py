'''
Tools for physics-based constraints and derivations for CREDIT models
--------------------------------------------------------------------------
Content:
    - physics_pressure_level
        - pressure_integral_midpoint
        - pressure_integral_trapz
        - weighted_sum
        - total_dry_air_mass
        - total_column_water

Reference:
    - https://journals.ametsoc.org/view/journals/clim/34/10/JCLI-D-20-0676.1.xml
    - https://doi.org/10.1175/JCLI-D-13-00018.1
    - https://github.com/ai2cm/ace/tree/main/fme/fme/core

Yingkai Sha
ksha@ucar.edu
'''



import torch
from credit.physics_constants import *

class physics_pressure_level:
    '''
    Pressure level physics

    Attributes:
        upper_air_pressure (torch.Tensor): pressure levels in Pa.
        lon (torch.Tensor): longitude in degrees.
        lat (torch.Tensor): latitude in degrees.
        pressure_thickness (torch.Tensor): pressure thickness between levels.
        dx, dy (torch.Tensor): grid spacings in longitude and latitude.
        area (torch.Tensor): area of grid cells.
        integral (function): vertical integration method (midpoint or trapezoidal).
    '''
    
    def __init__(self,
                 lon: torch.Tensor,
                 lat: torch.Tensor,
                 upper_air_pressure: torch.Tensor,
                 midpoint: bool = False):
        '''
        Initialize the class with longitude, latitude, and pressure levels.

        All inputs must be in the same torch device.

        Full order of dimensions:  (batch, time, level, latitude, longitude)
        
        Args:
            lon (torch.Tensor): Longitude in degrees.
            lat (torch.Tensor): Latitude in degrees.
            upper_air_pressure (torch.Tensor): Pressure levels in Pa.
            midpoint (bool): True if vertical level quantities are midpoint values
                      otherwise False
            
        '''
        self.lon = lon
        self.lat = lat
        self.upper_air_pressure = upper_air_pressure
        
        # ========================================================================= #
        # compute pressure level thickness
        self.pressure_thickness = self.upper_air_pressure.diff(dim=-1)
        
        # # ========================================================================= #
        # # compute grid spacings
        # lat_rad = torch.deg2rad(self.lat)
        # lon_rad = torch.deg2rad(self.lon)
        # self.dy = torch.gradient(lat_rad * RAD_EARTH, dim=0)[0]
        # self.dx = torch.gradient(lon_rad * RAD_EARTH, dim=1)[0] * torch.cos(lat_rad)

        # ========================================================================= #
        # compute gtid area
        # area = R^2 * d_sin(lat) * d_lon
        lat_rad = torch.deg2rad(self.lat)
        lon_rad = torch.deg2rad(self.lon)
        sin_lat_rad = torch.sin(lat_rad)
        d_phi = torch.gradient(sin_lat_rad, dim=0, edge_order=2)[0]
        d_lambda = torch.gradient(lon_rad, dim=1, edge_order=2)[0]
        d_lambda = (d_lambda + torch.pi) % (2 * torch.pi) - torch.pi
        self.area = torch.abs(RAD_EARTH**2 * d_phi * d_lambda)
        
        # ========================================================================== #
        # vertical integration method
        if midpoint:
            self.integral = self.pressure_integral_midpoint
        else:
            self.integral = self.pressure_integral_trapz
            
    def pressure_integral_midpoint(self, q_mid: torch.Tensor) -> torch.Tensor:
        '''
        Compute the pressure level integral of a given quantity; assuming its mid point
        values are pre-computed
        
        Args:
            q_mid: the quantity with dims of (batch_size, time, level-1, latitude, longitude)
    
        Returns:
            Pressure level integrals of q
        '''
        num_dims = len(q_mid.shape)
        
        if num_dims == 5:  # (batch_size, time, level, latitude, longitude)
            delta_p = self.pressure_thickness.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q_area = q_mid * delta_p
            q_trapz = torch.sum(q_area, dim=2)
        
        elif num_dims == 4:  # (batch_size, level, latitude, longitude) or (time, level, latitude, longitude)
            delta_p = self.pressure_thickness.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q_area = q_mid * delta_p  # Trapezoidal rule
            q_trapz = torch.sum(q_area, dim=1)
        
        elif num_dims == 3:  # (level, latitude, longitude)
            delta_p = self.pressure_thickness.unsqueeze(-1).unsqueeze(-1)  # Expand for broadcasting
            q_area = q_mid * delta_p
            q_trapz = torch.sum(q_area, dim=0)
        
        else:
            raise ValueError(f"Unsupported tensor dimensions: {q.shape}")
        
        return q_trapz
        
    def pressure_integral_trapz(self, q: torch.Tensor) -> torch.Tensor:
        '''
        Compute the pressure level integral of a given quantity using the trapezoidal rule.
        
        Args:
            q: the quantity with dims of (batch_size, time, level, latitude, longitude)
    
        Returns:
            Pressure level integrals of q
        '''
        num_dims = len(q.shape)
        
        if num_dims == 5:  # (batch_size, time, level, latitude, longitude)
            delta_p = self.pressure_thickness.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q_area = 0.5 * (q[:, :, :-1, :, :] + q[:, :, 1:, :, :]) * delta_p
            q_trapz = torch.sum(q_area, dim=2)
        
        elif num_dims == 4:  # (batch_size, level, latitude, longitude) or (time, level, latitude, longitude)
            delta_p = self.pressure_thickness.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q_area = 0.5 * (q[:, :-1, :, :] + q[:, 1:, :, :]) * delta_p  # Trapezoidal rule
            q_trapz = torch.sum(q_area, dim=1)
        
        elif num_dims == 3:  # (level, latitude, longitude)
            delta_p = self.pressure_thickness.unsqueeze(-1).unsqueeze(-1)  # Expand for broadcasting
            q_area = 0.5 * (q[:-1, :, :] + q[1:, :, :]) * delta_p
            q_trapz = torch.sum(q_area, dim=0)
        
        else:
            raise ValueError(f"Unsupported tensor dimensions: {q.shape}")
        
        return q_trapz

    def weighted_sum(self,
                     q: torch.Tensor, 
                     axis: Dict[tuple, None] = None, 
                     keepdims: bool = False) -> torch.Tensor:
        '''
        Compute the weighted sum of a given quantity for PyTorch tensors.
        
        Args:
            data: the quantity to be summed (PyTorch tensor)
            axis: dims to compute the sum (can be int or tuple of ints)
            keepdims: whether to keep the reduced dimensions or not
    
        Returns:
            Weighted sum (PyTorch tensor)
        '''
        q_w = q * self.area
        q_sum = torch.sum(q_w, dim=axis, keepdim=keepdims)
        return q_sum

    def total_dry_air_mass(self, q: torch.Tensor) -> torch.Tensor:
        '''
        Compute the total mass of dry air over the entire globe [kg]
        '''
        mass_dry_per_area = self.integral(1-q) / GRAVITY # kg/m^2
        # weighted sum on latitude and longitude dimensions
        mass_dry_sum = self.weighted_sum(mass_dry_per_area, axis=(-2, -1)) # kg
        
        return mass_dry_sum

    def total_column_water(self, q: torch.Tensor) -> torch.Tensor:
        '''
        Compute total column water (TCW) per air column [kg/m2]
        '''
        TWC = self.integral(q) / GRAVITY # kg/m^2
        
        return TWC
