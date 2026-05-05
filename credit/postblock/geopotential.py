import torch
import numpy as np
import xarray as xr


def pressure_on_interfaces(surface_pressure: torch.Tensor, model_a_half: torch.Tensor, model_b_half: torch.Tensor):
    """
    Calculate pressure on the interfaces of atmospheric hybrid sigma-pressure model levels.
    Conversion is `pressure_3d = a + b * SP`.
    The `a` and `b` coefficients are defined at the interfaces of the levels rather than at the level centers where
    the mass variables are defined.

    Args:
        surface_pressure (torch.Tensor): (time, latitude, longitude) or (latitude, longitude) grid in units of Pa.
        model_a_half (torch.Tensor): coefficient a at each model level interface in units of Pa.
        model_b_half (torch.Tensor): coefficient b at each model level interface, which is unitless.

    Returns:
        Pressure on each model level interface.
    """
    model_a_3d = model_a_half.reshape(-1, 1, 1)
    model_b_3d = model_b_half.reshape(-1, 1, 1)
    pressure_3d_half = model_a_3d + model_b_3d * surface_pressure
    return pressure_3d_half


def geopotential(
    surface_geopotential: torch.Tensor,
    surface_pressure: torch.Tensor,
    temperature: torch.Tensor,
    specific_humidity: torch.Tensor,
    model_a_half: torch.Tensor,
    model_b_half: torch.Tensor,
):
    """
    Calculate geopotential (m**2 s**-2) from hybrid sigma-pressure atmospheric level data.

    This calculation assumes that the 3D fields are oriented in the vertical/levels
    dimension such that pressure increases with increasing index value (top of atmosphere to surface).

    Args:
        surface_geopotential (torch.Tensor): surface geopotential (m**2 s**-2)
        surface_pressure (torch.Tensor): surface pressure (Pa)
        temperature (torch.Tensor): temperature (K)
        specific_humidity (torch.Tensor): specific humidity (kg kg**-1)
        model_a_half (torch.Tensor): pressure component a at each model level interface (Pa)
        model_b_half (torch.Tensor): sigma component b at each model level interface (unitless)

    Returns:
        geopotential (torch.Tensor): geopotential on each model level (m**2 s**-2)
    """
    RDGAS = 287.06
    GAMMA = 0.609133
    pressure_inter = pressure_on_interfaces(surface_pressure, model_a_half, model_b_half)
    pi_upper = pressure_inter[:-1][::-1]
    pi_lower = pressure_inter[1:][::-1]
    pressure_center = 0.5 * (pi_upper + pi_lower)
    dlogp = torch.log(pi_lower / pi_upper)
    dlogp_center = torch.log(pi_lower / pressure_center)
    virtual_temperature = (temperature * (1.0 + GAMMA * specific_humidity))[::-1]
    geopotential_interfaces = surface_geopotential + torch.cumsum(RDGAS * virtual_temperature * dlogp, dim=0)
    geopotential_top = geopotential_interfaces[-1] + RDGAS * virtual_temperature[-1] * 2.0 * np.log(2.0)
    geopotential_interfaces = torch.concat([geopotential_interfaces, geopotential_top], dim=0)
    geopotential_centers = 0.5 * (geopotential_interfaces[:-1] + RDGAS * virtual_temperature * dlogp_center)
    return geopotential_centers


class GeopotentialDiagnostic(torch.nn.Module):
    def __init__(
        self,
        surface_geopotential_var: str,
        surface_pressure_var: str,
        temperature_var: str,
        specific_humidity_var: str,
        level_info_file: str,
        model_a_half_var: str,
        model_b_half_var: str,
    ):
        super().__init__()
        self.surface_geopotential_var = surface_geopotential_var
        self.surface_geopotential_var = surface_pressure_var
        self.temperature_var = temperature_var
        self.specific_humidity_var = specific_humidity_var
        self.level_info_file = level_info_file
        self.model_a_half_var = model_a_half_var
        self.model_b_half_var = model_b_half_var
        with xr.open_dataset(self.level_info_file) as level_info:
            self.model_a_half = torch.Tensor(level_info[self.model_a_half_var].values)
            self.model_b_half = torch.Tensor(level_info[self.model_a_half_var].values)
        return

    def forward(self, x):
        return geopotential(
            x[self.surface_geopotential_var],
            x[self.surface_geopotential_var],
            x[self.temperature_var],
            x[self.specific_humidity_var],
            self.model_a_half,
            self.model_b_half,
        )
