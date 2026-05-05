import torch
import numpy as np


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
    model_b_half,
):
    """
    Calculate geopotential (m**2 s**-2) from hybrid sigma-pressure atmospheric level data.

    This calculation assumes that the 3D fields are oriented in the vertical/levels
    dimension such that pressure increases with increasing index value (top of atmosphere to surface).

    Args:
        surface_geopotential (torch.Tensor):
        surface_pressure:
        temperature:
        specific_humidity:
        model_a_half:
        model_b_half:

    Returns:

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
    geopotential_centers = 0.5 * (geopotential_interfaces[:-1] + RDGAS * virtual_temperature)
    return geopotential_centers
