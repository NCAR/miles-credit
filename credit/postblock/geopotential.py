import torch
import numpy as np
import xarray as xr
from credit.metadata import get_meta_file_path


def pressure_on_interfaces(
    surface_pressure: torch.Tensor,
    model_a_half: torch.Tensor,
    model_b_half: torch.Tensor,
    model_top_pressure: float = 1.0,
):
    """
    Calculate pressure on the interfaces of atmospheric hybrid sigma-pressure model levels.
    Conversion is `pressure_3d = a + b * SP`.
    The `a` and `b` coefficients are defined at the interfaces of the levels rather than at the level centers where
    the mass variables are defined.

    Args:
        surface_pressure (torch.Tensor): (time, latitude, longitude) or (latitude, longitude) grid in units of Pa.
        model_a_half (torch.Tensor): coefficient a at each model level interface in units of Pa.
        model_b_half (torch.Tensor): coefficient b at each model level interface, which is unitless.
        model_top_pressure (float): pressure at model top (default 1 Pa).

    Returns:
        Pressure on each model level interface.
    """
    pressure_3d_half = model_a_half + model_b_half * surface_pressure
    pressure_3d_half = torch.where(pressure_3d_half > 0, pressure_3d_half, model_top_pressure)
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
    RVGAS = 461.51
    GAMMA = RVGAS / RDGAS - 1.0
    pressure_inter = pressure_on_interfaces(surface_pressure, model_a_half, model_b_half)
    pi_upper = torch.flip(pressure_inter[:-1], (0,))
    pi_lower = torch.flip(pressure_inter[1:], (0,))
    dlogp = torch.log(pi_lower / pi_upper)
    alpha = 1.0 - ((pi_upper / (pi_lower - pi_upper)) * dlogp)
    virtual_temperature = torch.flip((temperature * (1.0 + GAMMA * specific_humidity)), (0,))
    geopotential_interfaces = surface_geopotential + torch.cumsum(RDGAS * virtual_temperature * dlogp, dim=0)
    geopotential_centers = geopotential_interfaces + RDGAS * virtual_temperature * alpha
    geopotential_centers = torch.flip(geopotential_centers, (0,))
    return geopotential_centers


class GeopotentialDiagnostic(torch.nn.Module):
    def __init__(
        self,
        output_name: str = "geopotential",
        surface_geopotential_var: str = "geopotential_at_surface",
        surface_pressure_var: str = "surface_pressure",
        temperature_var: str = "temperature",
        specific_humidity_var: str = "specific_humidity",
        level_info_file: str = "ERA5_Lev_Info.nc",
        model_a_half_var: str = "a_half",
        model_b_half_var: str = "b_half",
    ):
        super().__init__()
        self.output_name = output_name
        self.surface_geopotential_var = surface_geopotential_var
        self.surface_pressure_var = surface_pressure_var
        self.temperature_var = temperature_var
        self.specific_humidity_var = specific_humidity_var
        self.level_info_file = get_meta_file_path(level_info_file)
        self.model_a_half_var = model_a_half_var
        self.model_b_half_var = model_b_half_var
        with xr.open_dataset(self.level_info_file) as level_info:
            self.model_a_half = torch.Tensor(level_info[self.model_a_half_var].values)
            self.model_b_half = torch.Tensor(level_info[self.model_b_half_var].values)
        return

    def forward(self, pred_dict: dict, chunk_size=1000):
        pred = pred_dict["prediction"]
        pred_shape = pred[self.temperature_var].shape
        pred_flat = {}
        for input_var in [
            self.surface_geopotential_var,
            self.surface_pressure_var,
            self.temperature_var,
            self.specific_humidity_var,
        ]:
            pred_per = torch.permute(pred[input_var], (0, 2, 3, 4, 5, 1))
            pred_flat[input_var] = pred_per.reshape(np.prod(pred_per.shape[:-1]), pred_per.shape[-1])
        vgeo = torch.vmap(geopotential, (0, 0, 0, 0, None, None), chunk_size=chunk_size)
        geo_out = vgeo(
            pred_flat[self.surface_geopotential_var],
            pred_flat[self.surface_pressure_var],
            pred_flat[self.temperature_var],
            pred_flat[self.specific_humidity_var],
            self.model_a_half,
            self.model_b_half,
        ).reshape(pred_shape[0], pred_shape[2], pred_shape[3], pred_shape[4], pred_shape[5], pred_shape[1])
        pred[self.output_name] = torch.permute(geo_out, (0, 5, 1, 2, 3, 4))
        return pred_dict
