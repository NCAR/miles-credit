import torch
import numpy as np
import xarray as xr
from credit.metadata import get_meta_file_path
from typing import Iterable
from functools import partial


def pressure_on_interfaces(
    surface_pressure: torch.Tensor,
    model_a_half: torch.Tensor,
    model_b_half: torch.Tensor,
    model_top_pressure: float = 0.57,
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
    model_a_3d = model_a_half
    model_b_3d = model_b_half
    pressure_3d_half = model_a_3d + model_b_3d * surface_pressure
    pressure_3d_half = torch.where(pressure_3d_half > 0, pressure_3d_half, model_top_pressure)
    return pressure_3d_half


def geopotential(
    surface_geopotential: torch.Tensor,
    surface_pressure: torch.Tensor,
    temperature: torch.Tensor,
    specific_humidity: torch.Tensor,
    model_a_half: torch.Tensor,
    model_b_half: torch.Tensor,
    flip_vertical: bool = True,
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
        flip_vertical (bool): whether to flip the vertical dimension of the input tensors. Default True.

    Returns:
        geopotential (torch.Tensor): geopotential on each model level (m**2 s**-2)
    """
    RDGAS = 287.06
    RVGAS = 461.51
    GAMMA = RVGAS / RDGAS - 1.0
    pressure_inter = pressure_on_interfaces(surface_pressure, model_a_half, model_b_half)
    pi_upper = pressure_inter[:-1]
    pi_lower = pressure_inter[1:]
    if flip_vertical:
        pi_upper = torch.flip(pi_upper, (0,))
        pi_lower = torch.flip(pi_lower, (0,))
    dlogp = torch.log(pi_lower / pi_upper)
    alpha = 1.0 - ((pi_upper / (pi_lower - pi_upper)) * dlogp)
    virtual_temperature = temperature * (1.0 + GAMMA * specific_humidity)
    if flip_vertical:
        virtual_temperature = torch.flip(virtual_temperature, (0,))
    geopotential_interfaces = surface_geopotential + torch.cumsum(RDGAS * virtual_temperature * dlogp, dim=0)
    # I flipped the sign here, and it lines up much better with ERA5 geopotential
    geopotential_centers = geopotential_interfaces - RDGAS * virtual_temperature * alpha
    if flip_vertical:
        geopotential_centers = torch.flip(geopotential_centers, (0,))
    return geopotential_centers


class GeopotentialDiagnostic(torch.nn.Module):
    """
    GeopotentialDiagnostic is a neural network module used for computing geopotential
    diagnostics using multi-dimensional input data.

    This class processes geophysical variables such as surface geopotential, surface
    pressure, temperature, and specific humidity to calculate geopotential fields.
    The input data is expected to conform to a specific format, and the class makes
    use of auxiliary metadata files that describe model-specific level information.

    Attributes:
        output_name (str): The key used in the dataset to store the computed
            geopotential diagnostic output.
        dataset_name (str): The name of the dataset from which input variables
            will be retrieved.
        chunk_size (int): The chunk size used for vectorized computations
            to optimize memory usage during processing.
        data_keys (Iterable[str]): The keys in the input data dictionary that
            will be processed (e.g., "prediction", "target").
        surface_geopotential_var (str): The key for the surface geopotential variable
            in the dataset.
        surface_pressure_var (str): The key for the surface pressure variable
            in the dataset.
        temperature_var (str): The key for the temperature variable in the dataset.
        specific_humidity_var (str): The key for the specific humidity variable
            in the dataset.
        flip_vertical (bool): Whether to flip the vertical dimension of the input tensors. Default True
        level_info_file (str): The filename of the auxiliary metadata file that
            stores information about model levels.
        model_a_half_var (str): The variable name for the `a` (pressure) hybrid sigma-pressure coefficient in
            the level information file.
        model_b_half_var (str): The variable name for the `b` (sigma) hybrid sigma-pressure coefficient parameter in
            the level information file.
    """

    def __init__(
        self,
        output_name: str = "ARCO_ERA5/derived_diagnostic/3d/geopotential",
        dataset_name: str = "ARCO_ERA5",
        chunk_size: int = 1000,
        data_keys: Iterable[str] = ("prediction", "target"),
        surface_geopotential_var: str = "ARCO_ERA5/static/2d/geopotential_at_surface",
        surface_pressure_var: str = "ARCO_ERA5/prognostic/2d/surface_pressure",
        temperature_var: str = "ARCO_ERA5/prognostic/3d/temperature",
        specific_humidity_var: str = "ARCO_ERA5/prognostic/3d/specific_humidity",
        flip_vertical: bool = True,
        level_info_file: str = "ERA5_Lev_Info.nc",
        model_a_half_var: str = "a_half",
        model_b_half_var: str = "b_half",
    ):
        super().__init__()
        self.output_name = output_name
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size
        self.data_keys = data_keys
        self.surface_geopotential_var = surface_geopotential_var
        self.surface_pressure_var = surface_pressure_var
        self.temperature_var = temperature_var
        self.specific_humidity_var = specific_humidity_var
        self.flip_vertical = flip_vertical
        self.level_info_file = get_meta_file_path(level_info_file)
        self.model_a_half_var = model_a_half_var
        self.model_b_half_var = model_b_half_var
        with xr.open_dataset(self.level_info_file) as level_info:
            self.model_a_half = torch.Tensor(level_info[self.model_a_half_var].values)
            self.model_b_half = torch.Tensor(level_info[self.model_b_half_var].values)
        return

    def forward(self, data_dict: dict):
        """
        Processes a dictionary of input data, rearranges dimensions, computes derived quantities
        using a custom function `geopotential`, and updates the data dictionary with the results.

        Args:
            data_dict (dict): Input dictionary containing data corresponding to various
                data types. The data for each type is expected to be organized into specified
                attributes (e.g., temperature, specific humidity).

        Returns:
            dict: Updated data dictionary, where new computed fields are added to the
            relevant dataset, preserving the original structure.

        Raises:
            ValueError: If any required data type is not found in the input `data_dict`.
        """
        for data_type in self.data_keys:
            if data_type not in data_dict:
                raise ValueError(f"Data key {data_type} not found in data_dict.")
            data = data_dict[data_type]
            pred_shape = list(data[self.dataset_name][self.temperature_var].shape)  # (B, n_levels, n_time, H, W)
            pred_flat = {}
            dsn = self.dataset_name
            for input_var in [
                self.surface_geopotential_var,
                self.surface_pressure_var,
                self.temperature_var,
                self.specific_humidity_var,
            ]:
                new_dim_order = tuple([0] + list(range(2, len(data[dsn][input_var].shape))) + [1])
                pred_per = torch.permute(data[dsn][input_var], new_dim_order)  # (B, n_time, H, W, n_levels)
                total_shape = int(np.prod(pred_per.shape[:-1]))
                pred_flat[input_var] = pred_per.reshape(total_shape, pred_per.shape[-1])
            vgeo = torch.vmap(
                partial(geopotential, flip_vertical=self.flip_vertical),
                (0, 0, 0, 0, None, None),
                chunk_size=self.chunk_size,
            )
            geo_out = vgeo(
                pred_flat[self.surface_geopotential_var],
                pred_flat[self.surface_pressure_var],
                pred_flat[self.temperature_var],
                pred_flat[self.specific_humidity_var],
                self.model_a_half,
                self.model_b_half,
            ).reshape(*[pred_shape[0]] + pred_shape[2:] + [pred_shape[1]])  # (B, n_time, H, W, n_levels)
            final_dim_order = tuple([0] + [len(pred_shape) - 1] + list(range(1, len(pred_shape) - 1)))
            data[dsn][self.output_name] = torch.permute(geo_out, final_dim_order)
        return data_dict
