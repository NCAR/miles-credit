import torch
import numpy as np
import xarray as xr
from credit.metadata import get_meta_file_path
from credit.postblock.base import BasePostblock
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
        model_top_pressure (float): pressure at model top (default 0.57 Pa).

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


class GeopotentialDiagnostic(BasePostblock):
    """
    GeopotentialDiagnostic is a neural network module used for computing geopotential
    diagnostics using multi-dimensional input data.

    This class processes geophysical variables such as surface geopotential, surface
    pressure, temperature, and specific humidity to calculate geopotential fields.
    The class makes use of auxiliary metadata files that describe model-specific
    level information.

    Follows the same batch-dict protocol as ``Reconstruct`` and
    ``BridgeScalerTransform``: it operates on the nested output dict at
    ``batch_dict[key]`` (default ``"y_processed"``), which has the form
    ``{source: {var_key: tensor}}`` where ``var_key`` is
    ``"source/field_type/dim/varname"``. The source for each variable is
    derived from the first path component of its ``var_key``. The static
    surface geopotential is read from ``batch_dict[static_source_key]``
    (default ``"ic_raw"``), which has the same nested form, since static
    fields are not part of the reconstructed model output. The result is
    written back into ``batch_dict[key]`` under ``output_name``.

    Attributes:
        output_name (str): The var_key used to store the computed
            geopotential diagnostic output in ``batch_dict[key]``.
        chunk_size (int): The chunk size used for vectorized computations
            to optimize memory usage during processing.
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
        key (str): entry in ``batch_dict`` holding the nested output dict written
            by ``Reconstruct`` (default: ``"y_processed"``).
        static_source_key (str): entry in ``batch_dict`` holding the nested raw IC
            dict that provides static fields (default: ``"ic_raw"``).
    """

    def __init__(
        self,
        output_name: str = "ARCO_ERA5/derived_diagnostic/3d/geopotential",
        chunk_size: int = 1000,
        surface_geopotential_var: str = "ARCO_ERA5/static/2d/geopotential_at_surface",
        surface_pressure_var: str = "ARCO_ERA5/prognostic/2d/surface_pressure",
        temperature_var: str = "ARCO_ERA5/prognostic/3d/temperature",
        specific_humidity_var: str = "ARCO_ERA5/prognostic/3d/specific_humidity",
        flip_vertical: bool = True,
        level_info_file: str = "ERA5_Lev_Info.nc",
        model_a_half_var: str = "a_half",
        model_b_half_var: str = "b_half",
        key: str = "y_processed",
        static_source_key: str = "ic_raw",
        levels: list[int] | None = None,
    ):
        super().__init__()
        self.output_name = output_name
        self.chunk_size = chunk_size
        self.key = key
        self.surface_geopotential_var = surface_geopotential_var
        self.surface_pressure_var = surface_pressure_var
        self.temperature_var = temperature_var
        self.specific_humidity_var = specific_humidity_var
        self.flip_vertical = flip_vertical
        self.level_info_file = get_meta_file_path(level_info_file)
        self.model_a_half_var = model_a_half_var
        self.model_b_half_var = model_b_half_var
        self.static_source_key = static_source_key
        self.levels = levels
        with xr.open_dataset(self.level_info_file) as level_info:
            a_all = torch.Tensor(level_info[self.model_a_half_var].values)
            b_all = torch.Tensor(level_info[self.model_b_half_var].values)
        if levels is not None:
            half_idx = [lv - 1 for lv in levels] + [levels[-1]]
            self.model_a_half = a_all[half_idx]
            self.model_b_half = b_all[half_idx]
        else:
            self.model_a_half = a_all
            self.model_b_half = b_all
        return

    def forward(self, batch_dict: dict):
        """
        Computes geopotential from the nested output dict and writes it back into the batch dict.

        Args:
            batch_dict (dict): batch dict containing ``key`` and ``static_source_key``
                entries, each of the form ``{source: {var_key: tensor}}`` with tensors
                of shape ``(B, n_levels, n_time, H, W)``.

        Returns:
            dict: The same ``batch_dict`` with ``output_name`` added under
            ``batch_dict[key][source]``.

        Raises:
            ValueError: If ``key`` or ``static_source_key`` is not found in ``batch_dict``.
        """
        for required_key in (self.key, self.static_source_key):
            if required_key not in batch_dict:
                raise ValueError(f"Key {required_key!r} not found in batch_dict.")
        nested = batch_dict[self.key]  # {source: {var_key: tensor}}
        static_nested = batch_dict[self.static_source_key]
        temp_source = self.temperature_var.split("/")[0]
        pred_shape = list(nested[temp_source][self.temperature_var].shape)  # (B, n_levels, n_time, H, W)
        pred_flat = {}
        for input_var in [
            self.surface_geopotential_var,
            self.surface_pressure_var,
            self.temperature_var,
            self.specific_humidity_var,
        ]:
            source = input_var.split("/")[0]
            src = static_nested[source] if input_var == self.surface_geopotential_var else nested[source]
            new_dim_order = tuple([0] + list(range(2, len(src[input_var].shape))) + [1])
            pred_per = torch.permute(src[input_var], new_dim_order)  # (B, n_time, H, W, n_levels)
            total_shape = int(np.prod(pred_per.shape[:-1]))
            pred_flat[input_var] = pred_per.reshape(total_shape, pred_per.shape[-1])
        device = pred_flat[self.surface_pressure_var].device
        pred_flat = {k: v.to(device) for k, v in pred_flat.items()}
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
            self.model_a_half.to(device),
            self.model_b_half.to(device),
        ).reshape(*[pred_shape[0]] + pred_shape[2:] + [pred_shape[1]])  # (B, n_time, H, W, n_levels)
        final_dim_order = tuple([0] + [len(pred_shape) - 1] + list(range(1, len(pred_shape) - 1)))
        out_source = self.output_name.split("/")[0]
        nested.setdefault(out_source, {})[self.output_name] = torch.permute(geo_out, final_dim_order)
        return batch_dict
