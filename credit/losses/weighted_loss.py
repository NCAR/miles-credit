import torch
import xarray as xr
import numpy as np
import logging

from credit.losses.base_losses import base_losses
from credit.losses.spectral import SpectralLoss2D
from credit.losses.power import PSDLoss


logger = logging.getLogger(__name__)


def latitude_weights(conf):
    """Calculate latitude-based weights for loss function.
    This function calculates weights based on latitude values
    to be used in loss functions for geospatial data. The weights
    are derived from the cosine of the latitude and normalized
    by their mean.

    Args:
        conf (dict): Configuration dictionary containing the
            path to the latitude weights file.

    Returns:
        torch.Tensor: A 2D tensor of weights with dimensions
            corresponding to latitude and longitude.
    """
    # Open the dataset and extract latitude and longitude information
    ds = xr.open_dataset(conf["loss"]["latitude_weights"])
    lat = torch.from_numpy(ds["latitude"].values).float()
    lon_dim = ds["longitude"].shape[0]

    # Calculate weights using PyTorch operations
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()

    # Create a 2D tensor of weights
    L = weights.unsqueeze(1).expand(-1, lon_dim)

    return L


def variable_weights(conf, channels, frames):
    """Create variable-specific weights for different atmospheric
    and surface channels.

    This function loads weights for different atmospheric variables
    (e.g., U, V, T, Q) and surface variables (e.g., SP, t2m) from
    the configuration file. It then combines them into a single
    weight tensor for use in loss calculations.

    Args:
        conf (dict): Configuration dictionary containing the
            variable weights.
        channels (int): Number of channels for atmospheric variables.
        frames (int): Number of time frames.

    Returns:
        torch.Tensor: A tensor containing the combined weights for
            all variables.
    """
    # Load weights for U, V, T, Q
    varname_upper_air = conf["data"]["variables"]
    varname_surface = conf["data"]["surface_variables"]
    varname_diagnostics = conf["data"]["diagnostic_variables"]

    # surface + diag channels
    N_channels_single = len(varname_surface) + len(varname_diagnostics)

    weights_upper_air = torch.tensor([conf["loss"]["variable_weights"][var] for var in varname_upper_air]).view(
        1, channels * frames, 1, 1
    )

    weights_single = torch.tensor(
        [conf["loss"]["variable_weights"][var] for var in (varname_surface + varname_diagnostics)]
    ).view(1, N_channels_single, 1, 1)

    # Combine all weights along the color channel
    var_weights = torch.cat([weights_upper_air, weights_single], dim=1)

    return var_weights


class VariableTotalLoss2D(torch.nn.Module):
    """Custom loss function class for 2D geospatial data
    with optional spectral and power loss components.

    This class defines a loss function that combines a base loss
    (e.g., L1, MSE) with optional spectral and power loss components
    for 2D geospatial data. The loss function can incorporate latitude
    and variable-specific weights.

    Args:
        conf (dict): Configuration dictionary containing loss
            function settings and weights.
        validation (bool, optional): If True, the loss function
            is used in validation mode. Defaults to False.
    """

    def __init__(self, conf, validation=False):
        """Initialize the VariableTotalLoss2D.

        Args:
            conf (str): path to config file.
            training_loss (str): Loss equation to use during training.
            vars (list): Atmospheric, surface, and diagnostic variable names.
            lat_weights (str): Path to upper latitude weights file.
            var_weights (bool): Whether to use variable weights during training.
            use_spectral_loss (bool): Whether to use spectral loss during training.

            varname_surface (list): List of surface variable names.
            varname_dyn_forcing (list): List of dynamic forcing variable names.
            varname_forcing (list): List of forcing variable names.
            varname_static (list): List of static variable names.
            varname_diagnostic (list): List of diagnostic variable names.
            filenames (list): List of filenames for upper air data.
            filename_surface (list, optional): List of filenames for surface data.
            filename_dyn_forcing (list, optional): List of filenames for dynamic forcing data.
            filename_forcing (str, optional): Filename for forcing data.
            filename_static (str, optional): Filename for static data.
            filename_diagnostic (list, optional): List of filenames for diagnostic data.
            history_len (int, optional): Length of the history sequence. Default is 2.
            forecast_len (int, optional): Length of the forecast sequence. Default is 0.
            transform (callable, optional): Transformation function to apply to the data.
            seed (int, optional): Random seed for reproducibility. Default is 42.
            skip_periods (int, optional): Number of periods to skip between samples.
            one_shot(bool, optional): Whether to return all states or just
                                    the final state of the training target. Default is None
            max_forecast_len (int, optional): Maximum length of the forecast sequence.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.

        Returns:
            sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.

        """
        super(VariableTotalLoss2D, self).__init__()

        self.conf = conf
        self.training_loss = conf["loss"]["training_loss"]

        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"] if "levels" in conf["model"] else conf["model"]["frames"]

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars

        self.lat_weights = None
        if conf["loss"]["use_latitude_weights"]:
            logger.info("Using latitude weights in loss calculations")
            self.lat_weights = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # ------------------------------------------------------------- #
        # variable weights
        # order: upper air --> surface --> diagnostics
        self.var_weights = None
        if conf["loss"]["use_variable_weights"]:
            logger.info("Using variable weights in loss calculations")

            var_weights = [
                value if isinstance(value, list) else [value] for value in conf["loss"]["variable_weights"].values()
            ]

            var_weights = np.array([item for sublist in var_weights for item in sublist])

            self.var_weights = torch.from_numpy(var_weights)
        # ------------------------------------------------------------- #

        self.use_spectral_loss = conf["loss"]["use_spectral_loss"]
        if self.use_spectral_loss:
            self.spectral_lambda_reg = conf["loss"]["spectral_lambda_reg"]
            self.spectral_loss_surface = SpectralLoss2D(
                wavenum_init=conf["loss"]["spectral_wavenum_init"], reduction="none"
            )

        self.use_power_loss = conf["loss"]["use_power_loss"] if "use_power_loss" in conf["loss"] else False
        if self.use_power_loss:
            self.power_lambda_reg = conf["loss"]["spectral_lambda_reg"]
            self.power_loss = PSDLoss(wavenum_init=conf["loss"]["spectral_wavenum_init"])

        self.validation = validation
        if conf["loss"]["training_loss"] == "KCRPS":  # for ensembles, load same loss for train and valid
            self.loss_fn = base_losses(conf, reduction="none", validation=False)
        elif self.validation:
            if "validation_loss" in conf["loss"]:
                self.loss_fn = base_losses(conf, reduction="none", validation=True)
            else:
                self.loss_fn = torch.nn.L1Loss(reduction="none")
        else:
            self.loss_fn = base_losses(conf, reduction="none", validation=False)

    def forward(self, target, pred):
        """Calculate the total loss for the given target and prediction.

        This method computes the base loss between the target and prediction,
        applies latitude and variable weights, and optionally adds spectral
        and power loss components.

        Args:
            target (torch.Tensor): Ground truth tensor.
            pred (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: The computed loss value.

        """
        # User defined loss
        loss = self.loss_fn(target, pred)

        # Latitutde and variable weights
        loss_dict = {}
        for i, var in enumerate(self.vars):
            var_loss = loss[:, i]

            if self.lat_weights is not None:
                var_loss = torch.mul(var_loss, self.lat_weights.to(target.device))

            if self.var_weights is not None:
                var_loss *= self.var_weights[i].to(target.device)

            loss_dict[f"loss_{var}"] = var_loss.mean()

        loss = torch.mean(torch.stack(list(loss_dict.values())))

        # Add the spectral loss
        if not self.validation and self.use_power_loss:
            loss += self.power_lambda_reg * self.power_loss(target, pred, weights=self.lat_weights)

        if not self.validation and self.use_spectral_loss:
            loss += self.spectral_lambda_reg * self.spectral_loss_surface(target, pred, weights=self.lat_weights).mean()

        return loss
