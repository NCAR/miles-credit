
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class LESLoss2D(torch.nn.Module):
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
        super(LESLoss2D, self).__init__()

        self.conf = conf
        # self.training_loss = conf["loss"]["training_loss"]

        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"]
        
        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += [f"{v}_{k}" for v in diag_vars for k in range(levels)]

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
                value if isinstance(value, list) else [value]
                for value in conf["loss"]["variable_weights"].values()
            ]

            var_weights = np.array(
                [item for sublist in var_weights for item in sublist]
            )

            self.var_weights = torch.from_numpy(var_weights)
        # ------------------------------------------------------------- #

        self.use_spectral_loss = conf["loss"]["use_spectral_loss"]
        if self.use_spectral_loss:
            self.spectral_lambda_reg = conf["loss"]["spectral_lambda_reg"]
            self.spectral_loss_surface = SpectralLoss2D(
                wavenum_init=conf["loss"]["spectral_wavenum_init"], reduction="none"
            )

        self.use_power_loss = (
            conf["loss"]["use_power_loss"]
            if "use_power_loss" in conf["loss"]
            else False
        )
        if self.use_power_loss:
            self.power_lambda_reg = conf["loss"]["spectral_lambda_reg"]
            self.power_loss = PSDLoss(
                wavenum_init=conf["loss"]["spectral_wavenum_init"]
            )

        self.validation = validation
        
        if self.validation:
            self.loss_fn = nn.L1Loss(reduction="none")
        else:
            self.loss_fn = nn.L1Loss(reduction="none")
            #load_loss(self.training_loss, reduction="none")

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
            loss += self.power_lambda_reg * self.power_loss(
                target, pred, weights=self.lat_weights
            )

        if not self.validation and self.use_spectral_loss:
            loss += (
                self.spectral_lambda_reg
                * self.spectral_loss_surface(
                    target, pred, weights=self.lat_weights
                ).mean()
            )

        return loss
