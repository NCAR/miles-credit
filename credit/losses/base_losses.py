import torch.nn as nn
import logging

from credit.losses.logcosh import LogCoshLoss
from credit.losses.xtanh import XTanhLoss
from credit.losses.xsigmoid import XSigmoidLoss
from credit.losses.msle import MSLELoss
from credit.losses.kcrps import KCRPSLoss
from credit.losses.spectral import SpectralLoss2D
from credit.losses.power import PSDLoss
from credit.losses.almost_fair_crps import AlmostFairKCRPSLoss

logger = logging.getLogger(__name__)


def base_losses(conf, reduction="mean", validation=False):
    """Load a specified loss function by its type.

    Args:
        conf (dict): Configuration dictionary containing loss settings.
        reduction (str, optional): Default reduction method if not specified in parameters.
        validation (bool): Use validation loss settings if True, else training loss.

    Returns:
        torch.nn.Module: Instantiated loss function.
    """
    loss_key = "validation_loss" if validation else "training_loss"
    params_key = "validation_loss_parameters" if validation else "training_loss_parameters"

    loss_type = conf["loss"][loss_key]
    loss_params = conf["loss"].get(params_key, {})

    # Ensure 'reduction' is included
    if "reduction" not in loss_params:
        loss_params["reduction"] = reduction

    logger.info(f"Loaded the {loss_type} loss function with parameters: {loss_params}")

    # Standard loss registry
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "msle": MSLELoss,
        "huber": nn.HuberLoss,
        "logcosh": LogCoshLoss,
        "xtanh": XTanhLoss,
        "xsigmoid": XSigmoidLoss,
        "KCRPS": KCRPSLoss,
        "almost-fair-crps": AlmostFairKCRPSLoss,
        "spectral": SpectralLoss2D,
        "power": PSDLoss,
    }

    if loss_type in losses:
        return losses[loss_type](**loss_params)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported")
