import torch.nn as nn
import logging

from credit.losses.logcosh import LogCoshLoss
from credit.losses.xtanh import XTanhLoss
from credit.losses.xsigmoid import XSigmoidLoss
from credit.losses.msle import MSLELoss
from credit.losses.crps import RingCRPSLoss
from credit.losses.kcrps import KCRPSLoss
from credit.losses.spectral import SpectralLoss2D
from credit.losses.power import PSDLoss
from credit.losses.almost_fair_crps import AlmostFairKCRPSLoss
from credit.losses.covariance import CovarianceWeightedMSELoss

logger = logging.getLogger(__name__)


LOSSES = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "msle": MSLELoss,
    "huber": nn.HuberLoss,
    "logcosh": LogCoshLoss,
    "xtanh": XTanhLoss,
    "xsigmoid": XSigmoidLoss,
    "KCRPS": KCRPSLoss,
    "almost-fair-crps": AlmostFairKCRPSLoss,
    "ring-crps": RingCRPSLoss,
    "spectral": SpectralLoss2D,
    "power": PSDLoss,
    "covmse": CovarianceWeightedMSELoss,
}
CRPS_LOSSES = frozenset(name for name in LOSSES if "crps" in name.casefold())


def is_crps_loss(loss_type):
    return loss_type in CRPS_LOSSES


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

    mode = "validation" if validation else "train"
    logger.info(f"Loaded the {loss_type} loss function ({mode}) with parameters: {loss_params}")

    if loss_type in LOSSES:
        return LOSSES[loss_type](**loss_params)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported")
