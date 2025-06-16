import logging
from credit.losses.base_losses import base_losses
from credit.losses.weighted_loss import VariableTotalLoss2D


logger = logging.getLogger(__name__)


def load_loss(conf, reduction="none", validation=False):
    """
    Load the appropriate loss function based on the configuration.

    This function determines whether to use a weighted custom loss wrapper
    (such as `VariableTotalLoss2D`) when latitude or variable weights are enabled,
    or to load a standard or custom loss via `available_losses`.

    If in validation mode and a separate validation loss is specified in the config,
    that loss type will be used. Otherwise, the training loss is used.

    Args:
        conf (dict): Configuration dictionary. Must contain a 'loss' section with keys like:
                     - 'training_loss' (str): The primary loss function name.
                     - 'validation_loss' (optional, str): An alternate loss for validation.
                     - 'use_latitude_weights' (bool): Whether to use latitude-based weighting.
                     - 'use_variable_weights' (bool): Whether to use variable-specific weighting.
        reduction (str, optional): Reduction method to apply to the loss ('mean', 'sum', or 'none').
                                   Default is 'none'.
        validation (bool, optional): Whether the loss is being used for validation. Defaults to False.

    Returns:
        torch.nn.Module: A loss function instance, either weighted (`VariableTotalLoss2D`)
                         or a standard/custom loss from `available_losses`.

    Raises:
        ValueError: If the requested loss type is not recognized in `available_losses`.
    """
    loss_conf = conf["loss"]
    use_weighted_loss = loss_conf.get("use_latitude_weights", False) or loss_conf.get("use_variable_weights", False)

    if use_weighted_loss:
        logger.info("Loaded the VariableTotalLoss2D loss wrapper class for applying latititude or variable weights")
        return VariableTotalLoss2D(conf, validation=validation)

    # Select loss type for training or validation

    return base_losses(conf, reduction=reduction, validation=validation)
