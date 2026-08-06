import importlib
import inspect
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config-driven dispatch: maps config-file keys → class (used by load_loss).
# Registry entries are either:
#   (module_path: str, class_name: str)  — built-in lazy entries
#   cls: type                             — externally registered classes
_LOSS_REGISTRY = {
    "mse": ("torch.nn", "MSELoss"),
    "mae": ("torch.nn", "L1Loss"),
    "msle": ("credit.losses.msle", "MSLELoss"),
    "huber": ("torch.nn", "HuberLoss"),
    "logcosh": ("credit.losses.gen_1.logcosh", "LogCoshLoss"),
    "xtanh": ("credit.losses.gen_1.xtanh", "XTanhLoss"),
    "xsigmoid": ("credit.losses.gen_1.xsigmoid", "XSigmoidLoss"),
    "KCRPS": ("credit.losses.gen_1.kcrps", "KCRPSLoss"),
    "almost-fair-crps": ("credit.losses.gen_1.almost_fair_crps", "AlmostFairKCRPSLoss"),
    "ring-crps": ("credit.losses.crps", "RingCRPSLoss"),
    "spectral": ("credit.losses.gen_1.spectral", "SpectralLoss2D"),
    "power": ("credit.losses.gen_1.power", "PSDLoss"),
    "covmse": ("credit.losses.gen_1.covariance", "CovarianceWeightedMSELoss"),
    "base": ("credit.losses.base", "BaseLoss"),
}

# Names of registered losses that are CRPS-family (ensemble) losses, used by
# the gen2 trainer to decide whether trainer.ensemble_size > 1 is required.
CRPS_LOSSES = frozenset(name for name in _LOSS_REGISTRY if "crps" in name.casefold())


def is_crps_loss(loss_type):
    return loss_type in CRPS_LOSSES


# Direct-import table: maps Python class names → class for lazy module attribute access.
# Enables ``from credit.losses import LogCoshLoss`` without eager imports; kept for backward compatibility.
_CLASS_SOURCES = {
    "MSLELoss": ("credit.losses.msle", "MSLELoss"),
    "LogCoshLoss": ("credit.losses.gen_1.logcosh", "LogCoshLoss"),
    "XTanhLoss": ("credit.losses.gen_1.xtanh", "XTanhLoss"),
    "XSigmoidLoss": ("credit.losses.gen_1.xsigmoid", "XSigmoidLoss"),
    "KCRPSLoss": ("credit.losses.gen_1.kcrps", "KCRPSLoss"),
    "AlmostFairKCRPSLoss": ("credit.losses.gen_1.almost_fair_crps", "AlmostFairKCRPSLoss"),
    "SpectralLoss2D": ("credit.losses.gen_1.spectral", "SpectralLoss2D"),
    "PSDLoss": ("credit.losses.gen_1.power", "PSDLoss"),
    "CovarianceWeightedMSELoss": ("credit.losses.gen_1.covariance", "CovarianceWeightedMSELoss"),
}


# ---------------------------------------------------------------------------
# Module __getattr__: called when a name is not found via normal attribute lookup.
# Resolves names listed in _CLASS_SOURCES lazily so submodules are only imported on first access.
# Example: ``from credit.losses import LogCoshLoss`` triggers __getattr__("LogCoshLoss"),
#          which imports credit.losses.gen_1.logcosh on the spot and returns the class.
def __getattr__(name):
    if name == "VariableTotalLoss2D":
        from credit.losses.gen_1.weighted_loss import VariableTotalLoss2D

        return VariableTotalLoss2D
    if name == "BaseLoss":
        from credit.losses.base import BaseLoss

        return BaseLoss
    if name == "DownscalingLoss":
        from credit.losses.gen_1.downscaling_loss import DownscalingLoss

        return DownscalingLoss
    if name in _CLASS_SOURCES:
        module_path, class_name = _CLASS_SOURCES[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise AttributeError(f"Cannot import {name!r}: optional dependencies missing.") from exc
    raise AttributeError(f"module 'credit.losses' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Registration
def register_loss(loss_type):
    """Decorator that adds an external loss class to the loss registry.

    The class must inherit from ``torch.nn.Module`` so that it can be used
    as a loss function in PyTorch training.

    Args:
        loss_type: Key used in the config ``loss.training_loss`` or ``loss.validation_loss`` field.

    Example::

        import torch.nn as nn
        from credit.losses import register_loss

        @register_loss("my_loss")
        class MyLoss(nn.Module):
            ...
    """

    def decorator(cls):
        # No lazy import needed here: torch.nn is already imported at the top of this module.
        # isinstance(cls, type) guards against passing an instance or function;
        # issubclass then confirms it inherits nn.Module, which is required for training.
        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            raise TypeError(f"register_loss: '{cls.__name__}' must inherit from torch.nn.Module.")
        if loss_type in _LOSS_REGISTRY:  # warn instead of silently overwriting
            logger.warning(f"register_loss: overwriting existing registry entry for '{loss_type}'")
        _LOSS_REGISTRY[loss_type] = cls  # store the class under the given key
        return cls  # must return the class, otherwise it becomes None after decoration

    return decorator


def _load_loss_entry(loss_type):
    """Return the class for a registered loss type, importing lazily if needed.

    Raises:
        ValueError: If loss_type is not in _LOSS_REGISTRY.
        ImportError: If the loss's module cannot be imported (missing optional dependencies).
    """
    if loss_type not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            f"Available types: {sorted(_LOSS_REGISTRY)}. "
            "Register a custom loss with @register_loss or via custom_objects in your config."
        )
    entry = _LOSS_REGISTRY[loss_type]
    if isinstance(entry, tuple):
        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise ImportError(
                f"Loss type '{loss_type}' requires optional dependencies that are not installed. Original error: {exc}"
            ) from exc
    return entry  # externally registered class stored directly


# ---------------------------------------------------------------------------
# Loss construction & loading
def _instantiate_loss(conf, reduction="mean", validation=False):
    """Look up and instantiate a loss from the registry by name.

    This is the low-level factory used internally by ``load_loss``,
    ``VariableTotalLoss2D``, and ``DownscalingLoss``.  Call ``load_loss``
    instead — it handles weighted/downscaling wrappers and registers custom
    objects from the config before dispatching here.

    Args:
        conf (dict): Configuration dictionary containing loss settings.
        reduction (str, optional): Default reduction method if not specified in parameters.
        validation (bool): Use validation loss settings if True, else training loss.

    Returns:
        torch.nn.Module: Instantiated loss function.

    Raises:
        ValueError: If the requested loss type is not in ``_LOSS_REGISTRY``.
    """
    loss_key = "validation_loss" if validation else "training_loss"
    params_key = "validation_loss_parameters" if validation else "training_loss_parameters"

    loss_type = conf["loss"][loss_key]
    loss_params = conf["loss"].get(params_key, {})

    if "reduction" not in loss_params:
        loss_params["reduction"] = reduction

    mode = "validation" if validation else "train"
    logger.info(f"Loaded the {loss_type} loss function ({mode}) with parameters: {loss_params}")

    if loss_type in _LOSS_REGISTRY:
        return _load_loss_entry(loss_type)(**loss_params)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported")


def load_loss(conf, reduction="none", validation=False):
    """Load the appropriate loss function based on the configuration.

    Determines whether to use a weighted custom loss wrapper
    (such as ``VariableTotalLoss2D``) when latitude or variable weights are
    enabled, or to load a standard or custom loss directly from the registry.

    If in validation mode and a separate validation loss is specified in the
    config, that loss type will be used. Otherwise, the training loss is used.

    Args:
        conf (dict): Configuration dictionary. Must contain a 'loss' section in one of two formats:

                     New-style (Gen 2, mirrors preblocks/postblocks)::
                         loss:
                           type: base            # registry loss name
                           args: {...}           # constructor kwargs

                     Legacy flat keys:
                     - 'training_loss' (str): The primary loss function name.
                     - 'validation_loss' (optional, str): An alternate loss for validation.
                     - 'use_latitude_weights' (bool): Whether to use latitude-based weighting.
                     - 'use_variable_weights' (bool): Whether to use variable-specific weighting.
                     A downscaling config is detected by the presence of ``conf["data"]["datasets"]``
                     (the multi-source dataset key); when present, ``DownscalingLoss`` is returned
                     regardless of the weight flags.
        reduction (str, optional): Reduction method to apply to the loss ('mean', 'sum', or 'none').
                                   Default is 'none'.
        validation (bool, optional): Whether the loss is being used for validation. Defaults to False.

    Returns:
        torch.nn.Module: A loss function instance: ``BaseLoss`` (or another ``type``-dispatched
                         loss) for new-style sections, ``DownscalingLoss`` for downscaling configs,
                         ``VariableTotalLoss2D`` when latitude/variable weights are enabled,
                         or a standard/custom loss from the registry otherwise.

    Raises:
        ValueError: If the requested loss type is not recognized in the registry.
    """
    from credit.registry import (
        load_custom_objects,
    )  # imported here so that importing credit.losses does not automatically load credit.registry
    from credit.losses.gen_1.weighted_loss import VariableTotalLoss2D
    from credit.losses.gen_1.downscaling_loss import DownscalingLoss

    load_custom_objects(conf)  # register any custom classes listed under custom_objects in the config

    loss_conf = conf["loss"]

    mode = "validation" if validation else "train"

    # New-style {type, args} loss section (Gen 2; mirrors the preblocks/
    # postblocks structure). Self-contained: no latitude/variable-weight or
    # downscaling wrapper dispatch applies.
    if "type" in loss_conf:
        loss_type = loss_conf["type"]
        args = dict(loss_conf.get("args") or {})
        if loss_type == "base":
            from credit.datasets.gen_2.channel_utils import ChannelSchema
            from credit.losses.base import BaseLoss

            logger.info("Loaded BaseLoss (%s)", mode)
            return BaseLoss(channel_schema=ChannelSchema.load_or_from_config(conf), validation=validation, **args)
        cls = _load_loss_entry(loss_type)
        if "reduction" in inspect.signature(cls.__init__).parameters:
            args.setdefault("reduction", reduction)
        logger.info(f"Loaded the {loss_type} loss function ({mode}) with parameters: {args}")
        return cls(**args)

    is_downscaling = "datasets" in conf["data"]
    # downscaling could also use_variable_weights, so it needs to come first
    if is_downscaling:
        from credit.losses.gen_1.downscaling_loss import DownscalingLoss

        logger.info("Loaded DownscalingLoss (%s)", mode)
        return DownscalingLoss(conf, validation=validation)

    use_weighted_loss = loss_conf.get("use_latitude_weights", False) or loss_conf.get("use_variable_weights", False)

    if not validation and loss_conf["training_loss"] == "ring-crps" and use_weighted_loss:
        raise ValueError(
            "ring-crps returns a scalar and cannot be combined with "
            "use_latitude_weights / use_variable_weights (VariableTotalLoss2D "
            "needs an elementwise loss). Disable the weights for ring-crps training."
        )

    if use_weighted_loss:
        from credit.losses.gen_1.weighted_loss import VariableTotalLoss2D

        logger.info("Loaded VariableTotalLoss2D (%s)", mode)
        return VariableTotalLoss2D(conf, validation=validation)

    return _instantiate_loss(conf, reduction=reduction, validation=validation)
