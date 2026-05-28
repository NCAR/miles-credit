"""
transforms/__init__.py
----------------------
Factory and lazy-import registry for CREDIT transform classes.

Callers that only need the factory::

    from credit.transforms import load_transforms
    transform = load_transforms(conf)

Callers that import a class directly (backward-compatible)::

    from credit.transforms import Normalize_ERA5_and_Forcing

Modules are imported on first use so optional heavy dependencies (numba,
BridgeScaler, …) are never loaded unless that transform type is actually
requested.  Add a new entry to ``_TRANSFORM_REGISTRY`` to register a new
scaler type without touching ``load_transforms`` itself.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from torchvision import transforms as tforms

logger = logging.getLogger(__name__)

# Maps conf["data"]["scaler_type"] -> {"scaler": ..., "to_tensor": ...}
# Each value is either None (no scaler) or a (module_path, class_name) tuple.
_TRANSFORM_REGISTRY: dict[str, dict[str, Any]] = {
    "std_new": {
        "scaler": ("credit.transforms.transforms_global", "Normalize_ERA5_and_Forcing"),
        "to_tensor": ("credit.transforms.transforms_global", "ToTensor_ERA5_and_Forcing"),
    },
    "std_cached": {
        "scaler": None,
        "to_tensor": ("credit.transforms.transforms_global", "ToTensor_ERA5_and_Forcing"),
    },
    "quantile-cached": {
        "scaler": ("credit.transforms.transforms_quantile", "NormalizeState_Quantile_Bridgescalar"),
        "to_tensor": ("credit.transforms.transforms_quantile", "ToTensor_BridgeScaler"),
    },
    "bridgescaler": {
        "scaler": ("credit.transforms.transforms_quantile", "BridgescalerScaleState"),
        "to_tensor": ("credit.transforms.deprecated._transforms", "ToTensor"),
    },
    "std-les": {
        "scaler": ("credit.transforms.transforms_les", "NormalizeLES"),
        "to_tensor": ("credit.transforms.transforms_les", "ToTensorLES"),
    },
    "std-wrf": {
        "scaler": ("credit.transforms.transforms_wrf", "NormalizeWRF"),
        "to_tensor": ("credit.transforms.transforms_wrf", "ToTensorWRF"),
    },
}

# Backward-compatible lazy re-exports: supports `from credit.transforms import X`
# without eagerly importing every submodule on package load.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ToTensor": ("credit.transforms.deprecated._transforms", "ToTensor"),
    "NormalizeLES": ("credit.transforms.transforms_les", "NormalizeLES"),
    "ToTensorLES": ("credit.transforms.transforms_les", "ToTensorLES"),
    "NormalizeWRF": ("credit.transforms.transforms_wrf", "NormalizeWRF"),
    "ToTensorWRF": ("credit.transforms.transforms_wrf", "ToTensorWRF"),
    "Normalize_ERA5_and_Forcing": ("credit.transforms.transforms_global", "Normalize_ERA5_and_Forcing"),
    "ToTensor_ERA5_and_Forcing": ("credit.transforms.transforms_global", "ToTensor_ERA5_and_Forcing"),
    "BridgescalerScaleState": ("credit.transforms.transforms_quantile", "BridgescalerScaleState"),
    "NormalizeState_Quantile_Bridgescalar": (
        "credit.transforms.transforms_quantile",
        "NormalizeState_Quantile_Bridgescalar",
    ),
    "ToTensor_BridgeScaler": ("credit.transforms.transforms_quantile", "ToTensor_BridgeScaler"),
}


def __getattr__(name: str):
    """Lazily resolve package-level names listed in ``_LAZY_EXPORTS``."""
    if name in _LAZY_EXPORTS:
        module_path, class_name = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_path), class_name)
    raise AttributeError(f"module 'credit.transforms' has no attribute '{name}'")


def load_transforms(conf: dict, scaler_only: bool = False):
    """Load and compose transform objects from a CREDIT config.

    Args:
        conf: Full training/inference config dict.  Reads
            ``conf["data"]["scaler_type"]`` to select the transform pair.
        scaler_only: When True, return only the normalisation/scaler instance
            without the ToTensor step.  Used by postblocks that apply the
            inverse transform at inference time.

    Returns:
        When ``scaler_only=True``: a single scaler instance (or None for
        ``std_cached``).  Otherwise a ``torchvision.transforms.Compose`` of
        ``[scaler, to_tensor]`` (scaler omitted when it is None).

    Raises:
        ValueError: If ``scaler_type`` is not registered in
            ``_TRANSFORM_REGISTRY``.
    """
    scaler_type = conf["data"]["scaler_type"]
    entry = _TRANSFORM_REGISTRY.get(scaler_type)
    if entry is None:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Valid options: {list(_TRANSFORM_REGISTRY)}")

    scaler_spec = entry["scaler"]
    transform_scaler = None
    if scaler_spec is not None:
        module_path, class_name = scaler_spec
        cls = getattr(importlib.import_module(module_path), class_name)
        transform_scaler = cls(conf)

    if scaler_only:
        return transform_scaler

    module_path, class_name = entry["to_tensor"]
    cls = getattr(importlib.import_module(module_path), class_name)
    to_tensor = cls(conf)

    steps = [transform_scaler, to_tensor] if transform_scaler is not None else [to_tensor]
    return tforms.Compose(steps)
