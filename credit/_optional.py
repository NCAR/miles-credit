"""Helpers for importing optional third-party dependencies with an actionable error message.

``torch-harmonics`` and ``segmentation-models-pytorch`` are not core dependencies of
``miles-credit``: they are only needed by SKEBS, spherical-harmonic noise/filters, and spectral
verification. Code that needs them calls :func:`require` at the point of use, so importing a model
or postblock never pulls them in.
"""

import importlib
from types import ModuleType

#: Optional modules mapped to the ``miles-credit`` extra that installs them.
OPTIONAL_EXTRAS = {
    "torch_harmonics": "spectral",
    "segmentation_models_pytorch": "spectral",
}


def require(module_name: str, feature: str) -> ModuleType:
    """Import an optional dependency, or raise an ImportError that says how to install it.

    Args:
        module_name: Importable module name, e.g. ``"torch_harmonics"``.
        feature: Short description of what needs the module, used in the error message.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as err:
        extra = OPTIONAL_EXTRAS.get(module_name)
        hint = f"pip install 'miles-credit[{extra}]'" if extra else f"pip install {module_name}"
        raise ImportError(
            f"{feature} requires the optional dependency '{module_name}', which could not be imported "
            f"({err}). Install it with: {hint}"
        ) from err


def is_available(module_name: str) -> bool:
    """Return True if ``module_name`` can be imported."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True
