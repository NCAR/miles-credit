import importlib
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config-driven dispatch: maps config-file keys → class (used by build_postblocks).
# Registry entries are either:
#   (module_path: str, class_name: str)  — built-in lazy entries
#   cls: type                             — externally registered classes
_POSTBLOCK_REGISTRY = {
    "reconstruct": ("credit.postblock.reconstruct", "Reconstruct"),
    "bridgescaler_transform": ("credit.postblock.scaler", "BridgeScalerTransform"),
    "exp_transform": ("credit.postblock.exp", "ExpTransform"),
    "square_transform": ("credit.postblock.square", "SquareTransform"),
    "wet_mask_samudra": ("credit.postblock.wet_mask_samudra", "WetMaskBlock"),
    "mslp_diagnostic": ("credit.postblock.mslp", "MSLPDiagnostic"),
    "tracer_fixer": ("credit.postblock.gen1", "TracerFixer"),
    "global_mass_fixer": ("credit.postblock.gen1", "GlobalMassFixer"),
    "global_water_fixer": ("credit.postblock.gen1", "GlobalWaterFixer"),
    "global_energy_fixer": ("credit.postblock.gen1", "GlobalEnergyFixer"),
    "geopotential_diagnostic": ("credit.postblock.geopotential", "GeopotentialDiagnostic"),
    "pressure_interp_diagnostic": ("credit.postblock.pressure_interp", "PressureInterpDiagnostic"),
    "hybrid_level_interp": ("credit.postblock.hybrid_interp", "HybridLevelInterp"),
}

# Direct-import table: maps Python class names → class for lazy module attribute access.
# Enables ``from credit.postblock import Reconstruct`` without eager imports; kept for backward compatibility.
_CLASS_SOURCES = {
    "Reconstruct": ("credit.postblock.reconstruct", "Reconstruct"),
    "BridgeScalerTransform": ("credit.postblock.scaler", "BridgeScalerTransform"),
    "WetMaskBlock": ("credit.postblock.wet_mask_samudra", "WetMaskBlock"),
    "MSLPDiagnostic": ("credit.postblock.mslp", "MSLPDiagnostic"),
    "TracerFixer": ("credit.postblock.gen1", "TracerFixer"),
    "GlobalMassFixer": ("credit.postblock.gen1", "GlobalMassFixer"),
    "GlobalWaterFixer": ("credit.postblock.gen1", "GlobalWaterFixer"),
    "GlobalEnergyFixer": ("credit.postblock.gen1", "GlobalEnergyFixer"),
    "GeopotentialDiagnostic": ("credit.postblock.geopotential", "GeopotentialDiagnostic"),
    "PressureInterpDiagnostic": ("credit.postblock.pressure_interp", "PressureInterpDiagnostic"),
    "HybridLevelInterp": ("credit.postblock.hybrid_interp", "HybridLevelInterp"),
}


# ---------------------------------------------------------------------------
# Module __getattr__: called when a name is not found via normal attribute lookup.
# Resolves names listed in _CLASS_SOURCES lazily so submodules are only imported on first access.
# Example: ``from credit.postblock import Reconstruct`` triggers __getattr__("Reconstruct"),
#          which imports credit.postblock.reconstruct on the spot and returns the class.
def __getattr__(name):
    if name in _CLASS_SOURCES:
        module_path, class_name = _CLASS_SOURCES[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise AttributeError(f"Cannot import {name!r}: optional dependencies missing.") from exc
    raise AttributeError(f"module 'credit.postblock' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Registration
def register_postblock(block_type):
    """Decorator that adds an external postblock class to the postblock registry.

    The class must inherit from :class:`credit.postblock.base.BasePostblock` so
    that the correct ``forward(batch: dict) -> dict`` signature is enforced.

    Args:
        block_type: Key used in the config ``postblocks.<phase>.<block>.type`` field.

    Example::

        from credit.postblock import register_postblock
        from credit.postblock.base import BasePostblock

        @register_postblock("my_postblock")
        class MyPostBlock(BasePostblock):
            def forward(self, batch: dict) -> dict:
                ...
    """

    def decorator(cls):
        from credit.postblock.base import BasePostblock  # imported here to avoid loading it at module import time

        # isinstance(cls, type) guards against passing an instance or function;
        # issubclass then confirms it inherits the required base class.
        if not (isinstance(cls, type) and issubclass(cls, BasePostblock)):
            raise TypeError(
                f"register_postblock: '{cls.__name__}' must inherit from credit.postblock.base.BasePostblock."
            )
        if block_type in _POSTBLOCK_REGISTRY:  # warn instead of silently overwriting
            logger.warning(f"register_postblock: overwriting existing registry entry for '{block_type}'")
        _POSTBLOCK_REGISTRY[block_type] = cls  # store the class under the given key
        return cls  # must return the class, otherwise it becomes None after decoration

    return decorator


def _load_postblock_entry(block_type):
    """Return the class for a registered postblock type, importing lazily if needed.

    Raises:
        ValueError: If block_type is not in _POSTBLOCK_REGISTRY.
        ImportError: If the postblock's module cannot be imported (missing optional dependencies).
    """
    if block_type not in _POSTBLOCK_REGISTRY:
        raise ValueError(
            f"Unknown postblock type '{block_type}'. "
            f"Available types: {sorted(_POSTBLOCK_REGISTRY)}. "
            "Register a custom postblock with @register_postblock or via custom_objects in your config."
        )
    entry = _POSTBLOCK_REGISTRY[block_type]
    if isinstance(entry, tuple):
        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise ImportError(
                f"Postblock type '{block_type}' requires optional dependencies that are not installed. "
                f"Original error: {exc}"
            ) from exc
    return entry  # externally registered class stored directly


# ---------------------------------------------------------------------------
# Building & applying
_VALID_SECTIONS = {"per_step", "post_rollout"}


def _build_postblock_section(section_cfg: dict) -> nn.ModuleDict:
    modules = {}
    for name, block_cfg in section_cfg.items():
        block_type = block_cfg["type"]
        if block_type not in _POSTBLOCK_REGISTRY:
            raise KeyError(
                f"Unknown postblock type {block_type!r} (block name: {name!r}). "
                f"Available types: {sorted(_POSTBLOCK_REGISTRY)}. "
                "Register a custom postblock with @register_postblock or via custom_objects in your config."
            )
        modules[name] = _load_postblock_entry(block_type)(**(block_cfg.get("args") or {}))
    return nn.ModuleDict(modules)


def build_postblocks(conf: dict, phase: str = "per_step") -> nn.ModuleDict:
    """Instantiate postblocks for a single phase from the full CREDIT config.

    Config format::

        postblocks:
          per_step:          # run after every forward pass in the rollout loop
            reconstruct:
              type: reconstruct
            inverse_scale:
              type: bridgescaler_transform
              args:
                method: inverse_transform
                scaler_path: /path/to/scaler.json
          post_rollout:      # run once after all rollout steps complete
            mass_fixer:
              type: global_mass_fixer
              args: ...

    Typical usage — build once per phase, store separately::

        step_postblocks    = build_postblocks(conf, phase="per_step")
        rollout_postblocks = build_postblocks(conf, phase="post_rollout")

        # inside rollout loop, after each forward pass:
        full_data_dict = apply_postblocks(step_postblocks, full_data_dict)

        # once after rollout loop completes:
        apply_postblocks(rollout_postblocks, full_data_dict)

    Args:
        conf: the full CREDIT config dict.
        phase: which section to build — ``"per_step"`` or ``"post_rollout"``.

    Returns:
        ``nn.ModuleDict`` of instantiated blocks for the requested phase.

    Raises:
        ValueError: if the config contains keys other than ``"per_step"`` / ``"post_rollout"``,
            or if ``phase`` is not one of those values.
        KeyError: if a block's ``type`` value is not in ``_POSTBLOCK_REGISTRY``.
    """
    from credit.registry import (
        load_custom_objects,
    )  # imported here so that importing credit.postblock does not automatically load credit.registry

    load_custom_objects(conf)  # register any custom classes listed under custom_objects in the config
    cfg = conf.get("postblocks") or {}  # handles both absent key and explicit null in the config
    unknown = set(cfg) - _VALID_SECTIONS
    if unknown:
        raise ValueError(
            f"build_postblocks: unexpected top-level keys {sorted(unknown)}. "
            "Expected only 'per_step' and/or 'post_rollout'. "
            "If you are using the old flat postblock format, migrate to the two-section layout."
        )
    if phase not in _VALID_SECTIONS:
        raise ValueError(f"build_postblocks: phase must be one of {sorted(_VALID_SECTIONS)}, got {phase!r}.")
    return _build_postblock_section(cfg.get(phase) or {})


def apply_postblocks(postblocks: nn.ModuleDict, batch_dict: dict) -> dict:
    """Apply a postblock group built by ``build_postblocks``.

    Args:
        postblocks: ``nn.ModuleDict`` built by ``build_postblocks`` for a single phase.
        batch_dict: dict containing at minimum ``"y_pred"`` and ``"metadata"``.

    Returns:
        The same ``batch_dict`` after all blocks in the group have run.
    """
    for block in postblocks.values():
        batch_dict = block(batch_dict)
    return batch_dict
