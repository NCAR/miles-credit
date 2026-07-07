import importlib
import logging
import torch
import torch.nn as nn
from credit.preblock.concat import ConcatToTensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config-driven dispatch: maps config-file keys → class (used by build_preblocks).
# Registry entries are either:
#   (module_path: str, class_name: str)  — built-in lazy entries
#   cls: type                             — externally registered classes
_PREBLOCK_REGISTRY = {
    "log_transform": ("credit.preblock.log", "LogTransform"),
    "sqrt_transform": ("credit.preblock.sqrt", "SqrtTransform"),
    "regrid": ("credit.preblock.regrid", "Regridder"),
    "concat": ("credit.preblock.concat", "ConcatToTensor"),
    "era5_normalizer": ("credit.preblock.norm", "ERA5Normalizer"),
    "fill_values": ("credit.preblock.fill_values", "FillValues"),
    "bridgescaler_transform": ("credit.preblock.scaler", "BridgeScalerTransform"),
}

# Direct-import table: maps Python class names → class for lazy module attribute access.
# Enables ``from credit.preblock import LogTransform`` without eager imports; kept for backward compatibility.
_CLASS_SOURCES = {
    "LogTransform": ("credit.preblock.log", "LogTransform"),
    "SqrtTransform": ("credit.preblock.sqrt", "SqrtTransform"),
    "Regridder": ("credit.preblock.regrid", "Regridder"),
    "ConcatToTensor": ("credit.preblock.concat", "ConcatToTensor"),
    "ERA5Normalizer": ("credit.preblock.norm", "ERA5Normalizer"),
    "FillValues": ("credit.preblock.fill_values", "FillValues"),
    "BridgeScalerTransform": ("credit.preblock.scaler", "BridgeScalerTransform"),
}


# ---------------------------------------------------------------------------
# Module __getattr__: called when a name is not found via normal attribute lookup.
# Resolves names listed in _CLASS_SOURCES lazily so submodules are only imported on first access.
# Example: ``from credit.preblock import LogTransform`` triggers __getattr__("LogTransform"),
#          which imports credit.preblock.log on the spot and returns the class.
def __getattr__(name):
    if name in _CLASS_SOURCES:
        module_path, class_name = _CLASS_SOURCES[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise AttributeError(f"Cannot import {name!r}: optional dependencies missing.") from exc
    raise AttributeError(f"module 'credit.preblock' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Registration
def register_preblock(block_type):
    """Decorator that adds an external preblock class to the preblock registry.

    The class must inherit from :class:`credit.preblock.base.BasePreblock` so
    that the correct ``forward(batch: dict) -> dict`` signature is enforced.

    Args:
        block_type: Key used in the config ``preblocks.<phase>.<block>.type`` field.

    Example::

        from credit.preblock import register_preblock
        from credit.preblock.base import BasePreblock

        @register_preblock("my_preblock")
        class MyPreBlock(BasePreblock):
            def forward(self, batch: dict) -> dict:
                ...
    """

    def decorator(cls):
        from credit.preblock.base import BasePreblock  # imported here to avoid loading it at module import time

        # isinstance(cls, type) guards against passing an instance or function;
        # issubclass then confirms it inherits the required base class.
        if not (isinstance(cls, type) and issubclass(cls, BasePreblock)):
            raise TypeError(f"register_preblock: '{cls.__name__}' must inherit from credit.preblock.base.BasePreblock.")
        if block_type in _PREBLOCK_REGISTRY:  # warn instead of silently overwriting
            logger.warning(f"register_preblock: overwriting existing registry entry for '{block_type}'")
        _PREBLOCK_REGISTRY[block_type] = cls  # store the class under the given key
        return cls  # must return the class, otherwise it becomes None after decoration

    return decorator


def _load_preblock_entry(block_type):
    """Return the class for a registered preblock type, importing lazily if needed.

    Raises:
        ValueError: If block_type is not in _PREBLOCK_REGISTRY.
        ImportError: If the preblock's module cannot be imported (missing optional dependencies).
    """
    if block_type not in _PREBLOCK_REGISTRY:
        raise ValueError(
            f"unknown preblock type '{block_type}'. "
            f"Available types: {sorted(_PREBLOCK_REGISTRY)}. "
            "Register a custom preblock with @register_preblock or via custom_objects in your config."
        )
    entry = _PREBLOCK_REGISTRY[block_type]
    if isinstance(entry, tuple):
        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise ImportError(
                f"Preblock type '{block_type}' requires optional dependencies that are not installed. "
                f"Original error: {exc}"
            ) from exc
    return entry  # externally registered class stored directly


# ---------------------------------------------------------------------------
# Building & applying
_VALID_SECTIONS = {"ic_only", "per_step"}


def _build_preblock_section(section_cfg: dict) -> nn.ModuleDict:
    modules = {}
    for name, block_cfg in section_cfg.items():
        block_type = block_cfg["type"]
        modules[name] = _load_preblock_entry(block_type)(**(block_cfg.get("args") or {}))
    return nn.ModuleDict(modules)


def build_preblocks(conf: dict, phase: str = "per_step") -> nn.ModuleDict:
    """Instantiate preblocks for a single phase from the full CREDIT config.

    Config format::

        preblocks:
          ic_only:          # run once at t=0 on the raw batch (e.g. static regrid)
            regrid_static:
              type: regrid
              args: ...
          per_step:         # run every rollout step (e.g. log_transform, concat)
            log_transform:
              type: log_transform
            concat:
              type: concat

    Typical usage — build once per phase, store separately::

        ic_preblocks   = build_preblocks(conf, phase="ic_only")
        step_preblocks = build_preblocks(conf, phase="per_step")

        # t=0: run both in sequence
        ic_preprocessed    = apply_preblocks(ic_preblocks, batch, device=device)
        preprocessed_batch = apply_preblocks(step_preblocks, ic_preprocessed, device=device)

        # t>0: run per_step only
        preprocessed_batch = apply_preblocks(step_preblocks, rollout_batch, device=device)

    Args:
        conf: the full CREDIT config dict.
        phase: which section to build — ``"ic_only"`` or ``"per_step"``.

    Returns:
        ``nn.ModuleDict`` of instantiated blocks for the requested phase.

    Raises:
        ValueError: if the config contains keys other than ``"ic_only"`` / ``"per_step"``,
            if ``phase`` is not one of those values, or if a block's ``type`` value
            is not in ``_PREBLOCK_REGISTRY``.
    """
    from credit.registry import (
        load_custom_objects,
    )  # imported here so that importing credit.preblock does not automatically load credit.registry

    load_custom_objects(conf)  # register any custom classes listed under custom_objects in the config
    cfg = conf.get("preblocks") or {}  # handles both absent key and explicit null in the config
    unknown = set(cfg) - _VALID_SECTIONS
    if unknown:
        raise ValueError(
            f"build_preblocks: unexpected top-level keys {sorted(unknown)}. "
            "Expected only 'ic_only' and/or 'per_step'. "
            "If you are using the old flat preblock format, migrate to the two-section layout."
        )
    if phase not in _VALID_SECTIONS:
        raise ValueError(f"build_preblocks: phase must be one of {sorted(_VALID_SECTIONS)}, got {phase!r}.")
    return _build_preblock_section(cfg.get(phase) or {})


def attach_channel_schema(preblocks: nn.ModuleDict, schema) -> None:
    """Attach a ``ChannelSchema`` to every ConcatToTensor block in a preblock group.

    The schema is runtime state (built from config / loaded from save_loc), not a
    config arg, so it is injected after ``build_preblocks`` rather than through
    the registry. No-op for groups without a concat block or when schema is None.
    """
    if schema is None:
        return
    for block in preblocks.values():
        if isinstance(block, ConcatToTensor):
            block.set_schema(schema)


def _run_preblock_group(group: nn.ModuleDict, batch: dict, device=None):
    """Sequentially applies a group of preblocks, returning the transformed batch."""
    from credit.preblock.concat import ConcatToTensor  # needed for isinstance check

    meta = None
    target = None
    to_device = True
    concat_ran = False

    for preblock in group.values():
        result = preblock(batch)
        if isinstance(result, tuple):
            if len(result) == 3:
                batch, target, meta = result
            else:
                batch, meta = result
        else:
            batch = result
        if isinstance(preblock, ConcatToTensor):
            to_device = preblock.to_device
            concat_ran = True

    if not concat_ran:
        return batch

    out = {"x": batch}
    if meta is not None:
        out["metadata"] = meta
    if target is not None:
        out["y"] = target

    if device is not None and to_device:
        out = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    return out


def _move_batch_to_device(batch, device):
    """Recursively move all tensors in a nested dict to device."""
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def apply_preblocks_before_scaler(preblocks: nn.ModuleDict, batch: dict, device=None):
    from credit.preblock.scaler import BridgeScalerTransform  # needed for isinstance check

    for preblock in preblocks.values():
        if isinstance(preblock, BridgeScalerTransform):
            break
        result = preblock(batch)
        if isinstance(result, tuple):
            if len(result) == 3:
                batch, target, meta = result
            else:
                batch, meta = result
        else:
            batch = result
    if device is not None:
        batch = _move_batch_to_device(batch, device)
    return batch


def apply_preblocks(
    preblocks: nn.ModuleDict,
    batch: dict,
    device=None,
) -> dict:
    """Apply a preblock group built by ``build_preblocks``.

    Args:
        preblocks: ``nn.ModuleDict`` built by ``build_preblocks`` for a single phase.
        batch: nested variable dict from the dataset (or a prior preblock pass).
        device: move output tensors here after concat.

    Returns:
        When concat has run: ``{"x": tensor}`` plus optional ``"y"`` (if a target was
        produced) and ``"metadata"`` (if metadata was produced) keys.
        Otherwise: the transformed nested batch dict (pre-concat).
    """
    return _run_preblock_group(preblocks, batch, device)
