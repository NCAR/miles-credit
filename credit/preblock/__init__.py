import logging

import torch
import torch.nn as nn

from credit.preblock.log import LogTransform
from credit.preblock.sqrt import SqrtTransform
from credit.preblock.regrid import Regridder
from credit.preblock.concat import ConcatToTensor
from credit.preblock.norm import ERA5Normalizer

# bridgescaler is an optional dependency; guard against environments without it
try:
    from credit.preblock.scaler import BridgeScalerTransformer

    _BRIDGESCALER_AVAILABLE = True
except (ImportError, Exception) as _e:
    logging.warning(f"BridgeScalerTransformer unavailable (numba/NumPy conflict): {_e}")
    BridgeScalerTransformer = None
    _BRIDGESCALER_AVAILABLE = False

PREBLOCK_REGISTRY = {
    "log_transform": LogTransform,
    "sqrt_transform": SqrtTransform,
    "regrid": Regridder,
    "concat": ConcatToTensor,
    "era5_normalizer": ERA5Normalizer,
}

if _BRIDGESCALER_AVAILABLE:
    PREBLOCK_REGISTRY["bridgescaler_transform"] = BridgeScalerTransformer

_VALID_SECTIONS = {"ic_only", "per_step"}


def _build_preblock_section(section_cfg: dict) -> nn.ModuleDict:
    modules = {}
    for name, block_cfg in section_cfg.items():
        modules[name] = PREBLOCK_REGISTRY[block_cfg["type"]](**(block_cfg.get("args") or {}))
    return nn.ModuleDict(modules)


def build_preblocks(preblock_cfg: dict | None = None, phase: str = "per_step") -> nn.ModuleDict:
    """Instantiate preblocks for a single phase from a two-section config.

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

        ic_preblocks   = build_preblocks(cfg, phase="ic_only")
        step_preblocks = build_preblocks(cfg, phase="per_step")

        # t=0: run both in sequence
        ic_preprocessed    = apply_preblocks(ic_preblocks, batch, device=device)
        preprocessed_batch = apply_preblocks(step_preblocks, ic_preprocessed, device=device)

        # t>0: run per_step only
        preprocessed_batch = apply_preblocks(step_preblocks, rollout_batch, device=device)

    Args:
        preblock_cfg: the full ``preblocks`` config dict (both sections).
        phase: which section to build — ``"ic_only"`` or ``"per_step"``.

    Returns:
        ``nn.ModuleDict`` of instantiated blocks for the requested phase.

    Raises:
        ValueError: if the config contains keys other than ``"ic_only"`` / ``"per_step"``,
            or if ``phase`` is not one of those values.
    """
    cfg = preblock_cfg or {}
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


def _run_preblock_group(group: nn.ModuleDict, batch: dict, device=None):
    """Sequentially applies a group of preblocks, returning the transformed batch."""
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
        When concat has run: ``{"x": tensor, "y": tensor, "metadata": ...}``.
        Otherwise: the transformed nested batch dict (pre-concat).
    """
    return _run_preblock_group(preblocks, batch, device)
