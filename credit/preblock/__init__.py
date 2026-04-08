import logging

import torch.nn as nn

from credit.preblock.log import LogTransform
from credit.preblock.sqrt import SqrtTransform
from credit.preblock.regrid import Regridder
from credit.preblock.concat import ConcatPreblock as ConcatPreblock
from credit.preblock.concat import ConcatToTensor

# bridgescaler depends on numba which requires NumPy ≤ 2.2
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
}

if _BRIDGESCALER_AVAILABLE:
    PREBLOCK_REGISTRY["bridgescaler_transform"] = BridgeScalerTransformer


def build_preblocks(preblock_cfg: dict) -> nn.ModuleDict:
    """
    Instantiates all preblocks from the config's 'preblocks' section.

    Args:
        preblock_cfg: the full preblocks dict from the config, e.g.:
            {
                'era5_log_transform': {'type': 'log_transform', 'args': {...}},
                'era5_z_transform':   {'type': 'z_transform',   'args': {...}},
            }

    Returns:
        nn.ModuleDict of instantiated preblocks, ordered as in config.
    """
    return nn.ModuleDict(
        {
            name: PREBLOCK_REGISTRY[block_cfg["type"]](**(block_cfg.get("args") or {}))
            for name, block_cfg in preblock_cfg.items()
        }
    )


def apply_preblocks(preblocks: nn.ModuleDict, batch: dict):
    """Sequentially applies all preblocks, then returns (x, y, metadata) tuple.

    Supports two terminal preblock styles:
    - ``ConcatPreblock`` (Gen2): adds ``batch["x"]`` and ``batch["y"]`` to the dict,
      then apply_preblocks extracts them and returns a tuple for interface consistency.
    - ``ConcatToTensor`` (Gen1/legacy): called automatically if no Gen2 preblock set "x".

    Returns:
        (x_tensor, y_tensor, metadata_dict)
    """
    for preblock in preblocks.values():
        result = preblock(batch)
        if isinstance(result, dict):
            batch = result
        else:
            # Legacy: a preblock returned a tuple directly (e.g. ConcatToTensor used standalone)
            return result

    # If ConcatPreblock already assembled x/y into the batch dict, extract them
    if "x" in batch:
        return batch["x"], batch.get("y", None), batch.get("metadata", {})

    # Fallback: Gen1 path — run ConcatToTensor as before
    return PREBLOCK_REGISTRY["concat"]()(batch)
