import logging

import torch.nn as nn

from credit.preblock.log import LogTransform
from credit.preblock.sqrt import SqrtTransform
from credit.preblock.regrid import Regridder
from credit.preblock.concat import ConcatToTensor
from credit.preblock.norm import ERA5Normalizer

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
    "era5_normalizer": ERA5Normalizer,
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
    """Sequentially applies transform preblocks (dict→dict), then concatenates to tensors.

    Concatenation is always performed last via ConcatToTensor and is not configurable.
    """
    for preblock in preblocks.values():
        batch = preblock(batch)
    return PREBLOCK_REGISTRY["concat"]()(batch)
