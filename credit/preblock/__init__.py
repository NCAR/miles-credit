import logging

import torch.nn as nn

from credit.preblock.log import LogTransform
from credit.preblock.sqrt import SqrtTransform
from credit.preblock.regrid import Regridder
from credit.preblock.concat import ConcatToTensor, ConcatToTensorV1
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
    "concat_v1": ConcatToTensorV1,
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

    If a block with type ``concat_v1`` is present it is stored under the
    reserved key ``_concat_override`` and used by ``apply_preblocks`` in place
    of the default ``ConcatToTensor``.  This is the mechanism for running
    ``rollout_to_netcdf_gen2.py`` against a V1-trained checkpoint.

    Returns:
        nn.ModuleDict of instantiated preblocks, ordered as in config.
    """
    blocks = {}
    for name, block_cfg in preblock_cfg.items():
        block_type = block_cfg["type"]
        if block_type == "concat_v1":
            blocks["_concat_override"] = PREBLOCK_REGISTRY["concat_v1"](**(block_cfg.get("args") or {}))
        else:
            blocks[name] = PREBLOCK_REGISTRY[block_type](**(block_cfg.get("args") or {}))
    return nn.ModuleDict(blocks)


def apply_preblocks(preblocks: nn.ModuleDict, batch: dict):
    """Sequentially applies transform preblocks (dict→dict).

    Skips the reserved ``_concat_override`` key — that entry is used by callers
    (e.g. rollout_to_netcdf_gen2) to select the right ConcatToTensor variant;
    it is not itself a transform preblock.
    """
    for name, preblock in preblocks.items():
        if name == "_concat_override":
            continue
        batch = preblock(batch)
    return batch
