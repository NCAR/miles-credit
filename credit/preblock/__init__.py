import torch.nn as nn
from credit.preblock.log import LogTransform
from credit.preblock.sqrt import SqrtTransform
from credit.preblock.scaler import BridgeScaleTransformer
from credit.preblock.regrid import Regridder

PREBLOCK_REGISTRY = {
    "log_transform": LogTransform,
    "sqrt_transform": SqrtTransform,
    "bridgescaler_transform": BridgeScaleTransformer,
    "regrid": Regridder,
}


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
        {name: PREBLOCK_REGISTRY[block_cfg["type"]](**block_cfg["args"]) for name, block_cfg in preblock_cfg.items()}
    )


def apply_preblocks(preblocks: nn.ModuleDict, batch: dict) -> dict:
    """Sequentially applies all preblocks to a batch dict."""
    for preblock in preblocks.values():
        batch = preblock(batch)
    return batch
