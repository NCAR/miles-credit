import torch.nn as nn

from credit.preblock.concat import ConcatPreblock
from credit.preblock.regrid import Regrid
from credit.preblock.scaler import Scaler

__all__ = ["ConcatPreblock", "apply_preblocks", "Regrid", "Scaler"]


def apply_preblocks(preblocks: nn.ModuleDict, batch: dict) -> dict:
    """Sequentially applies all preblocks to a batch dict."""
    for preblock in preblocks.values():
        batch = preblock(batch)
    return batch
