import torch
import torch.nn as nn

from credit.preblock.log import LogTransform
from credit.preblock.sqrt import SqrtTransform
from credit.preblock.regrid import Regridder
from credit.preblock.concat import ConcatToTensor
from credit.preblock.norm import ERA5Normalizer
from credit.preblock.scaler import BridgeScalerTransformer


PREBLOCK_REGISTRY = {
    "log_transform": LogTransform,
    "sqrt_transform": SqrtTransform,
    "regrid": Regridder,
    "concat": ConcatToTensor,
    "era5_normalizer": ERA5Normalizer,
    "bridgescaler_transform": BridgeScalerTransformer,
}


def build_preblocks(preblock_cfg: dict | None = None) -> nn.ModuleDict:
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
            for name, block_cfg in (preblock_cfg or {}).items()
        }
    )


def apply_preblocks(preblocks: nn.ModuleDict, batch: dict, device=None):
    """Sequentially applies transform preblocks (dict→dict).

    Returns a dict with keys:
        "input"  — concatenated tensor (if concat preblock present) or nested batch dict
        "meta"   — metadata dict (only present if concat preblock ran)
        "target" — target tensor (only present if target data was in the batch)

    Tensors in the output are moved to ``device`` unless the concat preblock was
    configured with ``to_device: false``.
    """
    meta = None
    target = None
    to_device = True
    concat_ran = False
    for preblock in preblocks.values():
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

    out = {"input": batch}
    if meta is not None:
        out["metadata"] = meta
    if target is not None:
        out["target"] = target

    if device is not None and to_device:
        out = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    return out


def apply_preblocks_before_scaler(preblocks: nn.ModuleDict, batch: dict, device=None):
    for preblock in preblocks.values():
        if isinstance(preblock, BridgeScalerTransformer):
            break
        result = preblock(batch)
        if isinstance(result, tuple):
            if len(result) == 3:
                batch, target, meta = result
            else:
                batch, meta = result
        else:
            batch = result
    return batch
