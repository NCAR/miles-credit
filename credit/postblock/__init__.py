import torch
import torch.nn as nn

from credit.postblock.reconstruct import Reconstruct
from credit.postblock.wet_mask_samudra import WetMaskBlock
from credit.postblock.scaler import BridgeScalerTransformer


POSTBLOCK_REGISTRY = {
    "reconstruct": Reconstruct,
    "bridgescaler_transform": BridgeScalerTransformer,
    "wet_mask_samudra": WetMaskBlock,
}


def build_postblocks(postblock_cfg: dict | None = None) -> nn.ModuleDict:
    """Instantiates all postblocks from the config's ``postblocks`` section.

    ``per_step`` defaults to ``True`` for every block and can be set per-block
    in the config as a signal to the trainer about call timing.

    Args:
        postblock_cfg: the full postblocks dict from the config, e.g.::

            postblocks:
              inverse_scale:
                type: bridgescaler_transform
                args:
                  method: inverse_transform
                  scaler_path: /path/to/scaler.json
                  per_step: false
              mass_fixer:
                type: global_mass_fixer
                args: ...

    Returns:
        ``nn.ModuleDict`` of instantiated postblocks, ordered as in config.
        ``Reconstruct`` is NOT included here — it is hardcoded in
        ``apply_postblocks``.
    """
    modules = {}
    for name, block_cfg in (postblock_cfg or {}).items():
        block_type = block_cfg["type"]
        args = block_cfg.get("args") or {}
        instance = POSTBLOCK_REGISTRY[block_type](**args)
        instance.per_step = block_cfg.get("per_step", False)
        modules[name] = instance
    return nn.ModuleDict(modules)


def apply_postblocks(postblocks: nn.ModuleDict, y_pred: torch.Tensor, metadata: dict) -> dict:
    """Reconstructs ``y_pred`` into a variable dict, then applies all postblocks.

    ``Reconstruct`` is always run first (hardcoded via the registry).
    All registered postblocks then run sequentially on the resulting dict.

    Args:
        postblocks: ``nn.ModuleDict`` built by ``build_postblocks``.
        y_pred:     Flat model output tensor, shape ``(B, C, H, W)``.
        metadata:   Metadata dict from ``apply_preblocks``, must contain
                    ``metadata["_channel_map"]["output"]``.

    Returns:
        Dict with keys ``"prediction"`` and ``"metadata"``, possibly further
        transformed by registered postblocks.
    """
    batch = POSTBLOCK_REGISTRY["reconstruct"]()(y_pred, metadata)

    for postblock in postblocks.values():
        batch = postblock(batch)

    return batch
