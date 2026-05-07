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


def apply_postblocks(postblocks: nn.ModuleDict, batch_dict: dict) -> dict:
    """Applies all postblocks sequentially on a shared batch dict.

    The caller is responsible for adding ``"prediction"`` (flat model output
    tensor) to ``batch_dict`` before calling this function. Any additional data
    needed by postblocks (e.g. ``"_raw"`` for a pre-transform batch) should
    also be added to ``batch_dict`` by the caller beforehand.

    Args:
        postblocks:  ``nn.ModuleDict`` built by ``build_postblocks``.
        batch_dict:  Dict containing at minimum ``"prediction"`` and ``"meta"``.

    Returns:
        The same ``batch_dict`` after all postblocks have run. ``"prediction"``
        will be a nested variable dict after ``Reconstruct`` runs.
    """
    blocks = list(postblocks.values())
    if not blocks:
        raise RuntimeError(
            'No postblocks configured. "postblocks" must be present in the config, '
            'with "reconstruct" as the first postblock.'
        )
    if not isinstance(blocks[0], Reconstruct):
        raise RuntimeError(
            '"reconstruct" must be present as the first postblock '
            "to reconstruct the output tensor into a named variable dict."
        )

    for block in blocks:
        batch_dict = block(batch_dict)

    return batch_dict
