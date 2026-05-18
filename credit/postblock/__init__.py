import torch.nn as nn

from credit.postblock.reconstruct import Reconstruct
from credit.postblock.wet_mask_samudra import WetMaskBlock
from credit.postblock.scaler import BridgeScalerTransformer
from credit.postblock.mslp import MSLPDiagnostic
from credit.postblock.mslp import mslp_from_surface_pressure as mslp_from_surface_pressure


POSTBLOCK_REGISTRY = {
    "reconstruct": Reconstruct,
    "bridgescaler_transform": BridgeScalerTransformer,
    "wet_mask_samudra": WetMaskBlock,
    "mslp_diagnostic": MSLPDiagnostic,
}


def build_postblocks(postblock_cfg: dict | None = None) -> nn.ModuleDict:
    """Instantiates all postblocks from the config's ``postblocks`` section.

    ``per_step`` defaults to ``False`` for every block and can be set per-block
    in the config as a signal to the trainer about call timing.

    Args:
        postblock_cfg: the full postblocks dict from the config, e.g.::

            postblocks:
              reconstruct:
                type: reconstruct
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
    tensor) and ``"meta"`` (metadata from ``apply_preblocks``) to ``batch_dict``
    before calling. Any additional data needed by postblocks (e.g. ``"input"``,
    ``"target"``, ``"_raw"``) should also be added by the caller beforehand.

    ``Reconstruct`` must be the first registered postblock — it converts
    ``batch_dict["prediction"]`` from a flat tensor into a nested variable dict.
    Subsequent postblocks operate on that nested dict via their ``key=`` arg.

    Args:
        postblocks:  ``nn.ModuleDict`` built by ``build_postblocks``.
        batch_dict:  Dict containing at minimum ``"prediction"`` (flat tensor)
                     and ``"meta"`` (with ``_channel_map["output"]``).

    Returns:
        The same ``batch_dict`` after all postblocks have run. ``"prediction"``
        will be a nested variable dict after ``Reconstruct`` runs.

    Raises:
        RuntimeError: if ``postblocks`` is empty or ``Reconstruct`` is not first.
    """
    blocks = list(postblocks.values())

    for block in blocks:
        batch_dict = block(batch_dict)

    return batch_dict
