import torch.nn as nn

from credit.postblock.reconstruct import Reconstruct
from credit.postblock.wet_mask_samudra import WetMaskBlock
from credit.postblock.scaler import BridgeScalerTransformer
from credit.postblock.tracer_fixer import TracerFixer
from credit.postblock.mass_fixer import GlobalMassFixer
from credit.postblock.water_fixer import GlobalWaterFixer
from credit.postblock.energy_fixer import GlobalEnergyFixer


POSTBLOCK_REGISTRY = {
    "reconstruct": Reconstruct,
    "bridgescaler_transform": BridgeScalerTransformer,
    "wet_mask_samudra": WetMaskBlock,
    "tracer_fixer": TracerFixer,
    "global_mass_fixer": GlobalMassFixer,
    "global_water_fixer": GlobalWaterFixer,
    "global_energy_fixer": GlobalEnergyFixer,
}

_VALID_SECTIONS = {"per_step", "post_rollout"}


def _build_postblock_section(section_cfg: dict) -> nn.ModuleDict:
    modules = {}
    for name, block_cfg in section_cfg.items():
        modules[name] = POSTBLOCK_REGISTRY[block_cfg["type"]](**(block_cfg.get("args") or {}))
    return nn.ModuleDict(modules)


def build_postblocks(postblock_cfg: dict | None = None, phase: str = "per_step") -> nn.ModuleDict:
    """Instantiate postblocks for a single phase from a two-section config.

    Config format::

        postblocks:
          per_step:          # run after every forward pass in the rollout loop
            reconstruct:
              type: reconstruct
            inverse_scale:
              type: bridgescaler_transform
              args:
                method: inverse_transform
                scaler_path: /path/to/scaler.json
          post_rollout:      # run once after all rollout steps complete
            mass_fixer:
              type: global_mass_fixer
              args: ...

    Typical usage — build once per phase, store separately::

        step_postblocks    = build_postblocks(cfg, phase="per_step")
        rollout_postblocks = build_postblocks(cfg, phase="post_rollout")

        # inside rollout loop, after each forward pass:
        full_data_dict = apply_postblocks(step_postblocks, full_data_dict)

        # once after rollout loop completes:
        apply_postblocks(rollout_postblocks, full_data_dict)

    Args:
        postblock_cfg: the full ``postblocks`` config dict (both sections).
        phase: which section to build — ``"per_step"`` or ``"post_rollout"``.

    Returns:
        ``nn.ModuleDict`` of instantiated blocks for the requested phase.

    Raises:
        ValueError: if the config contains keys other than ``"per_step"`` / ``"post_rollout"``,
            or if ``phase`` is not one of those values.
    """
    cfg = postblock_cfg or {}
    unknown = set(cfg) - _VALID_SECTIONS
    if unknown:
        raise ValueError(
            f"build_postblocks: unexpected top-level keys {sorted(unknown)}. "
            "Expected only 'per_step' and/or 'post_rollout'. "
            "If you are using the old flat postblock format, migrate to the two-section layout."
        )
    if phase not in _VALID_SECTIONS:
        raise ValueError(f"build_postblocks: phase must be one of {sorted(_VALID_SECTIONS)}, got {phase!r}.")
    return _build_postblock_section(cfg.get(phase) or {})


def apply_postblocks(postblocks: nn.ModuleDict, batch_dict: dict) -> dict:
    """Apply a postblock group built by ``build_postblocks``.

    Args:
        postblocks: ``nn.ModuleDict`` built by ``build_postblocks`` for a single phase.
        batch_dict: dict containing at minimum ``"predicted"`` and ``"metadata"``.

    Returns:
        The same ``batch_dict`` after all blocks in the group have run.
    """
    for block in postblocks.values():
        batch_dict = block(batch_dict)
    return batch_dict
