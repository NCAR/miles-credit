"""Model conversion for domain parallelism.

Walks a model's module tree and replaces layers that need inter-GPU
communication with domain-parallel equivalents. Local operations
(1x1 convolutions, channel-wise norms, activations, etc.) are unchanged.
"""

import logging
import torch.nn as nn

from credit.domain_parallel.layers import (
    DomainParallelConv2d,
    DomainParallelConv3d,
    DomainParallelConvTranspose2d,
    DomainParallelConvTranspose3d,
    DomainParallelGroupNorm,
)
from credit.domain_parallel.manager import DomainParallelManager

logger = logging.getLogger(__name__)


def _needs_halo_conv2d(conv):
    """Check if a Conv2d needs halo exchange (kernel > 1 along H)."""
    if isinstance(conv.kernel_size, int):
        k_h = conv.kernel_size
    else:
        k_h = conv.kernel_size[0]
    return k_h > 1


def _needs_halo_conv3d(conv):
    """Check if a Conv3d needs halo exchange (kernel > 1 along H dim)."""
    if isinstance(conv.kernel_size, int):
        k_h = conv.kernel_size
    else:
        k_h = conv.kernel_size[1]  # (T, H, W) -> H is index 1
    return k_h > 1


def _needs_halo_conv_transpose3d(conv):
    """Check if a ConvTranspose3d needs halo exchange (kernel > stride along H)."""
    if isinstance(conv.kernel_size, int):
        k_h = conv.kernel_size
    else:
        k_h = conv.kernel_size[1]  # H in (Kz, Kh, Kw)

    if isinstance(conv.stride, int):
        s_h = conv.stride
    else:
        s_h = conv.stride[1]

    return k_h > s_h


def _needs_halo_conv_transpose2d(conv):
    """Check if a ConvTranspose2d needs halo exchange."""
    if isinstance(conv.kernel_size, int):
        k_h = conv.kernel_size
    else:
        k_h = conv.kernel_size[0]

    if isinstance(conv.stride, int):
        s_h = conv.stride
    else:
        s_h = conv.stride[0]

    # kernel > stride means overlap, needs halo
    return k_h > s_h


def _replace_module(parent, name, old_module, new_module):
    """Replace a child module in the parent."""
    setattr(parent, name, new_module)


def convert_to_domain_parallel(model, manager, shard_dim=-2, custom_converters=None):
    """Convert a model to use domain-parallel layers.

    Walks the module tree and replaces:
    - nn.Conv2d (kernel>1 in H) -> DomainParallelConv2d
    - nn.Conv3d (kernel>1 in H) -> DomainParallelConv3d
    - nn.ConvTranspose2d (kernel>stride in H) -> DomainParallelConvTranspose2d
    - nn.ConvTranspose3d (kernel>stride in H) -> DomainParallelConvTranspose3d
    - nn.GroupNorm -> DomainParallelGroupNorm

    Leaves unchanged:
    - 1x1 Conv2d (FeedForward, projections)
    - Custom LayerNorm (channel-wise, no spatial reduction)
    - Attention modules (windowed, local within shard)
    - PixelShuffle, activations, Dropout

    For custom module types that wrap a Conv2d internally (e.g. PeriodicConv2d),
    pass a custom_converters dict so the whole wrapper is replaced at once and
    its inner Conv2d children are not double-processed:

        from credit.models.unet_diffusion import PeriodicConv2d
        from credit.domain_parallel.layers import DomainParallelPeriodicConv2d

        convert_to_domain_parallel(
            model, manager,
            custom_converters={
                PeriodicConv2d: lambda m: DomainParallelPeriodicConv2d(m),
            }
        )

    Args:
        model: The nn.Module to convert.
        manager: DomainParallelManager instance.
        shard_dim: Spatial dimension being sharded (-2 for H in BCHW).
        custom_converters: Optional dict mapping module type -> callable(module)
            returning the replacement module. Custom types are checked before
            the built-in rules, and their children are skipped.

    Returns:
        The model with replaced layers (modified in-place).
    """
    counts = {
        "conv2d": 0, "conv3d": 0,
        "conv_transpose2d": 0, "conv_transpose3d": 0,
        "group_norm": 0, "custom": 0,
    }

    # For Conv3d, the shard_dim in 5D tensor is different
    shard_dim_5d = 3  # H in (B, C, T, H, W)

    custom_converters = custom_converters or {}

    # Track IDs of modules that have already been scheduled for replacement
    # so we skip their children (avoids double-processing inner convs of
    # custom wrappers like PeriodicConv2d).
    replaced_ids = set()

    # We need to iterate over a list of (parent, name, module) because
    # we're modifying the tree during iteration
    replacements = []

    for _parent_name, parent_module in model.named_modules():
        if id(parent_module) in replaced_ids:
            continue

        for name, module in parent_module.named_children():
            if id(module) in replaced_ids:
                continue

            # --- Custom converters first ---
            matched_custom = False
            for custom_type, factory in custom_converters.items():
                if isinstance(module, custom_type):
                    replacements.append((parent_module, name, factory(module)))
                    counts["custom"] += 1
                    # Mark all descendants so they are not also replaced
                    for _, desc in module.named_modules():
                        replaced_ids.add(id(desc))
                    matched_custom = True
                    break

            if matched_custom:
                continue

            # --- Built-in rules ---
            if isinstance(module, nn.Conv2d) and _needs_halo_conv2d(module):
                replacements.append(
                    (parent_module, name, DomainParallelConv2d(module, shard_dim=shard_dim))
                )
                counts["conv2d"] += 1

            elif isinstance(module, nn.Conv3d) and _needs_halo_conv3d(module):
                replacements.append(
                    (parent_module, name, DomainParallelConv3d(module, shard_dim=shard_dim_5d))
                )
                counts["conv3d"] += 1

            elif isinstance(module, nn.ConvTranspose3d) and _needs_halo_conv_transpose3d(module):
                replacements.append(
                    (
                        parent_module,
                        name,
                        DomainParallelConvTranspose3d(module, shard_dim=shard_dim_5d),
                    )
                )
                counts["conv_transpose3d"] += 1

            elif isinstance(module, nn.ConvTranspose2d) and _needs_halo_conv_transpose2d(module):
                replacements.append(
                    (
                        parent_module,
                        name,
                        DomainParallelConvTranspose2d(module, shard_dim=shard_dim),
                    )
                )
                counts["conv_transpose2d"] += 1

            elif isinstance(module, nn.GroupNorm):
                replacements.append(
                    (parent_module, name, DomainParallelGroupNorm(module))
                )
                counts["group_norm"] += 1

    # Apply replacements
    for parent, name, new_module in replacements:
        _replace_module(parent, name, None, new_module)

    logger.info(
        f"Domain-parallel conversion: replaced {counts['conv2d']} Conv2d, "
        f"{counts['conv3d']} Conv3d, {counts['conv_transpose2d']} ConvTranspose2d, "
        f"{counts['conv_transpose3d']} ConvTranspose3d, "
        f"{counts['group_norm']} GroupNorm, "
        f"{counts['custom']} custom layers"
    )

    return model


def validate_sharding_constraints(model, local_h, window_sizes=None):
    """Validate that the local spatial dimension is compatible with the model.

    Checks that local_h is divisible by all window sizes at all encoder levels,
    accounting for stride-based downsampling.

    Args:
        model: The model to validate against.
        local_h: Local H dimension per domain-parallel rank.
        window_sizes: List of window sizes at each encoder level.
            If None, attempts to extract from the model.

    Returns:
        List of warning messages (empty if all checks pass).
    """
    warnings = []

    if window_sizes is not None:
        h = local_h
        for i, ws in enumerate(window_sizes):
            ws_h = ws[0] if isinstance(ws, (tuple, list)) else ws
            if h % ws_h != 0:
                warnings.append(
                    f"Level {i}: local H={h} not divisible by window_size={ws_h}. "
                    f"Attention will fail. Consider adjusting domain_parallel_size or padding."
                )
            h = h // 2  # assumes stride=2 at each level

    return warnings
