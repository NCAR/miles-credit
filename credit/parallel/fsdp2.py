"""FSDP2 wrapping for CREDIT v2 models.

Uses torch.distributed._composable.fsdp.fully_shard (FSDP2) instead of
the legacy FullyShardedDataParallel (FSDP1). FSDP2 is composable with TP
and does not require a module wrapper — parameters are sharded in-place as
DTensors.

Sharding granularity for WXFormer v2:
  - Each Transformer encoder block  (one per depth layer per level)
  - Each UpBlock / UpBlockPS decoder block
  - The full model (outermost shard)

Checkpoint I/O:
  - Use torch.distributed.checkpoint.state_dict.get_model_state_dict /
    set_model_state_dict (DCP) rather than torch.save/load.
  - A helper is provided: fsdp2_state_dict / fsdp2_load_state_dict.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_fsdp2(model: nn.Module, dp_mesh, conf: dict) -> nn.Module:
    """Apply FSDP2 to model using the data-parallel submesh.

    Shards Transformer and UpBlock/UpBlockPS submodules first, then
    wraps the whole model.

    Args:
        model: Raw (or TP-converted) model.
        dp_mesh: 1-D DeviceMesh for the data-parallel dimension.
            Pass None to shard over the default global mesh.
        conf: Full training config dict (reads trainer.amp for mp_policy).

    Returns:
        model with fully_shard applied (in-place, returns same object).
    """
    from torch.distributed._composable.fsdp import fully_shard

    mp_policy = _build_mp_policy(conf)

    kwargs = {}
    if dp_mesh is not None:
        kwargs["mesh"] = dp_mesh
    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy

    ac_conf = conf.get("trainer", {}).get("activation_checkpoint", False)

    # Import WXFormer-specific block types — guarded so this works even if
    # wxformer_v2 is not installed.
    shard_types = _get_shard_types()

    count = 0
    for module in model.modules():
        if shard_types and isinstance(module, tuple(shard_types)):
            if ac_conf:
                _apply_activation_checkpoint(module)
            fully_shard(module, **kwargs)
            count += 1

    # Outermost shard
    fully_shard(model, **kwargs)
    logger.info(f"FSDP2: sharded {count} submodules + top-level model")

    # SpectralNorm registers weight_u / weight_v as fp32 buffers but FSDP2
    # casts weight_orig (the parameter) to param_dtype. The power-iteration
    # hook then fails with a dtype mismatch when computing torch.mv(weight_mat, v).
    # Fix: cast all spectral-norm buffers to match param_dtype after sharding.
    if mp_policy is not None:
        _fix_spectral_norm_dtype(model, mp_policy.param_dtype)

    return model


def _fix_spectral_norm_dtype(model: nn.Module, param_dtype: torch.dtype) -> None:
    """Cast spectral norm u/v buffers to match the FSDP2 parameter dtype.

    SpectralNorm registers `weight_u` and `weight_v` as fp32 buffers.
    When FSDP2 casts `weight_orig` to bfloat16, the power-iteration
    `torch.mv(weight_mat, v)` fails with a dtype mismatch.
    This walks all modules and casts matching buffers in-place.
    """
    sn_buffer_names = {"weight_u", "weight_v"}
    count = 0
    for module in model.modules():
        for buf_name in list(module._buffers):
            if buf_name in sn_buffer_names and module._buffers[buf_name] is not None:
                module._buffers[buf_name] = module._buffers[buf_name].to(param_dtype)
                count += 1
    if count:
        logger.info(f"SpectralNorm: cast {count} u/v buffers to {param_dtype}")


def _build_mp_policy(conf: dict):
    """Build FSDP2 MixedPrecision policy from config.

    SpectralNorm registers weight_orig as a parameter and performs power
    iteration using fp32 u/v buffers. Any MixedPrecisionPolicy that casts
    parameters or inputs creates a dtype conflict inside torch.autocast
    (at::autocast::prioritize fails on mixed fp32/bf16 tensors in torch.mv).

    Strategy:
    - use_spectral_norm=True: return None (no policy). FSDP2 sharding still
      provides memory savings; the trainer disables manual autocast for fsdp2
      mode, so all compute runs in fp32. Override via fsdp2_mp_policy to opt in.
    - use_spectral_norm=False: use bfloat16 MixedPrecisionPolicy (full AMP).
    """
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy

    trainer = conf.get("trainer", {})
    if not trainer.get("amp", False):
        return None

    mp_conf = trainer.get("fsdp2_mp_policy", {})
    has_spectral_norm = conf.get("model", {}).get("use_spectral_norm", False)

    # If spectral norm is present and user hasn't explicitly overridden, skip
    # MixedPrecisionPolicy entirely. fp32 compute + sharding for memory.
    if has_spectral_norm and not mp_conf:
        logger.info(
            "SpectralNorm detected: MixedPrecisionPolicy disabled for FSDP2 "
            "(fp32 compute with sharding). Set trainer.fsdp2_mp_policy to override."
        )
        return None

    param_dtype = _parse_dtype(mp_conf.get("param_dtype", "bfloat16"))
    reduce_dtype = _parse_dtype(mp_conf.get("reduce_dtype", "float32"))
    output_dtype = _parse_dtype(mp_conf.get("output_dtype", "bfloat16"))

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=True,
    )


def _parse_dtype(s: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if s not in mapping:
        raise ValueError(f"Unknown dtype '{s}'. Choose from {list(mapping)}")
    return mapping[s]


def _get_shard_types():
    """Return model block types that should get their own FSDP2 shard."""
    types = []
    try:
        from credit.models.wxformer.wxformer_v2 import Transformer, UpBlock, UpBlockPS

        types.extend([Transformer, UpBlock, UpBlockPS])
    except ImportError:
        pass
    try:
        from credit.models.crossformer.crossformer import Transformer as V1Transformer

        types.append(V1Transformer)
    except ImportError:
        pass
    return types


def _apply_activation_checkpoint(module: nn.Module) -> None:
    """Apply no-reentrant activation checkpointing to a module."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
    )

    try:
        checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    except Exception as e:
        logger.warning(f"Activation checkpoint failed for {type(module).__name__}: {e}")


# ---------------------------------------------------------------------------
# Checkpoint helpers for FSDP2
# ---------------------------------------------------------------------------


def fsdp2_state_dict(model: nn.Module) -> dict:
    """Gather a full (unsharded) state dict from an FSDP2 model on all ranks.

    Returns the full state dict on every rank so rank 0 can save it.
    """
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        StateDictOptions,
    )

    opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=False)
    return get_model_state_dict(model, options=opts)


def fsdp2_load_state_dict(model: nn.Module, state_dict: dict) -> None:
    """Load a full state dict into an FSDP2 model."""
    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict,
        StateDictOptions,
    )

    opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    set_model_state_dict(model, state_dict, options=opts)
