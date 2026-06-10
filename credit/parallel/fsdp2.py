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

    # AC must come before fully_shard so the CheckpointWrapper is what gets sharded.
    if ac_conf:
        _apply_activation_checkpointing(model)

    count = 0
    for module in model.modules():
        if _is_shardable(module, ac_conf):
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


def _has_fsdp2_shard(module: nn.Module) -> bool:
    """Return True if module opted into per-block FSDP2 sharding.

    Any nn.Module subclass can opt in by declaring::

        class MyBlock(nn.Module):
            _fsdp2_shard = True
    """
    return bool(getattr(type(module), "_fsdp2_shard", False))


def _apply_activation_checkpointing(model: nn.Module) -> None:
    """Apply no-reentrant AC to all modules that declared ``_fsdp2_shard = True``.

    Uses apply_activation_checkpointing so replacements happen in-place in parent
    modules. Must be called before fully_shard so the CheckpointWrapper is what
    gets sharded.
    """
    import functools
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=_has_fsdp2_shard,
    )
    logger.info("FSDP2: activation checkpointing applied to _fsdp2_shard blocks")


def _is_shardable(module: nn.Module, ac_enabled: bool) -> bool:
    """Return True if module should receive its own FSDP2 shard.

    With AC enabled, shard_type blocks are wrapped in CheckpointWrapper.
    We shard the CheckpointWrapper, not the inner module — otherwise
    model.modules() visits both and fully_shard is called twice on the
    same parameters, corrupting the mesh state.
    """
    if ac_enabled:
        inner = getattr(module, "_checkpoint_wrapped_module", None)
        return inner is not None and _has_fsdp2_shard(inner)
    return _has_fsdp2_shard(module)


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


def fsdp2_optimizer_state_dict(model: nn.Module, optimizer) -> dict:
    """Gather a full (unsharded) optimizer state dict from an FSDP2 model.

    A raw ``optimizer.state_dict()`` under FSDP2 contains this rank's DTensor
    SHARDS only — saving that from rank 0 silently drops every other rank's
    optimizer state. This gathers the full state on every rank so rank 0 can
    save a checkpoint that survives resume (at any world size).
    """
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        StateDictOptions,
    )

    opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=False)
    return get_optimizer_state_dict(model, optimizer, options=opts)


def fsdp2_load_optimizer_state_dict(model: nn.Module, optimizer, state_dict: dict) -> None:
    """Load a full optimizer state dict into an FSDP2-sharded optimizer.

    Params that never received gradients (unused modules, frozen layers) have
    no saved state: AdamW creates per-param state lazily on the first step
    with a non-None grad. ``set_optimizer_state_dict`` however requires an
    entry for every optimizer param and raises KeyError otherwise (e.g.
    WXFormer's cube_embedding with patch sizes of 1 is allocated but never
    called). Synthesize lazy-init (zero) state for those params — equivalent
    to their pre-first-gradient condition.
    """
    from torch.distributed.checkpoint.state_dict import (
        set_optimizer_state_dict,
        StateDictOptions,
    )

    state = state_dict.get("state", {})
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    patched = []
    for fqn, p in model.named_parameters():
        if id(p) in opt_param_ids and fqn not in state:
            # p.shape on a DTensor is the full logical shape.
            state[fqn] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros(p.shape, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(p.shape, dtype=torch.float32),
            }
            patched.append(fqn)
    if patched:
        logger.info(
            f"FSDP2 optimizer load: synthesized lazy-init state for {len(patched)} stateless params: {patched[:4]}{'...' if len(patched) > 4 else ''}"
        )

    opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict, options=opts)
