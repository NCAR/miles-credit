"""Tensor Parallelism for CREDIT v2 models.

Two implementations live here:

1. Native DTensor TP (``apply_native_tensor_parallel``) — torch's
   ``parallelize_module`` with Colwise/RowwiseParallel. Blocks opt in by
   declaring a ``_tp_plan`` dict (see wxformer_next). This is the supported
   path (issue #415): params keep FQNs and logical shapes, checkpointing
   comes free through the DCP full-state APIs, and ``fully_shard`` composes
   on top over a 2D (dp, tp) mesh.

2. Legacy hand-rolled module swapping (``apply_tensor_parallel`` and the
   Tp*Conv2d/Tp*Linear wrappers) — DISABLED: its contiguous column slicing
   scrambles fused qkv boundaries and the backward all-reduce at the
   column-parallel input is missing. The entry point raises; the layer
   classes remain only as reference and for their unit tests.

Legacy opt-in protocol (unused while disabled):

    class MyBlock(nn.Module):
        _tp_col = "proj_up"   # attribute path for the column-parallel layer
        _tp_row = "proj_out"  # attribute path for the row-parallel layer

Paths may be dotted (e.g. ``"layers.1"``) to address layers nested inside a
Sequential or other container.

The column-parallel layer receives the full input and produces a sharded output
(no all_reduce). The row-parallel layer receives the sharded input and issues an
all_reduce SUM so the rest of the graph sees the full output.

Supported layer types: ``nn.Conv2d`` (kernel 1×1 only) and ``nn.Linear``.

Gradient flow:
  - TpColConv2d / TpColLinear: no all_reduce (output is sharded,
    consumed by the paired Row layer).
  - TpRowConv2d / TpRowLinear: all_reduce SUM over tp_group before returning.

Usage:
    from credit.parallel import apply_tensor_parallel
    model = apply_tensor_parallel(model, tp_mesh)
"""

import logging
import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _assert_plain_1x1(conv: nn.Conv2d, cls_name: str) -> None:
    """Require a plain 1×1 conv: the Tp wrappers rebuild the layer with default
    stride/padding/dilation/groups, so anything non-default would be silently
    dropped rather than replicated."""
    assert conv.kernel_size in ((1, 1), 1), f"{cls_name} only supports 1x1 convolutions"
    assert conv.stride in ((1, 1), 1), f"{cls_name}: stride {conv.stride} unsupported (must be 1)"
    assert conv.padding in ((0, 0), 0), f"{cls_name}: padding {conv.padding} unsupported (must be 0)"
    assert conv.dilation in ((1, 1), 1), f"{cls_name}: dilation {conv.dilation} unsupported (must be 1)"
    assert conv.groups == 1, f"{cls_name}: groups {conv.groups} unsupported (must be 1)"


# ---------------------------------------------------------------------------
# Column-parallel Conv2d (k=1)
# ---------------------------------------------------------------------------


class TpColConv2d(nn.Module):
    """Column-parallel 1×1 Conv2d: each rank owns out_channels // tp output channels.

    Input: full (B, C_in, H, W).
    Output: sharded (B, C_out // tp, H, W) — no all_reduce needed here.
    """

    def __init__(self, conv: nn.Conv2d, tp_group):
        super().__init__()
        _assert_plain_1x1(conv, "TpColConv2d")
        self.tp_group = tp_group
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)

        out_ch = conv.out_channels
        assert out_ch % tp_size == 0, f"out_channels={out_ch} not divisible by tp_size={tp_size}"
        chunk = out_ch // tp_size
        start, end = tp_rank * chunk, (tp_rank + 1) * chunk

        has_bias = conv.bias is not None
        self.conv = nn.Conv2d(conv.in_channels, chunk, 1, bias=has_bias)
        with torch.no_grad():
            self.conv.weight.copy_(conv.weight[start:end])
            if has_bias:
                self.conv.bias.copy_(conv.bias[start:end])

    def forward(self, x):
        return self.conv(x)  # (B, C_out//tp, H, W)


# ---------------------------------------------------------------------------
# Row-parallel Conv2d (k=1)
# ---------------------------------------------------------------------------


class TpRowConv2d(nn.Module):
    """Row-parallel 1×1 Conv2d: each rank owns in_channels // tp input channels.

    Input: sharded (B, C_in // tp, H, W) — from paired TpColConv2d.
    Output: full (B, C_out, H, W) after all_reduce SUM over tp_group.
    """

    def __init__(self, conv: nn.Conv2d, tp_group):
        super().__init__()
        _assert_plain_1x1(conv, "TpRowConv2d")
        self.tp_group = tp_group
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)

        in_ch = conv.in_channels
        assert in_ch % tp_size == 0, f"in_channels={in_ch} not divisible by tp_size={tp_size}"
        chunk = in_ch // tp_size
        start, end = tp_rank * chunk, (tp_rank + 1) * chunk

        # Bias is kept as a full replicated parameter and added AFTER the
        # all_reduce. Putting it inside the conv (zeroed on rank != 0) is wrong:
        # every rank's bias copy receives the same gradient, so the *summed*
        # effective bias trains at tp_size × the intended learning rate (and
        # weight decay treats the copies independently). Adding it post-reduce
        # keeps the replicas identical and the effective LR correct.
        self.conv = nn.Conv2d(chunk, conv.out_channels, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(conv.weight[:, start:end])
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = self.conv(x)  # partial output
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ---------------------------------------------------------------------------
# Column-parallel Linear
# ---------------------------------------------------------------------------


class TpColLinear(nn.Module):
    """Column-parallel Linear: each rank owns out_features // tp output neurons.

    Input: full (*, C_in).
    Output: sharded (*, C_out // tp) — no all_reduce needed here.
    """

    def __init__(self, linear: nn.Linear, tp_group):
        super().__init__()
        self.tp_group = tp_group
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)

        out_f = linear.out_features
        assert out_f % tp_size == 0, f"out_features={out_f} not divisible by tp_size={tp_size}"
        chunk = out_f // tp_size
        start, end = tp_rank * chunk, (tp_rank + 1) * chunk

        has_bias = linear.bias is not None
        self.linear = nn.Linear(linear.in_features, chunk, bias=has_bias)
        with torch.no_grad():
            self.linear.weight.copy_(linear.weight[start:end])
            if has_bias:
                self.linear.bias.copy_(linear.bias[start:end])

    def forward(self, x):
        return self.linear(x)  # (*, out_features // tp)


# ---------------------------------------------------------------------------
# Row-parallel Linear
# ---------------------------------------------------------------------------


class TpRowLinear(nn.Module):
    """Row-parallel Linear: each rank owns in_features // tp input neurons.

    Input: sharded (*, C_in // tp) — from paired TpColLinear.
    Output: full (*, C_out) after all_reduce SUM over tp_group.
    """

    def __init__(self, linear: nn.Linear, tp_group):
        super().__init__()
        self.tp_group = tp_group
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)

        in_f = linear.in_features
        assert in_f % tp_size == 0, f"in_features={in_f} not divisible by tp_size={tp_size}"
        chunk = in_f // tp_size
        start, end = tp_rank * chunk, (tp_rank + 1) * chunk

        # Full replicated bias added AFTER the all_reduce — see TpRowConv2d for
        # why an in-layer bias zeroed on rank != 0 trains at tp_size × the LR.
        self.linear = nn.Linear(chunk, linear.out_features, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(linear.weight[:, start:end])
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = self.linear(x)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _tp_group_from_mesh(tp_mesh):
    """Extract the underlying ProcessGroup from a 1D DeviceMesh."""
    return tp_mesh.get_group()


# ---------------------------------------------------------------------------
# Generic path helpers (dotted paths like "layers.1" into Sequentials)
# ---------------------------------------------------------------------------


def _rgetattr(obj, path: str):
    """Get a nested attribute/index using a dotted path."""
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _rsetattr(obj, path: str, val) -> None:
    """Set a nested attribute/index using a dotted path."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = val
    else:
        setattr(obj, last, val)


# ---------------------------------------------------------------------------
# Generic col/row dispatchers
# ---------------------------------------------------------------------------


def _to_col_parallel(layer: nn.Module, tp_group):
    if isinstance(layer, nn.Conv2d):
        return TpColConv2d(layer, tp_group)
    if isinstance(layer, nn.Linear):
        return TpColLinear(layer, tp_group)
    raise TypeError(f"_to_col_parallel: unsupported layer type {type(layer)}")


def _to_row_parallel(layer: nn.Module, tp_group):
    if isinstance(layer, nn.Conv2d):
        return TpRowConv2d(layer, tp_group)
    if isinstance(layer, nn.Linear):
        return TpRowLinear(layer, tp_group)
    raise TypeError(f"_to_row_parallel: unsupported layer type {type(layer)}")


# ---------------------------------------------------------------------------
# Native DTensor tensor parallelism (issue #415)
# ---------------------------------------------------------------------------


def _has_spectral_norm(layer: nn.Module) -> bool:
    """True if spectral norm wraps the layer's weight.

    Covers both registration styles: the hook-based ``nn.utils.spectral_norm``
    used by the wxformer family's ``apply_spectral_norm`` (registers
    ``weight_orig``/``weight_u``/``weight_v``) and the parametrize-based
    ``nn.utils.parametrizations.spectral_norm`` (registers
    ``parametrizations.weight``).
    """
    if hasattr(layer, "weight_orig"):
        return True
    parametrizations = getattr(layer, "parametrizations", None)
    return parametrizations is not None and "weight" in parametrizations


def supports_native_tp(model: nn.Module) -> bool:
    """True if any submodule declares a native ``_tp_plan``.

    Models opt in per block::

        class MyBlock(nn.Module):
            _tp_plan = {"to_q": "colwise", "to_out": "rowwise"}

    Currently only wxformer_next's transformer blocks declare plans (and
    CubeSphereWxFormer by inheritance).
    """
    return any(getattr(type(m), "_tp_plan", None) for m in model.modules())


def apply_native_tensor_parallel(model: nn.Module, tp_mesh) -> nn.Module:
    """Apply torch-native DTensor TP to all blocks that declare ``_tp_plan``.

    Uses ``parallelize_module`` with ColwiseParallel/RowwiseParallel, so the
    sharded parameters become DTensors that keep their FQNs and logical
    shapes — checkpointing, EMA, and optimizer state work through the same
    DCP full-state APIs as FSDP2, and ``fully_shard`` over the dp submesh
    composes on top (2D mesh, the torchtitan configuration).

    Block protocol (mirrors the legacy ``_tp_col``/``_tp_row``):
      - ``_tp_plan``: dict mapping dotted submodule paths (must resolve to
        nn.Linear) to ``"colwise"`` or ``"rowwise"``.
      - ``_tp_constraints(instance, tp_size)`` (optional staticmethod):
        raises on invalid configurations, e.g. heads % tp_size != 0.

    Requirements enforced here with clear errors:
      - every planned layer is an nn.Linear (the refactored wxformer_next
        projections; 1x1 convs are NOT supported — that was the legacy path)
      - colwise out_features / rowwise in_features divisible by the TP degree

    Spectral norm reduces (does not break) TP: the power-iteration hook mixes
    plain-tensor u/v buffers with DTensor weights, so blocks whose planned
    layers carry spectral norm are SKIPPED — left fully replicated on every
    TP rank — with a warning. The skip is per block, never per layer: a
    colwise layer's sharded output is only valid feeding its rowwise partner,
    so the whole colwise/rowwise group stays together. Replicated blocks are
    still correct (TP peers see identical inputs by the dp contract, and the
    trainer's sync_replicated_gradients pins the replicas), but for full TP
    set model.use_spectral_norm: false.

    Args:
        model: The model to convert (modified in place).
        tp_mesh: 1-D DeviceMesh for the tensor-parallel dimension.

    Returns:
        model (same object), with ``_tp_group`` stashed for the trainer's
        replicated-gradient sync.
    """
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    styles = {"colwise": ColwiseParallel, "rowwise": RowwiseParallel}
    tp_size = tp_mesh.size()
    count = 0
    skipped = 0

    seen = set()
    for module in model.modules():
        mid = id(module)
        if mid in seen:
            continue
        seen.add(mid)

        plan_spec = getattr(type(module), "_tp_plan", None)
        if not plan_spec:
            continue

        # Spectral norm on any planned layer disqualifies the WHOLE block:
        # a colwise layer's sharded output is only consumed correctly by its
        # rowwise partner, so the colwise/rowwise group must shard (or stay
        # replicated) as a unit. A replicated block is still correct — TP
        # peers see identical inputs (dp contract) and a preceding rowwise
        # all_reduce already restored the full activation.
        if any(_has_spectral_norm(_rgetattr(module, path)) for path in plan_spec):
            skipped += 1
            continue

        check_fn = getattr(type(module), "_tp_constraints", None)
        if check_fn is not None:
            check_fn(module, tp_size)

        plan = {}
        for path, style in plan_spec.items():
            layer = _rgetattr(module, path)
            if not isinstance(layer, nn.Linear):
                raise TypeError(
                    f"Native TP: {type(module).__name__}._tp_plan[{path!r}] resolves to "
                    f"{type(layer).__name__}, but parallelize_module requires nn.Linear. "
                    "Conv projections must be refactored to Linear (issue #415)."
                )
            if style == "colwise" and layer.out_features % tp_size != 0:
                raise ValueError(
                    f"Native TP: {type(module).__name__}.{path} out_features="
                    f"{layer.out_features} not divisible by tp_size={tp_size}."
                )
            if style == "rowwise" and layer.in_features % tp_size != 0:
                raise ValueError(
                    f"Native TP: {type(module).__name__}.{path} in_features="
                    f"{layer.in_features} not divisible by tp_size={tp_size}."
                )
            plan[path] = styles[style]()

        parallelize_module(module, tp_mesh, plan)
        count += 1

    if count == 0 and skipped == 0:
        raise ValueError(
            "apply_native_tensor_parallel: no blocks with _tp_plan found. Check "
            "supports_native_tp(model) before calling, or use a model that opts in "
            "(wxformer_next family)."
        )
    if skipped and count:
        logger.warning(
            f"Native TP is REDUCED: {count} block(s) sharded, {skipped} block(s) skipped "
            "because their projection layers carry spectral norm (the power-iteration "
            "hook is incompatible with DTensor sharding). Skipped blocks stay fully "
            "replicated on every TP rank. For full tensor parallelism set "
            "model.use_spectral_norm: false."
        )
    elif skipped:
        logger.warning(
            f"Native TP had NO effect: all {skipped} _tp_plan block(s) carry spectral "
            f"norm, so nothing was sharded. Every parameter is replicated and the "
            f"{tp_size} TP ranks repeat identical compute with no memory savings. "
            "Training remains correct (replicas are kept in sync), but set "
            "model.use_spectral_norm: false to actually shard the model, or set "
            "trainer.parallelism.tensor: 1 to reclaim the ranks for data parallelism."
        )
    else:
        logger.info(f"Native tensor parallelism applied to {count} block(s), degree={tp_size}")

    # Stash the group so the trainer can sync replicated-parameter gradients
    # across the TP dimension at accumulation boundaries. _tp_native marks
    # the model for the checkpoint paths: native TP params keep their FQNs
    # and logical shapes, so the DCP full-state APIs save/load them like any
    # FSDP2 DTensor (the legacy warn-on-save / raise-on-resume guards do not
    # apply).
    model._tp_group = _tp_group_from_mesh(tp_mesh)
    model._tp_native = True
    return model


def _is_tp_sharded_param(p) -> bool:
    """True if p is a DTensor sharded along a mesh dim named 'tp'.

    Works both before fully_shard (1-D ('tp',) mesh from parallelize_module)
    and after (2-D ('dp', 'tp') mesh from the FSDP2 composition), and is
    wrapper-agnostic (no FQN matching, so AC/FSDP2 prefixes don't matter).
    """
    placements = getattr(p, "placements", None)
    if placements is None:
        return False
    names = p.device_mesh.mesh_dim_names or ()
    return any(name == "tp" and placement.is_shard() for name, placement in zip(names, placements))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_tensor_parallel(model: nn.Module, tp_mesh) -> nn.Module:
    """Walk model and apply TP to all blocks that declare ``_tp_col``/``_tp_row``.

    Any ``nn.Module`` subclass can opt in by setting two class attributes::

        class MyBlock(nn.Module):
            _tp_col = "proj_up"   # dotted path to the column-parallel layer
            _tp_row = "proj_out"  # dotted path to the row-parallel layer

    Paths may address layers inside Sequentials (e.g. ``"layers.1"``).
    Supported layer types: ``nn.Conv2d`` (1×1 only) and ``nn.Linear``.

    Converts in-place. Safe to call before apply_fsdp2.

    Args:
        model: The model to convert.
        tp_mesh: 1-D DeviceMesh for the tensor-parallel dimension.

    Returns:
        model (same object, modified in-place).
    """
    raise NotImplementedError(
        "Tensor parallelism (trainer.parallelism.tensor > 1) is disabled for "
        "this model: the hand-rolled column sharding slices fused projections "
        "(e.g. WXFormer's to_qkv) across q/k/v boundaries, and the backward "
        "all-reduce at the column-parallel input (Megatron's 'f' operator) is "
        "missing, so tensor > 1 silently trains wrong outputs and gradients. "
        "Native DTensor TP (issue #415) is available for models that declare "
        "_tp_plan blocks — currently the wxformer_next family (model type "
        "nextgen_wxformer). Use one of those, or set "
        "trainer.parallelism.tensor: 1."
    )
    tp_group = _tp_group_from_mesh(tp_mesh)
    count = 0

    seen = set()
    for module in model.modules():
        mid = id(module)
        if mid in seen:
            continue
        seen.add(mid)

        col_path = getattr(module, "_tp_col", None)
        row_path = getattr(module, "_tp_row", None)
        if col_path is None or row_path is None:
            continue

        # Optional per-block constraint check — implement _tp_constraints(instance, tp_size)
        # on a block class to catch invalid configurations early (e.g. heads % tp_size != 0).
        check_fn = getattr(module, "_tp_constraints", None)
        if check_fn is not None:
            check_fn(module, dist.get_world_size(tp_group))

        col_layer = _rgetattr(module, col_path)
        row_layer = _rgetattr(module, row_path)
        _rsetattr(module, col_path, _to_col_parallel(col_layer, tp_group))
        _rsetattr(module, row_path, _to_row_parallel(row_layer, tp_group))
        count += 1

    if count == 0:
        logger.warning(
            "apply_tensor_parallel: no blocks with _tp_col/_tp_row found — "
            "tensor parallelism had no effect. Add these class attributes to "
            "your model's transformer blocks to enable TP."
        )
    else:
        logger.info(f"Tensor parallelism applied to {count} block(s)")
    # Stash the group so the trainer can sync replicated-parameter gradients
    # across the TP dimension at accumulation boundaries.
    model._tp_group = tp_group
    return model


def sync_replicated_gradients(model: nn.Module, tp_group) -> None:
    """Average gradients of replicated (non-TP-sharded) params across the TP group.

    The TP-sharded weights are genuinely different per rank and must NOT be
    synced: legacy Tp col/row wrapper params, and (native path) DTensors with
    a Shard placement on the 'tp' mesh dim. Everything else (embeddings,
    norms, non-TP blocks, and the replicated row-parallel biases) holds an
    identical copy on every TP rank. Their gradients are identical in exact
    arithmetic given identical inputs, but with data=none nothing enforces
    that, and nondeterministic kernels drift the replicas apart over a long
    run. Averaging at the accumulation boundary pins the replicas together.

    See credit.parallel.collectives.allreduce_grads_avg for the bucketing and
    DTensor handling. TP peers share the same dp coordinate, so their DTensor
    shard shapes are identical.
    """
    from credit.parallel.collectives import allreduce_grads_avg

    sharded_ids = set()
    for m in model.modules():
        if isinstance(m, (TpColConv2d, TpRowConv2d)):
            sharded_ids.update(id(p) for p in m.conv.parameters())
        elif isinstance(m, (TpColLinear, TpRowLinear)):
            sharded_ids.update(id(p) for p in m.linear.parameters())

    allreduce_grads_avg(
        (
            p.grad
            for p in model.parameters()
            if p.grad is not None and id(p) not in sharded_ids and not _is_tp_sharded_param(p)
        ),
        tp_group,
    )
