"""Tensor Parallelism for CREDIT v2 models.

Applies column/row parallelism to transformer blocks that opt in by declaring
two class attributes:

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
        assert conv.kernel_size == (1, 1) or conv.kernel_size == 1, "TpColConv2d only supports 1x1 convolutions"
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
        assert conv.kernel_size == (1, 1) or conv.kernel_size == 1, "TpRowConv2d only supports 1x1 convolutions"
        self.tp_group = tp_group
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)

        in_ch = conv.in_channels
        assert in_ch % tp_size == 0, f"in_channels={in_ch} not divisible by tp_size={tp_size}"
        chunk = in_ch // tp_size
        start, end = tp_rank * chunk, (tp_rank + 1) * chunk

        has_bias = conv.bias is not None
        self.conv = nn.Conv2d(chunk, conv.out_channels, 1, bias=has_bias)
        with torch.no_grad():
            self.conv.weight.copy_(conv.weight[:, start:end])
            if has_bias:
                # Only rank 0 contributes the bias to avoid double-counting.
                if tp_rank == 0:
                    self.conv.bias.copy_(conv.bias)
                else:
                    self.conv.bias.zero_()

    def forward(self, x):
        out = self.conv(x)  # partial output
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
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

        has_bias = linear.bias is not None
        self.linear = nn.Linear(chunk, linear.out_features, bias=has_bias)
        with torch.no_grad():
            self.linear.weight.copy_(linear.weight[:, start:end])
            if has_bias:
                if tp_rank == 0:
                    self.linear.bias.copy_(linear.bias)
                else:
                    self.linear.bias.zero_()

    def forward(self, x):
        out = self.linear(x)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
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
    tp_group = _tp_group_from_mesh(tp_mesh)
    count = 0

    seen = set()
    for module in model.modules():
        mid = id(module)
        if mid in seen:
            continue
        seen.add(mid)

        col_path = getattr(type(module), "_tp_col", None)
        row_path = getattr(type(module), "_tp_row", None)
        if col_path is None or row_path is None:
            continue

        # Optional per-block constraint check — implement _tp_constraints(instance, tp_size)
        # on a block class to catch invalid configurations early (e.g. heads % tp_size != 0).
        check_fn = getattr(type(module), "_tp_constraints", None)
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
    return model
