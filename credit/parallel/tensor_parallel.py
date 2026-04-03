"""Tensor Parallelism for CREDIT v2 WXFormer models.

Applies column/row parallelism to the two dense linear bottlenecks in each
transformer block:

  FeedForward / FeedForwardSwiGLU (Conv2d k=1):
    proj_up   (or layers[1]) → TpColConv2d  (output channels sharded)
    proj_out  (or layers[4]) → TpRowConv2d  (input channels sharded, all_reduce)

  GridAttention (nn.Linear):
    to_qkv  → TpColLinear   (output features sharded → Q/K/V heads split)
    to_out  → TpRowLinear   (input features sharded, all_reduce)

  Window Attention (Conv2d k=1):
    to_qkv  → TpColConv2d
    to_out  → TpRowConv2d

All other layers (norms, convolutions with spatial extent, embeddings,
PixelShuffle) are left unchanged — they are either local operations or
too expensive to partition.

Gradient flow:
  - TPColConv2d / TPColLinear: no all_reduce needed (output is sharded,
    consumed by the paired Row layer).
  - TPRowConv2d / TPRowLinear: all_reduce SUM over tp_group before returning,
    so the rest of the graph sees the full reduced output.

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


def _convert_feedforward(ff_module, tp_group):
    """Convert FeedForward (Sequential with two 1x1 Conv2d) to TP."""
    # layers: LayerNorm, Conv2d(dim→dim*mult), GELU, Dropout, Conv2d(dim*mult→dim)
    seq = ff_module.layers
    for i, layer in enumerate(seq):
        if isinstance(layer, nn.Conv2d) and layer.kernel_size in ((1, 1), 1):
            first_conv_idx = i
            break
    else:
        return ff_module

    # Find the two Conv2d layers (first = Col, second = Row)
    conv_indices = [i for i, layer in enumerate(seq) if isinstance(layer, nn.Conv2d)]
    if len(conv_indices) < 2:
        return ff_module

    col_idx, row_idx = conv_indices[0], conv_indices[-1]
    seq[col_idx] = TpColConv2d(seq[col_idx], tp_group)
    seq[row_idx] = TpRowConv2d(seq[row_idx], tp_group)
    return ff_module


def _convert_swiglu(swiglu_module, tp_group):
    """Convert FeedForwardSwiGLU to TP."""
    # proj_up: Conv2d(dim, hidden*2, 1) → Col (output sharded, gate+val both sharded)
    # proj_out: Conv2d(hidden, dim, 1) → Row (input is hidden//tp, all_reduce)
    swiglu_module.proj_up = TpColConv2d(swiglu_module.proj_up, tp_group)
    swiglu_module.proj_out = TpRowConv2d(swiglu_module.proj_out, tp_group)
    return swiglu_module


def _convert_grid_attention(attn_module, tp_group):
    """Convert GridAttention to TP (uses nn.Linear — clean column/row split)."""
    # to_qkv: Linear(dim, inner_dim*3) → Col
    # to_out: Linear(inner_dim, dim) → Row
    attn_module.to_qkv = TpColLinear(attn_module.to_qkv, tp_group)
    attn_module.to_out = TpRowLinear(attn_module.to_out, tp_group)
    return attn_module


def _convert_window_attention(attn_module, tp_group):
    """Convert window Attention (Conv2d k=1 QKV) to TP."""
    # to_qkv: Conv2d(dim, inner_dim*3, 1) → Col
    # to_out: Conv2d(inner_dim, dim, 1) → Row
    attn_module.to_qkv = TpColConv2d(attn_module.to_qkv, tp_group)
    attn_module.to_out = TpRowConv2d(attn_module.to_out, tp_group)
    return attn_module


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_tensor_parallel(model: nn.Module, tp_mesh) -> nn.Module:
    """Walk model and apply TP to all eligible transformer blocks.

    Converts in-place. Safe to call before apply_fsdp2.

    Args:
        model: The model to convert (e.g. CrossFormer / WXFormer).
        tp_mesh: 1-D DeviceMesh for the tensor-parallel dimension.

    Returns:
        model (same object, modified in-place).
    """
    try:
        from credit.models.wxformer.crossformer import (
            FeedForward,
            Attention,
        )
    except ImportError:
        logger.warning("crossformer not found — tensor parallelism skipped")
        return model

    tp_group = _tp_group_from_mesh(tp_mesh)

    counts = {"ff": 0, "window_attn": 0}

    # Walk named children; replace in parent
    for parent_name, parent in model.named_modules():
        for name, module in list(parent.named_children()):
            if isinstance(module, FeedForward):
                setattr(parent, name, _convert_feedforward(module, tp_group))
                counts["ff"] += 1
            elif isinstance(module, Attention):
                setattr(parent, name, _convert_window_attention(module, tp_group))
                counts["window_attn"] += 1

    logger.info(f"Tensor parallelism applied: {counts['ff']} FF, {counts['window_attn']} WindowAttn")
    return model
