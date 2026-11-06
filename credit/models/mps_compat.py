"""Workarounds for Apple MPS kernels that return silently wrong results.

Apple's MPS backend returns wrong ``Conv2d`` output (~100% relative error, no warning
or exception) once ``in_channels * kH * kW`` exceeds ``2**16`` (seen on torch 2.13 /
macOS 26). The CrossFormer-family models (CrossFormer, CAMulator, WXFormer) hit it in
their first ``CrossEmbedLayer``: e.g. CAMulator's 136 input channels with a 32x32 kernel.

:func:`patch_mps_large_kernel_convs` reroutes every affected layer through a forward that,
**only when the input is on MPS**, evaluates the convolution as a sum over input-channel
chunks that stay under the limit -- exact by linearity, up to float summation order. On
CPU/CUDA the original forward runs unchanged, so it is safe to apply unconditionally;
:func:`credit.models.load_model` and :func:`credit.models.load_model_name` do so for
every model they build.
"""

import types

import torch
import torch.nn.functional as F

# Half the observed failure threshold (2**16), for margin.
MPS_CONV_MAX_CIN_KERNEL = 2**15


def chunked_conv2d(input, weight, bias, stride, padding, dilation, max_cin_kernel=MPS_CONV_MAX_CIN_KERNEL):
    """``F.conv2d`` (groups=1) as a sum over input-channel chunks with ``in_chunk * kH * kW <= max_cin_kernel``."""
    chunk = max(1, max_cin_kernel // (weight.shape[2] * weight.shape[3]))
    out = None
    for s in range(0, weight.shape[1], chunk):
        o = F.conv2d(input[:, s : s + chunk], weight[:, s : s + chunk], None, stride, padding, dilation, 1)
        out = o if out is None else out + o
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def _mps_safe_conv_forward(self, input, weight, bias):
    if input.device.type != "mps":
        return type(self)._conv_forward(self, input, weight, bias)
    return chunked_conv2d(input, weight, bias, self.stride, self.padding, self.dilation)


def _needs_patch(mod: torch.nn.Module) -> bool:
    return (
        isinstance(mod, torch.nn.Conv2d)
        and mod.groups == 1
        and mod.padding_mode == "zeros"
        and isinstance(mod.padding, tuple)  # string padding ("same") is not handled by the chunked path
        and mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1] > MPS_CONV_MAX_CIN_KERNEL
    )


def patch_mps_large_kernel_convs(model: torch.nn.Module) -> int:
    """Make the model's MPS-unsafe ``Conv2d`` layers MPS-safe, in place.

    Idempotent. Returns the number of layers that are routed through the MPS-safe
    forward (whether patched by this call or an earlier one).
    """
    n = 0
    for mod in model.modules():
        if not _needs_patch(mod):
            continue
        if not getattr(mod, "_credit_mps_safe", False):
            mod._conv_forward = types.MethodType(_mps_safe_conv_forward, mod)
            mod._credit_mps_safe = True
        n += 1
    return n
