"""Workaround for an Apple MPS kernel that returns silently wrong results.

On MPS, ``Conv2d`` returns wrong output (no warning or exception) once ``in_channels * kH * kW`` exceeds about
``2**16``; it is still the case on torch 2.13. CAMulator's first ``CrossEmbedLayer`` crosses it (136 input channels,
32x32 and 16x16 kernels): a CAMulator forward on MPS differs from CPU by up to ~10% of the output's range.

:func:`patch_mps_large_kernel_convs` routes every affected layer through a forward that, **only when the input is on
MPS**, evaluates the convolution as a sum over input-channel chunks that stay under the limit. That is exact by
linearity, up to float summation order. On CPU and CUDA the original forward runs unchanged, so the patch is safe to
apply unconditionally.

Ported from the earlier ``credit/models/mps_compat.py`` (local branches only, never merged into CREDIT); here it is
applied by a custom model instead of by ``credit.models.load_model``.
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
    for start in range(0, weight.shape[1], chunk):
        part = F.conv2d(
            input[:, start : start + chunk], weight[:, start : start + chunk], None, stride, padding, dilation, 1
        )
        out = part if out is None else out + part
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def _mps_safe_conv_forward(self, input, weight, bias):
    if input.device.type != "mps":
        return type(self)._conv_forward(self, input, weight, bias)
    return chunked_conv2d(input, weight, bias, self.stride, self.padding, self.dilation)


def _needs_patch(module):
    return (
        isinstance(module, torch.nn.Conv2d)
        and module.groups == 1
        and module.padding_mode == "zeros"
        and isinstance(module.padding, tuple)  # string padding ("same") is not handled by the chunked path
        and module.in_channels * module.kernel_size[0] * module.kernel_size[1] > MPS_CONV_MAX_CIN_KERNEL
    )


def patch_mps_large_kernel_convs(model):
    """Make the model's MPS-unsafe ``Conv2d`` layers MPS-safe, in place.

    Idempotent. Returns the number of layers routed through the MPS-safe forward (patched by this call or an earlier
    one). ``copy.deepcopy`` rebinds the patched forward to the copy, so EMA-style model copies stay correct.
    """
    n = 0
    for module in model.modules():
        if not _needs_patch(module):
            continue
        if not getattr(module, "_credit_mps_safe", False):
            module._conv_forward = types.MethodType(_mps_safe_conv_forward, module)
            module._credit_mps_safe = True
        n += 1
    return n
