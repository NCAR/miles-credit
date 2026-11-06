"""Tests for credit.models.mps_compat (Apple MPS large-kernel Conv2d workaround)."""

import copy

import pytest
import torch

from credit.models import load_model, register_model
from credit.models.mps_compat import (
    MPS_CONV_MAX_CIN_KERNEL,
    chunked_conv2d,
    patch_mps_large_kernel_convs,
)

MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _big_conv(**kw):
    # 136 * 16 * 16 = 34816 > 2**15 -> must be patched (CAMulator's first layer is 136 x 32x32)
    return torch.nn.Conv2d(136, 8, kernel_size=16, **kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"stride": 2, "padding": 7},
        {"stride": 1, "padding": (3, 5), "dilation": 2},
        {"stride": 4, "padding": 0, "bias": False},
    ],
)
def test_chunked_conv2d_matches_conv2d(kw):
    conv = _big_conv(**kw).double()
    x = torch.randn(2, 136, 40, 48, dtype=torch.float64)
    ref = conv(x)
    # small chunk limit forces many chunks, including a ragged last one
    out = chunked_conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, max_cin_kernel=7 * 256)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)


def test_patch_selects_only_unsafe_convs_and_is_idempotent():
    model = torch.nn.Sequential(
        _big_conv(padding=2),
        torch.nn.Conv2d(8, 8, 3),  # small: untouched
        torch.nn.Conv2d(136, 136, 16, groups=136),  # grouped: untouched
        torch.nn.Conv2d(136, 8, 16, padding="same"),  # string padding: untouched
    )
    assert 136 * 16 * 16 > MPS_CONV_MAX_CIN_KERNEL
    assert patch_mps_large_kernel_convs(model) == 1
    assert "_conv_forward" in vars(model[0])
    assert all("_conv_forward" not in vars(m) for m in model[1:])
    patched = model[0]._conv_forward
    assert patch_mps_large_kernel_convs(model) == 1
    assert model[0]._conv_forward is patched


def test_patched_conv_is_unchanged_off_mps():
    conv = _big_conv(stride=2, padding=7)
    x = torch.randn(1, 136, 32, 32)
    ref = conv(x)
    patch_mps_large_kernel_convs(conv)
    assert torch.equal(conv(x), ref)  # CPU takes the original path: bit-identical


def test_deepcopy_rebinds_to_copy():
    # EMA and similar keep a deepcopy of the model; the patched forward must use the copy's weights
    conv = _big_conv(padding=2)
    patch_mps_large_kernel_convs(conv)
    twin = copy.deepcopy(conv)
    with torch.no_grad():
        twin.weight.zero_()
        twin.bias.zero_()
    x = torch.randn(1, 136, 24, 24)
    assert torch.count_nonzero(twin(x)) == 0
    assert torch.count_nonzero(conv(x)) > 0


def test_load_model_patches_registered_models():
    from credit.models.base_model import BaseModel

    @register_model("test_mps_bigconv_model")
    class BigConvModel(BaseModel):
        def __init__(self):
            super().__init__()
            self.embed = _big_conv(padding=2)

        def forward(self, x):
            return self.embed(x)

    model = load_model({"model": {"type": "test_mps_bigconv_model"}, "save_loc": "/tmp"})
    assert "_conv_forward" in vars(model.embed)


@pytest.mark.skipif(not MPS, reason="requires Apple MPS")
def test_patched_conv_on_mps_matches_cpu():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(136, 64, kernel_size=32, stride=2, padding=15)
    x = torch.randn(1, 136, 96, 144)
    ref = conv(x)
    patch_mps_large_kernel_convs(conv)
    out = conv.to("mps")(x.to("mps")).cpu()
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel < 1e-4, f"MPS chunked conv relative error {rel:.2e}"
