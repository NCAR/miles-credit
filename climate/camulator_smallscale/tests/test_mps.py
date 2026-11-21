"""The Apple MPS large-kernel Conv2d workaround (camulator_smallscale.mps)."""

import copy

import pytest
import torch

from camulator_smallscale.mps import MPS_CONV_MAX_CIN_KERNEL, chunked_conv2d, patch_mps_large_kernel_convs

MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _big_conv(**kw):
    # 136 * 16 * 16 = 34816 > 2**15, so it must be patched (CAMulator's first layer is 136 channels x 32x32).
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
    # A small chunk limit forces many chunks, including a ragged last one.
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
    # EMA and similar keep a deepcopy of the model; the patched forward must use the copy's weights.
    conv = _big_conv(padding=2)
    patch_mps_large_kernel_convs(conv)
    twin = copy.deepcopy(conv)
    with torch.no_grad():
        twin.weight.zero_()
        twin.bias.zero_()
    x = torch.randn(1, 136, 24, 24)
    assert torch.count_nonzero(twin(x)) == 0
    assert torch.count_nonzero(conv(x)) > 0


@pytest.mark.skipif(not MPS, reason="needs Apple MPS")
def test_patched_conv_matches_cpu_on_mps():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(136, 16, kernel_size=32, stride=4, padding=14)  # CAMulator's first-layer shape
    x = torch.randn(1, 136, 192, 288)
    ref = conv(x)
    conv_mps = copy.deepcopy(conv).to("mps")
    unpatched = conv_mps(x.to("mps")).cpu()
    patch_mps_large_kernel_convs(conv_mps)
    patched = conv_mps(x.to("mps")).cpu()
    scale = ref.abs().max()
    assert (patched - ref).abs().max() / scale < 1e-4
    # Documents the bug this works around; if it ever starts passing, MPS has been fixed.
    assert (unpatched - ref).abs().max() / scale > 1e-2
