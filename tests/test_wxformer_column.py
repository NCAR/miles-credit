"""Tests for WXFormerColumn's residual connection.

Companion to the equivalent tests in test_cubed_wxformer.py -- both models
build the residual the same way (prognostic channels of the last input frame,
zero-padded for output-only/diagnostic channels), so both need the same
coverage.
"""

import torch
import torch.nn as nn
import pytest

from credit.models.wxformer.wxformer_column import WXFormerColumn


def _tiny_model_conf(**overrides):
    kwargs = dict(
        image_height=32,
        image_width=64,
        frames=1,
        channels=2,
        surface_channels=2,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=(8, 16, 32, 64),
        depth=(1, 1, 1, 1),
        dim_head=4,
        global_window_size=(2, 2, 2, 1),
        local_window_size=2,
        cross_embed_kernel_sizes=((2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        col_attn_heads=2,
        # Off by default here: spectral norm reparametrizes weight / spectral_norm(weight),
        # which is a division by zero (-> NaN) once _zero_init_head zeroes the raw weight.
        use_spectral_norm=False,
    )
    kwargs.update(overrides)
    return kwargs


def _zero_init_head(model):
    """Zero both convs in up_block4 so the predicted delta is exactly zero,
    isolating the residual for an exact equality check."""
    nn.init.zeros_(model.up_block4[0].weight)
    nn.init.zeros_(model.up_block4[0].bias)
    nn.init.zeros_(model.up_block4[2].weight)
    nn.init.zeros_(model.up_block4[2].bias)


def _expected_residual(x, c_prog, c_out):
    """Prognostic channels of the last input frame, zero-padded for any
    output-only (diagnostic) channels -- never copied from input-only fields."""
    b, _, h, w = x.shape
    return torch.cat([x[:, :c_prog], torch.zeros(b, c_out - c_prog, h, w)], dim=1)


@pytest.mark.parametrize(
    "input_only_channels,output_only_channels",
    [
        (1, 0),  # no output-only channels (Cout == C_prognostic)
        (1, 3),  # output_only < input_only (Cout < Cin)
        (0, 2),  # no input-only channels at all
        (1, 6),  # output_only far exceeds input_only -> Cout > Cin
    ],
)
def test_zero_init_head_gives_persistence(input_only_channels, output_only_channels):
    """Zero-init head => delta 0 => output equals the residual exactly: the
    prognostic part of the last input frame, with output-only (diagnostic)
    channels held at zero rather than copied from unrelated input-only fields.
    Covers Cout > Cin, which used to raise a shape error."""
    model = WXFormerColumn(
        **_tiny_model_conf(input_only_channels=input_only_channels, output_only_channels=output_only_channels)
    )
    model.eval()
    _zero_init_head(model)

    c_prog = 2 * 2 + 2  # channels * levels + surface_channels
    c_in = c_prog + input_only_channels
    x = torch.randn(1, c_in, 1, 32, 64)
    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, model.output_channels, 1, 32, 64)
    expected = _expected_residual(x[:, :, 0], c_prog, model.output_channels)
    assert torch.allclose(y[:, :, 0], expected, atol=0.0)
