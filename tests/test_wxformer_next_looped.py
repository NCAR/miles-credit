"""Tests for WXFormerNextLooped (credit/models/wxformer/wxformer_next_looped.py).

Covers the properties the looped-bottleneck design (docs/looped_transformer_claude_handoff.md,
Experiment A) is supposed to have: parameter count independent of loop count, gradient
reaching the shared refiner and the per-iteration gates, an exact no-op at gate=0, and
the same residual-channel correctness already required of the other wxformer models.
"""

import pytest
import torch
import torch.nn as nn

from credit.models import load_model
from credit.models.wxformer.wxformer_next_looped import WXFormerNextLooped


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
        bottleneck_loop_depth=1,
        use_spectral_norm=False,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("bottleneck_loops", [1, 2, 4])
def test_forward_shape_roundtrips(bottleneck_loops):
    model = WXFormerNextLooped(**_tiny_model_conf(bottleneck_loops=bottleneck_loops))
    model.eval()

    c_in = 2 * 2 + 2 + 1
    c_out = 2 * 2 + 2
    x = torch.randn(1, c_in, 1, 32, 64)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, c_out, 1, 32, 64)


def test_bottleneck_loops_rejects_less_than_one():
    with pytest.raises(ValueError, match="bottleneck_loops"):
        WXFormerNextLooped(**_tiny_model_conf(bottleneck_loops=0))


def test_param_count_independent_of_loop_count():
    """The whole point of sharing weights across passes: looping the same
    refiner block more times must not grow the parameter count (beyond the
    trivially small per-iteration embedding/gate vectors)."""
    counts = {}
    for k in (1, 2, 4, 8):
        model = WXFormerNextLooped(**_tiny_model_conf(bottleneck_loops=k))
        # Exclude the per-iteration embedding/gate -- those are O(k * dim) and
        # O(k) respectively, deliberately NOT shared, and negligible in size.
        shared_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if "iter_embedding" not in name and "loop_gate" not in name
        )
        counts[k] = shared_params
    assert len(set(counts.values())) == 1, f"shared parameter count varies with loop count: {counts}"


def test_gradient_reaches_refiner_and_gates():
    model = WXFormerNextLooped(**_tiny_model_conf(bottleneck_loops=3))
    x = torch.randn(1, 2 * 2 + 2 + 1, 1, 32, 64, requires_grad=True)
    y = model(x)
    y.mean().backward()

    assert model.loop_gate.grad is not None
    assert torch.isfinite(model.loop_gate.grad).all()
    assert (model.loop_gate.grad != 0).any(), "no iteration's gate received gradient signal"

    refiner_grad_norms = [p.grad.norm().item() for p in model.bottleneck_refiner.parameters() if p.grad is not None]
    assert refiner_grad_norms, "bottleneck_refiner received no gradients at all"
    assert any(g > 0 for g in refiner_grad_norms), "every bottleneck_refiner gradient was exactly zero"


def test_zero_gate_makes_loop_a_no_op():
    """With every loop_gate forced to exactly 0, z = z + 0*delta every pass --
    the bottleneck output must equal the pre-loop encoded features exactly,
    regardless of loop count. This is the guard against the refiner secretly
    re-adding z on its own (which would break this no-op property)."""
    model = WXFormerNextLooped(**_tiny_model_conf(bottleneck_loops=3))
    model.eval()
    with torch.no_grad():
        model.loop_gate.zero_()

    captured = {}
    orig_forward = model.bottleneck_refiner.forward

    def _spy(z, c, k):
        delta = orig_forward(z, c, k)
        captured.setdefault("z_before", []).append(z.clone())
        return delta

    model.bottleneck_refiner.forward = _spy

    x = torch.randn(1, 2 * 2 + 2 + 1, 1, 32, 64)
    with torch.no_grad():
        model(x)

    # Every pass must see the SAME z (the untouched encoded features `c`),
    # since gate=0 means no pass ever actually updates it.
    z0 = captured["z_before"][0]
    for z_k in captured["z_before"][1:]:
        assert torch.equal(z_k, z0), "z changed across passes despite loop_gate == 0"


def test_registered_in_model_registry():
    conf = {"model": {"type": "wxformer_next_looped", **_tiny_model_conf(bottleneck_loops=2)}}
    model = load_model(conf)
    assert isinstance(model, WXFormerNextLooped)


def test_zero_init_head_gives_persistence():
    """Zero-init the final decoder conv => delta 0 => output equals the
    residual exactly: the prognostic part of the last input frame, zero-padded
    for output-only (diagnostic) channels."""
    model = WXFormerNextLooped(**_tiny_model_conf(output_only_channels=3, bottleneck_loops=2))
    model.eval()
    nn.init.zeros_(model.up_block4[0].weight)
    nn.init.zeros_(model.up_block4[0].bias)
    nn.init.zeros_(model.up_block4[2].weight)
    nn.init.zeros_(model.up_block4[2].bias)

    c_prog = 2 * 2 + 2  # channels * levels + surface_channels
    x = torch.randn(1, c_prog + 1, 1, 32, 64)
    with torch.no_grad():
        y = model(x)

    expected = torch.cat([x[:, :c_prog, 0], torch.zeros(1, model.output_channels - c_prog, 32, 64)], dim=1)
    assert torch.allclose(y[:, :, 0], expected, atol=0.0)
