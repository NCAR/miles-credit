"""Tests for credit.trainers.grad_clip (trainer.grad_max_norm: 'dynamic')."""

import math

import pytest
import torch

from credit.trainers.grad_clip import AdaptiveGradClipper, global_grad_norm


def _params_with_grad(norm):
    """Two parameters whose gradients have the given total L2 norm."""
    a = torch.nn.Parameter(torch.zeros(3))
    b = torch.nn.Parameter(torch.zeros(4))
    a.grad = torch.full((3,), 1.0)
    b.grad = torch.full((4,), 1.0)
    scale = norm / math.sqrt(7.0)
    a.grad.mul_(scale)
    b.grad.mul_(scale)
    return [a, b]


def test_global_grad_norm_single_process():
    params = _params_with_grad(5.0)
    assert float(global_grad_norm(params, distributed=False)) == pytest.approx(5.0)


def test_warmup_never_clips():
    clipper = AdaptiveGradClipper(factor=2.0, ema_decay=0.9, warmup_steps=3)
    for norm in (1.0, 1.0, 100.0):
        params = _params_with_grad(norm)
        assert clipper.clip_(params, distributed=False) == pytest.approx(norm)
        assert float(global_grad_norm(params, distributed=False)) == pytest.approx(norm)


def test_spike_is_clipped_to_factor_times_ema():
    clipper = AdaptiveGradClipper(factor=2.0, ema_decay=0.9, warmup_steps=5)
    for _ in range(5):
        clipper.clip_(_params_with_grad(1.0), distributed=False)
    assert clipper.threshold() == pytest.approx(2.0)

    normal = _params_with_grad(1.5)
    clipper.clip_(normal, distributed=False)
    assert float(global_grad_norm(normal, distributed=False)) == pytest.approx(1.5)

    spike = _params_with_grad(1.0e6)
    assert clipper.clip_(spike, distributed=False) == pytest.approx(1.0e6)
    clipped = float(global_grad_norm(spike, distributed=False))
    assert clipped < 3.0


def test_spike_does_not_loosen_threshold():
    """The EMA absorbs the post-clip norm, so one spike barely moves it."""
    clipper = AdaptiveGradClipper(factor=2.0, ema_decay=0.9, warmup_steps=5)
    for _ in range(20):
        clipper.clip_(_params_with_grad(1.0), distributed=False)
    before = clipper.threshold()
    clipper.clip_(_params_with_grad(1.0e6), distributed=False)
    assert clipper.threshold() <= before * 1.2


def test_nonfinite_norm_left_alone():
    clipper = AdaptiveGradClipper(warmup_steps=0)
    params = _params_with_grad(1.0)
    params[0].grad[0] = float("nan")
    state = clipper.state_dict()
    assert math.isnan(clipper.clip_(params, distributed=False))
    assert clipper.state_dict() == state


def test_state_roundtrip(tmp_path):
    clipper = AdaptiveGradClipper(warmup_steps=2)
    for norm in (1.0, 2.0, 3.0):
        clipper.clip_(_params_with_grad(norm), distributed=False)
    clipper.save(str(tmp_path))
    resumed = AdaptiveGradClipper(warmup_steps=2)
    assert resumed.load(str(tmp_path))
    assert resumed.state_dict() == clipper.state_dict()
    assert resumed.threshold() == pytest.approx(clipper.threshold())
    assert not AdaptiveGradClipper().load(str(tmp_path / "missing"))


def test_from_config_and_validation():
    clipper = AdaptiveGradClipper.from_config({"dynamic_grad_clip": {"factor": 3.0, "warmup_steps": 10}})
    assert clipper.factor == 3.0 and clipper.warmup_steps == 10
    assert AdaptiveGradClipper.from_config({}).factor == 2.0
    with pytest.raises(ValueError):
        AdaptiveGradClipper(factor=1.0)
    with pytest.raises(ValueError):
        AdaptiveGradClipper(ema_decay=1.0)
