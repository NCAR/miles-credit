"""Tests for scale_sdl_noise (the inference noise_scale override for SDL ensemble models)."""

import pytest
import torch

from credit.models.wxformer.stochastic_decomposition_layer import StochasticDecompositionLayer, scale_sdl_noise


def _model():
    return torch.nn.Sequential(
        StochasticDecompositionLayer(noise_dim=4, feature_channels=3, noise_factor=0.1),
        torch.nn.Sequential(StochasticDecompositionLayer(noise_dim=4, feature_channels=3, noise_factor=0.2)),
        torch.nn.Linear(3, 3),
    )


def _noise_factors(model):
    return [m.noise_factor.item() for m in model.modules() if isinstance(m, StochasticDecompositionLayer)]


def test_scales_every_nested_layer():
    model = _model()

    assert scale_sdl_noise(model, 0.5) == 2
    assert _noise_factors(model) == pytest.approx([0.05, 0.1])


@pytest.mark.parametrize("noise_scale", [None, 1.0])
def test_none_or_one_is_a_no_op(noise_scale):
    model = _model()

    assert scale_sdl_noise(model, noise_scale) == 0
    assert _noise_factors(model) == pytest.approx([0.1, 0.2])


def test_zero_removes_injected_noise():
    layer = StochasticDecompositionLayer(noise_dim=4, feature_channels=3)
    scale_sdl_noise(layer, 0.0)
    feature_map = torch.randn(2, 3, 5, 5)
    noise = torch.randn(2, 4)

    torch.testing.assert_close(layer(feature_map, noise), feature_map)


def test_model_without_sdl_layers_warns(caplog):
    with caplog.at_level("WARNING"):
        assert scale_sdl_noise(torch.nn.Linear(3, 3), 0.5) == 0
    assert "no StochasticDecompositionLayer" in caplog.text
