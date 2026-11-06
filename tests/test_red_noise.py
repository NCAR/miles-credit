"""Red-in-time noise: the AR(1) generators, red SDL noise (model.red_noise) and the targeted_red_noise postblock."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import xarray as xr

from credit.ensemble.red_noise import (
    AR1Noise,
    SphericalAR1Pattern,
    ar1_phi,
    red_noise_errors,
    reset_noise_state,
    spectral_truncation_fraction,
)
from credit.models.wxformer.crossformer_ensemble import CrossFormerWithNoise
from credit.postblock import build_postblocks
from credit.postblock.red_noise import TargetedRedNoise, targeted_red_noise_errors

EARTH_RADIUS_KM = 6371.0


def _lag_corr(series: torch.Tensor, lag: int) -> float:
    """Pooled lag correlation of a (time, ...) series of zero-mean, unit-variance draws."""
    a, b = series[:-lag].flatten(), series[lag:].flatten()
    return float((a * b).mean() / (a.std() * b.std()))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def test_ar1_phi():
    assert ar1_phi(None, 6) == 0.0
    assert ar1_phi(0, 6) == 0.0
    assert ar1_phi(24, 6) == pytest.approx(math.exp(-0.25))


def test_ar1_noise_starts_stationary_and_follows_the_recursion():
    phi = 0.7
    torch.manual_seed(0)
    e1, e2 = torch.randn(3, 5), torch.randn(3, 5)
    torch.manual_seed(0)
    noise = AR1Noise(phi)
    s1 = noise.sample((3, 5), "cpu")
    s2 = noise.sample((3, 5), "cpu")
    assert torch.equal(s1, e1)  # the first draw is the white draw itself
    torch.testing.assert_close(s2, phi * e1 + math.sqrt(1 - phi**2) * e2)

    noise.reset()
    torch.manual_seed(0)
    assert torch.equal(noise.sample((3, 5), "cpu"), e1)
    # a new shape (e.g. a smaller last batch) restarts the stream
    assert noise.sample((2, 5), "cpu").shape == (2, 5)


def test_ar1_noise_marginal_and_autocorrelation():
    torch.manual_seed(1)
    noise = AR1Noise(0.8)
    series = torch.stack([noise.sample((4000,), "cpu").clone() for _ in range(60)])
    assert series.var(dim=1).mean().item() == pytest.approx(1.0, abs=0.03)
    assert series[0].var().item() == pytest.approx(1.0, abs=0.08)  # stationary from the first step
    assert _lag_corr(series, 1) == pytest.approx(0.8, abs=0.02)
    assert _lag_corr(series, 3) == pytest.approx(0.8**3, abs=0.02)


def test_ar1_noise_rejects_bad_phi():
    with pytest.raises(ValueError, match="phi"):
        AR1Noise(1.0)


@pytest.fixture(scope="module")
def pattern_draws():
    torch.manual_seed(2)
    pattern = SphericalAR1Pattern(48, 96, length_scale_km=1500.0, phi=0.0)
    return pattern.innovation(3000, "cpu")


def test_spherical_pattern_has_unit_variance_everywhere(pattern_draws):
    var = pattern_draws.var(dim=0)
    assert pattern_draws.mean().abs().item() < 0.02
    assert var[0].mean().item() == pytest.approx(1.0, abs=0.08)  # pole
    assert var[-1].mean().item() == pytest.approx(1.0, abs=0.08)  # other pole
    assert var[22:26].mean().item() == pytest.approx(1.0, abs=0.04)  # equator
    assert var.mean().item() == pytest.approx(1.0, abs=0.03)


def test_spherical_pattern_correlation_length(pattern_draws):
    nlat, nlon = pattern_draws.shape[1:]
    length_scale = 1500.0
    row = nlat // 2  # equiangular nodes include both poles: this row is 1.9 deg from the equator
    lat = math.radians(90 - 180 * row / (nlat - 1))
    eq = pattern_draws[:, row]
    for k in (1, 2, 3):
        d = EARTH_RADIUS_KM * math.cos(lat) * 2 * math.pi * k / nlon
        corr = float((eq * eq.roll(k, dims=-1)).mean() / eq.var())
        assert corr == pytest.approx(math.exp(-(d**2) / (2 * length_scale**2)), abs=0.04)
    for k in (2, 4):  # meridional, at about 45 N
        i = nlat // 4
        d = EARTH_RADIUS_KM * math.pi * k / (nlat - 1)
        a, b = pattern_draws[:, i], pattern_draws[:, i + k]
        corr = float((a * b).mean() / (a.std() * b.std()))
        assert corr == pytest.approx(math.exp(-(d**2) / (2 * length_scale**2)), abs=0.04)


def test_spherical_pattern_autocorrelation():
    torch.manual_seed(3)
    phi = ar1_phi(26.9, 6)  # ~0.8
    pattern = SphericalAR1Pattern(24, 48, length_scale_km=2000.0, phi=phi)
    series = torch.stack([pattern.sample(16, "cpu").clone() for _ in range(200)])
    assert series[0].var().item() == pytest.approx(1.0, abs=0.15)
    assert _lag_corr(series, 1) == pytest.approx(phi, abs=0.03)
    assert _lag_corr(series, 4) == pytest.approx(phi**4, abs=0.05)
    pattern.reset()
    assert pattern.noise.state is None


def test_spherical_pattern_rejects_length_scale_below_grid():
    assert spectral_truncation_fraction(24, 48, 100.0) > 0.01
    with pytest.raises(ValueError, match="too short"):
        SphericalAR1Pattern(24, 48, length_scale_km=100.0, phi=0.0)


# ---------------------------------------------------------------------------
# Red SDL noise (R1)
# ---------------------------------------------------------------------------

_ARCH = dict(
    image_height=32,
    image_width=64,
    frames=1,
    channels=2,
    surface_channels=1,
    input_only_channels=1,
    output_only_channels=1,
    levels=2,
    dim=(32, 64, 128, 256),
    depth=(1, 1, 1, 1),
    global_window_size=(4, 2, 1, 1),
    local_window_size=2,
    cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
    cross_embed_strides=(2, 2, 2, 2),
    padding_conf={"activate": True, "mode": "earth", "pad_lat": [16, 16], "pad_lon": [16, 16]},
    use_spectral_norm=False,  # unconverged power iterations would blow the outputs up past the noise
    noise_latent_dim=8,
)
_RED = {"tau_hours": 6 / math.log(2), "timestep_hours": 6}  # phi = 0.5


def _reference_forward(model, x):
    """CrossFormerWithNoise.forward and StochasticDecompositionLayer.forward as they were before red noise.

    Copied from commit 36b89c2c, so the white default can be checked bit for bit.
    """

    def sdl(layer, feature_map, noise):
        batch, channels, height, width = feature_map.shape
        pixel_noise = layer.noise_factor * torch.randn(batch, channels, height, width, device=feature_map.device)
        style = layer.noise_transform(noise).view(batch, channels, 1, 1)
        return feature_map + pixel_noise * style * layer.modulation

    if model.correlated:
        noise = torch.randn(x.size(0), model.noise_latent_dim, device=x.device)
    x_copy = None
    if model.use_post_block:
        x_copy = x.clone().detach()
    if model.use_padding:
        x = model.padding_opt.pad(x)
    if model.patch_width > 1 and model.patch_height > 1:
        x = model.cube_embedding(x)
    if model.frames > 1:
        b, c, t, h, w = x.shape
        x = x.reshape(b, c * t, h, w)
    else:
        x = x.squeeze(2)
    encodings = []
    for k, (cel, transformer) in enumerate(model.layers):
        x = cel(x)
        x = transformer(x)
        if model.encoder_noise and k < len(model.layers) - 1:
            if not model.correlated:
                noise = torch.randn(x.size(0), model.noise_latent_dim, device=x.device)
            x = sdl(model.encoder_noise_layers[k], x, noise)
        encodings.append(x)
    for up_block, inject, skip in (
        (model.up_block1, model.noise_inject1, encodings[2]),
        (model.up_block2, model.noise_inject2, encodings[1]),
        (model.up_block3, model.noise_inject3, encodings[0]),
    ):
        x = up_block(x)
        if not model.correlated:
            noise = torch.randn(x.size(0), model.noise_latent_dim, device=x.device)
        x = sdl(inject, x, noise)
        x = torch.cat([x, skip], dim=1)
    x = model.up_block4(x)
    if model.use_padding:
        x = model.padding_opt.unpad(x)
    if model.use_interp:
        x = F.interpolate(x, size=(model.image_height, model.image_width), mode="bilinear")
    b, _, h, w = x.shape
    x = x.view(b, model.base_output_channels, model.output_frames, h, w)
    if model.use_post_block:
        x = model.postblock({"y_pred": x, "x": x_copy})
    return x


def _input(batch=2):
    torch.manual_seed(10)
    return torch.randn(batch, 2 * 2 + 1 + 1, 1, 32, 64)


_NOISE_MODES = [
    dict(encoder_noise=False, correlated=False),
    dict(encoder_noise=True, correlated=False),
    dict(encoder_noise=True, correlated=True),
]


@pytest.mark.parametrize("mode", _NOISE_MODES)
def test_white_sdl_is_bit_identical_to_the_previous_forward(mode):
    torch.manual_seed(0)
    model = CrossFormerWithNoise(**_ARCH, **mode).eval()
    x = _input()
    with torch.no_grad():
        for seed in (1, 2):  # several steps: no hidden state may build up
            torch.manual_seed(seed)
            expected = _reference_forward(model, x)
            after_expected = torch.randn(1)
            torch.manual_seed(seed)
            actual = model(x)
            assert torch.equal(actual, expected)
            assert torch.equal(torch.randn(1), after_expected)  # same random numbers consumed


@pytest.mark.parametrize("mode", _NOISE_MODES)
def test_red_sdl_first_step_equals_white_and_consumes_the_same_draws(mode):
    torch.manual_seed(0)
    white = CrossFormerWithNoise(**_ARCH, **mode).eval()
    red = CrossFormerWithNoise(**_ARCH, **mode, red_noise=_RED).eval()
    assert red.state_dict().keys() == white.state_dict().keys()  # checkpoints are unchanged
    red.load_state_dict(white.state_dict())
    x = _input()
    with torch.no_grad():
        torch.manual_seed(5)
        y_white = white(x)
        after_white = torch.randn(1)
        torch.manual_seed(5)
        y_red = red(x)
        assert torch.equal(y_red, y_white)
        assert torch.equal(torch.randn(1), after_white)
        # the second step draws the same innovations, but red noise remembers the first draw
        torch.manual_seed(6)
        y_white = white(x)
        torch.manual_seed(6)
        assert not torch.allclose(red(x), y_white)


def test_red_sdl_streams_follow_ar1_and_reset():
    torch.manual_seed(0)
    model = CrossFormerWithNoise(**_ARCH, encoder_noise=False, red_noise=_RED)
    model.train()  # trainable noise layers: the carried state must still hold no graph
    x = _input(batch=4)
    layer = model.noise_inject3
    states, latents = [], []
    for _ in range(3):
        model(x).sum().backward()
        states.append(layer.pixel_ar1.state.clone())
        latents.append(model._latent_ar1[("decoder", 2)].state.clone())
        assert not layer.pixel_ar1.state.requires_grad
        assert not model._latent_ar1[("decoder", 2)].state.requires_grad
    pixel = torch.stack(states)
    assert pixel.var().item() == pytest.approx(1.0, abs=0.03)
    assert _lag_corr(pixel, 1) == pytest.approx(0.5, abs=0.03)
    assert latents[0].shape == (4, _ARCH["noise_latent_dim"])

    reset_noise_state(model)
    assert layer.pixel_ar1.state is None
    assert all(stream.state is None for stream in model._latent_ar1.values())


def test_red_sdl_per_point_tau_and_apply_to():
    model = CrossFormerWithNoise(
        **_ARCH,
        encoder_noise=False,
        red_noise={"tau_hours": [480, None, 0], "timestep_hours": 6, "apply_to": ["pixel"]},
    )
    assert model.noise_inject1.pixel_ar1.phi == pytest.approx(math.exp(-6 / 480))
    assert model.noise_inject2.pixel_ar1 is None and model.noise_inject3.pixel_ar1 is None
    assert model._latent_ar1 == {}  # latents stay white


@pytest.mark.parametrize(
    "red_noise, mode, match",
    [
        ({"timestep_hours": 6}, {}, "tau_hours is required"),
        ({"tau_hours": -1, "timestep_hours": 6}, {}, "tau_hours must be"),
        ({"tau_hours": [1, 2], "timestep_hours": 6}, {}, "3 entries"),
        ({"tau_hours": [1, 2, 3], "timestep_hours": 6}, {"encoder_noise": True}, "decoder only"),
        ({"tau_hours": 24}, {}, "timestep_hours"),
        ({"tau_hours": 24, "timestep_hours": 6, "apply_to": ["style"]}, {}, "apply_to"),
        ({"tau_hours": 24, "timestep_hours": 6, "rho": 0.9}, {}, "unknown keys"),
    ],
)
def test_red_sdl_rejects_bad_config(red_noise, mode, match):
    assert any(match in e for e in red_noise_errors(red_noise, **{"encoder_noise": False, **mode}))
    with pytest.raises(ValueError, match="model.red_noise"):
        CrossFormerWithNoise(**_ARCH, **{"encoder_noise": False, **mode}, red_noise=red_noise)


# ---------------------------------------------------------------------------
# Targeted red noise postblock (R2)
# ---------------------------------------------------------------------------

_H, _W, _LEV = 24, 48, 4
_TREFHT = "SRC/prognostic/2d/TREFHT"
_T = "SRC/prognostic/3d/T"
_Q = "SRC/prognostic/3d/Qtot"
_ICE = "SRC/dynamic_forcing/2d/ICEFRAC"


@pytest.fixture
def maps(tmp_path):
    """Land on the western half; a sigma map of 2.0 for TREFHT."""
    land = np.zeros((_H, _W), np.float32)
    land[:, : _W // 2] = 1.0
    path = tmp_path / "maps.nc"
    xr.Dataset(
        {"LANDFRAC": (("lat", "lon"), land), "TREFHT_sigma": (("lat", "lon"), np.full((_H, _W), 2.0, np.float32))}
    ).to_netcdf(path)
    return str(path)


def _block(maps, tau_days=1.1205, **kw):
    fields = [
        {"var": _TREFHT, "sigma": {"path": maps, "var": "TREFHT_sigma"}, "tau_days": tau_days, "pattern": "sfc_T"},
        {"var": _T, "levels": [-1], "sigma": 0.5, "tau_days": tau_days, "pattern": "sfc_T"},
        {"var": _Q, "levels": [-1], "sigma": 1e-4, "tau_days": tau_days},
    ]
    args = dict(fields=fields, length_scale_km=2000.0, land_file=maps, land_var="LANDFRAC", ice_var=_ICE)
    return TargetedRedNoise(**{**args, **kw})


def _batch(batch=3, ice_batch=None, requires_grad=False):
    ice = torch.zeros(ice_batch or batch, 1, 1, _H, _W)
    ice[:, :, :, :4, _W // 2 :] = 1.0  # sea ice in the eastern (ocean) half, polar rows
    y = {
        _TREFHT: torch.zeros(batch, 1, 1, _H, _W, requires_grad=requires_grad),
        _T: torch.zeros(batch, _LEV, 1, _H, _W),
        _Q: torch.zeros(batch, _LEV, 1, _H, _W),
    }
    return {"y_processed": {"SRC": y}, "x_physical": {"SRC": {_ICE: ice}}}


def test_targeted_red_noise_is_masked_shared_and_on_the_chosen_level(maps):
    block = _block(maps)
    out = block(_batch())["y_processed"]["SRC"]
    trefht, t, q = out[_TREFHT][:, 0, 0], out[_T][:, :, 0], out[_Q][:, :, 0]

    open_ocean = torch.zeros(_H, _W, dtype=torch.bool)
    open_ocean[4:, _W // 2 :] = True
    assert (trefht[:, open_ocean] == 0).all() and (t[:, -1][:, open_ocean] == 0).all()
    assert (trefht[:, ~open_ocean] != 0).all()  # land and sea ice
    assert (t[:, :-1] == 0).all() and (q[:, :-1] == 0).all()  # only the lowest level
    # TREFHT and T share one pattern: their perturbations differ only by sigma (2.0 vs 0.5)
    torch.testing.assert_close(t[:, -1], 0.25 * trefht)
    assert list(block.patterns) == ["sfc_T", f"{_Q}[-1]"]
    assert block.state_dict() == {}  # nothing stateful or static lands in a checkpoint


def test_targeted_red_noise_amplitude_and_memory(maps):
    torch.manual_seed(4)
    block = _block(maps)
    phi = ar1_phi(1.1205 * 24, 6)
    series = []
    for _ in range(150):
        series.append(block(_batch(batch=8))["y_processed"]["SRC"][_TREFHT][:, 0, 0, :, : _W // 2].clone())
    series = torch.stack(series)  # land only: sigma = 2.0, mask = 1
    assert series.std().item() == pytest.approx(2.0, rel=0.05)
    assert series[0].std().item() == pytest.approx(2.0, rel=0.15)  # stationary from the first step
    assert _lag_corr(series, 1) == pytest.approx(phi, abs=0.03)
    assert _lag_corr(series, 5) == pytest.approx(phi**5, abs=0.05)

    reset_noise_state(torch.nn.ModuleDict({"red": block}))
    assert all(pattern.noise.state is None for pattern in block.patterns.values())


def test_targeted_red_noise_keeps_the_graph_and_repeats_ice_for_batch_ensembles(maps):
    block = _block(maps)
    batch = _batch(batch=4, ice_batch=2, requires_grad=True)
    leaf = batch["y_processed"]["SRC"][_TREFHT]
    out = block(batch)["y_processed"]["SRC"][_TREFHT]
    out.sum().backward()
    assert torch.equal(leaf.grad, torch.ones_like(leaf))


def test_targeted_red_noise_without_mask_is_global(maps):
    block = _block(maps, land_file=None, ice_var=None)
    out = block(_batch())["y_processed"]["SRC"][_TREFHT]
    assert (out != 0).all()


def test_targeted_red_noise_checks_the_grid(maps):
    block = _block(maps)
    batch = _batch()
    batch["y_processed"]["SRC"] = {k: v[..., :-1] for k, v in batch["y_processed"]["SRC"].items()}
    with pytest.raises(ValueError, match="grid"):
        block(batch)


def test_targeted_red_noise_builds_from_config(maps):
    conf = {
        "postblocks": {
            "per_step": {
                "red": {
                    "type": "targeted_red_noise",
                    "args": {
                        "fields": [{"var": _TREFHT, "sigma": 0.3, "tau_days": 11}],
                        "land_file": maps,
                        "ice_var": _ICE,
                    },
                }
            }
        }
    }
    assert isinstance(build_postblocks(conf)["red"], TargetedRedNoise)


@pytest.mark.parametrize(
    "fields, match",
    [
        ([], "non-empty list"),
        ([{"var": _TREFHT, "sigma": 0.3}], "tau_days"),
        ([{"var": _TREFHT, "sigma": -1, "tau_days": 11}], "sigma must be"),
        ([{"var": _TREFHT, "sigma": {"path": "x.nc"}, "tau_days": 11}], "{path, var}"),
        ([{"var": "TREFHT", "sigma": 0.3, "tau_days": 11}], "variable key"),
        ([{"var": _T, "levels": "31", "sigma": 0.3, "tau_days": 11}], "levels"),
        (
            [{"var": _T, "sigma": 0.3, "tau_days": 11}, {"var": _T, "levels": [3], "sigma": 0.3, "tau_days": 11}],
            "twice",
        ),
        (
            [
                {"var": _TREFHT, "sigma": 0.3, "tau_days": 11, "pattern": "p"},
                {"var": _T, "levels": [3], "sigma": 0.3, "tau_days": 16, "pattern": "p"},
            ],
            "must share tau_days",
        ),
        ([{"var": _TREFHT, "sigma": 0.3, "tau_days": 11, "amplitude": 1}], "unknown keys"),
    ],
)
def test_targeted_red_noise_rejects_bad_fields(fields, match):
    assert any(match in e for e in targeted_red_noise_errors(fields))
    with pytest.raises(ValueError, match="targeted_red_noise"):
        TargetedRedNoise(fields=fields)


def test_run_forecast_resets_noise_state_before_the_first_step(monkeypatch):
    from credit.trainers import rollout_utils

    calls = []
    monkeypatch.setattr(rollout_utils, "reset_noise_state", lambda *modules: calls.append(modules))
    model, postblocks = torch.nn.Linear(1, 1), torch.nn.ModuleDict()
    with pytest.raises(StopIteration):  # stop at the IC batch: the reset must already have happened
        rollout_utils.run_forecast(
            conf={"data": {"timestep": "6h"}},
            n_steps=1,
            save_dir="",
            ic_preblocks=None,
            step_preblocks=None,
            step_postblocks=postblocks,
            rollout_postblocks=None,
            model=model,
            batch_iter=iter([]),
            device="cpu",
            pool=None,
            save_output_fn=None,
        )
    assert calls == [(model, postblocks)]
