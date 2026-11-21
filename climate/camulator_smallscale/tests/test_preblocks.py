"""StartStatePerturbation: seeded start-state perturbations for twin runs (camulator_smallscale.preblocks)."""

import numpy as np
import pytest
import torch
import xarray as xr

from camulator_smallscale.preblocks import StartStatePerturbation
from camulator_smallscale.spectral import SphericalSpectra, band_sums

B, L, T, H, W = 2, 3, 1, 33, 64
T3 = "CESM/prognostic/3d/T"
PS = "CESM/prognostic/2d/PS"
SST = "CESM/dynamic_forcing/2d/SST"


def _batch(init_ns=0):
    g = torch.Generator().manual_seed(0)
    state = {
        T3: 250.0 + torch.randn(B, L, T, H, W, generator=g),
        PS: 1.0e5 + torch.randn(B, 1, T, H, W, generator=g),
        SST: 290.0 + torch.randn(B, 1, T, H, W, generator=g),
    }
    return {
        "input": {"CESM": dict(state)},
        "target": {"CESM": {T3: state[T3].clone()}},
        "metadata": {"CESM": {"input_datetime": torch.tensor([init_ns])}},
    }


def _delta(block, batch, key=T3):
    return block(batch)["input"]["CESM"][key] - batch["input"]["CESM"][key]


def test_zero_amplitude_is_a_no_op():
    batch = _batch()
    out = StartStatePerturbation([T3, PS], amplitude=0.0)(batch)
    assert torch.equal(out["input"]["CESM"][T3], batch["input"]["CESM"][T3])


def test_white_noise_amplitude_and_scope():
    batch = _batch()
    block = StartStatePerturbation([T3], amplitude=0.01)
    out = block(batch)
    delta = out["input"]["CESM"][T3] - batch["input"]["CESM"][T3]
    assert abs(delta.std().item() / 0.01 - 1) < 0.05
    assert torch.equal(out["input"]["CESM"][PS], batch["input"]["CESM"][PS])  # not listed
    assert torch.equal(out["target"]["CESM"][T3], batch["target"]["CESM"][T3])  # start state only
    assert out["input"]["CESM"] is not batch["input"]["CESM"]  # the caller's dict is not mutated


def test_same_seed_and_start_time_give_the_same_perturbation():
    a = _delta(StartStatePerturbation([T3], amplitude=0.1, seed=3), _batch(init_ns=10))
    b = _delta(StartStatePerturbation([T3], amplitude=0.1, seed=3), _batch(init_ns=10))
    other_time = _delta(StartStatePerturbation([T3], amplitude=0.1, seed=3), _batch(init_ns=20))
    other_seed = _delta(StartStatePerturbation([T3], amplitude=0.1, seed=4), _batch(init_ns=10))
    fixed = _delta(StartStatePerturbation([T3], amplitude=0.1, seed=3, per_init_seed=False), _batch(init_ns=10))
    fixed_later = _delta(StartStatePerturbation([T3], amplitude=0.1, seed=3, per_init_seed=False), _batch(init_ns=20))
    assert torch.equal(a, b)
    assert not torch.equal(a, other_time) and not torch.equal(a, other_seed)
    assert torch.equal(fixed, fixed_later)


def test_amplitude_per_variable_and_partial_keys():
    batch = _batch()
    out = StartStatePerturbation(["CESM/prognostic"], amplitude={"T": 0.5, "PS": 0.0})(batch)
    assert not torch.equal(out["input"]["CESM"][T3], batch["input"]["CESM"][T3])
    assert torch.equal(out["input"]["CESM"][PS], batch["input"]["CESM"][PS])
    assert torch.equal(out["input"]["CESM"][SST], batch["input"]["CESM"][SST])


def test_band_noise_lives_in_its_band_with_the_requested_rms():
    batch = _batch()
    # float64 state: in float32, adding 0.2 K to ~250 K rounds at ~1e-5 and that white round-off leaks out of the band
    batch["input"]["CESM"][T3] = batch["input"]["CESM"][T3].double()
    block = StartStatePerturbation([T3], amplitude=0.2, kind="band", lmin=10, lmax=20)
    delta = _delta(block, batch).double()
    spectra = SphericalSpectra(H, W, dtype=torch.float64)
    power = spectra.power(delta)
    inside, outside = band_sums(power, [(10, 20)]).sum(), power.sum() - band_sums(power, [(10, 20)]).sum()
    # Not exactly 0: on this 33-latitude grid, measuring wavenumber l is exact only for content up to degree 32 - l,
    # so band power aliases slightly into higher degrees in the analysis (the same limit as in test_spectral).
    assert outside / inside < 1e-4
    rms = spectra.area_mean_square(delta).sqrt()
    torch.testing.assert_close(rms, torch.full_like(rms, 0.2), rtol=1e-4, atol=0)


def test_file_perturbation(tmp_path):
    field = np.random.default_rng(0).standard_normal((L, H, W))
    path = tmp_path / "perturbation.nc"
    xr.Dataset({"T": (("level", "lat", "lon"), field)}).to_netcdf(path)
    delta = _delta(StartStatePerturbation([T3], amplitude=0.1, kind="file", path=str(path)), _batch())
    expected = 0.1 * torch.as_tensor(field, dtype=torch.float32)[None, :, None].expand(B, L, T, H, W)
    torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-4)
    with pytest.raises(KeyError, match="PS"):
        StartStatePerturbation([PS], amplitude=1.0, kind="file", path=str(path))(_batch())


def test_bad_arguments():
    with pytest.raises(ValueError, match="kind"):
        StartStatePerturbation([T3], amplitude=1.0, kind="red")
    with pytest.raises(ValueError, match="lmin"):
        StartStatePerturbation([T3], amplitude=1.0, kind="band", lmin=20, lmax=10)
    with pytest.raises(ValueError, match="path"):
        StartStatePerturbation([T3], amplitude=1.0, kind="file")
