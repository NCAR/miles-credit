"""AMSEHFLoss: HFLoss plus the AMSE adjustment (camulator_smallscale.losses)."""

import numpy as np
import pytest
import torch
import xarray as xr

from camulator_gen2.losses import HFLoss
from camulator_smallscale.losses import AMSEHFLoss
from credit.losses.base import BaseLoss

B, C, H, W = 4, 6, 33, 64  # an equiangular grid with both poles, like CAMulator's
L = 33
MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
CHANNEL_MAP = {
    "CESM/prognostic/3d/T": {"slice": slice(0, 4)},
    "CESM/diagnostic/2d/PRECT": {"slice": slice(4, 6)},
}


@pytest.fixture(scope="module")
def lat_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("statics") / "statics.nc"
    lat = np.linspace(-90.0, 90.0, H).astype(np.float32)
    xr.Dataset(coords={"latitude": lat, "longitude": np.arange(W) * 360.0 / W}).to_netcdf(path)
    return str(path)


def _coefficients(seed, slope=-2.0):
    """Random real-field coefficients with power ~ l**slope."""
    g = torch.Generator().manual_seed(seed)
    coeffs = torch.randn(B, C, L, L, dtype=torch.complex64, generator=g)
    coeffs = torch.tril(coeffs) * torch.arange(1, L + 1, dtype=torch.float32)[:, None] ** ((slope - 1) / 2)
    coeffs[..., 0] = coeffs[..., 0].real.to(torch.complex64)
    return coeffs


def _field(coeffs):
    from torch_harmonics import InverseRealSHT

    return InverseRealSHT(H, W, lmax=L, mmax=L, grid="equiangular")(coeffs)


def _same_spectrum_random_phases(coeffs, seed):
    """A field with exactly the power of ``coeffs`` at every (l, m) and independent phases (decorrelated)."""
    g = torch.Generator().manual_seed(seed)
    phases = torch.exp(1j * 2 * np.pi * torch.rand(coeffs.shape, generator=g)).to(torch.complex64)
    out = coeffs.abs() * phases
    out[..., 0] = coeffs[..., 0].abs() * torch.sign(torch.randn(coeffs[..., 0].shape, generator=g))
    return _field(out)


def _batch(pred, target, penalty=0.0):
    return {
        "y": target,
        "y_pred_hf": pred,
        "y_pred": pred,
        "water_conservation_loss": torch.tensor(penalty),
        "metadata": {"target": {"_channel_map": CHANNEL_MAP}},
    }


@pytest.fixture(scope="module")
def truth():
    return _field(_coefficients(0))


def test_is_an_hfloss_and_a_baseloss(lat_file):
    loss = AMSEHFLoss(lat_file)
    assert isinstance(loss, HFLoss) and isinstance(loss, BaseLoss)


def test_equals_hfloss_bit_for_bit_when_prediction_is_the_truth(lat_file, truth):
    batch = _batch(truth.clone(), truth, penalty=0.3)
    assert torch.equal(AMSEHFLoss(lat_file)(batch), HFLoss(lat_file)(batch))


def test_matches_hfloss_when_spectra_match(lat_file, truth):
    shifted = torch.roll(truth, 9, dims=-1)  # same spectrum, different phases
    batch = _batch(shifted, truth)
    hf = HFLoss(lat_file, water_penalty_weight=0)(batch)
    amse = AMSEHFLoss(lat_file, water_penalty_weight=0)(batch)
    torch.testing.assert_close(amse, hf, rtol=1e-5, atol=0)


def test_validation_is_exactly_hfloss(lat_file, truth):
    smoothed = 0.5 * truth
    batch = _batch(smoothed, truth, penalty=0.3)
    with torch.no_grad():
        assert torch.equal(AMSEHFLoss(lat_file)(batch), HFLoss(lat_file)(batch))


def test_damping_is_penalized(lat_file, truth):
    batch = _batch(0.6 * truth, truth)
    hf = HFLoss(lat_file, water_penalty_weight=0)(batch)
    amse_loss = AMSEHFLoss(lat_file, water_penalty_weight=0)
    amse = amse_loss(batch)
    assert amse > hf
    assert all(amse_loss.last_var_losses[f"{name}_amse"] > 0 for name in ("T", "PRECT"))


def test_decorrelated_prediction_is_best_at_full_amplitude(lat_file):
    """The defining AMSE property: MSE damps a decorrelated prediction toward zero, AMSE keeps its amplitude."""
    coeffs = _coefficients(1, slope=-3.0)
    truth = _field(coeffs)
    decorrelated = _same_spectrum_random_phases(coeffs, seed=2)
    alphas = np.round(np.arange(0.5, 1.51, 0.05), 2)
    hf = HFLoss(lat_file, water_penalty_weight=0)
    amse = AMSEHFLoss(lat_file, water_penalty_weight=0)
    hf_values = [hf(_batch(a * decorrelated, truth)).item() for a in alphas]
    amse_values = [amse(_batch(a * decorrelated, truth)).item() for a in alphas]
    assert alphas[int(np.argmin(hf_values))] == 0.5  # MSE: as small as the scan allows
    assert abs(alphas[int(np.argmin(amse_values))] - 1.0) <= 0.05


def test_channel_weights_scale_the_adjustment(lat_file, truth):
    batch = _batch(0.6 * truth, truth)
    plain = AMSEHFLoss(lat_file, water_penalty_weight=0)
    weighted = AMSEHFLoss(lat_file, water_penalty_weight=0, variable_weights={"PRECT": 3.0})
    plain(batch)
    weighted(batch)
    assert weighted.last_var_losses["PRECT_amse"] == pytest.approx(3 * plain.last_var_losses["PRECT_amse"], rel=1e-5)
    assert weighted.last_var_losses["T_amse"] == pytest.approx(plain.last_var_losses["T_amse"], rel=1e-5)


def test_amse_variables_restricts_the_adjustment(lat_file, truth):
    batch = _batch(0.6 * truth, truth)
    only_t = AMSEHFLoss(lat_file, water_penalty_weight=0, amse_variables=["T"])
    total = only_t(batch)
    assert only_t.last_var_losses["PRECT_amse"] == 0
    assert only_t.last_var_losses["T_amse"] > 0
    hf = HFLoss(lat_file, water_penalty_weight=0)(batch)
    adjustment = only_t.channel_adjustment(batch["y_pred_hf"], batch["y"])
    torch.testing.assert_close(total, hf + (adjustment * torch.tensor([1, 1, 1, 1, 0, 0])).mean())
    with pytest.raises(KeyError, match="QFLX"):
        AMSEHFLoss(lat_file, amse_variables=["QFLX"])(batch)


def test_needs_an_mse(lat_file):
    with pytest.raises(ValueError, match="MSE"):
        AMSEHFLoss(lat_file, train_loss="mae")


def test_gradients_are_finite_when_the_prediction_has_no_power(lat_file, truth):
    pred = torch.zeros_like(truth, requires_grad=True)
    AMSEHFLoss(lat_file, water_penalty_weight=0)(_batch(pred, truth)).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


@pytest.mark.skipif(not MPS, reason="needs Apple MPS")
def test_mps_matches_cpu(lat_file, truth):
    pred = 0.7 * torch.roll(truth, 5, dims=-1)
    cpu = AMSEHFLoss(lat_file, water_penalty_weight=0)(_batch(pred, truth))
    mps_batch = _batch(pred.to("mps"), truth.to("mps"))
    mps = AMSEHFLoss(lat_file, water_penalty_weight=0)(mps_batch).cpu()
    torch.testing.assert_close(mps, cpu, rtol=1e-4, atol=0)
