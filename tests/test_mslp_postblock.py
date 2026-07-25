"""Tests for the MSLP postblock: mslp_from_surface_pressure and MSLPDiagnostic."""

import torch
import pytest

from credit.postblock.mslp import mslp_from_surface_pressure, MSLPDiagnostic
from credit.physics_constants import GRAVITY, RDGAS

_LAPSE_RATE = 0.0065


# ---------------------------------------------------------------------------
# mslp_from_surface_pressure — unit tests
# ---------------------------------------------------------------------------


def _scalar(val):
    return torch.tensor([[val]], dtype=torch.float32)


def test_near_flat_returns_surface_pressure():
    """When surface geopotential is negligible, MSLP == SP."""
    sp = _scalar(101325.0)
    t = _scalar(288.0)
    phis = _scalar(0.0)  # sea level
    mslp = mslp_from_surface_pressure(sp, t, phis)
    assert torch.allclose(mslp, sp, rtol=1e-5)


def test_mslp_greater_than_sp_for_elevated_surface():
    """Above sea level, MSLP must exceed SP (pressure increases downward)."""
    sp = _scalar(85000.0)
    t = _scalar(275.0)
    phis = _scalar(1500.0 * GRAVITY)  # 1500 m elevation
    mslp = mslp_from_surface_pressure(sp, t, phis)
    assert mslp.item() > sp.item()


def test_mslp_reasonable_magnitude():
    """MSLP should be in a physically reasonable range (900–1100 hPa)."""
    sp = _scalar(85000.0)
    t = _scalar(275.0)
    phis = _scalar(1500.0 * GRAVITY)
    mslp = mslp_from_surface_pressure(sp, t, phis)
    assert 90000.0 < mslp.item() < 110000.0


def test_warm_surface_branch():
    """temp > 290.5 K triggers alpha=0 and T_eff = 0.5*(290.5+T)."""
    sp = _scalar(101325.0)
    t = _scalar(295.0)  # warm, > 290.5
    phis = _scalar(500.0 * GRAVITY)
    mslp = mslp_from_surface_pressure(sp, t, phis)
    assert mslp.item() > sp.item()
    assert torch.isfinite(mslp).all()


def test_cold_surface_branch():
    """temp < 255 K triggers T_eff = 0.5*(255+T)."""
    sp = _scalar(101325.0)
    t = _scalar(240.0)  # very cold, < 255
    phis = _scalar(500.0 * GRAVITY)
    mslp = mslp_from_surface_pressure(sp, t, phis)
    assert mslp.item() > sp.item()
    assert torch.isfinite(mslp).all()


def test_intermediate_branch_alpha_modified():
    """T_surf <= 290.5 and T_sl > 290.5: alpha = Rd*(290.5-T)/sgp."""
    sp = _scalar(101325.0)
    t = _scalar(288.0)
    # choose height so tto > 290.5: height > (290.5-288)/0.0065 ~ 384 m
    phis = _scalar(400.0 * GRAVITY)
    mslp = mslp_from_surface_pressure(sp, t, phis)
    assert mslp.item() > sp.item()
    assert torch.isfinite(mslp).all()


def test_batch_broadcast():
    """Accepts (B, T, H, W) inputs and returns same shape."""
    B, T, H, W = 4, 2, 8, 16
    sp = torch.rand(B, T, H, W) * 20000 + 85000
    t = torch.rand(B, T, H, W) * 40 + 260
    phis = torch.rand(B, T, H, W) * 2000 * GRAVITY
    mslp = mslp_from_surface_pressure(sp, t, phis)
    assert mslp.shape == (B, T, H, W)
    assert torch.isfinite(mslp).all()


def test_no_nan_on_zero_geopotential_grid():
    """All-zero geopotential (ocean grid) should give MSLP == SP everywhere."""
    B, H, W = 2, 32, 64
    sp = torch.rand(B, H, W) * 5000 + 98000
    t = torch.rand(B, H, W) * 40 + 260
    phis = torch.zeros(B, H, W)
    mslp = mslp_from_surface_pressure(sp, t, phis)
    assert torch.allclose(mslp, sp, rtol=1e-5)
    assert torch.isfinite(mslp).all()


def test_tto_uses_height_not_raw_geopotential():
    """
    Regression: the sea-level temperature branch test must use height (sgp/g),
    not raw geopotential.  At low elevation the two differ by a factor of ~9.8;
    using raw geopotential would trigger the alpha-modified branch at far too low
    an elevation.

    We pick T=288 K and height=100 m so tto = 288 + 0.0065*100 = 288.65 K < 290.5.
    With the bug (tto = T + lapse*sgp = 288 + 0.0065*100*9.8 ≈ 294.4 K > 290.5),
    the wrong alpha branch fires.
    """
    sp = _scalar(101325.0)
    t = _scalar(288.0)
    height = 100.0  # metres → tto = 288.65 K → default branch
    phis = _scalar(height * GRAVITY)

    # tto with correct formula
    tto_correct = 288.0 + _LAPSE_RATE * height  # 288.65 < 290.5 → default branch
    assert tto_correct <= 290.5, "test setup: should stay in default branch"

    mslp = mslp_from_surface_pressure(sp, t, phis)
    # in the default branch alpha = LAPSE_RATE*Rd/g
    alpha = _LAPSE_RATE * RDGAS / GRAVITY
    x = (height * GRAVITY) / (RDGAS * 288.0)
    import math

    expected = sp.item() * math.exp(x * (1.0 - 0.5 * alpha * x + (alpha * x) ** 2 / 3.0))
    assert abs(mslp.item() - expected) < 1.0, (
        f"MSLP {mslp.item():.2f} Pa vs expected {expected:.2f} Pa — possible tto units bug"
    )


# ---------------------------------------------------------------------------
# MSLPDiagnostic — integration / shape tests
# ---------------------------------------------------------------------------

SRC = "ARCO_ERA5"
SP_VAR = f"{SRC}/prognostic/2d/surface_pressure"
T2M_VAR = f"{SRC}/prognostic/2d/2m_temperature"
PHIS_VAR = f"{SRC}/static/2d/geopotential_at_surface"
OUTPUT_VAR = f"{SRC}/derived_diagnostic/2d/mean_sea_level_pressure"


def _make_batch(B=2, n_time=1, H=8, W=16):
    """Build a minimal batch dict as if Reconstruct has already run.

    Prognostic outputs live in ``y_processed``; the static PHIS field lives in
    ``ic_raw``, mirroring the rollout state dict in trainers/rollout_utils.py.
    """
    sp = torch.full((B, 1, n_time, H, W), 101_000.0)
    t2m = torch.full((B, 1, n_time, H, W), 285.0)
    phis = torch.full((B, 1, 1, H, W), 9_810.0)  # ~1000 m elevation
    return {
        "y_processed": {SRC: {SP_VAR: sp, T2M_VAR: t2m}},
        "ic_raw": {SRC: {PHIS_VAR: phis}},
    }


def test_mslp_diagnostic_output_shape():
    """Output tensor has shape (B, 1, n_time, H, W)."""
    B, n_time, H, W = 2, 1, 8, 16
    batch = _make_batch(B, n_time, H, W)
    mod = MSLPDiagnostic()
    out = mod(batch)
    assert OUTPUT_VAR in out["y_processed"][SRC]
    assert out["y_processed"][SRC][OUTPUT_VAR].shape == (B, 1, n_time, H, W)


def test_mslp_diagnostic_greater_than_sp():
    """MSLP > SP when surface is above sea level."""
    batch = _make_batch()
    mod = MSLPDiagnostic()
    out = mod(batch)
    result = out["y_processed"][SRC][OUTPUT_VAR]
    sp = batch["y_processed"][SRC][SP_VAR]
    assert (result > sp).all()


def test_mslp_diagnostic_finite():
    """No NaN or Inf for physically plausible random inputs."""
    B, n_time, H, W = 3, 2, 16, 32
    sp = torch.rand(B, 1, n_time, H, W) * 20_000 + 85_000
    t2m = torch.rand(B, 1, n_time, H, W) * 40 + 255
    phis = torch.rand(B, 1, 1, H, W) * 2_000 * GRAVITY
    batch = {"y_processed": {SRC: {SP_VAR: sp, T2M_VAR: t2m}}, "ic_raw": {SRC: {PHIS_VAR: phis}}}
    mod = MSLPDiagnostic()
    out = mod(batch)
    assert torch.isfinite(out["y_processed"][SRC][OUTPUT_VAR]).all()


def test_mslp_diagnostic_sea_level_identity():
    """At zero geopotential (ocean), MSLP equals SP exactly."""
    B, H, W = 2, 8, 16
    sp = torch.rand(B, 1, 1, H, W) * 5_000 + 98_000
    t2m = torch.rand(B, 1, 1, H, W) * 30 + 265
    phis = torch.zeros(B, 1, 1, H, W)
    batch = {"y_processed": {SRC: {SP_VAR: sp, T2M_VAR: t2m}}, "ic_raw": {SRC: {PHIS_VAR: phis}}}
    mod = MSLPDiagnostic()
    out = mod(batch)
    assert torch.allclose(out["y_processed"][SRC][OUTPUT_VAR], sp, rtol=1e-5)


def test_mslp_diagnostic_missing_key_raises():
    """A batch_dict missing the configured key raises ValueError."""
    batch = _make_batch()
    mod = MSLPDiagnostic(key="target_processed")
    with pytest.raises(ValueError, match="target_processed"):
        mod(batch)


def test_mslp_diagnostic_missing_static_key_raises():
    """A batch_dict missing the static source key raises ValueError."""
    batch = _make_batch()
    del batch["ic_raw"]
    mod = MSLPDiagnostic()
    with pytest.raises(ValueError, match="ic_raw"):
        mod(batch)


def test_mslp_diagnostic_custom_key():
    """The output dict can live under a non-default batch_dict key."""
    batch = _make_batch()
    batch["target_processed"] = batch.pop("y_processed")
    mod = MSLPDiagnostic(key="target_processed")
    out = mod(batch)
    assert OUTPUT_VAR in out["target_processed"][SRC]


def test_mslp_diagnostic_phis_broadcasts_over_time():
    """Static PHIS with n_time=1 broadcasts correctly over n_time > 1."""
    B, n_time, H, W = 2, 4, 8, 16
    sp = torch.full((B, 1, n_time, H, W), 90_000.0)
    t2m = torch.full((B, 1, n_time, H, W), 270.0)
    phis = torch.full((B, 1, 1, H, W), 5_000.0 * GRAVITY)  # static, n_time=1
    batch = {"y_processed": {SRC: {SP_VAR: sp, T2M_VAR: t2m}}, "ic_raw": {SRC: {PHIS_VAR: phis}}}
    mod = MSLPDiagnostic()
    out = mod(batch)
    result = out["y_processed"][SRC][OUTPUT_VAR]
    assert result.shape == (B, 1, n_time, H, W)
    assert torch.isfinite(result).all()
