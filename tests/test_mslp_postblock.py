"""Tests for the MSLP postblock: mslp_from_surface_pressure and MSLPCalculator."""

import torch
import pytest
from unittest.mock import MagicMock, patch

from credit.postblock.mslp import mslp_from_surface_pressure, MSLPCalculator
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
    expected = sp.item() * (1.0 - 0.5 * alpha * x + (alpha * x) ** 2 / 3.0) ** 0  # exp series
    import math

    expected = sp.item() * math.exp(x * (1.0 - 0.5 * alpha * x + (alpha * x) ** 2 / 3.0))
    assert abs(mslp.item() - expected) < 1.0, (
        f"MSLP {mslp.item():.2f} Pa vs expected {expected:.2f} Pa — possible tto units bug"
    )


# ---------------------------------------------------------------------------
# MSLPCalculator — integration / shape tests (no real physics file needed)
# ---------------------------------------------------------------------------


def _make_post_conf(sp_ind=0, t2m_ind=1, append=True, denorm=False):
    return {
        "mslp_calculator": {
            "activate": True,
            "sp_ind": sp_ind,
            "t2m_ind": t2m_ind,
            "surface_geopotential_name": "PHIS",
            "denorm": denorm,
            "append_output": append,
        },
        "data": {"save_loc_physics": "dummy_path"},
    }


def _make_mock_phis(H=8, W=16):
    import numpy as np

    ds = MagicMock()
    ds.__getitem__ = MagicMock(return_value=MagicMock(values=np.zeros((H, W), dtype="float32")))
    return ds


@pytest.fixture
def mslp_module():
    """MSLPCalculator with a mocked physics dataset."""
    H, W = 8, 16
    post_conf = _make_post_conf()
    with patch("credit.postblock.mslp.get_forward_data", return_value=_make_mock_phis(H, W)):
        mod = MSLPCalculator(post_conf)
    return mod


def test_mslp_calculator_appends_channel(mslp_module):
    """append_output=True adds one channel to y_pred."""
    B, V, T, H, W = 2, 5, 1, 8, 16
    y_pred = torch.rand(B, V, T, H, W) * 20000 + 88000
    x = {"y_pred": y_pred, "x": torch.zeros(B, V, T, H, W)}
    out = mslp_module(x)
    assert out["y_pred"].shape == (B, V + 1, T, H, W)


def test_mslp_calculator_stores_mslp_key(mslp_module):
    """x['mslp'] is always written, shape (B, 1, T, H, W)."""
    B, V, T, H, W = 2, 5, 1, 8, 16
    y_pred = torch.rand(B, V, T, H, W) * 20000 + 88000
    x = {"y_pred": y_pred, "x": torch.zeros(B, V, T, H, W)}
    mslp_module(x)
    assert "mslp" in x
    assert x["mslp"].shape == (B, 1, T, H, W)


def test_mslp_calculator_no_append():
    """append_output=False: y_pred channels unchanged."""
    H, W = 8, 16
    post_conf = _make_post_conf(append=False)
    with patch("credit.postblock.mslp.get_forward_data", return_value=_make_mock_phis(H, W)):
        mod = MSLPCalculator(post_conf)
    B, V, T = 2, 5, 1
    y_pred = torch.rand(B, V, T, H, W) * 20000 + 88000
    x = {"y_pred": y_pred.clone(), "x": torch.zeros(B, V, T, H, W)}
    mod(x)
    assert x["y_pred"].shape == (B, V, T, H, W)


def test_mslp_calculator_no_nan():
    """No NaN or Inf in output for random plausible inputs."""
    H, W = 16, 32
    post_conf = _make_post_conf()
    with patch("credit.postblock.mslp.get_forward_data", return_value=_make_mock_phis(H, W)):
        mod = MSLPCalculator(post_conf)
    B, V, T = 3, 10, 2
    y_pred = torch.rand(B, V, T, H, W)
    # put realistic SP and T in channels 0 and 1
    y_pred[:, 0, ...] = torch.rand(B, T, H, W) * 20000 + 85000  # SP: 85–105 kPa
    y_pred[:, 1, ...] = torch.rand(B, T, H, W) * 40 + 255  # T: 255–295 K
    x = {"y_pred": y_pred, "x": torch.zeros(B, V, T, H, W)}
    mod(x)
    assert torch.isfinite(x["mslp"]).all()
    assert torch.isfinite(x["y_pred"]).all()


def test_mslp_calculator_phis_buffer_moves_with_module():
    """phis buffer follows the module to a new device (CPU→CPU here)."""
    H, W = 8, 16
    post_conf = _make_post_conf()
    with patch("credit.postblock.mslp.get_forward_data", return_value=_make_mock_phis(H, W)):
        mod = MSLPCalculator(post_conf)
    mod = mod.cpu()
    assert mod.phis.device.type == "cpu"
