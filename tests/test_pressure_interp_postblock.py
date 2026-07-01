"""Tests for the pressure interpolation postblock.

Covers the per-column torch function (including parity with the numba
reference implementations in ``credit/interp.py``) and the
``PressureInterpDiagnostic`` postblock protocol.
"""

from functools import partial

import numpy as np
import pytest
import torch

from credit.interp import (
    interp_geopotential_to_pressure_levels,
    interp_hybrid_to_pressure_levels,
    interp_temperature_to_pressure_levels,
)
from credit.physics_constants import GRAVITY, RDGAS
from credit.postblock.geopotential import GeopotentialDiagnostic
from credit.postblock.pressure_interp import (
    PressureInterpDiagnostic,
    interp_column_to_pressure_levels,
)

# ---------------------------------------------------------------------------
# interp_column_to_pressure_levels — unit tests
# ---------------------------------------------------------------------------

N_LEVELS = 20
SIGMA_MID = (np.arange(N_LEVELS) + 0.5) / N_LEVELS  # top → surface
MODEL_A = 3000.0 * (1.0 - SIGMA_MID)  # Pa
MODEL_B = SIGMA_MID**2


def _column_tensors(surface_pressure=101_325.0, surface_height_m=0.0):
    """Build one physically plausible float64 column (top → surface)."""
    sp = torch.tensor(surface_pressure, dtype=torch.float64)
    sgp = torch.tensor(surface_height_m * GRAVITY, dtype=torch.float64)
    a = torch.tensor(MODEL_A, dtype=torch.float64)
    b = torch.tensor(MODEL_B, dtype=torch.float64)
    pressure = a + b * sp
    temperature = torch.tensor(216.0 + 72.0 * SIGMA_MID, dtype=torch.float64)
    geopotential = sgp + RDGAS * 260.0 * torch.log(sp / pressure)
    return sp, sgp, a, b, pressure, temperature, geopotential


def test_column_exact_for_linear_in_log_pressure():
    """A field linear in log(p) is interpolated exactly at in-range levels."""
    sp, sgp, a, b, pressure, temperature, geopotential = _column_tensors()
    field = 2.0 * torch.log(pressure) + 1.0
    plevs = torch.tensor([10_000.0, 50_000.0, 85_000.0], dtype=torch.float64)
    out = interp_column_to_pressure_levels(field.unsqueeze(0), temperature, geopotential, sp, sgp, a, b, plevs)
    expected = 2.0 * torch.log(plevs) + 1.0
    assert torch.allclose(out[0], expected, rtol=1e-10)


def test_column_constant_extrapolation_below_ground():
    """Fields (not T/Z) take the lowest-model-level value below the surface."""
    sp, sgp, a, b, pressure, temperature, geopotential = _column_tensors(surface_pressure=70_000.0)
    field = torch.randn(N_LEVELS, dtype=torch.float64)
    plevs = torch.tensor([85_000.0, 100_000.0], dtype=torch.float64)  # both below ground
    out = interp_column_to_pressure_levels(field.unsqueeze(0), temperature, geopotential, sp, sgp, a, b, plevs)
    assert torch.allclose(out[0], field[-1].expand(2))


def test_column_constant_extrapolation_above_model_top():
    """Targets above the model top take the top-level value."""
    sp, sgp, a, b, pressure, temperature, geopotential = _column_tensors()
    field = torch.randn(N_LEVELS, dtype=torch.float64)
    plevs = torch.tensor([1.0], dtype=torch.float64)  # far above model top
    out = interp_column_to_pressure_levels(field.unsqueeze(0), temperature, geopotential, sp, sgp, a, b, plevs)
    assert torch.allclose(out[0], field[0].expand(1))


def test_column_temperature_warmer_below_ground():
    """Below-ground temperature extrapolation increases T along the lapse rate."""
    sp, sgp, a, b, pressure, temperature, geopotential = _column_tensors(
        surface_pressure=70_000.0, surface_height_m=3000.0
    )
    plevs = torch.tensor([50_000.0, 85_000.0, 100_000.0], dtype=torch.float64)
    out = interp_column_to_pressure_levels(
        torch.zeros((0, N_LEVELS), dtype=torch.float64), temperature, geopotential, sp, sgp, a, b, plevs
    )
    t_out, z_out = out[0], out[1]
    # deeper below ground → warmer T, lower Z
    assert t_out[2] > t_out[1] > temperature[-1]
    assert z_out[2] < z_out[1] < sgp
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Parity with the numba reference implementations in credit/interp.py
# ---------------------------------------------------------------------------


def test_parity_with_interp_py_reference():
    """The vmapped torch column function matches the numba routines in credit/interp.py.

    The 2x2 grid covers all effective-lapse-rate branches: sea level (below
    2000 m), 1000 m, the 2000-2500 m blend zone, and a >2500 m plateau.
    """
    rng = np.random.default_rng(42)
    heights_m = np.array([[0.0, 1000.0], [2200.0, 3000.0]])
    ny, nx = heights_m.shape
    sgp = heights_m * GRAVITY
    sp = 101_325.0 * np.exp(-heights_m / 8000.0)

    pressure = MODEL_A[:, None, None] + MODEL_B[:, None, None] * sp  # (levels, y, x)
    temperature = 216.0 + 72.0 * SIGMA_MID[:, None, None] + rng.normal(0.0, 1.0, (N_LEVELS, ny, nx))
    geopotential = sgp + RDGAS * 260.0 * np.log(sp / pressure)
    u_wind = rng.normal(0.0, 15.0, (N_LEVELS, ny, nx))
    q = 1e-5 + 0.01 * SIGMA_MID[:, None, None] ** 3 * np.ones((N_LEVELS, ny, nx))
    plevs = np.array([30_000.0, 50_000.0, 70_000.0, 85_000.0, 92_500.0, 100_000.0])

    ref_u = interp_hybrid_to_pressure_levels(u_wind, pressure, plevs)
    ref_q = interp_hybrid_to_pressure_levels(q, pressure, plevs)
    ref_t = interp_temperature_to_pressure_levels(temperature, pressure, plevs, sp, sgp, geopotential)
    ref_z = interp_geopotential_to_pressure_levels(geopotential, pressure, plevs, sp, sgp, temperature)

    def _flat(arr_3d):  # (levels, y, x) → (y*x, levels)
        return torch.tensor(arr_3d.reshape(N_LEVELS, -1).T, dtype=torch.float64)

    fields = torch.stack([_flat(u_wind), _flat(q)], dim=1)  # (N, 2, levels)
    vinterp = torch.vmap(
        partial(interp_column_to_pressure_levels, temp_height=150.0),
        in_dims=(0, 0, 0, 0, 0, None, None, None),
        chunk_size=2,
    )
    out = vinterp(
        fields,
        _flat(temperature),
        _flat(geopotential),
        torch.tensor(sp.ravel(), dtype=torch.float64),
        torch.tensor(sgp.ravel(), dtype=torch.float64),
        torch.tensor(MODEL_A, dtype=torch.float64),
        torch.tensor(MODEL_B, dtype=torch.float64),
        torch.tensor(plevs, dtype=torch.float64),
    )  # (N, 4, n_plev)

    def _unflat(column_major):  # (y*x, n_plev) → (n_plev, y, x)
        return column_major.T.reshape(plevs.size, ny, nx).numpy()

    np.testing.assert_allclose(_unflat(out[:, 0]), ref_u, rtol=1e-8)
    np.testing.assert_allclose(_unflat(out[:, 1]), ref_q, rtol=1e-8)
    np.testing.assert_allclose(_unflat(out[:, 2]), ref_t, rtol=1e-8)
    np.testing.assert_allclose(_unflat(out[:, 3]), ref_z, rtol=1e-8, atol=1e-6)


# ---------------------------------------------------------------------------
# PressureInterpDiagnostic — postblock protocol tests
# ---------------------------------------------------------------------------

SRC = "ARCO_ERA5"
T_VAR = f"{SRC}/prognostic/3d/temperature"
Q_VAR = f"{SRC}/prognostic/3d/specific_humidity"
U_VAR = f"{SRC}/prognostic/3d/u_component_of_wind"
V_VAR = f"{SRC}/prognostic/3d/v_component_of_wind"
SP_VAR = f"{SRC}/prognostic/2d/surface_pressure"
PHIS_VAR = f"{SRC}/static/2d/geopotential_at_surface"
GEO_VAR = f"{SRC}/derived_diagnostic/3d/geopotential"

N_ERA5_LEVELS = 137


def _make_batch(B=2, n_time=1, H=4, W=8, surface_pressure=70_000.0, surface_height_m=3000.0):
    """Batch dict with ERA5-like profiles on all 137 levels, PHIS in ic_raw."""
    temp = torch.linspace(200.0, 290.0, N_ERA5_LEVELS).view(1, -1, 1, 1, 1).expand(B, -1, n_time, H, W).clone()
    q = torch.linspace(1e-6, 0.012, N_ERA5_LEVELS).view(1, -1, 1, 1, 1).expand(B, -1, n_time, H, W).clone()
    u = torch.randn(B, N_ERA5_LEVELS, n_time, H, W) * 15.0
    v = torch.randn(B, N_ERA5_LEVELS, n_time, H, W) * 10.0
    sp = torch.full((B, 1, n_time, H, W), surface_pressure)
    phis = torch.full((B, 1, 1, H, W), surface_height_m * GRAVITY)
    return {
        "y_processed": {SRC: {T_VAR: temp, Q_VAR: q, U_VAR: u, V_VAR: v, SP_VAR: sp}},
        "ic_raw": {SRC: {PHIS_VAR: phis}},
    }


@pytest.fixture(scope="module")
def interpolated_batch():
    """Run the geopotential_diagnostic → pressure_interp_diagnostic chain once."""
    torch.manual_seed(0)
    batch = _make_batch()
    batch = GeopotentialDiagnostic()(batch)
    mod = PressureInterpDiagnostic(pressure_levels=[500.0, 850.0, 1000.0])
    return mod(batch)


def test_postblock_output_shapes_and_finite(interpolated_batch):
    """One (B, n_plev, n_time, H, W) output per variable, all finite."""
    out = interpolated_batch["y_processed"][SRC]
    for var in ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind", "geopotential"]:
        key = f"{SRC}/derived_diagnostic/3d/{var}_PRES"
        assert key in out
        assert out[key].shape == (2, 3, 1, 4, 8)
        assert torch.isfinite(out[key]).all()


def test_postblock_constant_extrapolation_below_ground(interpolated_batch):
    """Below ground (850/1000 hPa with 700 hPa surface), u equals its lowest-level value."""
    out = interpolated_batch["y_processed"][SRC]
    u_pres = out[f"{SRC}/derived_diagnostic/3d/u_component_of_wind_PRES"]
    u_bottom = interpolated_batch["y_processed"][SRC][U_VAR][:, -1]  # (B, n_time, H, W)
    assert torch.allclose(u_pres[:, 1], u_bottom)
    assert torch.allclose(u_pres[:, 2], u_bottom)


def test_postblock_temperature_extrapolation_below_ground(interpolated_batch):
    """Below-ground temperature follows the lapse-rate extrapolation, not constant."""
    out = interpolated_batch["y_processed"][SRC]
    t_pres = out[f"{SRC}/derived_diagnostic/3d/temperature_PRES"]
    t_bottom = interpolated_batch["y_processed"][SRC][T_VAR][:, -1]
    assert (t_pres[:, 2] > t_pres[:, 1]).all()  # warmer with depth
    assert (t_pres[:, 1] > t_bottom).all()
    z_pres = out[f"{SRC}/derived_diagnostic/3d/geopotential_PRES"]
    assert (z_pres[:, 2] < z_pres[:, 1]).all()  # geopotential decreases with depth
    assert (z_pres[:, 1] < 3000.0 * GRAVITY).all()


def test_postblock_missing_key_raises():
    """A batch_dict missing the configured key raises ValueError."""
    batch = _make_batch()
    del batch["ic_raw"]
    mod = PressureInterpDiagnostic()
    with pytest.raises(ValueError, match="ic_raw"):
        mod(batch)
