"""
validate_tisr_parity.py
-----------------------
Standalone validation script that checks the PyTorch TISR implementation
(tisr.py) has numerical parity with the original JAX/GraphCast
solar_radiation.py reference implementation.

This script lives under scripts/ and is not part of the main test suite.
It is intended to be run manually by anyone who wants to verify the refactor.

Setup
-----
1. Install additional dependencies:

       pip install jax jaxlib chex

2. Download the GraphCast reference file into the scripts/ directory:

       wget https://raw.githubusercontent.com/google-deepmind/graphcast/main/graphcast/solar_radiation.py

Usage
-----
Run from the scripts/ directory:

    pytest validate_tisr_parity.py
"""

import importlib.util
import sys
import os

# Ensure the repo root is on sys.path so that tisr.py can resolve
# 'from credit.datasets.base_dataset import BaseDataset' when loaded below.
# This makes the script runnable from any working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import pytest
import torch

# ── Path configuration ────────────────────────────────────────────────────────
GRAPHCAST_SOLAR_RADIATION_PATH = "graphcast_solar_radiation.py"
TISR_PATH                      = "../credit/datasets/tisr.py"

# ── Original JAX implementation (loaded from absolute path) ──────────────────
_spec = importlib.util.spec_from_file_location(
    "solar_radiation",
    GRAPHCAST_SOLAR_RADIATION_PATH,
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules["solar_radiation"] = _mod

jax_get_j2000_days                   = _mod._get_j2000_days
jax_get_orbital_parameters           = _mod._get_orbital_parameters
jax_get_solar_sin_altitude           = _mod._get_solar_sin_altitude
jax_get_radiation_flux               = _mod._get_radiation_flux
jax_get_toa_incident_solar_radiation = _mod.get_toa_incident_solar_radiation
jax_get_tsi                          = _mod.get_tsi
era5_tsi_data                        = _mod.era5_tsi_data

import jax
jax.config.update("jax_enable_x64", True)  # match PyTorch float64 throughout
import jax.numpy as jnp

# ── PyTorch implementation (loaded from absolute path) ────────────────────────
_spec2 = importlib.util.spec_from_file_location(
    "tisr",
    TISR_PATH,
)
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
sys.modules["tisr"] = _mod2

_era5_tsi_data               = _mod2._era5_tsi_data
_get_tsi                     = _mod2._get_tsi
torch_get_j2000_days         = _mod2._get_j2000_days
torch_get_orbital_parameters = _mod2._get_orbital_parameters
_get_solar_time              = _mod2._get_solar_time
_get_hour_angle              = _mod2._get_hour_angle
_get_cosine_zenith_angle     = _mod2._get_cosine_zenith_angle
_get_instantaneous_toa_tisr  = _mod2._get_instantaneous_toa_tisr
_get_integrated_toa_tisr     = _mod2._get_integrated_toa_tisr

# ── Tolerances ────────────────────────────────────────────────────────────────
# float64 gives ~15 decimal digits; remaining differences are cross-library
# trig/polynomial divergence between JAX and PyTorch float64.
RTOL = 1e-5
ATOL = 1e-4


def to_np(x) -> np.ndarray:
    """Convert JAX or PyTorch tensor to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# Representative timestamps — chosen to exercise diverse solar geometry:
#   * March equinox (near-zero declination)
#   * June solstice (max N declination)
#   * September equinox
#   * December solstice (max S declination)
#   * High noon on equator, midnight, and an arbitrary mid-day time
TIMESTAMPS = [
    "2020-03-20 12:00:00",  # March equinox
    "2020-06-21 00:00:00",  # June solstice midnight
    "2020-06-21 12:00:00",  # June solstice noon
    "2020-09-22 06:00:00",  # September equinox dawn
    "2020-12-21 18:00:00",  # December solstice dusk
    "2000-01-01 12:00:00",  # J2000 reference epoch
    "1989-11-08 21:00:00",  # Example from ERA5 docs
]


# =============================================================================
# 1. TSI interpolation
# =============================================================================

class TestTSI:
    """_get_tsi must return identical interpolated TSI values as the JAX reference."""

    def test_tsi_values_match(self):
        tsi_data_jax = era5_tsi_data()
        times_torch, values_torch = _era5_tsi_data()

        jax_result = jax_get_tsi(TIMESTAMPS, tsi_data_jax)
        torch_result = _get_tsi(TIMESTAMPS, times_torch, values_torch)

        np.testing.assert_allclose(
            to_np(torch_result), np.array(jax_result),
            rtol=RTOL, atol=ATOL,
            err_msg="TSI interpolation diverges between implementations",
        )


# =============================================================================
# 2. J2000 day conversion
# =============================================================================

class TestJ2000Days:
    """_get_j2000_days must agree with the JAX reference."""

    @pytest.mark.parametrize("ts", TIMESTAMPS)
    def test_j2000_days_match(self, ts):
        jax_val = float(jax_get_j2000_days(pd.Timestamp(ts)))
        torch_val = float(torch_get_j2000_days(pd.Timestamp(ts)).item())
        assert abs(jax_val - torch_val) < 1e-4, \
            f"J2000 days differ for {ts}: JAX={jax_val}, PyTorch={torch_val}"


# =============================================================================
# 3. Orbital parameters
# =============================================================================

class TestOrbitalParameters:
    """Every orbital parameter must match the JAX reference."""

    KEYS = [
        "sin_declination",
        "cos_declination",
        "eq_of_time_seconds",
        "solar_distance_au",
    ]

    @pytest.mark.parametrize("ts", TIMESTAMPS)
    def test_orbital_params_match(self, ts):
        j2000 = float(jax_get_j2000_days(pd.Timestamp(ts)))

        jax_op = jax_get_orbital_parameters(jnp.array(j2000))
        torch_op = torch_get_orbital_parameters(
            torch.tensor([j2000], dtype=torch.float64)
        )

        for key in self.KEYS:
            jax_val = float(np.asarray(getattr(jax_op, key)))
            torch_val = float(torch_op[key].item())
            assert abs(jax_val - torch_val) < ATOL, (
                f"{key} mismatch at {ts}: JAX={jax_val:.6f}, PyTorch={torch_val:.6f}"
            )


# =============================================================================
# 4. Solar time
# =============================================================================

class TestSolarTime:
    """Solar time derived from PyTorch orbital params must match JAX."""

    def test_solar_time_matches_jax(self):
        for ts_str in TIMESTAMPS:
            j2000 = float(jax_get_j2000_days(pd.Timestamp(ts_str)))
            jax_op = jax_get_orbital_parameters(jnp.array(j2000))
            jax_solar_time = float(jax_op.rotational_phase
                                   + jax_op.eq_of_time_seconds / 86400.0)

            torch_op = torch_get_orbital_parameters(
                torch.tensor([j2000], dtype=torch.float64)
            )
            torch_solar_time = _get_solar_time(
                torch_op["rotational_phase"], torch_op["eq_of_time_seconds"]
            ).item()

            assert abs(jax_solar_time - torch_solar_time) < 1e-9, (
                f"Solar time mismatch at {ts_str}: "
                f"JAX={jax_solar_time:.10f}, PyTorch={torch_solar_time:.10f}"
            )


# =============================================================================
# 5. Solar geometry / hour angle
# =============================================================================

class TestSolarGeometry:
    """cos_zenith (PyTorch) must match sin_altitude (JAX) — they are the same quantity."""

    def test_cosine_zenith_matches_jax_sin_altitude(self):
        """
        The JAX code computes sin(altitude) = cos(zenith_angle), so both
        quantities are the same thing. Compare directly for a set of
        (lat, lon, timestamp) combinations.
        """
        ts = "2020-06-21 12:00:00"
        lat_deg = np.array([0.0, 45.0, -45.0, 90.0, -90.0])
        lon_deg = np.array([0.0, 90.0, -90.0, 180.0, 45.0])

        j2000 = float(jax_get_j2000_days(pd.Timestamp(ts)))
        jax_op = jax_get_orbital_parameters(jnp.array(j2000))

        sin_lat_jax = jnp.sin(jnp.radians(lat_deg))
        cos_lat_jax = jnp.cos(jnp.radians(lat_deg))
        lon_rad_jax = jnp.radians(lon_deg)
        jax_vals = np.asarray(jax_get_solar_sin_altitude(
            jax_op, sin_lat_jax, cos_lat_jax, lon_rad_jax
        ))
        jax_vals_clamped = np.maximum(jax_vals, 0.0)

        torch_op = torch_get_orbital_parameters(
            torch.tensor([[j2000]], dtype=torch.float64)
        )
        solar_time = _get_solar_time(
            torch_op["rotational_phase"], torch_op["eq_of_time_seconds"]
        )
        ha = _get_hour_angle(solar_time, torch.tensor(lon_deg, dtype=torch.float64))
        cos_zenith = _get_cosine_zenith_angle(
            cos_declination=torch_op["cos_declination"],
            sin_declination=torch_op["sin_declination"],
            latitude=torch.tensor(lat_deg, dtype=torch.float64),
            hour_angle=ha,
        )

        np.testing.assert_allclose(
            to_np(cos_zenith.squeeze()), jax_vals_clamped,
            rtol=RTOL, atol=ATOL,
            err_msg="cos_zenith (PyTorch) vs sin_altitude (JAX) mismatch",
        )


# =============================================================================
# 6. Instantaneous TOA TISR
# =============================================================================

class TestInstantaneousFlux:
    """Instantaneous flux (W/m2) must match the JAX _get_radiation_flux."""

    @pytest.mark.parametrize("ts", [
        "2020-03-20 12:00:00",
        "2020-06-21 12:00:00",
        "2020-12-21 00:00:00",
    ])
    def test_instantaneous_flux_matches_jax(self, ts):
        lat_deg = np.array([[-90.0], [-45.0], [0.0], [45.0], [90.0]])
        lon_deg = np.array([0.0, 90.0, 180.0, 270.0])

        tsi_data_jax = era5_tsi_data()
        tsi_scalar = float(jax_get_tsi([ts], tsi_data_jax)[0])
        j2000 = float(jax_get_j2000_days(pd.Timestamp(ts)))

        jax_flux = np.asarray(jax_get_radiation_flux(
            j2000_days=jnp.array(j2000),
            sin_latitude=jnp.sin(jnp.radians(lat_deg)),
            cos_latitude=jnp.cos(jnp.radians(lat_deg)),
            longitude=jnp.radians(lon_deg),
            tsi=jnp.array(tsi_scalar),
        ))

        torch_op = torch_get_orbital_parameters(
            torch.tensor([[[j2000]]], dtype=torch.float64)
        )
        lat_t = torch.tensor(lat_deg, dtype=torch.float64).unsqueeze(0)
        lon_t = torch.tensor(lon_deg, dtype=torch.float64).unsqueeze(0)
        solar_time = _get_solar_time(torch_op["rotational_phase"], torch_op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, lon_t)
        cz = _get_cosine_zenith_angle(
            torch_op["cos_declination"], torch_op["sin_declination"], lat_t, ha
        )
        solar_factor = 1.0 / torch_op["solar_distance_au"] ** 2
        torch_flux = _get_instantaneous_toa_tisr(
            tsi=torch.tensor([[[tsi_scalar]]], dtype=torch.float64),
            solar_factor=solar_factor,
            cos_zenith=cz,
        ).squeeze(0)

        np.testing.assert_allclose(
            to_np(torch_flux), jax_flux,
            rtol=RTOL, atol=ATOL,
            err_msg=f"Instantaneous TOA flux mismatch at {ts}",
        )


# =============================================================================
# 7. Integrated TOA TISR — end-to-end
# =============================================================================

class TestIntegratedTISR:
    """Integrated TISR must match the JAX get_toa_incident_solar_radiation."""

    @pytest.mark.parametrize("ts_str", [
        "2020-03-20 12:00:00",
        "2020-06-21 12:00:00",
        "2020-09-22 06:00:00",
        "2020-12-21 18:00:00",
        "1989-11-08 21:00:00",   # ERA5 docs example
    ])
    def test_integrated_tisr_matches_jax(self, ts_str):
        """
        Drives both pipelines with a small 5x4 lat/lon grid and checks
        that the 1-hour integrated TISR values agree to within tolerance.
        """
        lat_deg = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
        lon_deg = np.array([0.0, 90.0, 180.0, 270.0])
        integration_period = pd.Timedelta(hours=1)
        num_steps = 360

        jax_result = np.asarray(jax_get_toa_incident_solar_radiation(
            timestamps=[ts_str],
            latitude=lat_deg,
            longitude=lon_deg,
            integration_period=integration_period,
            num_integration_bins=num_steps,
            use_jit=False,
        ))[0]  # shape (lat, lon)

        t = pd.Timestamp(ts_str)
        ts = pd.date_range(
            end=t,
            periods=num_steps + 1,
            freq=integration_period / num_steps,
        )
        times_torch, tsi_values_torch = _era5_tsi_data()
        tsi = _get_tsi(ts, times_torch, tsi_values_torch).unsqueeze(-1).unsqueeze(-1)
        j2000_days = torch_get_j2000_days(ts).unsqueeze(-1).unsqueeze(-1)
        op = torch_get_orbital_parameters(j2000_days)
        lat_t = torch.tensor(lat_deg, dtype=torch.float64).reshape(1, -1, 1)
        lon_t = torch.tensor(lon_deg, dtype=torch.float64).reshape(1, 1, -1)
        solar_time = _get_solar_time(op["rotational_phase"], op["eq_of_time_seconds"])
        ha = _get_hour_angle(solar_time, lon_t)
        cz = _get_cosine_zenith_angle(op["cos_declination"], op["sin_declination"], lat_t, ha)
        solar_factor = 1.0 / op["solar_distance_au"] ** 2
        inst = _get_instantaneous_toa_tisr(tsi, solar_factor, cz)
        torch_result = to_np(
            _get_integrated_toa_tisr(inst, integration_period, num_steps)
        )  # shape (lat, lon)

        np.testing.assert_allclose(
            torch_result, jax_result,
            rtol=RTOL, atol=ATOL,
            err_msg=f"Integrated TISR mismatch at {ts_str}",
        )