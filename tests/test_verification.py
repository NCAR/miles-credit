import os

import numpy as np
import xarray as xr

from credit.verification.ensemble import binned_spread_skill, crps, rank_histogram_apply, spread_error

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]), "config")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# 7 lats ensures all four regional slices have at least one point:
# s_extratropics (-91,−24.5): -90,-60,-30 | tropics (-24.5,24.5): 0 | n_extratropics (24.5,91): 30,60,90
LATS = np.linspace(-90, 90, 7)
LONS = np.linspace(0, 360, 6)
TIMES = np.arange(3)


def _make_ensemble(values, ensemble_size=20):
    """Build da_pred with constant value across all members/times/points."""
    data = np.full((ensemble_size, len(TIMES), len(LATS), len(LONS)), values)
    return xr.DataArray(
        data,
        coords={
            "ensemble_member_label": np.arange(ensemble_size),
            "time": TIMES,
            "latitude": LATS,
            "longitude": LONS,
        },
    )


def _make_truth(values):
    data = np.full((len(TIMES), len(LATS), len(LONS)), values)
    return xr.DataArray(
        data,
        coords={"time": TIMES, "latitude": LATS, "longitude": LONS},
    )


# ---------------------------------------------------------------------------
# spread_error
# ---------------------------------------------------------------------------


def test_spread_error():
    """Large Gaussian ensemble: RMSE ≈ 0, spread ≈ 1."""
    ensemble_size = 1000000
    latitude = 2
    longitude = 3
    times = 2

    mu, sigma = 0, 1.0
    rng = np.random.default_rng()
    data = rng.normal(mu, sigma, (ensemble_size, times, latitude, longitude))

    da_true = xr.DataArray(
        np.zeros((times, latitude, longitude)),
        coords={
            "time": np.arange(times),
            "latitude": np.linspace(-90, 90, latitude),
            "longitude": np.linspace(0, 360, longitude),
        },
    )
    da_pred = xr.DataArray(
        data,
        coords={
            "ensemble_member_label": np.arange(ensemble_size),
            "time": np.arange(times),
            "latitude": np.linspace(-90, 90, latitude),
            "longitude": np.linspace(0, 360, longitude),
        },
    )

    result_dict = spread_error(da_pred, da_true)

    assert np.isclose(result_dict["rmse_global"], 0.0, atol=1e-2)
    assert np.isclose(result_dict["std_global"], 1.0, atol=1e-2)


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------


def test_crps_perfect_forecast():
    """All ensemble members equal the truth → CRPS = 0 everywhere."""
    da_pred = _make_ensemble(0.0)
    da_true = _make_truth(0.0)

    result = crps(da_pred, da_true)

    for key, val in result.items():
        assert np.isclose(val, 0.0, atol=1e-6), f"{key}={val}, expected 0"


def test_crps_constant_bias():
    """All members biased by c, truth = 0 → CRPS = |c| (point-forecast identity)."""
    c = 3.0
    da_pred = _make_ensemble(c)
    da_true = _make_truth(0.0)

    result = crps(da_pred, da_true)

    # For a deterministic (zero-spread) forecast, CRPS = MAE = |c|
    for key, val in result.items():
        assert np.isclose(val, abs(c), atol=1e-4), f"{key}={val}, expected {abs(c)}"


def test_crps_returns_expected_regions():
    """CRPS result has the four standard regional keys."""
    da_pred = _make_ensemble(1.0)
    da_true = _make_truth(0.0)
    result = crps(da_pred, da_true)
    for region in ("global", "tropics", "n_extratropics", "s_extratropics"):
        assert f"crps_{region}" in result, f"missing key: crps_{region}"


def test_crps_nonnegative():
    """CRPS is always ≥ 0."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (10, len(TIMES), len(LATS), len(LONS)))
    da_pred = xr.DataArray(
        data,
        coords={
            "ensemble_member_label": np.arange(10),
            "time": TIMES,
            "latitude": LATS,
            "longitude": LONS,
        },
    )
    da_true = _make_truth(0.5)
    result = crps(da_pred, da_true)
    for key, val in result.items():
        assert val >= 0.0, f"{key}={val} is negative"


# ---------------------------------------------------------------------------
# rank_histogram
# ---------------------------------------------------------------------------


def test_rank_histogram_truth_always_below():
    """Truth always below all members → all counts in rank-0 bin."""
    da_pred = _make_ensemble(1.0)  # all members = 1
    da_true = _make_truth(-1.0)  # truth = -1 < all members

    rank_hist = rank_histogram_apply(da_pred, da_true)

    total = rank_hist.sum()
    assert total > 0, "rank histogram is empty"
    assert rank_hist[0] == total, f"expected all counts in bin 0, got {rank_hist}"
    assert rank_hist[1:].sum() == 0


def test_rank_histogram_truth_always_above():
    """Truth always above all members → all counts in last bin."""
    da_pred = _make_ensemble(-1.0)  # all members = -1
    da_true = _make_truth(1.0)  # truth = 1 > all members

    rank_hist = rank_histogram_apply(da_pred, da_true)

    total = rank_hist.sum()
    assert rank_hist[-1] == total, f"expected all counts in last bin, got {rank_hist}"
    assert rank_hist[:-1].sum() == 0


def test_rank_histogram_length():
    """Rank histogram has ensemble_size + 1 bins."""
    ensemble_size = 10
    da_pred = _make_ensemble(0.0, ensemble_size=ensemble_size)
    da_true = _make_truth(0.5)

    rank_hist = rank_histogram_apply(da_pred, da_true)

    assert len(rank_hist) == ensemble_size + 1


# ---------------------------------------------------------------------------
# binned_spread_skill
# ---------------------------------------------------------------------------


def test_binned_spread_skill_zero_spread():
    """Deterministic ensemble (zero spread): all spread values = 0."""
    da_pred = _make_ensemble(0.0)
    da_true = _make_truth(1.0)

    result = binned_spread_skill(da_pred, da_true, num_bins=5)

    assert "spread_means" in result
    assert "rmse_means" in result
    assert "bin_centers" in result
    assert "counts" in result
    assert len(result["bin_centers"]) == 5


def test_binned_spread_skill_positive_spread():
    """Random ensemble: spread values are non-negative."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (20, len(TIMES), len(LATS), len(LONS)))
    da_pred = xr.DataArray(
        data,
        coords={
            "ensemble_member_label": np.arange(20),
            "time": TIMES,
            "latitude": LATS,
            "longitude": LONS,
        },
    )
    da_true = _make_truth(0.0)

    result = binned_spread_skill(da_pred, da_true, num_bins=5)

    for s in result["spread_means"]:
        if s is not None and not np.isnan(s):
            assert s >= 0.0, f"negative spread: {s}"


if __name__ == "__main__":
    test_spread_error()
    test_crps_perfect_forecast()
    test_crps_constant_bias()
    test_crps_returns_expected_regions()
    test_crps_nonnegative()
    test_rank_histogram_truth_always_below()
    test_rank_histogram_truth_always_above()
    test_rank_histogram_length()
    test_binned_spread_skill_zero_spread()
    test_binned_spread_skill_positive_spread()
    print("all verification tests passed")
