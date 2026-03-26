"""Unit tests for credit plot helpers: _build_channel_map and _build_denorm_stats.

All tests run on CPU with no real data files required.
_build_denorm_stats is tested with in-memory xarray datasets injected via monkeypatching.
"""

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

def _make_conf(vars_3d=None, vars_2d=None, diag_2d=None, n_levels=3):
    levels = list(range(n_levels))
    return {
        "data": {
            "source": {
                "ERA5": {
                    "level_coord": "level",
                    "levels": levels,
                    "variables": {
                        "prognostic": {
                            "vars_3D": vars_3d or [],
                            "vars_2D": vars_2d or [],
                        },
                        "diagnostic": {
                            "vars_2D": diag_2d or [],
                        },
                    },
                }
            },
            "mean_path": "/fake/mean.nc",
            "std_path":  "/fake/std.nc",
        }
    }


# ---------------------------------------------------------------------------
# _build_channel_map
# ---------------------------------------------------------------------------

class TestBuildChannelMap:
    def test_only_2d_variables(self):
        from credit.cli import _build_channel_map
        conf = _make_conf(vars_2d=["SP", "VAR_2T"], n_levels=3)
        cm = _build_channel_map(conf)
        assert cm == {"SP": [0], "VAR_2T": [1]}

    def test_only_3d_variables(self):
        from credit.cli import _build_channel_map
        conf = _make_conf(vars_3d=["U", "V"], n_levels=4)
        cm = _build_channel_map(conf)
        # U occupies indices 0-3, V occupies 4-7
        assert cm["U"] == [0, 1, 2, 3]
        assert cm["V"] == [4, 5, 6, 7]

    def test_3d_then_2d_ordering(self):
        from credit.cli import _build_channel_map
        conf = _make_conf(vars_3d=["T"], vars_2d=["SP"], n_levels=2)
        cm = _build_channel_map(conf)
        # T: channels 0-1, SP: channel 2
        assert cm["T"] == [0, 1]
        assert cm["SP"] == [2]

    def test_diagnostic_appended_after_prognostic(self):
        from credit.cli import _build_channel_map
        conf = _make_conf(vars_3d=["T"], vars_2d=["SP"], diag_2d=["precip"], n_levels=2)
        cm = _build_channel_map(conf)
        assert cm["T"]      == [0, 1]
        assert cm["SP"]     == [2]
        assert cm["precip"] == [3]

    def test_total_channel_count(self):
        from credit.cli import _build_channel_map
        conf = _make_conf(vars_3d=["U", "V"], vars_2d=["SP", "VAR_2T"],
                          diag_2d=["precip", "evap"], n_levels=5)
        cm = _build_channel_map(conf)
        all_channels = sorted(c for chans in cm.values() for c in chans)
        # 2 × 5 = 10 for 3D, 2 × 1 = 2 for prog 2D, 2 × 1 = 2 diag → 14 total
        assert all_channels == list(range(14))

    def test_empty_config(self):
        from credit.cli import _build_channel_map
        conf = _make_conf()
        cm = _build_channel_map(conf)
        assert cm == {}


# ---------------------------------------------------------------------------
# _build_denorm_stats
# ---------------------------------------------------------------------------

def _make_mock_datasets(vars_3d, vars_2d, diag_2d, n_levels, mean_val=10.0, std_val=2.0):
    """Build in-memory mean and std xarray Datasets for testing."""
    levels = list(range(n_levels))
    data_vars = {}
    for vn in vars_3d:
        data_vars[vn] = xr.DataArray(
            np.full(n_levels, mean_val, dtype=np.float32), dims=["level"],
            coords={"level": levels},
        )
    for vn in list(vars_2d) + list(diag_2d):
        data_vars[vn] = xr.DataArray(np.float32(mean_val))
    return xr.Dataset(data_vars)


class TestBuildDenormStats:
    def _patch_xr(self, monkeypatch, conf, mean_val=10.0, std_val=2.0):
        """Monkeypatch xr.open_dataset to return in-memory datasets."""
        src = conf["data"]["source"]["ERA5"]
        v = src["variables"]
        vars_3d = (v.get("prognostic") or {}).get("vars_3D", [])
        vars_2d = (v.get("prognostic") or {}).get("vars_2D", [])
        diag_2d = (v.get("diagnostic") or {}).get("vars_2D", [])
        n_levels = len(src["levels"])

        mean_ds = _make_mock_datasets(vars_3d, vars_2d, diag_2d, n_levels, mean_val=mean_val)
        std_ds  = _make_mock_datasets(vars_3d, vars_2d, diag_2d, n_levels, mean_val=std_val)

        call_count = {"n": 0}
        def _fake_open(path, **kwargs):
            ds = mean_ds if call_count["n"] == 0 else std_ds
            call_count["n"] += 1
            return ds

        import xarray as _xr
        monkeypatch.setattr(_xr, "open_dataset", _fake_open)

    def test_output_length_matches_total_channels(self, monkeypatch):
        from credit.cli import _build_denorm_stats
        conf = _make_conf(vars_3d=["U", "V"], vars_2d=["SP"], diag_2d=["precip"], n_levels=4)
        self._patch_xr(monkeypatch, conf)
        mean_arr, std_arr = _build_denorm_stats(conf)
        # 2×4 + 1 + 1 = 10
        assert len(mean_arr) == 10
        assert len(std_arr)  == 10

    def test_mean_values_propagated(self, monkeypatch):
        from credit.cli import _build_denorm_stats
        conf = _make_conf(vars_2d=["SP", "VAR_2T"], n_levels=2)
        self._patch_xr(monkeypatch, conf, mean_val=273.0, std_val=5.0)
        mean_arr, std_arr = _build_denorm_stats(conf)
        assert np.allclose(mean_arr, 273.0)
        assert np.allclose(std_arr,  5.0)

    def test_denorm_recovers_original_scale(self, monkeypatch):
        """x_norm * std + mean should recover original value."""
        from credit.cli import _build_denorm_stats
        conf = _make_conf(vars_2d=["SP"], n_levels=2)
        mean_val, std_val = 101325.0, 2000.0
        self._patch_xr(monkeypatch, conf, mean_val=mean_val, std_val=std_val)
        mean_arr, std_arr = _build_denorm_stats(conf)

        original = np.float32(103325.0)          # some physical SP value
        normalised = (original - mean_val) / std_val
        recovered = normalised * std_arr[0] + mean_arr[0]
        assert abs(recovered - original) < 1.0   # within 1 Pa

    def test_missing_variable_gives_passthrough(self, monkeypatch):
        """Variable not in stat files → mean=0, std=1 (no-op transform)."""
        from credit.cli import _build_denorm_stats
        conf = _make_conf(vars_2d=["SP", "MISSING_VAR"], n_levels=2)

        # Only SP is in the datasets
        src = conf["data"]["source"]["ERA5"]
        mean_ds = xr.Dataset({"SP": xr.DataArray(np.float32(101325.0))})
        std_ds  = xr.Dataset({"SP": xr.DataArray(np.float32(2000.0))})

        call_count = {"n": 0}
        def _fake_open(path, **kwargs):
            ds = mean_ds if call_count["n"] == 0 else std_ds
            call_count["n"] += 1
            return ds

        import xarray as _xr
        monkeypatch.setattr(_xr, "open_dataset", _fake_open)

        mean_arr, std_arr = _build_denorm_stats(conf)
        # SP: channel 0 — real stats
        assert abs(mean_arr[0] - 101325.0) < 1.0
        # MISSING_VAR: channel 1 — passthrough (mean=0, std=1)
        assert mean_arr[1] == pytest.approx(0.0)
        assert std_arr[1]  == pytest.approx(1.0)
