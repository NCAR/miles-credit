"""Unit tests for credit/trainers/preflight.py.

All tests run on CPU with no cluster or real data required.
"""

import time
import pytest

pytest.importorskip("credit.trainers.preflight", reason="preflight not available until v2/trainer-preblocks is merged")
from credit.trainers.preflight import estimate_dataloader_memory_gb, check_dataloader_startup  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conf(workers=4, prefetch=4, batch=1, h=721, w=1440, vars_3d=None, vars_2d=None, diag_2d=None, n_levels=13):
    vars_3d = vars_3d if vars_3d is not None else ["U", "V", "T"]
    vars_2d = vars_2d if vars_2d is not None else ["SP"]
    diag_2d = diag_2d if diag_2d is not None else []
    return {
        "trainer": {
            "thread_workers": workers,
            "prefetch_factor": prefetch,
            "train_batch_size": batch,
        },
        "model": {
            "image_height": h,
            "image_width": w,
        },
        "data": {
            "source": {
                "ERA5": {
                    "levels": list(range(n_levels)),
                    "variables": {
                        "prognostic": {"vars_3D": vars_3d, "vars_2D": vars_2d},
                        "diagnostic": {"vars_2D": diag_2d},
                    },
                }
            }
        },
    }


# ---------------------------------------------------------------------------
# estimate_dataloader_memory_gb — pure function tests
# ---------------------------------------------------------------------------


class TestEstimateDataloaderMemoryGb:
    def test_returns_float(self):
        assert isinstance(estimate_dataloader_memory_gb(_conf()), float)

    def test_zero_channels_returns_zero(self):
        est = estimate_dataloader_memory_gb(_conf(vars_3d=[], vars_2d=[], diag_2d=[], n_levels=13))
        assert est == 0.0

    def test_empty_conf_returns_zero(self):
        assert estimate_dataloader_memory_gb({}) == 0.0

    def test_scales_with_workers(self):
        base = estimate_dataloader_memory_gb(_conf(workers=1))
        doubled = estimate_dataloader_memory_gb(_conf(workers=2))
        assert abs(doubled / base - 2.0) < 1e-6

    def test_scales_with_prefetch(self):
        base = estimate_dataloader_memory_gb(_conf(prefetch=1))
        quad = estimate_dataloader_memory_gb(_conf(prefetch=4))
        assert abs(quad / base - 4.0) < 1e-6

    def test_scales_with_batch(self):
        base = estimate_dataloader_memory_gb(_conf(batch=1))
        octu = estimate_dataloader_memory_gb(_conf(batch=8))
        assert abs(octu / base - 8.0) < 1e-6

    def test_formula_1d_era5(self):
        """Spot-check with 1-degree ERA5: 3×13+1 = 40 channels."""
        conf = _conf(
            workers=4,
            prefetch=4,
            batch=1,
            h=721,
            w=1440,
            vars_3d=["U", "V", "T"],
            vars_2d=["SP"],
            diag_2d=[],
            n_levels=13,
        )
        total_ch = 3 * 13 + 1  # 40
        bytes_per_sample = 721 * 1440 * total_ch * 4 * 2
        expected_gb = (4 * 4 * 1 * bytes_per_sample) / 1e9
        assert abs(estimate_dataloader_memory_gb(conf) - expected_gb) < 0.01

    def test_025deg_era5_is_large(self):
        """0.25° ERA5 with typical settings should be >10 GB."""
        conf = _conf(
            workers=4,
            prefetch=4,
            batch=1,
            h=721,
            w=1440,
            vars_3d=["U", "V", "T", "Q", "Z"],
            vars_2d=["SP", "VAR_2T", "VAR_10U", "VAR_10V"],
            diag_2d=[],
            n_levels=37,
        )
        est = estimate_dataloader_memory_gb(conf)
        assert est > 10.0

    def test_missing_trainer_conf_uses_defaults(self):
        """No 'trainer' key → falls back to defaults (workers=4, prefetch=4, batch=1)."""
        conf = {
            "model": {"image_height": 721, "image_width": 1440},
            "data": {
                "source": {
                    "ERA5": {
                        "levels": list(range(13)),
                        "variables": {
                            "prognostic": {"vars_3D": ["T"], "vars_2D": []},
                            "diagnostic": {"vars_2D": []},
                        },
                    }
                }
            },
        }
        est = estimate_dataloader_memory_gb(conf)
        assert est > 0.0

    def test_raises_internally_returns_zero(self):
        """A config that triggers an internal exception returns 0.0 (lines 81-82)."""
        # levels is not iterable — len() will raise, hitting the except branch
        bad_conf = {
            "trainer": {"thread_workers": 4, "prefetch_factor": 4, "train_batch_size": 1},
            "model": {"image_height": 721, "image_width": 1440},
            "data": {
                "source": {
                    "ERA5": {
                        "levels": None,  # len(None) raises TypeError
                        "variables": {
                            "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},
                            "diagnostic": {"vars_2D": []},
                        },
                    }
                }
            },
        }
        result = estimate_dataloader_memory_gb(bad_conf)
        assert result == 0.0


# ---------------------------------------------------------------------------
# check_dataloader_startup — integration / side-effect tests
# ---------------------------------------------------------------------------


class _FastLoader:
    """Minimal loader that yields one dummy batch immediately."""

    def __iter__(self):
        import torch

        yield torch.zeros(1)


class _SlowLoader:
    """Loader that always blocks for longer than any reasonable timeout."""

    def __iter__(self):
        while True:
            time.sleep(9999)


class _ErrorLoader:
    """Loader that raises on first iteration."""

    def __iter__(self):
        raise ValueError("dataset exploded")


class TestCheckDataloaderStartup:
    def test_fast_loader_does_not_raise(self):
        """A loader that returns immediately should pass cleanly."""
        check_dataloader_startup(_conf(), _FastLoader(), rank=0, timeout_s=10.0)

    def test_non_rank0_is_no_op(self):
        """Non-rank-0 processes skip all checks — slow loader is fine."""
        # This must return immediately without touching the loader
        check_dataloader_startup(_conf(), _SlowLoader(), rank=1, timeout_s=0.001)

    def test_hang_raises_runtime_error(self):
        """A loader that hangs should raise RuntimeError within timeout + margin."""
        with pytest.raises(RuntimeError, match="DATA LOADING HANG DETECTED"):
            check_dataloader_startup(_conf(), _SlowLoader(), rank=0, timeout_s=0.1)

    def test_hang_message_contains_fixes(self):
        """Error message must include actionable fix suggestions."""
        with pytest.raises(RuntimeError) as exc_info:
            check_dataloader_startup(_conf(), _SlowLoader(), rank=0, timeout_s=0.1)
        msg = str(exc_info.value)
        assert "thread_workers" in msg
        assert "prefetch_factor" in msg

    def test_loader_error_is_reraised(self):
        """If the loader itself raises, that exception propagates."""
        with pytest.raises(ValueError, match="dataset exploded"):
            check_dataloader_startup(_conf(), _ErrorLoader(), rank=0, timeout_s=10.0)

    def test_timeout_s_respected(self):
        """Timeout should fire in roughly timeout_s seconds (within 3 s margin)."""
        t0 = time.monotonic()
        with pytest.raises(RuntimeError):
            check_dataloader_startup(_conf(), _SlowLoader(), rank=0, timeout_s=0.5)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"Timeout took too long: {elapsed:.1f}s"
