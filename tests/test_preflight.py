"""Unit tests for credit/trainers/preflight.py.

All tests run on CPU with no cluster or real data required.
"""

import time
import pytest

from credit.trainers.preflight import estimate_dataloader_memory_gib, check_dataloader_startup  # noqa: E402


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
# estimate_dataloader_memory_gib — pure function tests
# ---------------------------------------------------------------------------


class TestEstimateDataloaderMemoryGb:
    def test_returns_float(self):
        assert isinstance(estimate_dataloader_memory_gib(_conf()), float)

    def test_zero_channels_returns_zero(self):
        est = estimate_dataloader_memory_gib(_conf(vars_3d=[], vars_2d=[], diag_2d=[], n_levels=13))
        assert est == 0.0

    def test_empty_conf_returns_zero(self):
        assert estimate_dataloader_memory_gib({}) == 0.0

    def test_scales_with_workers(self):
        # workers x prefetch buffered batches, plus the one the main process holds.
        base = estimate_dataloader_memory_gib(_conf(workers=1))
        doubled = estimate_dataloader_memory_gib(_conf(workers=2))
        assert abs(doubled / base - (2 * 4 + 1) / (1 * 4 + 1)) < 1e-6

    def test_scales_with_prefetch(self):
        base = estimate_dataloader_memory_gib(_conf(prefetch=1))
        quad = estimate_dataloader_memory_gib(_conf(prefetch=4))
        assert abs(quad / base - (4 * 4 + 1) / (4 * 1 + 1)) < 1e-6

    def test_scales_with_batch(self):
        base = estimate_dataloader_memory_gib(_conf(batch=1))
        octu = estimate_dataloader_memory_gib(_conf(batch=8))
        assert abs(octu / base - 8.0) < 1e-6

    def test_formula_1d_era5(self):
        """Spot-check with 1-degree ERA5: 3×13+1 = 40 channels."""
        vars_3d = ["U", "V", "T"]
        vars_2d = ["SP"]
        n_levels = 13
        h = 721
        w = 1440
        conf = _conf(
            workers=4,
            prefetch=4,
            batch=1,
            h=h,
            w=w,
            vars_3d=vars_3d,
            vars_2d=vars_2d,
            diag_2d=[],
            n_levels=n_levels,
        )
        print(conf)
        total_ch = len(vars_3d) * n_levels + len(vars_2d)  # 40
        bytes_per_sample = h * w * total_ch * 4 * 2  # prognostic in the input and the target
        expected_gb = ((4 * 4 + 1) * 1 * bytes_per_sample) / 2**30
        predicted_gb = estimate_dataloader_memory_gib(conf)
        print(expected_gb)
        print(predicted_gb)
        assert abs(predicted_gb - expected_gb) < 0.01

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
        est = estimate_dataloader_memory_gib(conf)
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
        est = estimate_dataloader_memory_gib(conf)
        assert est > 0.0

    def test_raises_internally_returns_zero(self):
        """A config that triggers an internal exception returns 0.0 (lines 81-82)."""
        # A truthy-but-non-iterable levels makes resolve_num_levels' len() raise,
        # hitting the except branch. (levels: None is now handled gracefully — it
        # falls back to model.levels — so it no longer exercises this path.)
        bad_conf = {
            "trainer": {"thread_workers": 4, "prefetch_factor": 4, "train_batch_size": 1},
            "model": {"image_height": 721, "image_width": 1440},
            "data": {
                "source": {
                    "ERA5": {
                        "levels": 16,  # int is truthy but len(16) raises TypeError
                        "variables": {
                            "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},
                            "diagnostic": {"vars_2D": []},
                        },
                    }
                }
            },
        }
        result = estimate_dataloader_memory_gib(bad_conf)
        assert result == 0.0

    def test_null_levels_falls_back_to_model_levels(self):
        """levels: null (or omitted) is estimated using model.levels, not 0 (issue #432)."""
        conf = {
            "trainer": {"thread_workers": 4, "prefetch_factor": 4, "train_batch_size": 1},
            "model": {"image_height": 32, "image_width": 32, "levels": 8},
            "data": {
                "source": {
                    "ERA5": {
                        "levels": None,  # "all levels" -> resolved from model.levels
                        "variables": {
                            "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},
                            "diagnostic": {"vars_2D": []},
                        },
                    }
                }
            },
        }
        # 1 3D var * 8 levels + 1 2D var = 9 channels -> nonzero estimate
        assert estimate_dataloader_memory_gib(conf) > 0.0

    @staticmethod
    def _one_source(variables, history_len=None, **trainer):
        """One source on a grid where each float32 channel is exactly 1 GiB, no workers."""
        src = {"levels": [1, 2], "variables": variables}
        if history_len is not None:
            src["history_len"] = history_len
        return {
            "trainer": {"thread_workers": 0, **trainer},
            "model": {"image_height": 1, "image_width": 2**28},
            "data": {"source": {"ERA5": src}},
        }

    def test_channels_by_field_type_and_source(self):
        """Input: prognostic + static + dynamic_forcing; target: prognostic + diagnostic; all sources."""
        conf = self._one_source(
            {
                "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},  # 3 channels
                "diagnostic": {"vars_2D": ["TP"]},
                "static": {"vars_2D": ["lsm", "z"]},
                "dynamic_forcing": {"vars_2D": ["tisr"]},
            }
        )
        conf["data"]["source"]["SOLAR"] = {"variables": {"dynamic_forcing": {"vars_2D": ["tsi"]}}}
        # input 3 + 2 + 1 + 1, target 3 + 1
        assert estimate_dataloader_memory_gib(conf) == pytest.approx(11.0)

    def test_history_len_multiplies_the_input(self):
        variables = {"prognostic": {"vars_2D": ["SP"]}, "static": {"vars_2D": ["lsm"]}}
        # source-level history_len: 3 x (1 + 1) input + 1 target
        assert estimate_dataloader_memory_gib(self._one_source(variables, history_len=3)) == pytest.approx(7.0)
        # data-level history_len is the fallback
        conf = self._one_source(variables)
        conf["data"]["history_len"] = 2
        assert estimate_dataloader_memory_gib(conf) == pytest.approx(5.0)

    def test_validation_uses_valid_settings_and_validation_data(self):
        conf = self._one_source({"prognostic": {"vars_2D": ["SP"]}}, valid_batch_size=4, valid_thread_workers=0)
        # validation_data without its own source falls back to data's
        conf["validation_data"] = {"start_datetime": "2020-01-01"}
        assert estimate_dataloader_memory_gib(conf, validation=True) == pytest.approx(8.0)
        conf["validation_data"]["source"] = {"V": {"variables": {"prognostic": {"vars_2D": ["SP", "T2"]}}}}
        assert estimate_dataloader_memory_gib(conf, validation=True) == pytest.approx(16.0)


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


# ---------------------------------------------------------------------------
# synthetic_step: the GPU memory preflight's pass must leave no trace
# ---------------------------------------------------------------------------


def _tiny_model():
    import torch

    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.utils.spectral_norm(torch.nn.Linear(4, 8)),  # weight_u / weight_v buffers
        torch.nn.BatchNorm1d(8),  # running statistics
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )


def _snapshot(model, optimizer):
    from credit.trainers.preflight import _cpu_copy

    return (
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        _cpu_copy(optimizer.state_dict()),
        [g["lr"] for g in optimizer.param_groups],
        model.training,
    )


def _assert_same(a, b):
    import torch

    def same(x, y):
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(same(i, j) for i, j in zip(x, y))
        return x == y

    assert same(a, b)


@pytest.mark.parametrize("training", [True, False])
def test_synthetic_step_restores_weights_buffers_optimizer_and_mode(training):
    """A resumed job's Adam moments, step count and spectral-norm vectors must survive the preflight."""
    import torch

    from credit.trainers.preflight import synthetic_step

    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(3):  # real steps: Adam state exists, as after a resume
        model(torch.randn(16, 4)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    model.train(training)
    before = _snapshot(model, optimizer)
    synthetic_step(model, optimizer, torch.zeros(16, 4))
    _assert_same(_snapshot(model, optimizer), before)
    assert all(p.grad is None for p in model.parameters())


def test_synthetic_step_leaves_a_fresh_optimizer_without_state():
    import torch

    from credit.trainers.preflight import synthetic_step

    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    synthetic_step(model, optimizer, torch.zeros(16, 4))
    assert optimizer.state_dict()["state"] == {}
