"""Unit tests for trainer components: EMATracker, BaseTrainer.__init__, LinearWarmupCosineScheduler.

All tests run on CPU with no data files required.
"""

import torch
import torch.nn as nn
import pytest

try:
    from credit.trainers.base_trainer import EMATracker, BaseTrainer
    from credit.scheduler import LinearWarmupCosineScheduler

    _TRAINER_V2_AVAILABLE = True
except ImportError:
    _TRAINER_V2_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TRAINER_V2_AVAILABLE,
    reason="EMATracker / LinearWarmupCosineScheduler not available until v2/trainer-preblocks is merged",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model():
    return nn.Linear(4, 2)


def _minimal_conf(**trainer_overrides):
    """Return a minimal conf dict suitable for BaseTrainer.__init__."""
    trainer = {
        "mode": "none",
        "start_epoch": 0,
        "epochs": 10,
        "num_epoch": 5,
        "amp": False,
        "use_scheduler": False,
        "use_ema": False,
        "use_tensorboard": False,
        "skip_validation": False,
        "train_batch_size": 1,
    }
    trainer.update(trainer_overrides)
    return {
        "trainer": trainer,
        "save_loc": "/tmp/credit_test_trainer",
    }


class _ConcreteTrainer(BaseTrainer):
    """Minimal concrete subclass so we can instantiate BaseTrainer."""

    def train_one_epoch(self, *a, **kw):
        pass

    def validate(self, *a, **kw):
        return {}


# ---------------------------------------------------------------------------
# EMATracker
# ---------------------------------------------------------------------------


class TestEMATracker:
    def test_update_moves_shadow_toward_param(self):
        model = _tiny_model()
        ema = EMATracker(model, decay=0.9)

        # Record initial shadow
        initial = {k: v.clone() for k, v in ema.shadow.items()}

        # Change model params and update
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)
        ema.update(model)

        # Shadow should have moved toward 1.0 for every key
        for k in ema.shadow:
            assert not torch.equal(ema.shadow[k], initial[k]), f"Shadow did not update for key {k}"

    def test_swap_exchanges_weights(self):
        model = _tiny_model()
        # Set model weights to all-ones
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)
        ema = EMATracker(model, decay=0.9)

        # Force shadow to all-zeros
        for k in ema.shadow:
            ema.shadow[k].zero_()

        # After swap, model should hold zeros, shadow should hold ones
        ema.swap(model)
        for p in model.parameters():
            assert torch.allclose(p, torch.zeros_like(p)), "Model weights should be 0 after swap"

        # Second swap should restore ones
        ema.swap(model)
        for p in model.parameters():
            assert torch.allclose(p, torch.ones_like(p)), "Model weights should be 1 after double swap"

    def test_swap_is_idempotent_double_call(self):
        """Calling swap twice in a row returns to original state."""
        model = _tiny_model()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        ema = EMATracker(model, decay=0.9)

        ema.swap(model)
        ema.swap(model)

        for k, v in model.state_dict().items():
            assert torch.allclose(v, original[k]), f"Double swap not idempotent for key {k}"

    def test_state_dict_round_trip(self):
        model = _tiny_model()
        ema = EMATracker(model, decay=0.999)
        ema.step = 42

        state = ema.state_dict()
        ema2 = EMATracker(_tiny_model(), decay=0.5)
        ema2.load_state_dict(state)

        assert ema2.decay == 0.999
        assert ema2.step == 42
        for k in ema.shadow:
            assert torch.equal(ema.shadow[k], ema2.shadow[k])


# ---------------------------------------------------------------------------
# BaseTrainer.__init__
# ---------------------------------------------------------------------------


class TestBaseTrainerInit:
    def test_basic_fields_extracted(self, tmp_path):
        conf = _minimal_conf()
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)

        assert trainer.epochs == 10
        assert trainer.start_epoch == 0
        assert trainer.num_epoch == 5
        assert trainer.amp is False
        assert trainer.mode == "none"
        assert trainer.distributed is False
        assert trainer.rank == 0

    def test_device_is_cpu_when_no_cuda(self, tmp_path):
        conf = _minimal_conf()
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        # In a CPU-only test environment device should be cpu
        if not torch.cuda.is_available():
            assert trainer.device.type == "cpu"

    def test_ema_none_when_disabled(self, tmp_path):
        conf = _minimal_conf(use_ema=False)
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.ema is None

    def test_ema_created_when_enabled(self, tmp_path):
        conf = _minimal_conf(use_ema=True, ema_decay=0.999)
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert isinstance(trainer.ema, EMATracker)
        assert trainer.ema.decay == 0.999

    def test_tb_writer_none_when_disabled(self, tmp_path):
        conf = _minimal_conf(use_tensorboard=False)
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.tb_writer is None

    def test_training_metric_defaults(self, tmp_path):
        conf = _minimal_conf()
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        # skip_validation=False → metric is valid_loss
        assert trainer.training_metric == "valid_loss"

    def test_training_metric_train_loss_when_skip_validation(self, tmp_path):
        conf = _minimal_conf(skip_validation=True)
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.training_metric == "train_loss"


# ---------------------------------------------------------------------------
# LinearWarmupCosineScheduler
# ---------------------------------------------------------------------------


class TestLinearWarmupCosineScheduler:
    def _make_scheduler(self, base_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr=1e-5):
        model = _tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        scheduler = LinearWarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
        return optimizer, scheduler

    def test_lr_starts_near_zero(self):
        optimizer, scheduler = self._make_scheduler(base_lr=1e-3, warmup_steps=100)
        lr = optimizer.param_groups[0]["lr"]
        assert lr < 1e-4, f"Expected near-zero LR at step 0, got {lr}"

    def test_lr_ramps_during_warmup(self):
        optimizer, scheduler = self._make_scheduler(base_lr=1e-3, warmup_steps=100)
        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        # LR should be monotonically increasing during warmup
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1)), "LR should increase monotonically during warmup"

    def test_lr_peaks_at_base_lr(self):
        base_lr = 1e-3
        optimizer, scheduler = self._make_scheduler(base_lr=base_lr, warmup_steps=100)
        for _ in range(100):
            scheduler.step()
        peak = optimizer.param_groups[0]["lr"]
        assert abs(peak - base_lr) / base_lr < 0.02, f"Peak LR {peak} should be close to base_lr {base_lr}"

    def test_lr_decays_after_warmup(self):
        optimizer, scheduler = self._make_scheduler(base_lr=1e-3, warmup_steps=100, total_steps=500)
        # run through warmup
        for _ in range(100):
            scheduler.step()
        peak = optimizer.param_groups[0]["lr"]
        # run halfway through decay
        for _ in range(200):
            scheduler.step()
        mid = optimizer.param_groups[0]["lr"]
        assert mid < peak, f"LR should decrease after warmup: peak={peak}, mid={mid}"

    def test_lr_reaches_min_lr_at_total_steps(self):
        min_lr = 1e-5
        optimizer, scheduler = self._make_scheduler(base_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr=min_lr)
        for _ in range(1000):
            scheduler.step()
        final = optimizer.param_groups[0]["lr"]
        assert abs(final - min_lr) / min_lr < 0.05, f"Final LR {final} should be close to min_lr {min_lr}"

    def test_lr_never_below_min_lr(self):
        min_lr = 1e-5
        optimizer, scheduler = self._make_scheduler(base_lr=1e-3, warmup_steps=100, total_steps=500, min_lr=min_lr)
        for _ in range(600):  # past total_steps
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            assert lr >= min_lr - 1e-10, f"LR {lr} dropped below min_lr {min_lr}"


# ---------------------------------------------------------------------------
# Trainer subclass instantiation smoke tests
#
# These catch __init__ NameErrors / KeyErrors before any training runs.
# Each test only calls __init__ — no forward pass, no data files needed.
# ---------------------------------------------------------------------------


def _era5_v1_conf(**overrides):
    """Minimal conf for trainerERA5.Trainer (v1 data schema)."""
    base = _minimal_conf()
    base["data"] = {
        "forecast_len": 1,
        "history_len": 1,
    }
    base.update(overrides)
    return base


def _era5_v2_conf(**overrides):
    """Minimal conf for trainerERA5v2.Trainer (v2 nested data schema)."""
    base = _minimal_conf()
    base["data"] = {
        "forecast_len": 1,
        "scaler_type": "std_new",
        "source": {
            "ERA5": {
                "levels": [500, 850],
                "variables": {
                    "prognostic": {"vars_3D": ["temperature"], "vars_2D": ["SP"]},
                    "diagnostic": {"vars_3D": [], "vars_2D": []},
                    "dynamic_forcing": {"vars_2D": []},
                    "static": {"vars_2D": []},
                },
            }
        },
        "mean_path": "/dev/null",
        "std_path": "/dev/null",
    }
    base.update(overrides)
    return base


class TestTrainerSubclassInstantiation:
    """Smoke-test every Trainer subclass __init__ with a minimal conf.

    Regression guard: catches undefined-name bugs (post_self, data_self, etc.)
    introduced during refactors before any training is attempted.
    """

    def test_era5_trainer_init(self):
        from credit.trainers.trainerERA5 import Trainer

        t = Trainer(_tiny_model(), rank=0, conf=_era5_v1_conf())
        assert t.forecast_len == 1
        assert not t.flag_mass_conserve

    def test_era5_trainer_init_with_post_conf_inactive(self):
        from credit.trainers.trainerERA5 import Trainer

        conf = _era5_v1_conf()
        conf["model"] = {"post_conf": {"activate": False}}
        t = Trainer(_tiny_model(), rank=0, conf=conf)
        assert not t.flag_mass_conserve

    def test_era5v2_trainer_init(self):
        from unittest.mock import patch

        # ERA5Normalizer loads mean/std files at init — replace with identity nn.Module
        with patch("credit.trainers.trainerERA5v2.ERA5Normalizer", return_value=nn.Identity()):
            from credit.trainers.trainerERA5v2 import Trainer

            t = Trainer(_tiny_model(), rank=0, conf=_era5_v2_conf())
        assert t.forecast_len == 1

    def test_era5_ensemble_trainer_init(self):
        from credit.trainers.trainerERA5_ensemble import Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_era5_diffusion_trainer_init(self):
        from credit.trainers.trainerERA5_Diffusion import Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_les_trainer_init(self):
        from credit.trainers.trainerLES import Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_wrf_trainer_init(self):
        from credit.trainers.trainerWRF import Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_wrf_multi_trainer_init(self):
        from credit.trainers.trainerWRF_multi import Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_downscaling_trainer_init(self):
        from credit.trainers.trainer_downscaling import Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_om4_samudra_trainer_init(self):
        from credit.trainers.trainer_om4_samudra import Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())


# ---------------------------------------------------------------------------
# Multi-step training (forecast_len > 1) — ERA5 v2 trainer
# ---------------------------------------------------------------------------


def _era5_v2_multistep_conf(forecast_len, tmp_path):
    """Minimal conf for trainerERA5v2.Trainer with forecast_len > 1."""
    base = _minimal_conf()
    base["save_loc"] = str(tmp_path)
    base["trainer"]["batches_per_epoch"] = 1
    base["trainer"]["valid_batches_per_epoch"] = 1
    base["data"] = {
        "forecast_len": forecast_len,
        "retain_graph": False,
        "scaler_type": "std_new",
        "source": {
            "ERA5": {
                "levels": [],
                "variables": {
                    "prognostic": {"vars_3D": [], "vars_2D": ["a", "b", "c", "d"]},
                    "diagnostic": {"vars_3D": [], "vars_2D": []},
                    "dynamic_forcing": {"vars_2D": []},
                    "static": {"vars_2D": []},
                },
            }
        },
        "mean_path": "/dev/null",
        "std_path": "/dev/null",
    }
    return base


class _FakeDataset:
    """Stub dataset so the batches_per_epoch resolution logic doesn't crash."""

    pass


class _FakeLoader:
    """Minimal iterable loader that yields pre-assembled {x, y} batches."""

    def __init__(self, B, C, H, W, n_batches):
        self.dataset = _FakeDataset()
        self.sampler = None
        self._shape = (B, C, H, W)
        self._n = n_batches

    def __len__(self):
        return self._n

    def __iter__(self):
        import torch

        for _ in range(self._n):
            x = torch.randn(*self._shape)
            y = torch.randn(*self._shape)
            yield {"x": x, "y": y}


class TestERA5v2MultiStepTraining:
    """Verify forecast_len > 1 in the v2 trainer autoregressive loop."""

    def test_forecast_len_2_init(self, tmp_path):
        from unittest.mock import patch
        import torch.nn as nn
        from credit.trainers.trainerERA5v2 import Trainer

        conf = _era5_v2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        with patch("credit.trainers.trainerERA5v2.ERA5Normalizer", return_value=nn.Identity()):
            t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.forecast_len == 2
        assert t.backprop_on_timestep == [1, 2]
        assert t.static_dim_size == 0
        assert t.varnum_diag == 0

    def test_backprop_on_timestep_default_covers_all_steps(self, tmp_path):
        from unittest.mock import patch
        import torch.nn as nn
        from credit.trainers.trainerERA5v2 import Trainer

        for fl in [1, 2, 3]:
            conf = _era5_v2_multistep_conf(forecast_len=fl, tmp_path=tmp_path)
            with patch("credit.trainers.trainerERA5v2.ERA5Normalizer", return_value=nn.Identity()):
                t = Trainer(_tiny_model(), rank=0, conf=conf)
            assert t.backprop_on_timestep == list(range(1, fl + 1))

    def test_2step_train_one_epoch_runs(self, tmp_path):
        """2-step autoregressive loop completes on CPU with toy data."""
        import torch
        import torch.nn as nn
        from unittest.mock import patch
        from credit.trainers.trainerERA5v2 import Trainer

        B, C, H, W = 1, 4, 4, 4
        forecast_len = 2

        # 1x1 conv so shapes stay intact and model has real parameters
        model = nn.Conv2d(C, C, kernel_size=1, bias=False)

        conf = _era5_v2_multistep_conf(forecast_len=forecast_len, tmp_path=tmp_path)
        with patch("credit.trainers.trainerERA5v2.ERA5Normalizer", return_value=nn.Identity()):
            trainer = Trainer(model, rank=0, conf=conf)

        # _FakeLoader yields forecast_len * batches_per_epoch batches total
        loader = _FakeLoader(B, C, H, W, n_batches=forecast_len)

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        results = trainer.train_one_epoch(
            epoch=0,
            trainloader=loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=None,
            metrics=_metrics,
        )

        assert "train_loss" in results
        assert len(results["train_loss"]) == 1
        assert results["train_forecast_len"][-1] == forecast_len
        assert torch.isfinite(torch.tensor(results["train_loss"][0]))

    def test_2step_x_replaced_with_y_pred(self, tmp_path):
        """At t=2, prognostic channels of x must equal y_pred from t=1."""
        import torch
        import torch.nn as nn
        from unittest.mock import patch
        from credit.trainers.trainerERA5v2 import Trainer
        from credit.preblock import apply_preblocks

        B, C, H, W = 1, 4, 4, 4
        captured = {}

        class _ConstModel(nn.Module):
            """Returns a fixed tensor so we can track what y_pred was at t=1."""

            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(1))  # needs a param for optimizer

            def forward(self, x):
                out = x * self.w  # shape preserved, differentiable
                captured["last_x"] = x.detach().clone()
                captured["last_out"] = out.detach().clone()
                return out

        model = _ConstModel()
        conf = _era5_v2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        with patch("credit.trainers.trainerERA5v2.ERA5Normalizer", return_value=nn.Identity()):
            trainer = Trainer(model, rank=0, conf=conf)

        step = [0]
        x_at_step1_out = [None]

        original_apply = apply_preblocks

        def _patched_apply(preblocks, batch):
            batch = original_apply(preblocks, batch)
            step[0] += 1
            if step[0] == 1:
                x_at_step1_out[0] = batch["x"].clone()
            return batch

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        loader = _FakeLoader(B, C, H, W, n_batches=2)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        with patch("credit.trainers.trainerERA5v2.apply_preblocks", side_effect=_patched_apply):
            trainer.train_one_epoch(
                epoch=0,
                trainloader=loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                scheduler=None,
                metrics=_metrics,
            )

        # At t=2 the model sees x = y_pred from t=1 = x_at_step1_out (since model is identity-like)
        # captured["last_x"] is the x fed to the model at t=2
        assert captured["last_x"] is not None
        expected = x_at_step1_out[0]  # y_pred at t=1 = model(x_t1) = x_t1 * 1.0 ≈ x_t1
        torch.testing.assert_close(captured["last_x"], expected, atol=1e-5, rtol=1e-5)
