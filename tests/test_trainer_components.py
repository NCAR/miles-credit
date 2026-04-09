"""Unit tests for trainer components: EMATracker, BaseTrainer.__init__, LinearWarmupCosineScheduler,
and the load_trainer() factory.

All tests run on CPU with no data files required.
"""

import pytest
import torch
import torch.nn as nn

from credit.trainers.base_trainer import EMATracker, BaseTrainer
from credit.scheduler import LinearWarmupCosineScheduler


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


def _era5_gen2_conf(**overrides):
    """Minimal conf for trainerERA5gen2.Trainer (v2 nested data schema)."""
    base = _minimal_conf()
    base["data"] = {
        "forecast_len": 1,
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
    }
    base.update(overrides)
    return base


class TestTrainerSubclassInstantiation:
    """Smoke-test every Trainer subclass __init__ with a minimal conf.

    Regression guard: catches undefined-name bugs (post_self, data_self, etc.)
    introduced during refactors before any training is attempted.
    """

    def test_era5_trainer_init(self):
        from credit.trainers.trainerERA5gen1 import TrainerERA5Gen1 as Trainer

        t = Trainer(_tiny_model(), rank=0, conf=_era5_v1_conf())
        assert t.forecast_len == 1
        assert not t.flag_mass_conserve

    def test_era5_trainer_init_with_post_conf_inactive(self):
        from credit.trainers.trainerERA5gen1 import TrainerERA5Gen1 as Trainer

        conf = _era5_v1_conf()
        conf["model"] = {"post_conf": {"activate": False}}
        t = Trainer(_tiny_model(), rank=0, conf=conf)
        assert not t.flag_mass_conserve

    def test_era5gen2_trainer_init(self):
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        t = Trainer(_tiny_model(), rank=0, conf=_era5_gen2_conf())
        assert t.forecast_len == 1

    def test_era5_ensemble_trainer_init(self):
        from credit.trainers.trainerERA5_ensemble import TrainerERA5Ensemble as Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_era5_diffusion_trainer_init(self):
        from credit.trainers.trainerERA5_Diffusion import TrainerERA5Diffusion as Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_les_trainer_init(self):
        from credit.trainers.trainerLES import TrainerLES as Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_wrf_trainer_init(self):
        from credit.trainers.trainerWRF import TrainerWRF as Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_wrf_multi_trainer_init(self):
        from credit.trainers.trainerWRF_multi import TrainerWRFMulti as Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_downscaling_trainer_init(self):
        from credit.trainers.trainer_downscaling import TrainerDownscaling as Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())

    def test_om4_samudra_trainer_init(self):
        from credit.trainers.trainer_om4_samudra import TrainerSamudra as Trainer

        Trainer(_tiny_model(), rank=0, conf=_minimal_conf())


# ---------------------------------------------------------------------------
# Multi-step training (forecast_len > 1) — ERA5 Gen2 trainer
# ---------------------------------------------------------------------------


def _era5_gen2_multistep_conf(forecast_len, tmp_path):
    """Minimal conf for trainerERA5gen2.Trainer with forecast_len > 1."""
    base = _minimal_conf()
    base["save_loc"] = str(tmp_path)
    base["trainer"]["batches_per_epoch"] = 1
    base["trainer"]["valid_batches_per_epoch"] = 1
    base["data"] = {
        "forecast_len": forecast_len,
        "retain_graph": False,
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
    }
    return base


class _FakeDataset:
    """Stub dataset so the batches_per_epoch resolution logic doesn't crash."""

    pass


class _FakeLoader:
    """Minimal iterable loader that yields nested-format batches for trainerERA5gen2.

    Each batch has the structure expected by apply_preblocks / ConcatToTensor:
        batch["era5"]["input"][var_key]  -> (B, 1, H, W) tensor
        batch["era5"]["target"][var_key] -> (B, 1, H, W) tensor

    C variables are emitted so ConcatToTensor assembles (B, C, H, W) tensors,
    matching the original flat (B, C, H, W) shape expected by the test models.
    """

    def __init__(self, B, C, H, W, n_batches):
        self.dataset = _FakeDataset()
        self.sampler = None
        self._B = B
        self._C = C
        self._H = H
        self._W = W
        self._n = n_batches

    def __len__(self):
        return self._n

    def __iter__(self):
        import torch

        B, C, H, W = self._B, self._C, self._H, self._W
        for _ in range(self._n):
            input_vars = {f"era5/prognostic/2d/v{i}": torch.randn(B, 1, H, W) for i in range(C)}
            target_vars = {f"era5/prognostic/2d/v{i}": torch.randn(B, 1, H, W) for i in range(C)}
            yield {"era5": {"input": input_vars, "target": target_vars}}


class TestERA5Gen2MultiStepTraining:
    """Verify forecast_len > 1 in the v2 trainer autoregressive loop."""

    def test_forecast_len_2_init(self, tmp_path):
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.forecast_len == 2
        assert t.backprop_on_timestep == [1, 2]
        assert t.static_dim_size == 0
        assert t.varnum_diag == 0

    def test_backprop_on_timestep_default_covers_all_steps(self, tmp_path):
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        for fl in [1, 2, 3]:
            conf = _era5_gen2_multistep_conf(forecast_len=fl, tmp_path=tmp_path)
            t = Trainer(_tiny_model(), rank=0, conf=conf)
            assert t.backprop_on_timestep == list(range(1, fl + 1))

    def test_2step_train_one_epoch_runs(self, tmp_path):
        """2-step autoregressive loop completes on CPU with toy data."""
        import torch
        import torch.nn as nn
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        B, C, H, W = 1, 4, 4, 4
        forecast_len = 2

        # 1x1 conv so shapes stay intact and model has real parameters
        model = nn.Conv2d(C, C, kernel_size=1, bias=False)

        conf = _era5_gen2_multistep_conf(forecast_len=forecast_len, tmp_path=tmp_path)
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
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer
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
        conf = _era5_gen2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        trainer = Trainer(model, rank=0, conf=conf)

        step = [0]
        x_at_step1_out = [None]

        original_apply = apply_preblocks

        def _patched_apply(preblocks, batch):
            result = original_apply(preblocks, batch)
            x_raw, y_raw, _ = result
            step[0] += 1
            if step[0] == 1:
                x_at_step1_out[0] = x_raw.clone()
            return result

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        loader = _FakeLoader(B, C, H, W, n_batches=2)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        with patch("credit.trainers.trainerERA5gen2.apply_preblocks", side_effect=_patched_apply):
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

    def test_rollout_partial_channels_at_t2(self, tmp_path):
        """At t=2, ERA5Dataset returns only dynfrc channels.

        Verify the trainer correctly:
          - updates dynfrc slice (channels 0..n_dynfrc-1) from the new batch
          - preserves static slice (channels n_dynfrc..static_dim_size-1)
          - replaces prognostic slice (channels static_dim_size..) with y_pred
        """
        import torch
        import torch.nn as nn
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        # Layout: 2 dynfrc + 1 static + 3 prog  →  static_dim_size = 3, n_prog = 3
        N_DYNFRC, N_STATIC, N_PROG = 2, 1, 3
        STATIC_DIM = N_DYNFRC + N_STATIC  # 3
        TOTAL = N_DYNFRC + N_STATIC + N_PROG  # 6
        B, H, W = 1, 4, 4

        conf = _minimal_conf()
        conf["save_loc"] = str(tmp_path)
        conf["trainer"]["batches_per_epoch"] = 1
        conf["trainer"]["valid_batches_per_epoch"] = 1
        conf["data"] = {
            "forecast_len": 2,
            "retain_graph": False,
            "source": {
                "ERA5": {
                    "levels": [],
                    "variables": {
                        "prognostic": {"vars_3D": [], "vars_2D": [f"p{i}" for i in range(N_PROG)]},
                        "diagnostic": {"vars_3D": [], "vars_2D": []},
                        "dynamic_forcing": {"vars_2D": [f"df{i}" for i in range(N_DYNFRC)]},
                        "static": {"vars_2D": [f"st{i}" for i in range(N_STATIC)]},
                    },
                }
            },
        }

        # Model: outputs N_PROG channels, all zeros — makes y_pred easy to check.
        # Multiply by self.w so the output has a grad_fn for backprop.
        class _ZeroProgModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return torch.zeros(x.shape[0], N_PROG, x.shape[2], x.shape[3]) * self.w

        trainer = Trainer(_ZeroProgModel(), rank=0, conf=conf)
        assert trainer.static_dim_size == STATIC_DIM

        # Known fixed tensors for each channel group
        dynfrc_t1 = torch.full((B, N_DYNFRC, H, W), 1.0)
        static_ch = torch.full((B, N_STATIC, H, W), 2.0)
        prog_t1 = torch.full((B, N_PROG, H, W), 3.0)
        dynfrc_t2 = torch.full((B, N_DYNFRC, H, W), 9.0)  # new forcing at t=2

        step = [0]

        class _PartialLoader:
            """Step 1: full batch. Step 2: dynfrc only (simulates ERA5Dataset i>0)."""

            dataset = _FakeDataset()
            sampler = None

            def __len__(self):
                return 2

            def __iter__(self):
                # t=1 batch: all channel types present
                full_input = {}
                for i in range(N_DYNFRC):
                    full_input[f"era5/dynamic_forcing/2d/df{i}"] = dynfrc_t1[:, i : i + 1]
                for i in range(N_STATIC):
                    full_input[f"era5/static/2d/st{i}"] = static_ch[:, i : i + 1]
                for i in range(N_PROG):
                    full_input[f"era5/prognostic/2d/p{i}"] = prog_t1[:, i : i + 1]
                target = {f"era5/prognostic/2d/p{i}": prog_t1[:, i : i + 1] for i in range(N_PROG)}
                yield {"era5": {"input": full_input, "target": target}}

                # t=2 batch: only dynfrc (ERA5Dataset i>0 behavior)
                partial_input = {f"era5/dynamic_forcing/2d/df{i}": dynfrc_t2[:, i : i + 1] for i in range(N_DYNFRC)}
                yield {"era5": {"input": partial_input, "target": target}}

        captured_x = {}

        class _CapturingModel(_ZeroProgModel):
            def forward(self, x):
                captured_x[step[0]] = x.detach().clone()
                step[0] += 1
                return super().forward(x)

        trainer.model = _CapturingModel()
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.train_one_epoch(
            epoch=0,
            trainloader=_PartialLoader(),
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=None,
            metrics=lambda p, y: {"acc": 0.0, "mae": 0.0},
        )

        x_t2 = captured_x[1]  # x fed to model at t=2
        # dynfrc channels updated from t=2 batch
        torch.testing.assert_close(x_t2[:, :N_DYNFRC], dynfrc_t2, atol=1e-5, rtol=1e-5)
        # static channels preserved from t=1
        torch.testing.assert_close(x_t2[:, N_DYNFRC:STATIC_DIM], static_ch, atol=1e-5, rtol=1e-5)
        # prognostic channels replaced by y_pred (zeros from _ZeroProgModel)
        torch.testing.assert_close(x_t2[:, STATIC_DIM:], torch.zeros(B, N_PROG, H, W), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# load_trainer — factory function
# ---------------------------------------------------------------------------


class TestLoadTrainer:
    """Tests for credit.trainers.load_trainer()."""

    def test_valid_era5_type_returns_class(self):
        from credit.trainers import load_trainer
        from credit.trainers.trainerERA5gen1 import TrainerERA5Gen1 as Trainer

        result = load_trainer({"trainer": {"type": "era5"}})
        assert result is Trainer

    def test_valid_era5v2_type_returns_class(self):
        from credit.trainers import load_trainer
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        result = load_trainer({"trainer": {"type": "era5-gen2"}})
        assert result is Trainer

    def test_all_registered_types_return_a_class(self):
        """Every key in trainer_types must resolve to a callable without error."""
        from credit.trainers import load_trainer, trainer_types

        for name in trainer_types:
            result = load_trainer({"trainer": {"type": name}})
            assert callable(result), f"load_trainer('{name}') did not return a callable"

    def test_missing_type_key_raises_value_error(self):
        from credit.trainers import load_trainer

        with pytest.raises(ValueError, match="type"):
            load_trainer({"trainer": {}})

    def test_invalid_type_raises_value_error(self):
        from credit.trainers import load_trainer

        with pytest.raises(ValueError, match="not supported"):
            load_trainer({"trainer": {"type": "nonexistent-trainer-xyz"}})

    def test_original_conf_not_mutated(self):
        """load_trainer deep-copies conf internally; the caller's dict must be unchanged."""
        from credit.trainers import load_trainer

        conf = {"trainer": {"type": "era5", "extra_key": 99}}
        load_trainer(conf)
        assert conf["trainer"]["type"] == "era5"
        assert conf["trainer"]["extra_key"] == 99


# ---------------------------------------------------------------------------
# EMATracker — additional edge-case coverage
# ---------------------------------------------------------------------------


class TestEMATrackerEdgeCases:
    def test_ema_every_step_default_updates_every_call(self):
        """EMATracker.update() always updates shadow on every call (step always increments)."""
        model = _tiny_model()
        ema = EMATracker(model, decay=0.9)

        initial = {k: v.clone() for k, v in ema.shadow.items()}
        # Fill model with 1s so the update definitely moves the shadow
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)
        ema.update(model)

        for k in ema.shadow:
            assert not torch.equal(ema.shadow[k], initial[k])

    def test_swap_with_module_wrapper(self):
        """EMATracker.swap() handles models wrapped in a .module attribute (lines 93-96)."""

        class _WrappedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.Linear(4, 2)

            def state_dict(self):
                return self.module.state_dict()

            def load_state_dict(self, sd, **kw):
                self.module.load_state_dict(sd, **kw)

        # Fill the wrapped model with 1s
        wrapped = _WrappedModel()
        with torch.no_grad():
            for p in wrapped.module.parameters():
                p.fill_(1.0)

        ema = EMATracker(wrapped, decay=0.9)
        # Force shadow to zeros
        for k in ema.shadow:
            ema.shadow[k].zero_()

        # swap should move zeros into model, ones into shadow
        ema.swap(wrapped)
        for p in wrapped.module.parameters():
            assert torch.allclose(p, torch.zeros_like(p))

    def test_load_state_dict_missing_step_defaults_to_zero(self):
        """load_state_dict handles missing 'step' key (line 104: d.get('step', 0))."""
        model = _tiny_model()
        ema = EMATracker(model, decay=0.9)
        # Simulate old checkpoint without 'step'
        old_state = {"shadow": ema.shadow, "decay": 0.5}
        ema.load_state_dict(old_state)
        assert ema.step == 0
        assert ema.decay == 0.5

    def test_update_increments_step(self):
        """update() should increment self.step each call."""
        model = _tiny_model()
        ema = EMATracker(model, decay=0.9)
        assert ema.step == 0
        ema.update(model)
        assert ema.step == 1
        ema.update(model)
        assert ema.step == 2


# ---------------------------------------------------------------------------
# BaseTrainer — additional init / property coverage
# ---------------------------------------------------------------------------


class TestBaseTrainerAdditionalInit:
    def test_direction_max(self, tmp_path):
        """training_metric_direction='max' sets self.direction to max (line 160)."""
        conf = _minimal_conf(training_metric_direction="max")
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.direction is max

    def test_distributed_true_for_ddp_mode(self, tmp_path):
        """mode='ddp' should set distributed=True (line 137)."""
        conf = _minimal_conf(mode="ddp")
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.distributed is True

    def test_distributed_true_for_fsdp_mode(self, tmp_path):
        """mode='fsdp' should set distributed=True."""
        conf = _minimal_conf(mode="fsdp")
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.distributed is True

    def test_ema_loaded_from_checkpoint(self, tmp_path):
        """When checkpoint_ema.pt exists, EMA state should be restored (lines 170-173)."""
        import os

        # First build an EMA and save its state_dict to disk
        model = _tiny_model()
        ema = EMATracker(model, decay=0.999)
        ema.step = 77
        ema_state = {"model_state_dict": ema.state_dict()}
        ema_path = os.path.join(str(tmp_path), "checkpoint_ema.pt")
        torch.save(ema_state, ema_path)

        # Now build the trainer; it should load that EMA state
        conf = _minimal_conf(use_ema=True, ema_decay=0.999)
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.ema is not None
        assert trainer.ema.step == 77

    def test_tb_writer_none_for_nonzero_rank(self, tmp_path):
        """use_tensorboard=True but rank != 0 → tb_writer stays None (lines 181-194)."""
        conf = _minimal_conf(use_tensorboard=True)
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=1, conf=conf)
        assert trainer.tb_writer is None

    def test_model_state_dict_with_module_wrapper(self):
        """_model_state_dict unwraps DDP .module (lines 203, 208-211)."""

        class _Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.Linear(4, 2)

            def state_dict(self):
                return self.module.state_dict()

        wrapped = _Wrapped()
        sd = BaseTrainer._model_state_dict(wrapped)
        # Should return the underlying module's state dict
        assert any("weight" in k for k in sd)

    def test_load_model_state_dict_with_module_wrapper(self):
        """_load_model_state_dict loads into DDP .module (lines 208-211)."""

        class _Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = nn.Linear(4, 2)

            def state_dict(self):
                return self.module.state_dict()

            def load_state_dict(self, sd, **kw):
                # DDP wrapper doesn't have load_state_dict, the .module does
                pass

        wrapped = _Wrapped()
        sd = wrapped.module.state_dict()
        # Should not raise
        BaseTrainer._load_model_state_dict(wrapped, sd)

    def test_training_metric_explicit_override(self, tmp_path):
        """Explicit training_metric key in conf overrides default (lines 157-158)."""
        conf = _minimal_conf(training_metric="my_custom_metric")
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.training_metric == "my_custom_metric"

    def test_stop_after_epoch_flag(self, tmp_path):
        """stop_after_epoch flag is read from conf (line 152)."""
        conf = _minimal_conf(stop_after_epoch=True)
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.stop_after_epoch is True

    def test_save_metric_vars_list(self, tmp_path):
        """save_metric_vars can be a list of variable names (line 154)."""
        conf = _minimal_conf(save_metric_vars=["temperature", "wind"])
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.save_metric_vars == ["temperature", "wind"]

    def test_grad_max_norm_nonzero(self, tmp_path):
        """grad_max_norm extracted correctly from conf (line 144)."""
        conf = _minimal_conf(grad_max_norm=1.0)
        conf["save_loc"] = str(tmp_path)
        trainer = _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)
        assert trainer.grad_max_norm == 1.0


# ---------------------------------------------------------------------------
# BaseTrainer._save_checkpoint — non-fsdp branch
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def _build_trainer(self, tmp_path, **trainer_overrides):
        conf = _minimal_conf(**trainer_overrides)
        conf["save_loc"] = str(tmp_path)
        return _ConcreteTrainer(_tiny_model(), rank=0, conf=conf)

    def test_saves_checkpoint_file(self, tmp_path):
        """_save_checkpoint writes checkpoint.pt for rank-0, non-fsdp."""
        import os

        trainer = self._build_trainer(tmp_path)
        os.makedirs(str(tmp_path), exist_ok=True)

        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        trainer._save_checkpoint(epoch=0, optimizer=optimizer, scheduler=None, scaler=scaler)

        ckpt_path = tmp_path / "checkpoint.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 0
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt

    def test_saves_ema_checkpoint_when_ema_enabled(self, tmp_path):
        """When EMA is active, checkpoint_ema.pt is written alongside checkpoint.pt."""
        import os

        trainer = self._build_trainer(tmp_path, use_ema=True, ema_decay=0.999)
        os.makedirs(str(tmp_path), exist_ok=True)

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        trainer._save_checkpoint(epoch=5, optimizer=optimizer, scheduler=None, scaler=scaler)

        assert (tmp_path / "checkpoint_ema.pt").exists()
        ema_ckpt = torch.load(str(tmp_path / "checkpoint_ema.pt"), map_location="cpu", weights_only=False)
        assert "model_state_dict" in ema_ckpt
        assert ema_ckpt["epoch"] == 5

    def test_save_every_epoch_copies_checkpoint(self, tmp_path):
        """save_every_epoch=True should call copy_checkpoint (lines 272-273)."""
        import os
        from unittest.mock import patch

        trainer = self._build_trainer(tmp_path, save_every_epoch=True)
        os.makedirs(str(tmp_path), exist_ok=True)

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        with patch("credit.trainers.base_trainer.copy_checkpoint") as mock_copy:
            trainer._save_checkpoint(epoch=3, optimizer=optimizer, scheduler=None, scaler=scaler)
            mock_copy.assert_called_once()
            args = mock_copy.call_args[0]
            assert args[1] == 3  # epoch passed to copy_checkpoint

    def test_use_scheduler_false_passes_none_sched_state(self, tmp_path):
        """When use_scheduler=False, scheduler_state_dict in checkpoint is None (line 252)."""
        import os

        trainer = self._build_trainer(tmp_path, use_scheduler=False)
        os.makedirs(str(tmp_path), exist_ok=True)

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        trainer._save_checkpoint(epoch=0, optimizer=optimizer, scheduler=None, scaler=scaler)

        ckpt = torch.load(str(tmp_path / "checkpoint.pt"), map_location="cpu", weights_only=False)
        assert ckpt["scheduler_state_dict"] is None

    def test_nonzero_rank_does_not_write_checkpoint(self, tmp_path):
        """Non-rank-0 processes must not write checkpoint files (lines 255-296)."""
        import os

        conf = _minimal_conf()
        conf["save_loc"] = str(tmp_path)
        # Create trainer at rank=1
        trainer = _ConcreteTrainer(_tiny_model(), rank=1, conf=conf)
        os.makedirs(str(tmp_path), exist_ok=True)

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        trainer._save_checkpoint(epoch=0, optimizer=optimizer, scheduler=None, scaler=scaler)

        # File should NOT exist since rank != 0
        assert not (tmp_path / "checkpoint.pt").exists()


# ---------------------------------------------------------------------------
# TrainerERA5Gen2 — additional coverage
# ---------------------------------------------------------------------------


class TestTrainerERA5Gen2AdditionalCoverage:
    """Cover trainerERA5gen2 init branches and validate loop."""

    def test_backprop_on_timestep_explicit(self, tmp_path):
        """Explicit backprop_on_timestep in data conf is used directly (lines 102-103)."""
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=3, tmp_path=tmp_path)
        conf["data"]["backprop_on_timestep"] = [2, 3]

        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.backprop_on_timestep == [2, 3]

    def test_data_clamp_sets_flags(self, tmp_path):
        """data_clamp in conf sets flag_clamp, clamp_min, clamp_max (lines 107-115)."""
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["data"]["data_clamp"] = [-5.0, 5.0]

        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.flag_clamp is True
        assert t.clamp_min == -5.0
        assert t.clamp_max == 5.0

    def test_data_valid_overrides_valid_forecast_len(self, tmp_path):
        """data_valid block used when present (lines 118-120)."""
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        conf["data_valid"] = {"forecast_len": 4, "history_len": 2}

        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.valid_forecast_len == 4
        assert t.valid_history_len == 2

    def test_retain_graph_flag(self, tmp_path):
        """retain_graph flag extracted from conf (line 98)."""
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["data"]["retain_graph"] = True

        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.retain_graph is True

    def test_validate_one_epoch_runs(self, tmp_path):
        """validate() completes on CPU with toy data and returns valid_loss."""
        B, C, H, W = 1, 4, 4, 4

        model = nn.Conv2d(C, C, kernel_size=1, bias=False)
        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)

        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        trainer = Trainer(model, rank=0, conf=conf)

        loader = _FakeLoader(B, C, H, W, n_batches=1)

        def _metrics(pred, y):
            return {"acc": 0.5, "mae": 0.1}

        criterion = nn.MSELoss()

        results = trainer.validate(
            epoch=0,
            valid_loader=loader,
            criterion=criterion,
            metrics=_metrics,
        )

        assert "valid_loss" in results
        assert len(results["valid_loss"]) == 1
        assert torch.isfinite(torch.tensor(results["valid_loss"][0]))

    def test_train_with_ema_update(self, tmp_path):
        """EMA update is called when ema is not None (line 252-253)."""
        B, C, H, W = 1, 4, 4, 4

        model = nn.Conv2d(C, C, kernel_size=1, bias=False)
        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["trainer"]["use_ema"] = True
        conf["trainer"]["ema_decay"] = 0.999

        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        trainer = Trainer(model, rank=0, conf=conf)

        assert trainer.ema is not None
        initial_step = trainer.ema.step

        loader = _FakeLoader(B, C, H, W, n_batches=1)

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        trainer.train_one_epoch(
            epoch=0,
            trainloader=loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=None,
            metrics=_metrics,
        )

        assert trainer.ema.step == initial_step + 1

    def test_grad_max_norm_clipping(self, tmp_path):
        """grad_max_norm > 0 triggers gradient clipping (lines 245-246)."""
        B, C, H, W = 1, 4, 4, 4

        model = nn.Conv2d(C, C, kernel_size=1, bias=False)
        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["trainer"]["grad_max_norm"] = 0.01  # very small to actually clip

        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        trainer = Trainer(model, rank=0, conf=conf)

        loader = _FakeLoader(B, C, H, W, n_batches=1)

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Should complete without error even with aggressive clipping
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

    def test_scheduler_lambda_step_called(self, tmp_path):
        """lambda scheduler steps once per epoch at epoch start (lines 146-147)."""
        from unittest.mock import MagicMock
        from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as Trainer

        B, C, H, W = 1, 4, 4, 4

        model = nn.Conv2d(C, C, kernel_size=1, bias=False)
        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["trainer"]["use_scheduler"] = True
        conf["trainer"]["scheduler"] = {"scheduler_type": "lambda"}

        trainer = Trainer(model, rank=0, conf=conf)

        mock_scheduler = MagicMock()
        loader = _FakeLoader(B, C, H, W, n_batches=1)

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        trainer.train_one_epoch(
            epoch=0,
            trainloader=loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=mock_scheduler,
            metrics=_metrics,
        )

        # Lambda scheduler should have been called at epoch start
        mock_scheduler.step.assert_called()
