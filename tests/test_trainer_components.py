"""Unit tests for trainer components: EMATracker, BaseTrainer.__init__, LinearWarmupCosineScheduler,
and the load_trainer() factory.

All tests run on CPU with no data files required.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.utils as nnu
import warnings

try:
    from credit.trainers.base_trainer import EMATracker, BaseTrainer
    from credit.scheduler import LinearWarmupCosineScheduler

    _TRAINER_GEN2_AVAILABLE = True
except ImportError:
    _TRAINER_GEN2_AVAILABLE = False
warnings.filterwarnings("ignore", category=UserWarning)

pytestmark = pytest.mark.skipif(
    not _TRAINER_GEN2_AVAILABLE,
    reason="EMATracker / LinearWarmupCosineScheduler not available until v2/trainer-preblocks is merged",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model():
    return nn.Linear(4, 2)


class _ScaleModel(nn.Module):
    """Shape-preserving model: scales input by a learned scalar.
    Avoids cuDNN convolution so tests run on any GPU/driver version.
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.w


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


if _TRAINER_GEN2_AVAILABLE:

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
    """Minimal conf for trainer_gen2.Trainer (v2 nested data schema)."""
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
    base["preblocks"] = {"per_step": {"concat": {"type": "concat"}}}
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
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

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
    """Minimal conf for trainer_gen2.Trainer with forecast_len > 1."""
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
    base["preblocks"] = {"per_step": {"concat": {"type": "concat"}}}
    return base


class _FakeDataset:
    """Stub dataset so the batches_per_epoch resolution logic doesn't crash."""

    pass


class _FakeLoader:
    """Minimal iterable loader that yields nested-format batches for trainer_gen2.

    Each batch has the structure expected by apply_preblocks / ConcatToTensor:
        batch["input"]["era5"][var_key]  -> (B, 1, H, W) tensor
        batch["target"]["era5"][var_key] -> (B, 1, H, W) tensor

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
            input_vars = {f"era5/prognostic/2d/v{i}": torch.randn(B, 1, 1, H, W) for i in range(C)}
            target_vars = {f"era5/prognostic/2d/v{i}": torch.randn(B, 1, 1, H, W) for i in range(C)}
            yield {"input": {"era5": input_vars}, "target": {"era5": target_vars}}


class TestERA5Gen2MultiStepTraining:
    """Verify forecast_len > 1 in the v2 trainer autoregressive loop."""

    def test_forecast_len_2_init(self, tmp_path):
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.forecast_len == 2
        assert t.backprop_on_timestep == [1, 2]
        assert t.varnum_diag == 0

    def test_backprop_on_timestep_default_covers_all_steps(self, tmp_path):
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        for fl in [1, 2, 3]:
            conf = _era5_gen2_multistep_conf(forecast_len=fl, tmp_path=tmp_path)
            t = Trainer(_tiny_model(), rank=0, conf=conf)
            assert t.backprop_on_timestep == list(range(1, fl + 1))

    def test_2step_train_one_epoch_runs(self, tmp_path):
        """2-step autoregressive loop completes with toy data."""
        import torch
        import torch.nn as nn
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        B, C, H, W = 1, 4, 4, 4
        forecast_len = 2
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1x1 conv so shapes stay intact and model has real parameters
        model = _ScaleModel().to(device)

        conf = _era5_gen2_multistep_conf(forecast_len=forecast_len, tmp_path=tmp_path)
        conf["postblocks"] = {"per_step": {"reconstruct": {"type": "reconstruct"}}}
        trainer = Trainer(model, rank=0, conf=conf)

        # _FakeLoader yields forecast_len * batches_per_epoch batches total
        loader = _FakeLoader(B, C, H, W, n_batches=forecast_len)

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler(device.type, enabled=False)

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
        """At t=2, x must equal y_pred from t=1 (all vars are prognostic)."""
        import torch
        import torch.nn as nn
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        B, C, H, W = 1, 4, 4, 4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        captured = {}
        step = [0]

        class _CapturingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(1))

            def forward(self, x):
                out = x * self.w
                captured[step[0]] = {"x": x.detach().clone(), "out": out.detach().clone()}
                step[0] += 1
                return out

        model = _CapturingModel().to(device)
        conf = _era5_gen2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        conf["postblocks"] = {"per_step": {"reconstruct": {"type": "reconstruct"}}}
        trainer = Trainer(model, rank=0, conf=conf)

        loader = _FakeLoader(B, C, H, W, n_batches=2)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler(device.type, enabled=False)

        trainer.train_one_epoch(
            epoch=0,
            trainloader=loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=None,
            metrics=lambda p, y: {"acc": 0.0, "mae": 0.0},
        )

        # All vars are prognostic, model is identity (w=1), so y_pred at t=1 = x_t1.
        # Reconstruct + assemble_rollout_batch routes all channels through y_processed,
        # so x at t=2 must equal y_pred from t=1.
        y_pred_t1 = captured[0]["out"]
        x_t2 = captured[1]["x"]
        torch.testing.assert_close(x_t2, y_pred_t1, atol=1e-5, rtol=1e-5)

    def test_rollout_partial_channels_at_t2(self, tmp_path):
        """At t=2, ERA5Dataset returns only dynfrc channels.

        Verify assemble_rollout_batch correctly:
          - replaces prognostic channels with y_pred (via Reconstruct → y_processed)
          - preserves static channels from ic_preprocessed
          - updates dynfrc channels from the new batch
        Channel order follows FIELD_TYPE_RANK: prognostic < static < dynamic_forcing.
        """
        import torch
        import torch.nn as nn
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        N_PROG, N_STATIC, N_DYNFRC = 3, 1, 2
        B, H, W = 1, 4, 4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        conf["preblocks"] = {"per_step": {"concat": {"type": "concat"}}}
        conf["postblocks"] = {"per_step": {"reconstruct": {"type": "reconstruct"}}}

        # Known fixed tensors — 5D (B, 1, 1, H, W) to match ConcatToTensor expectations.
        dynfrc_t1 = torch.full((B, N_DYNFRC, 1, H, W), 1.0)
        static_ch = torch.full((B, N_STATIC, 1, H, W), 2.0)
        prog_t1 = torch.full((B, N_PROG, 1, H, W), 3.0)
        dynfrc_t2 = torch.full((B, N_DYNFRC, 1, H, W), 9.0)

        # Model: outputs N_PROG channels, all zeros — makes y_pred easy to verify.
        class _ZeroProgModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return torch.zeros(x.shape[0], N_PROG, *x.shape[2:], device=x.device) * self.w

        step = [0]
        captured_x = {}

        class _CapturingModel(_ZeroProgModel):
            def forward(self, x):
                captured_x[step[0]] = x.detach().clone()
                step[0] += 1
                return super().forward(x)

        class _PartialLoader:
            """Step 1: full batch. Step 2: dynfrc only (simulates ERA5Dataset i>0)."""

            dataset = _FakeDataset()
            sampler = None

            def __len__(self):
                return 2

            def __iter__(self):
                full_input = {}
                for i in range(N_PROG):
                    full_input[f"era5/prognostic/2d/p{i}"] = prog_t1[:, i : i + 1]
                for i in range(N_STATIC):
                    full_input[f"era5/static/2d/st{i}"] = static_ch[:, i : i + 1]
                for i in range(N_DYNFRC):
                    full_input[f"era5/dynamic_forcing/2d/df{i}"] = dynfrc_t1[:, i : i + 1]
                target = {f"era5/prognostic/2d/p{i}": prog_t1[:, i : i + 1] for i in range(N_PROG)}
                yield {"input": {"era5": full_input}, "target": {"era5": target}}

                partial_input = {f"era5/dynamic_forcing/2d/df{i}": dynfrc_t2[:, i : i + 1] for i in range(N_DYNFRC)}
                yield {"input": {"era5": partial_input}, "target": {"era5": target}}

        trainer = Trainer(_CapturingModel().to(device), rank=0, conf=conf)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler(device.type, enabled=False)
        trainer.train_one_epoch(
            epoch=0,
            trainloader=_PartialLoader(),
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=None,
            metrics=lambda p, y: {"acc": 0.0, "mae": 0.0},
        )

        x_t2 = captured_x[1]  # x fed to model at t=2, shape (B, TOTAL, 1, H, W)
        # Channel order: prognostic(0) < static(1) < dynamic_forcing(2) per FIELD_TYPE_RANK
        torch.testing.assert_close(
            x_t2[:, :N_PROG], torch.zeros(B, N_PROG, 1, H, W, device=device), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(x_t2[:, N_PROG : N_PROG + N_STATIC], static_ch.to(device), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(x_t2[:, N_PROG + N_STATIC :], dynfrc_t2.to(device), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# assemble_rollout_batch — direct unit tests
# ---------------------------------------------------------------------------


class TestAssembleRolloutBatch:
    """Direct unit tests for rollout_utils.assemble_rollout_batch.

    These verify the routing contract independently of the trainer loop:
      - prognostic / diagnostic → from y_processed (previous step's prediction)
      - dynamic_forcing          → from curr_batch["input"]
      - static (and unknown)     → always from ic_preprocessed["input"]
    """

    def test_static_always_comes_from_ic_preprocessed(self):
        """Static channels use ic_preprocessed values even when absent from curr_batch."""
        from credit.trainers.rollout_utils import assemble_rollout_batch

        STATIC_VAL = 42.0
        B, H, W = 1, 4, 4
        ic_preprocessed = {
            "input": {
                "era5": {
                    "era5/prognostic/2d/p": torch.full((B, 1, 1, H, W), 1.0),
                    "era5/static/2d/st": torch.full((B, 1, 1, H, W), STATIC_VAL),
                }
            }
        }
        y_processed = {"era5": {"era5/prognostic/2d/p": torch.full((B, 1, 1, H, W), 99.0)}}
        curr_batch = {"input": {"era5": {}}, "target": None}  # no static in curr_batch
        full_data_dict = {"ic_preprocessed": ic_preprocessed, "y_processed": y_processed}

        result = assemble_rollout_batch(full_data_dict, curr_batch)

        torch.testing.assert_close(
            result["input"]["era5"]["era5/static/2d/st"],
            torch.full((B, 1, 1, H, W), STATIC_VAL),
        )

    def test_prognostic_comes_from_y_processed(self):
        """Prognostic channels are routed from y_processed (previous step prediction)."""
        from credit.trainers.rollout_utils import assemble_rollout_batch

        PRED_VAL = 99.0
        B, H, W = 1, 4, 4
        ic_preprocessed = {"input": {"era5": {"era5/prognostic/2d/p": torch.full((B, 1, 1, H, W), 1.0)}}}
        y_processed = {"era5": {"era5/prognostic/2d/p": torch.full((B, 1, 1, H, W), PRED_VAL)}}
        curr_batch = {"input": {"era5": {}}, "target": None}
        full_data_dict = {"ic_preprocessed": ic_preprocessed, "y_processed": y_processed}

        result = assemble_rollout_batch(full_data_dict, curr_batch)

        torch.testing.assert_close(
            result["input"]["era5"]["era5/prognostic/2d/p"],
            torch.full((B, 1, 1, H, W), PRED_VAL),
        )

    def test_dynamic_forcing_comes_from_curr_batch(self):
        """Dynamic forcing channels are routed from curr_batch, not ic_preprocessed."""
        from credit.trainers.rollout_utils import assemble_rollout_batch

        DYNFRC_VAL = 7.0
        IC_VAL = 0.0
        B, H, W = 1, 4, 4
        ic_preprocessed = {"input": {"era5": {"era5/dynamic_forcing/2d/df": torch.full((B, 1, 1, H, W), IC_VAL)}}}
        y_processed = {"era5": {}}
        curr_batch = {
            "input": {"era5": {"era5/dynamic_forcing/2d/df": torch.full((B, 1, 1, H, W), DYNFRC_VAL)}},
            "target": None,
        }
        full_data_dict = {"ic_preprocessed": ic_preprocessed, "y_processed": y_processed}

        result = assemble_rollout_batch(full_data_dict, curr_batch)

        torch.testing.assert_close(
            result["input"]["era5"]["era5/dynamic_forcing/2d/df"],
            torch.full((B, 1, 1, H, W), DYNFRC_VAL),
        )

    def test_all_three_sources_assembled_together(self):
        """All three field types are assembled correctly in a single call."""
        from credit.trainers.rollout_utils import assemble_rollout_batch

        B, H, W = 1, 4, 4
        ic_preprocessed = {
            "input": {
                "era5": {
                    "era5/prognostic/2d/p": torch.full((B, 1, 1, H, W), 1.0),
                    "era5/static/2d/st": torch.full((B, 1, 1, H, W), 42.0),
                    "era5/dynamic_forcing/2d/df": torch.full((B, 1, 1, H, W), 0.0),
                }
            }
        }
        y_processed = {"era5": {"era5/prognostic/2d/p": torch.full((B, 1, 1, H, W), 99.0)}}
        curr_batch = {
            "input": {"era5": {"era5/dynamic_forcing/2d/df": torch.full((B, 1, 1, H, W), 7.0)}},
            "target": None,
        }
        full_data_dict = {"ic_preprocessed": ic_preprocessed, "y_processed": y_processed}

        result = assemble_rollout_batch(full_data_dict, curr_batch)

        torch.testing.assert_close(
            result["input"]["era5"]["era5/prognostic/2d/p"],
            torch.full((B, 1, 1, H, W), 99.0),
            msg="prognostic must come from y_processed",
        )
        torch.testing.assert_close(
            result["input"]["era5"]["era5/static/2d/st"],
            torch.full((B, 1, 1, H, W), 42.0),
            msg="static must come from ic_preprocessed",
        )
        torch.testing.assert_close(
            result["input"]["era5"]["era5/dynamic_forcing/2d/df"],
            torch.full((B, 1, 1, H, W), 7.0),
            msg="dynamic_forcing must come from curr_batch",
        )

    def test_y_processed_not_dict_raises_type_error(self):
        """TypeError raised when y_processed is not a dict (Reconstruct absent from chain)."""
        from credit.trainers.rollout_utils import assemble_rollout_batch

        full_data_dict = {
            "ic_preprocessed": {"input": {"era5": {}}},
            "y_processed": torch.randn(1, 4, 4, 4),  # wrong type — flat tensor
        }
        with pytest.raises(TypeError, match="y_processed"):
            assemble_rollout_batch(full_data_dict, {"input": {}})

    def test_target_forwarded_from_curr_batch(self):
        """The assembled batch's target is the same object as curr_batch['target']."""
        from credit.trainers.rollout_utils import assemble_rollout_batch

        B, H, W = 1, 4, 4
        target = {"era5": {"era5/prognostic/2d/T": torch.randn(B, 1, 1, H, W)}}
        curr_batch = {"input": {"era5": {}}, "target": target}
        full_data_dict = {
            "ic_preprocessed": {"input": {"era5": {}}},
            "y_processed": {"era5": {}},
        }

        result = assemble_rollout_batch(full_data_dict, curr_batch)

        assert result["target"] is target


# ---------------------------------------------------------------------------
# load_trainer — factory function
# ---------------------------------------------------------------------------


class TestLoadTrainer:
    """Tests for credit.trainers.load_trainer()."""

    def test_valid_era5_type_returns_class(self):
        from credit.trainers import load_trainer
        from credit.trainers.trainerERA5gen1 import TrainerERA5Gen1 as Trainer

        result = load_trainer({"trainer": {"type": "era5-gen1"}})
        assert result is Trainer

    def test_valid_era5v2_type_returns_class(self):
        from credit.trainers import load_trainer
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

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

        conf = {"trainer": {"type": "era5-gen1", "extra_key": 99}}
        load_trainer(conf)
        assert conf["trainer"]["type"] == "era5-gen1"
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

    def test_load_state_dict_strips_legacy_spectral_norm_uv(self):
        """load_state_dict must strip weight_u/weight_v from checkpoints saved before
        the spectral-norm filter was added.  Averaged u/v vectors are not unit vectors;
        leaving them in the shadow causes sigma = u^T W v to underestimate the spectral
        norm, making weight = weight_orig / sigma blow up in eval mode on resume."""
        sn_model = nn.Sequential(
            nnu.spectral_norm(nn.Linear(8, 8)),
            nnu.spectral_norm(nn.Linear(8, 4)),
        )

        # Run a forward pass so u/v are initialised (they start unset)
        _ = sn_model(torch.randn(2, 8))

        ema = EMATracker(sn_model, decay=0.9999)
        # Confirm the filter already works during __init__
        assert not any(k.endswith("_u") or k.endswith("_v") for k in ema.shadow)

        # Simulate a pre-filter checkpoint: inject averaged (non-unit) u/v
        state = sn_model.state_dict()
        legacy_shadow = dict(ema.shadow)
        for k, v in state.items():
            if k.endswith("_u") or k.endswith("_v"):
                fake = torch.randn_like(v)
                legacy_shadow[k] = v.float() * 0.5 + fake * 0.3  # not a unit vector

        legacy_state = {"shadow": legacy_shadow, "decay": 0.9999, "step": 500}

        ema2 = EMATracker(sn_model, decay=0.9999)
        ema2.load_state_dict(legacy_state)

        # After load, shadow must have no u/v keys
        bad_keys = [k for k in ema2.shadow if k.endswith("_u") or k.endswith("_v")]
        assert bad_keys == [], f"load_state_dict left stale u/v keys in shadow: {bad_keys}"

        # swap + eval forward must produce a sane sigma (≈1.0, not tiny)
        ema2.swap(sn_model)
        sn_model.eval()
        with torch.no_grad():
            _ = sn_model(torch.randn(2, 8))
        for mod in sn_model.modules():
            if hasattr(mod, "weight_orig"):
                u = getattr(mod, "weight_u")
                v = getattr(mod, "weight_v")
                w = mod.weight_orig.reshape(mod.weight_orig.size(0), -1)
                sigma = (u @ w @ v).item()
                assert sigma > 0.1, f"sigma={sigma:.4f} too small — u/v still corrupted after fix"
        ema2.swap(sn_model)

    def test_spectral_norm_resume_val_loss_stable(self):
        """End-to-end: val loss must not explode when resuming from a legacy EMA checkpoint
        (one whose shadow contained weight_u / weight_v before the filter was added)."""

        sn_model = nn.Sequential(
            nnu.spectral_norm(nn.Linear(16, 32)),
            nn.ReLU(),
            nnu.spectral_norm(nn.Linear(32, 16)),
        )
        opt = torch.optim.Adam(sn_model.parameters(), lr=1e-3)
        ema = EMATracker(sn_model, decay=0.9999)

        x = torch.randn(8, 16)
        y = torch.randn(8, 16)

        # Train briefly
        sn_model.train()
        for _ in range(30):
            opt.zero_grad()
            loss = (sn_model(x) - y).pow(2).mean()
            loss.backward()
            opt.step()
            ema.update(sn_model)

        # Reference val loss (EMA weights, clean checkpoint)
        ema.swap(sn_model)
        sn_model.eval()
        with torch.no_grad():
            ref_loss = (sn_model(x) - y).pow(2).mean().item()
        ema.swap(sn_model)
        sn_model.train()

        # Build a legacy shadow (inject u/v as if saved by old code)
        state = sn_model.state_dict()
        legacy_shadow = dict(ema.shadow)
        for k, v in state.items():
            if k.endswith("_u") or k.endswith("_v"):
                fake = torch.randn_like(v)
                legacy_shadow[k] = v.float() * 0.5 + fake * 0.3
        legacy_ema_state = {"shadow": legacy_shadow, "decay": 0.9999, "step": ema.step}

        # Resume with patched load_state_dict
        ema2 = EMATracker(sn_model, decay=0.9999)
        ema2.load_state_dict(legacy_ema_state)

        ema2.swap(sn_model)
        sn_model.eval()
        with torch.no_grad():
            resumed_loss = (sn_model(x) - y).pow(2).mean().item()
        ema2.swap(sn_model)

        assert resumed_loss < ref_loss * 5, (
            f"Val loss exploded on resume: {resumed_loss:.4f} vs ref {ref_loss:.4f} "
            f"(ratio {resumed_loss / ref_loss:.1f}x) — legacy u/v stripping may be broken"
        )


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
        conf["trainer"]["load_weights"] = True
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
    """Cover trainer_gen2 init branches and validate loop."""

    def test_backprop_on_timestep_explicit(self, tmp_path):
        """Explicit backprop_on_timestep in data conf is used directly (lines 102-103)."""
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=3, tmp_path=tmp_path)
        conf["data"]["backprop_on_timestep"] = [2, 3]

        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.backprop_on_timestep == [2, 3]

    def test_data_clamp_sets_flags(self, tmp_path):
        """data_clamp in conf sets flag_clamp, clamp_min, clamp_max (lines 107-115)."""
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["data"]["data_clamp"] = [-5.0, 5.0]

        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.flag_clamp is True
        assert t.clamp_min == -5.0
        assert t.clamp_max == 5.0

    def test_data_valid_overrides_valid_forecast_len(self, tmp_path):
        """validation_data block used when present (lines 118-120)."""
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        conf["validation_data"] = {"forecast_len": 4, "history_len": 2}

        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.valid_forecast_len == 4
        assert t.valid_history_len == 2

    def test_retain_graph_flag(self, tmp_path):
        """retain_graph flag extracted from conf (line 98)."""
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["data"]["retain_graph"] = True

        t = Trainer(_tiny_model(), rank=0, conf=conf)

        assert t.retain_graph is True

    def test_validate_one_epoch_runs(self, tmp_path):
        """validate() completes with toy data and returns valid_loss."""
        B, C, H, W = 1, 4, 4, 4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = _ScaleModel().to(device)
        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)

        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = _ScaleModel().to(device)
        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["trainer"]["use_ema"] = True
        conf["trainer"]["ema_decay"] = 0.999

        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        trainer = Trainer(model, rank=0, conf=conf)

        assert trainer.ema is not None
        initial_step = trainer.ema.step

        loader = _FakeLoader(B, C, H, W, n_batches=1)

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler(device.type, enabled=False)

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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = _ScaleModel().to(device)
        conf = _era5_gen2_multistep_conf(forecast_len=1, tmp_path=tmp_path)
        conf["trainer"]["grad_max_norm"] = 0.01  # very small to actually clip

        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        trainer = Trainer(model, rank=0, conf=conf)

        loader = _FakeLoader(B, C, H, W, n_batches=1)

        def _metrics(pred, y):
            return {"acc": 0.0, "mae": 0.0}

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler(device.type, enabled=False)

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
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        B, C, H, W = 1, 4, 4, 4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = _ScaleModel().to(device)
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
        scaler = torch.amp.GradScaler(device.type, enabled=False)

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

    def test_rollout_postblocks_called_once_after_loop(self, tmp_path):
        """apply_postblocks is invoked with rollout_postblocks exactly once after the rollout loop.

        This guards against accidentally removing the post-rollout apply_postblocks call
        at the end of train_one_epoch (not inside the per-step loop).
        """
        from unittest.mock import patch
        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer
        import credit.trainers.trainer_gen2 as trainer_module

        B, C, H, W = 1, 4, 4, 4
        forecast_len = 2
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = _ScaleModel().to(device)
        conf = _era5_gen2_multistep_conf(forecast_len=forecast_len, tmp_path=tmp_path)
        conf["postblocks"] = {"per_step": {"reconstruct": {"type": "reconstruct"}}}

        trainer = Trainer(model, rank=0, conf=conf)
        loader = _FakeLoader(B, C, H, W, n_batches=forecast_len)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler(device.type, enabled=False)

        real_apply = trainer_module.apply_postblocks
        rollout_postblock_calls = []

        def _tracking_apply(postblocks, fdd):
            result = real_apply(postblocks, fdd)
            if postblocks is trainer.rollout_postblocks:
                rollout_postblock_calls.append(True)
            return result

        with patch.object(trainer_module, "apply_postblocks", side_effect=_tracking_apply):
            trainer.train_one_epoch(
                epoch=0,
                trainloader=loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                scheduler=None,
                metrics=lambda p, y: {"acc": 0.0},
            )

        assert len(rollout_postblock_calls) == 1, (
            "rollout_postblocks must be applied exactly once after the autoregressive loop"
        )

    def test_validate_multistep_completes_with_reconstruct(self, tmp_path):
        """validate() with valid_forecast_len=2 completes and returns valid_loss.

        At t=2 the validate loop calls assemble_rollout_batch, which requires
        full_data_dict['y_processed'] to exist (populated by Reconstruct at t=1).
        This test confirms the two-step validate path is end-to-end functional.
        """
        B, C, H, W = 1, 4, 4, 4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = _ScaleModel().to(device)
        conf = _era5_gen2_multistep_conf(forecast_len=2, tmp_path=tmp_path)
        conf["postblocks"] = {"per_step": {"reconstruct": {"type": "reconstruct"}}}

        from credit.trainers.trainer_gen2 import TrainerERA5Gen2 as Trainer

        trainer = Trainer(model, rank=0, conf=conf)
        # Two batches per validation step (one per rollout step)
        loader = _FakeLoader(B, C, H, W, n_batches=2)

        criterion = nn.MSELoss()

        results = trainer.validate(
            epoch=0,
            valid_loader=loader,
            criterion=criterion,
            metrics=lambda pred, y: {"acc": 0.5, "mae": 0.1},
        )

        assert "valid_loss" in results
        assert len(results["valid_loss"]) == 1
        assert torch.isfinite(torch.tensor(results["valid_loss"][0]))
        assert results["valid_forecast_len"][-1] == 2
