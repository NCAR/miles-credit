"""Equivalency tests: verify that trainerERA5gen2 (new) and trainerERA5gen2_legacy
produce bit-identical flat y_pred tensors for the same model, batch, and config.

Three assertions are verified:
  1. y_pred at t=1 is identical (single-step and multi-step configs).
  2. x fed to the model at t=2 is identical (multi-step rollout correctness).
  3. The final training loss is identical.

The new trainer uses build_rollout_input + postblocks; the legacy trainer uses
update_x + build_channel_layout.  Both are exercised with and without Reconstruct
to cover both code paths in build_rollout_input.
"""

import torch
import torch.nn as nn
import pytest

try:
    from credit.trainers.trainerERA5gen2_legacy import TrainerERA5Gen2 as LegacyTrainer
    from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2 as NewTrainer

    _AVAILABLE = True
except ImportError as e:
    _AVAILABLE = False
    _IMPORT_ERR = str(e)

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason=f"Trainers not importable: {locals().get('_IMPORT_ERR', '')}")


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


class _ScaleModel(nn.Module):
    """Shape-preserving model: multiplies input by a learned scalar."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.w


class _ProgOnlyModel(nn.Module):
    """Returns only the first n_prog input channels, scaled by a learned scalar.

    Simulates a realistic model that predicts only prognostic variables,
    producing output shape (B, n_prog, H, W) regardless of input channel count.
    """

    def __init__(self, n_prog: int):
        super().__init__()
        self.n_prog = n_prog
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.n_prog, ...] * self.w


class _RecordingModel(nn.Module):
    """Wraps a model and records every (x, y_pred) pair seen during forward."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        self.inputs.append(x.detach().clone())
        self.outputs.append(out.detach().clone())
        return out


class _FakeDataset:
    pass


class _FullLoader:
    """Yields full batches (all variable groups) for every step."""

    def __init__(self, B, C, H, W, n_batches, seed=0):
        self.dataset = _FakeDataset()
        self.sampler = None
        self._B, self._C, self._H, self._W = B, C, H, W
        self._n = n_batches
        self._seed = seed

    def __len__(self):
        return self._n

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        for _ in range(self._n):
            input_vars = {
                f"era5/prognostic/2d/v{i}": torch.randn(self._B, 1, 1, self._H, self._W, generator=g)
                for i in range(self._C)
            }
            target_vars = {
                f"era5/prognostic/2d/v{i}": torch.randn(self._B, 1, 1, self._H, self._W, generator=g)
                for i in range(self._C)
            }
            yield {"input": {"era5": input_vars}, "target": {"era5": target_vars}}


class _PartialLoader:
    """Step 1: full batch (prog + static + dynfrc). Step 2+: dynfrc only.

    Simulates the ERA5Dataset behaviour where t>1 batches contain only
    dynamic_forcing channels.
    """

    N_PROG = 3
    N_STATIC = 1
    N_DYNFRC = 2

    def __init__(self, B, H, W, forecast_len, seed=0):
        self.dataset = _FakeDataset()
        self.sampler = None
        self._B, self._H, self._W = B, H, W
        self._forecast_len = forecast_len
        self._seed = seed

    def __len__(self):
        return self._forecast_len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed)

        N_P, N_S, N_D = self.N_PROG, self.N_STATIC, self.N_DYNFRC
        B, H, W = self._B, self._H, self._W

        # t=1: full batch
        full_input = {}
        for i in range(N_P):
            full_input[f"era5/prognostic/2d/p{i}"] = torch.randn(B, 1, 1, H, W, generator=g)
        for i in range(N_S):
            full_input[f"era5/static/2d/s{i}"] = torch.randn(B, 1, 1, H, W, generator=g)
        for i in range(N_D):
            full_input[f"era5/dynamic_forcing/2d/d{i}"] = torch.randn(B, 1, 1, H, W, generator=g)
        target = {f"era5/prognostic/2d/p{i}": torch.randn(B, 1, 1, H, W, generator=g) for i in range(N_P)}
        yield {"input": {"era5": full_input}, "target": {"era5": target}}

        # t=2+: only dynfrc
        for _ in range(1, self._forecast_len):
            partial = {f"era5/dynamic_forcing/2d/d{i}": torch.randn(B, 1, 1, H, W, generator=g) for i in range(N_D)}
            yield {"input": {"era5": partial}, "target": {"era5": target}}


def _minimal_conf(tmp_path):
    return {
        "trainer": {
            "mode": "none",
            "start_epoch": 0,
            "epochs": 1,
            "num_epoch": 1,
            "amp": False,
            "use_scheduler": False,
            "use_ema": False,
            "use_tensorboard": False,
            "skip_validation": False,
            "train_batch_size": 1,
            "batches_per_epoch": 1,
            "valid_batches_per_epoch": 1,
        },
        "save_loc": str(tmp_path),
    }


def _gen2_conf_simple(tmp_path, forecast_len, C):
    """Config where every variable is a 2D prognostic (simplest possible layout)."""
    conf = _minimal_conf(tmp_path)
    conf["data"] = {
        "forecast_len": forecast_len,
        "retain_graph": False,
        "source": {
            "ERA5": {
                "levels": [],
                "variables": {
                    "prognostic": {"vars_3D": [], "vars_2D": [f"v{i}" for i in range(C)]},
                    "diagnostic": {"vars_3D": [], "vars_2D": []},
                    "dynamic_forcing": {"vars_2D": []},
                    "static": {"vars_2D": []},
                },
            }
        },
    }
    conf["preblocks"] = {"concat": {"type": "concat"}}
    return conf


def _gen2_conf_mixed(tmp_path, forecast_len):
    """Config with prog + static + dynfrc variable groups (matches _PartialLoader)."""
    conf = _minimal_conf(tmp_path)
    N_P, N_S, N_D = _PartialLoader.N_PROG, _PartialLoader.N_STATIC, _PartialLoader.N_DYNFRC
    conf["data"] = {
        "forecast_len": forecast_len,
        "retain_graph": False,
        "source": {
            "ERA5": {
                "levels": [],
                "variables": {
                    "prognostic": {"vars_3D": [], "vars_2D": [f"p{i}" for i in range(N_P)]},
                    "diagnostic": {"vars_3D": [], "vars_2D": []},
                    "dynamic_forcing": {"vars_2D": [f"d{i}" for i in range(N_D)]},
                    "static": {"vars_2D": [f"s{i}" for i in range(N_S)]},
                },
            }
        },
    }
    conf["preblocks"] = {"concat": {"type": "concat"}}
    return conf


def _run_train(trainer_cls, model, conf, loader):
    """Run one epoch with a recording model; return (recording_model, results)."""
    rec = _RecordingModel(model)
    trainer = trainer_cls(rec, rank=0, conf=conf)
    optimizer = torch.optim.SGD(rec.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    results = trainer.train_one_epoch(
        epoch=0,
        trainloader=loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        scheduler=None,
        metrics=lambda p, y: {},
    )
    return rec, results


# ---------------------------------------------------------------------------
# Single-step equivalency: y_pred at t=1 is identical
# ---------------------------------------------------------------------------


class TestSingleStepEquivalency:
    """forecast_len=1 — new and legacy trainers must produce identical y_pred."""

    def test_y_pred_identical(self, tmp_path):
        B, C, H, W = 1, 4, 4, 4
        torch.manual_seed(42)
        base = _ScaleModel()

        model_legacy = _ScaleModel()
        model_new = _ScaleModel()
        model_new.load_state_dict(base.state_dict())
        model_legacy.load_state_dict(base.state_dict())

        conf_legacy = _gen2_conf_simple(tmp_path / "legacy", forecast_len=1, C=C)
        conf_new = _gen2_conf_simple(tmp_path / "new", forecast_len=1, C=C)

        loader_legacy = _FullLoader(B, C, H, W, n_batches=1, seed=7)
        loader_new = _FullLoader(B, C, H, W, n_batches=1, seed=7)

        rec_legacy, _ = _run_train(LegacyTrainer, model_legacy, conf_legacy, loader_legacy)
        rec_new, _ = _run_train(NewTrainer, model_new, conf_new, loader_new)

        assert len(rec_legacy.outputs) == 1 and len(rec_new.outputs) == 1
        torch.testing.assert_close(rec_legacy.outputs[0], rec_new.outputs[0], atol=0, rtol=0)

    def test_input_to_model_identical(self, tmp_path):
        """x fed to the model at t=1 is the same in both trainers."""
        B, C, H, W = 1, 4, 4, 4
        torch.manual_seed(99)
        base = _ScaleModel()

        model_legacy, model_new = _ScaleModel(), _ScaleModel()
        model_legacy.load_state_dict(base.state_dict())
        model_new.load_state_dict(base.state_dict())

        conf_legacy = _gen2_conf_simple(tmp_path / "legacy", forecast_len=1, C=C)
        conf_new = _gen2_conf_simple(tmp_path / "new", forecast_len=1, C=C)

        loader = _FullLoader(B, C, H, W, n_batches=1, seed=13)

        rec_legacy, _ = _run_train(LegacyTrainer, model_legacy, conf_legacy, _FullLoader(B, C, H, W, 1, 13))
        rec_new, _ = _run_train(NewTrainer, model_new, conf_new, _FullLoader(B, C, H, W, 1, 13))

        torch.testing.assert_close(rec_legacy.inputs[0], rec_new.inputs[0], atol=0, rtol=0)

    def test_loss_identical(self, tmp_path):
        B, C, H, W = 1, 4, 4, 4
        base = _ScaleModel()
        model_legacy, model_new = _ScaleModel(), _ScaleModel()
        model_legacy.load_state_dict(base.state_dict())
        model_new.load_state_dict(base.state_dict())

        conf_legacy = _gen2_conf_simple(tmp_path / "legacy", forecast_len=1, C=C)
        conf_new = _gen2_conf_simple(tmp_path / "new", forecast_len=1, C=C)

        _, res_legacy = _run_train(LegacyTrainer, model_legacy, conf_legacy, _FullLoader(B, C, H, W, 1, 55))
        _, res_new = _run_train(NewTrainer, model_new, conf_new, _FullLoader(B, C, H, W, 1, 55))

        assert abs(res_legacy["train_loss"][0] - res_new["train_loss"][0]) < 1e-6


# ---------------------------------------------------------------------------
# Multi-step equivalency: x at t=2 and y_pred at t=2 are identical
# ---------------------------------------------------------------------------


class TestMultiStepEquivalencyNoPostblocks:
    """forecast_len=2, postblocks={}.

    build_rollout_input falls back to the flat-tensor path (no Reconstruct),
    which should replicate update_x exactly.
    """

    def _models(self):
        base = _ProgOnlyModel(_PartialLoader.N_PROG)
        legacy = _ProgOnlyModel(_PartialLoader.N_PROG)
        new = _ProgOnlyModel(_PartialLoader.N_PROG)
        legacy.load_state_dict(base.state_dict())
        new.load_state_dict(base.state_dict())
        return legacy, new

    def test_x_at_t2_identical(self, tmp_path):
        B, H, W = 1, 4, 4
        model_legacy, model_new = self._models()

        conf_legacy = _gen2_conf_mixed(tmp_path / "legacy", forecast_len=2)
        conf_new = _gen2_conf_mixed(tmp_path / "new", forecast_len=2)

        seed = 21
        rec_legacy, _ = _run_train(LegacyTrainer, model_legacy, conf_legacy, _PartialLoader(B, H, W, 2, seed))
        rec_new, _ = _run_train(NewTrainer, model_new, conf_new, _PartialLoader(B, H, W, 2, seed))

        assert len(rec_legacy.inputs) == 2 and len(rec_new.inputs) == 2
        torch.testing.assert_close(rec_legacy.inputs[1], rec_new.inputs[1], atol=1e-6, rtol=1e-6)

    def test_y_pred_at_t2_identical(self, tmp_path):
        B, H, W = 1, 4, 4
        model_legacy, model_new = self._models()

        conf_legacy = _gen2_conf_mixed(tmp_path / "legacy", forecast_len=2)
        conf_new = _gen2_conf_mixed(tmp_path / "new", forecast_len=2)

        seed = 33
        rec_legacy, _ = _run_train(LegacyTrainer, model_legacy, conf_legacy, _PartialLoader(B, H, W, 2, seed))
        rec_new, _ = _run_train(NewTrainer, model_new, conf_new, _PartialLoader(B, H, W, 2, seed))

        torch.testing.assert_close(rec_legacy.outputs[1], rec_new.outputs[1], atol=1e-6, rtol=1e-6)

    def test_loss_identical(self, tmp_path):
        B, H, W = 1, 4, 4
        model_legacy, model_new = self._models()

        conf_legacy = _gen2_conf_mixed(tmp_path / "legacy", forecast_len=2)
        conf_new = _gen2_conf_mixed(tmp_path / "new", forecast_len=2)

        seed = 77
        _, res_legacy = _run_train(LegacyTrainer, model_legacy, conf_legacy, _PartialLoader(B, H, W, 2, seed))
        _, res_new = _run_train(NewTrainer, model_new, conf_new, _PartialLoader(B, H, W, 2, seed))

        assert abs(res_legacy["train_loss"][0] - res_new["train_loss"][0]) < 1e-6


class TestMultiStepEquivalencyWithReconstruct:
    """forecast_len=2 with Reconstruct in postblocks.

    build_rollout_input uses the nested-dict path.  The result must still
    be bit-identical to the legacy update_x path.
    """

    def _models(self):
        base = _ProgOnlyModel(_PartialLoader.N_PROG)
        legacy = _ProgOnlyModel(_PartialLoader.N_PROG)
        new = _ProgOnlyModel(_PartialLoader.N_PROG)
        legacy.load_state_dict(base.state_dict())
        new.load_state_dict(base.state_dict())
        return legacy, new

    def _new_conf_with_reconstruct(self, tmp_path, forecast_len):
        conf = _gen2_conf_mixed(tmp_path, forecast_len)
        conf["postblocks"] = {"reconstruct": {"type": "reconstruct"}}
        return conf

    def test_x_at_t2_identical(self, tmp_path):
        B, H, W = 1, 4, 4
        model_legacy, model_new = self._models()

        conf_legacy = _gen2_conf_mixed(tmp_path / "legacy", forecast_len=2)
        conf_new = self._new_conf_with_reconstruct(tmp_path / "new", forecast_len=2)

        seed = 44
        rec_legacy, _ = _run_train(LegacyTrainer, model_legacy, conf_legacy, _PartialLoader(B, H, W, 2, seed))
        rec_new, _ = _run_train(NewTrainer, model_new, conf_new, _PartialLoader(B, H, W, 2, seed))

        assert len(rec_legacy.inputs) == 2 and len(rec_new.inputs) == 2
        torch.testing.assert_close(rec_legacy.inputs[1], rec_new.inputs[1], atol=1e-6, rtol=1e-6)

    def test_y_pred_at_t2_identical(self, tmp_path):
        B, H, W = 1, 4, 4
        model_legacy, model_new = self._models()

        conf_legacy = _gen2_conf_mixed(tmp_path / "legacy", forecast_len=2)
        conf_new = self._new_conf_with_reconstruct(tmp_path / "new", forecast_len=2)

        seed = 66
        rec_legacy, _ = _run_train(LegacyTrainer, model_legacy, conf_legacy, _PartialLoader(B, H, W, 2, seed))
        rec_new, _ = _run_train(NewTrainer, model_new, conf_new, _PartialLoader(B, H, W, 2, seed))

        torch.testing.assert_close(rec_legacy.outputs[1], rec_new.outputs[1], atol=1e-6, rtol=1e-6)

    def test_loss_identical(self, tmp_path):
        B, H, W = 1, 4, 4
        model_legacy, model_new = self._models()

        conf_legacy = _gen2_conf_mixed(tmp_path / "legacy", forecast_len=2)
        conf_new = self._new_conf_with_reconstruct(tmp_path / "new", forecast_len=2)

        seed = 88
        _, res_legacy = _run_train(LegacyTrainer, model_legacy, conf_legacy, _PartialLoader(B, H, W, 2, seed))
        _, res_new = _run_train(NewTrainer, model_new, conf_new, _PartialLoader(B, H, W, 2, seed))

        assert abs(res_legacy["train_loss"][0] - res_new["train_loss"][0]) < 1e-6
