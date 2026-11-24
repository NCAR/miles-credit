"""Mid-epoch (batch-level) checkpoint/resume for the gen2 trainer.

CPU-only: a tiny model with dropout (so the RNG restore matters), a
deterministic map-style dataset driven by the real
DistributedMultiStepBatchSampler, and a linear-warmup-cosine scheduler
(stepped per batch, so a wrong resume position shows up in the LR).
"""

import copy
import logging
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from credit.samplers import DistributedMultiStepBatchSampler
from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.trainer_gen2 import TrainerERA5Gen2
from credit.trainers.utils import load_model_states_and_optimizer

warnings.filterwarnings("ignore", category=UserWarning)

N_TIMES, BATCH_SIZE, C, H, W = 24, 2, 4, 4, 4
BATCHES_PER_EPOCH = N_TIMES // BATCH_SIZE  # 12


@pytest.fixture(autouse=True)
def _pin_cpu(monkeypatch):
    """Keep the trainer on CPU: select_device() would pick MPS on Apple Silicon."""
    import credit.trainers.base_trainer as base_trainer

    monkeypatch.setattr(base_trainer, "select_device", lambda *a, **kw: torch.device("cpu"))


class _SeqDataset(Dataset):
    """Each init time maps to fixed random fields, so batches are reproducible."""

    def __init__(self):
        self.datetimes = pd.date_range("2020-01-01", periods=N_TIMES, freq="6h")

    def __len__(self):
        return N_TIMES

    def __getitem__(self, key):
        t, _ = key
        g = torch.Generator().manual_seed(int(pd.Timestamp(t).value // 10**9))

        def fields():
            return {f"era5/prognostic/2d/v{i}": torch.randn(1, 1, H, W, generator=g) for i in range(C)}

        return {"input": {"era5": fields()}, "target": {"era5": fields()}}


class _DropoutScaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return nn.functional.dropout(x, p=0.3, training=self.training) * self.w


def _conf(save_loc, **trainer_overrides):
    trainer = {
        "mode": "none",
        "start_epoch": 0,
        "epochs": 1,
        "amp": False,
        "use_ema": False,
        "use_tensorboard": False,
        "skip_validation": True,
        "train_batch_size": BATCH_SIZE,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "use_scheduler": True,
        "scheduler": {"scheduler_type": "linear-warmup-cosine", "warmup_steps": 4, "total_steps": 40},
    }
    trainer.update(trainer_overrides)
    return {
        "save_loc": str(save_loc),
        "seed": 0,
        "trainer": trainer,
        "data": {
            "forecast_len": 1,
            "retain_graph": False,
            "source": {
                "era5": {
                    "levels": [],
                    "variables": {
                        "prognostic": {"vars_3D": [], "vars_2D": [f"v{i}" for i in range(C)]},
                        "diagnostic": {"vars_3D": [], "vars_2D": []},
                        "dynamic_forcing": {"vars_2D": []},
                        "static": {"vars_2D": []},
                    },
                }
            },
        },
        "preblocks": {"per_step": {"concat": {"type": "concat"}}},
        "postblocks": {"per_step": {"reconstruct": {"type": "reconstruct"}}},
    }


def _loader(num_workers=0):
    dataset = _SeqDataset()
    sampler = DistributedMultiStepBatchSampler(
        dataset, batch_size=BATCH_SIZE, num_forecast_steps=1, num_replicas=1, rank=0, shuffle=True, seed=0
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)


def _metrics(pred, y):
    return {"mae": (pred - y).abs().mean().item()}


def _train_epoch_with_trainer(conf, num_workers=0):
    """Build model/optimizer the way train_gen2 does, then run epoch start_epoch."""
    torch.manual_seed(0)
    conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(
        conf, _DropoutScaleModel(), torch.device("cpu")
    )
    trainer = TrainerERA5Gen2(model, rank=0, conf=conf)
    trainer._restore_resume_state(conf["trainer"]["start_epoch"])
    results = trainer.train_one_epoch(
        conf["trainer"]["start_epoch"], _loader(num_workers), optimizer, nn.MSELoss(), scaler, scheduler, _metrics
    )
    return trainer, optimizer, results


def _train_epoch(conf, num_workers=0):
    trainer, optimizer, results = _train_epoch_with_trainer(conf, num_workers)
    return trainer.model, optimizer, results


def _resume_conf(conf):
    conf = copy.deepcopy(conf)
    conf["trainer"].update(
        load_weights=True, load_optimizer=True, load_scheduler=True, load_scaler=True, reload_epoch=True
    )
    return conf


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def test_skip_first_batches_until_reset():
    sampler = _loader().batch_sampler
    sampler.set_epoch(3)
    full = list(sampler)

    sampler.set_epoch(3)
    sampler.skip_first_batches(5)
    assert list(sampler) == full[5:]
    # Not one-shot: a multi-worker DataLoader iterator calls iter(sampler) twice on creation.
    assert list(sampler) == full[5:]
    sampler.skip_first_batches(0)
    assert list(sampler) == full


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resumed_iterator_skips_and_later_passes_do_not(num_workers):
    """The trainer's pattern: skip, create the iterator, reset; cycle() re-iteration starts at batch 0."""
    loader = _loader(num_workers)
    loader.batch_sampler.set_epoch(0)
    full = [b["input"]["era5"]["era5/prognostic/2d/v0"] for b in loader]

    loader.batch_sampler.set_epoch(0)
    loader.batch_sampler.skip_first_batches(5)
    it = iter(loader)
    loader.batch_sampler.skip_first_batches(0)
    resumed = [b["input"]["era5"]["era5/prognostic/2d/v0"] for b in it]
    torch.testing.assert_close(resumed, full[5:])
    torch.testing.assert_close([b["input"]["era5"]["era5/prognostic/2d/v0"] for b in loader], full)


# ---------------------------------------------------------------------------
# Interval resolution / config validation
# ---------------------------------------------------------------------------


class _EndOfEpochOnly(BaseTrainer):
    """A trainer without mid-epoch checkpoint support (like every non-gen2 trainer)."""

    def train_one_epoch(self, *a, **kw):
        pass

    def validate(self, *a, **kw):
        return {}


class _Concrete(_EndOfEpochOnly):
    supports_mid_epoch_checkpoints = True


@pytest.mark.parametrize(
    "overrides, bpe, accum, expected",
    [
        ({}, 100, 1, 0),
        ({"checkpoints_per_epoch": 4}, 100, 1, 25),
        ({"checkpoints_per_epoch": 4}, 100, 4, 28),  # rounded up to an optimizer step
        ({"checkpoints_per_epoch": 3}, 100, 1, 34),
        ({"checkpoints_per_epoch": 1}, 100, 1, 100),  # end-of-epoch only -> no mid-epoch save
    ],
)
def test_mid_epoch_interval(tmp_path, overrides, bpe, accum, expected):
    trainer = _Concrete(nn.Linear(1, 1), rank=0, conf=_conf(tmp_path, **overrides))
    assert trainer._mid_epoch_checkpoint_interval(bpe, accum) == expected


def test_negative_checkpoints_per_epoch_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be >= 0"):
        _Concrete(nn.Linear(1, 1), rank=0, conf=_conf(tmp_path, checkpoints_per_epoch=-1))


def test_unsupported_trainer_ignores_checkpoints_per_epoch(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        trainer = _EndOfEpochOnly(nn.Linear(1, 1), rank=0, conf=_conf(tmp_path, checkpoints_per_epoch=4))
    assert trainer.checkpoints_per_epoch == 0
    assert "is ignored" in caplog.text


def test_legacy_fsdp_rejects_mid_epoch_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="legacy FSDP"):
        _Concrete(nn.Linear(1, 1), rank=0, conf=_conf(tmp_path, mode="fsdp", checkpoints_per_epoch=4))


def test_rescale_start_batch_on_layout_change():
    same = {"num_replicas": 4, "batch_size": 2}
    assert BaseTrainer._rescale_start_batch(10, same, dict(same)) == 10
    assert BaseTrainer._rescale_start_batch(10, None, same) == 10
    # 10 batches x 2 x 4 ranks = 80 samples; with 8 ranks x 2 that's 5 batches.
    assert BaseTrainer._rescale_start_batch(10, same, {"num_replicas": 8, "batch_size": 2}) == 5


# ---------------------------------------------------------------------------
# End to end: interrupted + resumed == uninterrupted
# ---------------------------------------------------------------------------


def test_mid_epoch_checkpoint_contents(tmp_path):
    _train_epoch(_conf(tmp_path, checkpoints_per_epoch=3))
    ckpt = torch.load(os.path.join(tmp_path, "checkpoint.pt"), weights_only=False)
    # Saves at 4 and 8; 12 is the end of the epoch, which fit() saves.
    assert ckpt["epoch"] == 0
    assert ckpt["batch_in_epoch"] == 8
    assert len(ckpt["train_results"]["train_loss"]) == 8
    assert ckpt["sampler_layout"] == {"num_replicas": 1, "batch_size": BATCH_SIZE}
    assert len(ckpt["rng_states"]) == 1
    assert not os.path.exists(os.path.join(tmp_path, "checkpoint.pt.tmp"))


def test_resume_matches_uninterrupted(tmp_path):
    ref_model, ref_opt, ref_results = _train_epoch(_conf(tmp_path / "ref"))

    # The full epoch writes mid-epoch checkpoints; the last one (batch 8) is
    # what a job killed during batches 9-12 would leave behind.
    run_dir = tmp_path / "run"
    conf = _conf(run_dir, checkpoints_per_epoch=3)
    _train_epoch(conf)

    resumed_conf = _resume_conf(conf)
    model, opt, results = _train_epoch(resumed_conf)

    assert resumed_conf["trainer"]["start_epoch"] == 0
    assert resumed_conf["trainer"]["resume_at_batch"] == 8
    np.testing.assert_allclose(results["train_loss"], ref_results["train_loss"], rtol=1e-6)
    assert len(results["train_loss"]) == BATCHES_PER_EPOCH
    torch.testing.assert_close(model.w, ref_model.w)
    assert opt.param_groups[0]["lr"] == pytest.approx(ref_opt.param_groups[0]["lr"])


def _assert_resume_matches(tmp_path, num_workers=0, **overrides):
    """Interrupted at the last mid-epoch checkpoint + resumed == one uninterrupted epoch."""
    ref_trainer, ref_opt, ref_results = _train_epoch_with_trainer(_conf(tmp_path / "ref", **overrides), num_workers)

    conf = _conf(tmp_path / "run", checkpoints_per_epoch=3, **overrides)
    _train_epoch(conf, num_workers)
    trainer, opt, results = _train_epoch_with_trainer(_resume_conf(conf), num_workers)

    assert trainer.resume_at_batch == 8
    np.testing.assert_allclose(results["train_loss"], ref_results["train_loss"], rtol=1e-6)
    torch.testing.assert_close(trainer.model.w, ref_trainer.model.w)
    assert opt.param_groups[0]["lr"] == pytest.approx(ref_opt.param_groups[0]["lr"])
    return ref_trainer, trainer


def test_resume_with_ema_matches_uninterrupted(tmp_path):
    ref_trainer, trainer = _assert_resume_matches(tmp_path, use_ema=True, ema_decay=0.5)
    torch.testing.assert_close(trainer.ema.state_dict(), ref_trainer.ema.state_dict())
    assert trainer.ema.step == ref_trainer.ema.step


def test_resume_with_grad_accum_matches_uninterrupted(tmp_path):
    # Interval ceil(12 / 3) = 4 is already a multiple of 2, so the last save is at batch 8.
    _assert_resume_matches(tmp_path, grad_accum_every=2)


def test_resume_with_lambda_scheduler_matches_uninterrupted(tmp_path):
    """The per-epoch lambda step is in the saved scheduler state, so resume must not step again."""
    _assert_resume_matches(tmp_path, scheduler={"scheduler_type": "lambda"})


def test_resume_with_dataloader_workers_matches_uninterrupted(tmp_path):
    """With num_workers > 0 the worker seeds differ after resume; a dataset that doesn't sample still matches."""
    _assert_resume_matches(tmp_path, num_workers=2)


def test_ema_checkpoint_mismatch_warns(tmp_path, caplog):
    conf = _conf(tmp_path, checkpoints_per_epoch=3, use_ema=True)
    _train_epoch(conf)
    # Simulate a kill between the two renames: checkpoint.pt at batch 8, EMA still from batch 4.
    ema_path = os.path.join(tmp_path, "checkpoint_ema.pt")
    ema = torch.load(ema_path, weights_only=False)
    ema["batch_in_epoch"] = 4
    torch.save(ema, ema_path)

    with caplog.at_level(logging.WARNING):
        _Concrete(_DropoutScaleModel(), rank=0, conf=_resume_conf(conf))
    assert "checkpoint_ema.pt (epoch 0, batch 4) does not match checkpoint.pt (epoch 0, batch 8)" in caplog.text


def test_ema_checkpoint_saved_with_model_checkpoint(tmp_path, caplog):
    conf = _conf(tmp_path, checkpoints_per_epoch=3, use_ema=True)
    _train_epoch(conf)
    assert torch.load(os.path.join(tmp_path, "checkpoint_ema.pt"), weights_only=False)["batch_in_epoch"] == 8
    assert not os.path.exists(os.path.join(tmp_path, "checkpoint_ema.pt.tmp"))
    with caplog.at_level(logging.WARNING):
        _Concrete(_DropoutScaleModel(), rank=0, conf=_resume_conf(conf))
    assert "does not match" not in caplog.text


def test_sampler_without_skip_fails_before_training(tmp_path):
    """A run that couldn't resume from its own mid-epoch checkpoints fails at the start, not on resume."""
    conf = _conf(tmp_path, checkpoints_per_epoch=3)
    _, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(
        conf, _DropoutScaleModel(), torch.device("cpu")
    )
    trainer = TrainerERA5Gen2(model, rank=0, conf=conf)
    plain_loader = DataLoader(_SeqDataset(), batch_size=BATCH_SIZE)
    with pytest.raises(RuntimeError, match="skip_first_batches"):
        trainer.train_one_epoch(0, plain_loader, optimizer, nn.MSELoss(), scaler, scheduler, _metrics)
    assert not os.path.exists(os.path.join(tmp_path, "checkpoint.pt"))


@pytest.mark.parametrize("has_log", [True, False])
def test_weights_only_reload_restarts_mid_epoch_checkpoint_at_batch_0(tmp_path, has_log):
    """Fresh optimizer/scheduler: resuming at the saved batch would restart the LR schedule mid-epoch."""
    conf = _conf(tmp_path, checkpoints_per_epoch=3)
    _train_epoch(conf)
    if has_log:  # an earlier epoch finished, as in a multi-epoch run
        pd.DataFrame({"epoch": [0]}).to_csv(tmp_path / "training_log.csv", index=False)

    weights_only = copy.deepcopy(conf)
    weights_only["trainer"].update(load_weights=True, reload_epoch=True)
    resumed_conf, *_ = load_model_states_and_optimizer(weights_only, _DropoutScaleModel(), torch.device("cpu"))
    assert resumed_conf["trainer"]["start_epoch"] == 0
    assert resumed_conf["trainer"].get("resume_at_batch", 0) == 0


def test_end_of_epoch_checkpoint_resumes_next_epoch(tmp_path):
    """A batch_in_epoch=0 checkpoint keeps the existing next-epoch behaviour."""
    conf = _conf(tmp_path)
    model, optimizer, results = _train_epoch(conf)
    trainer = TrainerERA5Gen2(model, rank=0, conf=conf)
    trainer._save_checkpoint(0, optimizer, None, torch.amp.GradScaler("cpu", enabled=False))

    resumed_conf = _resume_conf(conf)
    resumed_conf["trainer"]["use_scheduler"] = False
    load_model_states_and_optimizer(resumed_conf, _DropoutScaleModel(), torch.device("cpu"))
    assert resumed_conf["trainer"]["start_epoch"] == 1
    assert resumed_conf["trainer"]["resume_at_batch"] == 0


def test_resume_restores_optimizer_lr(tmp_path):
    """Rebuilding the scheduler must not clobber the restored LR (it reset to 0)."""
    conf = _conf(tmp_path, checkpoints_per_epoch=3)
    _, optimizer, _ = _train_epoch(conf)
    saved = torch.load(os.path.join(tmp_path, "checkpoint.pt"), weights_only=True)

    _, _, resumed_opt, _, _ = load_model_states_and_optimizer(
        _resume_conf(conf), _DropoutScaleModel(), torch.device("cpu")
    )
    assert resumed_opt.param_groups[0]["lr"] == pytest.approx(saved["optimizer_state_dict"]["param_groups"][0]["lr"])
    assert resumed_opt.param_groups[0]["lr"] > 0


class _Killed(Exception):
    pass


def _fit(conf, kill_after_batch=None):
    torch.manual_seed(0)
    conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(
        conf, _DropoutScaleModel(), torch.device("cpu")
    )
    trainer = TrainerERA5Gen2(model, rank=0, conf=conf)
    if kill_after_batch is not None:
        real_save = trainer._save_checkpoint

        def save_then_die(*args, resume_state=None, **kwargs):
            real_save(*args, resume_state=resume_state, **kwargs)
            if resume_state and resume_state["batch_in_epoch"] == kill_after_batch:
                raise _Killed

        trainer._save_checkpoint = save_then_die
    trainer.fit(conf, _loader(), None, optimizer, nn.MSELoss(), nn.MSELoss(), scaler, scheduler, _metrics)
    return model


def test_fit_killed_mid_epoch_resumes_to_same_result(tmp_path):
    ref_model = _fit(_conf(tmp_path / "ref", epochs=2))
    ref_log = pd.read_csv(tmp_path / "ref" / "training_log.csv")

    conf = _conf(tmp_path / "run", epochs=2, checkpoints_per_epoch=3)
    with pytest.raises(_Killed):
        _fit(copy.deepcopy(conf), kill_after_batch=8)
    # The kill happens in epoch 0, before any end-of-epoch save or log row.
    assert not os.path.exists(tmp_path / "run" / "training_log.csv")

    model = _fit(_resume_conf(conf))
    log = pd.read_csv(tmp_path / "run" / "training_log.csv")

    torch.testing.assert_close(model.w, ref_model.w)
    assert list(log["epoch"]) == [0, 1]
    np.testing.assert_allclose(log["train_loss"], ref_log["train_loss"], rtol=1e-6)
