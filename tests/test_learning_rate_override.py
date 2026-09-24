"""Tests for credit.trainers.utils.apply_learning_rate_override (trainer.update_learning_rate)."""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from credit.scheduler import LinearWarmupCosineScheduler
from credit.trainers.utils import apply_learning_rate_override


def _optimizer(lr):
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=lr)


def test_fresh_run_keeps_warmup_at_zero():
    """Step 0 of a linear warmup must stay at 0, not jump to the full LR."""
    opt = _optimizer(1e-3)
    sched = LinearWarmupCosineScheduler(opt, warmup_steps=100, total_steps=1000)
    apply_learning_rate_override(opt, sched, 1e-3)
    assert opt.param_groups[0]["lr"] == 0.0
    opt.step()
    sched.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-5) < 1e-12


def test_resume_rescales_scheduler_base():
    """A new LR on resume becomes the scheduler's base and keeps its position."""
    opt = _optimizer(1e-3)
    sched = LinearWarmupCosineScheduler(opt, warmup_steps=100, total_steps=1000)
    for _ in range(50):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 5e-4) < 1e-12
    apply_learning_rate_override(opt, sched, 2e-3)
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-12
    assert sched.base_lrs == [2e-3]
    opt.step()
    sched.step()
    assert abs(opt.param_groups[0]["lr"] - 2e-3 * 51 / 100) < 1e-12


def test_no_scheduler_sets_lr_directly():
    opt = _optimizer(1e-3)
    apply_learning_rate_override(opt, None, 5e-4)
    assert opt.param_groups[0]["lr"] == 5e-4


def test_plateau_scheduler_sets_lr_directly():
    """ReduceLROnPlateau has no base_lrs, so the LR is overwritten as before."""
    opt = _optimizer(1e-3)
    sched = ReduceLROnPlateau(opt)
    apply_learning_rate_override(opt, sched, 5e-4)
    assert opt.param_groups[0]["lr"] == 5e-4
