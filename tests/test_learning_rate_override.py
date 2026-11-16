"""Tests for credit.trainers.utils.apply_learning_rate_override (trainer.update_learning_rate)."""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from credit.scheduler import CosineAnnealingWarmupRestarts, LinearWarmupCosineScheduler
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


def _restarts(opt):
    return CosineAnnealingWarmupRestarts(opt, first_cycle_steps=1000, max_lr=1e-3, min_lr=1e-6, warmup_steps=100)


def test_restarts_fresh_run_keeps_warmup():
    """base_lrs holds min_lr here; the override must not scale by learning_rate / min_lr."""
    opt = _optimizer(1e-3)
    sched = _restarts(opt)
    apply_learning_rate_override(opt, sched, 1e-3)
    assert abs(opt.param_groups[0]["lr"] - 1e-6) < 1e-15
    assert sched.base_lrs == [1e-6]
    for _ in range(100):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-12  # end of warmup reaches the peak


def test_restarts_resume_rescales_peak_and_keeps_position():
    """A new LR on resume replaces max_lr; min_lr stays, the cosine position is kept."""
    opt = _optimizer(1e-3)
    sched = _restarts(opt)
    for _ in range(550):  # mid-decay: cosine halfway between the peak and min_lr
        opt.step()
        sched.step()
    midpoint = 1e-6 + (1e-3 - 1e-6) / 2
    assert abs(opt.param_groups[0]["lr"] - midpoint) < 1e-12
    apply_learning_rate_override(opt, sched, 2e-3)
    assert abs(opt.param_groups[0]["lr"] - (1e-6 + (2e-3 - 1e-6) / 2)) < 1e-12
    assert sched.base_lrs == [1e-6]
    assert sched.base_max_lr == sched.max_lr == 2e-3
    for _ in range(450):  # end of the cycle decays back to min_lr
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-6) < 1e-12


def test_restarts_override_keeps_gamma_decay():
    """In a later cycle the peak is base_max_lr * gamma**cycle, rescaled by the same factor."""
    opt = _optimizer(1e-3)
    sched = CosineAnnealingWarmupRestarts(opt, first_cycle_steps=10, max_lr=1e-3, min_lr=0.0, gamma=0.5)
    for _ in range(10):
        opt.step()
        sched.step()
    assert sched.cycle == 1 and abs(sched.max_lr - 5e-4) < 1e-15
    apply_learning_rate_override(opt, sched, 2e-3)
    assert abs(sched.max_lr - 1e-3) < 1e-15
    opt.step()
    sched.step()
    assert abs(sched.max_lr - 1e-3) < 1e-15
