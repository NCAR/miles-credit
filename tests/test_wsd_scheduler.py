"""WarmupStableDecayScheduler ("warmup-stable-decay"): values, config loading, per-batch stepping and resume."""

import math

import pytest
import torch

from credit.scheduler import WarmupStableDecayScheduler, load_scheduler, update_on_batch

A, PEAK, MIN = 4, 1e-3, 1e-5
# 40,000 optimizer steps x A micro-batches: warmup 1,000, constant to 34,000, 1-sqrt decay to 40,000
SPEC = {"warmup_steps": 1000 * A, "decay_start": 34000 * A, "total_steps": 40000 * A, "min_lr": MIN}


def _conf(**kw):
    return {"trainer": {"use_scheduler": True, "scheduler": {"scheduler_type": "warmup-stable-decay", **SPEC, **kw}}}


def _opt():
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=PEAK)


def _lr_at(sched, opt_step):
    sched.last_epoch = opt_step * A
    return sched.get_lr()[0]


@pytest.mark.parametrize(
    "opt_step, expected",
    [
        (0, 0.0),
        (500, PEAK / 2),
        (1000, PEAK),
        (34000, PEAK),
        (37000, MIN + (PEAK - MIN) * (1 - math.sqrt(0.5))),
        (40000, MIN),
        (41000, MIN),
    ],
)
def test_values(opt_step, expected):
    sched = load_scheduler(_opt(), _conf())
    assert isinstance(sched, WarmupStableDecayScheduler)
    assert _lr_at(sched, opt_step) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_linear_decay_and_argument_checks():
    assert _lr_at(load_scheduler(_opt(), _conf(decay="linear")), 37000) == pytest.approx(MIN + (PEAK - MIN) / 2)
    with pytest.raises(ValueError):
        WarmupStableDecayScheduler(_opt(), warmup_steps=10, decay_start=5, total_steps=20)
    with pytest.raises(ValueError):
        WarmupStableDecayScheduler(_opt(), warmup_steps=1, decay_start=5, total_steps=20, decay="cosine")


def test_stepped_per_batch():
    assert "warmup-stable-decay" in update_on_batch
    opt = _opt()
    sched = WarmupStableDecayScheduler(opt, warmup_steps=4, decay_start=8, total_steps=12, min_lr=MIN)
    lrs = []
    for _ in range(14):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[:5] == pytest.approx([0, PEAK / 4, PEAK / 2, 3 * PEAK / 4, PEAK])
    assert lrs[8] == pytest.approx(PEAK) and lrs[12:] == pytest.approx([MIN, MIN])
    assert all(a >= b for a, b in zip(lrs[4:], lrs[5:]))


def test_resume_through_scheduler_state_dict():
    """The trainer's resume path: load_scheduler builds a fresh scheduler, then loads scheduler_state_dict."""
    small = {"warmup_steps": 4, "decay_start": 8, "total_steps": 12}
    opt = _opt()
    sched = load_scheduler(opt, _conf(**small))
    for _ in range(9):  # into the decay
        opt.step()
        sched.step()
    state = sched.state_dict()
    lr_at_save = opt.param_groups[0]["lr"]
    ref = []
    for _ in range(4):
        opt.step()
        sched.step()
        ref.append(opt.param_groups[0]["lr"])

    opt2 = _opt()
    opt2.param_groups[0]["lr"] = lr_at_save  # the optimizer state dict carries the param-group LR
    sched2 = load_scheduler(opt2, _conf(**small))
    sched2.load_state_dict(state)
    got = []
    for _ in range(4):
        opt2.step()
        sched2.step()
        got.append(opt2.param_groups[0]["lr"])
    assert got == ref and sched2.last_epoch == 13 and ref[-1] == pytest.approx(MIN)
