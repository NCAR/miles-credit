"""Resuming from a checkpoint continues training exactly where it stopped.

Regression for a resumed job diverging after every restart: the GPU memory preflight stepped the real
optimizer on a synthetic batch (overwriting the restored Adam moments), and the first resumed step ran
at the new scheduler's step-0 learning rate (0 during warmup) instead of the schedule's.
"""

import copy
import os

import pytest
import torch

from credit.trainers.preflight import synthetic_step
from credit.trainers.utils import load_model_states_and_optimizer


def _conf(save_loc, reload):
    return {
        "save_loc": str(save_loc),
        "trainer": {
            "mode": "none",
            "parallelism": {"data": "none", "tensor": 1, "domain": 1},
            "amp": False,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "use_scheduler": True,
            "scheduler": {
                "scheduler_type": "linear-warmup-cosine",
                "warmup_steps": 3,
                "total_steps": 20,
                "min_lr": 1e-4,
            },  # fmt: skip
            "load_weights": reload,
            "load_optimizer": reload,
            "load_scaler": reload,
            "load_scheduler": reload,
            "reload_epoch": reload,
            "start_epoch": 0,
        },
    }


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.utils.spectral_norm(torch.nn.Linear(4, 16)), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    )


def _build(save_loc, reload):
    _, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(
        copy.deepcopy(_conf(save_loc, reload)), _model(), torch.device("cpu")
    )
    synthetic_step(model, optimizer, torch.zeros(8, 4))  # what the trainer's GPU preflight does at every start
    return model, optimizer, scheduler, scaler


def _train(model, optimizer, scheduler, batches):
    losses = []
    for x, y in batches:
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        losses.append(loss.item())
    return losses


@pytest.fixture
def batches():
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(8, 4, generator=g) for _ in range(12)]
    return [(x, x.flip(1) * 0.5) for x in xs]


def test_resume_continues_exactly_like_an_uninterrupted_run(tmp_path, batches):
    straight = _train(*_build(tmp_path / "a", reload=False)[:3], batches)

    save_loc = tmp_path / "b"
    save_loc.mkdir()
    model, optimizer, scheduler, scaler = _build(save_loc, reload=False)
    first = _train(model, optimizer, scheduler, batches[:6])
    torch.save({"epoch": 0, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict()},
               os.path.join(save_loc, "checkpoint.pt"))  # fmt: skip
    (save_loc / "training_log.csv").write_text("epoch\n0\n")

    model, optimizer, scheduler, _ = _build(save_loc, reload=True)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(scheduler.get_last_lr()[0])
    assert optimizer.param_groups[0]["lr"] > 0  # not the new scheduler's warmup step 0
    resumed = first + _train(model, optimizer, scheduler, batches[6:])

    assert resumed == pytest.approx(straight, rel=1e-6, abs=1e-9)
