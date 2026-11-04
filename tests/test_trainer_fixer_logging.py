"""Fixer-diagnostic logging in the gen2 trainer, and which columns reach training_log.csv."""

from collections import defaultdict
from types import SimpleNamespace

import torch

from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.trainer_gen2 import _collect_fixer_diagnostics


def test_collect_fixer_diagnostics_averages_and_pops():
    fixer_logs = defaultdict(list)
    for step_value in (1.0, 3.0):
        full_data_dict = {"fixer_diagnostics": {"energy_fixer/dT_K": torch.tensor([step_value, step_value + 2])}}
        _collect_fixer_diagnostics(fixer_logs, full_data_dict)
        assert "fixer_diagnostics" not in full_data_dict  # a later step must not re-log it
    assert fixer_logs == {"energy_fixer/dT_K": [2.0, 4.0]}

    _collect_fixer_diagnostics(fixer_logs, {})  # no fixers: no-op
    assert list(fixer_logs) == ["energy_fixer/dT_K"]


def test_fixer_and_raw_loss_columns_are_always_logged():
    trainer = SimpleNamespace(save_metric_vars=False, display_metrics=["rmse"])
    train = {"train_loss": [1.0], "train_rmse": [1.0], "train_fixer/energy_fixer/dT_K": [0.1]}
    valid = {"valid_loss": [1.0], "valid_loss_raw": [1.2], "valid_rmse/ERA5/prognostic/3d/T": [0.5]}
    names = BaseTrainer._select_log_metric_names(trainer, train, valid)
    assert {"loss", "rmse", "fixer/energy_fixer/dT_K", "loss_raw"} <= set(names)
    assert "rmse/ERA5/prognostic/3d/T" not in names  # per-variable columns still follow save_metric_vars
