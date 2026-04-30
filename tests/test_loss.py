import os

import torch

import credit.losses.weighted_loss as weighted_loss
from credit.losses.kcrps import KCRPSLoss
from credit.losses.covariance import CovarianceWeightedMSELoss
from credit.losses.weighted_loss import VariableTotalLoss2D

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]), "config")


def test_KCRPS():
    loss_fn = KCRPSLoss("none")
    batch_size = 2
    ensemble_size = 5

    target = torch.randn(batch_size, 10, 1, 40, 50)
    pred = torch.randn(batch_size * ensemble_size, 10, 1, 40, 50)

    loss = loss_fn(target, pred)
    assert not torch.isnan(loss).any()


def test_CovarianceWeightedMSELoss():
    loss_fn = CovarianceWeightedMSELoss()
    batch_size = 2
    target = torch.randn(batch_size, 10, 1, 40, 50)
    pred = torch.randn(batch_size, 10, 1, 40, 50)
    loss = loss_fn(target, pred)
    assert not torch.isnan(loss).any()
    assert loss > 0


def test_variable_total_loss_slices_latitude_weights_for_domain_shard(monkeypatch):
    class FakeDomainManager:
        domain_parallel_size = 2
        domain_rank = 1

    conf = {
        "data": {
            "variables": ["t"],
            "surface_variables": [],
            "diagnostic_variables": [],
        },
        "model": {"levels": 1},
        "loss": {
            "training_loss": "mse",
            "validation_loss": "mse",
            "use_latitude_weights": False,
            "use_variable_weights": False,
            "use_spectral_loss": False,
            "use_power_loss": False,
        },
    }
    loss_fn = VariableTotalLoss2D(conf)
    loss_fn.lat_weights = torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1)
    monkeypatch.setattr(weighted_loss, "get_domain_parallel_manager", lambda: FakeDomainManager())

    target = torch.zeros(1, 1, 3, 2)
    pred = torch.ones_like(target)

    loss = loss_fn(target, pred)

    assert torch.isclose(loss, torch.tensor(5.0))
