import os

import torch

import credit.domain_parallel.manager as domain_manager_mod
from credit.losses.gen_1.kcrps import KCRPSLoss
from credit.losses.gen_1.covariance import CovarianceWeightedMSELoss
from credit.losses.gen_1.weighted_loss import VariableTotalLoss2D

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
    monkeypatch.setattr(domain_manager_mod, "get_domain_parallel_manager", lambda: FakeDomainManager())

    target = torch.zeros(1, 1, 3, 2)
    pred = torch.ones_like(target)

    loss = loss_fn(target, pred)

    assert torch.isclose(loss, torch.tensor(5.0))


def test_all_registry_losses_resolve():
    """Every built-in _LOSS_REGISTRY entry imports and returns a class.

    Regression test: moving the Gen 1 loss modules into credit/losses/gen_1/
    left the registry module paths stale, and _load_loss_entry masks the
    resulting ModuleNotFoundError as "optional dependencies not installed".
    """
    from credit.losses import _LOSS_REGISTRY, _load_loss_entry

    for loss_type in _LOSS_REGISTRY:
        cls = _load_loss_entry(loss_type)
        assert isinstance(cls, type), f"{loss_type} resolved to {cls!r}, not a class"


def test_all_class_source_losses_importable():
    """``from credit.losses import <ClassName>`` works for every _CLASS_SOURCES entry."""
    import credit.losses as losses_pkg
    from credit.losses import _CLASS_SOURCES

    for class_name in _CLASS_SOURCES:
        assert isinstance(getattr(losses_pkg, class_name), type), class_name
