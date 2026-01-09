import os

import torch

from credit.losses.kcrps import KCRPSLoss
from credit.losses.covariance import CovarianceWeightedMSELoss

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
