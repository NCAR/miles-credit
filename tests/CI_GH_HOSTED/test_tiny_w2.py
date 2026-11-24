"""tests for artifacts produced by ci-gh-hosted.yml using tiny_w2.yml config.

TOL values in tests here were chosen empirically to be relatively lenient
but still nontrivial, based on values from actual training with this config.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml
from torch.utils.data import default_collate

from credit.cli._plot import _build_channel_map
from credit.datasets import MultiSourceDataset
from credit.models import load_model
from credit.preblock import apply_preblocks, build_preblocks

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "tiny_w2.yml")
# CREDIT_CI_DIR must be defined before running these tests (see also: test_tiny_w2_artifacts_exist)
CREDIT_CI_DIR = os.environ["CREDIT_CI_DIR"]
CHECKPOINT_FILE = os.path.join(CREDIT_CI_DIR, "tiny_w2/checkpoint.pt")
TRAINING_LOG_FILE = os.path.join(CREDIT_CI_DIR, "tiny_w2/training_log.csv")


def test_tiny_w2_artifacts_exist():
    """Ensure all artifacts required by these tests actually exist."""
    assert os.path.exists(CREDIT_CI_DIR)
    assert os.path.exists(CHECKPOINT_FILE)
    assert os.path.exists(TRAINING_LOG_FILE)


def test_training_log_looks_healthy():
    """Ensure that the training log csv file looks healthy and matches expectations:
    - 3 epochs total (as per CONFIG_FILE)
    - lr scheduler is working (lr always decreases across epochs)
    - train_loss decreases across all epochs
    - valid_loss decreases from epoch 0 to 1
    - valid_loss at epoch 2 is less than or "not too much greater than" at epoch 1
      (CONFIG_FILE tuned ema_decay for *fast* training, so some overfitting might occur.)
    """
    TRAIN_TOL = 0.9
    VALID_TOL01 = 0.7
    VALID_TOL12 = 1.5
    LR_TOL = 0.5
    log = pd.read_csv(TRAINING_LOG_FILE, index_col="epoch").to_xarray()
    assert log.sizes["epoch"] == 3
    log0 = log.sel(epoch=0)
    log1 = log.sel(epoch=1)
    log2 = log.sel(epoch=2)
    assert log1["train_loss"] / log0["train_loss"] < TRAIN_TOL
    assert log2["train_loss"] / log1["train_loss"] < TRAIN_TOL
    assert log1["valid_loss"] / log0["valid_loss"] < VALID_TOL01
    assert log2["valid_loss"] / log1["valid_loss"] < VALID_TOL12
    assert log1["lr"] / log0["lr"] < LR_TOL
    assert log2["lr"] / log1["lr"] < LR_TOL


@pytest.fixture(scope="module")
def init_pred_truth():
    """xarray Datasets of (initial conditions, prediction, ground truth),
    when considering the 0th and 1st time steps for the first batch of the validation dataset.
    Results have not been de-normalized; 0 corresponds to mean across training dataset.
    Variables are flattened across levels into f"{name}_{level}".

    CAUTION: avoid modifying the returned Datasets; they can be shared across tests.
    """
    conf = yaml.safe_load(open(CONFIG_FILE))

    model = load_model(conf, load_weights=False)
    ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    dataset = MultiSourceDataset({**conf["data"], **conf["validation_data"]}, return_target=True)
    t0 = dataset.datetimes[0]
    batch = apply_preblocks(build_preblocks(conf), default_collate([dataset[(t0, 0)]]), device="cpu")
    with torch.no_grad():
        y_pred = model(batch["x"])

    chmap = _build_channel_map(conf)
    levels = conf["data"]["source"]["ERA5"]["levels"]

    def to_ds(t):
        a = t.squeeze(0)
        if a.ndim == 4:  # (C, time, H, W) -> (C, H, W)
            a = a[:, 0]
        a = a.cpu().numpy()
        H, W = a.shape[-2:]
        lats = np.linspace(90, -90, H)  # still worth checking against ERA5_grid_schema.nc
        lons = np.linspace(0, 360, W, endpoint=False)
        data = {
            name: (("level", "lat", "lon"), a[ch]) if len(ch) > 1 else (("lat", "lon"), a[ch[0]])
            for name, ch in chmap.items()
        }
        return xr.Dataset(data, coords={"level": levels, "lat": lats, "lon": lons})

    def _flatten_levels(ds):
        """flatten levels from ds into separate variables as {name}_{lev}."""
        out = {}
        for name, da in ds.data_vars.items():
            if "level" in da.dims:
                for lev in da["level"].values:
                    out[f"{name}_{lev}"] = da.sel(level=lev, drop=True)
            else:
                out[name] = da
        return xr.Dataset(out)

    init = _flatten_levels(to_ds(batch["x"]))
    pred = _flatten_levels(to_ds(y_pred))
    truth = _flatten_levels(to_ds(batch["y"]))
    return init, pred, truth


def test_all_vars_present(init_pred_truth):
    """Ensure the set of data_vars agrees with what the CONFIG_FILE implies."""
    init, pred, truth = init_pred_truth
    expected_vars = {
        "temperature_500",
        "temperature_850",
        "specific_humidity_500",
        "specific_humidity_850",
        "surface_pressure",
        "2m_temperature",
    }
    assert set(init.data_vars) == expected_vars
    assert set(pred.data_vars) == expected_vars
    assert set(truth.data_vars) == expected_vars


def test_reasonable_spread(init_pred_truth):
    """Ensure that the model's spread is comparable to the truth's spread"""
    _, pred, truth = init_pred_truth
    pred_spread = pred.std(("lat", "lon"))
    truth_spread = truth.std(("lat", "lon"))
    ratios = (pred_spread / truth_spread).to_array()
    assert np.all(0.3 < ratios)
    assert np.all(ratios < 3.0)


def mae(f, truth):
    """mean absolute error across lat/lon dimensions"""
    return abs(f - truth).mean(("lat", "lon"))


def test_pred_beats_gauss(init_pred_truth):
    """Ensure that the model's prediction beats gaussian noise.
    (init_pred_truth was not denormalized, so just use mean=0, std=1 here.)
    """
    BASE_TOL = 0.6
    Q500_TOL = 0.9  # q at level=500 is harder for the model to learn quickly.
    init, pred, truth = init_pred_truth
    rng = np.random.default_rng(0)
    noise = init.map(lambda da: da.copy(data=rng.standard_normal(da.shape)))
    gauss = 0 + 1 * noise
    ratios_ds = mae(pred, truth) / mae(gauss, truth)
    TOLS_DS = BASE_TOL + xr.zeros_like(ratios_ds)
    TOLS_DS["specific_humidity_500"] = Q500_TOL
    ratios = ratios_ds.to_array()
    TOLS = TOLS_DS.to_array()
    assert np.all(ratios < TOLS)


def test_pred_beats_zeros(init_pred_truth):
    """Ensure that the model's prediction beats a zero baseline.
    (init_pred_truth was not denormalized, so zero implies mean across training dataset.)
    Does not check specific_humidity at level=500, which is hard for the model to learn quickly.
    """
    TOL = 0.9
    init, pred, truth = init_pred_truth
    zeros = xr.zeros_like(init)
    ratios_ds = mae(pred, truth) / mae(zeros, truth)
    ratios_ds = ratios_ds.drop_vars("specific_humidity_500")
    ratios = ratios_ds.to_array()
    assert np.all(ratios < TOL)
