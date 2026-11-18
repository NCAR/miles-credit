"""Tests for credit.losses.variances: loss-space variances measured by credit preprocess."""

import json

import numpy as np
import pytest
import torch
from credit.losses import BaseLoss
from credit.losses.variances import (
    LossVarianceAccumulator,
    load_loss_variances,
    observed_masks,
    resolve_target_variances,
    save_loss_variances,
    target_twin_conf,
)

SRC = "ERA5"
VAR_T = f"{SRC}/prognostic/3d/T"
VAR_SST = f"{SRC}/prognostic/2d/SST"
VAR_MASK = f"{SRC}/static/2d/LANDMASK"


def _reference(x, mask, lat_w):
    """Direct per-level temporal / pooled variances and observed fraction. x: (S, L, H, W)."""
    S, L, H, W = x.shape
    w = np.broadcast_to(lat_w[:, None], (H, W))
    temporal, pooled, observed = [], [], []
    for lev in range(L):
        num = den = 0.0
        vals, weights = [], []
        for h in range(H):
            for c in range(W):
                obs = x[mask[:, lev, h, c], lev, h, c]
                if obs.size >= 2:
                    num += w[h, c] * obs.size * obs.var()
                    den += w[h, c] * obs.size
                    vals.append(obs)
                    weights.append(np.full(obs.size, w[h, c]))
        temporal.append(num / den)
        v, ww = np.concatenate(vals), np.concatenate(weights)
        mu = np.sum(ww * v) / ww.sum()
        pooled.append(np.sum(ww * (v - mu) ** 2) / ww.sum())
        observed.append(np.sum(w * mask[:, lev].sum(0)) / (w.sum() * S))
    return np.array(temporal), np.array(pooled), np.array(observed)


def test_accumulator_matches_direct_computation():
    rng = np.random.default_rng(0)
    S, L, H, W = 24, 2, 3, 4
    # Large offset per cell (spatial contrast) plus small temporal noise: pooled >> temporal.
    x = 300.0 + 50.0 * rng.standard_normal((1, L, H, W)) + rng.standard_normal((S, L, H, W))
    mask = rng.random((S, L, H, W)) > 0.2
    mask[:, 0, 0, 0] = False  # one never-observed cell
    lat_w = np.array([0.5, 1.5, 1.0])

    acc = LossVarianceAccumulator()
    for chunk in np.array_split(np.arange(S), 4):  # (B, L, T, H, W) batches of 6
        xb = torch.tensor(x[chunk])[:, :, None]
        mb = torch.tensor(mask[chunk])[:, :, None]
        acc.update({VAR_T: xb}, {VAR_T: mb})
    out = acc.finalize(torch.tensor(lat_w))[VAR_T]

    temporal, pooled, observed = _reference(x, mask, lat_w)
    assert out["n_samples"] == S
    np.testing.assert_allclose(out["temporal_variance"], temporal, rtol=1e-9)
    np.testing.assert_allclose(out["pooled_variance"], pooled, rtol=1e-9)
    np.testing.assert_allclose(out["observed_fraction"], observed, rtol=1e-12)
    assert out["pooled_variance"][0] > 100 * out["temporal_variance"][0]


def test_accumulator_all_unobserved_level_is_nan():
    acc = LossVarianceAccumulator()
    x = torch.randn(4, 2, 1, 3, 3, dtype=torch.float64)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 1] = False
    acc.update({VAR_T: x}, {VAR_T: mask})
    out = acc.finalize()[VAR_T]
    assert np.isfinite(out["temporal_variance"][0]) and np.isnan(out["temporal_variance"][1])
    assert out["observed_fraction"] == pytest.approx([1.0, 0.0])


def test_observed_masks_nan_value_and_mask_var_rules():
    pre_sst = torch.tensor([[[[[float("nan"), 290.0], [0.0, 295.0]]]]])  # (1, 1, 1, 2, 2)
    landmask = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    t3d = torch.zeros(1, 2, 1, 2, 2)
    pre = {"input": {SRC: {VAR_MASK: landmask}}, "target": {SRC: {VAR_SST: pre_sst, VAR_T: t3d}}}
    # y_target_processed has no NaN left (filled after the scaler).
    target = {VAR_SST: torch.nan_to_num(pre_sst, nan=280.0), VAR_T: t3d}

    plain = observed_masks(target, pre)
    assert plain[VAR_SST].flatten().tolist() == [False, True, True, True]
    assert plain[VAR_T].all()

    by_value = observed_masks(target, pre, {VAR_SST: {"op": "!=", "value": 0}})
    assert by_value[VAR_SST].flatten().tolist() == [False, True, False, True]

    # A 0/1 mask variable, broadcast over levels; a prefix selects every prognostic variable.
    by_mask_var = observed_masks(target, pre, {f"{SRC}/prognostic": {"mask_var": VAR_MASK, "op": "<", "value": 0.5}})
    assert by_mask_var[VAR_SST].flatten().tolist() == [False, True, True, False]
    assert by_mask_var[VAR_T].shape == t3d.shape
    assert by_mask_var[VAR_T][0, :, 0].flatten().tolist() == [False, True, True, False] * 2


def test_observed_masks_missing_mask_var_raises():
    target = {VAR_SST: torch.zeros(1, 1, 1, 2, 2)}
    with pytest.raises(KeyError, match="mask_var"):
        observed_masks(target, {"target": {SRC: dict(target)}}, {VAR_SST: {"mask_var": VAR_MASK, "value": 0}})


def test_target_twin_conf_keeps_only_target_blocks():
    conf = {
        "postblocks": {
            "per_step": {
                "reconstruct": {"type": "reconstruct", "args": {"detach": False}},
                "scaler": {"type": "bridgescaler_transform", "args": {"method": "inverse_transform"}},
                "reconstruct_target": {"type": "reconstruct", "args": {"in_key": "y", "out_key": "y_target_processed"}},
                "scaler_target": {"type": "bridgescaler_transform", "args": {"key": "y_target_processed"}},
                "tracer_fixer": {"type": "tracer_fixer", "args": {"tracer_thres": 0.0}},
            }
        }
    }
    twin = target_twin_conf(conf)["postblocks"]["per_step"]
    assert list(twin) == ["reconstruct_target", "scaler_target"]
    assert list(conf["postblocks"]["per_step"]) == [
        "reconstruct",
        "scaler",
        "reconstruct_target",
        "scaler_target",
        "tracer_fixer",
    ]
    with pytest.raises(ValueError, match="y_target_processed"):
        target_twin_conf({"postblocks": {"per_step": {"reconstruct": {"type": "reconstruct"}}}})


def _write_file(tmp_path, variables):
    path = str(tmp_path / "loss_variances.json")
    save_loss_variances(path, variables, {"test": True})
    return path


def test_load_loss_variances_types_and_mask_correction(tmp_path):
    path = _write_file(
        tmp_path,
        {
            VAR_T: {"temporal_variance": [4.0, 9.0], "pooled_variance": [40.0, 90.0], "observed_fraction": [1.0, 0.5]},
        },
    )
    assert json.load(open(path))["meta"]["test"] is True
    assert load_loss_variances(path)[VAR_T].tolist() == [4.0, 4.5]
    assert load_loss_variances(path, mask_correction=False)[VAR_T].tolist() == [4.0, 9.0]
    assert load_loss_variances(path, "pooled")[VAR_T].tolist() == [40.0, 45.0]
    with pytest.raises(ValueError, match="variance_type"):
        load_loss_variances(path, "tendency")


def test_resolve_prefers_file_and_falls_back_to_scaler(tmp_path):
    from bridgescaler import save_scaler_dict
    from bridgescaler.distributed_tensor import DStandardScalerTensor

    scalers = {}
    for var_key, var in {VAR_T: [100.0, 25.0], VAR_SST: [7.0]}.items():
        s = DStandardScalerTensor(channels_last=False)
        s.mean_x_ = torch.zeros(len(var))
        s.var_x_ = torch.tensor(var)
        s.x_columns_ = list(range(len(var)))
        s.n_ = 10
        s._fit = True
        scalers[var_key] = s
    scaler_path = str(tmp_path / "scaler.json")
    save_scaler_dict({"target": {SRC: scalers}}, scaler_path)
    path = _write_file(
        tmp_path,
        {VAR_T: {"temporal_variance": [2.0, 3.0], "pooled_variance": [2.0, 3.0], "observed_fraction": [1.0, 1.0]}},
    )
    per_level = resolve_target_variances(scaler_path, variance_path=path, per_level=True)
    assert per_level[VAR_T].tolist() == [2.0, 3.0]
    assert per_level[VAR_SST].tolist() == [7.0]
    assert resolve_target_variances(scaler_path, variance_path=path)[VAR_T] == pytest.approx(2.5)
    assert resolve_target_variances(None, variance_path=path)[VAR_T] == pytest.approx(2.5)


def test_base_loss_uses_variance_file(tmp_path):
    path = _write_file(
        tmp_path,
        {
            VAR_T: {"temporal_variance": [1e-14, 1e-5], "pooled_variance": [1.0, 1.0], "observed_fraction": [1, 1]},
            VAR_SST: {"temporal_variance": [0.25], "pooled_variance": [100.0], "observed_fraction": [0.5]},
        },
    )
    loss_fn = BaseLoss(training_loss="mse", variance_path=path, normalize_weights=False)
    g = torch.Generator().manual_seed(0)
    pred = {VAR_T: torch.randn(2, 2, 1, 3, 4, generator=g), VAR_SST: torch.randn(2, 1, 1, 3, 4, generator=g)}
    target = {k: v + torch.randn(v.shape, generator=g) for k, v in pred.items()}
    with torch.no_grad():
        loss = loss_fn({"y_processed": {SRC: pred}, "y_target_processed": {SRC: target}})

    def level_mse(k):
        return ((pred[k] - target[k]) ** 2).mean(dim=(0, 2, 3, 4)).double().numpy()

    expected = np.mean([np.mean(level_mse(VAR_T) / np.array([1e-14, 1e-5])), level_mse(VAR_SST)[0] / (0.25 * 0.5)])
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_metric_uses_variance_file(tmp_path):
    from credit.metrics.base import BaseCombinedMetric

    path = _write_file(
        tmp_path,
        {VAR_SST: {"temporal_variance": [1e-15], "pooled_variance": [1e-15], "observed_fraction": [1.0]}},
    )
    metric = BaseCombinedMetric(metrics={"rmse": {}}, variance_path=path, normalize_weights=False)
    pred = {VAR_SST: torch.full((1, 1, 1, 2, 2), 2e-8)}
    target = {VAR_SST: torch.zeros(1, 1, 1, 2, 2)}
    out = metric({"y_processed": {SRC: pred}, "y_target_processed": {SRC: target}})
    # No absolute variance floor: RMSE 2e-8 / sigma 3.16e-8.
    assert out["rmse"] == pytest.approx(2e-8 / np.sqrt(1e-15), rel=1e-4)
