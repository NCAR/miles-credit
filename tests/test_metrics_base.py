"""Tests for credit.metrics.base (BaseVariableMetric, BaseCombinedMetric, built-ins)
and the credit.metrics registry / load_metric factory."""

import numpy as np
import pytest
import torch
from credit.datasets.gen_2.channel_utils import ChannelSchema
from credit.metrics import (
    BaseCombinedMetric,
    BaseVariableMetric,
    BiasMetric,
    LatWeightedMetrics,
    LatWeightedMetricsClimatology,
    LatWeightedMetricsEnsemble,
    LogVarianceRatioMetric,
    MAEMetric,
    MSEMetric,
    R2ScoreMetric,
    RMSEMetric,
    load_metric,
)
from credit.metrics.anomaly import (
    AnomalyCorrelationCoefficientMetric,
    ForecastActivityMetric,
)

# ---------------------------------------------------------------------------
# Fixtures (adapted from tests/test_losses_base.py)
# ---------------------------------------------------------------------------

VAR_T = "ERA5/prognostic/3d/T"
VAR_SP = "ERA5/prognostic/2d/SP"
VAR_PRECIP = "ERA5/diagnostic/2d/precip"
VAR_MSLP = "ERA5/diagnostic/2d/MSLP_computed"  # postblock-computed diagnostic (not in data layout)
N_LEVELS = 3

# Per-variable physical scales (sigma^2) written into the synthetic scaler file.
VARIANCES = {VAR_T: [100.0, 25.0, 4.0], VAR_SP: [1.0e6], VAR_PRECIP: [1.0e-8]}


def _make_scaler_file(tmp_path):
    from bridgescaler import save_scaler_dict
    from bridgescaler.distributed_tensor import DStandardScalerTensor

    scalers = {}
    for var_key, variances in VARIANCES.items():
        n = len(variances)
        s = DStandardScalerTensor(channels_last=False)
        s.mean_x_ = torch.zeros(n)
        s.var_x_ = torch.tensor(variances, dtype=torch.float32)
        s.x_columns_ = list(range(n))
        s.n_ = 100
        s._fit = True
        scalers[var_key] = s
    path = str(tmp_path / "scaler.json")
    save_scaler_dict({"target": {"ERA5": scalers}}, path)
    return path


def _make_schema():
    """ChannelSchema for T(3D) + SP(2D) prognostic and precip(2D) diagnostic."""
    return ChannelSchema.from_config(
        {
            "data": {
                "source": {
                    "ERA5": {
                        "levels": list(range(N_LEVELS)),
                        "variables": {
                            "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},
                            "diagnostic": {"vars_2D": ["precip"]},
                        },
                    }
                }
            },
            "model": {"levels": N_LEVELS},
        }
    )


def _make_conf(scaler_path, tmp_path, metric_args=None):
    """Full config with the new-style {type, args} metrics section."""
    args = {
        "metrics": {"rmse": {}, "mae": {}, "bias": {}},
        "var_weighting": "inverse_variance",
        "scaler_path": scaler_path,
    }
    if metric_args:
        args.update(metric_args)
    return {
        "save_loc": str(tmp_path),
        "data": {
            "source": {
                "ERA5": {
                    "levels": list(range(N_LEVELS)),
                    "variables": {
                        "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},
                        "diagnostic": {"vars_2D": ["precip"]},
                    },
                }
            }
        },
        "model": {"levels": N_LEVELS},
        "metrics": {"type": "combined", "args": args},
    }


def _make_metric(metric_cls, scaler_path, metric_name=None, **kwargs):
    """A BaseVariableMetric subclass wired to the synthetic channel schema."""
    kwargs.setdefault("var_weighting", "inverse_variance")
    if kwargs["var_weighting"] == "inverse_variance":
        kwargs.setdefault("scaler_path", scaler_path)
    kwargs.setdefault("channel_schema", _make_schema())
    return metric_cls(metric_name=metric_name or metric_cls.__name__.lower().replace("metric", ""), **kwargs)


def _make_state_dict(batch=2, height=4, width=5, seed=0, computed=False):
    g = torch.Generator().manual_seed(seed)
    pred = {
        VAR_T: torch.randn(batch, N_LEVELS, 1, height, width, generator=g),
        VAR_SP: torch.randn(batch, 1, 1, height, width, generator=g) * 1000.0,
        VAR_PRECIP: torch.rand(batch, 1, 1, height, width, generator=g) * 1e-4,
    }
    if computed:
        pred[VAR_MSLP] = torch.randn(batch, 1, 1, height, width, generator=g) * 100.0
    target = {k: v + 0.1 * torch.randn(v.shape, generator=g) for k, v in pred.items() if k != VAR_MSLP}
    return {
        "y_processed": {"ERA5": pred},
        "y_target_processed": {"ERA5": target},
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


def test_base_variable_metric_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BaseVariableMetric(metric_name="x")


# ---------------------------------------------------------------------------
# Per-variable math
# ---------------------------------------------------------------------------


def test_rmse_per_variable(tmp_path):
    metric = _make_metric(RMSEMetric, _make_scaler_file(tmp_path), var_weighting="none")
    state = _make_state_dict()
    out = metric(state)
    assert metric.var_keys == [VAR_T, VAR_SP, VAR_PRECIP]
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        expected = torch.sqrt(torch.mean((p - t) ** 2)).item()
        assert out[f"rmse/{var_key}"] == pytest.approx(expected, rel=1e-5)


def test_mse_per_variable(tmp_path):
    metric = _make_metric(MSEMetric, _make_scaler_file(tmp_path), var_weighting="none")
    state = _make_state_dict()
    out = metric(state)
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        expected = torch.mean((p - t) ** 2).item()
        assert out[f"mse/{var_key}"] == pytest.approx(expected, rel=1e-5)


def test_mae_per_variable(tmp_path):
    metric = _make_metric(MAEMetric, _make_scaler_file(tmp_path), var_weighting="none")
    state = _make_state_dict()
    out = metric(state)
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        expected = torch.mean(torch.abs(p - t)).item()
        assert out[f"mae/{var_key}"] == pytest.approx(expected, rel=1e-5)


def test_bias_per_variable(tmp_path):
    metric = _make_metric(BiasMetric, _make_scaler_file(tmp_path), var_weighting="none")
    state = _make_state_dict()
    out = metric(state)
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        expected = torch.mean(p - t).item()
        assert out[f"bias/{var_key}"] == pytest.approx(expected, rel=1e-5)


def test_r2score_per_variable(tmp_path):
    metric = _make_metric(R2ScoreMetric, _make_scaler_file(tmp_path), var_weighting="none")
    state = _make_state_dict()
    out = metric(state)
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        ss_res = torch.mean((p - t) ** 2).item()
        target_mean = torch.mean(t).item()
        ss_tot = torch.mean((t - target_mean) ** 2).item()
        expected = 1.0 - ss_res / (ss_tot + 1e-12)
        assert out[f"r2score/{var_key}"] == pytest.approx(expected, abs=1e-5)


def test_r2score_perfect_forecast(tmp_path):
    """R² = 1 when pred == target."""
    metric = _make_metric(R2ScoreMetric, _make_scaler_file(tmp_path), var_weighting="none")
    g = torch.Generator().manual_seed(42)
    pred = {
        VAR_T: torch.randn(2, N_LEVELS, 1, 4, 5, generator=g),
        VAR_SP: torch.randn(2, 1, 1, 4, 5, generator=g) * 1000.0,
        VAR_PRECIP: torch.rand(2, 1, 1, 4, 5, generator=g) * 1e-4,
    }
    state = {"y_processed": {"ERA5": pred}, "y_target_processed": {"ERA5": {k: v.clone() for k, v in pred.items()}}}
    out = metric(state)
    for var_key in metric.var_keys:
        assert out[f"r2score/{var_key}"] == pytest.approx(1.0, abs=1e-4)


def test_r2score_climatology_forecast(tmp_path):
    """R² ≈ 0 when pred == target mean (no skill beyond climatology)."""
    metric = _make_metric(R2ScoreMetric, _make_scaler_file(tmp_path), var_weighting="none")
    g = torch.Generator().manual_seed(42)
    target = {
        VAR_T: torch.randn(2, N_LEVELS, 1, 4, 5, generator=g),
        VAR_SP: torch.randn(2, 1, 1, 4, 5, generator=g) * 1000.0,
        VAR_PRECIP: torch.rand(2, 1, 1, 4, 5, generator=g) * 1e-4,
    }
    # pred = broadcast target mean (constant field)
    pred = {}
    for var_key, t in target.items():
        pred[var_key] = t.mean().expand_as(t).contiguous()
    state = {"y_processed": {"ERA5": pred}, "y_target_processed": {"ERA5": target}}
    out = metric(state)
    for var_key in metric.var_keys:
        # R² is near 0; small deviations from the epsilon in the denominator
        # are largest for low-variance variables (e.g. precip ~1e-8).
        assert out[f"r2score/{var_key}"] == pytest.approx(0.0, abs=1e-2)


def test_log_variance_ratio_per_variable(tmp_path):
    """Log variance ratio matches manual log10(var_pred) - log10(var_target)."""
    metric = _make_metric(
        LogVarianceRatioMetric, _make_scaler_file(tmp_path), metric_name="log_variance_ratio", var_weighting="none"
    )
    state = _make_state_dict()
    out = metric(state)
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        var_pred = torch.mean((p - p.mean()) ** 2)
        var_target = torch.mean((t - t.mean()) ** 2)
        expected = torch.log10(var_pred + 1e-12) - torch.log10(var_target + 1e-12)
        assert out[f"log_variance_ratio/{var_key}"] == pytest.approx(expected.item(), rel=1e-4), var_key


def test_log_variance_ratio_equal_variance(tmp_path):
    """Log variance ratio = 0 when pred == target."""
    metric = _make_metric(
        LogVarianceRatioMetric, _make_scaler_file(tmp_path), metric_name="log_variance_ratio", var_weighting="none"
    )
    g = torch.Generator().manual_seed(42)
    pred = {
        VAR_T: torch.randn(2, N_LEVELS, 1, 4, 5, generator=g),
        VAR_SP: torch.randn(2, 1, 1, 4, 5, generator=g) * 1000.0,
        VAR_PRECIP: torch.rand(2, 1, 1, 4, 5, generator=g) * 1e-4,
    }
    state = {"y_processed": {"ERA5": pred}, "y_target_processed": {"ERA5": {k: v.clone() for k, v in pred.items()}}}
    out = metric(state)
    for var_key in metric.var_keys:
        assert out[f"log_variance_ratio/{var_key}"] == pytest.approx(0.0, abs=1e-4), var_key


def test_log_variance_ratio_smooth_forecast_negative(tmp_path):
    """A damped forecast (lower variance) produces a negative log variance ratio."""
    metric = _make_metric(
        LogVarianceRatioMetric, _make_scaler_file(tmp_path), metric_name="log_variance_ratio", var_weighting="none"
    )
    state = _make_state_dict()
    # Damped forecast: pred = 0.5 * original_pred (halves the variance → log10(0.25) ≈ -0.60)
    pred_damped = {k: 0.5 * v for k, v in state["y_processed"]["ERA5"].items()}
    state_damped = {"y_processed": {"ERA5": pred_damped}, "y_target_processed": state["y_target_processed"]}
    out = metric(state_damped)
    for var_key in metric.var_keys:
        assert out[f"log_variance_ratio/{var_key}"] < 0.0, var_key


def test_log_variance_ratio_sharp_forecast_positive(tmp_path):
    """An amplified forecast (higher variance) produces a positive log variance ratio."""
    metric = _make_metric(
        LogVarianceRatioMetric, _make_scaler_file(tmp_path), metric_name="log_variance_ratio", var_weighting="none"
    )
    # Use a clean state where pred and target have the same variance baseline.
    g = torch.Generator().manual_seed(42)
    base = {
        VAR_T: torch.randn(2, N_LEVELS, 1, 4, 5, generator=g),
        VAR_SP: torch.randn(2, 1, 1, 4, 5, generator=g) * 1000.0,
        VAR_PRECIP: torch.randn(2, 1, 1, 4, 5, generator=g) * 1e-4,
    }
    # Amplified forecast: pred = 2.0 * base (4x the variance → log10(4) ≈ +0.60)
    pred_sharp = {k: 2.0 * v for k, v in base.items()}
    state = {"y_processed": {"ERA5": pred_sharp}, "y_target_processed": {"ERA5": base}}
    out = metric(state)
    for var_key in metric.var_keys:
        assert out[f"log_variance_ratio/{var_key}"] > 0.0, var_key


def test_log_variance_ratio_custom_eps(tmp_path):
    """Custom eps is used inside the log10 terms."""
    metric = _make_metric(
        LogVarianceRatioMetric,
        _make_scaler_file(tmp_path),
        metric_name="log_variance_ratio",
        var_weighting="none",
        eps=1.0,
    )
    state = _make_state_dict()
    out = metric(state)
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        var_pred = torch.mean((p - p.mean()) ** 2)
        var_target = torch.mean((t - t.mean()) ** 2)
        expected = torch.log10(var_pred + 1.0) - torch.log10(var_target + 1.0)
        assert out[f"log_variance_ratio/{var_key}"] == pytest.approx(expected.item(), rel=1e-4), var_key


def test_log_variance_ratio_bias_invariant(tmp_path):
    """Log variance ratio is unaffected by a constant bias (translation-invariant).

    Variance is mathematically translation-invariant; in float32 the invariance
    is approximate for extreme bias magnitudes relative to the field scale, so
    we use a moderate bias and a loose tolerance.
    """
    metric = _make_metric(
        LogVarianceRatioMetric, _make_scaler_file(tmp_path), metric_name="log_variance_ratio", var_weighting="none"
    )
    state = _make_state_dict()
    out_no_bias = metric(state)
    # Add a moderate constant bias (comparable to the field scale, not 1e6
    # which would destroy float32 precision for small-magnitude variables).
    state_biased = {
        "y_processed": {"ERA5": {k: v + 10.0 for k, v in state["y_processed"]["ERA5"].items()}},
        "y_target_processed": state["y_target_processed"],
    }
    out_biased = metric(state_biased)
    for var_key in metric.var_keys:
        assert out_biased[f"log_variance_ratio/{var_key}"] == pytest.approx(
            out_no_bias[f"log_variance_ratio/{var_key}"], abs=1e-2
        ), var_key


def test_load_metric_log_variance_ratio(tmp_path):
    """load_metric dispatches 'log_variance_ratio' type."""
    conf = _make_conf(_make_scaler_file(tmp_path), tmp_path)
    conf["metrics"] = {"type": "log_variance_ratio", "args": {"var_weighting": "none"}}
    metric = load_metric(conf)
    assert isinstance(metric, LogVarianceRatioMetric)


# ---------------------------------------------------------------------------
# Combination / weighting
# ---------------------------------------------------------------------------


def test_forward_combination_none_mode(tmp_path):
    metric = _make_metric(MSEMetric, _make_scaler_file(tmp_path), var_weighting="none")
    state = _make_state_dict()
    out = metric(state)

    expected_per_var = {}
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        expected_per_var[var_key] = torch.mean((p - t) ** 2).item()
    expected = np.mean(list(expected_per_var.values()))
    assert out["mse"] == pytest.approx(expected, rel=1e-5)
    assert metric.last_var_scores == pytest.approx(expected_per_var, rel=1e-5)


def test_manual_weights(tmp_path):
    manual = {VAR_T: 2.0, VAR_SP: 1.0, VAR_PRECIP: 0.5}
    metric = _make_metric(
        MSEMetric,
        _make_scaler_file(tmp_path),
        var_weighting="manual",
        variable_weights=manual,
        normalize_weights=False,
    )
    state = _make_state_dict()
    out = metric(state)
    expected = np.mean([manual[k] * metric.last_var_scores[k] for k in manual])
    assert out["mse"] == pytest.approx(expected, rel=1e-5)


def test_forward_inverse_variance_weights(tmp_path):
    metric = _make_metric(MSEMetric, _make_scaler_file(tmp_path))
    state = _make_state_dict()
    out = metric(state)

    raw = {
        VAR_T: 1.0 / np.mean(VARIANCES[VAR_T]),
        VAR_SP: 1.0 / 1.0e6,
        VAR_PRECIP: 1.0 / 1.0e-8,
    }
    mean_w = np.mean(list(raw.values()))
    expected = np.mean([raw[k] / mean_w * metric.last_var_scores[k] for k in raw])
    assert out["mse"] == pytest.approx(expected, rel=1e-5)


def test_learnable_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="learnable"):
        _make_metric(MSEMetric, _make_scaler_file(tmp_path), var_weighting="learnable")


def test_inverse_variance_requires_scaler_path(tmp_path):
    with pytest.raises(ValueError, match="scaler_path"):
        MSEMetric(metric_name="mse", var_weighting="inverse_variance", channel_schema=_make_schema())


def test_bad_weighting_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="var_weighting"):
        _make_metric(MSEMetric, _make_scaler_file(tmp_path), var_weighting="bogus")


# ---------------------------------------------------------------------------
# Latitude weighting
# ---------------------------------------------------------------------------


def test_latitude_weights_applied(tmp_path):
    metric = _make_metric(MSEMetric, _make_scaler_file(tmp_path), var_weighting="none")
    height = 4
    metric.lat_weights = torch.arange(1, height + 1, dtype=torch.float32)

    pred = {k: torch.ones(1, n, 1, height, 5) for k, n in ((VAR_T, N_LEVELS), (VAR_SP, 1), (VAR_PRECIP, 1))}
    target = {k: torch.zeros_like(v) for k, v in pred.items()}
    state = {"y_processed": {"ERA5": pred}, "y_target_processed": {"ERA5": target}}
    out = metric(state)
    # every elementwise entry is 1; weighted mean over H with weights 1..4 = 2.5
    assert out[f"mse/{VAR_SP}"] == pytest.approx(np.mean([1, 2, 3, 4]), rel=1e-5)


# ---------------------------------------------------------------------------
# Computed diagnostics
# ---------------------------------------------------------------------------


def test_computed_diagnostics_included(tmp_path):
    metric = _make_metric(
        MSEMetric, _make_scaler_file(tmp_path), var_weighting="none", include_computed_diagnostics=True
    )
    state = _make_state_dict(computed=True)
    state["y_target_processed"]["ERA5"][VAR_MSLP] = torch.zeros_like(state["y_processed"]["ERA5"][VAR_MSLP])
    out = metric(state)
    assert VAR_MSLP in metric.last_var_scores
    assert metric.var_keys == [VAR_T, VAR_SP, VAR_PRECIP, VAR_MSLP]
    assert f"mse/{VAR_MSLP}" in out


def test_computed_diagnostics_skipped(tmp_path):
    metric = _make_metric(
        MSEMetric, _make_scaler_file(tmp_path), var_weighting="none", include_computed_diagnostics=False
    )
    state = _make_state_dict(computed=True)
    out = metric(state)
    assert VAR_MSLP not in metric.last_var_scores
    assert metric.var_keys == [VAR_T, VAR_SP, VAR_PRECIP]
    assert f"mse/{VAR_MSLP}" not in out


def test_computed_diagnostics_missing_target_raises(tmp_path):
    metric = _make_metric(
        MSEMetric, _make_scaler_file(tmp_path), var_weighting="none", include_computed_diagnostics=True
    )
    state = _make_state_dict(computed=True)  # VAR_MSLP only in pred
    with pytest.raises(KeyError, match="y_target_processed"):
        metric(state)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_target_dict_raises(tmp_path):
    metric = _make_metric(MSEMetric, _make_scaler_file(tmp_path))
    state = _make_state_dict()
    del state["y_target_processed"]
    with pytest.raises(KeyError, match="y_target_processed"):
        metric(state)


def test_missing_y_processed_raises(tmp_path):
    metric = _make_metric(MSEMetric, _make_scaler_file(tmp_path))
    state = _make_state_dict()
    del state["y_processed"]
    with pytest.raises(KeyError, match="y_processed"):
        metric(state)


def test_metrics_run_under_no_grad(tmp_path):
    metric = _make_metric(MSEMetric, _make_scaler_file(tmp_path), var_weighting="none")
    state = _make_state_dict()
    # pred tensors carry no grad; metrics must not require grad on the output
    pred_t = state["y_processed"]["ERA5"][VAR_T].requires_grad_(True)
    state["y_processed"]["ERA5"][VAR_T] = pred_t
    out = metric(state)
    assert isinstance(out["mse"], float)
    # metric must not have built a graph
    assert all(isinstance(v, float) for v in out.values())


# ---------------------------------------------------------------------------
# BaseCombinedMetric
# ---------------------------------------------------------------------------


def test_combined_metric_returns_union(tmp_path):
    scaler_path = _make_scaler_file(tmp_path)
    schema = _make_schema()
    metric = BaseCombinedMetric(
        channel_schema=schema,
        metrics={"rmse": {}, "mae": {}},
        var_weighting="none",
    )
    state = _make_state_dict()
    out = metric(state)

    expected_keys = set()
    for name in ("rmse", "mae"):
        for var_key in (VAR_T, VAR_SP, VAR_PRECIP):
            expected_keys.add(f"{name}/{var_key}")
        expected_keys.add(name)
    assert set(out.keys()) == expected_keys

    # aggregates match each child run separately
    rmse = RMSEMetric(metric_name="rmse", var_weighting="none", channel_schema=schema)
    mae = MAEMetric(metric_name="mae", var_weighting="none", channel_schema=schema)
    rmse_out = rmse(state)
    mae_out = mae(state)
    assert out["rmse"] == pytest.approx(rmse_out["rmse"], rel=1e-5)
    assert out["mae"] == pytest.approx(mae_out["mae"], rel=1e-5)
    assert out[f"rmse/{VAR_T}"] == pytest.approx(rmse_out[f"rmse/{VAR_T}"], rel=1e-5)


def test_combined_metric_empty_rejected(tmp_path):
    with pytest.raises(ValueError, match="metrics"):
        BaseCombinedMetric(metrics={}, channel_schema=_make_schema())


def test_combined_metric_unknown_metric_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown metric type"):
        BaseCombinedMetric(metrics={"bogus": {}}, channel_schema=_make_schema())


def test_combined_metric_inverse_variance(tmp_path):
    scaler_path = _make_scaler_file(tmp_path)
    metric = BaseCombinedMetric(
        channel_schema=_make_schema(),
        metrics={"rmse": {}},
        var_weighting="inverse_variance",
        scaler_path=scaler_path,
    )
    state = _make_state_dict()
    out = metric(state)
    assert "rmse" in out
    assert f"rmse/{VAR_T}" in out


# ---------------------------------------------------------------------------
# load_metric factory
# ---------------------------------------------------------------------------


def test_load_metric_returns_combined(tmp_path):
    conf = _make_conf(_make_scaler_file(tmp_path), tmp_path)
    metric = load_metric(conf)
    assert isinstance(metric, BaseCombinedMetric)
    assert set(metric.metric_modules) == {"rmse", "mae", "bias"}


def test_load_metric_defaults_to_training_metrics(tmp_path):
    conf = _make_conf(_make_scaler_file(tmp_path), tmp_path)
    del conf["metrics"]
    metric = load_metric(conf)
    assert isinstance(metric, BaseCombinedMetric)
    assert set(metric.metric_modules) == {"rmse", "r2score", "bias"}


def test_load_metric_unknown_type_raises(tmp_path):
    conf = _make_conf(_make_scaler_file(tmp_path), tmp_path)
    conf["metrics"]["type"] = "bogus"
    with pytest.raises(ValueError, match="Unknown metric type"):
        load_metric(conf)


def test_load_metric_single_type(tmp_path):
    conf = _make_conf(_make_scaler_file(tmp_path), tmp_path)
    conf["metrics"] = {"type": "rmse", "args": {"var_weighting": "none"}}
    metric = load_metric(conf)
    assert isinstance(metric, RMSEMetric)


# ---------------------------------------------------------------------------
# Anomaly metrics (ACC, forecast activity)
# ---------------------------------------------------------------------------


def _make_anomaly_metric(metric_cls, scaler_path=None, metric_name=None, **kwargs):
    """An anomaly metric wired to the synthetic channel schema, no lat weights."""
    kwargs.setdefault("var_weighting", "none")
    kwargs.setdefault("channel_schema", _make_schema())
    if metric_name is None:
        metric_name = metric_cls.__name__.lower().replace("metric", "")
    return metric_cls(metric_name=metric_name, **kwargs)


def _make_climatology_dict():
    """Per-variable climatology fields matching _make_state_dict shapes."""
    g = torch.Generator().manual_seed(99)
    return {
        VAR_T: torch.randn(1, N_LEVELS, 1, 4, 5, generator=g) * 0.5,
        VAR_SP: torch.randn(1, 1, 1, 4, 5, generator=g) * 500.0,
        VAR_PRECIP: torch.rand(1, 1, 1, 4, 5, generator=g) * 1e-5,
    }


def test_acc_perfect_forecast(tmp_path):
    """ACC = 1 when pred == target (same anomaly pattern)."""
    clim = _make_climatology_dict()
    metric = _make_anomaly_metric(AnomalyCorrelationCoefficientMetric, metric_name="acc", climatology=clim)
    g = torch.Generator().manual_seed(7)
    pred = {
        VAR_T: torch.randn(2, N_LEVELS, 1, 4, 5, generator=g),
        VAR_SP: torch.randn(2, 1, 1, 4, 5, generator=g) * 1000.0,
        VAR_PRECIP: torch.rand(2, 1, 1, 4, 5, generator=g) * 1e-4,
    }
    state = {"y_processed": {"ERA5": pred}, "y_target_processed": {"ERA5": {k: v.clone() for k, v in pred.items()}}}
    out = metric(state)
    for var_key in metric.var_keys:
        assert out[f"acc/{var_key}"] == pytest.approx(1.0, abs=2e-2), var_key


def test_acc_matches_manual_computation(tmp_path):
    """ACC matches the manual dot-product / (||d_f|| * ||d_t||) formula."""
    clim = _make_climatology_dict()
    metric = _make_anomaly_metric(AnomalyCorrelationCoefficientMetric, metric_name="acc", climatology=clim)
    state = _make_state_dict()
    out = metric(state)
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        c = clim[var_key].to(p.device, p.dtype).expand_as(p)
        d_f = (p - c) - (p - c).mean()
        d_t = (t - c) - (t - c).mean()
        expected = (d_f * d_t).mean() / (torch.sqrt((d_f**2).mean() + 1e-12) * torch.sqrt((d_t**2).mean() + 1e-12))
        assert out[f"acc/{var_key}"] == pytest.approx(expected.item(), rel=1e-4), var_key


def test_acc_climatology_from_validation_data(tmp_path):
    """ACC works when climatology is accumulated from validation batches."""
    metric = _make_anomaly_metric(AnomalyCorrelationCoefficientMetric, metric_name="acc")
    # First batch — establishes running mean climatology.
    state1 = _make_state_dict(seed=0)
    out1 = metric(state1)
    assert "acc" in out1
    # Second batch — climatology updated, ACC should still be finite.
    state2 = _make_state_dict(seed=1)
    out2 = metric(state2)
    assert "acc" in out2
    for var_key in metric.var_keys:
        assert np.isfinite(out2[f"acc/{var_key}"]), var_key


def test_activity_matches_manual_computation(tmp_path):
    """Forecast activity (SDAF) matches sqrt(mean(d_f^2))."""
    clim = _make_climatology_dict()
    metric = _make_anomaly_metric(ForecastActivityMetric, metric_name="activity", climatology=clim)
    state = _make_state_dict()
    out = metric(state)
    for var_key in metric.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        c = clim[var_key].to(p.device, p.dtype).expand_as(p)
        d_f = (p - c) - (p - c).mean()
        expected = torch.sqrt((d_f**2).mean() + 1e-12)
        assert out[f"activity/{var_key}"] == pytest.approx(expected.item(), rel=1e-4), var_key


def test_activity_reduced_for_smooth_forecast(tmp_path):
    """A damped forecast has lower activity than the full forecast."""
    clim = _make_climatology_dict()
    metric = _make_anomaly_metric(ForecastActivityMetric, metric_name="activity", climatology=clim)
    state = _make_state_dict()
    # Damped forecast: pred = 0.5 * original_pred + 0.5 * clim (closer to climatology)
    pred_damped = {}
    for var_key, p in state["y_processed"]["ERA5"].items():
        c = clim[var_key].to(p.device, p.dtype).expand_as(p)
        pred_damped[var_key] = 0.5 * p + 0.5 * c
    state_damped = {"y_processed": {"ERA5": pred_damped}, "y_target_processed": state["y_target_processed"]}
    out_full = metric(state)
    out_damped = metric(state_damped)
    for var_key in metric.var_keys:
        assert out_damped[f"activity/{var_key}"] < out_full[f"activity/{var_key}"], var_key


def test_anomaly_metrics_in_combined(tmp_path):
    """ACC and activity work inside a BaseCombinedMetric."""
    clim = _make_climatology_dict()
    metric = BaseCombinedMetric(
        channel_schema=_make_schema(),
        metrics={"acc": {"climatology": clim}, "activity": {"climatology": clim}},
        var_weighting="none",
    )
    state = _make_state_dict()
    out = metric(state)
    assert "acc" in out
    assert "activity" in out
    for var_key in (VAR_T, VAR_SP, VAR_PRECIP):
        assert f"acc/{var_key}" in out
        assert f"activity/{var_key}" in out


def test_load_metric_acc(tmp_path):
    """load_metric dispatches 'acc' type to AnomalyCorrelationCoefficientMetric."""
    conf = _make_conf(_make_scaler_file(tmp_path), tmp_path)
    conf["metrics"] = {"type": "acc", "args": {"var_weighting": "none"}}
    metric = load_metric(conf)
    assert isinstance(metric, AnomalyCorrelationCoefficientMetric)


def test_load_metric_activity(tmp_path):
    """load_metric dispatches 'activity' type to ForecastActivityMetric."""
    conf = _make_conf(_make_scaler_file(tmp_path), tmp_path)
    conf["metrics"] = {"type": "activity", "args": {"var_weighting": "none"}}
    metric = load_metric(conf)
    assert isinstance(metric, ForecastActivityMetric)


# ---------------------------------------------------------------------------
# Backward-compat re-exports
# ---------------------------------------------------------------------------


def test_gen1_metrics_reexport():
    assert LatWeightedMetrics is not None
    assert LatWeightedMetricsClimatology is not None
    assert LatWeightedMetricsEnsemble is not None
