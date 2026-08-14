"""
common.py
---------
Built-in concrete univariate metrics for the Gen 2 metrics framework.

Each class subclasses :class:`credit.metrics.base.BaseVariableMetric` and
implements :meth:`compute_variable` (the elementwise error tensor) and, where
needed, :meth:`reduce` (finalization of the per-variable scalar after the
spatial mean). Latitude weighting is applied in
:meth:`BaseVariableMetric.forward` before the mean, so these only define the
error functional.

These metrics are registered in :data:`credit.metrics._METRIC_REGISTRY` under
the keys ``"rmse"``, ``"mse"``, ``"mae"``, ``"bias"``, ``"r2score"``, and
``"log_variance_ratio"`` and are therefore available directly from the config
``metrics`` section::

    metrics:
      type: combined
      args:
        metrics: {rmse: {}, mae: {}, bias: {}, r2score: {}, log_variance_ratio: {}}

Code example::

    from credit.metrics.common import RMSEMetric

    metric = RMSEMetric(metric_name="rmse", var_weighting="none")
    scores = metric(full_data_dict)   # {"rmse/ERA5/.../T": 1.2, "rmse": 0.9, ...}
"""

import torch

from credit.metrics.base import BaseVariableMetric

__all__ = [
    "BiasMetric",
    "LogVarianceRatioMetric",
    "MAEMetric",
    "MSEMetric",
    "R2ScoreMetric",
    "RMSEMetric",
]


class MSEMetric(BaseVariableMetric):
    """Mean squared error per variable (elementwise ``(pred - target) ** 2``)."""

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target) ** 2


class MAEMetric(BaseVariableMetric):
    """Mean absolute error per variable (elementwise ``abs(pred - target)``)."""

    scale_power = 1  # linear in sigma

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target)


class BiasMetric(BaseVariableMetric):
    """Signed mean bias per variable (elementwise ``pred - target``).

    The aggregate is the weighted mean of per-variable signed biases and may
    be negative.
    """

    scale_power = 1  # linear in sigma

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred - target


class RMSEMetric(BaseVariableMetric):
    """Root mean squared error per variable.

    ``compute_variable`` returns the elementwise squared error; ``reduce``
    takes the square root of the spatial mean to yield the per-variable RMSE.
    """

    scale_power = 1  # sqrt of a quadratic score -> linear in sigma

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target) ** 2

    def reduce(self, score: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(score)


class R2ScoreMetric(BaseVariableMetric):
    """Coefficient of determination (R²) per variable.

    R² = 1 - SS_res / SS_tot, where SS_res is the (latitude-weighted) residual
    sum of squares ``mean(w * (pred - target)²)`` and SS_tot is the
    (latitude-weighted) total sum of squares ``mean(w * (target - mean_w(target))²)``.

    - R² = 1: perfect forecast.
    - R² = 0: forecast is no better than predicting the target mean.
    - R² < 0: forecast is worse than the target-mean baseline.

    Unlike the simple elementwise metrics (MSE, MAE, ...), R² requires the
    latitude-weighted target mean, so it overrides :meth:`_score_variable`
    rather than :meth:`compute_variable`/:meth:`reduce`.
    """

    scale_power = 0  # already normalized by the target variance

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Not used — _score_variable is overridden instead. Provided so the
        # abstract method is satisfied and the class can be instantiated.
        return (pred - target) ** 2

    def _score_variable(self, pred: torch.Tensor, target: torch.Tensor, var_key: str | None = None) -> torch.Tensor:
        lat_w = self._lat_w(pred)
        w = lat_w if lat_w is not None else 1.0
        ss_res = (w * (pred - target) ** 2).mean()
        target_mean = (w * target).mean()
        ss_tot = (w * (target - target_mean) ** 2).mean()
        return 1.0 - ss_res / (ss_tot + 1e-12)


class LogVarianceRatioMetric(BaseVariableMetric):
    """Log10 ratio of predicted to target spatial variance per variable.

    ``score = log10(var_pred + eps) - log10(var_target + eps)``

    where ``var_pred`` and ``var_target`` are the area-weighted spatial
    variances (deviation from the area-weighted mean) of the forecast and
    target fields, respectively.

    - ``score = 0``: forecast and target have equal variance.
    - ``score < 0``: forecast is **smoother** than target (reduced variability;
      typical of ML emulators and ensemble averaging).
    - ``score > 0``: forecast is **sharper** than target (enhanced variability;
      can indicate noise amplification).

    The log10 transform makes the metric symmetric around 0: ``+1`` means the
    forecast variance is 10× the target, ``-1`` means 1/10×. This is
    preferable to a raw ratio for aggregation and interpretation.

    .. note::
        Variance is translation-invariant, so this metric is **insensitive to
        bias** — it isolates the smoothness/sharpness question from systematic
        error. This is a strength when comparing ML emulators (which may have
        different bias characteristics) but means the metric should be used
        alongside a bias-sensitive metric (e.g. RMSE, BiasMetric) for a
        complete picture.

    .. warning::
        Without a climatology reference, the variance includes both the
        "signal" (anomaly) and the "mean state" variability. When a
        climatology is available, :class:`ForecastActivityMetric` (SDAF) is
        more informative because it measures the anomaly variance. This
        metric is a reasonable proxy when no climatology is available.

    Args:
        eps: small constant added inside both log10 terms to avoid
            ``log10(0)``. Default ``1e-12`` is appropriate for physical-unit
            fields; increase it (e.g. ``1e-6``) for normalized data where
            variances are very small.

    Example (config)::

        metrics:
          type: combined
          args:
            metrics:
              rmse: {}
              log_variance_ratio:
                eps: 1.0e-12
            var_weighting: "none"
    """

    scale_power = 0  # a log ratio is dimensionless

    def __init__(self, *args, eps: float = 1e-12, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = float(eps)

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Not used — _score_variable is overridden instead.
        return (pred - target) ** 2

    def _score_variable(self, pred: torch.Tensor, target: torch.Tensor, var_key: str | None = None) -> torch.Tensor:
        lat_w = self._lat_w(pred)
        w = lat_w if lat_w is not None else 1.0
        pred_mean = (w * pred).mean()
        target_mean = (w * target).mean()
        var_pred = (w * (pred - pred_mean) ** 2).mean()
        var_target = (w * (target - target_mean) ** 2).mean()
        return torch.log10(var_pred + self.eps) - torch.log10(var_target + self.eps)
