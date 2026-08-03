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
the keys ``"rmse"``, ``"mse"``, ``"mae"``, ``"bias"``, and ``"r2score"`` and
are therefore available directly from the config ``metrics`` section::

    metrics:
      type: combined
      args:
        metrics: {rmse: {}, mae: {}, bias: {}, r2score: {}}

Code example::

    from credit.metrics.common import RMSEMetric

    metric = RMSEMetric(metric_name="rmse", var_weighting="none")
    scores = metric(full_data_dict)   # {"rmse/ERA5/.../T": 1.2, "rmse": 0.9, ...}
"""

import torch

from credit.metrics.base import BaseVariableMetric

__all__ = ["BiasMetric", "MAEMetric", "MSEMetric", "R2ScoreMetric", "RMSEMetric"]


class MSEMetric(BaseVariableMetric):
    """Mean squared error per variable (elementwise ``(pred - target) ** 2``)."""

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target) ** 2


class MAEMetric(BaseVariableMetric):
    """Mean absolute error per variable (elementwise ``abs(pred - target)``)."""

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target)


class BiasMetric(BaseVariableMetric):
    """Signed mean bias per variable (elementwise ``pred - target``).

    The aggregate is the weighted mean of per-variable signed biases and may
    be negative.
    """

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred - target


class RMSEMetric(BaseVariableMetric):
    """Root mean squared error per variable.

    ``compute_variable`` returns the elementwise squared error; ``reduce``
    takes the square root of the spatial mean to yield the per-variable RMSE.
    """

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

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Not used — _score_variable is overridden instead. Provided so the
        # abstract method is satisfied and the class can be instantiated.
        return (pred - target) ** 2

    def _score_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lat_w = self._lat_w(pred)
        w = lat_w if lat_w is not None else 1.0
        ss_res = (w * (pred - target) ** 2).mean()
        target_mean = (w * target).mean()
        ss_tot = (w * (target - target_mean) ** 2).mean()
        return 1.0 - ss_res / (ss_tot + 1e-12)
