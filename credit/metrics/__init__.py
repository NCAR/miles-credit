"""credit.metrics — verification metrics for CREDIT training and rollout.

This package hosts two metric generations:

* ``credit.metrics.gen_1.metrics`` — the legacy flat-tensor metrics
  (``LatWeightedMetrics`` and friends) used by the Gen 1 / LES / WRF trainers.
  They are re-exported here so the historical import
  ``from credit.metrics import LatWeightedMetrics`` keeps working after the
  ``credit/metrics.py`` module was folded into this package.
* ``credit.metrics.base`` — the Gen 2 per-variable metrics
  (``BaseVariableMetric`` / ``BaseCombinedMetric`` and the built-in
  ``RMSEMetric`` / ``MSEMetric`` / ``MAEMetric`` / ``BiasMetric``) that score
  the postblock ``full_data_dict`` in physical units, mirroring
  :class:`credit.losses.base.BaseLoss`.

Construction is config-driven via :func:`load_metric`, dispatched on
``conf["metrics"]["type"]`` (mirroring :func:`credit.losses.load_loss`).
"""

import importlib
import inspect
import logging

from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_METRIC_TYPES = ("rmse", "r2score", "bias")

# ---------------------------------------------------------------------------
# Config-driven dispatch: maps config-file keys -> class (used by load_metric).
# Registry entries are either:
#   (module_path: str, class_name: str)  — built-in lazy entries
#   cls: type                             — externally registered classes
_METRIC_REGISTRY = {
    "rmse": ("credit.metrics.common", "RMSEMetric"),
    "mse": ("credit.metrics.common", "MSEMetric"),
    "mae": ("credit.metrics.common", "MAEMetric"),
    "bias": ("credit.metrics.common", "BiasMetric"),
    "r2score": ("credit.metrics.common", "R2ScoreMetric"),
    "log_variance_ratio": ("credit.metrics.common", "LogVarianceRatioMetric"),
    "acc": ("credit.metrics.anomaly", "AnomalyCorrelationCoefficientMetric"),
    "activity": ("credit.metrics.anomaly", "ForecastActivityMetric"),
    "combined": ("credit.metrics.base", "BaseCombinedMetric"),
}

# Direct-import table for lazy module attribute access (mirrors credit.losses).
_CLASS_SOURCES = {
    "BaseVariableMetric": ("credit.metrics.base", "BaseVariableMetric"),
    "BaseCombinedMetric": ("credit.metrics.base", "BaseCombinedMetric"),
    "RMSEMetric": ("credit.metrics.common", "RMSEMetric"),
    "MSEMetric": ("credit.metrics.common", "MSEMetric"),
    "MAEMetric": ("credit.metrics.common", "MAEMetric"),
    "BiasMetric": ("credit.metrics.common", "BiasMetric"),
    "R2ScoreMetric": ("credit.metrics.common", "R2ScoreMetric"),
    "LogVarianceRatioMetric": ("credit.metrics.common", "LogVarianceRatioMetric"),
    "AnomalyCorrelationCoefficientMetric": ("credit.metrics.anomaly", "AnomalyCorrelationCoefficientMetric"),
    "ForecastActivityMetric": ("credit.metrics.anomaly", "ForecastActivityMetric"),
    "LatWeightedMetrics": ("credit.metrics.gen_1.metrics", "LatWeightedMetrics"),
    "LatWeightedMetricsClimatology": ("credit.metrics.gen_1.metrics", "LatWeightedMetricsClimatology"),
    "LatWeightedMetricsEnsemble": ("credit.metrics.gen_1.metrics", "LatWeightedMetricsEnsemble"),
}


# ---------------------------------------------------------------------------
# Module __getattr__: lazy attribute resolution so submodules are only imported
# on first access. Mirrors credit/losses/__init__.py.
def __getattr__(name):
    if name in _CLASS_SOURCES:
        module_path, class_name = _CLASS_SOURCES[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise AttributeError(f"Cannot import {name!r}: optional dependencies missing.") from exc
    raise AttributeError(f"module 'credit.metrics' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Registration
def register_metric(metric_type):
    """Decorator that adds an external metric class to the metric registry.

    The class must inherit from :class:`credit.metrics.base.BaseVariableMetric`
    (or otherwise accept the Gen 2 ``full_data_dict`` forward contract).

    Args:
        metric_type: Key used in the config ``metrics`` section.

    Example (Python decorator)::

        from credit.metrics import register_metric
        from credit.metrics.base import BaseVariableMetric

        @register_metric("my_metric")
        class MyMetric(BaseVariableMetric):
            ...

    Example (config ``custom_objects``)::

        custom_objects:
          MyMetric:
            object_type: metric
            module_path: mypackage.metrics

        metrics:
          type: combined
          args:
            metrics: {rmse: {}, MyMetric: {}}
    """

    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            raise TypeError(f"register_metric: '{cls.__name__}' must inherit from torch.nn.Module.")
        if metric_type in _METRIC_REGISTRY:
            logger.warning(f"register_metric: overwriting existing registry entry for '{metric_type}'")
        _METRIC_REGISTRY[metric_type] = cls
        return cls

    return decorator


def _load_metric_entry(metric_type):
    """Return the class for a registered metric type, importing lazily if needed.

    Raises:
        ValueError: If metric_type is not in _METRIC_REGISTRY.
        ImportError: If the metric's module cannot be imported.
    """
    if metric_type not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric type '{metric_type}'. "
            f"Available types: {sorted(_METRIC_REGISTRY)}. "
            "Register a custom metric with @register_metric."
        )
    entry = _METRIC_REGISTRY[metric_type]
    if isinstance(entry, tuple):
        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise ImportError(
                f"Metric type '{metric_type}' requires optional dependencies that are not installed. "
                f"Original error: {exc}"
            ) from exc
    return entry


# ---------------------------------------------------------------------------
# Metric construction & loading
def load_metric(conf, validation=False):
    """Load a metric (or combined metric) from the config.

    Dispatches on ``conf["metrics"]["type"]``, mirroring
    :func:`credit.losses.load_loss`. The Gen 2 ``combined`` type returns a
    :class:`credit.metrics.base.BaseCombinedMetric` holding one or more
    :class:`credit.metrics.base.BaseVariableMetric` subclasses; any other
    registered ``type`` returns a single metric instance. When the ``metrics``
    section is omitted, the default combined metric contains ``rmse``,
    ``r2score``, and ``bias`` with uniform variable weighting.

    Args:
        conf (dict): Configuration dictionary. An optional ``metrics``
            section uses the new-style ``{type, args}`` format; when omitted,
            ``rmse``, ``r2score``, and ``bias`` are loaded with uniform
            variable weighting::

                metrics:
                  type: combined
                  args:
                    metrics: {rmse: {}, mae: {}, bias: {}}
                    var_weighting: "inverse_variance"
                    scaler_path: "/path/scaler.json"
                    use_latitude_weights: true
                    latitude_weights: "/path/static.zarr"
        validation (bool, optional): Reserved for API symmetry with
            :func:`credit.losses.load_loss`. Currently unused by the metric
            classes (metrics are not optimized).

    Returns:
        torch.nn.Module: A metric instance callable as
        ``metrics(full_data_dict) -> dict[str, float]``.

    Raises:
        ValueError: If the requested metric type is not in ``_METRIC_REGISTRY``.
    """
    from credit.registry import load_custom_objects

    load_custom_objects(conf)

    metrics_conf = conf.get("metrics")
    if not metrics_conf:
        metrics_conf = {
            "type": "combined",
            "args": {
                "metrics": {metric_type: {} for metric_type in DEFAULT_METRIC_TYPES},
                "var_weighting": "none",
            },
        }
    metric_type = metrics_conf["type"]
    args = dict(metrics_conf.get("args") or {})

    mode = "validation" if validation else "train"
    logger.info(f"Loaded the {metric_type} metric ({mode}) with parameters: {args}")

    if metric_type == "combined":
        from credit.datasets.gen_2.channel_utils import ChannelSchema
        from credit.metrics.base import BaseCombinedMetric

        return BaseCombinedMetric(channel_schema=ChannelSchema.load_or_from_config(conf), **args)

    cls = _load_metric_entry(metric_type)
    args.setdefault("metric_name", metric_type)
    if "channel_schema" in inspect.signature(cls.__init__).parameters:
        from credit.datasets.gen_2.channel_utils import ChannelSchema

        args.setdefault("channel_schema", ChannelSchema.load_or_from_config(conf))
    return cls(**args)
