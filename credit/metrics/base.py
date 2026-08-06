"""
base.py
-------
Gen 2 per-variable verification metrics on the postblocks state dictionary.

The Gen 2 trainer's ``per_step`` postblock chain turns the raw model output into
``full_data_dict["y_processed"]`` — a nested ``{source: {var_key: tensor}}``
dict in **physical units** (Reconstruct + inverse bridgescaler, plus any
physics fixers). The metric classes in this module score each variable
against the matching entry of ``full_data_dict["y_target_processed"]`` (the
same chain applied to the flat target ``y``) and combine the per-variable
scores, mirroring :class:`credit.losses.base.BaseLoss` but for *evaluation*
rather than training: metrics run under ``torch.no_grad`` and return detached
Python floats for logging.

Required postblocks config (per_step phase) — identical to BaseLoss::

    postblocks:
      per_step:
        reconstruct:        {type: reconstruct, args: {detach: false}}
        scaler:             {type: bridgescaler_transform, args: {... method: inverse_transform}}
        reconstruct_target: {type: reconstruct, args: {in_key: "y", out_key: "y_target_processed"}}
        scaler_target:      {type: bridgescaler_transform, args: {... method: inverse_transform, key: "y_target_processed"}}

Metric config (``conf["metrics"]``, mirroring the ``loss`` / ``preblocks``
``{type, args}`` structure)::

    metrics:
      type: combined
      args:
        metrics:                         # registry names of univariate metrics
          rmse: {}
          mae: {}
          bias: {}
        var_weighting: "inverse_variance"  # inverse_variance | manual | none
        scaler_path: "/path/scaler.json"   # required for inverse_variance
        variable_weights: {}                # manual multipliers per var_key (all modes)
        normalize_weights: true            # rescale combination weights to mean 1
        include_computed_diagnostics: true  # score postblock-computed diagnostics
        use_latitude_weights: true          # cos(lat) spatial weighting per variable
        latitude_weights: "/path/static.zarr"

Scored variables: every variable in the data target layout (prognostic AND
diagnostic variables from ``conf["data"]["source"]``) is always scored.
Variables that only appear in ``y_processed`` because a postblock computed
them from prognostic variables (e.g. ``mslp_diagnostic``) are "computed
diagnostics": they are scored only when ``include_computed_diagnostics: true``,
and then require a matching entry in ``y_target_processed``.

``var_weighting`` reuses the BaseLoss modes **except** ``learnable`` (metrics
are not optimized):

  - ``inverse_variance`` (default): per-variable weight
    ``1 / sigma_v ** scale_power``, with ``sigma_v`` read from the fitted
    bridgescaler at ``scaler_path``. ``scale_power`` is a class attribute of
    each metric giving the power of sigma its score carries, so the weighted
    aggregate is dimensionless regardless of the metric's order: 2 for MSE,
    1 for RMSE / MAE / bias / activity, and 0 for already-normalized scores
    such as R2 and ACC (which ignore the scaler entirely). Weighting every
    metric by ``1 / sigma^2`` would over-weight low-variance variables
    quadratically for the linear metrics and is meaningless for the
    dimensionless ones.
  - ``manual``: weights come from ``variable_weights`` alone.
  - ``none``: uniform combination; only sensible when variables are already
    on comparable scales.

``variable_weights`` multipliers apply on top of every mode (default 1.0).

Code example — building a combined metric directly::

    from credit.metrics.base import BaseCombinedMetric
    from credit.datasets.gen_2.channel_utils import ChannelSchema

    schema = ChannelSchema.from_config(conf)
    metric = BaseCombinedMetric(
        channel_schema=schema,
        metrics={"rmse": {}, "mae": {}, "bias": {}},
        var_weighting="inverse_variance",
        scaler_path="/path/scaler.json",
        use_latitude_weights=True,
        latitude_weights="/path/static.zarr",
    )
    scores = metric(full_data_dict)   # {"rmse/ERA5/.../T": 1.2, "rmse": 0.9, ...}

Code example — registering a custom univariate metric::

    from credit.metrics import register_metric
    from credit.metrics.base import BaseVariableMetric

    @register_metric("huber_metric")
    class HuberMetric(BaseVariableMetric):
        def compute_variable(self, pred, target):
            return torch.nn.functional.huber_loss(pred, target, reduction="none")

And in config::

    metrics:
      type: combined
      args:
        metrics: {rmse: {}, huber_metric: {delta: 1.0}}
"""

import logging
from abc import ABC, abstractmethod

import torch
from torch import nn

from credit.losses.base import (
    BaseLoss,
    _cos_lat_weights,
    _load_target_variances,
)

logger = logging.getLogger(__name__)

METRIC_VAR_WEIGHTING_MODES = ("inverse_variance", "manual", "none")


class BaseVariableMetric(nn.Module, ABC):
    """Abstract base class for a univariate per-variable verification metric.

    ``forward`` takes the trainer's ``full_data_dict`` and scores
    ``full_data_dict["y_processed"]`` against
    ``full_data_dict["y_target_processed"]`` variable by variable, then combines
    the per-variable scores across variables (see the module docstring for the
    full config format and weighting modes). Unlike
    :class:`credit.losses.base.BaseLoss`, metrics run under ``torch.no_grad``
    and return detached Python floats for logging; there is no ``learnable``
    weighting mode and no gradient/detach check.

    Subclasses implement :meth:`compute_variable` (elementwise error tensor)
    and optionally override :meth:`reduce` (default identity; e.g. RMSE takes
    ``sqrt``). Latitude weighting is applied to the elementwise tensor before
    the spatial mean, exactly as in BaseLoss.

    Args:
        metric_name: name used to namespace the returned scores
            (``"{metric_name}/{var_key}"`` and ``"{metric_name}"`` aggregate).
        var_weighting: combination weighting — ``"inverse_variance"`` (default),
            ``"manual"``, or ``"none"``. ``"learnable"`` is rejected.
        scaler_path: bridgescaler dict JSON used to read per-variable variances;
            required for ``inverse_variance``.
        variable_weights: manual multipliers per var_key (all modes, default 1.0).
        normalize_weights: rescale static combination weights to mean 1.
        include_computed_diagnostics: score postblock-computed diagnostics
            (variables present in ``y_processed`` but not in the data target
            layout). Data diagnostics are always scored.
        use_latitude_weights: apply cos(lat) spatial weighting per variable.
        latitude_weights: path to a dataset with a ``latitude`` coordinate
            (required when ``use_latitude_weights`` is True).
        channel_schema: optional :class:`ChannelSchema` fixing the data target
            variable layout; when None the scored variables are discovered from
            the state dict on the first forward pass.

    Attributes:
        last_var_scores: ``{var_key: float}`` detached per-variable scores
            (pre-combination) from the most recent forward pass.
        var_keys: scored variable list, resolved on the first forward pass.
        scale_power: class attribute; see below.
    """

    #: Power of the variable's standard deviation carried by this metric's
    #: per-variable score, used by ``var_weighting="inverse_variance"`` to make
    #: the combined aggregate dimensionless. The weight is
    #: ``1 / sigma ** scale_power``, i.e. ``variance ** (-scale_power / 2)``.
    #:
    #:   * ``2`` — quadratic in sigma (MSE). ``1 / sigma^2`` recovers the
    #:     normalized-space score exactly, matching ``BaseLoss``.
    #:   * ``1`` — linear in sigma (RMSE, MAE, bias, forecast activity).
    #:   * ``0`` — already dimensionless (R2, ACC, log variance ratio); the
    #:     scaler variance is not consulted at all.
    #:
    #: Subclasses that are not quadratic MUST override this, otherwise
    #: ``inverse_variance`` over-weights low-variance variables.
    scale_power: int = 2

    def __init__(
        self,
        metric_name: str,
        var_weighting: str = "inverse_variance",
        scaler_path: str | None = None,
        variable_weights: dict | None = None,
        normalize_weights: bool = True,
        include_computed_diagnostics: bool = True,
        use_latitude_weights: bool = False,
        latitude_weights: str | None = None,
        channel_schema=None,
        **kwargs,
    ):
        super().__init__()
        self.metric_name = metric_name

        if var_weighting == "learnable":
            raise ValueError(
                "BaseVariableMetric: var_weighting='learnable' is not supported for metrics "
                "(metrics are not optimized). Use 'inverse_variance', 'manual', or 'none'."
            )
        if var_weighting not in METRIC_VAR_WEIGHTING_MODES:
            raise ValueError(f"var_weighting must be one of {METRIC_VAR_WEIGHTING_MODES}; got {var_weighting!r}")
        self.var_weighting = var_weighting
        self.manual_weights = {k: float(v) for k, v in (variable_weights or {}).items()}
        self.normalize_weights = bool(normalize_weights)
        self.include_computed_diagnostics = bool(include_computed_diagnostics)

        self.lat_weights = None
        self._lat_w_key = None
        self._lat_w_cached = None
        if use_latitude_weights:
            if not latitude_weights:
                raise ValueError("latitude_weights (path) is required when use_latitude_weights=True.")
            self.lat_weights = _cos_lat_weights(latitude_weights)  # (H,)

        # Data target variables (prognostic + diagnostic from the data config):
        # always scored. Variables appearing in y_processed but not listed here
        # are postblock-computed diagnostics, handled per include_computed_diagnostics.
        self.data_var_keys = None
        if channel_schema is not None:
            self.data_var_keys = [entry["var_key"] for entry in channel_schema.target_layout]

        self._variances = None
        # A dimensionless metric (scale_power == 0) never consults the scaler,
        # so requiring scaler_path for it would be a pointless config burden.
        if self.var_weighting == "inverse_variance" and self.scale_power != 0:
            if not scaler_path:
                raise ValueError(
                    f"scaler_path is required for var_weighting='{self.var_weighting}' "
                    f"with metric '{metric_name}' (scale_power={self.scale_power})."
                )
            self._variances = _load_target_variances(scaler_path)

        self._combination_weights = None  # {var_key: float}; built at first forward
        self.var_keys = None  # full scoring list, resolved at first forward
        # Populated by forward(); initialized here so the documented attribute
        # exists (and logging code reading it is safe) before the first pass.
        self.last_var_scores = {}

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the elementwise (un-reduced) error tensor for one variable.

        The tensor must broadcast against ``pred`` (typically the same shape);
        :meth:`forward` applies latitude weighting and takes the spatial mean.
        For example, MSE returns ``(pred - target) ** 2``.
        """

    def reduce(self, score: torch.Tensor) -> torch.Tensor:
        """Finalize the per-variable scalar after the spatial mean.

        Default is identity; subclasses override (e.g. RMSE returns
        ``torch.sqrt``).
        """
        return score

    # ------------------------------------------------------------------
    # Setup helpers (mirror BaseLoss, minus the learnable path)
    # ------------------------------------------------------------------

    def _resolve_var_keys(self, pred: dict, target: dict) -> list:
        if self.data_var_keys is not None:
            var_keys = list(self.data_var_keys)
        else:
            var_keys = sorted(target.keys())
        extras = sorted(k for k in pred if k not in set(var_keys))
        if extras and self.include_computed_diagnostics:
            var_keys += extras
        elif extras:
            logger.info(
                "BaseVariableMetric: skipping computed diagnostics %s (include_computed_diagnostics=False).", extras
            )
        return var_keys

    def _build_static_weights(self, var_keys) -> dict:
        weights = {}
        for var_key in var_keys:
            manual = self.manual_weights.get(var_key, 1.0)
            if self.var_weighting == "inverse_variance":
                if self.scale_power == 0:
                    # Already dimensionless (R2, ACC, ...) — dividing by a
                    # variance would distort the aggregate, not normalize it.
                    weights[var_key] = manual
                    continue
                variance = self._variances.get(var_key)
                if variance is None:
                    logger.warning(
                        "BaseVariableMetric: no scaler variance found for '%s'; falling back to weight 1.0. "
                        "Add it to the scaler at scaler_path or set a manual entry in variable_weights.",
                        var_key,
                    )
                    weights[var_key] = manual
                else:
                    # 1 / sigma**scale_power, so the weighted score is dimensionless
                    # for any metric order (MSE 2, RMSE/MAE/bias 1).
                    weights[var_key] = manual / max(variance, 1e-12) ** (self.scale_power / 2)
            elif self.var_weighting == "manual":
                if var_key not in self.manual_weights:
                    logger.warning("BaseVariableMetric: no variable_weights entry for '%s'; using 1.0.", var_key)
                weights[var_key] = manual
            else:  # "none"
                weights[var_key] = manual

        if self.normalize_weights:
            mean_w = sum(weights.values()) / len(weights)
            if mean_w > 0:
                weights = {k: v / mean_w for k, v in weights.items()}
        return weights

    def _lat_w(self, target: torch.Tensor) -> "torch.Tensor | None":
        """Sharded, device-resident latitude weights shaped (1, 1, 1, H, 1)."""
        if self.lat_weights is None:
            return None
        from credit.parallel.domain import shard_lat_weights

        key = (target.shape[-2], target.device)
        if self._lat_w_key != key:
            w = self.lat_weights.view(1, 1, 1, -1, 1)
            self._lat_w_cached = shard_lat_weights(w, target.shape[-2]).to(device=target.device)
            self._lat_w_key = key
        return self._lat_w_cached

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _score_variable(self, pred: torch.Tensor, target: torch.Tensor, var_key: str | None = None) -> torch.Tensor:
        """Compute the per-variable scalar score from pred and target tensors.

        Default implementation calls :meth:`compute_variable` for the
        elementwise error, applies latitude weighting, takes the spatial mean,
        and finalizes via :meth:`reduce`. Subclasses that need access to the
        full target tensor (e.g. R², which requires the target mean) or to the
        variable key (e.g. anomaly metrics that look up a per-variable
        climatology) override this method instead of
        ``compute_variable``/``reduce``.

        Args:
            pred: forecast tensor for one variable, shape (B, C, T, H, W).
            target: validating "truth" tensor, same shape as ``pred``.
            var_key: the variable's registry key (e.g.
                ``"ERA5/prognostic/3d/temperature"``); ``None`` when the
                caller does not supply it (backward compatibility).
        """
        elementwise = self.compute_variable(pred, target)
        if elementwise.shape != pred.shape:
            raise ValueError(
                f"BaseVariableMetric: compute_variable returned shape "
                f"{tuple(elementwise.shape)}, expected elementwise {tuple(pred.shape)}."
            )
        lat_w = self._lat_w(pred)
        if lat_w is not None:
            elementwise = elementwise * lat_w
        return self.reduce(elementwise.mean())

    def forward(self, full_data_dict: dict) -> dict:
        if "y_processed" not in full_data_dict:
            raise KeyError(
                "BaseVariableMetric requires full_data_dict['y_processed'] — add a 'reconstruct' postblock "
                "to postblocks.per_step."
            )
        if "y_target_processed" not in full_data_dict:
            raise KeyError(
                "BaseVariableMetric requires full_data_dict['y_target_processed'] — add to postblocks.per_step: "
                "reconstruct_target: {type: reconstruct, args: {in_key: 'y', out_key: 'y_target_processed'}} "
                "followed by a bridgescaler_transformer postblock with key: 'y_target_processed'."
            )

        pred = BaseLoss._flatten_state(full_data_dict["y_processed"], "y_processed")
        target = BaseLoss._flatten_state(full_data_dict["y_target_processed"], "y_target_processed")

        if self.var_keys is None:
            self.var_keys = self._resolve_var_keys(pred, target)
            self._combination_weights = self._build_static_weights(self.var_keys)
        var_keys = self.var_keys
        weights = self._combination_weights

        var_scores = {}
        self.last_var_scores = {}
        with torch.no_grad():
            for var_key in var_keys:
                if var_key not in pred:
                    raise KeyError(
                        f"BaseVariableMetric: variable '{var_key}' missing from y_processed — "
                        "channel schema and postblocks output disagree."
                    )
                if var_key not in target:
                    raise KeyError(
                        f"BaseVariableMetric: variable '{var_key}' missing from y_target_processed. If this is a "
                        "postblock-computed diagnostic, apply the same compute postblock to the target twin "
                        "(key: 'y_target_processed') or set include_computed_diagnostics: false."
                    )
                # Physical-unit magnitudes (e.g. SP ~1e5 Pa) overflow fp16 — score in fp32.
                p = pred[var_key].float()
                t = target[var_key].float().to(p.device)
                var_score = self._score_variable(p, t, var_key=var_key)
                var_scores[var_key] = var_score
                self.last_var_scores[var_key] = var_score.item()

            aggregate = torch.stack([weights[var_key] * var_scores[var_key] for var_key in var_keys]).mean().item()

        out = {f"{self.metric_name}/{var_key}": self.last_var_scores[var_key] for var_key in var_keys}
        out[f"{self.metric_name}"] = aggregate
        return out


class BaseCombinedMetric(nn.Module):
    """Container metric combining several :class:`BaseVariableMetric` subclasses.

    Holds one :class:`BaseVariableMetric` per registry name listed in
    ``metrics`` and shares the cross-metric weighting configuration (variable
    weighting, latitude weighting, computed-diagnostics policy, channel schema)
    across all of them. ``forward`` returns the union of each child's scores,
    namespaced by each child's ``metric_name`` (e.g. ``"rmse/ERA5/.../T"``,
    ``"rmse"``, ``"mae/ERA5/.../T"``, ``"mae"``).

    Args:
        metrics: ``{metric_name: per_metric_args}`` mapping. Each name must be
            in the metric registry (``credit.metrics._METRIC_REGISTRY``).
        var_weighting: combination weighting passed to every child —
            ``"inverse_variance"`` (default), ``"manual"``, or ``"none"``.
        scaler_path: bridgescaler dict JSON; required for ``inverse_variance``.
        variable_weights: manual multipliers per var_key (all modes, default 1.0).
        normalize_weights: rescale static combination weights to mean 1.
        include_computed_diagnostics: score postblock-computed diagnostics.
        use_latitude_weights: apply cos(lat) spatial weighting per variable.
        latitude_weights: path to a dataset with a ``latitude`` coordinate.
        channel_schema: optional :class:`ChannelSchema` fixing the data target
            variable layout.

    Attributes:
        metric_modules: ``{metric_name: BaseVariableMetric}`` child instances.

    Example (config)::

        metrics:
          type: combined
          args:
            metrics: {rmse: {}, mae: {}, bias: {}}
            var_weighting: "inverse_variance"
            scaler_path: "/path/scaler.json"
            use_latitude_weights: true
            latitude_weights: "/path/static.zarr"

    Example (code)::

        metric = BaseCombinedMetric(
            channel_schema=schema,
            metrics={"rmse": {}, "mae": {}},
            var_weighting="none",
        )
        scores = metric(full_data_dict)
    """

    def __init__(
        self,
        metrics: dict,
        var_weighting: str = "inverse_variance",
        scaler_path: str | None = None,
        variable_weights: dict | None = None,
        normalize_weights: bool = True,
        include_computed_diagnostics: bool = True,
        use_latitude_weights: bool = False,
        latitude_weights: str | None = None,
        channel_schema=None,
        **kwargs,
    ):
        super().__init__()
        if not metrics:
            raise ValueError("BaseCombinedMetric: 'metrics' must list at least one metric.")

        # Shared kwargs forwarded into every child BaseVariableMetric.
        shared = {
            "var_weighting": var_weighting,
            "scaler_path": scaler_path,
            "variable_weights": variable_weights,
            "normalize_weights": normalize_weights,
            "include_computed_diagnostics": include_computed_diagnostics,
            "use_latitude_weights": use_latitude_weights,
            "latitude_weights": latitude_weights,
            "channel_schema": channel_schema,
        }

        # Local import to avoid a circular import at module load
        # (credit.metrics imports credit.metrics.base lazily).
        from credit.metrics import _load_metric_entry

        self.metric_modules = nn.ModuleDict()
        for metric_name, per_metric_args in metrics.items():
            cls = _load_metric_entry(metric_name)
            if not (isinstance(cls, type) and issubclass(cls, BaseVariableMetric)):
                raise TypeError(
                    f"BaseCombinedMetric: metric '{metric_name}' maps to {cls!r} which is not a "
                    "BaseVariableMetric subclass."
                )
            args = dict(shared)
            args.update(per_metric_args or {})
            self.metric_modules[metric_name] = cls(metric_name=metric_name, **args)

    def forward(self, full_data_dict: dict) -> dict:
        out = {}
        for module in self.metric_modules.values():
            out.update(module(full_data_dict))
        return out


# ---------------------------------------------------------------------------
# Built-in concrete univariate metrics live in credit.metrics.common, and are
# registered in credit.metrics._METRIC_REGISTRY under the keys "rmse", "mse",
# "mae", and "bias". See credit/metrics/common.py for the implementations.
