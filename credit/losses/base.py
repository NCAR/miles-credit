"""
base.py
-------
BaseLoss: per-variable loss computed on postblocks
output rather than on the flat, normalized ``y_pred`` tensor.

The Gen 2 trainer's ``per_step`` postblock chain turns the raw model output
into ``full_data_dict["y_processed"]`` — a nested ``{source: {var_key:
tensor}}`` dict in **physical units** (Reconstruct + inverse bridgescaler,
plus any physics fixers). BaseLoss scores each variable against the matching
entry of ``full_data_dict["y_target_processed"]`` (the same chain applied to
the flat target ``y``) and combines the per-variable scores.

Required postblocks config (per_step phase)::

    postblocks:
      per_step:
        reconstruct:        {type: reconstruct, args: {detach: false}}
        scaler:             {type: bridgescaler_transform, args: {... method: inverse_transform}}
        reconstruct_target: {type: reconstruct, args: {in_key: "y", out_key: "y_target_processed"}}
        scaler_target:      {type: bridgescaler_transform, args: {... method: inverse_transform, key: "y_target_processed"}}

``detach: false`` on the prediction Reconstruct is REQUIRED for training —
otherwise ``y_processed`` carries no gradient and BaseLoss raises.

Loss config (``conf["loss"]``, mirroring the preblocks/postblocks structure)::

    loss:
      type: base
      args:
        training_loss: "mse"             # univariate loss from the credit loss registry
        base_loss_parameters: {}         # kwargs for the base loss (reduction forced to "none")
        validation_loss: "mae"           # optional different univariate loss for validation
        validation_loss_parameters: {}
        base_loss_overrides:             # optional per-variable univariate loss
          ERA5/diagnostic/2d/total_precipitation: {loss: "mae", parameters: {}}
        var_weighting: "inverse_variance"  # inverse_variance | manual | learnable | none
        scaler_path: "/path/scaler.json"   # bridgescaler dict; required for inverse_variance/learnable
        variable_weights: {}             # manual multipliers per var_key (all modes)
        normalize_weights: true          # rescale combination weights to mean 1
        per_level_variance: true         # inverse_variance: one variance per level (channel)
        include_computed_diagnostics: true
        use_latitude_weights: false      # cos(lat) spatial weighting per variable
        latitude_weights: "/path/static.zarr"
        pred_key: "y_processed"          # prediction to score; y_processed_raw with a snapshot postblock
        valid_pred_key: null             # validation's prediction key (default: pred_key)

Scoring before the budget fixers: a global fixer (mass, water, energy) that
rescales or shifts a field makes a loss on its output blind to that field's
raw global mean or scale, so nothing pulls the raw prediction back toward the
truth in that direction (Chapman, Schreck & Sha 2026, arXiv:2607.18416). A
``snapshot`` postblock placed before the fixers copies the prediction to
``y_processed_raw``; ``pred_key: y_processed_raw`` trains on that copy while
the fixed ``y_processed`` still rolls forward. ``valid_pred_key: y_processed``
validates on the delivered prediction; the trainer then also logs the raw
validation loss as ``valid_loss_raw``.

Scored variables: every variable in the data target layout (prognostic AND
diagnostic variables from ``conf["data"]["source"]``) is always scored.
Variables that only appear in ``y_processed`` because a postblock computed
them from prognostic variables (e.g. ``mslp_diagnostic``,
``geopotential_diagnostic``) are "computed diagnostics": they are scored only
when ``include_computed_diagnostics: true``, and then require a matching
entry in ``y_target_processed`` (i.e. the same compute postblock applied to
the target twin with ``key: y_target_processed``).

Physical-unit variables have wildly different variances (SP ~1e5 Pa vs
specific humidity ~1e-3), so an unweighted physical-space MSE is dominated by
the largest-scale variables. ``var_weighting`` handles this:

  - ``inverse_variance`` (default): weight ``1 / sigma_{v,l}^2`` for each level
    ``l`` of variable ``v``, with the variances read from the fitted bridgescaler
    at ``scaler_path`` (DStandardScalerTensor ``var_x_`` directly;
    DQuantileScalerTensor via a t-digest centroid moment estimate; numpy
    DeepStandardScaler via ``sd_``). The scaler normalizes each level
    separately, so per-level weights make an MSE equal to the normalized-space
    MSE (up to how the variable's levels are averaged). Spatial scalers
    (``DSpatialStandardScalerTensor``) contribute each level's variance averaged
    over its observed grid points. ``per_level_variance: false`` restores the old
    single level-mean variance per variable, which leaves low-variance levels
    (e.g. upper-air specific humidity) almost unweighted.
  - ``manual``: weights come from ``variable_weights`` alone.
  - ``learnable``: Kendall-Gal uncertainty weighting — per-variable
    ``log sigma_v^2`` is a learnable parameter (initialized from the scaler
    stats when available), ``L = mean_v(m_v * exp(-s_v) * L_v + 0.5 * s_v)``.
    The training application must add the criterion's parameters to the
    optimizer (``train_gen2`` does this automatically); checkpointing of these
    parameters is currently not wired — they re-initialize on resume
    (tracked in issue #473).
    Learnable mode requires a channel schema and does not support computed
    diagnostics.
  - ``none``: uniform combination; only sensible when variables are already
    on comparable scales.

``variable_weights`` multipliers apply on top of every mode (default 1.0).
"""

import inspect
import logging
import math
import os

import numpy as np
import torch
from torch import nn

from credit.losses import _load_loss_entry, is_crps_loss

logger = logging.getLogger(__name__)

VAR_WEIGHTING_MODES = ("inverse_variance", "manual", "learnable", "none")


def _scaler_channel_variance(scaler) -> "torch.Tensor | None":
    """Per-channel variance from a fitted bridgescaler object, or None.

    Supports DStandardScalerTensor (``var_x_`` directly), DQuantileScalerTensor
    (second moment estimated from the t-digest centroids — a slight
    underestimate due to centroid clustering, adequate for weighting), and the
    numpy DeepStandardScaler (``sd_``).
    """
    var = getattr(scaler, "var_x_", None)
    if var is not None:
        return torch.as_tensor(var, dtype=torch.float32).flatten()

    means_list = getattr(scaler, "centroids_mean_tensor", None)
    weights_list = getattr(scaler, "centroids_weight_tensor", None)
    if means_list is not None and weights_list is not None:
        variances = []
        for means, weights in zip(means_list, weights_list):
            m = torch.as_tensor(means, dtype=torch.float32)
            w = torch.as_tensor(weights, dtype=torch.float32)
            total = w.sum()
            if total <= 0:
                variances.append(torch.tensor(1.0))
                continue
            mu = (w * m).sum() / total
            variances.append((w * (m - mu) ** 2).sum() / total)
        return torch.stack(variances).flatten()

    sd = getattr(scaler, "sd_", None)
    if sd is not None:
        return torch.as_tensor(np.asarray(sd), dtype=torch.float32).flatten() ** 2

    return None


def exp_transform_specs(conf: dict) -> list:
    """``exp_transform`` postblocks that write physical values into ``y_processed``.

    Returns ``[{"variables": [...], "base": str, "eps": float}, ...]`` from
    ``conf["postblocks"]["per_step"]``. A variable listed here is scored in
    physical units, but its scaler statistics were fit on the log-transformed
    values (the matching ``log_transform`` preblock), so its variance has to be
    converted before it can weight the loss (see ``_load_target_variances``).
    """
    per_step = (conf.get("postblocks") or {}).get("per_step") or {}
    specs = []
    for block in per_step.values():
        if not isinstance(block, dict) or block.get("type") != "exp_transform":
            continue
        args = block.get("args") or {}
        if args.get("key", "y_processed") != "y_processed":
            continue
        specs.append(
            {
                "variables": list(args.get("variables") or []),
                "base": str(args.get("base", "e")),
                "eps": float(args.get("eps", 1e-8)),
            }
        )
    return specs


def _matching_exp_spec(var_key: str, specs: list) -> "dict | None":
    """First spec selecting ``var_key``: exact, a "/"-delimited prefix, or ``[]`` (all)."""
    for spec in specs:
        variables = spec["variables"]
        if not variables or any(var_key == v or var_key.startswith(v + "/") for v in variables):
            return spec
    return None


def _physical_variance_from_log(mean: torch.Tensor, var: torch.Tensor, base: str, eps: float) -> torch.Tensor:
    """Per-channel variance of ``x`` given the mean/variance of ``y = log_b(x + eps) - log_b(eps)``.

    First-order (delta-method) estimate about the log-space mean: with ``m`` and
    ``s2`` the natural-log mean and variance, ``Var(x) ~= exp(2m) * s2``. The exact
    lognormal moment ``(exp(s2) - 1) * exp(2m + s2)`` is not used: fields such as
    specific humidity are strongly skewed in log space (log variance > 2 near the
    surface), where the lognormal formula overestimates the physical variance by an
    order of magnitude. The two agree when ``s2`` is small (e.g. surface pressure).
    This is a fallback for scalers fit before ``credit preprocess`` recorded exact
    physical-unit statistics (``_load_physical_scalers``).
    """
    ln_base = {"e": 1.0, "2": math.log(2.0), "10": math.log(10.0)}[base]
    log_b_eps = math.log(eps) / ln_base
    m = (mean.double() + log_b_eps) * ln_base
    s2 = var.double() * ln_base**2
    return torch.exp(2 * m) * s2


def _load_physical_scalers(scaler_path: str) -> dict:
    """``{var_key: scaler}`` from the physical-unit stats sidecar of ``scaler_path``, or ``{}``."""
    from bridgescaler import load_scaler_dict

    from credit.preblock.scaler import physical_stats_path

    path = physical_stats_path(scaler_path)
    if not os.path.isfile(path):
        return {}
    return {
        var_key: scaler
        for source_scalers in load_scaler_dict(path).get("target", {}).values()
        for var_key, scaler in source_scalers.items()
    }


def _channel_grid_stat(scaler, name: str) -> "torch.Tensor | None":
    """Scaler statistic ``name`` (``var_x_`` / ``mean_x_``) as ``(C, K)``: channels by grid points.

    ``K`` is 1 for a per-channel scaler and the flattened grid for
    ``DSpatialStandardScalerTensor``. Never-observed entries (``n_ == 0``) are NaN.
    """
    stat = getattr(scaler, name, None)
    if stat is None:
        return None
    stat = torch.as_tensor(stat, dtype=torch.float64)
    stat = stat.reshape(stat.shape[0], -1) if stat.ndim else stat.reshape(1, 1)
    n = getattr(scaler, "n_", None)
    if n is not None and not isinstance(n, (int, float)):
        n = torch.as_tensor(n)
        if n.numel() == stat.numel():
            stat = torch.where(n.reshape(stat.shape) > 0, stat, torch.nan)
    return stat


def _physical_channel_variance(scaler) -> "torch.Tensor | None":
    """``(C, K)`` variance from a physical-units sidecar scaler, or None if it is unusable."""
    var = _channel_grid_stat(scaler, "var_x_")
    if var is None:
        flat = _scaler_channel_variance(scaler)
        var = None if flat is None else flat.double().reshape(-1, 1)
    if var is None or torch.isinf(var).any() or torch.isnan(var).all():
        return None
    return var


def _load_target_variances(scaler_path: str, exp_transforms: "list | None" = None, per_level: bool = False) -> dict:
    """Flatten a bridgescaler dict's ``"target"`` slice to ``{var_key: variance}``.

    ``exp_transforms`` (from ``exp_transform_specs``) marks variables whose scaler
    was fit in log space but which are scored in physical units; their variances
    are converted to physical units so inverse-variance weights match the loss.
    The exact physical variance from the ``*_physical.json`` sidecar that
    ``credit preprocess`` writes (``credit.preblock.scaler.physical_stats_path``)
    is used when present; otherwise it is estimated from the log-space statistics
    (``_physical_variance_from_log``).

    With ``per_level=False`` each variance is one float, averaged over all levels
    (and grid points, for a spatial scaler). With ``per_level=True`` it is a
    ``(C,)`` tensor with one variance per channel (level), averaged over that
    channel's observed grid points only.
    """
    from bridgescaler import load_scaler_dict

    from credit.preblock.scaler import physical_stats_path

    exp_transforms = exp_transforms or []
    target_scalers = load_scaler_dict(os.path.expandvars(scaler_path))["target"]
    physical_scalers = _load_physical_scalers(scaler_path) if exp_transforms else {}
    variances = {}
    for source_scalers in target_scalers.values():
        for var_key, scaler in source_scalers.items():
            grid_var = _channel_grid_stat(scaler, "var_x_")
            if grid_var is None:
                # Quantile / numpy scalers: one statistic per channel already.
                flat = _scaler_channel_variance(scaler)
                grid_var = None if flat is None else flat.double().reshape(-1, 1)
            if grid_var is None or torch.isinf(grid_var).any() or torch.isnan(grid_var).all():
                continue
            channel_var = grid_var
            spec = _matching_exp_spec(var_key, exp_transforms)
            exact = _physical_channel_variance(physical_scalers[var_key]) if var_key in physical_scalers else None
            if spec is not None and exact is not None and exact.shape == channel_var.shape:
                logger.info(
                    "'%s' is scored in physical units but scaled in log space: using the exact "
                    "physical variance %.4g from %s.",
                    var_key,
                    float(exact.nanmean()),
                    physical_stats_path(scaler_path),
                )
                channel_var = exact
            elif spec is not None:
                mean = _channel_grid_stat(scaler, "mean_x_")
                if mean is None:
                    logger.warning(
                        "'%s' is exp-transformed back to physical units, but its scaler has no "
                        "mean_x_ to convert the log-space variance; using the log-space variance. "
                        "Set a variable_weights entry for it.",
                        var_key,
                    )
                else:
                    phys = _physical_variance_from_log(mean, channel_var, spec["base"], spec["eps"])
                    if torch.isinf(phys).any():
                        logger.warning(
                            "'%s': log-to-physical variance conversion overflowed; using the "
                            "log-space variance. Set a variable_weights entry for it.",
                            var_key,
                        )
                    else:
                        logger.info(
                            "'%s' is scored in physical units but scaled in log space: level-mean "
                            "variance %.4g (log) -> %.4g (physical).",
                            var_key,
                            float(channel_var.nanmean()),
                            float(phys.nanmean()),
                        )
                        channel_var = phys
            per_channel = channel_var.nanmean(dim=1)
            if not torch.isfinite(per_channel).all():
                continue
            variances[var_key] = per_channel.float() if per_level else float(channel_var.nanmean())
    return variances


def _cos_lat_weights(path: str) -> torch.Tensor:
    """Cos-latitude weights (H,) normalized to mean 1, from a grid dataset."""
    import xarray as xr

    ds = xr.open_dataset(os.path.expandvars(path))
    lat = torch.tensor(ds["latitude"].values, dtype=torch.float32)
    weights = torch.cos(torch.deg2rad(lat))
    return weights / weights.mean()


def _instantiate_univariate_loss(name: str, params: dict) -> nn.Module:
    """Instantiate a registry loss as an elementwise univariate loss.

    CRPS-family losses are rejected, with one exception: classes that declare
    ``supports_elementwise = True`` (e.g. ``ring-crps``, whose ensemble
    dimension lives across data-parallel ranks rather than in the tensor, so
    its local score is elementwise like any other univariate loss).
    """
    cls = _load_loss_entry(name)
    if is_crps_loss(name) and not getattr(cls, "supports_elementwise", False):
        raise ValueError(
            f"BaseLoss training_loss '{name}' is an ensemble CRPS loss and cannot be used "
            "as a univariate per-variable loss. Choose an elementwise loss (mse, mae, huber, ...) "
            "or ring-crps (ensemble across data-parallel ranks)."
        )
    params = dict(params)
    if "reduction" in inspect.signature(cls.__init__).parameters:
        params["reduction"] = "none"
    return cls(**params)


class BaseLoss(nn.Module):
    """Per-variable univariate loss on the Gen 2 postblocks state dictionary.

    ``forward`` takes the trainer's ``full_data_dict`` and scores
    ``full_data_dict["y_processed"]`` against
    ``full_data_dict["y_target_processed"]`` variable by variable, then
    combines the scores across variables (see module docstring for the full
    config format and weighting modes).

    Args:
        training_loss: name of the univariate loss in the credit loss registry
            (``"mse"``, ``"mae"``, ``"huber"``, ``"logcosh"``, ...).
        base_loss_parameters: kwargs for the univariate loss (``reduction`` is
            forced to ``"none"``).
        validation_loss: optional different univariate loss name for validation.
        validation_loss_parameters: kwargs for the validation univariate loss.
        base_loss_overrides: ``{var_key: {"loss": name, "parameters": {...}}}``
            to use a different univariate loss for specific variables.
        var_weighting: combination weighting — ``"inverse_variance"`` (default),
            ``"manual"``, ``"learnable"``, or ``"none"``.
        scaler_path: bridgescaler dict JSON used to read per-variable variances;
            required for ``inverse_variance`` and ``learnable``.
        variable_weights: manual multipliers per var_key (all modes, default 1.0).
        normalize_weights: rescale static combination weights to mean 1.
        per_level_variance: with ``inverse_variance``, weight each level of a
            variable by its own scaler variance (default). False uses one
            level-mean variance per variable (the pre-2026-10 behavior).
        include_computed_diagnostics: score postblock-computed diagnostics
            (variables present in ``y_processed`` but not in the data target
            layout, e.g. from ``mslp_diagnostic``). Data diagnostics are always
            scored.
        use_latitude_weights: apply cos(lat) spatial weighting per variable.
        latitude_weights: path to a dataset with a ``latitude`` coordinate
            (required when ``use_latitude_weights`` is True).
        channel_schema: optional ``ChannelSchema`` fixing the data target
            variable layout; when None the scored variables are discovered from
            the state dict on the first forward pass.
        validation: construct the validation variant (uses ``validation_loss``
            and ``valid_pred_key`` when given).
        pred_key: batch-dict key of the prediction to score (default
            ``y_processed``). Point it at a ``snapshot`` postblock's ``out_key``
            (e.g. ``y_processed_raw``) to score the prediction before the
            global budget fixers that follow the snapshot.
        valid_pred_key: prediction key for the validation variant; defaults to
            ``pred_key``. Setting it to ``y_processed`` validates (and selects
            checkpoints) on the fully corrected prediction while training on
            the raw one.
        exp_transforms: ``exp_transform`` postblock specs (``exp_transform_specs``;
            ``load_loss`` fills this from the config). Variables they cover are
            scored in physical units but scaled in log space, so their
            inverse-variance weights use the variance converted to physical units.

    Attributes:
        last_var_losses: ``{var_key: float}`` detached per-variable scores
            (pre-combination) from the most recent forward pass — the trainer
            logs these alongside the combined loss.
        var_keys: scored variable list, resolved on the first forward pass.
        log_variance: learnable per-variable ``log sigma_v^2`` parameter
            (``var_weighting="learnable"`` only, else None).
    """

    def __init__(
        self,
        training_loss: str = "mse",
        base_loss_parameters: dict | None = None,
        validation_loss: str | None = None,
        validation_loss_parameters: dict | None = None,
        base_loss_overrides: dict | None = None,
        var_weighting: str = "inverse_variance",
        scaler_path: str | None = None,
        variable_weights: dict | None = None,
        normalize_weights: bool = True,
        per_level_variance: bool = True,
        include_computed_diagnostics: bool = True,
        use_latitude_weights: bool = False,
        latitude_weights: str | None = None,
        channel_schema=None,
        validation: bool = False,
        exp_transforms: list | None = None,
        pred_key: str = "y_processed",
        valid_pred_key: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.validation = validation
        self.train_pred_key = pred_key
        self.pred_key = (valid_pred_key or pred_key) if validation else pred_key

        base_name = training_loss
        base_params = dict(base_loss_parameters or {})
        if validation and validation_loss is not None:
            base_name = validation_loss
            base_params = dict(validation_loss_parameters or {})
        self.base_loss = _instantiate_univariate_loss(base_name, base_params)
        if validation and is_crps_loss(base_name):
            logger.warning(
                "BaseLoss: validation is using '%s'. Validation keeps data-parallel dataset "
                "sharding, so ranks hold different samples and the cross-rank spread term is "
                "meaningless. Set loss.args.validation_loss to a deterministic loss (e.g. mae).",
                base_name,
            )

        # The trainer keys its ensemble setup (shared batches per dp rank,
        # per-rank member seeds) off loss.args.training_loss alone, so a CRPS
        # loss hidden in an override would run with wrong batching.
        for var_key, spec in (base_loss_overrides or {}).items():
            if is_crps_loss(spec["loss"]):
                raise ValueError(
                    f"BaseLoss: base_loss_overrides['{var_key}'] uses '{spec['loss']}', but CRPS "
                    "losses are only supported as the training_loss itself (the trainer's ensemble "
                    "setup keys off loss.args.training_loss and would not see the override)."
                )
        self.overrides = {
            var_key: _instantiate_univariate_loss(spec["loss"], spec.get("parameters", {}))
            for var_key, spec in (base_loss_overrides or {}).items()
        }

        if var_weighting not in VAR_WEIGHTING_MODES:
            raise ValueError(f"var_weighting must be one of {VAR_WEIGHTING_MODES}; got {var_weighting!r}")
        self.var_weighting = var_weighting
        self.manual_weights = {k: float(v) for k, v in (variable_weights or {}).items()}
        self.normalize_weights = bool(normalize_weights)
        self.per_level_variance = bool(per_level_variance)
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

        self._variances = None  # {var_key: (C,) per-level variance}
        if self.var_weighting in ("inverse_variance", "learnable"):
            if not scaler_path:
                raise ValueError(f"scaler_path is required for var_weighting='{self.var_weighting}'.")
            self._variances = _load_target_variances(scaler_path, exp_transforms, per_level=True)

        # {var_key: (C,) or (1,) tensor} for static modes; built at first forward
        self._combination_weights = None
        self.log_variance = None  # nn.Parameter for learnable mode
        self.var_keys = None  # full scoring list, resolved at first forward
        # Populated by forward(); initialized here so the documented attribute
        # exists (and logging code reading it is safe) before the first pass.
        self.last_var_losses = {}
        if self.var_weighting == "learnable":
            if self.data_var_keys is None:
                raise ValueError(
                    "var_weighting='learnable' requires a channel schema (channel_schema.yaml or a config "
                    "from which ChannelSchema can be derived) so the learnable parameters exist at init."
                )
            init = torch.zeros(len(self.data_var_keys))
            for i, var_key in enumerate(self.data_var_keys):
                if var_key in self._variances:
                    init[i] = float(np.log(max(float(self._variances[var_key].mean()), 1e-12)))
            self.log_variance = nn.Parameter(init)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_var_keys(self, pred: dict, target: dict) -> list:
        """Scored variable list: data target variables plus (optionally)
        postblock-computed diagnostics found only in ``y_processed``."""
        if self.data_var_keys is not None:
            var_keys = list(self.data_var_keys)
        else:
            var_keys = sorted(target.keys())
        extras = sorted(k for k in pred if k not in set(var_keys))
        if extras and self.include_computed_diagnostics:
            if self.var_weighting == "learnable":
                raise ValueError(
                    f"BaseLoss: computed diagnostics {extras} are not supported with "
                    "var_weighting='learnable' (learnable parameters are fixed at init). "
                    "Set include_computed_diagnostics: false or use a static var_weighting."
                )
            var_keys += extras
        elif extras:
            logger.info("BaseLoss: skipping computed diagnostics %s (include_computed_diagnostics=False).", extras)
        return var_keys

    def _inverse_variance(self, var_key: str, n_levels: int) -> "torch.Tensor | None":
        """``1 / variance`` for ``var_key``: ``(n_levels,)`` per level, or ``(1,)`` level-mean."""
        variance = self._variances.get(var_key)
        if variance is None:
            logger.warning(
                "BaseLoss: no scaler variance found for '%s'; falling back to weight 1.0. "
                "Add it to the scaler at scaler_path or set a manual entry in variable_weights.",
                var_key,
            )
            return None
        variance = variance.double()
        if self.per_level_variance and variance.numel() not in (1, n_levels):
            logger.warning(
                "BaseLoss: the scaler has %d channels for '%s' but the loss sees %d levels; "
                "using the level-mean variance.",
                variance.numel(),
                var_key,
                n_levels,
            )
            variance = variance.mean().reshape(1)
        elif not self.per_level_variance:
            variance = variance.mean().reshape(1)
        if not (variance > 0).all():
            logger.warning("BaseLoss: non-positive scaler variance for '%s'; falling back to weight 1.0.", var_key)
            return None
        return 1.0 / variance

    def _build_static_weights(self, var_keys, n_levels: dict) -> dict:
        """Build combination weights for the static modes (none/manual/inverse_variance).

        Each weight is a ``(C,)`` per-level or ``(1,)`` per-variable float64 tensor.
        """
        weights = {}
        for var_key in var_keys:
            manual = torch.tensor([self.manual_weights.get(var_key, 1.0)], dtype=torch.float64)
            if self.var_weighting == "inverse_variance":
                inv_var = self._inverse_variance(var_key, n_levels[var_key])
                weights[var_key] = manual if inv_var is None else manual * inv_var
            elif self.var_weighting == "manual":
                if var_key not in self.manual_weights:
                    logger.warning("BaseLoss: no variable_weights entry for '%s'; using 1.0.", var_key)
                weights[var_key] = manual
            else:  # "none"
                weights[var_key] = manual

        if self.normalize_weights:
            # A variable's loss is the level mean of its weighted per-level losses,
            # so its effective weight is the level mean of its weights.
            mean_w = sum(float(w.mean()) for w in weights.values()) / len(weights)
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

    @staticmethod
    def _flatten_state(state: dict, name: str) -> dict:
        """``{source: {var_key: tensor}}`` -> ``{var_key: tensor}``."""
        flat = {}
        for source_vars in state.values():
            flat.update(source_vars)
        if not flat:
            raise ValueError(f"BaseLoss: '{name}' is empty — check the per_step postblocks config.")
        return flat

    def forward(self, full_data_dict: dict, pred_key: str | None = None) -> torch.Tensor:
        """Combined loss of ``full_data_dict[pred_key]`` against ``y_target_processed``.

        ``pred_key`` defaults to the key this instance was built with
        (``pred_key``, or ``valid_pred_key`` for the validation variant).
        """
        pred_key = pred_key or self.pred_key
        if pred_key not in full_data_dict:
            if pred_key == "y_processed":
                raise KeyError(
                    "BaseLoss requires full_data_dict['y_processed'] — add a 'reconstruct' postblock "
                    "(with args.detach: false for training) to postblocks.per_step."
                )
            raise KeyError(
                f"BaseLoss: full_data_dict['{pred_key}'] is missing — add a snapshot postblock "
                f"(type: snapshot, args: {{out_key: {pred_key}}}) to postblocks.per_step."
            )
        if "y_target_processed" not in full_data_dict:
            raise KeyError(
                "BaseLoss requires full_data_dict['y_target_processed'] — add to postblocks.per_step: "
                "reconstruct_target: {type: reconstruct, args: {in_key: 'y', out_key: 'y_target_processed'}} "
                "followed by a bridgescaler_transformer postblock with key: 'y_target_processed'."
            )

        pred = self._flatten_state(full_data_dict[pred_key], pred_key)
        target = self._flatten_state(full_data_dict["y_target_processed"], "y_target_processed")

        if torch.is_grad_enabled() and not any(t.requires_grad for t in pred.values()):
            raise RuntimeError(
                f"BaseLoss: {pred_key} carries no gradient (all tensors detached). Set "
                "'detach: false' on the reconstruct postblock in postblocks.per_step so the "
                "loss backpropagates through the postblocks into the model."
            )

        if self.var_keys is None:
            self.var_keys = self._resolve_var_keys(pred, target)
            if self.var_weighting != "learnable":
                n_levels = {k: pred[k].shape[1] for k in self.var_keys if k in pred}
                self._combination_weights = self._build_static_weights(self.var_keys, n_levels)
        var_keys = self.var_keys
        weights = self._combination_weights

        var_losses = {}
        weighted_losses = {}
        self.last_var_losses = {}
        # Device of the scored tensors, captured in the loop below so the
        # learnable branch does not depend on a leaked loop variable. Falls
        # back to the parameter's own device, making the .to() a no-op.
        device = self.log_variance.device if self.log_variance is not None else None
        for var_key in var_keys:
            if var_key not in pred:
                raise KeyError(
                    f"BaseLoss: variable '{var_key}' missing from {pred_key} — "
                    "channel schema and postblocks output disagree."
                )
            if var_key not in target:
                raise KeyError(
                    f"BaseLoss: variable '{var_key}' missing from y_target_processed. If this is a "
                    "postblock-computed diagnostic, apply the same compute postblock to the target twin "
                    "(key: 'y_target_processed') or set include_computed_diagnostics: false."
                )
            # Physical-unit magnitudes (e.g. SP ~1e5 Pa) overflow fp16 — score in fp32.
            p = pred[var_key].float()
            device = p.device
            t = target[var_key].float().to(device)
            loss_fn = self.overrides.get(var_key, self.base_loss)
            elementwise = loss_fn(t, p)
            if elementwise.shape != p.shape:
                raise ValueError(
                    f"BaseLoss: base loss for '{var_key}' returned shape {tuple(elementwise.shape)}, "
                    f"expected elementwise {tuple(p.shape)}. Use a loss with reduction='none'."
                )
            lat_w = self._lat_w(p)
            if lat_w is not None:
                elementwise = elementwise * lat_w
            # Mean over every dim but the level (channel) dim 1; levels are
            # equal-sized, so the level mean equals the full elementwise mean.
            level_loss = elementwise.mean(dim=[d for d in range(elementwise.ndim) if d != 1])
            var_loss = level_loss.mean()
            var_losses[var_key] = var_loss
            self.last_var_losses[var_key] = var_loss.detach().item()
            if weights is not None:
                w = weights[var_key].to(device=device, dtype=level_loss.dtype)
                weighted_losses[var_key] = (w * level_loss).mean()

        if self.var_weighting == "learnable":
            log_var = self.log_variance.to(device)
            terms = []
            for i, var_key in enumerate(var_keys):
                manual = self.manual_weights.get(var_key, 1.0)
                terms.append(manual * torch.exp(-log_var[i]) * var_losses[var_key] + 0.5 * log_var[i])
            return torch.stack(terms).mean()

        return torch.stack([weighted_losses[var_key] for var_key in var_keys]).mean()
