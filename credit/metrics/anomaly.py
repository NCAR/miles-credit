"""
anomaly.py
----------
Anomaly-based verification metrics for the Gen 2 metrics framework.

Implements the anomaly correlation coefficient (ACC) and forecast activity
(SDAF) as described in Bonavita & Geer (2026), "Forecast verification using
information and noise", *Q. J. R. Meteorol. Soc.*, 152, e70109,
:doi:`10.1002/qj.70109`.

Both metrics require a **climatology** field — a per-variable spatial mean
representing the long-term average state. The climatology is subtracted from
forecast and truth before computing anomalies, and the area-weighted mean of
the anomaly is removed (debiasing) following the operational practice
described in Appendix A of the paper.

Climatology sources (in priority order):

1. **``climatology_path``** — a path to an xarray-readable dataset
   (netCDF/Zarr) whose data variables are named by the **short name** of
   each var_key (the last path component, e.g. ``"temperature"`` for
   ``"ERA5/prognostic/3d/temperature"``). The dataset must have ``latitude``
   and ``longitude`` coordinates; 3-D variables additionally need a ``level``
   (or ``levels``) coordinate matching the data config.

2. **Validation data (default)** — when no ``climatology_path`` is given the
   climatology is accumulated as a running mean of the target
   (``y_target_processed``) fields seen during validation. On the first
   batch the climatology equals that batch's target mean; it converges to the
   full validation mean as more batches are processed. This is an online
   approximation — for exact results either provide a climatology file or
   ensure the metric sees all validation data before the scores are logged.

3. **``climatology``** — a pre-computed ``{var_key: torch.Tensor}`` dict
   supplied programmatically (each tensor broadcastable to the variable's
   spatial shape). Overrides both ``climatology_path`` and online
   accumulation.

Config example (combined metric with ACC and activity)::

    metrics:
      type: combined
      args:
        metrics:
          rmse: {}
          acc:
            climatology_path: /path/to/climatology.nc
          activity:
            climatology_path: /path/to/climatology.nc
        var_weighting: "none"
        use_latitude_weights: true
        latitude_weights: /path/to/static.zarr

Code example::

    from credit.metrics.anomaly import AnomalyCorrelationCoefficientMetric

    metric = AnomalyCorrelationCoefficientMetric(
        metric_name="acc",
        var_weighting="none",
        use_latitude_weights=True,
        latitude_weights="/path/to/static.zarr",
    )
    scores = metric(full_data_dict)
    # {"acc/ERA5/.../T": 0.95, "acc": 0.92, ...}

Mathematical definitions (following Bonavita & Geer 2026, Appendix A)
with area weights ``w_i`` normalised to mean 1:

- Debiased forecast anomaly: ``d_f = x_f - x_c - mean_w(x_f - x_c)``
- Debiased truth anomaly:     ``d_t = x_t - x_c - mean_w(x_t - x_c)``
- Forecast activity (SDAF):   ``SDAF = sqrt(mean_w(d_f^2))``
- Truth activity (SDAV):      ``SDAV = sqrt(mean_w(d_t^2))``
- ACC:                        ``ACC = mean_w(d_f * d_t) / (SDAF * SDAV)``
"""

import logging
import os

import torch

from credit.losses.base import BaseLoss
from credit.metrics.base import BaseVariableMetric

logger = logging.getLogger(__name__)

__all__ = [
    "AnomalyCorrelationCoefficientMetric",
    "ForecastActivityMetric",
]

_EPS = 1e-12


class _AnomalyMetricBase(BaseVariableMetric):
    """Shared climatology management for anomaly-based metrics.

    Subclasses implement :meth:`_score_anomaly` which receives the debiased
    forecast and truth anomaly tensors (latitude-weighted mean already
    subtracted) plus the latitude weights.

    Args:
        climatology_path: path to an xarray-readable climatology dataset.
            Variables in the file are looked up by the short name (last
            component) of each var_key.
        climatology: pre-computed ``{var_key: Tensor}`` dict; takes priority
            over ``climatology_path`` and online accumulation.
        accumulate_climatology: when True (default) and no file/dict is
            provided, accumulate the target mean across forward calls. Set
            False to use the current batch's target mean as the climatology
            for that batch only.
    """

    def __init__(
        self,
        *args,
        climatology_path: str | None = None,
        climatology: dict | None = None,
        accumulate_climatology: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.climatology_path = climatology_path
        self.accumulate_climatology = bool(accumulate_climatology)

        # Fixed climatology (from file or dict) — never changes after init.
        self._climatology: dict[str, torch.Tensor] = {}
        if climatology is not None:
            self._climatology = {k: v.float() for k, v in climatology.items()}
        elif climatology_path:
            self._load_climatology_from_file(climatology_path)

        # Online running mean state.
        self._clim_sum: dict[str, torch.Tensor] = {}
        self._clim_count: int = 0

    # ------------------------------------------------------------------
    # Climatology loading
    # ------------------------------------------------------------------

    def _load_climatology_from_file(self, path: str) -> None:
        """Load per-variable climatology fields from an xarray dataset."""
        import xarray as xr

        ds = xr.open_dataset(os.path.expandvars(path))
        for var_name in ds.data_vars:
            # Store under every var_key whose short name matches.
            # The full mapping happens lazily in _get_clim via short-name
            # lookup, so here we just store by short name.
            self._climatology[var_name] = torch.as_tensor(ds[var_name].values, dtype=torch.float32)
        logger.info(
            "Loaded climatology from %s with variables: %s",
            path,
            sorted(self._climatology),
        )

    @staticmethod
    def _short_name(var_key: str) -> str:
        """Last path component of a var_key (e.g. 'temperature')."""
        return var_key.rsplit("/", 1)[-1]

    def _get_clim(self, var_key: str, ref: torch.Tensor) -> torch.Tensor | None:
        """Return the climatology field for *var_key* broadcastable to *ref*.

        Priority: fixed dict (by full key, then short name) → online running
        mean → None (caller falls back to batch mean).
        """
        # Fixed climatology — try full var_key first, then short name.
        if var_key in self._climatology:
            clim = self._climatology[var_key]
        else:
            short = self._short_name(var_key)
            if short in self._climatology:
                clim = self._climatology[short]
            else:
                clim = None

        if clim is not None:
            clim = clim.to(device=ref.device, dtype=ref.dtype)
            if clim.ndim < ref.ndim and ref.ndim == 5:
                # ref is (B, C, T, H, W): axis 1 is channel/level, axis 2 is
                # time, and the last two axes are spatial. Climatology fields
                # are stored without batch or time axes — (lat, lon) for 2-D
                # variables and (level, lat, lon) for 3-D — so insert the
                # missing axes at their correct positions. Left-padding with
                # unsqueeze(0) instead would land `level` on the time axis,
                # which fails to broadcast for every 3-D variable.
                leading = clim.shape[:-2]
                if len(leading) > 1:
                    raise ValueError(
                        f"{type(self).__name__}: climatology for '{var_key}' has shape "
                        f"{tuple(clim.shape)}; expected (lat, lon) for a 2-D variable or "
                        "(level, lat, lon) for a 3-D variable."
                    )
                n_levels = leading[0] if leading else 1
                if n_levels not in (1, ref.shape[1]):
                    raise ValueError(
                        f"{type(self).__name__}: climatology for '{var_key}' has {n_levels} "
                        f"levels but the field has {ref.shape[1]}. Check that the climatology "
                        "file's level coordinate matches the data config."
                    )
                return clim.reshape(1, n_levels, 1, *clim.shape[-2:])
            while clim.ndim < ref.ndim:
                clim = clim.unsqueeze(0)
            return clim

        # Online running mean.
        if var_key in self._clim_sum and self._clim_count > 0:
            clim = self._clim_sum[var_key] / self._clim_count
            return clim.to(device=ref.device, dtype=ref.dtype)

        return None

    def _update_climatology(self, target_flat: dict[str, torch.Tensor]) -> None:
        """Accumulate target fields for the online running mean."""
        if not self.accumulate_climatology:
            return
        if self._climatology:
            return  # fixed climatology — no need to accumulate
        for var_key, t in target_flat.items():
            batch_size = t.shape[0]
            t_detached = t.detach().float()
            if var_key in self._clim_sum:
                self._clim_sum[var_key] += t_detached.sum(dim=0)
            else:
                self._clim_sum[var_key] = t_detached.sum(dim=0)
        # All variables in one batch share the same batch size.
        self._clim_count += batch_size

    # ------------------------------------------------------------------
    # Forward — override to manage climatology before scoring
    # ------------------------------------------------------------------

    def forward(self, full_data_dict: dict) -> dict:
        if "y_processed" not in full_data_dict:
            raise KeyError(
                f"{type(self).__name__} requires full_data_dict['y_processed'] — "
                "add a 'reconstruct' postblock to postblocks.per_step."
            )
        if "y_target_processed" not in full_data_dict:
            raise KeyError(
                f"{type(self).__name__} requires full_data_dict['y_target_processed'] — "
                "add to postblocks.per_step: reconstruct_target + bridgescaler_transform."
            )

        pred = BaseLoss._flatten_state(full_data_dict["y_processed"], "y_processed")
        target = BaseLoss._flatten_state(full_data_dict["y_target_processed"], "y_target_processed")

        # Accumulate the online climatology before scoring.
        self._update_climatology(target)

        if self.var_keys is None:
            self.var_keys = self._resolve_var_keys(pred, target)
            self._combination_weights = self._build_static_weights(self.var_keys)
        var_keys = self.var_keys
        weights = self._combination_weights

        var_scores: dict[str, torch.Tensor] = {}
        self.last_var_scores: dict[str, float] = {}
        with torch.no_grad():
            for var_key in var_keys:
                if var_key not in pred:
                    raise KeyError(f"{type(self).__name__}: variable '{var_key}' missing from y_processed.")
                if var_key not in target:
                    raise KeyError(f"{type(self).__name__}: variable '{var_key}' missing from y_target_processed.")
                p = pred[var_key].float()
                t = target[var_key].float().to(p.device)

                clim = self._get_clim(var_key, p)
                if clim is None:
                    # Fallback: use this batch's target mean as climatology.
                    logger.debug(
                        "%s: no climatology for '%s'; using batch target mean.",
                        type(self).__name__,
                        var_key,
                    )
                    clim = t.mean(dim=0, keepdim=True)

                # Broadcast clim to match p if needed.
                clim = clim.to(p.device, p.dtype)
                if clim.shape != p.shape:
                    clim = clim.expand_as(p)

                lat_w = self._lat_w(p)
                w = lat_w if lat_w is not None else 1.0

                # Debiased anomalies (Appendix A, eqs A1–A2, A31–A32).
                d_f = (p - clim) - (w * (p - clim)).mean()
                d_t = (t - clim) - (w * (t - clim)).mean()

                var_score = self._score_anomaly(d_f, d_t, w)
                var_scores[var_key] = var_score
                self.last_var_scores[var_key] = var_score.item()

            aggregate = torch.stack([weights[var_key] * var_scores[var_key] for var_key in var_keys]).mean().item()

        out = {f"{self.metric_name}/{var_key}": self.last_var_scores[var_key] for var_key in var_keys}
        out[f"{self.metric_name}"] = aggregate
        return out

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _score_anomaly(self, d_f: torch.Tensor, d_t: torch.Tensor, w: torch.Tensor | float) -> torch.Tensor:
        """Score from debiased forecast anomaly *d_f*, truth anomaly *d_t*,
        and area weights *w* (normalised to mean 1, or 1.0 if no lat weights).

        Subclasses must override this method.
        """
        raise NotImplementedError


class AnomalyCorrelationCoefficientMetric(_AnomalyMetricBase):
    """Anomaly correlation coefficient (ACC) per variable.

    ACC measures the cosine of the angle between the debiased forecast and
    truth anomaly vectors in the area-weighted state space (Bonavita & Geer
    2026, eq. A8). It is bounded by ±1 and is insensitive to bias and
    forecast activity.

    - ACC = 1: perfect anomaly pattern.
    - ACC = 0: no correlation with the observed anomaly.
    - ACC < 0: negatively correlated (worse than climatology).

    Requires a climatology (see :class:`_AnomalyMetricBase`).
    """

    scale_power = 0  # a correlation is dimensionless

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Not used — _score_anomaly is the hook for anomaly metrics.
        return (pred - target) ** 2

    def _score_anomaly(self, d_f: torch.Tensor, d_t: torch.Tensor, w: torch.Tensor | float) -> torch.Tensor:
        sdaf = torch.sqrt((w * d_f**2).mean() + _EPS)
        sdav = torch.sqrt((w * d_t**2).mean() + _EPS)
        dot = (w * d_f * d_t).mean()
        return dot / (sdaf * sdav)


class ForecastActivityMetric(_AnomalyMetricBase):
    """Forecast activity (SDAF) per variable.

    SDAF is the area-weighted standard deviation of the debiased forecast
    anomaly (Bonavita & Geer 2026, eq. A5). It quantifies how much the
    forecast deviates from climatology — i.e. how "active" the forecast is.
    Unrealistically smooth forecasts (e.g. from ML emulators or ensemble
    averaging) have reduced SDAF.

    Requires a climatology (see :class:`_AnomalyMetricBase`).
    """

    scale_power = 1  # a standard deviation is linear in sigma

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Not used — _score_anomaly is the hook for anomaly metrics.
        return (pred - target) ** 2

    def _score_anomaly(self, d_f: torch.Tensor, d_t: torch.Tensor, w: torch.Tensor | float) -> torch.Tensor:
        return torch.sqrt((w * d_f**2).mean() + _EPS)
