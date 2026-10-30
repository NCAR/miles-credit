"""
variances.py
------------
Per-level target variances measured in **loss space** for BaseLoss and the
gen2 metrics.

BaseLoss scores ``y_processed`` against ``y_target_processed`` in physical
units, after every inverse transform in the postblock chain (``square_transform``,
``exp_transform``, ...). The fitted bridgescaler holds variances of the values
it saw, i.e. *before* those inverses, so its variances are in the wrong units
for any transformed variable. ``credit preprocess`` therefore makes a second
pass once the scaler is fit: it runs the per_step preblocks and the target-twin
postblocks (every ``postblocks.per_step`` block writing to
``y_target_processed``) and records, for every scored variable and level:

- ``temporal_variance``: variance over time at each grid point, averaged over
  the observed points (``climatology scores 1``);
- ``pooled_variance``: variance over time and space together (what a
  per-channel standard scaler normalizes by);
- ``observed_fraction``: fraction of the field that is observed.

Both averages are latitude-weighted with the loss's ``latitude_weights`` when
``use_latitude_weights`` is on. A point is observed where the target is finite
just before the scaler (the scaler's own definition) and, optionally, where a
``variance_masks`` rule holds, for fields masked by a value or by a 0/1 mask
variable instead of NaN.

Config (``loss.args``; the metrics read the same three keys)::

    loss:
      type: base
      args:
        variance_path: /path/loss_variances.json   # written by credit preprocess
        variance_type: temporal                    # temporal | pooled
        mask_correction: true                      # weight by 1 / (variance * observed_fraction)
        variance_batches: 250                      # preprocess only; default half the scaler batches
        variance_masks:                            # preprocess only; optional
          SRC/prognostic/2d/SST: {op: "!=", value: 0}                       # 0 means "no data"
          SRC/prognostic/2d/SOILWAT: {mask_var: SRC/static/2d/LANDMASK, op: ">", value: 0.5}
"""

import json
import logging
import os
from datetime import UTC, datetime

import torch

logger = logging.getLogger(__name__)

VARIANCE_TYPES = ("temporal", "pooled")

_OPS = {
    "==": torch.eq,
    "!=": torch.ne,
    "<": torch.lt,
    "<=": torch.le,
    ">": torch.gt,
    ">=": torch.ge,
}


def _match(var_key: str, selectors) -> "str | None":
    """The selector that names ``var_key`` exactly or as a "/"-delimited prefix."""
    for selector in selectors:
        if var_key == selector or var_key.startswith(selector + "/"):
            return selector
    return None


def target_twin_conf(conf: dict) -> dict:
    """Copy of ``conf`` whose ``postblocks.per_step`` holds only the target-twin blocks.

    A block belongs to the twin when its ``key`` or ``out_key`` arg is
    ``y_target_processed``; these rebuild exactly the tensors BaseLoss scores
    the prediction against.
    """
    per_step = (conf.get("postblocks") or {}).get("per_step") or {}
    twin = {}
    for name, block in per_step.items():
        args = (block or {}).get("args") or {} if isinstance(block, dict) else {}
        if "y_target_processed" in (args.get("key"), args.get("out_key")):
            twin[name] = block
    if not twin:
        raise ValueError(
            "No postblocks.per_step block writes to 'y_target_processed'; loss-space variances need the "
            "target twin (reconstruct with out_key: y_target_processed, then its inverse transforms)."
        )
    out = dict(conf)
    out["postblocks"] = {"per_step": twin}
    return out


def _lookup(batch: dict, var_key: str) -> "torch.Tensor | None":
    source = var_key.split("/")[0]
    for data_type in ("target", "input"):
        tensor = (batch.get(data_type) or {}).get(source, {}).get(var_key)
        if tensor is not None:
            return tensor
    return None


def observed_masks(target: dict, pre_scaler_batch: dict, mask_rules: "dict | None" = None) -> dict:
    """``{var_key: bool tensor}``: where each target variable counts as observed.

    Args:
        target: flat ``{var_key: tensor}`` of the loss-space target.
        pre_scaler_batch: the nested batch just before the scaler (NaN marks
            missing data there; later preblocks may fill it).
        mask_rules: ``{var_key or prefix: {"op", "value", "mask_var"?}}``. The
            comparison runs on ``mask_var`` (any input or target variable,
            broadcast over levels) or, without it, on the variable itself.
    """
    mask_rules = mask_rules or {}
    masks = {}
    for var_key, tensor in target.items():
        pre = _lookup(pre_scaler_batch, var_key)
        if pre is not None and pre.shape == tensor.shape:
            mask = torch.isfinite(pre.to(tensor.device))
        else:
            if pre is not None:
                logger.warning(
                    "Loss variances: '%s' has shape %s before the scaler but %s in y_target_processed; "
                    "counting every point as observed.",
                    var_key,
                    tuple(pre.shape),
                    tuple(tensor.shape),
                )
            mask = torch.isfinite(tensor)
        selector = _match(var_key, mask_rules)
        if selector is not None:
            rule = mask_rules[selector]
            mask_var = rule.get("mask_var")
            source = _lookup(pre_scaler_batch, mask_var) if mask_var else (pre if pre is not None else tensor)
            if source is None:
                raise KeyError(f"variance_masks['{selector}']: mask_var '{mask_var}' is not in the batch.")
            source = source.to(tensor.device)
            keep = _OPS[rule.get("op", "!=")](source, float(rule["value"])) & torch.isfinite(source)
            mask = mask & keep.expand_as(mask)
        masks[var_key] = mask
    return masks


class LossVarianceAccumulator:
    """Per-grid-point running sums (count, sum, sum of squares) in float64.

    Sums are additive, so per-rank accumulators combine with one all-reduce.
    Raw float64 sums lose about ``1e-16 * (mean / std)**2`` relative precision in
    a grid point's variance: negligible for geophysical fields (surface pressure,
    mean/std ~100, loses ~1e-12).
    Tensors are ``(..., L, ..., H, W)`` with the level at dim 1 and the grid in
    the last two dims; every other dim (batch, time) is a sample dim.
    """

    def __init__(self):
        self.sums = {}  # var_key -> (3, L, H, W): count, sum, sum of squares
        self.n_samples = {}  # var_key -> samples seen per grid point, observed or not

    def update(self, target: dict, masks: dict) -> None:
        for var_key, tensor in target.items():
            # Move, then cast: a combined .to("cpu", torch.float64) returns zeros from MPS (torch 2.x).
            x = tensor.detach().cpu().to(torch.float64)
            mask = masks[var_key].to("cpu")
            dims = [d for d in range(x.ndim) if d not in (1, x.ndim - 2, x.ndim - 1)]
            xz = torch.where(mask, x, torch.zeros((), dtype=x.dtype))
            batch_sums = torch.stack([mask.to(x.dtype).sum(dims), xz.sum(dims), (xz * xz).sum(dims)])
            if var_key in self.sums:
                self.sums[var_key] += batch_sums
            else:
                self.sums[var_key] = batch_sums
            n = 1
            for d in dims:
                n *= x.shape[d]
            self.n_samples[var_key] = self.n_samples.get(var_key, 0) + n

    def all_reduce(self) -> None:
        """Sum the accumulators across data-parallel ranks (no-op single-process)."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return
        for var_key in sorted(self.sums):
            dist.all_reduce(self.sums[var_key])
            n = torch.tensor([self.n_samples[var_key]], dtype=torch.float64)
            dist.all_reduce(n)
            self.n_samples[var_key] = int(n.item())

    def finalize(self, lat_weights: "torch.Tensor | None" = None) -> dict:
        """``{var_key: {"temporal_variance", "pooled_variance", "observed_fraction": [L], "n_samples"}}``."""
        out = {}
        for var_key, (count, total, total_sq) in self.sums.items():
            height, width = count.shape[-2:]
            if lat_weights is None:
                w = torch.ones(height, 1, dtype=torch.float64)
            else:
                w = lat_weights.to(torch.float64).view(-1, 1)
                if w.shape[0] != height:
                    raise ValueError(f"latitude weights have {w.shape[0]} rows but '{var_key}' has {height}.")
            w = w.expand(height, width)
            valid = count >= 2
            n = torch.where(valid, count, torch.ones_like(count))
            mean = total / n
            local_var = (total_sq / n - mean * mean).clamp_min(0.0)
            wn = torch.where(valid, w * count, torch.zeros_like(count))  # weight of each point's samples
            wn_sum = wn.sum((-2, -1))
            safe = torch.where(wn_sum > 0, wn_sum, torch.ones_like(wn_sum))
            temporal = (wn * local_var).sum((-2, -1)) / safe
            grand = (wn * mean).sum((-2, -1)) / safe
            # Law of total variance: within-point plus between-point, stable for large means (PS ~1e5 Pa).
            between = (wn * (mean - grand[:, None, None]) ** 2).sum((-2, -1)) / safe
            pooled = temporal + between
            observed = (w * count).sum((-2, -1)) / (w.sum() * self.n_samples[var_key])
            nan = torch.full_like(temporal, float("nan"))
            out[var_key] = {
                "temporal_variance": torch.where(wn_sum > 0, temporal, nan).tolist(),
                "pooled_variance": torch.where(wn_sum > 0, pooled, nan).tolist(),
                "observed_fraction": observed.tolist(),
                "n_samples": self.n_samples[var_key],
            }
        return out


def save_loss_variances(path: str, variables: dict, meta: dict) -> None:
    """Write the variance file atomically (``meta`` + ``variables``)."""
    path = os.path.expandvars(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = {"created": datetime.now(UTC).isoformat(timespec="seconds"), **meta}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"meta": meta, "variables": variables}, f, indent=1, allow_nan=True)
    os.replace(tmp, path)


def load_loss_variances(path: str, variance_type: str = "temporal", mask_correction: bool = True) -> dict:
    """``{var_key: (L,) tensor}`` of effective per-level variances from a variance file.

    With ``mask_correction`` the variance is multiplied by the observed
    fraction: unobserved points add (almost) no error to the loss mean, which
    shrinks a masked variable's score by that fraction, and a weight of
    ``1 / (variance * fraction)`` undoes it. Levels with no observed data come
    back NaN.
    """
    if variance_type not in VARIANCE_TYPES:
        raise ValueError(f"variance_type must be one of {VARIANCE_TYPES}; got {variance_type!r}")
    with open(os.path.expandvars(path)) as f:
        data = json.load(f)
    variances = {}
    for var_key, entry in data["variables"].items():
        variance = torch.tensor(entry[f"{variance_type}_variance"], dtype=torch.float64)
        if mask_correction:
            variance = variance * torch.tensor(entry["observed_fraction"], dtype=torch.float64)
        variances[var_key] = variance
    return variances


def resolve_target_variances(
    scaler_path: "str | None",
    exp_transforms: "list | None" = None,
    variance_path: "str | None" = None,
    variance_type: str = "temporal",
    mask_correction: bool = True,
    per_level: bool = False,
) -> dict:
    """Variances for inverse-variance weighting: the variance file where it covers a
    variable, the bridgescaler at ``scaler_path`` otherwise.

    Returns ``(L,)`` tensors with ``per_level`` and level-mean floats without.
    """
    from credit.losses.base import _load_target_variances

    variances = _load_target_variances(scaler_path, exp_transforms, per_level=per_level) if scaler_path else {}
    if variance_path:
        from_file = load_loss_variances(variance_path, variance_type, mask_correction)
        for var_key, variance in from_file.items():
            if not torch.isfinite(variance).all():
                logger.warning(
                    "'%s' has levels with no observed data in %s; keeping the scaler variance.", var_key, variance_path
                )
                continue
            variances[var_key] = variance.float() if per_level else float(variance.mean())
        missing = sorted(k for k in variances if k not in from_file)
        if missing:
            logger.warning("Variables not in %s use scaler variances: %s", variance_path, missing)
    return variances
