from numbers import Real

import torch

from credit.ensemble.red_noise import SphericalAR1Pattern, ar1_phi
from credit.postblock.base import BasePostblock

# ECMWF's three-scale SPPT pattern (Palmer et al. 2009): standard deviation, e-folding time, correlation length.
DEFAULT_SCALES = (
    {"sigma": 0.52, "tau_days": 0.25, "length_scale_km": 500.0},
    {"sigma": 0.18, "tau_days": 3.0, "length_scale_km": 1000.0},
    {"sigma": 0.06, "tau_days": 30.0, "length_scale_km": 2000.0},
)
_FIELD_KEYS = {"var", "levels"}
_SCALE_KEYS = {"sigma", "tau_days", "length_scale_km"}


def _positive(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and value > 0


def _src(var_key: str) -> str:
    return var_key.split("/")[0]


def tendency_noise_errors(fields, scales=DEFAULT_SCALES, clip=1.0, lead_time_periods=6) -> list[str]:
    """Problems with a ``tendency_noise`` block's arguments (empty list if they are valid).

    Shared by :class:`TendencyNoise` (which raises on them) and ``credit check``.
    """
    errors = []
    if not isinstance(fields, (list, tuple)) or not fields:
        errors.append("fields must be a non-empty list of {var[, levels]} mappings")
        fields = []
    seen = set()
    for i, field in enumerate(fields):
        where = f"fields[{i}]"
        if not isinstance(field, dict):
            errors.append(f"{where} must be a mapping, got {type(field).__name__}")
            continue
        unknown = set(field) - _FIELD_KEYS
        if unknown:
            errors.append(f"{where}: unknown keys {sorted(unknown)} (accepted: {sorted(_FIELD_KEYS)})")
        var = field.get("var")
        if not isinstance(var, str) or "/" not in var:
            errors.append(f"{where}.var must be a variable key such as CESM/prognostic/3d/T, got {var!r}")
        elif var in seen:
            errors.append(f"{where}: {var} is listed twice")
        else:
            seen.add(var)
        levels = field.get("levels")
        if levels is not None and (
            not isinstance(levels, (list, tuple))
            or not levels
            or not all(isinstance(lv, int) and not isinstance(lv, bool) for lv in levels)
        ):
            errors.append(f"{where}.levels must be a non-empty list of level indices, got {levels!r}")
    if not isinstance(scales, (list, tuple)) or not scales:
        errors.append("scales must be a non-empty list of {sigma, tau_days, length_scale_km} mappings")
        scales = []
    for i, scale in enumerate(scales):
        where = f"scales[{i}]"
        if not isinstance(scale, dict):
            errors.append(f"{where} must be a mapping, got {type(scale).__name__}")
            continue
        if set(scale) != _SCALE_KEYS:
            errors.append(f"{where} needs exactly the keys {sorted(_SCALE_KEYS)}, got {sorted(scale)}")
        for key in sorted(_SCALE_KEYS & set(scale)):
            if not _positive(scale[key]):
                errors.append(f"{where}.{key} must be a positive number, got {scale[key]!r}")
    if not _positive(clip) or clip > 1:
        errors.append(f"clip must be in (0, 1] so that 1 + r stays non-negative, got {clip!r}")
    if not _positive(lead_time_periods):
        errors.append(f"lead_time_periods must be the model step in hours, got {lead_time_periods!r}")
    return errors


class TendencyNoise(BasePostblock):
    """Multiplicative, SPPT-style noise on each step's change of chosen prognostic fields.

    Every step it rescales the model's predicted change by a random factor,

        y' = x + (1 + r) * (y - x)

    where ``x`` is the step's input state and ``y`` the prediction, both in
    physical units. ``r`` is one random field shared by every perturbed field
    and level, as in SPPT: the sum over ``scales`` of ``sigma`` times a
    unit-variance, spatially smooth pattern with correlation length
    ``length_scale_km`` that is AR(1) in time with e-folding time ``tau_days``
    (:class:`credit.ensemble.red_noise.SphericalAR1Pattern`), clipped to
    ``[-clip, clip]``. One independent stream per batch row (ensemble member);
    the patterns start from their stationary distribution.

    The noise scales with the size of the change, so it is large where the
    model predicts big changes and vanishes where it predicts none. Unlike SPPT
    in a GCM, an emulator's change includes the resolved dynamics (advection),
    so winds and the dynamics are perturbed too; leave the winds out of
    ``fields`` for a thermodynamic-only variant.

    Place it in ``postblocks.per_step`` after the inverse scaler (it works in
    physical units) and before the tracer clamp and the conservation fixers, so
    the clamp keeps tracers non-negative and the fixers close each step's budget
    with the perturbation included. The patterns restart at every trajectory
    (``reset_noise_state``, called by the gen2 trainer at ``t == 1`` and by
    ``run_forecast``). To use it at inference only, put the postblock chain that
    contains it under ``inference.postblocks``. It needs the full grid, so it
    does not work with domain parallelism.

    Args:
        fields: one mapping per perturbed prognostic field:

            - ``var``: prediction key, e.g. ``CESM/prognostic/3d/T``; the same
              key must be in the step's physical inputs.
            - ``levels`` (3-D fields): level indices to perturb, e.g.
              ``[10, ..., 31]`` for CAM's levels below about 100 hPa. Omitted,
              every level is perturbed.

        scales: the pattern's components, each ``{sigma, tau_days,
            length_scale_km}``. Default: ECMWF's three scales (0.52, 6 h,
            500 km), (0.18, 3 d, 1000 km), (0.06, 30 d, 2000 km).
        clip: bound on ``|r|``, at most 1 so that ``1 + r >= 0``.
        lead_time_periods: model step in hours.
        input_source_key: batch-dict entry holding the step's physical inputs (default ``x_physical``).
        key: nested prediction dict to perturb (default ``y_processed``).
    """

    def __init__(
        self,
        fields: list,
        scales: list = DEFAULT_SCALES,
        clip: float = 1.0,
        lead_time_periods: float = 6,
        input_source_key: str = "x_physical",
        key: str = "y_processed",
    ):
        super().__init__()
        errors = tendency_noise_errors(fields, scales, clip, lead_time_periods)
        if errors:
            raise ValueError("tendency_noise: " + "; ".join(errors))
        self.fields = [dict(field) for field in fields]
        self.scales = [dict(scale) for scale in scales]
        self.clip = clip
        self.lead_time_periods = lead_time_periods
        self.input_source_key = input_source_key
        self.key = key

        # One pattern per scale, built on the first call (it needs the grid); plain attributes, not buffers.
        self.patterns = []

    def reset_noise_state(self):
        """Restart every pattern from its stationary distribution."""
        for pattern in self.patterns:
            pattern.reset()

    def sample_factor(self, batch: int, height: int, width: int, device) -> torch.Tensor:
        """Draw this step's ``r``, shape (batch, H, W)."""
        if not self.patterns:
            for scale in self.scales:
                phi = ar1_phi(scale["tau_days"] * 24.0, self.lead_time_periods)
                self.patterns.append(SphericalAR1Pattern(height, width, scale["length_scale_km"], phi))
        r = sum(scale["sigma"] * pattern.sample(batch, device) for scale, pattern in zip(self.scales, self.patterns))
        return r.clamp(-self.clip, self.clip)

    def forward(self, batch_dict: dict) -> dict:
        pred = batch_dict[self.key]
        state = batch_dict[self.input_source_key]
        first = self.fields[0]["var"]
        ref = pred[_src(first)][first]
        batch, height, width, device = ref.shape[0], ref.shape[-2], ref.shape[-1], ref.device

        r = self.sample_factor(batch, height, width, device)

        for field in self.fields:
            var = field["var"]
            y = pred[_src(var)][var]
            try:
                x = state[_src(var)][var]
            except KeyError:
                raise KeyError(f"tendency_noise: {var} is not in the step's physical inputs ({self.input_source_key}).")
            x = x[:, :, -1:].to(device, y.dtype)  # newest step of the input window (dim 2 is time)
            if x.shape[0] != batch:  # a batch-axis ensemble repeats each input sample
                x = x.repeat_interleave(batch // x.shape[0], dim=0)
            if x.shape != y.shape:
                raise ValueError(
                    f"tendency_noise: {var} input {tuple(x.shape)} and prediction {tuple(y.shape)} differ."
                )
            factor = r.to(y.dtype).reshape(batch, *([1] * (y.dim() - 3)), height, width)
            delta = factor * (y - x)
            if field.get("levels") is not None:
                on_level = torch.zeros(y.shape[1], dtype=y.dtype, device=device)
                on_level[list(field["levels"])] = 1.0
                delta = delta * on_level.reshape(1, -1, *([1] * (y.dim() - 2)))
            pred[_src(var)][var] = y + delta  # out of place: y may be on the autograd graph
        return batch_dict
