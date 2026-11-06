import os
from numbers import Real

import numpy as np
import torch

from credit.ensemble.red_noise import SphericalAR1Pattern, ar1_phi
from credit.postblock.base import BasePostblock

_FIELD_KEYS = {"var", "levels", "sigma", "tau_days", "pattern"}


def _positive(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and value > 0


def _default_pattern(field: dict) -> str:
    """Without a ``pattern`` name every field gets its own pattern."""
    levels = field.get("levels")
    return field.get("var", "") if levels is None else f"{field.get('var', '')}{list(levels)}"


def _src(var_key: str) -> str:
    return var_key.split("/")[0]


def targeted_red_noise_errors(fields, length_scale_km=500.0, lead_time_periods=6) -> list[str]:
    """Problems with a ``targeted_red_noise`` block's arguments (empty list if they are valid).

    Shared by :class:`TargetedRedNoise` (which raises on them) and ``credit check``.
    """
    if not isinstance(fields, (list, tuple)) or not fields:
        return ["fields must be a non-empty list of {var, sigma, tau_days[, levels, pattern]} mappings"]
    errors = []
    pattern_tau = {}
    levels_by_var = {}
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
            errors.append(f"{where}.var must be a variable key such as CESM/prognostic/2d/TREFHT, got {var!r}")
        tau = field.get("tau_days")
        if not _positive(tau):
            errors.append(f"{where}.tau_days must be a positive number, got {tau!r}")
        sigma = field.get("sigma")
        if isinstance(sigma, dict):
            if set(sigma) != {"path", "var"} or not all(isinstance(sigma[k], str) and sigma[k] for k in sigma):
                errors.append(f"{where}.sigma as a map needs exactly two strings, {{path, var}}; got {sigma!r}")
        elif not _positive(sigma):
            errors.append(f"{where}.sigma must be a positive number or a {{path, var}} map, got {sigma!r}")
        levels = field.get("levels")
        if levels is not None and (
            not isinstance(levels, (list, tuple))
            or not levels
            or not all(isinstance(lv, int) and not isinstance(lv, bool) for lv in levels)
        ):
            errors.append(f"{where}.levels must be a non-empty list of level indices, got {levels!r}")
            levels = None
        if isinstance(var, str):
            taken = levels_by_var.setdefault(var, [])
            mine = None if levels is None else set(levels)
            if any(other is None or mine is None or other & mine for other in taken):
                errors.append(f"{where}: {var} is perturbed twice (overlapping levels)")
            taken.append(mine)
        name = field.get("pattern", _default_pattern(field))
        if not isinstance(name, str):
            errors.append(f"{where}.pattern must be a name, got {name!r}")
        elif _positive(tau):
            if name in pattern_tau and pattern_tau[name] != tau:
                errors.append(
                    f"{where}: fields sharing pattern {name!r} must share tau_days ({pattern_tau[name]} vs {tau})"
                )
            pattern_tau.setdefault(name, tau)
    if not _positive(length_scale_km):
        errors.append(f"length_scale_km must be a positive number, got {length_scale_km!r}")
    if not _positive(lead_time_periods):
        errors.append(f"lead_time_periods must be the model step in hours, got {lead_time_periods!r}")
    return errors


def _load_map(path: str, var: str) -> torch.Tensor:
    """A 2-D (lat, lon) field from a NetCDF file, NaN set to 0."""
    import xarray as xr

    with xr.open_dataset(os.path.expandvars(path)) as ds:
        if var not in ds:
            raise KeyError(f"targeted_red_noise: {path} has no variable {var!r}; it has {sorted(ds.data_vars)}.")
        values = ds[var].squeeze().values
    if values.ndim != 2:
        raise ValueError(f"targeted_red_noise: {path}:{var} must be a 2-D (lat, lon) map, got shape {values.shape}.")
    return torch.as_tensor(np.nan_to_num(values.astype(np.float32)))


class TargetedRedNoise(BasePostblock):
    """Add slow, spatially smooth noise to chosen fields over land and sea ice, in physical units.

    A stochastic term for the part of the emulator's one-step error that has
    weeks of memory and that its inputs cannot predict (hidden land and
    sea-ice surface state). Every step it adds

        field += sigma(x) * mask(x) * pattern(x, t)

    where ``pattern`` is a unit-variance isotropic random field with
    correlation length ``length_scale_km`` that is AR(1) in time with
    e-folding time ``tau_days`` (:class:`credit.ensemble.red_noise.SphericalAR1Pattern`),
    one independent stream per batch row (ensemble member). It starts from its
    stationary distribution, so the added noise has standard deviation
    ``sigma * mask`` at every step, the first included.

    ``mask = 1 - (1 - land) * (1 - ice)``: the land fraction from a static
    file (CAM's ``LANDFRAC``; ``LANDM_COSLAT`` is a smoothed coastal transition
    field that reaches far over the ocean) joined with a sea-ice fraction read
    from the physical inputs of the step (``ICEFRAC``). With neither, the noise
    is global.

    Place it in ``postblocks.per_step`` after the inverse scaler (it works in
    physical units) and before the tracer clamp and the conservation fixers,
    so that the clamp keeps tracers non-negative and the fixers close each
    step's budget with the perturbation included. After the fixers its global
    integral would become the next step's baseline, and red noise would
    random-walk the global energy and water. A perturbed prognostic feeds the
    next step; a perturbed diagnostic only adds spread.

    The pattern restarts at every trajectory (``reset_noise_state``, called by
    the gen2 trainer at ``t == 1`` and by ``run_forecast``). To use it at
    inference only, put the postblock chain that contains it under
    ``inference.postblocks``. It needs the full grid, so it does not work with
    domain parallelism, which hands postblocks a spatial shard.

    Args:
        fields: one mapping per perturbed field:

            - ``var``: prediction key, e.g. ``CESM/prognostic/2d/TREFHT``.
            - ``sigma``: per-step standard deviation in the field's units: a
              number, or ``{path, var}`` naming a (lat, lon) map on the model
              grid, in the data's orientation.
            - ``tau_days``: e-folding time of the pattern.
            - ``levels`` (3-D fields): level indices to perturb, e.g. ``[31]``
              for CAM's lowest level. Omitted, every level is perturbed.
            - ``pattern`` (optional): fields with the same name share one
              pattern (and must share ``tau_days``); otherwise each field has
              its own.

        lead_time_periods: model step in hours.
        length_scale_km: correlation length L; correlation is ``~exp(-d**2 / (2 L**2))``.
        land_file, land_var: static land-fraction map (e.g. ``LANDFRAC`` in the CAM statics file).
        ice_var: sea-ice fraction key in the physical inputs, e.g. ``CESM/dynamic_forcing/2d/ICEFRAC``.
        input_source_key: batch-dict entry holding the physical inputs (default ``x_physical``).
        key: nested prediction dict to perturb (default ``y_processed``).
    """

    def __init__(
        self,
        fields: list,
        lead_time_periods: float = 6,
        length_scale_km: float = 500.0,
        land_file: str | None = None,
        land_var: str = "LANDFRAC",
        ice_var: str | None = None,
        input_source_key: str = "x_physical",
        key: str = "y_processed",
    ):
        super().__init__()
        errors = targeted_red_noise_errors(fields, length_scale_km, lead_time_periods)
        if errors:
            raise ValueError("targeted_red_noise: " + "; ".join(errors))
        self.fields = [{**field, "pattern": field.get("pattern", _default_pattern(field))} for field in fields]
        self.lead_time_periods = lead_time_periods
        self.length_scale_km = length_scale_km
        self.ice_var = ice_var
        self.input_source_key = input_source_key
        self.key = key

        self.register_buffer("land", _load_map(land_file, land_var) if land_file else None, persistent=False)
        for i, field in enumerate(self.fields):
            if isinstance(field["sigma"], dict):
                self.register_buffer(f"sigma_map_{i}", _load_map(**field["sigma"]), persistent=False)

        # One pattern per name, built on the first call (it needs the grid); a plain attribute, not a buffer.
        self.patterns = {}

    def reset_noise_state(self):
        """Restart every pattern from its stationary distribution."""
        for pattern in self.patterns.values():
            pattern.reset()

    def _check_grid(self, name: str, tensor: torch.Tensor, height: int, width: int) -> None:
        if tuple(tensor.shape[-2:]) != (height, width):
            raise ValueError(
                f"targeted_red_noise: {name} is {tuple(tensor.shape[-2:])} but the prediction grid is {(height, width)}."
            )

    def _mask(self, batch_dict: dict, batch: int, height: int, width: int, device) -> torch.Tensor | None:
        """``1 - (1 - land) * (1 - ice)``, shape (batch or 1, H, W); None when there is no mask."""
        uncovered = None
        if self.land is not None:
            self._check_grid("the land map", self.land, height, width)
            uncovered = 1.0 - self.land.to(device).clamp(0.0, 1.0)[None]
        if self.ice_var is not None:
            ice = batch_dict[self.input_source_key][_src(self.ice_var)][self.ice_var]
            self._check_grid(self.ice_var, ice, height, width)
            ice = ice.reshape(ice.shape[0], -1, height, width)[:, -1].to(device, torch.float32).clamp(0.0, 1.0)
            if ice.shape[0] != batch:  # a batch-axis ensemble repeats each input sample
                ice = ice.repeat_interleave(batch // ice.shape[0], dim=0)
            uncovered = (1.0 - ice) if uncovered is None else uncovered * (1.0 - ice)
        return None if uncovered is None else 1.0 - uncovered

    def forward(self, batch_dict: dict) -> dict:
        pred = batch_dict[self.key]
        first = self.fields[0]["var"]
        ref = pred[_src(first)][first]
        batch, height, width, device = ref.shape[0], ref.shape[-2], ref.shape[-1], ref.device

        if not self.patterns:
            for field in self.fields:
                if field["pattern"] not in self.patterns:
                    phi = ar1_phi(field["tau_days"] * 24.0, self.lead_time_periods)
                    self.patterns[field["pattern"]] = SphericalAR1Pattern(height, width, self.length_scale_km, phi)

        mask = self._mask(batch_dict, batch, height, width, device)
        draws = {name: pattern.sample(batch, device) for name, pattern in self.patterns.items()}

        for i, field in enumerate(self.fields):
            var = field["var"]
            values = pred[_src(var)][var]
            sigma = getattr(self, f"sigma_map_{i}", None)
            if sigma is None:
                sigma = field["sigma"]
            else:
                self._check_grid(f"the sigma map of {var}", sigma, height, width)
                sigma = sigma.to(device)
            delta = sigma * draws[field["pattern"]]
            if mask is not None:
                delta = delta * mask
            # (B, H, W) -> (B, 1, ..., 1, H, W), then onto the chosen levels (dim 1)
            delta = delta.to(values.dtype).reshape(batch, *([1] * (values.dim() - 3)), height, width)
            if field.get("levels") is not None:
                on_level = torch.zeros(values.shape[1], dtype=values.dtype, device=device)
                on_level[list(field["levels"])] = 1.0
                delta = delta * on_level.reshape(1, -1, *([1] * (values.dim() - 2)))
            pred[_src(var)][var] = values + delta  # out of place: values may be on the autograd graph
        return batch_dict
