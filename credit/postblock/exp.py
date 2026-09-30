import logging
import math

import torch

from credit.postblock.base import BasePostblock
from credit.preblock._utils import (
    _parse_variable_selection,
)  # shared utility — lives in preblock but used by both pre and postblocks

logger = logging.getLogger(__name__)


class ExpTransform(BasePostblock):
    """Inverse of the LogTransform preblock: converts log-space values to physical space.

    Inverts ``y = log_base(x + eps) - log_base(eps)`` back to ``x = base^(y + log_base(eps)) - eps``.

    ``eps`` and ``base`` must match those used in the corresponding LogTransform preblock.

    ``variables`` supports the same shorthand as the scaler: an empty list
    transforms every variable; partial paths (e.g. ``"era5/prognostic"``) expand
    to all variables under that hierarchy. Expansion happens lazily on the first
    forward call.

    Config example::

        type: "exp_transform"
        args:
            variables:
                - "era5/prognostic/3d/Q"
            eps: 1.0e-8      # must match LogTransform eps
            base: "e"        # must match LogTransform base
            max_output: 1.0e+15  # optional overflow guard; null disables

        # or inverse-transform all variables:
        type: "exp_transform"
        args:
            variables: []

    Overflow guard
    --------------
    An untrained model emits normalized values many sigma from the mean, and
    exponentiating those produces finite but astronomical physical values that
    overflow fp32 when a squared-error loss squares them (``|x| > 1.8e19``).
    ``max_output`` caps the exponent so the result never exceeds it, bounding
    the first batches until the model's output scale settles.

    **Set this per config rather than relying on the default.** The default of
    1e15 only guarantees the run does not die: it is deliberately far beyond any
    geophysical field so it can never silently clip real data, but a cap that
    loose still admits enormous gradients just below it, because the gradient of
    a squared error through ``exp`` scales as ``x**2``. For a field whose real
    values are O(10), a cap near 1e3 keeps a large safety margin while holding
    those gradients to something an optimizer (and ``grad_max_norm``) can handle:

    ::

        model z   |grad| @ max_output=1e15   |grad| @ max_output=1e3
        0         1.3e-4                     1.3e-4   (cap inactive)
        2         2.2e8                      0
        4         1.4e24                     0

    The clamp zeroes the gradient of the elements it caps. That is still
    strictly better than the alternative: an overflowing loss is ``inf``, which
    destroys the gradient for *every* element, not just the saturated ones.
    Only the upper end is bounded; small values are untouched.
    """

    _WARN_WINDOW = 100  # forward calls to check for saturation before giving up on warning

    def __init__(
        self,
        variables: list[str],
        eps: float = 1e-8,
        base: str = "e",
        key: str = "y_processed",
        max_output: float | None = 1e15,
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.key = key  # key in batch_dict where Reconstruct writes the split output (default: "y_processed")
        # Saturation bookkeeping: the guard is only expected to engage on the first
        # batches, so the (device-syncing) check runs for a bounded warm-up window
        # and then switches off for the rest of training.
        self._warned = False
        self._calls = 0
        # _eps and _log_eps stored as Python floats — they broadcast to any device
        # without an explicit .to() call, and log(eps) is a constant so we compute it once.
        self._eps = float(eps)

        if base == "e":
            self._log_eps = math.log(self._eps)
        elif base == "2":
            self._log_eps = math.log2(self._eps)
        elif base == "10":
            self._log_eps = math.log10(self._eps)
        else:
            raise ValueError(f"Unsupported base '{base}'. Choose from: 'e', '2', '10'.")

        self._base = base

        # Cap on the exponent argument, so base**arg <= max_output.
        self._max_arg = None
        if max_output is not None:
            # YAML 1.1 only resolves an exponent-form scalar to a float when the
            # exponent carries an explicit sign, so "1.0e6" arrives as a str while
            # "1.0e+6" arrives as a float. Coerce rather than making the config
            # author remember which spelling works.
            try:
                max_output = float(max_output)
            except (TypeError, ValueError):
                raise ValueError(f"max_output must be a number or None, got {max_output!r}.") from None
            if max_output <= 0:
                raise ValueError(f"max_output must be positive or None, got {max_output}.")
            log_of = {"e": math.log, "2": math.log2, "10": math.log10}[base]
            self._max_arg = log_of(float(max_output))
        self._max_output = max_output

    def _exp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base-specific exponentiation: e^x, 2^x, or 10^x."""
        if self._base == "e":
            return torch.exp(x)
        elif self._base == "2":
            return torch.exp2(x)
        else:  # "10"
            return torch.pow(10.0, x)  # torch has no torch.exp10; torch.pow is the standard alternative

    def forward(self, batch_dict: dict) -> dict:
        if not self.variables_expanded:
            # batch_dict[self.key] is {source: {var_key: tensor}} — one level shallower
            # than the preblock's {data_type: {source: {var_key: tensor}}}. Wrap with a
            # dummy key so _parse_variable_selection can traverse it with its standard logic.
            wrapped = {"_": batch_dict[self.key]}
            self.variables = _parse_variable_selection(self.variables, wrapped, data_types=["_"])
            self.variables_expanded = True
        nested = batch_dict[self.key]  # {source: {var_key: tensor}}
        for var_key in self.variables:
            source = var_key.split("/")[0]  # e.g. "era5" from "era5/prognostic/3d/Q"
            if source not in nested or var_key not in nested[source]:
                continue  # variable not present in this batch — skip silently
            y = nested[source][var_key]  # value in log-space (to be inverted to physical space)
            arg = y + self._log_eps
            if self._max_arg is not None:
                if not self._warned and self._calls < self._WARN_WINDOW:
                    n_capped = int((arg > self._max_arg).sum())
                    if n_capped:
                        self._warned = True
                        logger.warning(
                            "ExpTransform: capped %d value(s) of '%s' at max_output=%g. Expected while the "
                            "model is untrained; if it persists past the first epoch the log-space scaler "
                            "statistics or the model output scale need attention.",
                            n_capped,
                            var_key,
                            self._max_output,
                        )
                arg = torch.clamp(arg, max=self._max_arg)
            nested[source][var_key] = self._exp(arg) - self._eps  # x = base^(y + log_base(eps)) - eps
        self._calls += 1
        return batch_dict
