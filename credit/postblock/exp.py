import math

import torch

from credit.postblock.base import BasePostblock
from credit.preblock._utils import (
    _parse_variable_selection,
)  # shared utility — lives in preblock but used by both pre and postblocks


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

        # or inverse-transform all variables:
        type: "exp_transform"
        args:
            variables: []
    """

    def __init__(
        self,
        variables: list[str],
        eps: float = 1e-8,
        base: str = "e",
        key: str = "y_processed",
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.key = key  # key in batch_dict where Reconstruct writes the split output (default: "y_processed")
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
            nested[source][var_key] = self._exp(y + self._log_eps) - self._eps  # x = base^(y + log_base(eps)) - eps
        return batch_dict
