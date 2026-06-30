import math

import torch

from credit.preblock.base import BasePreblock
from credit.preblock._utils import _parse_variable_selection


class LogTransform(BasePreblock):
    """Applies a log transformation with an eps offset to specified variables in a batch dict.

    Applies ``y = log_base(x + eps) - log_base(eps)`` so that ``y = 0`` when
    ``x = 0``, regardless of ``eps``. Use ``ExpTransform`` in the postblock to
    invert this with matching ``base`` and ``eps``. Input values should satisfy
    ``x >= -eps``; values below this produce NaN silently.

    Operates on ``batch[data_type][source][var_key]`` for each requested
    ``data_type`` (default: ``["input", "target"]``).

    ``variables`` supports the same shorthand as the scaler: an empty list
    transforms every variable; partial paths (e.g. ``"era5/prognostic"``) expand
    to all variables under that hierarchy. Expansion happens lazily on the first
    forward call.

    Config example::

        type: "log_transform"
        args:
            variables:
                - "era5/prognostic/3d/Q"
            data_types:     # optional, defaults to ["input", "target"]
                - "input"
                - "target"
            base: "e"       # optional, default "e". Options: "e", "2", "10"
            eps: 1.0e-8     # optional, default 1e-8

        # or transform all variables:
        type: "log_transform"
        args:
            variables: []
    """

    def __init__(
        self,
        variables: list[str],
        data_types: list[str] = None,
        base: str = "e",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.data_types = data_types or ["input", "target"]
        self._eps = float(eps)  # Python float — broadcasts to any device without an explicit .to() call

        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(
                f"Invalid data_types {invalid}. "
                f"Valid options are {self.VALID_DATA_TYPES}. "
                f"Preblocks never operate on 'metadata'."
            )

        self._base = base  # stored for dispatch in _log()
        # log(eps) is a constant for a given base, so compute it once here rather than every forward call.
        if base == "e":
            self._log_eps = math.log(self._eps)
        elif base == "2":
            self._log_eps = math.log2(self._eps)
        elif base == "10":
            self._log_eps = math.log10(self._eps)
        else:
            raise ValueError(f"Unsupported log base '{base}'. Choose from: 'e', '2', '10'.")

    def _log(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base-specific logarithm: log_e(x), log_2(x), or log_10(x)."""
        if self._base == "e":
            return torch.log(x)
        elif self._base == "2":
            return torch.log2(x)
        else:  # "10"
            return torch.log10(x)

    def forward(self, batch: dict) -> dict:
        if not self.variables_expanded:
            self.variables = _parse_variable_selection(self.variables, batch, self.data_types)
            self.variables_expanded = True
        batch = self._copy_batch(batch)  # shallow copy — avoids mutating the caller's dict
        for var_key in self.variables:
            source = var_key.split("/")[0]  # e.g. "era5" from "era5/prognostic/3d/Q"

            for data_type in self.data_types:
                if data_type not in batch:
                    continue  # data type absent in this batch (e.g. no "target" during inference)
                if source not in batch[data_type]:
                    raise KeyError(f"LogTransform: source '{source}' not found in batch['{data_type}'].")
                if var_key not in batch[data_type][source]:
                    continue  # variable absent in this data type (e.g. statics only exist in "input")

                x = batch[data_type][source][var_key]
                batch[data_type][source][var_key] = (
                    self._log(x + self._eps) - self._log_eps
                )  # y = log_base(x + eps) - log_base(eps)

        return batch
