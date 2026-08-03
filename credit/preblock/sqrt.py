import torch

from credit.preblock.base import BasePreblock
from credit.preblock._utils import _parse_variable_selection


class SqrtTransform(BasePreblock):
    """Applies a square-root transformation to specified variables in a batch dict.

    Applies ``y = sqrt(x)``. Use ``SquareTransform`` in the postblock to invert
    this (``x = y^2``). Input values must be non-negative; negative values
    produce NaN silently.

    Operates on ``batch[data_type][source][var_key]`` for each requested
    ``data_type`` (default: ``["input", "target"]``).

    ``variables`` supports the same shorthand as the scaler: an empty list
    transforms every variable; partial paths (e.g. ``"era5/prognostic"``) expand
    to all variables under that hierarchy. Expansion happens lazily on the first
    forward call.

    Config example::

        type: "sqrt_transform"
        args:
            variables:
                - "era5/prognostic/3d/Q"
            data_types:     # optional, defaults to ["input", "target"]
                - "input"
                - "target"

        # or transform all variables:
        type: "sqrt_transform"
        args:
            variables: []
    """

    def __init__(self, variables: list[str], data_types: list[str] = None):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.data_types = data_types or ["input", "target"]

        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(
                f"Invalid data_types {invalid}. "
                f"Valid options are {self.VALID_DATA_TYPES}. "
                f"Preblocks never operate on 'metadata'."
            )

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
                    raise KeyError(f"SqrtTransform: source '{source}' not found in batch['{data_type}'].")
                if var_key not in batch[data_type][source]:
                    continue  # variable absent in this data type (e.g. statics only exist in "input")

                x = batch[data_type][source][var_key]
                batch[data_type][source][var_key] = torch.sqrt(x)  # y = sqrt(x)

        return batch
