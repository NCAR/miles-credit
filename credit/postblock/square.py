import torch

from credit.postblock.base import BasePostblock
from credit.preblock._utils import (
    _parse_variable_selection,
)  # shared utility — lives in preblock but used by both pre and postblocks


class SquareTransform(BasePostblock):
    """Inverse of the SqrtTransform preblock: converts sqrt-space values to physical space.

    Inverts ``y = sqrt(x)`` back to ``x = y^2``.

    Note: values may be slightly negative due to floating-point effects;
    squaring maps these to small positive values, preserving the non-negative physical
    constraint for quantities such as specific humidity or precipitation.

    ``variables`` supports the same shorthand as the scaler: an empty list
    transforms every variable; partial paths (e.g. ``"era5/prognostic"``) expand
    to all variables under that hierarchy. Expansion happens lazily on the first
    forward call.

    Config example::

        type: "square_transform"
        args:
            variables:
                - "era5/prognostic/3d/Q"

        # or inverse-transform all variables:
        type: "square_transform"
        args:
            variables: []
    """

    def __init__(self, variables: list[str], key: str = "y_processed"):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.key = key  # key in batch_dict where Reconstruct writes the split output (default: "y_processed")

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
            y = nested[source][var_key]  # value in sqrt-space (to be inverted to physical space)
            nested[source][var_key] = torch.square(y)  # x = y^2
        return batch_dict
