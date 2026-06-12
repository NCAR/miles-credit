import torch

from credit.preblock.base import BasePreblock
from ._utils import _parse_variable_selection


class FillNan(BasePreblock):
    """Replace NaN values with a constant fill value for selected variables.

    Walks a nested batch dict of the form ``batch[data_type][source][var_key]``
    and replaces any NaN entries with ``fill_value`` for the selected variables.
    Only NaNs are touched; ``+/-inf`` and finite values are left unchanged.

    ``variables`` may contain full or partial names (e.g. ``"era5/prognostic/3d"``
    expands to every variable beneath it) and is expanded against the first batch
    via :func:`credit.preblock._utils._parse_variable_selection`. An empty or
    omitted ``variables`` list means "every variable". Specifying a subset lets
    several ``FillNan`` preblocks coexist, each filling different variables with a
    different value.

    Config example::

        type: "fill_nan"
        args:
            variables:                  # optional, defaults to all variables
                - "era5/prognostic/3d/Q"
            data_types:                 # optional, defaults to ['input', 'target']
                - "input"
                - "target"
            fill_value: 0.0             # optional, default 0.0
    """

    def __init__(
        self,
        variables: list[str] = None,
        data_types: list[str] = None,
        fill_value: float = 0.0,
    ):
        super().__init__()
        self.variables = variables or []
        self.variables_expanded = False
        self.data_types = data_types or ["input", "target"]
        self.fill_value = fill_value

        # Validate data_types at init
        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(f"Invalid data_types {invalid}. Valid options are {self.VALID_DATA_TYPES}. ")

    def forward(self, batch: dict) -> dict:
        if not self.variables_expanded:
            self.variables = _parse_variable_selection(self.variables, batch, self.data_types)
            self.variables_expanded = True

        for var_key in self.variables:
            source = var_key.split("/")[0]

            for data_type in self.data_types:
                if data_type not in batch:
                    continue
                if source not in batch[data_type]:
                    continue
                if var_key not in batch[data_type][source]:
                    continue

                x = batch[data_type][source][var_key]
                batch[data_type][source][var_key] = torch.where(
                    torch.isnan(x),
                    torch.tensor(self.fill_value, dtype=x.dtype, device=x.device),
                    x,
                )

        return batch
