from bridgescaler import load_scaler_dict, scale_var_dict
from credit.postblock.base import BasePostblock


class BridgeScalerTransformer(BasePostblock):
    """Scaling postblock using a fitted bridgescaler dict.

    Applies per-variable scaling (or its inverse) to the nested prediction dict
    at ``batch_dict[key]``, which has the form
    ``batch_dict[key][source][var_key]`` where ``var_key`` is
    ``"source/field_type/dim/varname"`` (e.g. ``"era5/prognostic/3d/T"``).

    Defaults to operating on ``"y_processed"`` — the nested dict written by
    ``Reconstruct``. Use ``method="inverse_transform"`` to convert normalized
    predictions back to physical units before physics fixers.

    The scaler dict must have been fit with ``bridgescaler.scale_var_dict``
    using the same nested structure and saved with ``bridgescaler.save_scaler_dict``.

    Example config::

        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables:
                - "era5/prognostic/3d/T"
                - "era5/prognostic/3d/U"
            method: "inverse_transform"
    """

    def __init__(self, scaler_path: str, variables: list[str], method: str, key: str = "y_processed"):
        super().__init__()
        self.variables = variables
        self.method = method
        self.scaler_path = scaler_path
        self.key = key
        self.scaler = load_scaler_dict(scaler_path)

    def forward(self, batch_dict: dict) -> dict:
        batch_dict[self.key] = scale_var_dict(batch_dict[self.key], self.scaler, self.method, self.variables)
        return batch_dict
