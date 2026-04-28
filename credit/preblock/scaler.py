from bridgescaler import load_scaler_dict, scale_var_dict
from credit.preblock.base import BasePreblock


class BridgeScalerTransformer(BasePreblock):
    """Scaling preblock using a fitted bridgescaler dict.

    Applies per-variable z-score scaling (or its inverse) to tensors in a
    nested batch dict of the form ``batch[source][data_type][var_key]``.

    The scaler dict must have been fit with ``bridgescaler.scale_var_dict``
    using the same nested structure and saved with ``bridgescaler.save_scaler_dict``.

    Example config::

        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables:
                - "era5/prognostic/3d/T"
                - "era5/prognostic/3d/U"
            method: "transform"
    """

    def __init__(self, scaler_path: str, variables: list[str], method: str):
        super().__init__()
        self.variables = variables
        self.method = method
        self.scaler_path = scaler_path
        self.scaler = load_scaler_dict(scaler_path)

    def forward(self, batch: dict) -> dict:
        return scale_var_dict(batch, self.scaler, self.method, self.variables)
