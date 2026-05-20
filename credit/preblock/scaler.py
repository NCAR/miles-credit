from bridgescaler import load_scaler_dict, scale_var_dict
from bridgescaler.distributed_tensor import DStandardScalerTensor, DQuantileScalerTensor, DMinMaxScalerTensor
from credit.preblock.base import BasePreblock
from os.path import exists, expandvars
from os import makedirs

_SCALER_REGISTRY = {
    "standard": DStandardScalerTensor,
    "quantile": DQuantileScalerTensor,
    "minmax": DMinMaxScalerTensor,
}


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

    def __init__(
        self, scaler_path: str, variables: list[str], method: str, scaler_type: str = "standard", scaler_params=None
    ):
        super().__init__()
        self.variables = variables
        self.method = method
        self.scaler_path = scaler_path
        if scaler_params is None:
            scaler_params = {}
        self.scaler_params = scaler_params
        if exists(expandvars(self.scaler_path)):
            self.scaler = load_scaler_dict(scaler_path)
        else:
            full_scaler_path = expandvars(self.scaler_path)
            makedirs(full_scaler_path.rsplit("/", 1)[0], exist_ok=True)
            self.scaler = _SCALER_REGISTRY[scaler_type](**scaler_params)

    def forward(self, batch: dict) -> dict:
        return scale_var_dict(batch, self.scaler, self.method, self.variables)

    def fit_scaler_batch(self, batch: dict) -> dict:
        return scale_var_dict(batch, self.scaler, "fit", self.variables)
