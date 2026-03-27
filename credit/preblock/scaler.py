from bridgescaler import load_scaler_dict
from credit.preblock.base import BasePreblock


class BridgeScaleTransformer(BasePreblock):
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
            data_types:       # optional, defaults to ['input', 'target']
                - "input"
                - "target"
    """

    def __init__(self, scaler_path: str, variables: list[str], method: str, data_types: list[str] = None):

        super().__init__()
        self.variables = variables
        self.data_types = data_types or ["input", "target"]
        self.method = method
        self.scaler_path = scaler_path
        self.scaler = load_scaler_dict(scaler_path)

        # Validate data_types at init
        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(f"Invalid data_types {invalid}. Valid options are {self.VALID_DATA_TYPES}. ")

    def forward(self, batch: dict) -> dict:
        for var_key in self.variables:
            source = var_key.split("/")[0]

            if source not in batch:
                raise KeyError(f"Source '{source}' not found in batch.")

            for data_type in self.data_types:
                # Silent skip: data_type absent (e.g. 'target' during inference)
                if data_type not in batch[source]:
                    continue

                # Silent skip: variable not present in this data_type
                if var_key not in batch[source][data_type]:
                    continue

                if var_key not in self.scaler[source][data_type]:
                    raise KeyError(
                        f"BridgeScaleTransformer: no fitted scaler for '{var_key}' "
                        f"in {source}/{data_type}. Check that the scaler was fit on this variable."
                    )

                x = batch[source][data_type][var_key]
                s = self.scaler[source][data_type][var_key]
                if self.method == "inverse":
                    batch[source][data_type][var_key] = s.inverse_transform(x)
                elif self.method == "transform":
                    batch[source][data_type][var_key] = s.transform(x)
                else:
                    raise ValueError(f"Unsupported method: {self.method}. Choose 'transform' or 'inverse'.")

        return batch
