from credit.preblock.base import BasePreblock
import torch


class SqrtTransform(BasePreblock):
    """
    Applies a sqrt transformation to specified variables in a batch dict.

    Expected dict structure:
        batch[source][data_type]['source/var_type/var_shape/var_name']

    Config example:
        type: "sqrt_transform"
        args:
            variables:
                - 'ERA5/prognostic/3D/Q'
            data_types:     # optional, defaults to ['input', 'target']
                - 'input'
                - 'target'
    """

    def __init__(self, variables: list[str], data_types: list[str] = None):
        super().__init__()
        self.variables = variables
        self.data_types = data_types or ["input", "target"]

        # Validate data_types at init
        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(
                f"Invalid data_types {invalid}. "
                f"Valid options are {self.VALID_DATA_TYPES}. "
                f"Preblocks never operate on 'metadata'."
            )

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)
        for var_key in self.variables:
            source = var_key.split("/")[0]

            for data_type in self.data_types:
                if data_type not in batch:
                    continue
                if source not in batch[data_type]:
                    raise KeyError(f"SqrtTransform: source '{source}' not found in batch['{data_type}'].")
                if var_key not in batch[data_type][source]:
                    continue

                x = batch[data_type][source][var_key]
                batch[data_type][source][var_key] = torch.sqrt(x)

        return batch
