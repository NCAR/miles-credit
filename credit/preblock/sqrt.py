from credit.preblock.base import BasePreblock
import torch


class SqrtTransform(BasePreblock):
    """
    Applies a log transformation to specified variables in a batch dict.

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
        for var_key in self.variables:
            source = var_key.split("/")[0]

            if source not in batch:
                raise KeyError(f"SqrtTransform: source '{source}' not found in batch.")

            for data_type in self.data_types:
                # Silent skip: data_type absent (e.g. 'target' during inference)
                if data_type not in batch[source]:
                    continue

                # Silent skip: variable not present in this data_type
                if var_key not in batch[source][data_type]:
                    print(batch[source][data_type].keys())
                    print(f"SqrtTransform: data_type '{data_type}' not found in batch.")
                    continue

                x = batch[source][data_type][var_key]
                # Out-of-place: safe for autograd
                batch[source][data_type][var_key] = torch.sqrt(x)

        return batch
