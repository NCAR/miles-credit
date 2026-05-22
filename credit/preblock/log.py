from credit.preblock.base import BasePreblock
import torch


class LogTransform(BasePreblock):
    """
    Applies a log transformation to specified variables in a batch dict.

    Expected dict structure:
        batch[source][data_type]['source/var_type/var_shape/var_name']

    Config example:
        type: "log_transform"
        args:
            variables:
                - 'ERA5/prognostic/3D/Q'
            data_types:     # optional, defaults to ['input', 'target']
                - 'input'
                - 'target'
            base: 'e'       # optional, default 'e'. Options: 'e', '2', '10'
            eps: 1.0e-8     # optional, default 1e-8
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
        self.data_types = data_types or ["input", "target"]
        self.eps = torch.tensor(eps)

        # Validate data_types at init
        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(f"Invalid data_types {invalid}. Valid options are {self.VALID_DATA_TYPES}. ")

        if base == "e":
            self._log_fn = torch.log
        elif base == "2":
            self._log_fn = torch.log2
        elif base == "10":
            self._log_fn = torch.log10
        else:
            raise ValueError(f"Unsupported log base '{base}'. Choose from: 'e', '2', '10'.")

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)
        for var_key in self.variables:
            source = var_key.split("/")[0]

            for data_type in self.data_types:
                if data_type not in batch:
                    continue
                if source not in batch[data_type]:
                    raise KeyError(f"LogTransform: source '{source}' not found in batch['{data_type}'].")
                if var_key not in batch[data_type][source]:
                    continue

                x = batch[data_type][source][var_key]
                batch[data_type][source][var_key] = self._log_fn(x + self.eps) - self._log_fn(self.eps)

        return batch
