from bridgescaler import load_scaler_dict, scale_var_dict
from bridgescaler.distributed_tensor import DStandardScalerTensor, DQuantileScalerTensor, DMinMaxScalerTensor
from credit.preblock.base import BasePreblock
from os.path import exists, expandvars
from os import makedirs
from ._utils import _parse_variable_selection

_SCALER_REGISTRY = {
    "standard": DStandardScalerTensor,
    "quantile": DQuantileScalerTensor,
    "minmax": DMinMaxScalerTensor,
}


def _combine_scaler_dicts(a, b):
    """Recursively merge two nested scaler dicts by summing matching leaf scalers.

    bridgescaler distributed scalers implement ``__add__`` as a running merge of
    fitted statistics, so summing per-batch (or per-rank) fits yields the combined
    fit over all the underlying data.
    """
    if isinstance(a, dict):
        return {k: _combine_scaler_dicts(a[k], b[k]) for k in a}
    return a + b


def combine_scaler_dicts(scaler_dicts):
    """Combine a list of nested scaler dicts into a single merged scaler dict.

    Args:
        scaler_dicts (list): Nested scaler dicts with identical structure, e.g. the
            per-rank results gathered in ``credit.applications.preprocess``.

    Returns:
        dict: A single nested scaler dict merging the statistics of every input.
    """
    combined = scaler_dicts[0]
    for scaler_dict in scaler_dicts[1:]:
        combined = _combine_scaler_dicts(combined, scaler_dict)
    return combined


class BridgeScalerTransformer(BasePreblock):
    """Scaling preblock using a dictionary of bridgescaler scalers to fit and transform CREDIT state dictionaries.

    Applies per-variable z-score scaling (or its inverse) to tensors in a
    nested batch dict of the form `batch[data_type][source][var_key]`.

    The scaler dict must have been fit with `bridgescaler.scale_var_dict`
    using the same nested structure and saved with `bridgescaler.save_scaler_dict`.

    Example config::

        type: "bridgescaler_transformer"
        args:
            scaler_path: "/path/to/scaler.json"
            variables:
                - "era5/prognostic/3d/T"
                - "era5/prognostic/3d/U"
            method: "transform"
    """

    def __init__(
        self,
        scaler_path: str,
        variables: list[str],
        method: str = "transform",
        scaler_type: str = "standard",
        scaler_params=None,
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.method = method
        self.scaler_path = expandvars(scaler_path)
        if scaler_params is None:
            scaler_params = {}
        self.scaler_params = scaler_params
        # ``scaler`` holds the nested dict of fitted scalers used for transform /
        # inverse_transform (loaded from disk, or accumulated by fit_scaler_batch).
        # ``scaler_template`` is the single unfitted prototype used to fit new data.
        if exists(expandvars(self.scaler_path)):
            self.scaler = load_scaler_dict(scaler_path)
            self.scaler_template = None
        else:
            full_scaler_path = expandvars(self.scaler_path)
            makedirs(full_scaler_path.rsplit("/", 1)[0], exist_ok=True)
            self.scaler = None
            self.scaler_template = _SCALER_REGISTRY[scaler_type](**scaler_params)

    def forward(self, batch: dict) -> dict:
        if not self.variables_expanded:
            self.variables = _parse_variable_selection(self.variables, batch)
            self.variables_expanded = True
        batch = self._copy_batch(batch)
        return scale_var_dict(batch, self.scaler, self.method, self.variables)

    def fit_scaler_batch(self, batch: dict) -> dict:
        """
        Fit the scaler dictionary to a batch of data. This will usually be called with `credit preprocess`. Repeated
        calls to fit_scaler_batch will perform a running average of the scaler parameters.

        Args:
            batch (dict): state dictionary

        Returns:
            dict: The accumulated nested dict of fitted scalers
            (`scaler[data_type][source][var_key]`) merged across every call so far.
            The same dict is stored on `self.scaler`.
        """
        if self.scaler_template is None:
            raise ValueError(
                f"Cannot fit: a scaler already exists at '{self.scaler_path}'. Remove it to refit from scratch."
            )
        if not self.variables_expanded:
            self.variables = _parse_variable_selection(self.variables, batch)
            self.variables_expanded = True
        fitted = scale_var_dict(batch, self.scaler_template, "fit", self.variables)
        # Merge this batch's fit into the running accumulation (running average of
        # the scaler statistics across batches).
        self.scaler = fitted if self.scaler is None else _combine_scaler_dicts(self.scaler, fitted)
        return self.scaler
