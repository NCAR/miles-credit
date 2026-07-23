import torch
from bridgescaler import load_scaler_dict, scale_var_dict
from bridgescaler.distributed_tensor import DStandardScalerTensor, DQuantileScalerTensor, DMinMaxScalerTensor
from credit.preblock.base import BasePreblock
from os.path import exists, expandvars
from os import makedirs
from ._utils import _parse_variable_selection, _flatten_spatial_tensors, _unflatten_spatial_tensors

_SCALER_REGISTRY = {
    "standard": DStandardScalerTensor,
    "quantile": DQuantileScalerTensor,
    "minmax": DMinMaxScalerTensor,
}


def _move_leaf_scaler_to_cpu(scaler):
    """Move all tensor attributes of a single fitted scaler object to CPU in-place."""
    for attr, val in vars(scaler).items():
        if torch.is_tensor(val):
            setattr(scaler, attr, val.cpu())
        elif isinstance(val, dict):
            for k, v in val.items():
                if torch.is_tensor(v):
                    val[k] = v.cpu()
    return scaler


def move_scaler_dict_to_cpu(scaler_dict):
    """Recursively move all tensor statistics in a nested scaler dict to CPU.

    Required before ``gather_object`` in multi-GPU runs: each rank's scaler
    holds tensors on its own device, and combining scalers from different
    devices raises a device-mismatch error.
    """
    if isinstance(scaler_dict, dict):
        return {k: move_scaler_dict_to_cpu(v) for k, v in scaler_dict.items()}
    return _move_leaf_scaler_to_cpu(scaler_dict)


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


class BridgeScalerTransform(BasePreblock):
    """Scaling preblock using a dictionary of bridgescaler scalers to fit and transform CREDIT state dictionaries.

    Applies per-variable z-score scaling (or its inverse) to tensors in a
    nested batch dict of the form ``batch[data_type][source][var_key]``.

    The scaler dict mirrors the batch structure — it is organized as
    ``{data_type: {source: {var_key: scaler}}}``. Variables that only exist in
    one data type (e.g. statics and dynamic forcings are ``"input"`` only) are
    therefore only scaled in that data type; no special configuration is needed.
    The scaler dict is produced by running ``credit preprocess``.

    ``variables`` accepts full variable keys (e.g. ``"era5/prognostic/3d/T"``),
    partial paths (e.g. ``"era5/prognostic"`` — expands to all variables whose
    key starts with that prefix), or an empty list (expands to all variables).
    The expansion is resolved lazily against the actual batch on the first
    forward call.

    ``data_types`` scopes which data types this scaler instance processes. Its
    main use case is multi-step rollout training, where the input is already
    scaled from the previous step and only the target needs to be scaled. Omit
    it (or pass ``["input", "target"]``) to scale both sides as usual.

    Config examples::

        # Scale specific variables in both input and target (default behaviour)
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables:
                - "era5/prognostic/3d/T"
                - "era5/prognostic/3d/U"
            method: "transform"

        # Scale all variables (input and target) by leaving variables empty
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables: []
            method: "transform"

        # Scale all prognostic variables using a partial path
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables:
                - "era5/prognostic"
            method: "transform"

        # Multi-step rollout: scale only the target (input already scaled)
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables: []
            method: "transform"
            data_types:
                - "target"

        # Grid-wise scaling: variables in `spatial_variables` are scaled per
        # gridpoint (one mean/variance per lat/lon cell) rather than per level.
        # Their scaler's x_columns_/mean_x_/var_x_ must have length == n_lat * n_lon.
        # `spatial_variables` entries must also be covered by `variables`.
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables: []
            method: "transform"
            spatial_variables:
                - "cesm/prognostic/2d/some_var"
    """

    def __init__(
        self,
        scaler_path: str,
        variables: list[str],
        method: str = "transform",
        scaler_type: str = "standard",
        scaler_params=None,
        data_types: list[str] = None,
        spatial_variables: list[str] = None,
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.method = method
        self.scaler_path = expandvars(scaler_path)
        self.data_types = data_types
        # Full keys or partial paths of variables that use grid-wise (per-gridpoint)
        # scaling instead of per-level scaling. Must also be selected by `variables`.
        self.spatial_variables = spatial_variables or []
        if scaler_params is None:
            scaler_params = {}
        self.scaler_params = scaler_params
        # ``scaler`` holds the nested dict of fitted scalers used for transform /
        # inverse_transform (loaded from disk, or accumulated by fit_scaler_batch).
        # ``scaler_template`` is the single unfitted prototype used to fit new data.
        if exists(self.scaler_path):
            self.scaler = load_scaler_dict(self.scaler_path)
            self.scaler_template = None
        else:
            full_scaler_path = self.scaler_path
            makedirs(full_scaler_path.rsplit("/", 1)[0], exist_ok=True)
            self.scaler = None
            self.scaler_template = _SCALER_REGISTRY[scaler_type](**scaler_params)

    def _expand_variables(self, batch: dict) -> None:
        self.variables = _parse_variable_selection(self.variables, batch, self.data_types)
        if self.spatial_variables:
            self.spatial_variables = _parse_variable_selection(self.spatial_variables, batch, self.data_types)
            missing = set(self.spatial_variables) - set(self.variables)
            if missing:
                raise ValueError(f"spatial_variables must also be selected by `variables` (missing: {missing}).")
        self.variables_expanded = True

    def forward(self, batch: dict) -> dict:
        if not self.variables_expanded:
            self._expand_variables(batch)
        batch = self._copy_batch(batch)  # shallow copy — avoids mutating the caller's dict
        if self.data_types is not None:
            # Slice to the requested data types — useful in multi-step training where
            # the input is already scaled but the target still needs to be scaled.
            sub_batch = {dt: batch[dt] for dt in self.data_types if dt in batch}
            sub_batch, spatial_shapes = _flatten_spatial_tensors(sub_batch, self.spatial_variables)
            scaled = scale_var_dict(sub_batch, self.scaler, self.method, self.variables)
            scaled = _unflatten_spatial_tensors(scaled, spatial_shapes)
            batch.update(scaled)  # write the scaled data types back into the full batch
            return batch
        batch, spatial_shapes = _flatten_spatial_tensors(batch, self.spatial_variables)
        scaled = scale_var_dict(batch, self.scaler, self.method, self.variables)
        return _unflatten_spatial_tensors(scaled, spatial_shapes)

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
            self._expand_variables(batch)
        # Fit only the requested data types; if data_types is None, fit the full batch.
        fit_batch = {dt: batch[dt] for dt in self.data_types if dt in batch} if self.data_types is not None else batch
        fit_batch, _ = _flatten_spatial_tensors(fit_batch, self.spatial_variables)
        fitted = scale_var_dict(fit_batch, self.scaler_template, "fit", self.variables)
        # Merge this batch's fit into the running accumulation (running average of
        # the scaler statistics across batches).
        self.scaler = fitted if self.scaler is None else _combine_scaler_dicts(self.scaler, fitted)
        return self.scaler
