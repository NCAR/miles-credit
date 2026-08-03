from os.path import expandvars

from bridgescaler import load_scaler_dict, scale_var_dict
from credit.postblock.base import BasePostblock
from credit.preblock._utils import (
    _parse_variable_selection,
    _flatten_spatial_tensors,
    _unflatten_spatial_tensors,
)  # shared utilities — live in preblock but used by both pre and postblocks


class BridgeScalerTransform(BasePostblock):
    """Scaling postblock using a fitted bridgescaler dict.

    Applies per-variable scaling to the nested output dict at ``batch_dict[key]``
    (default: ``"y_processed"``), which has the form ``{source: {var_key: tensor}}``
    where ``var_key`` is ``"source/field_type/dim/varname"`` (e.g. ``"era5/prognostic/3d/T"``).
    Typically used after ``Reconstruct`` to convert normalized values back to
    physical units before physics fixers.

    ``method`` controls which bridgescaler operation to apply (default:
    ``"inverse_transform"``). Any method name supported by the fitted scaler objects
    is valid (e.g. ``"transform"`` to scale instead of unscale).

    ``key`` selects which entry in ``batch_dict`` to operate on (default:
    ``"y_processed"``). Override if a different postblock writes its output under
    a different key.

    The same ``scaler.json`` produced by ``credit preprocess`` (which has the
    structure ``{data_type: {source: {var_key: scaler}}}``) can be shared with
    the preblock. The ``"target"`` slice is always used for inverse-transforming
    model output.

    ``variables`` supports the same shorthand as the preblock: an empty list
    scales every variable; partial paths (e.g. ``"era5/prognostic"``) expand to
    all variables under that hierarchy. Expansion happens lazily on the first
    forward call.

    ``spatial_variables`` scales the listed variables per gridpoint (one
    mean/variance per lat/lon cell) instead of per level — see the preblock's
    docstring for the full explanation. Entries must also be covered by
    ``variables``.

    Example config::

        # Inverse-transform specific variables
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables:
                - "era5/prognostic/3d/T"
                - "era5/prognostic/3d/U"

        # Inverse-transform all variables
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables: []

        # Inverse-transform with a grid-wise variable
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables: []
            spatial_variables:
                - "cesm/prognostic/2d/some_var"
    """

    def __init__(
        self,
        scaler_path: str,
        variables: list[str],
        method: str = "inverse_transform",
        key: str = "y_processed",
        spatial_variables: list[str] = None,
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.method = method
        self.scaler_path = expandvars(scaler_path)
        self.key = key  # key in batch_dict where Reconstruct writes the split output (default: "y_processed")
        # Full keys or partial paths of variables that use grid-wise (per-gridpoint)
        # scaling instead of per-level scaling. Must also be selected by `variables`.
        self.spatial_variables = spatial_variables or []
        full_scaler = load_scaler_dict(self.scaler_path)
        self.scaler = full_scaler["target"]

    def forward(self, batch_dict: dict) -> dict:
        if not self.variables_expanded:
            # batch_dict[self.key] is {source: {var_key: tensor}} — one level shallower
            # than the preblock's {data_type: {source: {var_key: tensor}}}. Wrap it with
            # a dummy top-level key so _parse_variable_selection can traverse it with its
            # standard logic.
            wrapped = {"_": batch_dict[self.key]}
            self.variables = _parse_variable_selection(self.variables, wrapped, data_types=["_"])
            if self.spatial_variables:
                self.spatial_variables = _parse_variable_selection(self.spatial_variables, wrapped, data_types=["_"])
                missing = set(self.spatial_variables) - set(self.variables)
                if missing:
                    raise ValueError(f"spatial_variables must also be selected by `variables` (missing: {missing}).")
            self.variables_expanded = True
        flattened, spatial_shapes = _flatten_spatial_tensors(batch_dict[self.key], self.spatial_variables)
        scaled = scale_var_dict(flattened, self.scaler, self.method, self.variables)
        batch_dict[self.key] = _unflatten_spatial_tensors(
            scaled, spatial_shapes
        )  # returns a new dict; reassign back into batch_dict
        return batch_dict
