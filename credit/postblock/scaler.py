from os.path import expandvars

from bridgescaler import load_scaler_dict, scale_var_dict
from credit.postblock.base import BasePostblock
from credit.preblock._utils import (
    _parse_variable_selection,
)  # shared utility — lives in preblock but used by both pre and postblocks


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
    the preblock. Use ``scaler_data_type`` to select which data type's statistics
    to apply — typically ``"target"`` for inverse-transforming model output.
    Set ``scaler_data_type=None`` only if the scaler was fit with a flat
    ``{source: {var_key: scaler}}`` structure (no data type split).

    ``variables`` supports the same shorthand as the preblock: an empty list
    scales every variable; partial paths (e.g. ``"era5/prognostic"``) expand to
    all variables under that hierarchy. Expansion happens lazily on the first
    forward call.

    Example config::

        # Inverse-transform specific variables using the target side of a shared scaler
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables:
                - "era5/prognostic/3d/T"
                - "era5/prognostic/3d/U"
            scaler_data_type: "target"   # default — slice to target statistics

        # Inverse-transform all variables
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables: []

        # Use a flat scaler (no data type split)
        type: "bridgescaler_transform"
        args:
            scaler_path: "/path/to/scaler.json"
            variables: []
            scaler_data_type: null
    """

    def __init__(
        self,
        scaler_path: str,
        variables: list[str],
        method: str = "inverse_transform",
        key: str = "y_processed",
        scaler_data_type: str = "target",
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.method = method
        self.scaler_path = expandvars(scaler_path)
        self.key = key  # key in batch_dict where Reconstruct writes the split output (default: "y_processed")
        full_scaler = load_scaler_dict(
            self.scaler_path
        )  # loads the scaler from disk — postblock never fits, only applies
        # Slice to the requested data type so the scaler structure matches {source: {var_key: scaler}},
        # which is one level shallower than the full {data_type: {source: {var_key: scaler}}} structure.
        # Set scaler_data_type=None only for flat scalers without a data type split.
        self.scaler = full_scaler[scaler_data_type] if scaler_data_type is not None else full_scaler

    def forward(self, batch_dict: dict) -> dict:
        if not self.variables_expanded:
            # batch_dict[self.key] is {source: {var_key: tensor}} — one level shallower
            # than the preblock's {data_type: {source: {var_key: tensor}}}. Wrap it with
            # a dummy top-level key so _parse_variable_selection can traverse it with its
            # standard logic.
            wrapped = {"_": batch_dict[self.key]}
            self.variables = _parse_variable_selection(self.variables, wrapped, data_types=["_"])
            self.variables_expanded = True
        batch_dict[self.key] = scale_var_dict(
            batch_dict[self.key], self.scaler, self.method, self.variables
        )  # returns a new dict; reassign back into batch_dict
        return batch_dict
