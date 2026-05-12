"""
reconstruct.py
--------------
Reconstruct: first postblock that splits the flat ``batch_dict["prediction"]``
tensor back into a nested variable dict in-place.

Reads ``batch_dict["meta"]["_channel_map"]["output"]`` built by
``ConcatToTensor`` to know which channels correspond to which variables.

Input
-----
batch_dict : dict
    Must contain:
      "prediction" — flat model output tensor, shape (B, C, H, W) or (B, C, T, H, W)
      "meta"       — metadata dict with ``_channel_map["output"]``
    May also contain "input", "target", "_raw", etc. — all passed through unchanged.

Output
------
The same ``batch_dict`` with ``"prediction"`` replaced by a nested dict:

    batch_dict["prediction"][source][dataset_type][field_type][dim][var_name]
        -> tensor of shape (B, n_levels, n_time, H, W)
"""

from credit.postblock.base import BasePostblock


class Reconstruct(BasePostblock):
    """Splits ``batch_dict["prediction"]`` from a flat tensor into a nested variable dict.

    Slices are read from ``batch_dict["meta"]["_channel_map"]["output"]``, built
    by ``ConcatToTensor`` and covering only prognostic + diagnostic variables.
    Each slice is unflattened from ``(B, n_levels * n_time, H, W)`` back to
    ``(B, n_levels, n_time, H, W)``. All other keys in ``batch_dict`` pass through.
    """

    def forward(self, batch_dict: dict) -> dict:
        y_pred = batch_dict["prediction"]
        output_map = batch_dict["meta"]["_channel_map"]["output"]

        # Flatten time dim if y_pred arrived as 5D (B, C, T, H, W) — unflatten needs 4D input
        if y_pred.dim() == 5:
            y_pred = y_pred.flatten(1, 2)

        prediction = {}
        for var_key, info in output_map.items():
            ch_slice = info["slice"]
            n_levels, n_time = info["orig_shape"]

            # Slice the flat channel dim: (B, n_levels*n_time, H, W)
            var_tensor = y_pred[:, ch_slice, ...]

            # Restore level and time dims: (B, n_levels, n_time, H, W)
            var_tensor = var_tensor.unflatten(1, (n_levels, n_time))

            # Build nested dict matching input convention: source/dataset_type/field_type/dim/var_name
            source, dataset_type, field_type, dim, var_name = var_key.split("/")
            prediction.setdefault(source, {}).setdefault(dataset_type, {}).setdefault(field_type, {}).setdefault(dim, {})[var_name] = var_tensor

        batch_dict["prediction"] = prediction
        return batch_dict
