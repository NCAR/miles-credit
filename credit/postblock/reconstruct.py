"""
reconstruct.py
--------------
Reconstruct: first hardcoded postblock that splits a flat model-output tensor
back into a nested dict keyed by variable path.

Reads ``metadata["_channel_map"]["output"]`` built by ``ConcatToTensor`` to
know which channels in ``y_pred`` correspond to which variables and what their
original shape was before flattening.

Input
-----
y_pred   : torch.Tensor, shape (B, C, H, W)
             Flat model output after ``flatten(1, 2)`` collapsed the time dim.
metadata : dict
             Must contain ``metadata["_channel_map"]["output"]``.

Output
------
dict with keys:

    "prediction" : {var_key: tensor}
                   Each tensor has shape (B, n_levels, T, H, W), restoring the
                   original level and time dimensions.
    "metadata"   : the input metadata dict, passed through unchanged.
"""

import torch
from credit.postblock.base import BasePostblock


class Reconstruct(BasePostblock):
    """Splits a flat ``y_pred`` tensor into a per-variable dict.

    Slices are read from ``metadata["_channel_map"]["output"]``, which is built
    by ``ConcatToTensor`` and covers only prognostic + diagnostic variables.
    Each slice is unflattened from ``(B, n_levels * T, H, W)`` back to
    ``(B, n_levels, T, H, W)``.
    """

    def forward(self, y_pred: torch.Tensor, metadata: dict) -> dict:
        output_map = metadata["_channel_map"]["output"]

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

            # Build nested dict matching input convention: source/data_type/dim/var_name
            source, data_type, dim, var_name = var_key.split("/")
            prediction.setdefault(source, {}).setdefault(data_type, {}).setdefault(dim, {})[var_name] = var_tensor

        return {"prediction": prediction, "metadata": metadata}
