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

        prediction = {}
        for var_key, info in output_map.items():
            ch_slice = info["slice"]
            n_levels, T = info["orig_shape"]

            # Slice the flat channel dim: (B, n_levels*T, H, W)
            var_tensor = y_pred[:, ch_slice, ...]

            # Restore level and time dims: (B, n_levels, T, H, W)
            var_tensor = var_tensor.unflatten(1, (n_levels, T))

            prediction[var_key] = var_tensor

        return {"prediction": prediction, "metadata": metadata}
