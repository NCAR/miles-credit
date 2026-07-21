"""
reconstruct.py
--------------
Reconstruct: first postblock that splits the flat ``batch_dict["y_pred"]``
tensor into a nested variable dict, writing the result to ``batch_dict["y_processed"]``.

Reads ``batch_dict["metadata"]["target"]["_channel_map"]`` built by
``ConcatToTensor`` to know which channels correspond to which variables.

Input
-----
batch_dict : dict
    Must contain::

      "y_pred"   — flat model output tensor, shape (B, C, H, W) or (B, C, T, H, W)
      "metadata" — metadata dict with ``["target"]["_channel_map"]``

    All other keys pass through unchanged.

Output
------
The same ``batch_dict`` with ``"y_processed"`` added as a nested dict:

    batch_dict["y_processed"][source][var_key]
        -> tensor of shape (B, n_levels, n_time, H, W)

``"y_pred"`` is left intact (grad-attached) for use in loss computation.
"""

from credit.postblock.base import BasePostblock


class Reconstruct(BasePostblock):
    """Splits ``batch_dict["y_pred"]`` into a nested variable dict at ``batch_dict["y_processed"]``.

    This must be the **first postblock** in the chain — all subsequent postblocks
    (scalers, transforms, physics fixers) read from ``batch_dict["y_processed"]``.

    Slices are read from ``batch_dict["metadata"]["target"]["_channel_map"]``, built
    by ``ConcatToTensor`` and covering only prognostic + diagnostic variables.
    Each slice is unflattened from ``(B, n_levels * n_time, H, W)`` back to
    ``(B, n_levels, n_time, H, W)``. ``y_pred`` is left untouched for loss
    computation. All other keys in ``batch_dict`` pass through unchanged.

    Config example::

        type: "reconstruct"
    """

    def forward(self, batch_dict: dict) -> dict:
        y_pred = batch_dict["y_pred"]
        output_map = batch_dict["metadata"]["target"]["_channel_map"]

        # Merge channel and time dims into a single channel dim if y_pred arrived as 5D (B, C, T, H, W)
        if y_pred.dim() == 5:
            y_pred = y_pred.flatten(1, 2)

        y_processed = {}
        for var_key, info in output_map.items():
            ch_slice = info["slice"]
            n_levels, n_time = info["orig_shape"]

            # Slice the flat channel dim: (B, n_levels*n_time, H, W)
            var_tensor = y_pred[
                :, ch_slice, ...
            ].detach()  # detach so postblock transforms don't affect gradients on y_pred

            # Restore level and time dims: (B, n_levels, n_time, H, W)
            var_tensor = var_tensor.unflatten(1, (n_levels, n_time))

            source = var_key.split("/")[0]  # e.g. "era5" from "era5/prognostic/3d/T"
            y_processed.setdefault(source, {})[var_key] = (
                var_tensor  # group by source, creating the sub-dict on first encounter
            )

        batch_dict["y_processed"] = y_processed
        return batch_dict
