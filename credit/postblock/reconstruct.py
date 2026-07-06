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
    Must contain:
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

import os
import torch
from credit.postblock.base import BasePostblock


class Reconstruct(BasePostblock):
    """Splits ``batch_dict["y_pred"]`` into a nested variable dict at ``batch_dict["y_processed"]``.

    Slices are read from ``batch_dict["metadata"]["target"]["_channel_map"]``, built
    by ``ConcatToTensor`` and covering only prognostic + diagnostic variables.
    Each slice is unflattened from ``(B, n_levels * n_time, H, W)`` back to
    ``(B, n_levels, n_time, H, W)``. ``y_pred`` is left untouched. All other
    keys in ``batch_dict`` pass through unchanged.

    Args:
        detach: when True (default) ``y_processed`` is detached from ``y_pred``'s
            autograd graph — appropriate when downstream postblocks only diagnose
            or produce output. Set ``detach=False`` when downstream postblocks
            (e.g. the conservation fixers) must feed corrections back into the
            training loss via ``FlattenToTensor``.
    """

    def __init__(self, detach: bool = True):
        super().__init__()
        self.detach = detach

    def forward(self, batch_dict: dict) -> dict:
        y_pred = batch_dict["y_pred"]
        output_map = batch_dict["metadata"]["target"]["_channel_map"]

        # Flatten time dim if y_pred arrived as 5D (B, C, T, H, W) — unflatten needs 4D input
        if y_pred.dim() == 5:
            y_pred = y_pred.flatten(1, 2)

        y_processed = {}
        for var_key, info in output_map.items():
            ch_slice = info["slice"]
            n_levels, n_time = info["orig_shape"]

            # Slice the flat channel dim: (B, n_levels*n_time, H, W)
            var_tensor = y_pred[:, ch_slice, ...]
            if self.detach:
                var_tensor = var_tensor.detach()

            # Restore level and time dims: (B, n_levels, n_time, H, W)
            var_tensor = var_tensor.unflatten(1, (n_levels, n_time))

            source = var_key.split("/")[0]
            y_processed.setdefault(source, {})[var_key] = var_tensor

        batch_dict["y_processed"] = y_processed
        return batch_dict


class FlattenToTensor(BasePostblock):
    """Rebuild the flat ``y_pred`` tensor from the nested ``y_processed`` dict.

    Inverse of ``Reconstruct``. Concatenates the per-variable tensors back into
    a single ``(B, C, H, W)`` tensor in the channel order given by
    ``batch_dict["metadata"]["target"]["_channel_map"]`` (sorted by slice start),
    so the result matches the model's original ``y_pred`` channel layout.

    Conservation fixers operate on ``y_processed`` in **physical units**, so this
    block forward-scales a copy of ``y_processed`` (via the same bridgescaler
    used elsewhere) before flattening — yielding a **normalized** ``y_pred`` for
    the loss while leaving the physical ``y_processed`` untouched (needed for
    autoregressive rollout assembly). When ``scaler_path`` is omitted the dict is
    flattened as-is (no scaling).

    Args:
        scaler_path: bridgescaler dict path; if None, no scaling is applied.
        variables: variable keys the scaler should transform.
        method: scaler method (default ``"transform"`` — physical -> normalized).
        key: source nested dict (default ``"y_processed"``).
        out_key: flat tensor destination (default ``"y_pred"``).
    """

    def __init__(
        self,
        scaler_path: str | None = None,
        variables: list[str] | None = None,
        method: str = "transform",
        key: str = "y_processed",
        out_key: str = "y_pred",
    ):
        super().__init__()
        self.scaler_path = os.path.expandvars(scaler_path) if scaler_path is not None else None
        self.variables = variables
        self.method = method
        self.key = key
        self.out_key = out_key
        if self.scaler_path is not None:
            from bridgescaler import load_scaler_dict

            self.scaler = load_scaler_dict(self.scaler_path)["target"]
        else:
            self.scaler = None

    def forward(self, batch_dict: dict) -> dict:
        nested = batch_dict[self.key]

        if self.scaler is not None:
            from bridgescaler import scale_var_dict

            # shallow-copy the nested structure so scale_var_dict (which reassigns
            # leaves in place) does not normalize the physical y_processed.
            work = {source: dict(vars_) for source, vars_ in nested.items()}
            work = scale_var_dict(work, self.scaler, self.method, self.variables)
        else:
            work = nested

        channel_map = batch_dict["metadata"]["target"]["_channel_map"]
        pieces = []
        for var_key in sorted(channel_map, key=lambda k: channel_map[k]["slice"].start):
            source = var_key.split("/")[0]
            tensor = work[source][var_key]
            # (B, n_levels, n_time, H, W) -> (B, n_levels * n_time, H, W)
            if tensor.dim() == 5:
                tensor = tensor.flatten(1, 2)
            pieces.append(tensor)

        batch_dict[self.out_key] = torch.cat(pieces, dim=1)
        return batch_dict
