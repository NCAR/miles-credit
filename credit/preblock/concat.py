"""
concat.py
---------
ConcatPreblock: assembles per-variable tensors from MultiSourceDataset batches
into model-ready x (input) and y (target) tensors.

Batch structure from MultiSourceDataset (after DataLoader collation)::

    batch = {
        "era5": {
            "input":  {"era5/prognostic/3d/temperature": (B, n_levels, T, H, W), ...},
            "target": {"era5/prognostic/3d/temperature": (B, n_levels, T, H, W), ...},
            "metadata": {...},
        },
        ...  # other sources (MRMS, etc.) if configured
    }

After ConcatPreblock::

    batch["x"]: (B, total_input_vars,  T, H, W)
    batch["y"]: (B, total_target_vars, T, H, W)

Assembly order:
    x: prognostic/3d, prognostic/2d, dynamic_forcing/2d, static/2d
    y: prognostic/3d, prognostic/2d, diagnostic/2d

Within each (field_type, dim) group, insertion order (= config order) is preserved.
"""

import torch
import torch.nn as nn

# Priority order for field_types in assembled x and y
_INPUT_FIELD_ORDER = ["prognostic", "dynamic_forcing", "static"]
_TARGET_FIELD_ORDER = ["prognostic", "diagnostic"]
_DIM_ORDER = ["3d", "2d"]


def _sort_key(key: str, field_order: list) -> tuple:
    """Return (field_priority, dim_priority) for a key of the form
    '{source}/{field_type}/{dim}/{varname}'."""
    parts = key.split("/")
    if len(parts) >= 4:
        field_type, dim = parts[1], parts[2]
    else:
        return (999, 999)
    fp = field_order.index(field_type) if field_type in field_order else 999
    dp = _DIM_ORDER.index(dim) if dim in _DIM_ORDER else 999
    return (fp, dp)


def _assemble(tensor_dict: dict, field_order: list) -> torch.Tensor:
    """Concatenate tensors from *tensor_dict* along dim=1 in field/dim priority order,
    preserving variable insertion order within each (field_type, dim) group."""
    groups: dict[tuple, list] = {}
    for key, tensor in tensor_dict.items():
        sk = _sort_key(key, field_order)
        groups.setdefault(sk, []).append(tensor)
    parts = []
    for sk in sorted(groups.keys()):
        parts.extend(groups[sk])
    return torch.cat(parts, dim=1)


class ConcatPreblock(nn.Module):
    """Assembles per-variable batch tensors into model-ready ``x`` and ``y``.

    Iterates over all source sub-dicts in the batch (e.g. ``batch["era5"]``),
    concatenates ``input`` tensors → ``batch["x"]`` and ``target`` tensors →
    ``batch["y"]`` along the variable/channel dimension (dim=1).

    Input/target tensors are expected to have shape ``(B, levels_or_1, T, H, W)``.
    Output ``x`` / ``y`` have shape ``(B, total_vars, T, H, W)``.
    """

    def forward(self, batch: dict) -> dict:
        x_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []

        for source_data in batch.values():
            if not isinstance(source_data, dict):
                continue
            if "input" in source_data:
                x_parts.append(_assemble(source_data["input"], _INPUT_FIELD_ORDER))
            if "target" in source_data:
                y_parts.append(_assemble(source_data["target"], _TARGET_FIELD_ORDER))

        if x_parts:
            batch["x"] = torch.cat(x_parts, dim=1) if len(x_parts) > 1 else x_parts[0]
        if y_parts:
            batch["y"] = torch.cat(y_parts, dim=1) if len(y_parts) > 1 else y_parts[0]

        return batch


