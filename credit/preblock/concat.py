import torch
from credit.preblock.base import BasePreblock

# Field types the model predicts (used to build the "output" channel map).
_PREDICTABLE_FIELD_TYPES = {"prognostic", "diagnostic"}


class ConcatToTensor(BasePreblock):
    """End-of-chain preblock that concatenates a nested batch dict of tensors
    into a single input tensor (and optionally a target tensor).

    Expects a batch dict of the form::

        batch[source][data_type][var_name] -> torch.Tensor

    where tensor shapes are (batch, n_levels, time, lat, lon) and concatenation
    is performed along dim=1 (channel). Traversal order follows key insertion
    order: for each source, all var_names under a data_type are concatenated,
    then the next source, and so on.

    ``metadata`` keys are passed through as-is (not concatenated).

    In addition to the tensors, two channel maps are attached to metadata under
    ``metadata["_channel_map"]``:

    * ``"input"``  — every variable and its slice in the concatenated input tensor.
    * ``"output"`` — prognostic + diagnostic variables only, with slices
      reindexed from 0 to match ``y_pred`` channel ordering.

    Each entry has the form::

        var_key -> {"slice": slice(start, end), "orig_shape": (n_levels, T)}

    Returns either::

        (input_tensor, metadata)                    # if no "target" data_type present
        (input_tensor, target_tensor, metadata)     # if "target" is present

    Example config::

        type: "concatenate_to_tensor"
        args: {}
    """

    def forward(self, batch):
        input_tensors = []
        target_tensors = []
        metadata = {}

        # Channel-map accumulators
        input_channel_map = {}
        output_channel_map = {}
        input_cursor = 0
        output_cursor = 0

        for source, data_types in batch.items():
            for data_type, variables in data_types.items():
                if data_type == "metadata":
                    metadata[source] = variables
                elif data_type == "input":
                    for var_key, tensor in variables.items():
                        input_tensors.append(tensor)
                        # tensor shape: (B, n_levels, T, H, W)
                        n_levels, T = tensor.shape[1], tensor.shape[2]
                        n_ch = n_levels * T
                        entry = {
                            "slice": slice(input_cursor, input_cursor + n_ch),
                            "orig_shape": (n_levels, T),
                        }
                        input_channel_map[var_key] = entry
                        input_cursor += n_ch

                        # Also track predictable variables for the output map
                        # Key format: source/field_type/dim/varname
                        parts = var_key.split("/")
                        if len(parts) >= 2 and parts[1] in _PREDICTABLE_FIELD_TYPES:
                            output_channel_map[var_key] = {
                                "slice": slice(output_cursor, output_cursor + n_ch),
                                "orig_shape": (n_levels, T),
                            }
                            output_cursor += n_ch

                elif data_type == "target":
                    for tensor in variables.values():
                        target_tensors.append(tensor)

        if not input_tensors:
            raise ValueError("No 'input' tensors found in batch.")

        metadata["_channel_map"] = {
            "input": input_channel_map,
            "output": output_channel_map,
        }

        input_tensor = torch.cat(input_tensors, dim=1)

        if target_tensors:
            target_tensor = torch.cat(target_tensors, dim=1)
            return input_tensor, target_tensor, metadata

        return input_tensor, metadata
