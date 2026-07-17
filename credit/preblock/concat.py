"""
concat.py
---------
ConcatToTensor: end-of-chain preblock that collapses a nested batch dict into a
flat (x, y, metadata) tuple. Used by build_preblocks/apply_preblocks.

Channel concat order is fully determined by the variable key structure
``{source}/{field_type}/{dim}/{varname}``:

  1. field_type rank: prognostic < dynamic_forcing < static < diagnostic
  2. dim rank: 3d < 2d
  3. within each (field_type, dim) bucket: original insertion order is
     preserved (Python sort is stable), which matches config list order.
"""

import pandas as pd
import torch
from credit.preblock.base import BasePreblock
from credit.datasets.gen_2.channel_layout import FIELD_TYPE_RANK as _FIELD_TYPE_RANK

# Field types the model predicts (used to build the "output" channel map).
_PREDICTABLE_FIELD_TYPES = {"prognostic", "diagnostic"}


def _channel_sort_key(item) -> tuple:
    """Sort key for items from variables.items(): (var_key, tensor).

    var_key has the form ``source/field_type/dim/varname``.
    """
    var_key = item[0]
    parts = var_key.split("/")
    ft = parts[1] if len(parts) > 1 else ""
    dim = parts[2] if len(parts) > 2 else ""
    return (_FIELD_TYPE_RANK.get(ft, len(_FIELD_TYPE_RANK)), 0 if dim == "3d" else 1)


class ConcatToTensor(BasePreblock):
    """End-of-chain preblock that concatenates a nested batch dict of tensors
    into a single input tensor (and optionally a target tensor).

    Expects a batch dict of the form::

        batch[data_type][source][var_name] -> torch.Tensor

    where tensor shapes are (batch, n_levels, time, lat, lon) and concatenation
    is performed along dim=1 (channel). Input tensors are sorted by
    ``_channel_sort_key`` before concatenation so the channel order matches
    the canonical variable schema regardless of insertion order in the batch.

    ``metadata`` keys are passed through as-is (not concatenated).

    In addition to the tensors, two channel maps are attached to metadata:

    * ``metadata["input"]["_channel_map"]``  — every variable and its slice in
      the concatenated input tensor.
    * ``metadata["target"]["_channel_map"]`` — prognostic + diagnostic variables
      only, with slices reindexed from 0 to match ``y_pred`` channel ordering.

    Each entry has the form::

        var_key -> {"slice": slice(start, end), "orig_shape": (n_levels, T)}

    Returns either::

        (input_tensor, metadata)                    # if no "target" data_type present
        (input_tensor, target_tensor, metadata)     # if "target" is present

    Example config::

        type: "concat"
        args:
          to_device: true   # set false to skip .to(device) in apply_preblocks
    """

    def __init__(self, to_device: bool = True):
        super().__init__()
        self.to_device = to_device
        # Optional ChannelSchema (set via set_schema, not config). When present:
        # target-less batches (inference) get the schema's full target map —
        # covering diagnostics — instead of the prognostic-only input-derived
        # fallback; batches WITH a target are validated against the schema once.
        self._schema = None
        self._schema_validated = False

    def set_schema(self, schema) -> None:
        """Attach a ``credit.datasets.gen_2.schema.ChannelSchema`` to this block."""
        self._schema = schema
        self._schema_validated = False

    def forward(self, batch: dict | tuple) -> tuple:
        if isinstance(batch, tuple):
            return batch  # already concatenated — pass through unchanged
        input_tensors = []
        target_tensors = []
        metadata: dict = {"input": {}, "target": {}}

        # Channel-map accumulators
        input_channel_map = {}
        output_channel_map = {}  # built from input — prognostic only (inference fallback)
        target_output_map = {}  # built from target — prognostic + diagnostic
        input_cursor = 0
        output_cursor = 0
        target_cursor = 0

        for data_type, sources in batch.items():
            if data_type == "metadata":
                for source, meta_dict in sources.items():
                    if "input_datetime" in meta_dict:
                        metadata["input"][source] = {"datetime": pd.DatetimeIndex(meta_dict["input_datetime"].numpy())}
                    if "target_datetime" in meta_dict:
                        metadata["target"][source] = {
                            "datetime": pd.DatetimeIndex(meta_dict["target_datetime"].numpy())
                        }
            elif data_type == "input":
                for source, variables in sources.items():
                    for var_key, tensor in sorted(variables.items(), key=_channel_sort_key):
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

                        # Build a separate output channel map for prognostic + diagnostic variables
                        # only. Its cursor starts at 0 because y_pred from the model contains
                        # only these predictable outputs — statics and dynamic forcings are
                        # inputs only and are absent from y_pred.
                        #
                        # This map is the fallback for the target channel map when the batch
                        # carries no target (e.g. rollout with return_target=False). The model
                        # predicts a single step (forecast_len == 1), so the output width is one
                        # per level regardless of the input's time dim. Using the input's T here
                        # would inflate the width by history_len and make Reconstruct unflatten
                        # the single-step y_pred with the wrong shape (crash when history_len > 1).
                        parts = var_key.split("/")
                        if len(parts) >= 2 and parts[1] in _PREDICTABLE_FIELD_TYPES:
                            out_ch = n_levels  # n_levels * 1 output step
                            output_channel_map[var_key] = {
                                "slice": slice(output_cursor, output_cursor + out_ch),
                                "orig_shape": (n_levels, 1),
                            }
                            output_cursor += out_ch

            elif data_type == "target":
                for source, variables in sources.items():
                    # Preserve insertion order so the channel map matches the
                    # order tensors are concatenated into ``y`` (and thus the
                    # model's ``y_pred`` channel order). The target includes
                    # output-only diagnostic variables, which the input does not.
                    for var_key, tensor in variables.items():
                        target_tensors.append(tensor)
                        n_levels, T = tensor.shape[1], tensor.shape[2]
                        n_ch = n_levels * T
                        target_output_map[var_key] = {
                            "slice": slice(target_cursor, target_cursor + n_ch),
                            "orig_shape": (n_levels, T),
                        }
                        target_cursor += n_ch

        if not input_tensors:
            raise ValueError("No 'input' tensors found in batch.")

        metadata["input"]["_channel_map"] = input_channel_map
        # Target map priority:
        #   1. built from an actual target (training/validation) — exact; checked
        #      against the schema once so a layout drift fails loudly, not silently;
        #   2. schema-derived (inference, no target) — covers diagnostics, which
        #      never appear in the input;
        #   3. input-derived prognostic-only map — legacy fallback when no schema
        #      is available; diagnostics will be missing from reconstruction.
        if target_output_map:
            if self._schema is not None and not self._schema_validated:
                self._schema.validate_channel_map(target_output_map, which="target")
                self._schema_validated = True
            metadata["target"]["_channel_map"] = target_output_map
        elif self._schema is not None:
            metadata["target"]["_channel_map"] = self._schema.target_channel_map()
        else:
            metadata["target"]["_channel_map"] = output_channel_map

        # Normalize device: rollout batches mix CPU (dataloader) and accelerator
        # (model output) tensors; torch.cat requires a uniform device.
        accel = next((t.device for t in input_tensors if t.device.type != "cpu"), None)
        if accel is not None:
            input_tensors = [t.to(accel) for t in input_tensors]
        input_tensor = torch.cat(input_tensors, dim=1).float()

        if target_tensors:
            target_tensor = torch.cat(target_tensors, dim=1).float()
            return input_tensor, target_tensor, metadata

        return input_tensor, metadata
