import torch
from credit.preblock.base import BasePreblock


class ConcatenateToTensor(BasePreblock):
    """End-of-chain preblock that concatenates a nested batch dict of tensors
    into a single input tensor (and optionally a target tensor).

    Expects a batch dict of the form::

        batch[source][data_type][var_name] -> torch.Tensor

    where tensor shapes are (batch, channel, time, lon, lat) and concatenation
    is performed along dim=1 (channel). Traversal order follows key insertion
    order: for each source, all var_names under a data_type are concatenated,
    then the next source, and so on.

    ``metadata`` keys are passed through as-is (not concatenated).

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

        for source, data_types in batch.items():
            for data_type, variables in data_types.items():
                if data_type == "metadata":
                    metadata[source] = variables
                elif data_type == "input":
                    for tensor in variables.values():
                        input_tensors.append(tensor)
                elif data_type == "target":
                    for tensor in variables.values():
                        target_tensors.append(tensor)

        if not input_tensors:
            raise ValueError("No 'input' tensors found in batch.")

        input_tensor = torch.cat(input_tensors, dim=1)

        if target_tensors:
            target_tensor = torch.cat(target_tensors, dim=1)
            return input_tensor, target_tensor, metadata

        return input_tensor, metadata
