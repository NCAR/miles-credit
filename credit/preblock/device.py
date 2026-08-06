"""Device placement for nested preblock state dictionaries."""

from collections.abc import Mapping

import torch

from credit.preblock.base import BasePreblock


def _move_to_device(value, device):
    """Recursively move tensors while preserving nested container structure."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


class ToDevice(BasePreblock):
    """Move every tensor in a nested full state dictionary to a device.

    This preblock is useful before GPU-only or accelerator-aware preblocks. It
    recursively visits all dictionaries, lists, and tuples in the state and
    leaves non-tensor metadata unchanged. Tensors already on the requested
    device are returned without a copy by PyTorch.

    Config example::

        preblocks:
          per_step:
            to_device:
              type: to_device
              args:
                device: cuda

    ``device`` may be any value accepted by :class:`torch.device`, including
    ``"cuda"``, ``"cuda:1"``, ``"mps"``, or ``"cpu"``. Place this block
    before preblocks that perform tensor operations on the full nested state.
    """

    def __init__(self, device: str | torch.device):
        super().__init__()
        try:
            self.device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"ToDevice: invalid device {device!r}.") from exc

    def forward(self, batch: dict) -> dict:
        return _move_to_device(batch, self.device)
