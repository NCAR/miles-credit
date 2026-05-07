import torch.nn as nn


class BasePreblock(nn.Module):
    """
    Base class for all preblocks. Enforces the forward signature and
    provides the from_config classmethod used by the registry.
    """

    VALID_DATA_TYPES = ("input", "target")

    def forward(self, batch: dict) -> dict:
        pass

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)
