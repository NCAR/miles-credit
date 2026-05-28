import torch.nn as nn


class BasePostblock(nn.Module):
    """Base class for all postblocks.

    Forward signature for all postblocks::

        forward(batch: dict) -> dict

    where ``batch`` is the reconstructed dict returned by ``Reconstruct``::

        {
            "prediction": {var_key: tensor, ...},
            "metadata":   {...},
        }
    """

    def forward(self, batch: dict) -> dict:
        pass

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)
