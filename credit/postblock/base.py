import torch.nn as nn


class BasePostblock(nn.Module):
    """Base class for all postblocks.

    Subclasses that need to run at every rollout step (e.g. conservation fixers)
    should set ``per_step = True``.  All others leave it at the default ``False``
    and will be called once after the full rollout, on the reconstructed dict.

    Forward signature for all postblocks::

        forward(batch: dict) -> dict

    where ``batch`` is the reconstructed dict returned by ``Reconstruct``::

        {
            "prediction": {var_key: tensor, ...},
            "metadata":   {...},
        }
    """

    per_step: bool = True

    def forward(self, batch: dict) -> dict:
        pass

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)
