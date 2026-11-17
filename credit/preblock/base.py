import torch.nn as nn


class BasePreblock(nn.Module):
    """
    Base class for all preblocks. Enforces the forward signature and
    provides the from_config classmethod used by the registry.
    """

    VALID_DATA_TYPES = ("input", "target")

    # True for a preblock that changes the variance of the variables it transforms
    # (log, sqrt, ...) and is undone by a postblock, so the loss scores them in
    # physical units. ``credit preprocess`` then records the physical-unit
    # statistics of its target variables (``fit_physical_target_stats``); such a
    # block must expose ``variables`` and ``data_types``. The inverse postblock is
    # listed in ``credit.losses.base.UNIT_TRANSFORMS``.
    changes_units = False

    def _copy_batch(self, batch: dict) -> dict:
        """Shallow-copy the batch so forward() never mutates the caller's dict.

        Creates new dict objects at the data_type and source levels; tensor
        values are shared (not copied) since preblocks must not mutate tensors
        in-place anyway (doing so would break autograd).
        """
        return {
            dt: ({src: dict(vars_) for src, vars_ in srcs.items()} if isinstance(srcs, dict) else srcs)
            for dt, srcs in batch.items()
        }

    def forward(self, batch: dict) -> dict:
        raise NotImplementedError

    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)
