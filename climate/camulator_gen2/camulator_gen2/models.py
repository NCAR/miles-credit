"""CAMulator with HF's fine-tune layer freezing.

HF's ``applications/train.py`` (``WillyChap/miles-credit@camulator_huggingface``)
froze parameters before building the optimizer, from two ``trainer`` settings:

- ``frozen_layers``: freeze every parameter whose name contains one of these strings;
- ``trainable_layers``: freeze everything, then unfreeze every parameter whose name
  contains one of these strings.

The optimizer then got only the trainable parameters. The gen2 trainer builds
its AdamW the same way (``filter(lambda p: p.requires_grad, ...)``) but has no
such settings, so this model applies them when it is built, before the
optimizer exists. The released checkpoint (epoch 69) was trained with
``trainable_layers: [up_block1, up_block2, up_block3, up_block4]``.
"""

import logging

from credit.models.camulator import Camulator

logger = logging.getLogger(__name__)


def freeze_layers(model, trainable_layers=None, frozen_layers=None):
    """Apply HF's ``frozen_layers`` then ``trainable_layers`` rules (substring match on parameter names)."""
    frozen_layers = list(frozen_layers or [])
    trainable_layers = list(trainable_layers or [])
    names = [name for name, _ in model.named_parameters()]
    for patterns, label in ((frozen_layers, "frozen_layers"), (trainable_layers, "trainable_layers")):
        unmatched = [p for p in patterns if not any(p in name for name in names)]
        if unmatched:
            raise ValueError(f"{label} {unmatched} match no parameter of {type(model).__name__}")
    if frozen_layers:
        for name, param in model.named_parameters():
            if any(p in name for p in frozen_layers):
                param.requires_grad = False
    if trainable_layers:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(p in name for p in trainable_layers):
                param.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.1f%%)", f"{n_trainable:,}", f"{n_total:,}", 100 * n_trainable / n_total
    )
    return model


class HFCamulator(Camulator):
    """:class:`credit.models.camulator.Camulator` that freezes layers as HF's ``train.py`` did.

    Args:
        trainable_layers: parameter-name substrings to train; everything else is frozen.
        frozen_layers: parameter-name substrings to freeze.
        **kwargs: passed to :class:`~credit.models.camulator.Camulator`.
    """

    def __init__(self, trainable_layers=None, frozen_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.trainable_layers = list(trainable_layers or [])
        self.frozen_layers = list(frozen_layers or [])
        freeze_layers(self, self.trainable_layers, self.frozen_layers)
