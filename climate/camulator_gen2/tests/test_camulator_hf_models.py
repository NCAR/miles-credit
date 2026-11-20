"""HFCamulator / freeze_layers against HF applications/train.py's freezing."""

import pytest
import torch
from torch import nn

from camulator_gen2.models import HFCamulator, freeze_layers


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_embed = nn.Linear(2, 2)
        self.up_block1 = nn.Linear(2, 2)
        self.up_block4 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))


def _hf_freeze(model, trainable_layers, frozen_layers):
    """HF applications/train.py: frozen_layers, then trainable_layers."""
    if frozen_layers:
        for name, param in model.named_parameters():
            if any(pat in name for pat in frozen_layers):
                param.requires_grad = False
    if trainable_layers:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(pat in name for pat in trainable_layers):
                param.requires_grad = True
    return {n: p.requires_grad for n, p in model.named_parameters()}


@pytest.mark.parametrize(
    "trainable, frozen",
    [(["up_block4"], []), (["up_block1", "up_block4"], []), ([], ["cross_embed"]), ([], [])],
)
def test_freeze_layers_matches_hf(trainable, frozen):
    ours = freeze_layers(_Toy(), trainable, frozen)
    assert {n: p.requires_grad for n, p in ours.named_parameters()} == _hf_freeze(_Toy(), trainable, frozen)


def test_unmatched_pattern_is_an_error():
    with pytest.raises(ValueError, match="up_block9"):
        freeze_layers(_Toy(), ["up_block9"])


def test_hf_camulator_freezes_everything_but_the_decoder():
    model = HFCamulator(
        trainable_layers=["up_block1", "up_block2", "up_block3", "up_block4"],
        image_height=32,
        image_width=64,
        frames=1,
        channels=1,
        surface_channels=1,
        input_only_channels=1,
        output_only_channels=1,
        levels=2,
        dim=(32, 64, 128, 256),
        depth=(1, 1, 1, 1),
        global_window_size=(2, 1, 1, 1),
        local_window_size=1,
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    # 40 tensors, as in the AdamW state saved in HF's checkpoint.pt00069.pt
    assert len(trainable) == 40 and all(n.startswith("up_block") for n in trainable)
    assert sum(not p.requires_grad for p in model.parameters()) == len(list(model.parameters())) - 40
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(optimizer.param_groups[0]["params"]) == len(trainable)
