"""CPU-only tests for loading pre-ZeroPad2d-wrapper checkpoints.

CrossEmbedLayer used to hold bare Conv2d modules (``convs.<i>.weight``); wrapping
each conv in nn.Sequential(ZeroPad2d, Conv2d) for exact "same" padding moved them
to ``convs.<i>.1.weight``. Covers the load_state_dict pre-hook that migrates them,
the explicit ``migrate_legacy_state_dict`` entry point used on the FSDP2 path, and
the guard for the removed ConvTranspose2d decoder, which cannot be migrated.
"""

import re

import pytest
import torch
import torch.nn as nn

from credit.models.wxformer.crossformer import (
    CrossEmbedLayer,
    CrossFormer,
    apply_spectral_norm,
    migrate_legacy_state_dict,
)


def _to_legacy(state_dict):
    """Rewrite a current state dict into the pre-wrapper layout."""
    return {re.sub(r"(convs\.\d+)\.1\.", r"\1.", k): v for k, v in state_dict.items()}


IMAGE_HW = (64, 64)
CHANNELS, LEVELS, SURFACE_CHANNELS, INPUT_ONLY = 2, 2, 2, 1


def _make_crossformer():
    """Smallest 4-stage CrossFormer that still exercises every CrossEmbedLayer."""
    return CrossFormer(
        image_height=IMAGE_HW[0],
        image_width=IMAGE_HW[1],
        frames=1,
        output_frames=1,
        channels=CHANNELS,
        surface_channels=SURFACE_CHANNELS,
        input_only_channels=INPUT_ONLY,
        levels=LEVELS,
        dim=(16, 32, 64, 128),
        dim_head=8,
        depth=(1, 1, 1, 1),
        global_window_size=(8, 4, 2, 1),
        local_window_size=2,
        cross_embed_kernel_sizes=((2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_spectral_norm=False,
    )


def _model_input():
    n_in = CHANNELS * LEVELS + SURFACE_CHANNELS + INPUT_ONLY
    return torch.randn(1, n_in, 1, *IMAGE_HW)


class TestCrossEmbedLegacyLoad:
    @pytest.mark.parametrize("spectral", [False, True], ids=["plain", "spectral_norm"])
    def test_legacy_keys_load_strict(self, spectral):
        """Old-format weights load with strict=True and reproduce the source layer."""
        torch.manual_seed(0)
        src = CrossEmbedLayer(7, 32, [2, 4], stride=2)
        if spectral:
            apply_spectral_norm(src)
        legacy = _to_legacy(src.state_dict())
        assert "convs.0.1.weight" not in legacy and any(k.startswith("convs.0.") for k in legacy)

        torch.manual_seed(1)  # different init, so a successful load is observable
        dst = CrossEmbedLayer(7, 32, [2, 4], stride=2)
        if spectral:
            apply_spectral_norm(dst)
        dst.load_state_dict(legacy, strict=True)

        for key, val in src.state_dict().items():
            assert torch.equal(dst.state_dict()[key], val), key

        x = torch.randn(2, 7, 16, 16)
        src.eval()
        dst.eval()
        with torch.no_grad():
            assert torch.allclose(src(x), dst(x))

    def test_new_format_still_loads(self):
        """The hook must not disturb current-format checkpoints."""
        torch.manual_seed(0)
        src = CrossEmbedLayer(7, 32, [2, 4], stride=2)
        torch.manual_seed(1)
        dst = CrossEmbedLayer(7, 32, [2, 4], stride=2)
        dst.load_state_dict(src.state_dict(), strict=True)
        for key, val in src.state_dict().items():
            assert torch.equal(dst.state_dict()[key], val), key

    def test_hook_warns(self, caplog):
        torch.manual_seed(0)
        layer = CrossEmbedLayer(7, 32, [2, 4], stride=2)
        legacy = _to_legacy(layer.state_dict())
        with caplog.at_level("WARNING"):
            layer.load_state_dict(legacy, strict=True)
        assert "Legacy CrossEmbedLayer checkpoint" in caplog.text


class TestMigrateLegacyStateDict:
    def test_full_model_round_trip(self):
        """A legacy whole-model checkpoint migrates and reproduces the source model."""
        torch.manual_seed(0)
        src = _make_crossformer().eval()
        legacy = _to_legacy(src.state_dict())

        torch.manual_seed(1)
        dst = _make_crossformer().eval()
        migrate_legacy_state_dict(dst, legacy)
        dst.load_state_dict(legacy, strict=True)

        x = _model_input()
        with torch.no_grad():
            assert torch.allclose(src(x), dst(x), atol=1e-6)

    def test_idempotent(self):
        """Running the migration twice is a no-op the second time."""
        torch.manual_seed(0)
        model = _make_crossformer()
        legacy = _to_legacy(model.state_dict())
        once = dict(migrate_legacy_state_dict(model, dict(legacy)))
        twice = migrate_legacy_state_dict(model, dict(once))
        assert set(once) == set(twice)
        assert set(once) == set(model.state_dict())

    def test_new_format_unchanged(self):
        torch.manual_seed(0)
        model = _make_crossformer()
        current = model.state_dict()
        out = migrate_legacy_state_dict(model, dict(current))
        assert set(out) == set(current)
        for key, val in current.items():
            assert torch.equal(out[key], val), key

    def test_sibling_crossembed_untouched(self):
        """Other models' unwrapped `convs` must not be rewritten by name matching."""
        from credit.models.camulator import CrossEmbedLayer as CamulatorCrossEmbed

        torch.manual_seed(0)
        layer = CamulatorCrossEmbed(7, 32, [2, 4], stride=2)
        before = set(layer.state_dict())
        after = set(migrate_legacy_state_dict(layer, dict(layer.state_dict())))
        assert after == before
        assert "convs.0.weight" in after


class TestTrainerCheckpointHook:
    """The wrapper the FSDP2 resume path relies on, where the module hooks fire too late."""

    def test_migrates_model_state_dict(self):
        from credit.trainers.utils import _migrate_legacy_checkpoint

        torch.manual_seed(0)
        model = _make_crossformer()
        checkpoint = {"model_state_dict": _to_legacy(model.state_dict()), "epoch": 3}
        _migrate_legacy_checkpoint(model, checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        assert checkpoint["epoch"] == 3  # untouched

    def test_noop_without_model_state_dict(self):
        from credit.trainers.utils import _migrate_legacy_checkpoint

        torch.manual_seed(0)
        _migrate_legacy_checkpoint(_make_crossformer(), {"epoch": 1})  # must not raise

    def test_noop_for_unrelated_architecture(self):
        from credit.trainers.utils import _migrate_legacy_checkpoint

        model = nn.Sequential(nn.Conv2d(3, 3, 1))
        checkpoint = {"model_state_dict": dict(model.state_dict())}
        _migrate_legacy_checkpoint(model, checkpoint)
        assert set(checkpoint["model_state_dict"]) == set(model.state_dict())


class TestRemovedDecoderGuard:
    def _legacy_convtranspose_checkpoint(self, model):
        """A legacy dict whose up_block4 is a bare ConvTranspose2d."""
        legacy = _to_legacy(model.state_dict())
        for key in [k for k in legacy if k.startswith("up_block4.")]:
            legacy.pop(key)
        legacy["up_block4.weight"] = torch.randn(4, 4, 4, 4)
        legacy["up_block4.bias"] = torch.randn(4)
        return legacy

    def test_migrate_raises(self):
        torch.manual_seed(0)
        model = _make_crossformer()
        legacy = self._legacy_convtranspose_checkpoint(model)
        with pytest.raises(RuntimeError, match="upsample_with_ps=False"):
            migrate_legacy_state_dict(model, legacy)

    def test_load_state_dict_raises(self):
        """The guard also fires on the plain load path, not just the explicit call."""
        torch.manual_seed(0)
        model = _make_crossformer()
        legacy = self._legacy_convtranspose_checkpoint(model)
        with pytest.raises(RuntimeError, match="upsample_with_ps=False"):
            model.load_state_dict(legacy, strict=False)

    def test_real_convtranspose_decoder_allowed(self):
        """crossformer_diffusion legitimately keeps up_block4 as a ConvTranspose2d."""

        class DiffusionLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = CrossEmbedLayer(7, 32, [2, 4], stride=2)
                self.up_block4 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)

        torch.manual_seed(0)
        model = DiffusionLike()
        legacy = _to_legacy(model.state_dict())
        assert "up_block4.weight" in legacy  # correct current key for this architecture

        migrate_legacy_state_dict(model, legacy)  # must not raise
        model.load_state_dict(legacy, strict=True)
