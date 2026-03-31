"""
Tests for regional WRF/downscaling model classes and the load_model factory.

Covers:
  - Helper utilities: get_pad2d, get_pad3d, apply_spectral_norm
  - Sub-module classes: CubeEmbedding, DownBlock, UpBlock  (dscale_wrf and swin_wrf variants)
  - WRFTransformer instantiation (minimal config, no forward pass)
  - DscaleTransformer instantiation (minimal config, no forward pass)
  - load_model factory: "wrf" and "dscale" type keys
  - load_model error paths: missing type, unknown type

All tests are CPU-only, using synthetic tensors — no real data files needed.
timm is a hard dependency of both model modules; tests are skipped if absent.
"""

import pytest

# ---------------------------------------------------------------------------
# timm availability guard
# ---------------------------------------------------------------------------

try:
    import timm  # noqa: F401

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

needs_timm = pytest.mark.skipif(not HAS_TIMM, reason="timm not installed")

# ---------------------------------------------------------------------------
# Minimal model configs
# ---------------------------------------------------------------------------

# A tiny WRFTransformer config that constructs fast on CPU.
# dim must be divisible by num_groups (32→4 groups of 8).
# patch sizes must evenly divide image dims.
# window_size must divide (image / patch / 2).
#   image=32, patch=4 → patches=8 → /2 = 4 → divisible by window_size=4 ✓
_PARAM_INTERIOR = {
    "image_height": 32,
    "image_width": 32,
    "patch_height": 4,
    "patch_width": 4,
    "levels": 2,
    "frames": 2,
    "frame_patch_size": 2,
    "channels": 2,
    "surface_channels": 1,
    "input_only_channels": 0,
    "output_only_channels": 0,
    "dim": 32,
}

_PARAM_OUTSIDE = {
    "image_height": 32,
    "image_width": 32,
    "patch_height": 4,
    "patch_width": 4,
    "levels": 2,
    "frames": 2,
    "frame_patch_size": 2,
    "channels": 2,
    "surface_channels": 1,
    "dim": 32,
}

_WRF_MODEL_CONF = {
    "param_interior": _PARAM_INTERIOR,
    "param_outside": _PARAM_OUTSIDE,
    "time_encode_dim": 4,
    "num_groups": 4,
    "num_heads": 2,
    "depth": 2,
    "window_size": 4,
    "use_spectral_norm": False,
    "interp": False,
    "drop_path": 0,
    "padding_conf": {"activate": False},
    "post_conf": {"activate": False},
}

# DscaleTransformer config
# image=32, patch=4 → 8 patches → /2 = 4 → divisible by window_size=4 ✓
_DSCALE_MODEL_CONF = {
    "image_height": 32,
    "image_width": 32,
    "patch_height": 4,
    "patch_width": 4,
    "total_input_channels": 5,
    "total_target_channels": 3,
    "time_encode_dim": 4,
    "frames": 2,
    "frame_patch_size": 2,
    "dim": 32,
    "num_groups": 4,
    "num_heads": 2,
    "depth": 2,
    "window_size": 4,
    "use_spectral_norm": False,
    "interp": False,
    "proj_drop": 0,
    "attn_drop": 0,
    "drop_path": 0,
    "padding_conf": {"activate": False},
    "post_conf": {"activate": False},
}


# ---------------------------------------------------------------------------
# TestPadHelpers — shared helpers in both modules
# ---------------------------------------------------------------------------


class TestPadHelpers:
    """Tests for get_pad3d and get_pad2d padding utilities."""

    @needs_timm
    def test_get_pad3d_no_padding_needed(self):
        from credit.models.dscale_wrf.dscale_wrf import get_pad3d

        result = get_pad3d((4, 8, 8), (4, 4, 4))
        assert result == (0, 0, 0, 0, 0, 0), f"Expected all zeros, got {result}"

    @needs_timm
    def test_get_pad3d_padding_needed(self):
        from credit.models.dscale_wrf.dscale_wrf import get_pad3d

        # 9 % 4 = 1 → needs 3 more → front=1, back=2
        pl, lat, lon = 9, 8, 8
        result = get_pad3d((pl, lat, lon), (4, 4, 4))
        total_front_back = result[4] + result[5]
        assert total_front_back == 3

    @needs_timm
    def test_get_pad3d_returns_6tuple(self):
        from credit.models.dscale_wrf.dscale_wrf import get_pad3d

        result = get_pad3d((5, 5, 5), (3, 3, 3))
        assert len(result) == 6

    @needs_timm
    def test_get_pad2d_returns_4tuple(self):
        from credit.models.dscale_wrf.dscale_wrf import get_pad2d

        result = get_pad2d((8, 8), (4, 4))
        assert len(result) == 4

    @needs_timm
    def test_get_pad2d_no_padding_needed(self):
        from credit.models.dscale_wrf.dscale_wrf import get_pad2d

        result = get_pad2d((8, 8), (4, 4))
        assert result == (0, 0, 0, 0)

    @needs_timm
    def test_get_pad3d_swin_wrf_variant(self):
        """swin_wrf has its own copy — verify it behaves identically."""
        from credit.models.dscale_wrf.dscale_wrf import get_pad3d as gp_dscale
        from credit.models.swin_wrf.swin_wrf import get_pad3d as gp_wrf

        args = ((5, 7, 11), (3, 4, 6))
        assert gp_dscale(*args) == gp_wrf(*args)

    @needs_timm
    def test_get_pad2d_swin_wrf_variant(self):
        from credit.models.dscale_wrf.dscale_wrf import get_pad2d as gp_dscale
        from credit.models.swin_wrf.swin_wrf import get_pad2d as gp_wrf

        args = ((7, 11), (4, 6))
        assert gp_dscale(*args) == gp_wrf(*args)


# ---------------------------------------------------------------------------
# TestApplySpectralNorm
# ---------------------------------------------------------------------------


class TestApplySpectralNorm:
    @needs_timm
    def test_apply_spectral_norm_dscale(self):
        import torch.nn as nn
        from credit.models.dscale_wrf.dscale_wrf import apply_spectral_norm

        model = nn.Sequential(nn.Linear(4, 4), nn.Conv2d(1, 1, 3))
        apply_spectral_norm(model)
        # After spectral norm, the linear layer has a weight_u attribute
        lin = model[0]
        assert hasattr(lin, "weight_u"), "spectral norm not applied to Linear"

    @needs_timm
    def test_apply_spectral_norm_swin_wrf(self):
        import torch.nn as nn
        from credit.models.swin_wrf.swin_wrf import apply_spectral_norm

        model = nn.Sequential(nn.Linear(4, 4))
        apply_spectral_norm(model)
        lin = model[0]
        assert hasattr(lin, "weight_u")


# ---------------------------------------------------------------------------
# TestCubeEmbedding
# ---------------------------------------------------------------------------


class TestCubeEmbedding:
    @needs_timm
    def test_instantiation_dscale(self):
        from credit.models.dscale_wrf.dscale_wrf import CubeEmbedding

        emb = CubeEmbedding(img_size=(2, 8, 8), patch_size=(2, 4, 4), in_chans=3, embed_dim=16)
        assert emb is not None

    @needs_timm
    def test_forward_shape_dscale(self):
        import torch
        from credit.models.dscale_wrf.dscale_wrf import CubeEmbedding

        emb = CubeEmbedding(img_size=(2, 8, 8), patch_size=(2, 4, 4), in_chans=3, embed_dim=16)
        # Input layout: (B, in_chans, T, Lat, Lon) — the forward labels T/C but Conv3d
        # treats dim-1 as the channel dimension, which is in_chans here.
        x = torch.randn(1, 3, 2, 8, 8)
        out = emb(x)
        # Expected: (B, embed_dim, T//pT, Lat//pH, Lon//pW) = (1, 16, 1, 2, 2)
        assert out.shape == (1, 16, 1, 2, 2), f"Unexpected shape {out.shape}"

    @needs_timm
    def test_instantiation_swin_wrf(self):
        from credit.models.swin_wrf.swin_wrf import CubeEmbedding

        emb = CubeEmbedding(img_size=(2, 8, 8), patch_size=(2, 4, 4), in_chans=5, embed_dim=16)
        assert emb is not None

    @needs_timm
    def test_forward_shape_swin_wrf(self):
        import torch
        from credit.models.swin_wrf.swin_wrf import CubeEmbedding

        emb = CubeEmbedding(img_size=(2, 8, 8), patch_size=(2, 4, 4), in_chans=5, embed_dim=16)
        # Input: (B, in_chans, T, Lat, Lon)
        x = torch.randn(1, 5, 2, 8, 8)
        out = emb(x)
        assert out.shape == (1, 16, 1, 2, 2)

    @needs_timm
    def test_no_norm_layer(self):
        import torch
        from credit.models.dscale_wrf.dscale_wrf import CubeEmbedding

        emb = CubeEmbedding((2, 8, 8), (2, 4, 4), 3, 16, norm_layer=None)
        # Input: (B, in_chans, T, Lat, Lon)
        x = torch.randn(1, 3, 2, 8, 8)
        out = emb(x)
        assert out.shape[1] == 16


# ---------------------------------------------------------------------------
# TestDownBlockUpBlock
# ---------------------------------------------------------------------------


class TestDownUpBlocks:
    @needs_timm
    def test_downblock_instantiation(self):
        from credit.models.dscale_wrf.dscale_wrf import DownBlock

        blk = DownBlock(in_chans=16, out_chans=16, num_groups=4)
        assert blk is not None

    @needs_timm
    def test_downblock_forward_halves_spatial(self):
        import torch
        from credit.models.dscale_wrf.dscale_wrf import DownBlock

        blk = DownBlock(in_chans=16, out_chans=16, num_groups=4)
        x = torch.randn(1, 16, 8, 8)
        out = blk(x)
        assert out.shape == (1, 16, 4, 4)

    @needs_timm
    def test_upblock_instantiation(self):
        from credit.models.dscale_wrf.dscale_wrf import UpBlock

        blk = UpBlock(in_chans=32, out_chans=16, num_groups=4)
        assert blk is not None

    @needs_timm
    def test_upblock_forward_doubles_spatial(self):
        import torch
        from credit.models.dscale_wrf.dscale_wrf import UpBlock

        blk = UpBlock(in_chans=32, out_chans=16, num_groups=4)
        x = torch.randn(1, 32, 4, 4)
        out = blk(x)
        assert out.shape == (1, 16, 8, 8)


# ---------------------------------------------------------------------------
# TestDscaleTransformerInit
# ---------------------------------------------------------------------------


class TestDscaleTransformerInit:
    """Instantiation tests only — no forward pass (requires GPU for full-scale)."""

    @needs_timm
    def test_import(self):
        from credit.models.dscale_wrf import DscaleTransformer  # noqa: F401

    @needs_timm
    def test_instantiation(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        conf = copy.deepcopy(_DSCALE_MODEL_CONF)
        model = DscaleTransformer(**conf)
        assert model is not None

    @needs_timm
    def test_is_base_model(self):
        from credit.models.dscale_wrf import DscaleTransformer
        from credit.models.base_model import BaseModel

        import copy

        model = DscaleTransformer(**copy.deepcopy(_DSCALE_MODEL_CONF))
        assert isinstance(model, BaseModel)

    @needs_timm
    def test_has_cube_embedding(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        model = DscaleTransformer(**copy.deepcopy(_DSCALE_MODEL_CONF))
        assert hasattr(model, "cube_embedding")

    @needs_timm
    def test_has_u_transformer(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        model = DscaleTransformer(**copy.deepcopy(_DSCALE_MODEL_CONF))
        assert hasattr(model, "u_transformer")

    @needs_timm
    def test_has_fc_layer(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        model = DscaleTransformer(**copy.deepcopy(_DSCALE_MODEL_CONF))
        assert hasattr(model, "fc")

    @needs_timm
    def test_has_film_layer(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        model = DscaleTransformer(**copy.deepcopy(_DSCALE_MODEL_CONF))
        assert hasattr(model, "film")

    @needs_timm
    def test_no_postblock_when_inactive(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        model = DscaleTransformer(**copy.deepcopy(_DSCALE_MODEL_CONF))
        assert not model.use_post_block

    @needs_timm
    def test_no_padding_when_inactive(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        model = DscaleTransformer(**copy.deepcopy(_DSCALE_MODEL_CONF))
        assert not model.use_padding

    @needs_timm
    def test_spectral_norm_disabled(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        conf = copy.deepcopy(_DSCALE_MODEL_CONF)
        conf["use_spectral_norm"] = False
        model = DscaleTransformer(**conf)
        assert not model.use_spectral_norm

    @needs_timm
    def test_parameter_count_positive(self):
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        model = DscaleTransformer(**copy.deepcopy(_DSCALE_MODEL_CONF))
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


# ---------------------------------------------------------------------------
# TestWRFTransformerInit
# ---------------------------------------------------------------------------


class TestWRFTransformerInit:
    """Instantiation tests for WRFTransformer (dual-encoder model)."""

    @needs_timm
    def test_import(self):
        from credit.models.swin_wrf import WRFTransformer  # noqa: F401

    @needs_timm
    def test_instantiation(self):
        from credit.models.swin_wrf import WRFTransformer

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        assert model is not None

    @needs_timm
    def test_is_base_model(self):
        from credit.models.swin_wrf import WRFTransformer
        from credit.models.base_model import BaseModel

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        assert isinstance(model, BaseModel)

    @needs_timm
    def test_has_dual_cube_embeddings(self):
        from credit.models.swin_wrf import WRFTransformer

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        assert hasattr(model, "cube_embedding_inside")
        assert hasattr(model, "cube_embedding_outside")

    @needs_timm
    def test_has_u_transformer(self):
        from credit.models.swin_wrf import WRFTransformer

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        assert hasattr(model, "u_transformer")

    @needs_timm
    def test_has_fc_and_film(self):
        from credit.models.swin_wrf import WRFTransformer

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        assert hasattr(model, "fc")
        assert hasattr(model, "film")

    @needs_timm
    def test_no_post_block_when_inactive(self):
        from credit.models.swin_wrf import WRFTransformer

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        assert not model.use_post_block

    @needs_timm
    def test_no_padding_when_inactive(self):
        from credit.models.swin_wrf import WRFTransformer

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        assert not model.use_padding

    @needs_timm
    def test_parameter_count_positive(self):
        from credit.models.swin_wrf import WRFTransformer

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    @needs_timm
    def test_out_chans_matches_config(self):
        """out_chans = channels*levels + surface + output_only = 2*2+1+0 = 5."""
        from credit.models.swin_wrf import WRFTransformer

        import copy

        model = WRFTransformer(**copy.deepcopy(_WRF_MODEL_CONF))
        expected = (
            _PARAM_INTERIOR["channels"] * _PARAM_INTERIOR["levels"]
            + _PARAM_INTERIOR["surface_channels"]
            + _PARAM_INTERIOR["output_only_channels"]
        )
        assert model.out_chans == expected


# ---------------------------------------------------------------------------
# TestLoadModelFactory — "wrf" and "dscale" type keys
# ---------------------------------------------------------------------------


class TestLoadModelFactory:
    @needs_timm
    def test_load_model_missing_type_raises(self):
        from credit.models import load_model

        with pytest.raises(ValueError, match="type"):
            load_model({"model": {}})

    @needs_timm
    def test_load_model_unknown_type_raises(self):
        from credit.models import load_model

        with pytest.raises(ValueError, match="not supported"):
            load_model({"model": {"type": "_not_a_real_model_xyz"}})

    @needs_timm
    def test_load_model_dscale_returns_dscale_transformer(self):
        from credit.models import load_model
        from credit.models.dscale_wrf import DscaleTransformer

        import copy

        conf = {"model": {"type": "dscale", **copy.deepcopy(_DSCALE_MODEL_CONF)}}
        model = load_model(conf)
        assert isinstance(model, DscaleTransformer)

    @needs_timm
    def test_load_model_wrf_returns_wrf_transformer(self):
        from credit.models import load_model
        from credit.models.swin_wrf import WRFTransformer

        import copy

        conf = {"model": {"type": "wrf", **copy.deepcopy(_WRF_MODEL_CONF)}}
        model = load_model(conf)
        assert isinstance(model, WRFTransformer)

    @needs_timm
    def test_load_model_dscale_no_weights_does_not_raise(self):
        from credit.models import load_model

        import copy

        conf = {"model": {"type": "dscale", **copy.deepcopy(_DSCALE_MODEL_CONF)}}
        model = load_model(conf, load_weights=False)
        assert model is not None

    @needs_timm
    def test_load_model_wrf_no_weights_does_not_raise(self):
        from credit.models import load_model

        import copy

        conf = {"model": {"type": "wrf", **copy.deepcopy(_WRF_MODEL_CONF)}}
        model = load_model(conf, load_weights=False)
        assert model is not None


# ---------------------------------------------------------------------------
# TestLoadFsdpOrCheckpointPolicy — wrf and dscale branches
# ---------------------------------------------------------------------------


class TestLoadFsdpOrCheckpointPolicy:
    @needs_timm
    def test_wrf_returns_set_with_swin_stage(self):
        from credit.models import load_fsdp_or_checkpoint_policy
        from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

        policy = load_fsdp_or_checkpoint_policy({"model": {"type": "wrf"}})
        assert SwinTransformerV2Stage in policy

    @needs_timm
    def test_dscale_returns_set_with_swin_stage(self):
        from credit.models import load_fsdp_or_checkpoint_policy
        from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

        policy = load_fsdp_or_checkpoint_policy({"model": {"type": "dscale"}})
        assert SwinTransformerV2Stage in policy

    @needs_timm
    def test_unknown_type_raises_os_error(self):
        from credit.models import load_fsdp_or_checkpoint_policy

        with pytest.raises(OSError):
            load_fsdp_or_checkpoint_policy({"model": {"type": "_unknown_model_"}})


# ---------------------------------------------------------------------------
# TestPackageImports — __init__.py re-exports
# ---------------------------------------------------------------------------


class TestPackageImports:
    @needs_timm
    def test_dscale_wrf_package_exports_class(self):
        from credit.models.dscale_wrf import DscaleTransformer
        from credit.models.dscale_wrf.dscale_wrf import DscaleTransformer as _Direct

        assert DscaleTransformer is _Direct

    @needs_timm
    def test_swin_wrf_package_exports_class(self):
        from credit.models.swin_wrf import WRFTransformer
        from credit.models.swin_wrf.swin_wrf import WRFTransformer as _Direct

        assert WRFTransformer is _Direct

    @needs_timm
    def test_models_init_exports_both(self):
        from credit.models import WRFTransformer, DscaleTransformer

        assert WRFTransformer is not None
        assert DscaleTransformer is not None
