"""CPU-only unit tests for native tensor parallelism on wxformer_next (issue #415).

Covers:
  - Conv -> Linear checkpoint remap (remap_conv_state_dict), incl. spectral norm
  - Numerical equivalence: conv-projection transformer (crossformer) vs the
    Linear-projection transformer (wxformer_next) with remapped weights
  - Full-model old-format checkpoint loading

The multi-GPU tp=2 vs tp=1 parity run lives in tests/manual/gen2_parallelism/.
"""

import pytest
import torch

from credit.models.wxformer.crossformer import (
    Transformer as ConvTransformer,
    apply_spectral_norm,
)
from credit.models.wxformer.wxformer_next import (
    NextGenWXFormer,
    Transformer as LinearTransformer,
    remap_conv_state_dict,
)


TINY_TRANSFORMER_KW = dict(local_window_size=4, global_window_size=2, depth=2, dim_head=4)


def _tiny_model_conf(use_spectral_norm=False):
    return dict(
        image_height=32,
        image_width=64,
        frames=1,
        channels=2,
        surface_channels=2,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=(8, 16, 32, 64),
        depth=(1, 1, 1, 1),
        dim_head=4,
        global_window_size=(2, 2, 2, 1),
        local_window_size=2,
        cross_embed_kernel_sizes=((2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        col_attn_heads=2,
        use_spectral_norm=use_spectral_norm,
    )


def _to_conv_format(state_dict):
    """Invert remap_conv_state_dict: build an old conv-format state dict from a
    new Linear-format one (fuse q/k/v, view 2D weights as 1x1 conv kernels)."""
    out = {}
    qkv = {}
    for key, val in state_dict.items():
        is_proj = False
        for name in ("to_q", "to_k", "to_v"):
            marker = f".{name}."
            if marker in key:
                stem, suffix = key.rsplit(marker, 1)
                qkv.setdefault((stem, suffix), {})[name] = val
                is_proj = True
                break
        if is_proj:
            continue
        if key.endswith((".to_out.weight", ".layers.1.weight", ".layers.4.weight")) and val.dim() == 2:
            out[key] = val.reshape(*val.shape, 1, 1)
        else:
            out[key] = val
    for (stem, suffix), parts in qkv.items():
        fused = torch.cat([parts["to_q"], parts["to_k"], parts["to_v"]], dim=0)
        if suffix == "weight":
            fused = fused.reshape(*fused.shape, 1, 1)
        out[f"{stem}.to_qkv.{suffix}"] = fused
    return out


# ---------------------------------------------------------------------------
# remap_conv_state_dict
# ---------------------------------------------------------------------------


class TestRemapConvStateDict:
    def test_qkv_split_and_conv_reshape(self):
        torch.manual_seed(0)
        conv_t = ConvTransformer(16, **TINY_TRANSFORMER_KW)
        sd = remap_conv_state_dict(conv_t.state_dict())

        assert not any(".to_qkv." in k for k in sd)
        # short attention of the first depth layer
        for name in ("to_q", "to_k", "to_v"):
            w = sd[f"layers.0.0.{name}.weight"]
            assert w.shape == (16, 16)
        # fused conv rows [q, k, v] split in order
        old = conv_t.state_dict()["layers.0.0.to_qkv.weight"]
        assert torch.equal(sd["layers.0.0.to_q.weight"], old[:16].reshape(16, 16))
        assert torch.equal(sd["layers.0.0.to_v.weight"], old[32:].reshape(16, 16))
        # to_out and FFN convs become 2D
        assert sd["layers.0.0.to_out.weight"].shape == (16, 16)
        assert sd["layers.0.1.layers.1.weight"].shape == (64, 16)
        assert sd["layers.0.1.layers.4.weight"].shape == (16, 64)

    def test_loads_strict_into_linear_transformer(self):
        torch.manual_seed(0)
        conv_t = ConvTransformer(16, **TINY_TRANSFORMER_KW)
        lin_t = LinearTransformer(16, **TINY_TRANSFORMER_KW)
        sd = remap_conv_state_dict(conv_t.state_dict())
        lin_t.load_state_dict(sd, strict=True)

    def test_spectral_norm_keys_remapped(self):
        torch.manual_seed(0)
        conv_t = ConvTransformer(16, **TINY_TRANSFORMER_KW)
        apply_spectral_norm(conv_t)
        lin_t = LinearTransformer(16, **TINY_TRANSFORMER_KW)
        apply_spectral_norm(lin_t)
        sd = remap_conv_state_dict(conv_t.state_dict())
        lin_t.load_state_dict(sd, strict=True)
        # u split in thirds, v copied
        old = conv_t.state_dict()
        assert torch.equal(sd["layers.0.0.to_q.weight_u"], old["layers.0.0.to_qkv.weight_u"][:16])
        assert torch.equal(sd["layers.0.0.to_k.weight_v"], old["layers.0.0.to_qkv.weight_v"])

    def test_idempotent_on_new_format(self):
        torch.manual_seed(0)
        lin_t = LinearTransformer(16, **TINY_TRANSFORMER_KW)
        sd = lin_t.state_dict()
        out = remap_conv_state_dict(sd)
        assert set(out) == set(sd)
        for k in sd:
            assert torch.equal(out[k], sd[k])

    def test_non_projection_convs_untouched(self):
        """3x3 convs (decoder, CrossEmbed) must stay 4D."""
        torch.manual_seed(0)
        model = NextGenWXFormer(**_tiny_model_conf())
        sd = remap_conv_state_dict(model.state_dict())
        assert sd["up_block1.conv.weight"].dim() == 4
        assert sd["layers.0.0.convs.0.weight"].dim() == 4

    def test_unexpected_qkv_suffix_raises(self):
        with pytest.raises(KeyError, match="to_qkv"):
            remap_conv_state_dict({"block.to_qkv.weight_garbage": torch.zeros(3)})


# ---------------------------------------------------------------------------
# Numerical equivalence: conv blocks vs Linear blocks with remapped weights
# ---------------------------------------------------------------------------


class TestConvLinearEquivalence:
    def test_transformer_outputs_identical(self):
        torch.manual_seed(0)
        conv_t = ConvTransformer(16, **TINY_TRANSFORMER_KW).eval()
        lin_t = LinearTransformer(16, **TINY_TRANSFORMER_KW).eval()
        lin_t.load_state_dict(remap_conv_state_dict(conv_t.state_dict()), strict=True)

        x = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            y_conv = conv_t(x)
            y_lin = lin_t(x)
        assert y_lin.shape == y_conv.shape
        assert torch.allclose(y_conv, y_lin, atol=1e-6), f"max diff {(y_conv - y_lin).abs().max().item()}"

    def test_gradients_flow(self):
        torch.manual_seed(0)
        lin_t = LinearTransformer(16, **TINY_TRANSFORMER_KW)
        x = torch.randn(1, 16, 8, 8)
        lin_t(x).mean().backward()
        for name, p in lin_t.named_parameters():
            assert p.grad is not None, f"no grad for {name}"


# ---------------------------------------------------------------------------
# Full-model: old conv-format checkpoint loads and reproduces outputs
# ---------------------------------------------------------------------------


class TestFullModelRemapRoundTrip:
    def test_old_format_checkpoint_loads_strict(self):
        torch.manual_seed(0)
        model = NextGenWXFormer(**_tiny_model_conf())
        old_sd = _to_conv_format(model.state_dict())
        # old format really is different
        assert any(".to_qkv." in k for k in old_sd)
        fresh = NextGenWXFormer(**_tiny_model_conf())
        fresh.load_state_dict(remap_conv_state_dict(old_sd), strict=True)

        model.eval()
        fresh.eval()
        x = torch.randn(1, 7, 1, 32, 64)
        with torch.no_grad():
            y_a = model(x)
            y_b = fresh(x)
        assert torch.allclose(y_a, y_b, atol=1e-6)

    def test_forward_shape_5d(self):
        torch.manual_seed(0)
        model = NextGenWXFormer(**_tiny_model_conf(use_spectral_norm=True))
        x = torch.randn(1, 7, 1, 32, 64)
        y = model(x)
        assert tuple(y.shape) == (1, 6, 1, 32, 64)
        assert not torch.isnan(y).any()
