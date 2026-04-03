"""
FSDP policy comparison test: hard-coded vs auto-detect for legacy models.

Run with:
  torchrun --nproc_per_node=2 tests/test_fsdp_policies.py

Tests crossformer, wxformer, unet, fuxi, swin with both the current hard-coded
wrap policy and the generic auto-detect path, then reports discovered layer sets
and whether FSDP wrap + fwd + bwd succeeds for each.
"""

import os
import sys
import time
import functools
import traceback

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

B, C_IN, C_OUT, H, W = 1, 20, 18, 32, 64


# ---------------------------------------------------------------------------
# Hard-coded policies (mirrors current credit/models/__init__.py)
# ---------------------------------------------------------------------------


def _hardcoded_policy(model_type):
    if "crossformer" in model_type or "unet" in model_type:
        from credit.models.crossformer.crossformer import Attention, DynamicPositionBias, FeedForward, CrossEmbedLayer

        return {Attention, DynamicPositionBias, FeedForward, CrossEmbedLayer}
    if "fuxi" in model_type or "wrf" in model_type or "dscale" in model_type:
        from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

        return {SwinTransformerV2Stage}
    if "swin" in model_type:
        from credit.models.swin.swin import (
            SwinTransformerV2CrBlock,
            WindowMultiHeadAttentionNoPos,
            WindowMultiHeadAttention,
        )

        return {SwinTransformerV2CrBlock, WindowMultiHeadAttentionNoPos, WindowMultiHeadAttention}
    return set()


# ---------------------------------------------------------------------------
# Auto-detect policy (mirrors generic path in load_fsdp_or_checkpoint_policy)
# ---------------------------------------------------------------------------


def _autodetect_policy(model):
    from collections import Counter

    try:
        from einops.layers.torch import Rearrange as _Rearrange
    except ImportError:
        _Rearrange = None
    _skip = {
        nn.Sequential,
        nn.ModuleList,
        nn.ModuleDict,
        nn.Linear,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.BatchNorm2d,
        nn.Conv2d,
        nn.Conv1d,
        nn.ConvTranspose2d,
        nn.Dropout,
        nn.GELU,
        nn.ReLU,
        nn.SiLU,
        nn.Identity,
        nn.Embedding,
        nn.AvgPool2d,
        nn.MaxPool2d,
    }
    if _Rearrange is not None:
        _skip.add(_Rearrange)
    _skip_names = {"LayerNorm"}
    type_counts = Counter(type(m) for m in model.modules())
    return {
        t
        for t, count in type_counts.items()
        if count > 1 and t not in _skip and issubclass(t, nn.Module) and t.__name__ not in _skip_names
    }


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _make_wxformer():
    from credit.models.wxformer.crossformer import CrossFormer

    return CrossFormer(
        in_channels=C_IN,
        out_channels=C_OUT,
        patch_size_factor=[2],
        frames=1,
        dim=[32],
        depth=[2],
        global_window_size=[4],
        local_window_size=2,
        cross_embed_kernel_sizes=[[2, 4]],
        cross_embed_strides=[2],
        num_heads=[4],
        attn_dropout=0.0,
        ff_dropout=0.0,
    )


def _make_crossformer():
    from credit.models.crossformer.crossformer import CrossFormer

    return CrossFormer(
        in_channels=C_IN,
        out_channels=C_OUT,
        patch_size_factor=[2],
        frames=1,
        dim=[32],
        depth=[2],
        global_window_size=[4],
        local_window_size=2,
        cross_embed_kernel_sizes=[[2, 4]],
        cross_embed_strides=[2],
        num_heads=[4],
        attn_dropout=0.0,
        ff_dropout=0.0,
    )


def _make_unet():
    from credit.models.unet.unet import SegmentationModel

    return SegmentationModel(
        in_channels=C_IN,
        out_channels=C_OUT,
        init_features=16,
        levels=2,
    )


def _make_fuxi():
    from credit.models.fuxi.fuxi import Fuxi

    return Fuxi(
        in_channels=C_IN,
        out_channels=C_OUT,
        patch_size=4,
        embed_dim=64,
        num_groups=8,
        num_heads=4,
        window_size=4,
        depth=2,
    )


def _make_swin():
    from credit.models.swin.swin import SwinTransformerV2Cr

    return SwinTransformerV2Cr(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        patch_size=2,
        embed_dim=64,
        depths=[2],
        num_heads=[4],
        window_size=4,
    )


MODELS = [
    ("wxformer", "crossformer", _make_wxformer),
    ("crossformer", "crossformer", _make_crossformer),
    ("unet", "unet", _make_unet),
    ("fuxi", "fuxi", _make_fuxi),
    ("swin", "swin", _make_swin),
]


# ---------------------------------------------------------------------------
# FSDP wrap + fwd + bwd helper
# ---------------------------------------------------------------------------


def _fsdp_run(model, policy_cls, device):
    def combined(module, recurse, nonwrapped_numel):
        p1 = (
            functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=policy_cls)(
                module, recurse, nonwrapped_numel
            )
            if policy_cls
            else False
        )
        p2 = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000)(module, recurse, nonwrapped_numel)
        return p1 or p2

    wrapped = FSDP(model, auto_wrap_policy=combined, device_id=device)
    x = torch.randn(B, C_IN, H, W, device=device)
    t0 = time.perf_counter()
    out = wrapped(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 5:
        out = out[:, :, 0]
    loss = out.mean()
    loss.backward()
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("\n=== FSDP policy comparison ===")
        print(f"Ranks: {dist.get_world_size()}   Device: {torch.cuda.get_device_name(device)}")
        print(f"{'Model':<14} {'Policy':<12} {'Layers found':<6} {'ms':>6}  {'Status'}")
        print("-" * 70)

    results = []

    for display_name, model_type, factory in MODELS:
        for policy_name, get_policy in [
            ("hard-coded", lambda mt=model_type, f=factory: _hardcoded_policy(mt)),
            ("auto-detect", lambda mt=model_type, f=factory: _autodetect_policy(f())),
        ]:
            try:
                model = factory().to(device)
                policy_cls = get_policy()
                elapsed = _fsdp_run(model, policy_cls, device)
                status = "PASS"
                n_layers = len(policy_cls)
                layer_names = sorted(c.__name__ for c in policy_cls)
            except Exception:
                elapsed = 0.0
                status = "FAIL"
                n_layers = 0
                layer_names = []
                if rank == 0:
                    traceback.print_exc()

            dist.barrier()
            results.append((display_name, policy_name, n_layers, layer_names, elapsed, status))

            if rank == 0:
                print(f"{display_name:<14} {policy_name:<12} {n_layers:<6} {elapsed:>6.0f}  {status}")

        if rank == 0:
            # Show layer name comparison
            hc = next(r for r in results if r[0] == display_name and r[1] == "hard-coded")
            ad = next(r for r in results if r[0] == display_name and r[1] == "auto-detect")
            hc_layers = set(hc[3])
            ad_layers = set(ad[3])
            only_hc = hc_layers - ad_layers
            only_ad = ad_layers - hc_layers
            if only_hc:
                print(f"  {'':14} hard-coded only: {sorted(only_hc)}")
            if only_ad:
                print(f"  {'':14} auto-detect only: {sorted(only_ad)}")
            if not only_hc and not only_ad and hc_layers:
                print(f"  {'':14} identical layer sets")
            print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
