"""CPU-only unit tests for native tensor parallelism on wxformer_next (issue #415).

Covers:
  - Conv -> Linear checkpoint remap (remap_conv_state_dict), incl. spectral norm
  - Numerical equivalence: conv-projection transformer (crossformer) vs the
    Linear-projection transformer (wxformer_next) with remapped weights
  - Full-model old-format checkpoint loading
  - apply_native_tensor_parallel: opt-in detection, plan construction,
    divisibility / non-Linear validation errors, and the reduced form:
    spectral-norm-wrapped blocks are skipped (replicated) with a warning

The multi-GPU tp=2 vs tp=1 parity run lives in tests/manual/gen2_parallelism/.
"""

import logging

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from credit.models.wxformer.crossformer import (
    Transformer as ConvTransformer,
    apply_spectral_norm,
)
from credit.models.wxformer.wxformer_next import (
    Attention,
    FeedForward,
    NextGenWXFormer,
    Transformer as LinearTransformer,
    remap_conv_state_dict,
)
from credit.parallel.tensor_parallel import (
    _is_tp_sharded_param,
    apply_native_tensor_parallel,
    supports_native_tp,
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


# ---------------------------------------------------------------------------
# apply_native_tensor_parallel (no dist init: parallelize_module is mocked)
# ---------------------------------------------------------------------------


def _fake_tp_mesh(tp_size=2):
    mesh = MagicMock()
    mesh.size.return_value = tp_size
    mesh.get_group.return_value = "fake_tp_group"
    return mesh


@pytest.fixture
def captured_plans(monkeypatch):
    """Mock parallelize_module and record (module, plan) for every call."""
    calls = []

    def fake_parallelize(module, mesh, plan):
        calls.append((module, plan))
        return module

    monkeypatch.setattr("torch.distributed.tensor.parallel.parallelize_module", fake_parallelize)
    return calls


class TestSupportsNativeTp:
    def test_true_for_wxformer_next(self):
        model = NextGenWXFormer(**_tiny_model_conf())
        assert supports_native_tp(model) is True

    def test_false_for_conv_transformer(self):
        assert supports_native_tp(ConvTransformer(16, **TINY_TRANSFORMER_KW)) is False

    def test_false_for_plain_module(self):
        assert supports_native_tp(nn.Linear(4, 4)) is False


class TestApplyNativeTensorParallel:
    def test_plan_construction_attention(self, captured_plans):
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        block = Attention(16, attn_type="short", window_size=4, dim_head=4)
        apply_native_tensor_parallel(block, _fake_tp_mesh(2))

        assert len(captured_plans) == 1
        _, plan = captured_plans[0]
        assert set(plan) == {"to_q", "to_k", "to_v", "to_out"}
        assert all(isinstance(plan[p], ColwiseParallel) for p in ("to_q", "to_k", "to_v"))
        assert isinstance(plan["to_out"], RowwiseParallel)

    def test_plan_construction_feedforward(self, captured_plans):
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        block = FeedForward(16)
        apply_native_tensor_parallel(block, _fake_tp_mesh(2))

        _, plan = captured_plans[0]
        assert isinstance(plan["layers.1"], ColwiseParallel)
        assert isinstance(plan["layers.4"], RowwiseParallel)

    def test_full_model_parallelizes_all_blocks(self, captured_plans):
        model = NextGenWXFormer(**_tiny_model_conf())
        out = apply_native_tensor_parallel(model, _fake_tp_mesh(2))

        # 4 stages x depth 1 x (2 attention + 2 FFN) blocks each
        assert len(captured_plans) == 16
        assert out is model
        assert model._tp_group == "fake_tp_group"

    def test_heads_not_divisible_raises(self, captured_plans):
        # dim=16, dim_head=4 -> heads=4; tp=3 does not divide 4
        block = Attention(16, attn_type="short", window_size=4, dim_head=4)
        with pytest.raises(ValueError, match="heads=4 not divisible by tp_size=3"):
            apply_native_tensor_parallel(block, _fake_tp_mesh(3))
        assert not captured_plans

    def test_ffn_width_not_divisible_raises(self, captured_plans):
        # dim=16, mult=4 -> out_features=64; tp=5 does not divide 64
        block = FeedForward(16)
        with pytest.raises(ValueError, match="not divisible by tp_size=5"):
            apply_native_tensor_parallel(block, _fake_tp_mesh(5))

    def test_spectral_norm_block_skipped_not_raised(self, captured_plans, caplog):
        block = Attention(16, attn_type="short", window_size=4, dim_head=4)
        apply_spectral_norm(block)
        with caplog.at_level(logging.WARNING, logger="credit.parallel.tensor_parallel"):
            out = apply_native_tensor_parallel(block, _fake_tp_mesh(2))
        assert not captured_plans
        assert "NO effect" in caplog.text
        assert "use_spectral_norm" in caplog.text
        # group still stashed so the trainer keeps the replicas pinned
        assert out._tp_group == "fake_tp_group"
        assert out._tp_native is True

    def test_partial_spectral_norm_skips_only_wrapped_blocks(self, captured_plans, caplog):
        # depth=2 -> 8 _tp_plan blocks (2 attention + 2 FFN per depth layer).
        # SN on the first depth layer's short attention + short FFN: those two
        # blocks skip, the other six shard.
        model = LinearTransformer(16, **TINY_TRANSFORMER_KW)
        apply_spectral_norm(model.layers[0][0])
        apply_spectral_norm(model.layers[0][1])
        with caplog.at_level(logging.WARNING, logger="credit.parallel.tensor_parallel"):
            apply_native_tensor_parallel(model, _fake_tp_mesh(2))
        assert len(captured_plans) == 6
        skipped = {id(model.layers[0][0]), id(model.layers[0][1])}
        assert all(id(m) not in skipped for m, _ in captured_plans)
        assert "6 block(s) sharded, 2 block(s) skipped" in caplog.text
        assert "use_spectral_norm: false" in caplog.text
        assert model._tp_group == "fake_tp_group"

    def test_full_model_with_spectral_norm_shards_nothing(self, captured_plans, caplog):
        # The real default config: use_spectral_norm=True wraps every Linear
        # inside the encoder Transformer stages, i.e. ALL _tp_plan layers, so
        # reduced TP is a documented no-op until spectral norm is disabled.
        model = NextGenWXFormer(**_tiny_model_conf(use_spectral_norm=True))
        with caplog.at_level(logging.WARNING, logger="credit.parallel.tensor_parallel"):
            out = apply_native_tensor_parallel(model, _fake_tp_mesh(2))
        assert not captured_plans
        assert "all 16 _tp_plan block(s)" in caplog.text
        assert "NO effect" in caplog.text
        assert out._tp_group == "fake_tp_group"

    def test_skipped_block_not_constraint_checked(self, captured_plans):
        # heads=4 is not divisible by tp=3, but the block is SN-wrapped and
        # skipped, so the constraint must not fire (it stays replicated).
        block = Attention(16, attn_type="short", window_size=4, dim_head=4)
        apply_spectral_norm(block)
        apply_native_tensor_parallel(block, _fake_tp_mesh(3))
        assert not captured_plans

    def test_parametrize_style_spectral_norm_also_skipped(self, captured_plans):
        # The wxformer apply_spectral_norm uses the hook style (weight_orig);
        # the parametrize-based variant must be detected too.
        block = FeedForward(16)
        nn.utils.parametrizations.spectral_norm(block.layers[1])
        apply_native_tensor_parallel(block, _fake_tp_mesh(2))
        assert not captured_plans

    def test_non_linear_target_raises(self, captured_plans):
        class BadBlock(nn.Module):
            _tp_plan = {"proj": "colwise"}

            def __init__(self):
                super().__init__()
                self.proj = nn.Conv2d(4, 8, 1)

        with pytest.raises(TypeError, match="requires nn.Linear"):
            apply_native_tensor_parallel(BadBlock(), _fake_tp_mesh(2))

    def test_no_optin_blocks_raises(self, captured_plans):
        with pytest.raises(ValueError, match="no blocks with _tp_plan"):
            apply_native_tensor_parallel(nn.Linear(4, 4), _fake_tp_mesh(2))


_REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])


def _gloo_tp_worker(rank, world, port, result_q, partial_sn=False):
    """Spawned worker: serial vs tp=2 forward/backward on a real gloo mesh.

    With partial_sn=True, spectral norm wraps the first depth layer's short
    attention + short FFN on BOTH models: those two blocks must be skipped
    (replicated full-width) while the other six shard, and the composed
    forward must still match serial."""
    import sys

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    import os

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from credit.models.wxformer.crossformer import apply_spectral_norm as _apply_sn
        from credit.models.wxformer.wxformer_next import Transformer
        from credit.parallel.tensor_parallel import _is_tp_sharded_param, apply_native_tensor_parallel

        kw = dict(local_window_size=4, global_window_size=2, depth=2, dim_head=4)
        torch.manual_seed(0)
        serial = Transformer(16, **kw).eval()
        tp_model = Transformer(16, **kw).eval()
        if partial_sn:
            for m in (serial, tp_model):
                _apply_sn(m.layers[0][0])
                _apply_sn(m.layers[0][1])
        tp_model.load_state_dict(serial.state_dict())

        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))
        tp_model = apply_native_tensor_parallel(tp_model, mesh["tp"])

        torch.manual_seed(7)
        x = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            diff = (serial(x) - tp_model(x)).abs().max().item()

        tp_model.train()
        tp_model(x).mean().backward()
        n_sharded = sum(1 for p in tp_model.parameters() if _is_tp_sharded_param(p))
        n_grads = sum(1 for p in tp_model.parameters() if p.grad is not None)
        n_total = sum(1 for p in tp_model.parameters())
        if rank == 0:
            result_q.put(("ok", diff, n_sharded, n_grads, n_total))
    except Exception as exc:  # surface worker failures to the test process
        if rank == 0:
            result_q.put(("error", repr(exc)))
        raise
    finally:
        dist.destroy_process_group()


class TestGlooTpParity:
    """Real 2-process DTensor run on CPU/gloo: the wiring test the mocked
    plan-construction tests cannot cover. tp=2 must reproduce the serial
    forward (to fp32 reassociation) and every param must receive a grad."""

    @staticmethod
    def _run(port, partial_sn=False):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        result_q = ctx.Queue()
        world = 2
        procs = [ctx.Process(target=_gloo_tp_worker, args=(r, world, port, result_q, partial_sn)) for r in range(world)]
        for p in procs:
            p.start()
        try:
            result = result_q.get(timeout=600)
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()

        assert result[0] == "ok", f"worker failed: {result[1]}"
        return result[1:]

    def test_tp2_matches_serial_forward_and_backward(self):
        diff, n_sharded, n_grads, n_total = self._run(29637)
        assert diff < 1e-5, f"tp=2 diverges from serial: max diff {diff}"
        # 2 depth layers x (2 attn x [q,k,v,out weights] + 2 FFN x [up w+b, down w])
        assert n_sharded == 28
        assert n_grads == n_total, f"missing grads: {n_grads}/{n_total}"

    def test_tp2_partial_spectral_norm_matches_serial(self):
        # Reduced TP: SN on the first short attention (4 sharded params lost)
        # and first short FFN (3 lost) — 21 sharded, the SN blocks replicated.
        # Composition both ways (sharded block -> replicated block -> sharded
        # block) must still reproduce the serial forward, and grads must reach
        # every param including the SN weight_orig tensors.
        diff, n_sharded, n_grads, n_total = self._run(29638, partial_sn=True)
        assert diff < 1e-5, f"reduced tp=2 diverges from serial: max diff {diff}"
        assert n_sharded == 28 - 7
        assert n_grads == n_total, f"missing grads: {n_grads}/{n_total}"


class TestIsTpShardedParam:
    def _fake_dtensor(self, names, shard_flags):
        p = MagicMock()
        p.device_mesh.mesh_dim_names = names
        placements = []
        for flag in shard_flags:
            pl = MagicMock()
            pl.is_shard.return_value = flag
            placements.append(pl)
        p.placements = tuple(placements)
        return p

    def test_plain_tensor_is_not_sharded(self):
        assert _is_tp_sharded_param(torch.randn(3)) is False

    def test_tp_shard_detected_1d_mesh(self):
        p = self._fake_dtensor(("tp",), (True,))
        assert _is_tp_sharded_param(p) is True

    def test_tp_replicate_not_sharded(self):
        # RowwiseParallel bias: Replicate on the tp mesh
        p = self._fake_dtensor(("tp",), (False,))
        assert _is_tp_sharded_param(p) is False

    def test_2d_mesh_dp_shard_only_not_tp_sharded(self):
        # FSDP2-only param after composition: sharded on dp, replicated on tp
        p = self._fake_dtensor(("dp", "tp"), (True, False))
        assert _is_tp_sharded_param(p) is False

    def test_2d_mesh_tp_shard_detected(self):
        p = self._fake_dtensor(("dp", "tp"), (True, True))
        assert _is_tp_sharded_param(p) is True
