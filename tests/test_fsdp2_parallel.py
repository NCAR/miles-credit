"""CPU-only unit tests for FSDP2 / tensor-parallel / domain-parallel infrastructure.

These tests cover:
  - Mesh dimension math (no dist init required)
  - shard_spatial correctness with a mock DomainParallelManager
  - unpad_shard_interp shape recovery
  - Parallelism config validation
"""

import pytest
import torch
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(rank, n):
    """Mock DomainParallelManager for rank `rank` of `n` total domain ranks."""
    m = MagicMock()
    m.domain_parallel_size = n
    m.domain_rank = rank
    m.is_first_domain_rank = rank == 0
    m.is_last_domain_rank = rank == n - 1
    # domain_slice returns the row slice this rank owns
    shard_h = None  # set per-test via tensor shape

    def _domain_slice(h):
        sh = h // n
        return slice(rank * sh, (rank + 1) * sh)

    m.domain_slice.side_effect = lambda h: _domain_slice(h)
    return m


# ---------------------------------------------------------------------------
# shard_spatial
# ---------------------------------------------------------------------------

from credit.parallel.domain import shard_spatial, unpad_shard_interp  # noqa: E402


class TestShardSpatial:
    def test_no_manager_passthrough(self):
        x = torch.randn(2, 4, 8, 16)
        assert shard_spatial(x, None) is x

    def test_single_domain_passthrough(self):
        m = _make_manager(0, 1)
        x = torch.randn(2, 4, 8, 16)
        assert shard_spatial(x, m) is x

    def test_shard_splits_h_evenly(self):
        H, W = 8, 16
        x = torch.arange(H * W).float().reshape(1, 1, H, W)
        for rank in range(2):
            m = _make_manager(rank, 2)
            shard = shard_spatial(x, m)
            assert shard.shape == (1, 1, H // 2, W)
            expected = x[..., rank * (H // 2) : (rank + 1) * (H // 2), :]
            assert torch.equal(shard, expected)

    def test_indivisible_h_raises(self):
        m = _make_manager(0, 3)
        x = torch.randn(1, 1, 8, 16)  # 8 % 3 != 0
        with pytest.raises(ValueError, match="divisible"):
            shard_spatial(x, m)

    def test_4d_and_5d_input(self):
        m = _make_manager(0, 2)
        x4 = torch.randn(2, 3, 8, 16)
        x5 = torch.randn(2, 3, 1, 8, 16)
        s4 = shard_spatial(x4, m)
        s5 = shard_spatial(x5, m)
        assert s4.shape == (2, 3, 4, 16)
        assert s5.shape == (2, 3, 1, 4, 16)


# ---------------------------------------------------------------------------
# unpad_shard_interp
# ---------------------------------------------------------------------------


def _make_padding_opt(pad_ns=(0, 0), pad_we=(0, 0)):
    p = MagicMock()
    p.pad_NS = pad_ns
    p.pad_WE = pad_we
    return p


class TestUnpadShardInterp:
    def test_no_padding_no_resize(self):
        """With zero padding and exact shard size, tensor passes through unchanged."""
        m = _make_manager(0, 2)
        p = _make_padding_opt()
        x = torch.randn(1, 3, 4, 16)  # already shard_h=4 of total H=8
        out = unpad_shard_interp(x, p, m, image_h=8, image_w=16)
        assert out.shape == (1, 3, 4, 16)

    def test_w_padding_removed(self):
        m = _make_manager(0, 2)
        p = _make_padding_opt(pad_we=(2, 3))
        # width = 16 + 2 + 3 = 21 (padded), should return width = 16
        x = torch.randn(1, 3, 4, 21)
        out = unpad_shard_interp(x, p, m, image_h=8, image_w=16)
        assert out.shape[-1] == 16

    def test_h_padding_removed_first_rank(self):
        """First domain rank should lose top padding rows."""
        m = _make_manager(0, 2)  # first rank
        p = _make_padding_opt(pad_ns=(2, 2))
        # shard_h target = 8 // 2 = 4; input has extra rows from top+bot padding
        # First rank: lose top=2, keep bot (interior boundary)
        x = torch.randn(1, 3, 8, 16)
        out = unpad_shard_interp(x, p, m, image_h=8, image_w=16)
        assert out.shape[-2] == 4

    def test_5d_input_unsqueezed(self):
        m = _make_manager(0, 2)
        p = _make_padding_opt()
        x = torch.randn(1, 3, 1, 4, 16)
        out = unpad_shard_interp(x, p, m, image_h=8, image_w=16)
        assert out.dim() == 5
        assert out.shape == (1, 3, 1, 4, 16)


# ---------------------------------------------------------------------------
# Parallelism config math (no dist init needed)
# ---------------------------------------------------------------------------


class TestParallelismConfigMath:
    """Verify dp_size = world_size // (tp * domain) arithmetic (the real helper)."""

    def _dp_size(self, world_size, tp, domain):
        from credit.parallel.mesh import dp_world_size

        return dp_world_size({"tensor": tp, "domain": domain}, world_size)

    def test_fsdp2_only_4gpu(self):
        assert self._dp_size(4, 1, 1) == 4

    def test_tp2_4gpu(self):
        assert self._dp_size(4, 2, 1) == 2

    def test_domain2_4gpu(self):
        assert self._dp_size(4, 1, 2) == 2

    def test_tp2_domain2_8gpu(self):
        assert self._dp_size(8, 2, 2) == 2

    def test_indivisible_raises(self):
        with pytest.raises(ValueError):
            self._dp_size(4, 3, 1)  # 4 % 3 != 0


# ---------------------------------------------------------------------------
# data_parallel_coords — the dataloader sampler contract
# ---------------------------------------------------------------------------

from credit.parallel import mesh as mesh_mod  # noqa: E402
from credit.parallel.mesh import data_parallel_coords, parse_parallelism_conf  # noqa: E402


class TestDataParallelCoords:
    """The sampler must shard data over the dp coordinate, not the global rank.

    Layout matches init_device_mesh / DomainParallelManager: row-major over
    (dp, tp, domain) with domain innermost, so dp_rank = g // (tp * domain).
    """

    def _coords(self, monkeypatch, world, rank, tp, domain):
        conf = {"trainer": {"parallelism": {"data": "fsdp2", "tensor": tp, "domain": domain}}}
        fake = MagicMock()
        fake.is_initialized.return_value = True
        fake.get_world_size.return_value = world
        fake.get_rank.return_value = rank
        monkeypatch.setattr(mesh_mod, "dist", fake)
        return data_parallel_coords(conf)

    def test_no_dist_falls_back(self):
        conf = {"trainer": {"parallelism": {"data": "none", "tensor": 1, "domain": 1}}}
        assert data_parallel_coords(conf) == (0, 1)

    def test_missing_parallelism_raises(self):
        with pytest.raises(ValueError, match="trainer.parallelism"):
            parse_parallelism_conf({"trainer": {"mode": "fsdp"}})

    def test_tp2_peers_share_dp_rank(self, monkeypatch):
        # 4 GPUs, tp=2: ranks (0,1) form one tp group → same dp_rank; (2,3) the other.
        assert self._coords(monkeypatch, 4, 0, 2, 1) == (0, 2)
        assert self._coords(monkeypatch, 4, 1, 2, 1) == (0, 2)
        assert self._coords(monkeypatch, 4, 2, 2, 1) == (1, 2)
        assert self._coords(monkeypatch, 4, 3, 2, 1) == (1, 2)

    def test_domain2_peers_share_dp_rank(self, monkeypatch):
        # Matches DomainParallelManager: domain groups are consecutive ranks.
        assert self._coords(monkeypatch, 4, 0, 1, 2) == (0, 2)
        assert self._coords(monkeypatch, 4, 1, 1, 2) == (0, 2)
        assert self._coords(monkeypatch, 4, 2, 1, 2) == (1, 2)
        assert self._coords(monkeypatch, 4, 3, 1, 2) == (1, 2)

    def test_tp2_domain2_8gpu(self, monkeypatch):
        # world=8, tp*domain=4: ranks 0-3 are dp_rank 0, ranks 4-7 are dp_rank 1.
        for g in range(4):
            assert self._coords(monkeypatch, 8, g, 2, 2) == (0, 2)
        for g in range(4, 8):
            assert self._coords(monkeypatch, 8, g, 2, 2) == (1, 2)

    def test_indivisible_raises(self, monkeypatch):
        with pytest.raises(ValueError):
            self._coords(monkeypatch, 4, 0, 3, 1)


# ---------------------------------------------------------------------------
# Tensor-parallel col→row parity (serial simulation of tp=2)
# ---------------------------------------------------------------------------

from credit.parallel import tensor_parallel as tp_mod  # noqa: E402
from credit.parallel.tensor_parallel import (  # noqa: E402
    TpColConv2d,
    TpColLinear,
    TpRowConv2d,
    TpRowLinear,
)


def _fake_tp_dist(monkeypatch, tp_size, tp_rank):
    fake = MagicMock()
    fake.get_world_size.return_value = tp_size
    fake.get_rank.return_value = tp_rank
    monkeypatch.setattr(tp_mod, "dist", fake)
    return fake


class TestTensorParallelParity:
    """Sum of per-rank partial outputs (+ bias once) must equal the serial layer.

    This is the regression test for the row-parallel bias bug: the bias must be
    a single replicated parameter added AFTER the all_reduce, not baked into the
    per-rank conv (which trains the effective bias at tp× the learning rate).
    """

    def test_conv_col_row_parity_tp2(self, monkeypatch):
        torch.manual_seed(0)
        up = torch.nn.Conv2d(8, 16, 1, bias=True)
        down = torch.nn.Conv2d(16, 8, 1, bias=True)
        x = torch.randn(2, 8, 4, 4)
        expected = down(up(x))

        partials = []
        rows = []
        for r in range(2):
            _fake_tp_dist(monkeypatch, 2, r)
            col = TpColConv2d(up, tp_group=None)
            row = TpRowConv2d(down, tp_group=None)
            rows.append(row)
            partials.append(row.conv(col(x)))  # partial, pre-reduce, no bias

        out = partials[0] + partials[1] + rows[0].bias.view(1, -1, 1, 1)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_linear_col_row_parity_tp2(self, monkeypatch):
        torch.manual_seed(0)
        up = torch.nn.Linear(8, 16, bias=True)
        down = torch.nn.Linear(16, 8, bias=True)
        x = torch.randn(2, 8)
        expected = down(up(x))

        partials = []
        rows = []
        for r in range(2):
            _fake_tp_dist(monkeypatch, 2, r)
            col = TpColLinear(up, tp_group=None)
            row = TpRowLinear(down, tp_group=None)
            rows.append(row)
            partials.append(row.linear(col(x)))

        out = partials[0] + partials[1] + rows[0].bias
        assert torch.allclose(out, expected, atol=1e-5)

    def test_row_bias_is_replicated_not_split(self, monkeypatch):
        """Every rank holds the FULL bias; the inner layer has none."""
        conv = torch.nn.Conv2d(8, 16, 1, bias=True)
        for r in range(2):
            _fake_tp_dist(monkeypatch, 2, r)
            row = TpRowConv2d(conv, tp_group=None)
            assert row.conv.bias is None
            assert torch.equal(row.bias, conv.bias)

    def test_non_default_conv_attrs_rejected(self, monkeypatch):
        _fake_tp_dist(monkeypatch, 2, 0)
        strided = torch.nn.Conv2d(8, 16, 1, stride=2)
        with pytest.raises(AssertionError):
            TpColConv2d(strided, tp_group=None)
        grouped = torch.nn.Conv2d(8, 16, 1, groups=2)
        with pytest.raises(AssertionError):
            TpRowConv2d(grouped, tp_group=None)


# ---------------------------------------------------------------------------
# apply_tensor_parallel is disabled pending the native-TP rewrite (issue #415)
# ---------------------------------------------------------------------------


class TestApplyTensorParallelDisabled:
    def test_raises_not_implemented(self):
        from credit.parallel.tensor_parallel import apply_tensor_parallel

        with pytest.raises(NotImplementedError, match="#415"):
            apply_tensor_parallel(torch.nn.Linear(4, 4), tp_mesh=None)


# ---------------------------------------------------------------------------
# shard_lat_weights — shared lat-weight sharding (loss + metrics)
# ---------------------------------------------------------------------------


class TestShardLatWeights:
    def test_matching_height_passthrough(self):
        from credit.parallel.domain import shard_lat_weights

        w = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
        assert shard_lat_weights(w, 6) is w

    def test_mismatch_without_manager_raises(self):
        from credit.parallel.domain import shard_lat_weights

        w = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
        with pytest.raises(ValueError, match="does not match"):
            shard_lat_weights(w, 3)

    def test_domain_shard_selected(self, monkeypatch):
        import credit.domain_parallel.manager as manager_mod
        from credit.parallel.domain import shard_lat_weights

        mgr = _make_manager(rank=1, n=2)
        monkeypatch.setattr(manager_mod, "get_domain_parallel_manager", lambda: mgr)
        w = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
        out = shard_lat_weights(w, 3)
        assert torch.equal(out.flatten(), torch.tensor([3.0, 4.0, 5.0]))

    def test_indivisible_raises(self, monkeypatch):
        import credit.domain_parallel.manager as manager_mod
        from credit.parallel.domain import shard_lat_weights

        mgr = _make_manager(rank=0, n=4)
        monkeypatch.setattr(manager_mod, "get_domain_parallel_manager", lambda: mgr)
        w = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
        with pytest.raises(ValueError, match="not divisible"):
            shard_lat_weights(w, 1)


# ---------------------------------------------------------------------------
# gather_spatial — between-step gather for domain-parallel multistep rollout
# ---------------------------------------------------------------------------


class TestGatherSpatial:
    def test_no_manager_passthrough(self):
        from credit.parallel.domain import gather_spatial

        x = torch.randn(1, 2, 4, 4)
        assert gather_spatial(x, None) is x

    def test_single_domain_passthrough(self):
        from credit.parallel.domain import gather_spatial

        x = torch.randn(1, 2, 4, 4)
        assert gather_spatial(x, _make_manager(0, 1)) is x


# ---------------------------------------------------------------------------
# fsdp2 helpers
# ---------------------------------------------------------------------------


class TestFsdp2Helpers:
    def test_fsdp2_is_applied_false_on_plain_module(self):
        from credit.parallel.fsdp2 import fsdp2_is_applied

        assert fsdp2_is_applied(torch.nn.Linear(4, 4)) is False

    def test_mp_policy_accepts_parse_dtype_aliases(self):
        """fsdp2_mp_policy goes through credit.mixed_precision.parse_dtype,
        so V1-accepted aliases (case-insensitive, 'half') must work."""
        from credit.parallel.fsdp2 import _build_mp_policy

        conf = {
            "trainer": {
                "amp": True,
                "fsdp2_mp_policy": {
                    "param_dtype": "Half",
                    "reduce_dtype": "FLOAT32",
                    "output_dtype": "bfloat16",
                },
            },
            "model": {"use_spectral_norm": False},
        }
        policy = _build_mp_policy(conf)
        assert policy.param_dtype == torch.float16
        assert policy.reduce_dtype == torch.float32

    def test_mp_policy_rejects_garbage_dtype(self):
        from credit.parallel.fsdp2 import _build_mp_policy

        conf = {
            "trainer": {"amp": True, "fsdp2_mp_policy": {"param_dtype": "floof32"}},
            "model": {"use_spectral_norm": False},
        }
        with pytest.raises(ValueError):
            _build_mp_policy(conf)
