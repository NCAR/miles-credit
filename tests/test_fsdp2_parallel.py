"""CPU-only unit tests for FSDP2 / tensor-parallel / domain-parallel infrastructure.

These tests cover:
  - Mesh dimension math (no dist init required)
  - _shard_spatial correctness with a mock DomainParallelManager
  - _unpad_shard_interp shape recovery
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
# _shard_spatial
# ---------------------------------------------------------------------------

from credit.trainers.trainerERA5v2 import _shard_spatial  # noqa: E402


class TestShardSpatial:
    def test_no_manager_passthrough(self):
        x = torch.randn(2, 4, 8, 16)
        assert _shard_spatial(x, None) is x

    def test_single_domain_passthrough(self):
        m = _make_manager(0, 1)
        x = torch.randn(2, 4, 8, 16)
        assert _shard_spatial(x, m) is x

    def test_shard_splits_h_evenly(self):
        H, W = 8, 16
        x = torch.arange(H * W).float().reshape(1, 1, H, W)
        for rank in range(2):
            m = _make_manager(rank, 2)
            shard = _shard_spatial(x, m)
            assert shard.shape == (1, 1, H // 2, W)
            expected = x[..., rank * (H // 2) : (rank + 1) * (H // 2), :]
            assert torch.equal(shard, expected)

    def test_indivisible_h_raises(self):
        m = _make_manager(0, 3)
        x = torch.randn(1, 1, 8, 16)  # 8 % 3 != 0
        with pytest.raises(ValueError, match="divisible"):
            _shard_spatial(x, m)

    def test_4d_and_5d_input(self):
        m = _make_manager(0, 2)
        x4 = torch.randn(2, 3, 8, 16)
        x5 = torch.randn(2, 3, 1, 8, 16)
        s4 = _shard_spatial(x4, m)
        s5 = _shard_spatial(x5, m)
        assert s4.shape == (2, 3, 4, 16)
        assert s5.shape == (2, 3, 1, 4, 16)


# ---------------------------------------------------------------------------
# _unpad_shard_interp
# ---------------------------------------------------------------------------

from credit.trainers.trainerERA5v2 import _unpad_shard_interp  # noqa: E402


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
        out = _unpad_shard_interp(x, p, m, image_h=8, image_w=16)
        assert out.shape == (1, 3, 4, 16)

    def test_w_padding_removed(self):
        m = _make_manager(0, 2)
        p = _make_padding_opt(pad_we=(2, 3))
        # width = 16 + 2 + 3 = 21 (padded), should return width = 16
        x = torch.randn(1, 3, 4, 21)
        out = _unpad_shard_interp(x, p, m, image_h=8, image_w=16)
        assert out.shape[-1] == 16

    def test_h_padding_removed_first_rank(self):
        """First domain rank should lose top padding rows."""
        m = _make_manager(0, 2)  # first rank
        p = _make_padding_opt(pad_ns=(2, 2))
        # shard_h target = 8 // 2 = 4; input has extra rows from top+bot padding
        # First rank: lose top=2, keep bot (interior boundary)
        x = torch.randn(1, 3, 8, 16)
        out = _unpad_shard_interp(x, p, m, image_h=8, image_w=16)
        assert out.shape[-2] == 4

    def test_5d_input_unsqueezed(self):
        m = _make_manager(0, 2)
        p = _make_padding_opt()
        x = torch.randn(1, 3, 1, 4, 16)
        out = _unpad_shard_interp(x, p, m, image_h=8, image_w=16)
        assert out.dim() == 5
        assert out.shape == (1, 3, 1, 4, 16)


# ---------------------------------------------------------------------------
# Parallelism config math (no dist init needed)
# ---------------------------------------------------------------------------


class TestParallelismConfigMath:
    """Verify dp_size = world_size // (tp * domain) arithmetic."""

    def _dp_size(self, world_size, tp, domain):
        total = tp * domain
        if world_size % total != 0:
            raise ValueError
        return world_size // total

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
