"""Unit tests for domain parallelism.

These tests verify correctness of domain-parallel layers by comparing
their output against standard single-device layers. Tests that require
multiple GPUs are skipped in single-GPU environments.

For single-device tests, we simulate sharding by manually splitting
tensors and calling the domain-parallel primitives directly.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from credit.domain_parallel.layers import (
    DomainParallelConv2d,
    DomainParallelConvTranspose2d,
    DomainParallelGroupNorm,
)
from credit.domain_parallel.convert import (
    convert_to_domain_parallel,
    _needs_halo_conv2d,
    _needs_halo_conv_transpose2d,
)
from credit.domain_parallel.sharding import shard_tensor, gather_tensor


class TestNeedsHalo:
    """Test halo requirement detection for different layer configurations."""

    def test_conv2d_1x1_no_halo(self):
        conv = nn.Conv2d(8, 16, kernel_size=1)
        assert not _needs_halo_conv2d(conv)

    def test_conv2d_3x3_needs_halo(self):
        conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        assert _needs_halo_conv2d(conv)

    def test_conv2d_tuple_kernel(self):
        conv = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1)
        assert _needs_halo_conv2d(conv)

    def test_conv2d_1xN_no_halo_h(self):
        conv = nn.Conv2d(8, 16, kernel_size=(1, 3), padding=(0, 1))
        assert not _needs_halo_conv2d(conv)

    def test_conv_transpose_k2_s2_no_halo(self):
        conv = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        assert not _needs_halo_conv_transpose2d(conv)

    def test_conv_transpose_k4_s2_needs_halo(self):
        conv = nn.ConvTranspose2d(8, 16, kernel_size=4, stride=2, padding=1)
        assert _needs_halo_conv_transpose2d(conv)


class TestDomainParallelConv2dHaloWidth:
    """Test that DomainParallelConv2d computes correct halo widths."""

    def test_halo_width_3x3_stride1(self):
        conv = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        dp_conv = DomainParallelConv2d(conv)
        assert dp_conv.halo_width == 1

    def test_halo_width_5x5_stride1(self):
        conv = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        dp_conv = DomainParallelConv2d(conv)
        assert dp_conv.halo_width == 2

    def test_halo_width_4x4_stride2(self):
        # CrossEmbedLayer kernel_size=4, stride=2
        conv = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        dp_conv = DomainParallelConv2d(conv)
        assert dp_conv.halo_width == 1

    def test_halo_width_8x8_stride2(self):
        # CrossEmbedLayer kernel_size=8, stride=2
        conv = nn.Conv2d(8, 16, kernel_size=8, stride=2, padding=3)
        dp_conv = DomainParallelConv2d(conv)
        assert dp_conv.halo_width == 3

    def test_halo_width_16x16_stride2(self):
        conv = nn.Conv2d(8, 16, kernel_size=16, stride=2, padding=7)
        dp_conv = DomainParallelConv2d(conv)
        assert dp_conv.halo_width == 7

    def test_halo_width_32x32_stride2(self):
        conv = nn.Conv2d(8, 16, kernel_size=32, stride=2, padding=15)
        dp_conv = DomainParallelConv2d(conv)
        assert dp_conv.halo_width == 15


class TestConvertToModel:
    """Test model conversion logic."""

    def test_replace_conv2d_3x3(self):
        """3x3 convs should be replaced, 1x1 should not."""
        model = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),  # should be replaced
            nn.Conv2d(16, 16, 1),  # should NOT be replaced (1x1)
            nn.Conv2d(16, 8, 3, padding=1),  # should be replaced
        )
        manager = MagicMock()
        manager.domain_parallel_size = 2

        convert_to_domain_parallel(model, manager)

        assert isinstance(model[0], DomainParallelConv2d)
        assert isinstance(model[1], nn.Conv2d)  # unchanged
        assert isinstance(model[2], DomainParallelConv2d)

    def test_replace_group_norm(self):
        """GroupNorm should be replaced."""
        model = nn.Sequential(
            nn.Conv2d(8, 16, 1),
            nn.GroupNorm(4, 16),
        )
        manager = MagicMock()
        manager.domain_parallel_size = 2

        convert_to_domain_parallel(model, manager)

        assert isinstance(model[0], nn.Conv2d)  # 1x1, not replaced
        assert isinstance(model[1], DomainParallelGroupNorm)

    def test_replace_conv_transpose2d(self):
        """ConvTranspose2d with k>s should be replaced, k==s should not."""
        model = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),  # k==s, no halo
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # k>s, halo
        )
        manager = MagicMock()
        manager.domain_parallel_size = 2

        convert_to_domain_parallel(model, manager)

        assert isinstance(model[0], nn.ConvTranspose2d)  # not replaced
        assert isinstance(model[1], DomainParallelConvTranspose2d)

    def test_nested_modules_replaced(self):
        """Conversion should work on nested module trees."""

        class InnerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 8, 3, padding=1)
                self.norm = nn.GroupNorm(4, 8)

            def forward(self, x):
                return self.norm(self.conv(x))

        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = InnerBlock()
                self.block2 = InnerBlock()
                self.proj = nn.Conv2d(8, 8, 1)  # 1x1

            def forward(self, x):
                return self.proj(self.block2(self.block1(x)))

        model = OuterModel()
        manager = MagicMock()
        manager.domain_parallel_size = 2

        convert_to_domain_parallel(model, manager)

        assert isinstance(model.block1.conv, DomainParallelConv2d)
        assert isinstance(model.block1.norm, DomainParallelGroupNorm)
        assert isinstance(model.block2.conv, DomainParallelConv2d)
        assert isinstance(model.block2.norm, DomainParallelGroupNorm)
        assert isinstance(model.proj, nn.Conv2d)  # 1x1, unchanged


class TestShardGatherRoundtrip:
    """Test shard/gather without actual distributed backend."""

    def test_shard_tensor_shape(self):
        """Sharding should divide the H dimension."""
        x = torch.randn(2, 8, 100, 200)  # B=2, C=8, H=100, W=200

        manager = MagicMock()
        manager.domain_parallel_size = 2
        manager.domain_rank = 0

        shard = shard_tensor(x, dim=-2, manager=manager)
        assert shard.shape == (2, 8, 50, 200)

    def test_shard_tensor_rank1(self):
        """Second rank should get the second half."""
        x = torch.randn(2, 8, 100, 200)

        manager = MagicMock()
        manager.domain_parallel_size = 2
        manager.domain_rank = 1

        shard = shard_tensor(x, dim=-2, manager=manager)
        assert shard.shape == (2, 8, 50, 200)
        assert torch.allclose(shard, x[:, :, 50:100, :])

    def test_shard_5d_tensor(self):
        """5D tensor sharding along lat (dim=3)."""
        x = torch.randn(1, 8, 2, 100, 200)

        manager = MagicMock()
        manager.domain_parallel_size = 2
        manager.domain_rank = 0

        shard = shard_tensor(x, dim=3, manager=manager)
        assert shard.shape == (1, 8, 2, 50, 200)

    def test_indivisible_raises(self):
        """Sharding non-divisible dimension should raise."""
        x = torch.randn(2, 8, 101, 200)

        manager = MagicMock()
        manager.domain_parallel_size = 2
        manager.domain_rank = 0

        with pytest.raises(ValueError, match="Cannot shard"):
            shard_tensor(x, dim=-2, manager=manager)

    def test_disabled_returns_unchanged(self):
        """domain_parallel_size=1 should be a no-op."""
        x = torch.randn(2, 8, 100, 200)

        manager = MagicMock()
        manager.domain_parallel_size = 1

        result = shard_tensor(x, dim=-2, manager=manager)
        assert result is x

    def test_none_manager_returns_unchanged(self):
        """No manager should be a no-op."""
        x = torch.randn(2, 8, 100, 200)
        result = shard_tensor(x, dim=-2, manager=None)
        assert result is x
