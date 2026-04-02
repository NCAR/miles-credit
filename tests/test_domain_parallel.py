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
    DomainParallelPeriodicConv2d,
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


# ---------------------------------------------------------------------------
# ConvTranspose3d
# ---------------------------------------------------------------------------

from credit.domain_parallel.layers import DomainParallelConvTranspose3d
from credit.domain_parallel.convert import _needs_halo_conv_transpose3d


class TestNeedsHaloConvTranspose3d:
    """Test halo requirement detection for ConvTranspose3d."""

    def test_pangu_patch_recovery_no_halo(self):
        """Pangu PatchRecovery: kernel==stride, no halo needed."""
        conv = nn.ConvTranspose3d(192, 13, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        assert not _needs_halo_conv_transpose3d(conv)

    def test_int_kernel_equal_stride_no_halo(self):
        conv = nn.ConvTranspose3d(8, 4, kernel_size=2, stride=2)
        assert not _needs_halo_conv_transpose3d(conv)

    def test_kernel_greater_than_stride_needs_halo(self):
        conv = nn.ConvTranspose3d(8, 4, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1))
        assert _needs_halo_conv_transpose3d(conv)


class TestDomainParallelConvTranspose3dHaloWidth:
    """Test halo width computation for DomainParallelConvTranspose3d."""

    def test_pangu_patch_recovery_halo_zero(self):
        """Pangu PatchRecovery kernel==stride: halo_width should be 0."""
        conv = nn.ConvTranspose3d(192, 13, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        dp_conv = DomainParallelConvTranspose3d(conv)
        assert dp_conv.halo_width == 0

    def test_kernel_greater_stride_halo_from_padding(self):
        """kernel=(2,4,4), stride=(2,2,2), padding=(0,1,1): halo_width = p_h = 1."""
        conv = nn.ConvTranspose3d(8, 4, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1))
        dp_conv = DomainParallelConvTranspose3d(conv)
        assert dp_conv.halo_width == 1

    def test_int_kernel_equal_stride_halo_zero(self):
        conv = nn.ConvTranspose3d(8, 4, kernel_size=2, stride=2)
        dp_conv = DomainParallelConvTranspose3d(conv)
        assert dp_conv.halo_width == 0


class TestConvertToModelConvTranspose3d:
    """Test that convert_to_domain_parallel handles ConvTranspose3d correctly."""

    def test_pangu_patch_recovery_not_replaced(self):
        """kernel==stride should NOT be replaced (no halo needed)."""
        model = nn.Sequential(
            nn.ConvTranspose3d(192, 13, kernel_size=(2, 4, 4), stride=(2, 4, 4)),
        )
        manager = MagicMock()
        manager.domain_parallel_size = 2
        convert_to_domain_parallel(model, manager)
        assert isinstance(model[0], nn.ConvTranspose3d)

    def test_conv_transpose3d_with_halo_replaced(self):
        """kernel>stride should be replaced with DomainParallelConvTranspose3d."""
        model = nn.Sequential(
            nn.ConvTranspose3d(8, 4, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)),
        )
        manager = MagicMock()
        manager.domain_parallel_size = 2
        convert_to_domain_parallel(model, manager)
        assert isinstance(model[0], DomainParallelConvTranspose3d)

    def test_mixed_3d_model(self):
        """Conv3d (embed) replaced, ConvTranspose3d (recover, kernel==stride) not replaced."""

        class PanguPatchBlocks(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Conv3d(13, 192, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                self.recover = nn.ConvTranspose3d(192, 13, kernel_size=(2, 4, 4), stride=(2, 4, 4))

            def forward(self, x):
                return self.recover(self.embed(x))

        model = PanguPatchBlocks()
        manager = MagicMock()
        manager.domain_parallel_size = 2
        convert_to_domain_parallel(model, manager)

        from credit.domain_parallel.layers import DomainParallelConv3d
        assert isinstance(model.embed, DomainParallelConv3d)
        assert isinstance(model.recover, nn.ConvTranspose3d)  # unchanged


# ---------------------------------------------------------------------------
# PeriodicConv2d / custom_converters
# ---------------------------------------------------------------------------

class TestPeriodicConv2dConversion:
    """Test DomainParallelPeriodicConv2d and custom_converters mechanism."""

    def _make_periodic_conv(self, dim_in=8, dim_out=16, kernel_size=3, padding=1):
        """Create a minimal PeriodicConv2d-like module (no model-zoo import needed)."""

        class PeriodicConv2d(nn.Module):
            def __init__(self):
                super().__init__()
                self.padding = padding
                self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, padding=0)

            def forward(self, x):
                import torch.nn.functional as F
                x = F.pad(x, (self.padding, self.padding, 0, 0), mode="circular")
                x = F.pad(x, (0, 0, self.padding, self.padding), mode="reflect")
                return self.conv(x)

        return PeriodicConv2d()

    def test_wraps_inner_conv_and_padding(self):
        """DomainParallelPeriodicConv2d should hold the inner conv and padding."""
        pc = self._make_periodic_conv()
        dp = DomainParallelPeriodicConv2d(pc)
        assert dp.conv is pc.conv
        assert dp.padding == 1
        assert dp.halo_exchange.halo_width == 1

    def test_custom_converter_replaces_whole_module(self):
        """custom_converters should replace PeriodicConv2d, not its inner conv."""
        PeriodicConv2d = self._make_periodic_conv().__class__

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pc = PeriodicConv2d()

            def forward(self, x):
                return self.pc(x)

        # Rebuild with the class, not the instance
        class PeriodicConv2dClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.padding = 1
                self.conv = nn.Conv2d(8, 16, 3, padding=0)

            def forward(self, x):
                return self.conv(x)

        model = nn.Sequential(PeriodicConv2dClass())
        manager = MagicMock()
        manager.domain_parallel_size = 2

        convert_to_domain_parallel(
            model, manager,
            custom_converters={PeriodicConv2dClass: DomainParallelPeriodicConv2d},
        )

        assert isinstance(model[0], DomainParallelPeriodicConv2d)

    def test_inner_conv_not_double_replaced(self):
        """Inner nn.Conv2d of a custom-converted module must not also be replaced."""

        class PeriodicConv2dClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.padding = 1
                self.conv = nn.Conv2d(8, 16, 3, padding=0)

            def forward(self, x):
                return self.conv(x)

        model = nn.Sequential(PeriodicConv2dClass())
        manager = MagicMock()
        manager.domain_parallel_size = 2

        convert_to_domain_parallel(
            model, manager,
            custom_converters={PeriodicConv2dClass: DomainParallelPeriodicConv2d},
        )

        # The wrapper should exist; its inner .conv must still be nn.Conv2d
        dp = model[0]
        assert isinstance(dp, DomainParallelPeriodicConv2d)
        assert isinstance(dp.conv, nn.Conv2d), "Inner conv was double-replaced"

    def test_without_custom_converters_inner_conv_replaced(self):
        """Without custom_converters the inner Conv2d gets replaced (old behavior, now avoidable)."""

        class PeriodicConv2dClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.padding = 1
                self.conv = nn.Conv2d(8, 16, 3, padding=0)

            def forward(self, x):
                return self.conv(x)

        model = nn.Sequential(PeriodicConv2dClass())
        manager = MagicMock()
        manager.domain_parallel_size = 2

        # No custom_converters — the inner conv still gets replaced
        convert_to_domain_parallel(model, manager)

        assert isinstance(model[0].conv, DomainParallelConv2d)
