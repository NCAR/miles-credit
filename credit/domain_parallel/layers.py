"""Domain-parallel layer wrappers.

Each wrapper replaces a standard PyTorch layer with a version that handles
halo exchange and/or distributed reductions for domain-parallel training.

Layers that are purely local (1x1 convolutions, channel-wise normalization,
activations, etc.) do not need wrappers and are left unchanged.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from credit.domain_parallel.halo_exchange import HaloExchange
from credit.domain_parallel.manager import get_domain_parallel_manager


def _run_forward_pre_hooks(module, x):
    """Run a module's forward pre-hooks without invoking its full forward pass.

    Required when bypassing module.__call__ (e.g. calling F.conv2d directly)
    but parameterization hooks such as SpectralNorm must still fire to
    recompute the weight attribute on the correct device before use.
    """
    for hook in module._forward_pre_hooks.values():
        hook(module, (x,))


class DomainParallelConv2d(nn.Module):
    """Domain-parallel Conv2d with automatic halo exchange.

    Before the convolution, exchanges halo rows along the sharding dimension
    (latitude by default) so boundary pixels see correct neighbors. After
    the convolution, the output has the correct local shard size.

    For strided convolutions (like in CrossEmbedLayer), the halo width is
    (kernel_size - stride) // 2 to account for the stride's effect on
    output boundaries.

    Args:
        conv: An existing nn.Conv2d module to wrap.
        shard_dim: Spatial dimension being sharded (-2 for H in BCHW).
    """

    def __init__(self, conv, shard_dim=-2):
        super().__init__()
        self.conv = conv
        self.shard_dim = shard_dim

        # Compute halo width along the sharded spatial dimension
        # For Conv2d, spatial dims are (H, W), corresponding to kernel_size indices
        # shard_dim=-2 corresponds to H, which is kernel_size[0]
        if isinstance(conv.kernel_size, int):
            k_h = conv.kernel_size
        else:
            k_h = conv.kernel_size[0]  # H dimension

        if isinstance(conv.stride, int):
            s_h = conv.stride
        else:
            s_h = conv.stride[0]

        if isinstance(conv.padding, int):
            p_h = conv.padding
        else:
            p_h = conv.padding[0]

        # For standard padding='same' style (padding=(k-1)//2, stride=1):
        #   halo = (k-1)//2 = padding
        # For strided convolutions (stride>1):
        #   halo = (k - stride) // 2
        if s_h == 1:
            self.halo_width = (k_h - 1) // 2
        else:
            self.halo_width = max(0, (k_h - s_h) // 2)

        self.halo_exchange = HaloExchange(self.halo_width, dim=shard_dim)

    def forward(self, x):
        if self.halo_width == 0:
            return self.conv(x)

        # Trigger forward pre-hooks (e.g. SpectralNorm) before bypassing __call__
        _run_forward_pre_hooks(self.conv, x)

        # Exchange halos
        x_padded = self.halo_exchange(x)

        # Run the convolution on the halo-padded input
        # The convolution's own padding handles the W (longitude) dimension,
        # but for the H (latitude) dimension we've already added the halo
        # so we need to adjust padding to avoid double-padding.
        #
        # Strategy: temporarily set H padding to 0, use halo padding instead
        orig_padding = self.conv.padding
        if isinstance(orig_padding, int):
            new_padding = (0, orig_padding)
        else:
            new_padding = (0, orig_padding[1])

        # Use F.conv2d directly to control padding
        out = F.conv2d(
            x_padded,
            self.conv.weight,
            self.conv.bias,
            self.conv.stride,
            new_padding,
            self.conv.dilation,
            self.conv.groups,
        )

        return out

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias


class DomainParallelConv3d(nn.Module):
    """Domain-parallel Conv3d with halo exchange along the lat dimension.

    Used for CubeEmbedding's patch-embedding Conv3d. The sharded dimension
    in the 5D tensor (B, C, T, H, W) is H (dim=-2 or dim=3).

    Args:
        conv: An existing nn.Conv3d module to wrap.
        shard_dim: Spatial dimension being sharded (3 for H in 5D BCTHW).
    """

    def __init__(self, conv, shard_dim=3):
        super().__init__()
        self.conv = conv
        self.shard_dim = shard_dim

        # kernel_size is (T, H, W) for Conv3d
        if isinstance(conv.kernel_size, int):
            k_h = conv.kernel_size
        else:
            k_h = conv.kernel_size[1]  # H dimension

        if isinstance(conv.stride, int):
            s_h = conv.stride
        else:
            s_h = conv.stride[1]

        if s_h == 1:
            self.halo_width = (k_h - 1) // 2
        else:
            self.halo_width = max(0, (k_h - s_h) // 2)

        self.halo_exchange = HaloExchange(self.halo_width, dim=shard_dim)

    def forward(self, x):
        if self.halo_width == 0:
            return self.conv(x)

        _run_forward_pre_hooks(self.conv, x)

        x_padded = self.halo_exchange(x)

        # Adjust padding: zero out H-dimension padding, keep T and W
        orig_padding = self.conv.padding
        if isinstance(orig_padding, int):
            new_padding = (orig_padding, 0, orig_padding)
        else:
            new_padding = (orig_padding[0], 0, orig_padding[2])

        out = F.conv3d(
            x_padded,
            self.conv.weight,
            self.conv.bias,
            self.conv.stride,
            new_padding,
            self.conv.dilation,
            self.conv.groups,
        )

        return out


class DomainParallelConvTranspose2d(nn.Module):
    """Domain-parallel ConvTranspose2d with halo exchange.

    For kernel=2, stride=2 (standard upsample): no halo needed, purely local.
    For kernel=4, stride=2, padding=1: halo of 1 needed.

    Args:
        conv: An existing nn.ConvTranspose2d module to wrap.
        shard_dim: Spatial dimension being sharded (-2 for H in BCHW).
    """

    def __init__(self, conv, shard_dim=-2):
        super().__init__()
        self.conv = conv
        self.shard_dim = shard_dim

        if isinstance(conv.kernel_size, int):
            k_h = conv.kernel_size
        else:
            k_h = conv.kernel_size[0]

        if isinstance(conv.stride, int):
            s_h = conv.stride
        else:
            s_h = conv.stride[0]

        if isinstance(conv.padding, int):
            p_h = conv.padding
        else:
            p_h = conv.padding[0]

        # For ConvTranspose2d, halo depends on how the output boundaries
        # depend on input boundaries. With padding p and kernel k, stride s:
        # The output for a given input pixel depends on k neighboring inputs.
        # halo = max(0, p) for strided transposed convolutions
        self.halo_width = p_h if k_h > s_h else 0

        self.halo_exchange = HaloExchange(self.halo_width, dim=shard_dim)

    def forward(self, x):
        if self.halo_width == 0:
            return self.conv(x)

        _run_forward_pre_hooks(self.conv, x)

        x_padded = self.halo_exchange(x)

        # Run transposed conv with adjusted padding
        orig_padding = self.conv.padding
        if isinstance(orig_padding, int):
            new_padding = (0, orig_padding)
        else:
            new_padding = (0, orig_padding[1])

        out = F.conv_transpose2d(
            x_padded,
            self.conv.weight,
            self.conv.bias,
            self.conv.stride,
            new_padding,
            self.conv.output_padding,
            self.conv.groups,
            self.conv.dilation,
        )

        # Trim extra output rows caused by (a) the halo input rows and
        # (b) removing padding_h from the convolution.
        # Each halo input row contributes stride_h output rows, and removing
        # padding_h adds padding_h more rows per side, so:
        #   trim = halo_width * stride_h + padding_h
        # Since halo_width == padding_h this simplifies to halo_width * (stride_h + 1).
        if isinstance(self.conv.stride, int):
            s_h = self.conv.stride
        else:
            s_h = self.conv.stride[0]

        trim = self.halo_width * (s_h + 1)
        if trim > 0:
            dim = self.shard_dim % out.ndim
            out = HaloExchange.trim(out, trim, trim, dim=dim)

        return out


class DomainParallelConvTranspose3d(nn.Module):
    """Domain-parallel ConvTranspose3d with halo exchange along the lat dimension.

    For kernel==stride (e.g. Pangu PatchRecovery with patch_size=(2,4,4)):
    no halo needed — purely local upsampling.
    For kernel>stride in H: halo exchange required.

    Args:
        conv: An existing nn.ConvTranspose3d module to wrap.
        shard_dim: Spatial dimension being sharded (3 for H in BCZHW).
    """

    def __init__(self, conv, shard_dim=3):
        super().__init__()
        self.conv = conv
        self.shard_dim = shard_dim

        # kernel_size is (Kz, Kh, Kw) for ConvTranspose3d; H is index 1
        if isinstance(conv.kernel_size, int):
            k_h = conv.kernel_size
        else:
            k_h = conv.kernel_size[1]

        if isinstance(conv.stride, int):
            s_h = conv.stride
        else:
            s_h = conv.stride[1]

        if isinstance(conv.padding, int):
            p_h = conv.padding
        else:
            p_h = conv.padding[1]

        self.halo_width = p_h if k_h > s_h else 0
        self.halo_exchange = HaloExchange(self.halo_width, dim=shard_dim)

    def forward(self, x):
        if self.halo_width == 0:
            return self.conv(x)

        _run_forward_pre_hooks(self.conv, x)

        x_padded = self.halo_exchange(x)

        # Zero out H-dimension padding; keep Z and W padding unchanged
        orig_padding = self.conv.padding
        if isinstance(orig_padding, int):
            new_padding = (orig_padding, 0, orig_padding)
        else:
            new_padding = (orig_padding[0], 0, orig_padding[2])

        out = F.conv_transpose3d(
            x_padded,
            self.conv.weight,
            self.conv.bias,
            self.conv.stride,
            new_padding,
            self.conv.output_padding,
            self.conv.groups,
            self.conv.dilation,
        )

        # Same trim logic as ConvTranspose2d: halo_width * (stride_h + 1).
        if isinstance(self.conv.stride, int):
            s_h = self.conv.stride
        else:
            s_h = self.conv.stride[1]

        trim = self.halo_width * (s_h + 1)
        if trim > 0:
            dim = self.shard_dim % out.ndim
            out = HaloExchange.trim(out, trim, trim, dim=dim)

        return out


class DomainParallelGroupNorm(nn.Module):
    """Domain-parallel GroupNorm with distributed statistics.

    GroupNorm computes mean and variance over (H, W) for each group of channels.
    With domain-sharded H, we need to all_reduce the statistics across the
    domain group before applying normalization.

    Args:
        norm: An existing nn.GroupNorm module to wrap.
    """

    def __init__(self, norm):
        super().__init__()
        self.num_groups = norm.num_groups
        self.num_channels = norm.num_channels
        self.eps = norm.eps
        self.affine = norm.affine
        if self.affine:
            self.weight = norm.weight
            self.bias = norm.bias
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        manager = get_domain_parallel_manager()
        if manager is None or manager.domain_parallel_size <= 1:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        B, C = x.shape[:2]
        spatial_dims = x.shape[2:]  # (H_local, W) or (H_local, W, ...)

        # Reshape to (B, num_groups, C//num_groups, *spatial)
        channels_per_group = C // self.num_groups
        x_grouped = x.view(B, self.num_groups, channels_per_group, *spatial_dims)

        # Reduce over dims 2, 3, 4, ... (everything except batch and group)
        reduce_dims = tuple(range(2, x_grouped.ndim))

        # Element count per group (all ranks have the same local size)
        local_count = 1
        for d in reduce_dims:
            local_count *= x_grouped.shape[d]
        global_count = local_count * manager.domain_parallel_size

        # All-reduce across domain group
        group = manager.domain_group

        # --- Two-pass mean-centered variance (numerically stable) ---
        # Pass 1: compute global mean
        local_sum = x_grouped.sum(dim=reduce_dims)  # (B, num_groups)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=group)
        mean = local_sum / global_count

        # Pass 2: compute global variance using the global mean
        mean_expanded = mean.view(B, self.num_groups, *([1] * len(reduce_dims)))
        x_centered = x_grouped - mean_expanded
        local_var_sum = (x_centered * x_centered).sum(dim=reduce_dims)  # (B, num_groups)
        dist.all_reduce(local_var_sum, op=dist.ReduceOp.SUM, group=group)
        var = local_var_sum / global_count

        # Normalize locally
        var_expanded = var.view(B, self.num_groups, *([1] * len(reduce_dims)))
        x_norm = x_centered / (var_expanded + self.eps).sqrt()

        # Reshape back to (B, C, *spatial)
        x_norm = x_norm.view(B, C, *spatial_dims)

        # Apply affine transform
        if self.affine:
            shape = [1, C] + [1] * len(spatial_dims)
            x_norm = x_norm * self.weight.view(shape) + self.bias.view(shape)

        return x_norm


class DomainParallelPeriodicConv2d(nn.Module):
    """Domain-parallel wrapper for PeriodicConv2d.

    PeriodicConv2d manually pads W with circular (longitude) and H with
    reflect (latitude) before calling an inner nn.Conv2d(padding=0).
    With domain-sharded H, reflect padding on H is wrong — boundary ranks
    would reflect against the shard edge instead of the true pole.

    This wrapper replaces the reflect padding on H with halo exchange,
    while keeping the circular padding on W unchanged.

    Args:
        periodic_conv: An existing PeriodicConv2d module to wrap.
            Must have a `.conv` (nn.Conv2d) and `.padding` (int) attribute.
        shard_dim: Spatial dimension being sharded (-2 for H in BCHW).
    """

    def __init__(self, periodic_conv, shard_dim=-2):
        super().__init__()
        self.conv = periodic_conv.conv  # inner nn.Conv2d(padding=0)
        self.padding = periodic_conv.padding
        self.shard_dim = shard_dim
        self.halo_exchange = HaloExchange(self.padding, dim=shard_dim)

    def forward(self, x):
        # Circular padding on W (longitude) — same as original
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode="circular")
        # Halo exchange on H (latitude) — replaces the reflect padding
        x = self.halo_exchange(x)
        return self.conv(x)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias


class DomainParallelInterpolate(nn.Module):
    """Domain-parallel bilinear interpolation.

    Exchanges a 1-row halo before interpolation so boundary pixels
    interpolate correctly, then trims the extra output rows.

    Args:
        size: Target output size (H, W).
        mode: Interpolation mode (default: 'bilinear').
        shard_dim: Spatial dimension being sharded (-2 for H in BCHW).
    """

    def __init__(self, size, mode="bilinear", shard_dim=-2):
        super().__init__()
        self.size = size
        self.mode = mode
        self.shard_dim = shard_dim
        self.halo_exchange = HaloExchange(1, dim=shard_dim)

    def forward(self, x):
        manager = get_domain_parallel_manager()
        if manager is None or manager.domain_parallel_size <= 1:
            return F.interpolate(x, size=self.size, mode=self.mode, align_corners=False)

        # Exchange 1-row halo
        x_padded = self.halo_exchange(x)

        # Compute padded target size
        h_in = x.shape[-2]
        h_padded = x_padded.shape[-2]
        h_target = self.size[0] if isinstance(self.size, (tuple, list)) else self.size
        w_target = self.size[1] if isinstance(self.size, (tuple, list)) else self.size

        # Scale target H proportionally to account for halo
        scale = h_padded / h_in
        h_target_padded = int(h_target * scale)

        # Interpolate
        out = F.interpolate(
            x_padded,
            size=(h_target_padded, w_target),
            mode=self.mode,
            align_corners=False,
        )

        # Trim the halo portion from the output
        # The halo of 1 input row maps to approximately (h_target / h_in) output rows
        trim_rows = (h_target_padded - h_target) // 2
        if trim_rows > 0:
            out = out[:, :, trim_rows : trim_rows + h_target, :]

        return out
