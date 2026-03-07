"""Halo exchange for domain-parallel operations.

Implements differentiable halo exchange along the sharding dimension using
point-to-point communication (isend/irecv). The backward pass performs the
reverse exchange so gradients flow correctly across domain boundaries.

For latitude sharding of weather data:
- The "previous" neighbor is the rank holding the region just north.
- The "next" neighbor is the rank holding the region just south.
- Edge ranks (poles) get zero-padded halos on their outer boundary,
  since TensorPadding already handled pole reflection before sharding.
"""

import torch
import torch.distributed as dist

from credit.domain_parallel.manager import get_domain_parallel_manager


class _HaloExchangeFunction(torch.autograd.Function):
    """Differentiable halo exchange.

    Forward: pads the tensor with halo rows received from neighbors.
    Backward: sends gradient halos back to the ranks that contributed them.
    """

    @staticmethod
    def forward(ctx, x, halo_width, dim, manager):
        ctx.halo_width = halo_width
        ctx.dim = dim
        ctx.manager = manager

        if halo_width == 0:
            return x

        prev_rank, next_rank = manager.neighbor_ranks()
        group = manager.domain_group
        device = x.device

        # Slices to send
        # Send to previous: the first halo_width rows of x along dim
        # Send to next: the last halo_width rows of x along dim
        send_to_prev = x.narrow(dim, 0, halo_width).contiguous()
        send_to_next = x.narrow(dim, x.shape[dim] - halo_width, halo_width).contiguous()

        # Buffers to receive
        recv_from_prev = torch.zeros_like(send_to_next)  # same shape
        recv_from_next = torch.zeros_like(send_to_prev)

        ops = []

        # Exchange with previous neighbor
        if prev_rank is not None:
            ops.append(dist.P2POp(dist.isend, send_to_prev, prev_rank, group=group))
            ops.append(dist.P2POp(dist.irecv, recv_from_prev, prev_rank, group=group))

        # Exchange with next neighbor
        if next_rank is not None:
            ops.append(dist.P2POp(dist.isend, send_to_next, next_rank, group=group))
            ops.append(dist.P2POp(dist.irecv, recv_from_next, next_rank, group=group))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # If no neighbor, the recv buffer stays zeros (zero-pad at boundary)

        # Concatenate: [recv_from_prev | x | recv_from_next]
        parts = []
        if prev_rank is not None or manager.is_first_domain_rank:
            parts.append(recv_from_prev if prev_rank is not None else recv_from_prev.zero_())
        parts.append(x)
        if next_rank is not None or manager.is_last_domain_rank:
            parts.append(recv_from_next if next_rank is not None else recv_from_next.zero_())

        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        halo_width = ctx.halo_width
        dim = ctx.dim
        manager = ctx.manager

        if halo_width == 0:
            return grad_output, None, None, None

        prev_rank, next_rank = manager.neighbor_ranks()
        group = manager.domain_group

        # The forward output was: [halo_prev | local | halo_next]
        # grad_output has the same structure.
        # We need to:
        # 1. Extract the gradient for the local portion
        # 2. Send the halo gradient portions back to their source ranks
        # 3. Add received gradient contributions to our local edges

        total_size = grad_output.shape[dim]
        has_prev = prev_rank is not None or manager.is_first_domain_rank
        has_next = next_rank is not None or manager.is_last_domain_rank

        start = halo_width if has_prev else 0
        local_size = total_size - (halo_width if has_prev else 0) - (halo_width if has_next else 0)
        end = start + local_size

        # Local gradient
        grad_local = grad_output.narrow(dim, start, local_size).contiguous()

        # Halo gradients to send back
        grad_halo_prev = grad_output.narrow(dim, 0, halo_width).contiguous() if has_prev else None
        grad_halo_next = (
            grad_output.narrow(dim, total_size - halo_width, halo_width).contiguous()
            if has_next
            else None
        )

        # Buffers for receiving gradient contributions from neighbors
        recv_from_prev = torch.zeros_like(grad_local.narrow(dim, 0, halo_width))
        recv_from_next = torch.zeros_like(
            grad_local.narrow(dim, local_size - halo_width, halo_width)
        )

        ops = []

        # Reverse exchange: send halo grads back, receive contributions
        if prev_rank is not None:
            ops.append(dist.P2POp(dist.isend, grad_halo_prev, prev_rank, group=group))
            ops.append(dist.P2POp(dist.irecv, recv_from_prev, prev_rank, group=group))

        if next_rank is not None:
            ops.append(dist.P2POp(dist.isend, grad_halo_next, next_rank, group=group))
            ops.append(dist.P2POp(dist.irecv, recv_from_next, next_rank, group=group))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Add the received gradient contributions to the edges of local gradient
        # recv_from_prev = gradient for our FIRST rows (which we sent to prev as halo)
        # recv_from_next = gradient for our LAST rows (which we sent to next as halo)
        grad_x = grad_local.clone()
        if prev_rank is not None:
            grad_x.narrow(dim, 0, halo_width).add_(recv_from_prev)
        if next_rank is not None:
            grad_x.narrow(dim, local_size - halo_width, halo_width).add_(recv_from_next)

        return grad_x, None, None, None


class HaloExchange(torch.nn.Module):
    """Halo exchange layer for domain-parallel operations.

    Pads the input tensor with halo rows from neighboring ranks along
    the sharding dimension. The operation is differentiable.

    Args:
        halo_width: Number of rows to exchange on each side.
        dim: Tensor dimension to exchange along (default: -2 for lat in BCHW).
    """

    def __init__(self, halo_width, dim=-2):
        super().__init__()
        self.halo_width = halo_width
        self.dim = dim

    def forward(self, x):
        manager = get_domain_parallel_manager()
        if manager is None or manager.domain_parallel_size <= 1:
            return x
        # Normalize negative dim
        dim = self.dim % x.ndim
        return _HaloExchangeFunction.apply(x, self.halo_width, dim, manager)

    @staticmethod
    def trim(x, halo_before, halo_after, dim=-2):
        """Trim halo rows from the output after a convolution.

        Args:
            x: Tensor with extra halo rows.
            halo_before: Number of rows to trim from the start.
            halo_after: Number of rows to trim from the end.
            dim: Dimension to trim along.
        """
        dim = dim % x.ndim
        total = x.shape[dim]
        end = total - halo_after if halo_after > 0 else total
        return x.narrow(dim, halo_before, end - halo_before)
