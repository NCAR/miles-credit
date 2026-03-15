"""Test halo exchange with inline definition."""
import torch
import torch.distributed as dist
import os
import sys


class InlineHaloFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, halo_width, dim, manager):
        ctx.halo_width = halo_width
        ctx.dim = dim
        ctx.manager = manager

        if halo_width == 0:
            return x

        prev_rank, next_rank = manager.neighbor_ranks()
        group = manager.domain_group

        send_to_prev = x.narrow(dim, 0, halo_width).contiguous()
        send_to_next = x.narrow(dim, x.shape[dim] - halo_width, halo_width).contiguous()

        recv_from_prev = torch.zeros_like(send_to_next)
        recv_from_next = torch.zeros_like(send_to_prev)

        ops = []
        if prev_rank is not None:
            ops.append(dist.P2POp(dist.isend, send_to_prev, prev_rank, group=group))
            ops.append(dist.P2POp(dist.irecv, recv_from_prev, prev_rank, group=group))
        if next_rank is not None:
            ops.append(dist.P2POp(dist.isend, send_to_next, next_rank, group=group))
            ops.append(dist.P2POp(dist.irecv, recv_from_next, next_rank, group=group))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        parts = []
        if prev_rank is not None or manager.is_first_domain_rank:
            parts.append(recv_from_prev if prev_rank is not None else recv_from_prev.zero_())
        parts.append(x)
        if next_rank is not None or manager.is_last_domain_rank:
            parts.append(recv_from_next if next_rank is not None else recv_from_next.zero_())

        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        with open("/tmp/inline_backward.txt", "a") as f:
            f.write(f"inline backward rank={dist.get_rank()}\n")

        halo_width = ctx.halo_width
        dim = ctx.dim
        manager = ctx.manager

        if halo_width == 0:
            return grad_output, None, None, None

        prev_rank, next_rank = manager.neighbor_ranks()
        group = manager.domain_group

        total_size = grad_output.shape[dim]
        has_prev = prev_rank is not None or manager.is_first_domain_rank
        has_next = next_rank is not None or manager.is_last_domain_rank

        start = halo_width if has_prev else 0
        local_size = total_size - (halo_width if has_prev else 0) - (halo_width if has_next else 0)

        grad_local = grad_output.narrow(dim, start, local_size).contiguous()
        grad_halo_prev = grad_output.narrow(dim, 0, halo_width).contiguous() if has_prev else None
        grad_halo_next = (
            grad_output.narrow(dim, total_size - halo_width, halo_width).contiguous()
            if has_next else None
        )

        recv_from_prev = torch.zeros_like(grad_local.narrow(dim, 0, halo_width))
        recv_from_next = torch.zeros_like(grad_local.narrow(dim, local_size - halo_width, halo_width))

        ops = []
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

        grad_x = grad_local.clone()
        if prev_rank is not None:
            grad_x.narrow(dim, 0, halo_width).add_(recv_from_prev)
        if next_rank is not None:
            grad_x.narrow(dim, local_size - halo_width, halo_width).add_(recv_from_next)

        return grad_x, None, None, None


def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    manager = initialize_domain_parallel(2, 2)

    if rank == 0:
        full_input = torch.ones(1, 1, 4, 2, device=device)
        for h in range(4):
            full_input[0, 0, h, :] = h + 1
    else:
        full_input = torch.empty(1, 1, 4, 2, device=device)
    dist.broadcast(full_input, src=0)

    local_input = shard_tensor(full_input, dim=-2, manager=manager).clone().requires_grad_(True)

    padded = InlineHaloFn.apply(local_input, 1, 2, manager)

    if rank == 0:
        print(f"padded.grad_fn = {padded.grad_fn}")
        print(f"padded.grad_fn type = {type(padded.grad_fn).__name__}")

    loss = padded.sum()
    loss.backward()

    dp_grad = gather_tensor(local_input.grad, dim=-2, manager=manager)

    if rank == 0:
        print(f"local_input.grad: {local_input.grad[0, 0, :, 0].tolist()}")
        print(f"Gathered: {dp_grad[0, 0, :, 0].tolist()}")
        print(f"Expected: [1.0, 2.0, 2.0, 1.0]")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
