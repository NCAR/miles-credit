"""Test custom autograd Function with non-tensor args and no save_for_backward."""
import torch
import torch.distributed as dist
import os
import sys


class CustomFnWithCtx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width, dim):
        print(f"  CustomFnWithCtx.forward rank={dist.get_rank()}", file=sys.stderr, flush=True)
        ctx.width = width
        ctx.dim = dim
        # Don't use save_for_backward, just like _HaloExchangeFunction
        with torch.no_grad():
            zeros = torch.zeros(1, 1, width, x.shape[-1], device=x.device)
            result = torch.cat([zeros, x, zeros], dim=dim)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        print(f"  CustomFnWithCtx.backward rank={dist.get_rank()}", file=sys.stderr, flush=True)
        width = ctx.width
        dim = ctx.dim
        total = grad_output.shape[dim]
        return grad_output.narrow(dim, width, total - 2 * width).contiguous(), None, None


def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    x = torch.randn(1, 1, 2, 4, requires_grad=True, device=device)
    y = CustomFnWithCtx.apply(x, 1, 2)

    if rank == 0:
        print(f"y.requires_grad = {y.requires_grad}")
        print(f"y.grad_fn = {y.grad_fn}")

    loss = y.sum()
    loss.backward()

    if rank == 0:
        print(f"x.grad = {x.grad[0,0,:,0].tolist()}")
        print(f"Expected: [1.0, 1.0]")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
