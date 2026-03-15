"""Test custom autograd Function with torchrun."""
import torch
import torch.distributed as dist
import os
import sys


class SimpleCustomFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        print(f"  SimpleCustomFn.forward rank={dist.get_rank()}", file=sys.stderr, flush=True)
        ctx.save_for_backward(x)
        zeros = torch.zeros(1, 1, 1, x.shape[-1], device=x.device)
        return torch.cat([zeros, x, zeros], dim=2)

    @staticmethod
    def backward(ctx, grad_output):
        print(f"  SimpleCustomFn.backward rank={dist.get_rank()}", file=sys.stderr, flush=True)
        x, = ctx.saved_tensors
        return grad_output[:, :, 1:-1, :]


def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    x = torch.randn(1, 1, 2, 4, requires_grad=True, device=device)
    y = SimpleCustomFn.apply(x)

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
