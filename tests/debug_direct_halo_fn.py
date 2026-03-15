"""Test _HaloExchangeFunction directly."""
import torch
import torch.distributed as dist
import os
import sys


def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.halo_exchange import _HaloExchangeFunction, _HALO_VERSION
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    if rank == 0:
        print(f"Halo version: {_HALO_VERSION}")

    manager = initialize_domain_parallel(2, 2)

    if rank == 0:
        full_input = torch.ones(1, 1, 4, 2, device=device)
        for h in range(4):
            full_input[0, 0, h, :] = h + 1
    else:
        full_input = torch.empty(1, 1, 4, 2, device=device)
    dist.broadcast(full_input, src=0)

    local_input = shard_tensor(full_input, dim=-2, manager=manager).clone().requires_grad_(True)

    if rank == 0:
        print(f"local_input: {local_input[0, 0, :, 0].tolist()}")

    # Call _HaloExchangeFunction.apply directly
    dim = 2  # H dimension
    padded = _HaloExchangeFunction.apply(local_input, 1, dim, manager)

    if rank == 0:
        print(f"padded: {padded[0, 0, :, 0].tolist()}")
        print(f"padded.requires_grad = {padded.requires_grad}")
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
