"""Trace the halo exchange backward step by step."""

import torch
import torch.distributed as dist
import os

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def main():
    local_rank, rank, world_size = setup_distributed()

    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.halo_exchange import HaloExchange
    import credit.domain_parallel.halo_exchange as halo_mod
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    # Enable debug
    halo_mod._DEBUG_BACKWARD = True

    manager = initialize_domain_parallel(world_size, world_size)
    device = torch.device(f"cuda:{rank}")

    # Simple test: 4 rows total, 2 per rank, halo_width=1
    if rank == 0:
        full_input = torch.ones(1, 1, 4, 2, device=device)
        # Make each row identifiable
        for h in range(4):
            full_input[0, 0, h, :] = h + 1  # row 0 = 1.0, row 1 = 2.0, etc.
    else:
        full_input = torch.empty(1, 1, 4, 2, device=device)
    dist.broadcast(full_input, src=0)

    if rank == 0:
        print(f"Full input: {full_input[0, 0, :, 0].tolist()}")

    local_input = shard_tensor(full_input, dim=-2, manager=manager).clone().requires_grad_(True)

    if rank == 0:
        print(f"Rank {rank} local_input: {local_input[0, 0, :, 0].tolist()}")

    halo = HaloExchange(halo_width=1, dim=-2)
    padded = halo(local_input)

    if rank == 0:
        print(f"Rank {rank} padded: {padded[0, 0, :, 0].tolist()}")
        print(f"padded.requires_grad = {padded.requires_grad}")
        print(f"padded.grad_fn = {padded.grad_fn}")
        print(f"padded.grad_fn type = {type(padded.grad_fn)}")
        # Walk the grad_fn chain
        fn = padded.grad_fn
        depth = 0
        while fn is not None and depth < 10:
            print(f"  grad_fn chain [{depth}]: {fn}")
            if hasattr(fn, 'next_functions'):
                for i, (nf, _) in enumerate(fn.next_functions):
                    print(f"    next_fn[{i}]: {nf}")
            fn = fn.next_functions[0][0] if fn.next_functions else None
            depth += 1

    loss = padded.sum()

    if rank == 0:
        print(f"\nCalling backward on rank {rank}...")
    dist.barrier()

    loss.backward()

    dp_grad = gather_tensor(local_input.grad, dim=-2, manager=manager)

    if rank == 0:
        print(f"\nRank {rank} local_input.grad: {local_input.grad[0, 0, :, 0].tolist()}")
        print(f"Gathered dp_grad: {dp_grad[0, 0, :, 0].tolist()}")
        print(f"Expected: [1.0, 2.0, 2.0, 1.0]")
        print(f"  (global edges get 1.0, boundary rows get 2.0)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
