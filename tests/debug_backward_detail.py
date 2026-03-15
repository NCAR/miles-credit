"""Focused backward gradient debug with tiny case."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import sys


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def debug_halo_backward_only():
    """Test halo exchange backward in isolation (identity forward, no conv)."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.halo_exchange import HaloExchange, _HaloExchangeFunction
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    manager = initialize_domain_parallel(world_size, world_size)

    if rank == 0:
        full_input = torch.randn(1, 1, 4, 4, device=device)
    else:
        full_input = torch.empty(1, 1, 4, 4, device=device)
    dist.broadcast(full_input, src=0)

    local_input = shard_tensor(full_input, dim=-2, manager=manager).clone().requires_grad_(True)

    # Forward: halo exchange
    halo = HaloExchange(halo_width=1, dim=-2)
    padded = halo(local_input)

    # Backward: just sum the padded tensor
    loss = padded.sum()
    loss.backward()

    dp_grad = gather_tensor(local_input.grad, dim=-2, manager=manager)

    if rank == 0:
        print("=== HALO EXCHANGE BACKWARD (no conv) ===")
        print(f"  local_input shape: {local_input.shape}")
        print(f"  padded shape: {padded.shape}")
        print(f"  local_input.grad:\n{local_input.grad[0, 0]}")

        # Expected: each element should have grad=1 from its own position in padded,
        # PLUS contributions from neighbors that received it as halo.
        # Interior elements: grad = 1
        # Edge elements: grad = 1 + 1 = 2 (if sent as halo to neighbor)
        # Global edge: grad = 1 (no neighbor to receive halo)
        print(f"\n  Expected: inner rows grad=1, boundary rows grad=2 (halo sent to neighbor)")
        print(f"  Global edges (row 0, row 3 of full) grad=1 (no neighbor)")


def debug_manual_backward():
    """Compare F.conv2d backward with padding vs without."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.halo_exchange import HaloExchange, _HaloExchangeFunction
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    manager = initialize_domain_parallel(world_size, world_size)

    # Tiny case: 1 input channel, 1 output channel, 3x3 conv
    torch.manual_seed(42)
    weight = torch.randn(1, 1, 3, 3, device=device)
    bias = torch.zeros(1, device=device)

    if rank == 0:
        full_input = torch.randn(1, 1, 8, 4, device=device)
    else:
        full_input = torch.empty(1, 1, 8, 4, device=device)
    dist.broadcast(full_input, src=0)

    # === Reference: standard conv with padding ===
    ref_input = full_input.clone().requires_grad_(True)
    ref_output = F.conv2d(ref_input, weight, bias, padding=(1, 1))
    ref_loss = ref_output.sum()
    ref_loss.backward()

    if rank == 0:
        print("\n=== MANUAL BACKWARD DEBUG ===")
        print(f"  Weight:\n{weight[0, 0]}")
        print(f"  Ref output shape: {ref_output.shape}")
        print(f"  Ref input grad (H rows), first W col:")
        for h in range(8):
            print(f"    Row {h}: {ref_input.grad[0, 0, h, 0].item():.6f}")

    # === DP: halo exchange + conv without H padding ===
    local_input = shard_tensor(full_input, dim=-2, manager=manager).clone().requires_grad_(True)

    halo = HaloExchange(halo_width=1, dim=-2)
    padded = halo(local_input)

    dp_output = F.conv2d(padded, weight, bias, padding=(0, 1))
    dp_loss = dp_output.sum()
    dp_loss.backward()

    dp_grad = gather_tensor(local_input.grad, dim=-2, manager=manager)

    if rank == 0:
        print(f"\n  DP grad (H rows), first W col:")
        for h in range(8):
            print(f"    Row {h}: {dp_grad[0, 0, h, 0].item():.6f}")

        print(f"\n  Diff (ref - dp), first W col:")
        for h in range(8):
            diff = ref_input.grad[0, 0, h, 0].item() - dp_grad[0, 0, h, 0].item()
            print(f"    Row {h}: {diff:.6f}")

    # === Also test: what does the conv backward give for padded input? ===
    if rank == 0:
        # On rank 0 only: manually create padded input and compute gradient
        padded_manual = torch.cat([
            torch.zeros(1, 1, 1, 4, device=device),
            full_input[:, :, 0:4, :],
            full_input[:, :, 4:5, :],  # halo from "rank 1"
        ], dim=2).requires_grad_(True)

        manual_output = F.conv2d(padded_manual, weight, bias, padding=(0, 1))
        manual_loss = manual_output.sum()
        manual_loss.backward()

        print(f"\n  Manual padded grad (all 6 rows), first W col:")
        for h in range(6):
            print(f"    Padded[{h}]: {padded_manual.grad[0, 0, h, 0].item():.6f}")

        print(f"\n  Expected local grad (rows 1-4 of padded):")
        for h in range(4):
            print(f"    Row {h}: {padded_manual.grad[0, 0, h+1, 0].item():.6f}")

        # Also check what F.conv2d gives with full padding
        print(f"\n  Full ref grad (first 4 rows), first W col:")
        for h in range(4):
            print(f"    Row {h}: {ref_input.grad[0, 0, h, 0].item():.6f}")


def debug_halo_backward_with_conv():
    """Step through the entire backward chain manually on rank 0."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    manager = initialize_domain_parallel(world_size, world_size)

    torch.manual_seed(42)
    weight = torch.randn(1, 1, 3, 3, device=device)
    bias = torch.zeros(1, device=device)

    if rank == 0:
        full_input = torch.randn(1, 1, 8, 4, device=device)
    else:
        full_input = torch.empty(1, 1, 8, 4, device=device)
    dist.broadcast(full_input, src=0)

    if rank == 0:
        print("\n=== CHECK: Does halo exchange backward add correctly? ===")
        # Manually compute what the gradient should be for rank 0
        # After halo exchange: padded = [zeros, input[0:4], input[4]]
        # After conv (no H pad): output = 4 rows
        # loss = sum(output)
        # d(loss)/d(padded) from conv backward:
        padded = torch.cat([
            torch.zeros(1, 1, 1, 4, device=device),
            full_input[:, :, 0:4, :],
            full_input[:, :, 4:5, :],
        ], dim=2).requires_grad_(True)

        output = F.conv2d(padded, weight, bias, padding=(0, 1))
        loss = output.sum()
        loss.backward()

        print(f"  d(loss_0)/d(padded), first W col:")
        for h in range(6):
            label = ["zeros", "input[0]", "input[1]", "input[2]", "input[3]", "halo(input[4])"][h]
            print(f"    padded[{h}] ({label}): {padded.grad[0, 0, h, 0].item():.6f}")

        print(f"\n  grad_local (rows 1-4 of padded grad), first W col:")
        for h in range(4):
            print(f"    local[{h}]: {padded.grad[0, 0, h+1, 0].item():.6f}")

        print(f"\n  grad_halo_next (row 5) = {padded.grad[0, 0, 5, 0].item():.6f}")
        print(f"  This should be sent to rank 1 and added to rank 1's first row")

        # Now compute rank 1's contribution back to rank 0
        padded1 = torch.cat([
            full_input[:, :, 3:4, :],  # halo from rank 0 (input[3])
            full_input[:, :, 4:8, :],
            torch.zeros(1, 1, 1, 4, device=device),
        ], dim=2).requires_grad_(True)

        output1 = F.conv2d(padded1, weight, bias, padding=(0, 1))
        loss1 = output1.sum()
        loss1.backward()

        print(f"\n  d(loss_1)/d(padded_1), first W col:")
        for h in range(6):
            label = ["halo(input[3])", "input[4]", "input[5]", "input[6]", "input[7]", "zeros"][h]
            print(f"    padded1[{h}] ({label}): {padded1.grad[0, 0, h, 0].item():.6f}")

        recv_from_next = padded1.grad[0, 0, 0, 0].item()  # rank 1 sends this to rank 0
        print(f"\n  grad_halo_prev from rank 1 = {recv_from_next:.6f}")
        print(f"  This should be added to rank 0's last row (local[3])")

        # Expected rank 0 gradient
        print(f"\n  Expected rank 0 gradient (local + corrections):")
        for h in range(4):
            val = padded.grad[0, 0, h+1, 0].item()
            if h == 3:  # last row
                val += recv_from_next
            print(f"    Row {h}: {val:.6f}")

        # Reference gradient
        ref_input = full_input.clone().requires_grad_(True)
        ref_output = F.conv2d(ref_input, weight, bias, padding=(1, 1))
        ref_loss = ref_output.sum()
        ref_loss.backward()

        print(f"\n  Reference gradient (rows 0-3), first W col:")
        for h in range(4):
            print(f"    Row {h}: {ref_input.grad[0, 0, h, 0].item():.6f}")

        print(f"\n  Diff (expected DP - reference), first W col:")
        for h in range(4):
            dp_val = padded.grad[0, 0, h+1, 0].item()
            if h == 3:
                dp_val += recv_from_next
            ref_val = ref_input.grad[0, 0, h, 0].item()
            print(f"    Row {h}: {dp_val - ref_val:.6e}")


def main():
    local_rank, rank, world_size = setup_distributed()
    if rank == 0:
        print(f"Detailed backward debug with {world_size} GPUs\n")

    try:
        debug_halo_backward_only()
    except Exception as e:
        if rank == 0:
            import traceback
            traceback.print_exc()

    dist.barrier()

    try:
        debug_manual_backward()
    except Exception as e:
        if rank == 0:
            import traceback
            traceback.print_exc()

    dist.barrier()

    try:
        debug_halo_backward_with_conv()
    except Exception as e:
        if rank == 0:
            import traceback
            traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
