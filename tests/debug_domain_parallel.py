"""Debug script for failing domain-parallel multi-GPU tests."""

import torch
import torch.nn as nn
import torch.distributed as dist
import os


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def debug_forward():
    """Debug the model conversion forward test."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.convert import convert_to_domain_parallel
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    manager = initialize_domain_parallel(world_size, world_size)

    torch.manual_seed(42)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
            self.norm = nn.GroupNorm(4, 8)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 4, 1)

        def forward(self, x):
            x = self.relu(self.norm(self.conv1(x)))
            return self.conv2(x)

    model_ref = SimpleModel().to(device)
    model_dp = SimpleModel().to(device)
    model_dp.load_state_dict(model_ref.state_dict())

    # Check weights match BEFORE conversion
    if rank == 0:
        print("=== WEIGHT CHECK BEFORE CONVERSION ===")
        for name in model_ref.state_dict():
            ref_w = model_ref.state_dict()[name]
            dp_w = model_dp.state_dict()[name]
            diff = (ref_w - dp_w).abs().max().item()
            print(f"  {name}: max diff = {diff:.2e}")

    convert_to_domain_parallel(model_dp, manager)

    # Check weights match AFTER conversion
    if rank == 0:
        print("\n=== WEIGHT CHECK AFTER CONVERSION ===")
        print(f"  conv2 type: {type(model_dp.conv2)}")
        w_diff = (model_ref.conv2.weight - model_dp.conv2.weight).abs().max().item()
        b_diff = (model_ref.conv2.bias - model_dp.conv2.bias).abs().max().item()
        print(f"  conv2.weight diff: {w_diff:.2e}")
        print(f"  conv2.bias diff: {b_diff:.2e}")
        print(f"  conv1 type: {type(model_dp.conv1)}")
        w_diff = (model_ref.conv1.weight - model_dp.conv1.conv.weight).abs().max().item()
        print(f"  conv1.weight diff: {w_diff:.2e}")

    # Test input
    if rank == 0:
        full_input = torch.randn(1, 4, 16, 32, device=device)
    else:
        full_input = torch.empty(1, 4, 16, 32, device=device)
    dist.broadcast(full_input, src=0)

    local_input = shard_tensor(full_input, dim=-2, manager=manager)

    with torch.no_grad():
        # Reference step by step
        ref_conv_out = model_ref.conv1(full_input)
        ref_norm_out = model_ref.norm(ref_conv_out)
        ref_relu_out = model_ref.relu(ref_norm_out)
        expected = model_ref.conv2(ref_relu_out)

        # DP step by step
        dp_conv_out = model_dp.conv1(local_input)
        dp_conv_gathered = gather_tensor(dp_conv_out, dim=-2, manager=manager)

        dp_norm_out = model_dp.norm(dp_conv_out)
        dp_norm_gathered = gather_tensor(dp_norm_out, dim=-2, manager=manager)

        dp_relu_out = model_dp.relu(dp_norm_out)
        dp_relu_gathered = gather_tensor(dp_relu_out, dim=-2, manager=manager)

        dp_final = model_dp.conv2(dp_relu_out)
        dp_output = gather_tensor(dp_final, dim=-2, manager=manager)

    if rank == 0:
        print("\n=== LAYER-BY-LAYER DIAGNOSTICS ===")
        conv_diff = (ref_conv_out - dp_conv_gathered).abs().max().item()
        norm_diff = (ref_norm_out - dp_norm_gathered).abs().max().item()
        relu_diff = (ref_relu_out - dp_relu_gathered).abs().max().item()
        final_diff = (expected - dp_output).abs().max().item()
        print(f"  After conv1:  max diff = {conv_diff:.2e}")
        print(f"  After norm:   max diff = {norm_diff:.2e}")
        print(f"  After relu:   max diff = {relu_diff:.2e}")
        print(f"  Final output: max diff = {final_diff:.2e}")

        # Check shapes
        print(f"\n  ref_relu_out shape: {ref_relu_out.shape}")
        print(f"  dp_relu_gathered shape: {dp_relu_gathered.shape}")
        print(f"  expected shape: {expected.shape}")
        print(f"  dp_output shape: {dp_output.shape}")

        # Check if relu introduces sign flips
        ref_near_zero = (ref_norm_out.abs() < 1e-3).sum().item()
        norm_sign_diff = ((ref_norm_out > 0) != (dp_norm_gathered > 0)).sum().item()
        relu_nonzero_diff = ((ref_relu_out > 0) != (dp_relu_gathered > 0)).sum().item()
        print(f"\n  Norm values near zero (<1e-3): {ref_near_zero}")
        print(f"  Sign differences after norm: {norm_sign_diff}")
        print(f"  Sign differences after relu: {relu_nonzero_diff}")

        # Check conv2 weight magnitudes
        print(f"\n  conv2 weight abs max: {model_ref.conv2.weight.abs().max().item():.4f}")
        print(f"  conv2 weight abs mean: {model_ref.conv2.weight.abs().mean().item():.4f}")


def debug_backward():
    """Debug the backward gradient test."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.convert import convert_to_domain_parallel
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    manager = initialize_domain_parallel(world_size, world_size)

    torch.manual_seed(42)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model_ref = TinyModel().to(device)
    model_dp = TinyModel().to(device)
    model_dp.load_state_dict(model_ref.state_dict())
    convert_to_domain_parallel(model_dp, manager)

    # Test input
    if rank == 0:
        full_input = torch.randn(1, 4, 8, 16, device=device)
    else:
        full_input = torch.empty(1, 4, 8, 16, device=device)
    dist.broadcast(full_input, src=0)

    # Reference backward
    ref_input = full_input.clone().requires_grad_(True)
    ref_output = model_ref(ref_input)
    ref_loss = ref_output.sum()
    ref_loss.backward()

    # DP backward
    local_input = shard_tensor(full_input, dim=-2, manager=manager)
    local_input = local_input.clone().requires_grad_(True)

    dp_output = model_dp(local_input)
    dp_loss = dp_output.sum()
    dp_loss.backward()

    # Gather output and gradient
    dp_output_gathered = gather_tensor(dp_output.detach(), dim=-2, manager=manager)
    dp_grad = gather_tensor(local_input.grad, dim=-2, manager=manager)

    if rank == 0:
        print("\n=== BACKWARD DIAGNOSTICS ===")

        # Check forward output first
        fwd_diff = (ref_output.detach() - dp_output_gathered).abs().max().item()
        print(f"  Forward output max diff: {fwd_diff:.2e}")

        # Check shapes
        print(f"  ref_output shape: {ref_output.shape}")
        print(f"  dp_output_gathered shape: {dp_output_gathered.shape}")
        print(f"  ref_input.grad shape: {ref_input.grad.shape}")
        print(f"  dp_grad shape: {dp_grad.shape}")

        # Compare gradients
        grad_diff = (ref_input.grad - dp_grad).abs()
        print(f"  Input grad max diff: {grad_diff.max().item():.2e}")
        print(f"  Input grad mean diff: {grad_diff.mean().item():.2e}")

        # Check where differences are largest
        max_idx = grad_diff.argmax()
        max_pos = []
        for s in reversed(grad_diff.shape):
            max_pos.append(max_idx % s)
            max_idx = max_idx // s
        max_pos.reverse()
        print(f"  Max diff at position: {max_pos}")
        print("  Boundary row (rank0 last / rank1 first): row 3 and 4")

        # Row-by-row gradient comparison
        print("\n  Row-by-row gradient max diff (H dim):")
        for h in range(ref_input.grad.shape[2]):
            row_diff = (ref_input.grad[:, :, h, :] - dp_grad[:, :, h, :]).abs().max().item()
            marker = " <-- boundary" if h in [3, 4] else ""
            print(f"    Row {h}: {row_diff:.6e}{marker}")

        # Check DP output matches reference (sanity check for forward)
        print("\n  DP output per-row max diff from ref:")
        for h in range(ref_output.shape[2]):
            row_diff = (ref_output.detach()[:, :, h, :] - dp_output_gathered[:, :, h, :]).abs().max().item()
            print(f"    Row {h}: {row_diff:.6e}")

        # Weight gradient comparison
        print("\n  Weight gradient comparison:")
        ref_w_grad = model_ref.conv.weight.grad
        dp_w_grad = model_dp.conv.conv.weight.grad
        print(f"  Weight grad shape ref: {ref_w_grad.shape}")
        print(f"  Weight grad shape dp: {dp_w_grad.shape}")
        w_grad_diff = (ref_w_grad - dp_w_grad).abs().max().item()
        print(f"  Weight grad max diff: {w_grad_diff:.2e}")


def main():
    local_rank, rank, world_size = setup_distributed()

    if rank == 0:
        print(f"Debug domain parallelism with {world_size} GPUs\n")

    try:
        debug_forward()
    except Exception as e:
        if rank == 0:
            import traceback

            print(f"Forward debug error: {e}")
            traceback.print_exc()

    dist.barrier()

    try:
        debug_backward()
    except Exception as e:
        if rank == 0:
            import traceback

            print(f"Backward debug error: {e}")
            traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
