"""Multi-GPU integration test for domain parallelism.

Run with:
    torchrun --nproc-per-node 2 tests/test_domain_parallel_multigpu.py

Verifies that a simple model produces the same output when run with
domain parallelism (sharded across 2 GPUs) vs single-GPU baseline.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import os
import sys


def setup_distributed():
    """Initialize distributed backend."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def test_halo_exchange():
    """Test that halo exchange correctly pads boundaries."""
    from credit.domain_parallel.halo_exchange import HaloExchange
    from credit.domain_parallel.manager import initialize_domain_parallel

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    manager = initialize_domain_parallel(world_size, world_size)

    # Create test data: each rank has different values
    # Rank 0: all 1s, Rank 1: all 2s
    local_data = torch.full((1, 4, 8, 16), float(rank + 1), device=device)

    halo = HaloExchange(halo_width=1, dim=-2)
    padded = halo(local_data)

    # Expected shape: (1, 4, 8+2, 16) = (1, 4, 10, 16) for interior ranks
    # Edge ranks get zero-padding on one side
    if world_size == 2:
        if rank == 0:
            # First rank: [zeros | local | halo_from_rank1]
            assert padded.shape == (1, 4, 10, 16), f"Shape mismatch: {padded.shape}"
            # First row should be zeros (no prev neighbor)
            assert padded[:, :, 0, :].sum() == 0, "First halo should be zeros"
            # Last row should be 2.0 (from rank 1's first row)
            assert torch.allclose(
                padded[:, :, -1, :],
                torch.full((1, 4, 16), 2.0, device=device),
            ), f"Last halo wrong: {padded[:, :, -1, 0]}"
        else:
            assert padded.shape == (1, 4, 10, 16), f"Shape mismatch: {padded.shape}"
            # First row should be 1.0 (from rank 0's last row)
            assert torch.allclose(
                padded[:, :, 0, :],
                torch.full((1, 4, 16), 1.0, device=device),
            ), f"First halo wrong: {padded[:, :, 0, 0]}"
            # Last row should be zeros (no next neighbor)
            assert padded[:, :, -1, :].sum() == 0, "Last halo should be zeros"

    if rank == 0:
        print("  PASSED: test_halo_exchange")


def test_domain_parallel_conv2d():
    """Test that DomainParallelConv2d matches single-GPU Conv2d."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.layers import DomainParallelConv2d
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    manager = initialize_domain_parallel(world_size, world_size)

    # Create a Conv2d and its domain-parallel wrapper
    torch.manual_seed(42)
    conv = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1).to(device)

    dp_conv = DomainParallelConv2d(conv, shard_dim=-2)

    # Create test input on rank 0 and broadcast
    if rank == 0:
        full_input = torch.randn(1, 4, 16, 32, device=device)
    else:
        full_input = torch.empty(1, 4, 16, 32, device=device)
    dist.broadcast(full_input, src=0)

    # Single-GPU reference
    with torch.no_grad():
        expected = conv(full_input)

    # Domain-parallel computation
    local_input = shard_tensor(full_input, dim=-2, manager=manager)
    with torch.no_grad():
        local_output = dp_conv(local_input)
    dp_output = gather_tensor(local_output, dim=-2, manager=manager)

    # Compare
    if rank == 0:
        max_diff = (expected - dp_output).abs().max().item()
        matches = torch.allclose(expected, dp_output, atol=1e-5)
        print(f"  Conv2d max diff: {max_diff:.2e}, matches: {matches}")
        assert matches, f"Conv2d output mismatch, max diff: {max_diff}"
        print("  PASSED: test_domain_parallel_conv2d")


def test_domain_parallel_groupnorm():
    """Test that DomainParallelGroupNorm matches single-GPU GroupNorm."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.layers import DomainParallelGroupNorm
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    manager = initialize_domain_parallel(world_size, world_size)

    # Create GroupNorm and its domain-parallel wrapper
    norm = nn.GroupNorm(4, 16).to(device)
    dp_norm = DomainParallelGroupNorm(norm).to(device)

    # Create test input
    if rank == 0:
        full_input = torch.randn(2, 16, 32, 64, device=device)
    else:
        full_input = torch.empty(2, 16, 32, 64, device=device)
    dist.broadcast(full_input, src=0)

    # Single-GPU reference
    with torch.no_grad():
        expected = norm(full_input)

    # Domain-parallel computation
    local_input = shard_tensor(full_input, dim=-2, manager=manager)
    with torch.no_grad():
        local_output = dp_norm(local_input)
    dp_output = gather_tensor(local_output, dim=-2, manager=manager)

    # Compare
    if rank == 0:
        max_diff = (expected - dp_output).abs().max().item()
        matches = torch.allclose(expected, dp_output, atol=1e-5)
        print(f"  GroupNorm max diff: {max_diff:.2e}, matches: {matches}")
        assert matches, f"GroupNorm output mismatch, max diff: {max_diff}"
        print("  PASSED: test_domain_parallel_groupnorm")


def test_model_conversion_forward():
    """Test that a converted model produces the same output as the original."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.convert import convert_to_domain_parallel
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Disable TF32 for this test to get precise float32 comparison.
    # TF32 uses reduced mantissa precision and produces different rounding
    # for different spatial sizes (local shard vs full tensor), causing
    # ~1e-3 differences that are not bugs in the domain-parallel logic.
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    manager = initialize_domain_parallel(world_size, world_size)

    # Simple test model
    torch.manual_seed(42)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
            self.norm = nn.GroupNorm(4, 8)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 4, 1)  # 1x1, stays local

        def forward(self, x):
            x = self.relu(self.norm(self.conv1(x)))
            return self.conv2(x)

    model_ref = SimpleModel().to(device)

    # Clone weights for domain-parallel version
    model_dp = SimpleModel().to(device)
    model_dp.load_state_dict(model_ref.state_dict())
    convert_to_domain_parallel(model_dp, manager)

    # Test input
    if rank == 0:
        full_input = torch.randn(1, 4, 16, 32, device=device)
    else:
        full_input = torch.empty(1, 4, 16, 32, device=device)
    dist.broadcast(full_input, src=0)

    # --- Layer-by-layer diagnostic ---
    local_input = shard_tensor(full_input, dim=-2, manager=manager)
    with torch.no_grad():
        # Reference: step by step
        ref_conv_out = model_ref.conv1(full_input)
        ref_norm_out = model_ref.norm(ref_conv_out)
        ref_relu_out = model_ref.relu(ref_norm_out)
        expected = model_ref.conv2(ref_relu_out)

        # DP: step by step
        dp_conv_out = model_dp.conv1(local_input)
        dp_conv_gathered = gather_tensor(dp_conv_out, dim=-2, manager=manager)

        dp_norm_out = model_dp.norm(dp_conv_out)
        dp_norm_gathered = gather_tensor(dp_norm_out, dim=-2, manager=manager)

        dp_relu_out = model_dp.relu(dp_norm_out)
        dp_final = model_dp.conv2(dp_relu_out)
        dp_output = gather_tensor(dp_final, dim=-2, manager=manager)

    if rank == 0:
        conv_diff = (ref_conv_out - dp_conv_gathered).abs().max().item()
        norm_diff = (ref_norm_out - dp_norm_gathered).abs().max().item()
        final_diff = (expected - dp_output).abs().max().item()
        print(f"  After conv1:  max diff = {conv_diff:.2e}")
        print(f"  After norm:   max diff = {norm_diff:.2e}")
        print(f"  Final output: max diff = {final_diff:.2e}")
        matches = torch.allclose(expected, dp_output, atol=1e-5)
        assert matches, f"Model output mismatch, max diff: {final_diff}"
        print("  PASSED: test_model_conversion_forward")

    # Restore TF32 settings
    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    torch.backends.cuda.matmul.allow_tf32 = prev_cuda_tf32


def test_backward_gradients():
    """Test that gradients flow correctly through domain-parallel layers."""
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

    # Reference model for gradient comparison
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

    # Reference backward (use sum so d(loss)/d(output)=1 everywhere)
    ref_input = full_input.clone().requires_grad_(True)
    ref_output = model_ref(ref_input)
    ref_loss = ref_output.sum()
    ref_loss.backward()

    # Domain-parallel backward
    local_input = shard_tensor(full_input, dim=-2, manager=manager)
    local_input.requires_grad_(True)

    output = model_dp(local_input)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert local_input.grad is not None, "Input gradient is None"
    assert local_input.grad.shape == local_input.shape, "Gradient shape mismatch"
    for name, param in model_dp.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    # Compare input gradients against reference
    dp_grad = gather_tensor(local_input.grad, dim=-2, manager=manager)
    if rank == 0:
        grad_diff = (ref_input.grad - dp_grad).abs().max().item()
        grad_matches = torch.allclose(ref_input.grad, dp_grad, atol=1e-5)
        print(f"  Input grad shape: {local_input.grad.shape}")
        print(f"  Input grad max diff: {grad_diff:.2e}, matches: {grad_matches}")
        assert grad_matches, f"Input gradient mismatch, max diff: {grad_diff}"
        print("  PASSED: test_backward_gradients")


def main():
    local_rank, rank, world_size = setup_distributed()

    if rank == 0:
        print(f"Running domain parallelism multi-GPU tests with {world_size} GPUs")
        print()

    tests = [
        ("Halo Exchange", test_halo_exchange),
        ("Domain-Parallel Conv2d", test_domain_parallel_conv2d),
        ("Domain-Parallel GroupNorm", test_domain_parallel_groupnorm),
        ("Model Conversion Forward", test_model_conversion_forward),
        ("Backward Gradients", test_backward_gradients),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        if rank == 0:
            print(f"[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            if rank == 0:
                print(f"  FAILED: {e}")
            failed += 1
        dist.barrier()

    if rank == 0:
        print()
        print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
        if failed > 0:
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
