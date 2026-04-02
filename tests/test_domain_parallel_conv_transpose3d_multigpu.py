"""Multi-GPU integration tests for DomainParallelConvTranspose3d.

Run with:
    torchrun --nproc-per-node 2 tests/test_domain_parallel_conv_transpose3d_multigpu.py

Verifies that DomainParallelConvTranspose3d produces the same output as a
single-GPU nn.ConvTranspose3d, covering:
  - Pangu PatchRecovery (kernel==stride, halo_width=0, passthrough)
  - General case (kernel>stride, halo exchange required)
  - Backward gradients for both cases
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


@pytest.mark.skipif(not dist.is_available() or not dist.is_initialized(),
                    reason="Distributed process group not initialized")
def test_pangu_patch_recovery_passthrough():
    """Pangu PatchRecovery: kernel==stride=(2,4,4), halo_width=0.

    The wrapper should be a transparent passthrough — output must match
    single-GPU ConvTranspose3d exactly.
    Input shape:  (B=1, C=192, Z=2, H_local, W=8)  — H sharded across 2 GPUs
    Output shape: (B=1, C=13,  Z=4, H_local*4, W=32)
    """
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.layers import DomainParallelConvTranspose3d
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    manager = initialize_domain_parallel(world_size, world_size)

    torch.manual_seed(42)
    # Pangu PatchRecovery parameters
    conv = nn.ConvTranspose3d(192, 13, kernel_size=(2, 4, 4), stride=(2, 4, 4)).to(device)
    dp_conv = DomainParallelConvTranspose3d(conv, shard_dim=3)

    assert dp_conv.halo_width == 0, f"Expected halo_width=0, got {dp_conv.halo_width}"

    # Full input: (1, 192, 2, 8, 8) — H=8 divides evenly across 2 ranks
    if rank == 0:
        full_input = torch.randn(1, 192, 2, 8, 8, device=device)
    else:
        full_input = torch.empty(1, 192, 2, 8, 8, device=device)
    dist.broadcast(full_input, src=0)

    with torch.no_grad():
        expected = conv(full_input)

    local_input = shard_tensor(full_input, dim=3, manager=manager)
    with torch.no_grad():
        local_output = dp_conv(local_input)
    dp_output = gather_tensor(local_output, dim=3, manager=manager)

    if rank == 0:
        max_diff = (expected - dp_output).abs().max().item()
        matches = torch.allclose(expected, dp_output, atol=1e-5)
        print(f"  PatchRecovery max diff: {max_diff:.2e}, matches: {matches}")
        assert matches, f"Output mismatch, max diff: {max_diff}"
        print("  PASSED: test_pangu_patch_recovery_passthrough")

    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    torch.backends.cuda.matmul.allow_tf32 = prev_cuda_tf32


@pytest.mark.skipif(not dist.is_available() or not dist.is_initialized(),
                    reason="Distributed process group not initialized")
def test_conv_transpose3d_with_halo():
    """General ConvTranspose3d: kernel=(2,4,4), stride=(2,2,2), padding=(0,1,1).

    k_h=4 > s_h=2, so halo_width = p_h = 1. The wrapper must exchange 1-row
    halos along H before the transposed convolution, and trim the output back
    to the correct local size.
    """
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.layers import DomainParallelConvTranspose3d
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Disable TF32 to avoid float32 rounding differences between local-shard
    # and full-tensor computations.
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    manager = initialize_domain_parallel(world_size, world_size)

    torch.manual_seed(42)
    conv = nn.ConvTranspose3d(
        8, 4, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)
    ).to(device)
    dp_conv = DomainParallelConvTranspose3d(conv, shard_dim=3)

    assert dp_conv.halo_width == 1, f"Expected halo_width=1, got {dp_conv.halo_width}"

    # Full input: (1, 8, 2, 8, 16) — H=8 divides evenly
    if rank == 0:
        full_input = torch.randn(1, 8, 2, 8, 16, device=device)
    else:
        full_input = torch.empty(1, 8, 2, 8, 16, device=device)
    dist.broadcast(full_input, src=0)

    with torch.no_grad():
        expected = conv(full_input)

    local_input = shard_tensor(full_input, dim=3, manager=manager)
    with torch.no_grad():
        local_output = dp_conv(local_input)
    dp_output = gather_tensor(local_output, dim=3, manager=manager)

    if rank == 0:
        max_diff = (expected - dp_output).abs().max().item()
        matches = torch.allclose(expected, dp_output, atol=1e-5)
        print(f"  ConvTranspose3d (halo) max diff: {max_diff:.2e}, matches: {matches}")
        assert matches, f"Output mismatch, max diff: {max_diff}"
        print("  PASSED: test_conv_transpose3d_with_halo")

    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    torch.backends.cuda.matmul.allow_tf32 = prev_cuda_tf32


@pytest.mark.skipif(not dist.is_available() or not dist.is_initialized(),
                    reason="Distributed process group not initialized")
def test_backward_gradients_passthrough():
    """Backward through the passthrough (halo_width=0) case."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.layers import DomainParallelConvTranspose3d
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    manager = initialize_domain_parallel(world_size, world_size)

    torch.manual_seed(42)
    conv_ref = nn.ConvTranspose3d(192, 13, kernel_size=(2, 4, 4), stride=(2, 4, 4)).to(device)
    conv_dp = nn.ConvTranspose3d(192, 13, kernel_size=(2, 4, 4), stride=(2, 4, 4)).to(device)
    conv_dp.load_state_dict(conv_ref.state_dict())
    dp_conv = DomainParallelConvTranspose3d(conv_dp, shard_dim=3)

    if rank == 0:
        full_input = torch.randn(1, 192, 2, 8, 8, device=device)
    else:
        full_input = torch.empty(1, 192, 2, 8, 8, device=device)
    dist.broadcast(full_input, src=0)

    # Reference backward
    ref_input = full_input.clone().requires_grad_(True)
    conv_ref(ref_input).sum().backward()

    # DP backward
    local_input = shard_tensor(full_input, dim=3, manager=manager).clone().requires_grad_(True)
    dp_conv(local_input).sum().backward()

    assert local_input.grad is not None, "Input gradient is None"
    for name, param in dp_conv.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    dp_grad = gather_tensor(local_input.grad, dim=3, manager=manager)
    if rank == 0:
        grad_diff = (ref_input.grad - dp_grad).abs().max().item()
        matches = torch.allclose(ref_input.grad, dp_grad, atol=1e-5)
        print(f"  Passthrough backward grad max diff: {grad_diff:.2e}, matches: {matches}")
        assert matches, f"Gradient mismatch, max diff: {grad_diff}"
        print("  PASSED: test_backward_gradients_passthrough")

    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    torch.backends.cuda.matmul.allow_tf32 = prev_cuda_tf32


@pytest.mark.skipif(not dist.is_available() or not dist.is_initialized(),
                    reason="Distributed process group not initialized")
def test_backward_gradients_with_halo():
    """Backward through the halo-exchange case."""
    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.layers import DomainParallelConvTranspose3d
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    manager = initialize_domain_parallel(world_size, world_size)

    torch.manual_seed(42)
    conv_ref = nn.ConvTranspose3d(
        8, 4, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)
    ).to(device)
    conv_dp = nn.ConvTranspose3d(
        8, 4, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)
    ).to(device)
    conv_dp.load_state_dict(conv_ref.state_dict())
    dp_conv = DomainParallelConvTranspose3d(conv_dp, shard_dim=3)

    if rank == 0:
        full_input = torch.randn(1, 8, 2, 8, 16, device=device)
    else:
        full_input = torch.empty(1, 8, 2, 8, 16, device=device)
    dist.broadcast(full_input, src=0)

    # Reference backward
    ref_input = full_input.clone().requires_grad_(True)
    conv_ref(ref_input).sum().backward()

    # DP backward
    local_input = shard_tensor(full_input, dim=3, manager=manager).clone().requires_grad_(True)
    dp_conv(local_input).sum().backward()

    assert local_input.grad is not None, "Input gradient is None"
    for name, param in dp_conv.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    dp_grad = gather_tensor(local_input.grad, dim=3, manager=manager)
    if rank == 0:
        grad_diff = (ref_input.grad - dp_grad).abs().max().item()
        matches = torch.allclose(ref_input.grad, dp_grad, atol=1e-5)
        print(f"  Halo backward grad max diff: {grad_diff:.2e}, matches: {matches}")
        assert matches, f"Gradient mismatch, max diff: {grad_diff}"
        print("  PASSED: test_backward_gradients_with_halo")

    torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    torch.backends.cuda.matmul.allow_tf32 = prev_cuda_tf32


def main():
    local_rank, rank, world_size = setup_distributed()

    if rank == 0:
        print(f"Running DomainParallelConvTranspose3d tests with {world_size} GPUs")
        print()

    tests = [
        ("Pangu PatchRecovery passthrough (kernel==stride)", test_pangu_patch_recovery_passthrough),
        ("ConvTranspose3d with halo (kernel>stride)",        test_conv_transpose3d_with_halo),
        ("Backward gradients — passthrough",                 test_backward_gradients_passthrough),
        ("Backward gradients — with halo",                   test_backward_gradients_with_halo),
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
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()
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
