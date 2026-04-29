"""
Multi-GPU test: verify chunked gather produces identical CRPS loss to the
original per-lat gather, and that gradients flow correctly.

Run with:
    torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/test_ensemble_gather.py
"""

import torch
import torch.distributed as dist
from credit.trainers.trainerERA5_ensemble import gather_tensor
from credit.losses.kcrps import KCRPSLoss


def setup():
    backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else "gloo"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    if backend == "nccl":
        torch.cuda.set_device(rank)
    return rank, dist.get_world_size(), backend


def teardown():
    dist.destroy_process_group()


def loss_perlat(y_pred, y, criterion, lat_size):
    """Original: one all_gather per latitude row."""
    total_loss = 0
    for i in range(lat_size):
        y_pred_slice = y_pred[:, :, :, i : i + 1].contiguous()
        y_slice = y[:, :, :, i : i + 1].contiguous()
        y_pred_slice = gather_tensor(y_pred_slice)
        total_loss += criterion(y_slice.to(y_pred_slice.dtype), y_pred_slice).mean() / lat_size
    return total_loss


def loss_chunked(y_pred, y, criterion, lat_size, lat_chunks=1):
    """New: one (or a few) all_gather calls."""
    chunk_size = (lat_size + lat_chunks - 1) // lat_chunks
    total_loss = 0
    for chunk_start in range(0, lat_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, lat_size)
        n_lats = chunk_end - chunk_start
        y_pred_chunk = y_pred[:, :, :, chunk_start:chunk_end].contiguous()
        y_chunk = y[:, :, :, chunk_start:chunk_end].contiguous()
        y_pred_chunk = gather_tensor(y_pred_chunk)
        total_loss += criterion(y_chunk.to(y_pred_chunk.dtype), y_pred_chunk).mean() * (n_lats / lat_size)
    return total_loss


def main():
    rank, world_size, backend = setup()
    device = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")

    torch.manual_seed(42 + rank)

    # Small spatial dims so the test is fast: (B, C, T, H, W)
    B, C, T, H, W = 1, 4, 1, 16, 32

    criterion = KCRPSLoss(reduction="mean")

    # Each rank holds one ensemble member — y is the same on all ranks
    torch.manual_seed(0)
    y = torch.randn(B, C, T, H, W, device=device)

    # y_pred differs per rank (each rank = one ensemble member)
    y_pred_orig = torch.randn(B, C, T, H, W, device=device, requires_grad=True)
    y_pred_new = y_pred_orig.detach().clone().requires_grad_(True)

    # ---- Test 1: loss values match ----
    loss_old = loss_perlat(y_pred_orig, y, criterion, H)
    loss_new_1 = loss_chunked(y_pred_new, y, criterion, H, lat_chunks=1)

    diff = (loss_old - loss_new_1).abs().item()
    if rank == 0:
        print(f"[rank 0] loss_perlat={loss_old.item():.8f}  loss_chunked(1)={loss_new_1.item():.8f}  diff={diff:.2e}")
        assert diff < 1e-5, f"Loss mismatch (1 chunk): {diff}"
        print("PASS: single-chunk loss matches per-lat loss")

    # ---- Test 2: chunked (lat_chunks=4) also matches ----
    y_pred_chunked = y_pred_orig.detach().clone().requires_grad_(True)
    loss_new_4 = loss_chunked(y_pred_chunked, y, criterion, H, lat_chunks=4)
    diff4 = (loss_old - loss_new_4).abs().item()
    if rank == 0:
        print(f"[rank 0] loss_chunked(4)={loss_new_4.item():.8f}  diff={diff4:.2e}")
        assert diff4 < 1e-5, f"Loss mismatch (4 chunks): {diff4}"
        print("PASS: 4-chunk loss matches per-lat loss")

    # ---- Test 3: gradients flow ----
    loss_old.backward()
    loss_new_1.backward()

    grad_diff = (y_pred_orig.grad - y_pred_new.grad).abs().max().item()
    if rank == 0:
        print(f"[rank 0] max grad diff (old vs new): {grad_diff:.2e}")
        assert grad_diff < 1e-5, f"Gradient mismatch: {grad_diff}"
        print("PASS: gradients match")

    if rank == 0:
        print("\nAll tests passed.")

    teardown()


if __name__ == "__main__":
    main()
