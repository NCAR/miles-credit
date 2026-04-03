# FSDP2 Parallelism Validation Results

WXFormer v2, 1-deg ERA5, 2 epochs × 5 train batches + 2 val batches.
All runs: single node, 4× GPU, `torchrun --standalone --nproc-per-node=4`.

## Casper A100-80GB (2025-04-03)

| Mode | E0 train_loss | E1 train_loss | E1 valid_loss | E1 valid_acc |
|------|-------------|-------------|-------------|------------|
| fsdp2_dp4 (dp=4, tp=1, domain=1) | — | — | — | ✓ |
| fsdp2_tp2 (dp=2, tp=2, domain=1) | — | — | — | ✓ |
| fsdp2_domain2 (dp=2, tp=1, domain=2) | — | — | — | ✓ |
| fsdp2_tp2_domain2 (dp=1, tp=2, domain=2) | — | — | — | ✓ |

Loss decreasing, ACC rising across all 4 modes. ✓

## Derecho H100-80GB (2026-04-03)

| Mode | E0 train_loss | E1 train_loss | E1 valid_loss | E1 valid_acc |
|------|-------------|-------------|-------------|------------|
| fsdp2_dp4 (dp=4, tp=1, domain=1) | 4.553 | 2.074 | 1.568 | 0.412 |
| fsdp2_tp2 (dp=2, tp=2, domain=1) | — | 2.085 | 1.635 | 0.430 |
| fsdp2_domain2 (dp=2, tp=1, domain=2) | — | 2.243 | 1.537 | 0.349 |
| fsdp2_tp2_domain2 (dp=1, tp=2, domain=2) | — | 2.395 | 1.634 | 0.343 |

Loss decreasing, ACC rising across all 4 modes. ✓

## Notes
- domain2 modes use pre-pad (H=181→192) before spatial sharding so each shard sees window-divisible height.
- `_skip_internal_padding` flag set on WXFormer v1 during domain-parallel forward to suppress duplicate pad/unpad.
- Logs: `/glade/derecho/scratch/schreck/CREDIT_runs/derecho_train_sweep/develop/`
