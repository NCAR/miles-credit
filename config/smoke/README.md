# Smoke test configs

Quick single-GPU configs for validating the `credit` CLI end-to-end on Casper.
Use these to reproduce bugs and confirm fixes before touching full-scale runs.

| Config | Purpose |
|--------|---------|
| `smoke_gen2_casper.yml` | **Primary CLI smoke test.** Gen2 trainer, 0.25° 16-level ERA5, 2-year window, 1 GPU V100. Exercises `credit train`, `credit rollout`, and `credit submit`. |
| `smoke_gen2_multistep_casper.yml` | Same as above with multi-step rollout training. |
| `smoke_v2branch_casper.yml` | Variant used during v2 branch integration testing. |
| `smoke_v2parallel_fsdp2_derecho.yml` | Pure FSDP2 parallelism (4 GPUs). |
| `smoke_v2parallel_ddp_derecho.yml` | Pure DDP parallelism (4 GPUs). |
| `smoke_v2parallel_domain_derecho.yml` | Domain Parallelism + DDP (4 GPUs). |
| `smoke_v2parallel_combo_derecho.yml` | Domain Parallelism + FSDP2 (4 GPUs). |
| `smoke_v2parallel_tp_fsdp_derecho.yml` | Tensor Parallelism + FSDP2 (4 GPUs). |
| `smoke_v2parallel_tp_domain_derecho.yml` | Tensor Parallelism + Domain Parallelism (4 GPUs). |

## Usage

```bash
# Train one epoch
credit train -c config/smoke/smoke_gen2_casper.yml

# Rollout (requires a checkpoint from training)
credit rollout -c config/smoke/smoke_gen2_casper.yml

# Dry-run PBS submission
credit submit --cluster casper -c config/smoke/smoke_gen2_casper.yml --dry-run
```

## Notes

- `save_loc` points to `/glade/derecho/scratch/schreck/credit_tests/smoke_gen2` — update to your own scratch path.
- Data paths under `data.source.ERA5` point to shared campaign storage readable by all NCAR staff.
- The model is intentionally small (`dim: [8,16,32,64]`) for fast iteration — not for science.
