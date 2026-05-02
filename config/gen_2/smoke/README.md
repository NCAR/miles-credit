# Smoke test configs

Quick single-GPU configs for validating the `credit` CLI end-to-end on Casper.
Use these to reproduce bugs and confirm fixes before touching full-scale runs.

| Config | Purpose |
|--------|---------|
| `smoke_gen2_casper.yml` | **Primary CLI smoke test.** Gen2 trainer, 0.25° 16-level ERA5, 2-year window, 1 GPU V100. Exercises `credit train`, `credit rollout`, and `credit submit`. |
| `smoke_gen2_multistep_casper.yml` | Same as above with multi-step rollout training. |
| `smoke_v2branch_casper.yml` | Variant used during v2 branch integration testing. |

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
- The model is intentionally small (`dim: [32,64,128,256]`) for fast iteration — not for science.
