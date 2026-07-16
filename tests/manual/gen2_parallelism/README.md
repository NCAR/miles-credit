# V2 parallelism smoke matrix

Manual HPC smoke test for the gen2 (`trainer.parallelism`) parallelism stack:
FSDP2, domain, tensor, DDP, and their combinations. Each config trains 1 epoch /
5 batches with a tiny model, so a full sweep is a few minutes of GPU time.

These are **not** run by GitHub CI — they need a multi-GPU Derecho node, NCCL,
and the `credit-main-derecho` conda env.

## Run the whole matrix

```bash
qsub tests/manual/gen2_parallelism/run_smoke.pbs
```

Results (per-mode PASS/FAIL + log tails) are written to
`/glade/derecho/scratch/$USER/tmp/v2parallel_smoke_logs/SUMMARY.txt`, with one
`<mode>.log` per run.

## Run a single mode by hand

```bash
torchrun --nnodes=1 --nproc-per-node=2 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 \
  applications/train_gen2.py \
  -c tests/manual/gen2_parallelism/configs/smoke_v2parallel_fsdp2_derecho.yml
```

## Configs and the GPU count each mode needs

`world = tensor * domain * dp`. FSDP2 only shards when `dp > 1`; with `dp == 1`
it self-skips by design (a warning is logged) and the other dims still run.

**Legacy tensor parallelism is disabled**: `apply_tensor_parallel` raises on
`tensor > 1` for models without a native `_tp_plan` because the legacy
hand-rolled sharding slices fused projections across q/k/v boundaries and
lacks the backward all-reduce. The old `tp_*` configs (model type `wxformer`)
are kept as negative references but are not in the matrix.

**Native TP (issue #415)** is supported for `nextgen_wxformer` (and
CubeSphereWxFormer by inheritance) via torch `parallelize_module`. Its
acceptance gate is its own run:

```bash
qsub tests/manual/gen2_parallelism/run_tp_parity.pbs
```

This trains `nextgen_wxformer` twice on 4 GPUs with the same dp layout and
seed — tp=1 (nproc=2, dp=2) vs tp=2 (nproc=4, dp=2 x tp=2) — and gates on
identical loss trajectories (`check_parity.py`, exact first, then rtol=1e-4
to allow float32 reassociation in the rowwise all_reduce). The parity configs
set `use_spectral_norm: false` so every `_tp_plan` block actually shards:
spectral norm's power-iteration hook is incompatible with DTensor-sharded
weights, so TP skips any wrapped colwise/rowwise group (the block stays
replicated full-width, with a warning). With the default
`use_spectral_norm: true` ALL of `nextgen_wxformer`'s transformer blocks are
wrapped, so TP shards nothing and warns loudly that it had no effect.

Every config trains **multistep** (`forecast_len: 2`), which exercises the
domain-parallel between-step gather (`gather_spatial` of `y_processed`) and
the rollout assembly in every mode. For each mode the matrix runs two phases:
a **fresh** run (train 1 epoch, save `checkpoint.pt`) and a **resume** run
(reload weights + optimizer + scaler + scheduler, train a second epoch) —
proving the save/load plumbing stays in sync per parallelism mode (FSDP2 DCP
full-state APIs, domain rewritten keys, and the sharded EMA shadow, which is
enabled in `fsdp2_ac` via `use_ema: true`).

| Config | data | tensor | domain | GPUs to exercise fully |
|--------|------|--------|--------|------------------------|
| `smoke_v2parallel_ddp_derecho.yml`      | ddp   | 1 | 1 | 2 |
| `smoke_v2parallel_fsdp2_derecho.yml`    | fsdp2 | 1 | 1 | 2 |
| `smoke_v2parallel_fsdp2_ac_derecho.yml` | fsdp2 | 1 | 1 | 2 (+ activation checkpoint + EMA) |
| `smoke_v2parallel_domain_derecho.yml`   | ddp   | 1 | 2 | 4 (dp=2, domain=2) |
| `smoke_v2parallel_combo_derecho.yml`    | fsdp2 | 1 | 2 | 4 (dp=2, domain=2) |
| `smoke_v2parallel_ac_casper.yml`        | none  | 1 | 1 | 1 (Casper V100, activation checkpoint, single-process) |

`run_smoke.pbs` covers the 5 Derecho configs (fresh + resume each, 10 runs) on
a single 4-GPU node, plus a ddp-vs-domain parity gate. `run_smoke_2node.pbs`
runs the combo config on 8 GPUs / 2 nodes so FSDP2 shards across nodes. The
`ac_casper` config is the single-GPU Casper variant; run it on Casper with
plain `python` (no launcher) to exercise the single-process path.
