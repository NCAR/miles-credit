# Training a Gen 2 Model

CREDIT v2 is the recommended path for all new experiments. It uses a cleaner nested
data schema with explicit variable categories (`prognostic`, `diagnostic`,
`dynamic_forcing`, `static`) and a unified `credit` command for everything.

## Training locally

On a laptop, CPU-only machine, or single-GPU workstation, skip PBS entirely and
run the trainer directly (after fitting the scalers once with
`credit preprocess -c my_experiment.yml`):

```bash
credit train -c my_experiment.yml
```

## Start Training on Casper or Derecho

```bash
# 1. Generate a config from a built-in template
credit begin     

# 2. Set your allocation in the pbs: section of my_experiment.yml, then submit.
#    Omit --chain to auto-compute the job count from ceil(trainer.epochs / trainer.num_epoch);
#    pass --chain N to override it.
credit submit --cluster casper  -c my_experiment.yml
credit submit --cluster derecho -c my_experiment.yml --chain 3   # explicit 3-job chain
```

That's it. `--chain 3` submits 3 back-to-back jobs via PBS `afterok` dependencies —
no manual resubmission needed. Without `--chain`, the job count is computed
automatically from the config.

:::{note}
**NCAR users**: data paths in the built-in configs point to
`/glade/campaign/cisl/aiml/ksha/CREDIT_data/` which is readable by all NCAR staff.
`save_loc` defaults to `/glade/derecho/scratch/$USER/CREDIT_runs/...` — no config
edits required to get started.
:::

## How many jobs do I need?

Rule of thumb: `--chain = ceil(total_epochs / epochs_per_job)`.

`num_epoch` in the trainer config controls how many epochs run per job submission
(default 5). `epochs` is the total training target (default 70).

| total epochs | epochs/job (`num_epoch`) | `--chain` |
|---|---|---|
| 70 | 5 | 14 |
| 70 | 10 | 7 |
| 100 | 10 | 10 |

Use `--dry-run` to inspect the PBS scripts before submitting:

```bash
credit submit --cluster derecho -c my_experiment.yml --chain 10 --dry-run
```

## Available configs

| Grid | File | Notes |
|------|------|-------|
| 1° | `config/gen_2/examples/example-v2026.2.yml` | Fully annotated starter: 1° ERA5 model-level, 6-hourly WXFormer, `USER SETTINGS` comments — the reference config |
| 1° | `config/gen_2/examples/example-end-to-end.yml` | Same 1° ERA5 setup with the newer `inference:` block; runs `credit preprocess` → `train` → `rollout` end to end |
| 0.25° | `config/gen_2/examples/wxformer_era5_025deg_6hr.yml` | Full-res 0.25° ERA5 pressure-level (721 × 1440, 13 levels), 6-hourly WXFormer |
| tiny (240×121) | `config/gen_2/examples/weatherbench2_era5_wxformer_tiny.yml` | Tiny WeatherBench2 ERA5 subset (2 levels, few variables) streamed from the cloud — runs on a laptop |

## What does a healthy training run look like?

After the first epoch, `train_loss` should be **O(1)** (roughly 1–3). It should
decrease steadily across epochs. If losses are > 100 or growing, something is wrong
with normalization or the data paths.

Check progress at any time:

```bash
# Quick check: tail the CSV log
tail -5 /glade/derecho/scratch/$USER/CREDIT_runs/my_run/training_log.csv

# Global map: truth vs prediction in physical units (saves to <save_loc>/plots/)
credit plot -c my_experiment.yml --field VAR_2T --denorm

# Visual dashboard: TensorBoard (see Monitoring with TensorBoard for SSH forwarding)
tensorboard --logdir /glade/derecho/scratch/$USER/CREDIT_runs/my_run/tensorboard
```

## Trainer configuration

Set `trainer.type: gen2` in your config. Key fields:

```yaml
trainer:
    type: gen2
    parallelism:
        data: ddp           # none | ddp | fsdp2
        tensor: 1           # tensor-parallel degree (1 = disabled)
        domain: 1           # domain-parallel shards (1 = disabled)
    train_batch_size: 8     # per-GPU; total = batch_size × n_gpus
    num_epoch: 5            # epochs per job submission
    epochs: &epochs 70      # total training target
    use_tensorboard: True   # write TensorBoard logs to save_loc/tensorboard/
    use_ema: True           # recommended: EMA shadow weights for checkpointing
    ema_decay: 0.9999
    use_scheduler: True
    scheduler:
        scheduler_type: linear-warmup-cosine
        warmup_steps: 1000
        total_steps: 500000
        min_lr: 1.0e-5
```

When `use_tensorboard: True`, metrics are written to `<save_loc>/tensorboard/` after each epoch.
Launch the viewer from any machine with access to the filesystem:

```bash
tensorboard --logdir /glade/derecho/scratch/$USER/my_run/tensorboard
```

See [Monitoring with TensorBoard](tensorboard.md) for port-forwarding instructions for Casper and Derecho.

## Gen2 parallelism: FSDP2, tensor parallel, and domain parallel

The gen2 trainer supports three independent parallelism axes, controlled by a
`parallelism:` block inside `trainer:`.

```yaml
trainer:
    type: gen2
    parallelism:
        data:   fsdp2   # "fsdp2" | "ddp" | "none"
        tensor: 1       # tensor-parallel degree (1 = disabled)
        domain: 1       # domain-parallel degree (1 = disabled)
```

The three axes compose freely. With `N` total GPUs:
`dp_size = N / (tensor × domain)`, where `dp_size` is the number of FSDP2 or DDP
data-parallel replicas.

**Data parallelism (`data:`)** — `fsdp2` shards model parameters and gradients
across the data-parallel group using PyTorch FSDP2. This is the recommended default
for large models. Use `ddp` if you need gradient debugging or the model fits
comfortably in one GPU's memory. With `fsdp2`, `amp: True` enables FSDP2's own
`MixedPrecisionPolicy` (bf16 by default; override via `trainer.fsdp2_mp_policy`)
instead of manual autocast — the trainer disables autocast and the GradScaler
automatically because the policy replaces both. With spectral norm the policy
is skipped (fp32 compute, sharding only) unless `fsdp2_mp_policy` is set
explicitly.

**Tensor parallelism (`tensor:`)** — splits each weight matrix column-wise across
`tensor` GPUs within a node. This reduces per-GPU activation memory at the cost of
intra-node all-reduce communication.

> **Currently disabled.** `tensor > 1` raises `NotImplementedError`: the legacy
> hand-rolled sharding slices fused projections (e.g. WXFormer's `to_qkv`)
> across q/k/v boundaries and lacks the backward all-reduce at the
> column-parallel input, so it trains mathematically wrong outputs and
> gradients. Native TP via torch's `parallelize_module` lands with issue #415.
> The protocol below documents the intended interface and stays in place for
> that rewrite.

**Adding TP support to a new model** — tensor parallelism uses an opt-in protocol.
Any `nn.Module` block that wants TP support declares two class attributes pointing
to its column-parallel and row-parallel projection layers:

```python
class MyBlock(nn.Module):
    _tp_col = "proj_up"  # attribute path for the column-parallel layer
    _tp_row = "proj_out"  # attribute path for the row-parallel layer
    ...
```

The path is resolved with `getattr`, so dotted paths work for layers nested
inside a `Sequential` (e.g. `"layers.1"`). Supported layer types are
`nn.Conv2d` (1×1 kernels only) and `nn.Linear`.

The column-parallel layer receives the **full** input and produces a
**sharded** output (no all-reduce). The row-parallel layer receives the
sharded input and issues an `all_reduce SUM`, so the rest of the graph
sees the full output. This is the standard Megatron-style col→row pairing.

WXFormer ships with this already wired up. `FeedForward` and `Attention`
in `credit/models/wxformer/crossformer.py` declare:

```python
class FeedForward(nn.Module):
    _tp_col = "layers.1"  # Conv2d(dim → dim*mult)
    _tp_row = "layers.4"  # Conv2d(dim*mult → dim)


class Attention(nn.Module):
    _tp_col = "to_qkv"  # Conv2d(dim → inner_dim*3)
    _tp_row = "to_out"  # Conv2d(inner_dim → dim)
```

Any model block that does **not** declare `_tp_col`/`_tp_row` is left
unchanged when `tensor > 1`. If no blocks in the model declare these
attributes, a warning is logged and TP is a no-op. There is no silent
wrong-answer failure mode.

**Domain parallelism (`domain:`)** — shards the spatial H dimension across `domain`
GPUs. Each rank processes a latitude band of height `H_padded / domain`. This is
useful when a single forward pass at high resolution exceeds GPU memory even with
FSDP2. First, we pre-pad the full tensor to a window-divisible height, then shard
before the model forward pass, and finally gather and unpad the outputs.

### Padding constraint for domain parallel

When `domain > 1`, the padded image height must satisfy:

```
H_padded % (domain × local_window_size × product_of_strides) == 0
```

For WXFormer with `local_window_size: 10` and `cross_embed_strides: [2, 2, 2, 2]`
(product = 16), the constraint is `H_padded % (domain × 160) == 0`. Set `pad_lat`
in `padding_conf` so that `image_height + sum(pad_lat)` meets this requirement. For
example, with `domain: 2` and `image_height: 640`, `pad_lat: [160, 160]` gives
`H_padded = 960`, `960 % 320 = 0`.

### Data sharding and rank layout 

The dataset sampler must shard samples over the **data-parallel** dimension
only, never over the global rank. Ranks that differ only in their tensor- or
domain-parallel coordinate must receive the **same** batch:

- **TP peers** compute partial outputs of the same activation; the row-parallel
  `all_reduce` sums them. If TP peers get different samples, the sum mixes
  partial outputs of different inputs, producing garbage activations. Worse, the
  replicated (non-TP) parameters then receive different gradients on each TP
  rank, and since nothing syncs across the tp dimension, the replicas silently
  drift apart.
- **Domain peers** hold different latitude bands of the same sample; the halo
  exchange passes boundary rows between them. Different samples per domain rank
  corrupt every halo.

`init_device_mesh` arranges ranks row-major over `(dp, tp, domain)` with dp
outermost and domain innermost (`DomainParallelManager` uses the same layout:
domain groups are consecutive ranks). For global rank `g`:

```
dp_rank = g // (tensor × domain)
dp_size = world_size // (tensor × domain)
```

`train_gen2.py` computes this via `credit.parallel.mesh.data_parallel_coords`
and passes `dp_rank` / `dp_size` to the dataloader. Two further rules follow:

1. **The sampler seed must be identical on every rank.** `DistributedSampler`
   has each rank take its slice of one shared permutation; per-rank seeds make
   each rank permute differently, silently duplicating and dropping samples.
   Per-epoch variation comes from `sampler.set_epoch(epoch)` (the gen2 trainer
   calls this), not from the seed.
2. **Model RNG (dropout etc.) is seeded by `dp_rank`, not the global rank**, so
   TP/domain peers generate identical masks while dp replicas still differ.

If you write a new entry point or trainer that supports the `parallelism:`
block, reuse `data_parallel_coords` — passing the global rank/world_size into a
dataloader is correct only when `tensor: 1` and `domain: 1`.

### Common configurations

| Mode | Config | GPUs | When to use |
|------|--------|------|-------------|
| FSDP2 only | `data: fsdp2, tensor: 1, domain: 1` | any | Default for large models |
| DDP | `data: ddp, tensor: 1, domain: 1` | any | Small models, debugging |
| Domain + DDP | `data: ddp, tensor: 1, domain: 2` | 4+ | Spatial sharding with data parallel |
| FSDP2 + domain | `data: fsdp2, tensor: 1, domain: 2` | 4+ | Very large spatial resolution |
| FSDP2 + TP | `data: fsdp2, tensor: 2, domain: 1` | 4+ | Reduce activation memory |
| TP + domain | `data: none, tensor: 2, domain: 2` | 4 | Maximum memory reduction |

### Submitting a parallel job

`credit submit` detects the `parallelism:` block and generates a `torchrun` launch
automatically. No extra flags are needed:

```bash
credit submit --cluster derecho -c config.yml --gpus 4
```

The generated script uses `torchrun --standalone --nproc-per-node=4`. For multi-node
runs, set `nodes: 2` (or more) in the `pbs:` block and `credit submit` handles the
`--nnodes` and `--rdzv` arguments.

### Multi-node launcher (derecho)

For multi-node derecho jobs, `credit submit` offers two launchers via `--launcher`:

- `mpiexec` (default) — launches one `python` process per GPU under cray-mpich; MPI's
  PMI environment variables tell each rank its identity.
- `pbsdsh` — spawns one `torchrun` per node with PBS Pro's native task launcher
  (no MPI process management), and runs NCCL over the `libfabric` module. Rank identity
  comes from torchrun (`LOCAL_RANK`/`RANK`/`WORLD_SIZE`).

```bash
# Default (mpiexec)
credit submit --cluster derecho -c config.yml --nodes 2

# pbsdsh + torchrun, NCCL over libfabric
credit submit --cluster derecho -c config.yml --nodes 2 --launcher pbsdsh
```

Both launchers apply to every `--mode` (`train`, `preprocess`, `rollout`, `realtime`) —
each mode requests `select=<nodes>` and launches across all of them. Single-node jobs always
use `torchrun --standalone` regardless of `--launcher`. The `pbsdsh` launcher bakes the
fully-resolved conda/module environment into a per-node script, because pbsdsh's spawned
shell does not inherit the job's loaded modules; NCCL uses the aws-ofi-nccl plugin over
libfabric for inter-node communication. Use `--dry-run` to compare the two scripts.

## Job submission

The `credit submit` command generates a ready-to-use PBS script and optionally calls `qsub`.
Resource settings are read from the `pbs:` section of your config (see above); CLI flags
override them when provided.

```{note}
`pbs.nodes` applies to *every* `--mode`, not just `train`. If your config sets
`nodes: 8` for training, `--mode rollout` / `realtime` / `preprocess` will also request
8 nodes unless you pass `--nodes 1`. The job plan printed before submission shows the
node count; use `--dry-run` to check the `select=` line first.
```

```bash
# Minimal — all settings come from the pbs: block in config.yml
credit submit --cluster derecho -c config.yml

# Override specific settings on the fly
credit submit --cluster derecho -c config.yml --nodes 2 --walltime 06:00:00

# Charge a different account for this run only
credit submit --cluster derecho -c config.yml --account NCAR0002

# Preview the generated PBS script without submitting
credit submit --cluster derecho -c config.yml --dry-run

# Casper
credit submit --cluster casper -c config.yml
```

See `credit submit --help` for the full option list.

## Resuming training

Wall-time limits on Casper (12 h) and Derecho mean a 70-epoch run typically needs
multiple job submissions. Two options:

### Option A — chain jobs upfront with `--chain N`

Submit all jobs at once before training starts. PBS `afterok` dependencies ensure each
job only starts after the previous one completes successfully:

```bash
# Submit 10 back-to-back jobs (job 1 fresh, jobs 2–10 auto-reload)
credit submit --cluster derecho -c config.yml --chain 10

# Same for Casper
credit submit --cluster casper -c config.yml --chain 10
```

If you estimate ~5 epochs per 12 h wall time and need 70 epochs total, `--chain 14`
covers the full run without any manual resubmission.

Use `--dry-run` to preview all scripts before submitting:

```bash
credit submit --cluster derecho -c config.yml --chain 10 --dry-run
```

### Option B — manual reload with `--reload`

Submit one job at a time. After each job completes, resubmit with `--reload`:

```bash
# First job
credit submit --cluster derecho -c config.yml

# Every subsequent job
credit submit --cluster derecho -c config.yml --reload
```

### Restarting a failed chain

If the cluster kills a job mid-run (preemption, node failure, etc.), the remaining
`afterok` jobs in the chain are automatically cancelled by PBS. To restart from the
last good checkpoint, combine `--reload` and `--chain`:

```bash
# Resume and re-queue 5 more jobs from the latest checkpoint
credit submit --cluster derecho -c config.yml --reload --chain 5
```

Job 1 picks up from the checkpoint; jobs 2–5 are chained behind it with `afterok`.
The epoch counter stays continuous because `reload_epoch: True` always reads the
next epoch from the checkpoint file.

Both options write `<save_loc>/config_reload.yml` with these five fields patched
automatically — no manual config editing required:

```yaml
load_weights: True
load_optimizer: True
load_scaler: True
load_scheduler: True
reload_epoch: True   # auto-detects next epoch from checkpoint
```

`reload_epoch: True` causes the trainer to read the epoch from the checkpoint and set
`start_epoch = checkpoint_epoch + 1`, so the epoch counter is always continuous.
