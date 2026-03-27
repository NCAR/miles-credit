# Training a Model

:::{note}
**New to CREDIT?** Jump straight to [Training with the v2 Data Schema](#training-with-the-v2-data-schema) —
it is the recommended path for all new experiments and uses a single `credit` command for everything.
The sections below document the legacy v1 workflow.
:::

CREDIT supports three modes for training a model. In your configuration file (`model.yml`), under the `trainer` field, you can set `mode` to one of the following:

- `None`: Trains on a single GPU without any special distributed settings.
- `ddp`: Uses **Distributed Data Parallel (DDP)** for multi-GPU training.
- `fsdp`: Uses **Fully Sharded Data Parallel (FSDP)** for multi-GPU training.

## Training on a Single GPU (No Distributed Training)

To start a training run from epoch 0, use:

```bash
credit_train -c config/model.yml
```

Ensure the `trainer` section in `model.yml` is set as follows:

```yaml
trainer:
    load_weights: False
    load_optimizer: False
    load_scaler: False
    load_scheduler: False
    reload_epoch: False
    start_epoch: 0
    num_epoch: 10
    epochs: &epochs 70
```

These settings ensure training starts at epoch 0 without loading any pre-existing weights. The model will train for 10 epochs and save a checkpoint (`checkpoint.pt`) to the `save_loc` directory as well as a `training_log.csv` file that will report on statistics such as the epoch number and the training and validation loss.

To continue training from epoch 11, update these settings:

```yaml
trainer:
    load_weights: True
    load_optimizer: True
    load_scaler: True
    load_scheduler: True
    reload_epoch: True
    start_epoch: 0
    num_epoch: 10
    epochs: &epochs 70
```

Setting `reload_epoch: True` ensures that training resumes from the last saved checkpoint and will automatically load `training_log.csv`. Once training has been run seven times, reaching epoch 70, the training process is complete.

## Training with Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP)

To train on multiple GPUs, set `mode` to `ddp` or `fsdp` in `model.yml`.

```yaml
trainer:
    mode: ddp  # Use 'fsdp' for Fully Sharded Data Parallel
```

Then, start training as usual:

```bash
credit_train -c config/model.yml
```

This command generates a PBS script and submits it via `qsub`.
Job resources are controlled by the `pbs:` section of your config — see below.

### PBS configuration in your config file

The `pbs:` block is the primary place to set your **allocation code**, walltime, node count,
conda environment, and other job parameters. You do not need to pass these on the command line
every time.

```yaml
# ---- Derecho ----------------------------------------------------------------
pbs:
    project: "NCAR0001"        # YOUR allocation code (PBS -A) — change this!
    job_name: "credit_v2"      # job name shown in qstat
    walltime: "12:00:00"       # wall-clock limit per job (HH:MM:SS)
    nodes: 1                   # number of nodes (derecho only; casper is always 1)
    ncpus: 64                  # CPUs per node
    ngpus: 4                   # GPUs per node
    mem: ‘480GB’               # memory per node
    queue: ‘main’              # queue name
    conda: "credit-derecho"    # conda env name or full path
```

```yaml
# ---- Casper -----------------------------------------------------------------
pbs:
    project: "NCAR0001"
    job_name: "credit_v2"
    walltime: "04:00:00"
    ncpus: 8
    ngpus: 1
    mem: ‘128GB’
    queue: ‘casper’
    gpu_type: ‘a100_80gb’      # a100_80gb, v100, h100, etc.
    conda: "credit"
```

**Resolution order** — the same setting can come from three places, highest priority first:

| Priority | Source | Example |
|---|---|---|
| 1 | CLI flag | `--account NCAR0001 --gpus 4` |
| 2 | `pbs:` section in config | `project: "NCAR0001"` |
| 3 | Built-in cluster default | 4 GPUs, 12 h walltime, etc. |

You can also export `PBS_ACCOUNT` in your shell as a global fallback for the account code
(useful if you work across multiple configs but always charge the same project).

## Running on Casper vs. Derecho

### Key Differences

| Feature          | Derecho          | Casper         |
|-----------------|-----------------|---------------|
| GPUs per node   | 4                | 1             |
| Total GPUs      | 32 (8 nodes × 4) | 1             |
| Memory          | 480GB            | 128GB         |
| Walltime        | 12:00:00         | 4:00:00       |
| GPU Type        | A100             | V100/A100/H100         |
| Queue          | `main`            | `casper`      |

Casper is best for **small-scale experiments**, while Derecho is designed for **large-scale, multi-node training**.
Derecho only has A100 GPUs with 40 Gb of memory. Casper has both 40 Gb and 80 Gb A100s along with a small
number of H100s with 80 Gb of memory.

---

## Training with the v2 Data Schema

CREDIT v2 is the recommended path for all new experiments. It uses a cleaner nested
data schema with explicit variable categories (`prognostic`, `diagnostic`,
`dynamic_forcing`, `static`) and a unified `credit` command for everything.

### Quickstart on Casper or Derecho

```bash
# 1. Activate the pre-built environment (NCAR users)
#    Casper:
conda activate /glade/u/home/schreck/.conda/envs/credit-casper
#    Derecho:
#    conda activate /glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221

# 2. Clone the repo and install
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
pip install -e . --no-deps

# 3. Generate a config from a built-in template
credit init --grid 1deg -o my_experiment.yml      # or --grid 0.25deg for full-res

# 4. Set your allocation in the pbs: section of my_experiment.yml, then submit.
#    --chain auto-computes job count from ceil(epochs / num_epoch) in the config.
credit submit --cluster casper  -c my_experiment.yml --chain 10
credit submit --cluster derecho -c my_experiment.yml --chain 10
```

That's it. `--chain 10` submits 10 back-to-back jobs via PBS `afterok` dependencies —
no manual resubmission needed.

:::{note}
**NCAR users**: data paths in the built-in configs point to
`/glade/campaign/cisl/aiml/ksha/CREDIT_data/` which is readable by all NCAR staff.
`save_loc` defaults to `/glade/derecho/scratch/$USER/CREDIT_runs/...` — no config
edits required to get started.
:::

### How many jobs do I need?

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

### Available configs

| Grid | File | Notes |
|------|------|-------|
| 1° | `config/wxformer_1dg_6hr_v2.yml` | ERA5 model-level — good starting point |
| 0.25° | `config/wxformer_025deg_6hr_v2.yml` | Full-res pressure-level, 13 levels |
| starter | `config/starter_v2.yml` | Minimal template with `USER SETTINGS` comments |

### What does a healthy training run look like?

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

### Trainer configuration

Set `trainer.type: era5-v2` in your config. Key fields:

```yaml
trainer:
    type: era5-v2
    mode: ddp               # none | ddp | fsdp
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

### Job submission

The `credit submit` command generates a ready-to-use PBS script and optionally calls `qsub`.
Resource settings are read from the `pbs:` section of your config (see above); CLI flags
override them when provided.

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

### Resuming training

Wall-time limits on Casper (12 h) and Derecho mean a 70-epoch run typically needs
multiple job submissions. Two options:

#### Option A — chain jobs upfront with `--chain N`

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

#### Option B — manual reload with `--reload`

Submit one job at a time. After each job completes, resubmit with `--reload`:

```bash
# First job
credit submit --cluster derecho -c config.yml

# Every subsequent job
credit submit --cluster derecho -c config.yml --reload
```

#### Restarting a failed chain

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
