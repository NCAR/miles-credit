# Training (Gen 1)
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
    job_name: "credit_gen2"      # job name shown in qstat
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
    job_name: "credit_gen2"
    walltime: "04:00:00"
    ncpus: 8
    ngpus: 1
    mem: ‘128GB’               # optional: omit to scale memory with ngpus
    queue: ‘casper’
    gpu_type: ‘a100_80gb’      # optional: a100_80gb, v100, h100, etc.
                               # omit it to run on any available NVIDIA GPGPU
    conda: "credit"
```

**Casper memory scaling** — when `mem` is not set (in the config or via `--mem`), `credit submit`
sizes the request to the share of the node the job occupies: 64GB for a single GPU, growing
linearly to (nearly) the whole node's memory when the job takes every GPU on that node type.
Node sizes come from the [Casper hardware
table](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/casper/#casper-hardware):

| `gpu_type` | GPUs/node | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---|---|---|---|---|---|
| unset (any GPGPU) / `v100` | 8 (1152GB) | 64GB | 224GB | 512GB | 1088GB |
| `a100_80gb` / `h100` | 4 (1024GB) | 64GB | 368GB | 960GB | — |

**Queue / cluster mismatch** — a `pbs:` block written for one machine is easy to reuse against the
other, and PBS only rejects the bad queue after the job is submitted. `credit submit` therefore fails
up front when the resolved queue belongs to the other cluster:

```
$ credit submit --cluster casper -c derecho_config.yml --dry-run
ERROR: queue 'main' is a Derecho queue, but this job targets casper.
Casper queues: casper, cpu, gpgpu, gpudev, htc, l40, largemem, rda, vis.
Fix the 'queue:' in the config's pbs block, or override it with --queue.
```

Pass `--queue casper` to override the config for a one-off run, or edit the block. Queues valid on
both machines (`gpudev`) and unrecognized/site-specific queue names are left alone. `credit check`
reports which cluster a config's queue implies.

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