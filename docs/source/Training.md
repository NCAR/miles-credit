# Training a Model

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

This command generates a **launch script (`launch.sh`)** and submits a job on **Derecho**, allocating the required number of nodes and GPUs. The settings for this job are controlled by the `pbs` field in `model.yml`.

### Example PBS Configuration (Derecho)

```yaml
pbs:
    conda: "credit-derecho"
    project: "NAML0001"
    job_name: "train_model"
    walltime: "12:00:00"
    nodes: 8
    ncpus: 64
    ngpus: 4
    mem: '480GB'
    queue: 'main'
```

- **`conda`**: The environment containing the `miles-credit` installation.
- **`project`**: Your project code.
- **`nodes`** and **`ngpus`**: The number of nodes and GPUs per node. In this example, `8 nodes × 4 GPUs` = **32 GPUs total**.

### Example `launch.sh` Script for Derecho

```bash
#!/bin/bash
#PBS -A NAML0001
#PBS -N train_model
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -q main
#PBS -j oe
#PBS -k eod
#PBS -r n

# Load modules
module load conda cuda cudnn mkl
conda activate credit-derecho

# Export environment variables
export LSCRATCH=/glade/derecho/scratch/schreck/
export LOGLEVEL=INFO

# Launch training
mpiexec --cpu-bind none --no-transfer \
    python applications/train.py -c model.yml --backend nccl
```

This script utilizes **MPI** for coordinating training across **multiple nodes and GPUs**. It includes necessary environment variables for Derecho’s system configuration. **Users should not need to modify this script**, as it is tailored for Derecho and may change with system updates.

## Running on Casper vs. Derecho

For **Casper**, modify `model.yml` as follows:

```yaml
pbs:
    conda: "credit"
    project: "NAML0001"
    job_name: "train_model"
    nodes: 1
    ncpus: 32
    ngpus: 4
    mem: '900GB'
    walltime: '4:00:00'
    gpu_type: 'a100'
    queue: 'casper'
```

Once again, to launch the job on Casper, run:

```bash
credit_train -c config/example-v2026.1.0.yml -l 1
```

This command generates a **launch script (`launch.sh`)**, which will look like:

```bash
#!/bin/bash -l
#PBS -N train_model
#PBS -l select=1:ncpus=32:ngpus=4:mem=900g:gpu_type=a100
#PBS -l walltime=4:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod
source ~/.bashrc
conda activate credit-casper
torchrun --standalone --nnodes 1 --nproc-per-node=4 applications/train.py -c config/example-v2026.1.0.yml
```

and note that the `torchrun` command is used rather than MPI. In order to utilize MPI,
PyTorch needs to be compiled from source on your own system against the MPI installation on that system.
`torchrun` can perform distributed training across all GPUs on a single node with minimal configuration
and is recommended for use on Casper or other servers focused on single node training.
It is possible to use `torchrun` for multi-node training orchestration but requires starting torchrun
instances separately on each node and coordinating communication. 

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

## Training with the v2 Data Schema (`train_v2.py`)

CREDIT supports a newer nested data schema that separates variables into explicit categories
(`prognostic`, `diagnostic`, `dynamic_forcing`, `static`) under a named source (e.g., `ERA5`).
This schema uses a dedicated entry point and trainer type.

### Key differences from v1

| Feature | v1 (`train.py`) | v2 (`train_v2.py`) |
|---|---|---|
| Entry point | `applications/train.py` | `applications/train_v2.py` |
| Trainer type | `era5` | `era5-v2` |
| Data config key | flat `variables`, `surface_variables`, etc. | nested `data.source.ERA5.variables.{prognostic,...}` |
| `forecast_len` semantics | `0` means 1 step | `1` means 1 step |
| Batch sampler | legacy `ERA5Dataset` | `MultiSourceDataset` + `DistributedMultiStepBatchSampler` |

### Quick start

Use the provided starter config as a template:

```bash
cp config/starter_v2.yml config/my_experiment.yml
# Edit save_loc and date ranges, then:
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    applications/train_v2.py -c config/my_experiment.yml
```

The starter config (`config/starter_v2.yml`) is pre-filled for 1-degree ERA5 model-level data
with a CrossFormer architecture. The sections users typically need to change are marked
with `USER SETTINGS` comments.

### Trainer configuration for v2

Set `trainer.type: era5-v2` in your config. The `mode` field works the same as v1:

```yaml
trainer:
    type: era5-v2
    mode: ddp               # none | ddp | fsdp
    train_batch_size: 8     # per-GPU; total = batch_size × n_gpus
    num_epoch: 5            # epochs per job submission
    epochs: &epochs 70      # total training target
    use_ema: True           # recommended: EMA shadow weights for checkpointing
    ema_decay: 0.9999
    use_scheduler: True
    scheduler:
        scheduler_type: linear-warmup-cosine
        warmup_steps: 1000
        total_steps: 500000
        min_lr: 1.0e-5
```

### Submitting to Casper (PBS)

Use `scripts/casper_v2.sh` for single-node Casper jobs. It uses `torchrun` internally
and supports an optional `NGPUS` override (defaults to 1):

```bash
# Single GPU
CONFIG=config/starter_v2.yml qsub scripts/casper_v2.sh

# Four GPUs on one node
NGPUS=4 CONFIG=config/starter_v2.yml qsub scripts/casper_v2.sh
```

The script requests one A100 80 GB node by default (`gpu_type=a100_80gb`). Edit the
`#PBS -l select=...` line if you need a different GPU type or more CPUs.

### Resuming training

After the first job completes (e.g., epoch 5 of 70), set the following in your config
and resubmit:

```yaml
trainer:
    load_weights: True
    load_optimizer: True
    load_scaler: True
    load_scheduler: True
    reload_epoch: True
```

CREDIT will automatically detect the latest checkpoint in `save_loc` and resume from it.
