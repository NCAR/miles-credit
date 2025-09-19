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
module load ncarenv/24.12 gcc/12.4.0 ncarcompilers cray-mpich/8.1.29 cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1
conda activate /path/to/my/miles-credit/derecho-env

# Export environment variables
export LSCRATCH=/glade/derecho/scratch/schreck/
export LOGLEVEL=INFO
# NCCL environment variables:
#  default NCCL and LibFabric environment variables are set by
#  the conda envionment upon activation, we will simply query them here.
#
#  It is good practice to make sure NCCL_NET is explicitly set to "AWS Libfabric"
#  this will cause the application to fail if it cannot locate the requested
#  network plugin, otherwise it is likely to fall back to a very slow and problematic
#  communication protocol.
export NCCL_NET="AWS Libfabric"
#  increased logging level for NCCL events (optional)
export NCCL_DEBUG=INFO
# print any NCCL/LibFabric related environment variables
env | egrep "NCCL|FI_" | sort -u

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
    ncpus: 8
    ngpus: 1
    mem: '128GB'
    walltime: '4:00:00'
    gpu_type: 'a100'
    queue: 'casper'
```

Once again, to launch the job on Casper, run:

```bash
credit_train -c config/model.yml -l 1
```

This command generates a **launch script (`launch.sh`)**, which will look like:

```bash
#!/bin/bash -l
#PBS -N train_model
#PBS -l select=1:ncpus=8:ngpus=1:mem=128g
#PBS -l walltime=4:00:00
#PBS -l gpu_type=a100
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod
source ~/.bashrc
conda activate credit
torchrun applications/train.py -c model.yml
```

and note that the ```torchrun``` command is used rather than MPIs. For now MPIs are not supported on Casper but that will change in a future release. But to get torch to run in distributed mode (either DDP or FSDP) we use torchrun to faciliate that (rather than setting that up manuanlly in train.py).

### Key Differences

| Feature          | Derecho          | Casper         |
|-----------------|-----------------|---------------|
| GPUs per node   | 4                | 1             |
| Total GPUs      | 32 (8 nodes × 4) | 1             |
| Memory          | 480GB            | 128GB         |
| Walltime        | 12:00:00         | 4:00:00       |
| GPU Type        | A100             | V100/A100/H100         |
| Queue          | `main`            | `casper`      |

Casper is best for **small-scale experiments**, while Derecho is designed for **large-scale, multi-node training**. Derecho only has A100 GPUs with 40 Gb of memory. Casper has both 40 Gb and 80 Gb A100s along with a small number of H100s with 80 Gb of memory.
