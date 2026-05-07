# Monitoring Training with TensorBoard

CREDIT writes training metrics to TensorBoard after every epoch when `use_tensorboard: True`
is set in the trainer config. Logs are written to `<save_loc>/tensorboard/`.

## Enabling TensorBoard

Add one line to the `trainer` block of your config:

```yaml
trainer:
    type: era5-gen2
    use_tensorboard: True   # <-- add this
    ...
```

TensorBoard is off by default so existing configs are unaffected. All production v2 configs
(`config/wxformer_025deg_6hr_v2.yml`, `config/wxformer_1dg_6hr_v2.yml`) have it enabled.

## What gets logged

Each scalar is grouped so that train and validation curves appear on the same chart:

| TensorBoard tag | Metric |
|-----------------|--------|
| `loss/train` | Training loss (mean over epoch) |
| `loss/valid` | Validation loss (mean over epoch) |
| `acc/train` | Training accuracy |
| `acc/valid` | Validation accuracy |
| `mae/train` | Training mean absolute error |
| `mae/valid` | Validation mean absolute error |
| `train/lr` | Learning rate |
| `forecast_len/train` | Rollout forecast length (auto-regressive curriculum) |

Additional per-variable metrics are logged if `save_metric_vars` is set in the config.

## Viewing logs locally

If your scratch filesystem is mounted locally (e.g. via SSHFS or on a login node):

```bash
tensorboard --logdir /glade/derecho/scratch/$USER/my_run/tensorboard
# then open http://localhost:6006 in your browser
```

## Viewing logs from Casper or Derecho (SSH port forwarding)

### Step 1 — start TensorBoard on the HPC node

SSH to a login node or submit an interactive job, then:

```bash
tensorboard --logdir /glade/derecho/scratch/$USER/my_run/tensorboard --port 6006
```

### Step 2 — forward the port to your laptop

In a **separate terminal** on your laptop:

```bash
# Replace <username> and <hostname> with your NCAR username and the login node
ssh -N -L 6006:localhost:6006 <username>@derecho.hpc.ucar.edu
# or for Casper:
ssh -N -L 6006:localhost:6006 <username>@casper.ucar.edu
```

### Step 3 — open in your browser

Navigate to [http://localhost:6006](http://localhost:6006).

### One-liner (no separate terminal)

```bash
ssh -L 6006:localhost:6006 <username>@casper.ucar.edu \
    "tensorboard --logdir /glade/derecho/scratch/$USER/my_run/tensorboard --port 6006"
```

## Comparing multiple runs

Point TensorBoard at a parent directory to overlay runs in the same chart:

```bash
# Each subdirectory becomes a separate run in the legend
tensorboard --logdir /glade/derecho/scratch/$USER/experiments/
```

This works well when `save_loc` is structured like:

```
experiments/
  wxformer_025deg_run1/tensorboard/
  wxformer_025deg_run2/tensorboard/
  wxformer_1deg_baseline/tensorboard/
```

## Resuming a run

TensorBoard appends new events to the existing log directory each time training resumes.
Epoch numbers are preserved (the trainer continues from `start_epoch`), so the loss curve
remains continuous across job restarts.

## Installation

TensorBoard is included with PyTorch and does not need a separate install. If it is somehow
missing from your environment:

```bash
pip install tensorboard
```
