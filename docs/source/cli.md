# The CREDIT Command-Line Interface

The `credit` command is the single entry point for everything CREDIT does:
creating a config, validating it, fitting preprocessing scalers, training a
model, generating forecasts, evaluating them, and submitting jobs to an HPC
scheduler.  This page gives a high-level tour of each subcommand and when you
would reach for it in a typical forecasting workflow.

If you are new to CREDIT, the [Quickstart](quickstart.md) walks through the
most common path in order; this page is a reference for the full set of
commands.

---

## The typical workflow at a glance

```text
credit begin        →  credit check  →  credit preprocess  →  credit train  →  credit rollout
```

1. **`credit begin`** — interactively create a starter config.
2. **`credit check`** — validate the config without running anything.
3. **`credit preprocess`** — fit the normalization scalers from your training data.
4. **`credit train`** — train the model.
5. **`credit rollout`** — generate forecasts from a trained checkpoint.

On an HPC cluster you replace steps 3–5 with `credit submit`, which writes and
queues a batch script for you.

---

## `credit begin` — create a starter config

```bash
credit begin
credit begin -c my_run.yml
```

An interactive wizard that asks a series of plain-language questions — which
dataset, which variables, what date range, how many GPUs — and writes a
complete, valid CREDIT config file.  It detects whether you are on an NCAR
system (Derecho/Casper), NERSC Perlmutter, or a laptop, and tailors the
defaults accordingly (scratch paths, PBS/SLURM blocks, parallelism mode).

**When to use it:** the first time you set up an experiment, or when you want a
known-good starting point that you can then hand-edit for advanced options.

> The wizard runs `credit check` on the generated file automatically, so you
> know the config is valid before you leave the wizard.

---

## `credit init` — copy a built-in template

```bash
credit init --grid 0.25deg -o my_config.yml
credit init --grid 1deg --model wxformer -o my_config.yml
```

Copies one of the pre-shipped example configs into your working directory.
Unlike `credit begin`, it does not ask any questions — you get the template
as-is and edit it yourself.

**When to use it:** when you already know which template you want and prefer to
edit a file directly rather than answer prompts.

---

## `credit check` — validate a config

```bash
credit check -c my_run.yml
credit check -c my_run.yml --deep     # also try constructing the model
credit check -c my_run.yml --strict   # fail on warnings, not just errors
credit check -c my_run.yml --json     # machine-readable output
```

Resolves every registry key the config names (model type, trainer type, loss
type, dataset type, each pre/postblock), checks each block's arguments against
the real constructor signatures, cross-checks the channel layout against the
model geometry, verifies the loss postblock chain, and checks that every file
the config points at exists.  Each finding comes with a suggested fix.

**When to use it:** before submitting a job, after editing a config, or in a
CI pipeline.  A clean `credit check` means the config will at least start
without import or shape errors.

---

## `credit preprocess` — fit normalization scalers

```bash
credit preprocess -c my_run.yml
```

Reads your training data and fits a BridgeScaler (standard scaler) JSON file
that the preblocks, loss, and metrics all reference.  This must run once before
training; the scaler file is written to `{save_loc}/standard_scaler.json`.

**When to use it:** once, after `credit begin` and before `credit train`.  If
you change your variable list or level selection, re-run it.

---

## `credit train` — train a model

```bash
credit train -c my_run.yml
credit train -c my_run.yml --backend gloo
```

Launches distributed training using the settings in your config (batch size,
learning rate, epochs, scheduler, EMA, etc.).  Checkpoints are written to
`{save_loc}/checkpoint.pt` after each epoch.

**When to use it:** on a workstation or after a batch job has started.  On HPC,
use `credit submit --mode train` instead so the job goes through the scheduler.

---

## `credit rollout` — generate forecasts

```bash
credit rollout -c my_run.yml
credit rollout -c my_run.yml --ensemble-size 10
```

Runs autoregressive forecasts from a trained checkpoint and saves the output
to NetCDF or Zarr.  The `inference:` section of your config controls the
forecast length, init times, output format, and variable selection.

**When to use it:** after training is complete and you want forecast fields for
evaluation or downstream analysis.

---

## `credit realtime` — operational-style single forecast

```bash
credit realtime -c my_run.yml --init-time 2024-01-15T00 --steps 40
```

Runs a single forecast from a specified initial time — useful for real-time or
near-real-time forecasting when you have a fresh analysis to initialize from.

**When to use it:** when you want one forecast from a specific time rather than
a batch of historical init times.

---

## `credit submit` — submit a job to the scheduler

```bash
# Preprocess + train on Casper
credit submit --cluster casper  --mode preprocess -c my_run.yml
credit submit --cluster casper  --mode train      -c my_run.yml --gpus 4

# Multi-node training on Derecho
credit submit --cluster derecho --mode train      -c my_run.yml --gpus 4 --nodes 2

# Rollout jobs
credit submit --cluster casper  --mode rollout    -c my_run.yml --jobs 1

# Realtime forecast job
credit submit --cluster casper  --mode realtime   -c my_run.yml --init-time 2024-01-15T00 --steps 40

# Preview the script without submitting
credit submit --cluster derecho -c my_run.yml --dry-run
```

Generates a PBS (Casper/Derecho) or SLURM (Perlmutter and other sites) batch
script and submits it.  For training, `--chain N` submits N back-to-back jobs
with `afterok` dependencies so each job resumes from the previous checkpoint
automatically.  `--reload` patches the config to resume from the last
checkpoint after a failure.

**When to use it:** on any HPC system where jobs must go through a scheduler.
On a laptop or interactive node, use `credit preprocess` / `credit train` /
`credit rollout` directly.

---

## `credit convert` — upgrade a v1 config to v2

```bash
credit convert -c old_v1_config.yml
credit convert -c old_v1_config.yml -o new_v2_config.yml
```

Interactively converts a legacy CREDIT v1 config to the gen2 nested data
schema, adding the new trainer features (EMA, TensorBoard, scheduler) and PBS
settings along the way.

**When to use it:** when you have an older config from a previous CREDIT
version and want to bring it up to the current schema.

---

## `credit plot` — quick visualization of a prediction

```bash
credit plot -c my_run.yml --field VAR_2T --denorm
credit plot -c my_run.yml --field U --level 5 --denorm
```

Loads a checkpoint, runs one forward pass on a validation sample, and produces
a 3-panel global map (truth | prediction | difference).  Runs on CPU — no GPU
required.

**When to use it:** to eyeball whether the model is producing realistic
weather patterns after a few epochs of training.

---

## `credit metrics` — WeatherBench2-style evaluation

```bash
credit metrics --netcdf /path/to/forecasts --out scores.csv
credit metrics --csv /path/to/scores --plot figures/ --label WXFormer-v2
```

Computes RMSE, ACC, and scorecard plots from forecast output in the
WeatherBench2 style.

**When to use it:** after rollout, to quantify forecast skill against ERA5
truth and compare against baselines.

---

## `credit ask` — AI assistant

```bash
credit ask "why is my training loss stuck at 2.5?"
credit ask -c my_run.yml "why did my training run crash?"
```

An AI assistant that can answer questions about CREDIT, debug configs, and
(when an Anthropic key is set) read files and run commands in agent mode.
Free providers (Groq, Google) work for quick questions; Anthropic enables full
agent mode.

**When to use it:** when you are stuck and want a guided answer rather than
reading the docs end-to-end.

---

## Quick reference

| Command             | What it does                           | Needs a trained model? |
|---------------------|----------------------------------------|------------------------|
| `credit begin`      | Interactive config wizard              | No                     |
| `credit init`       | Copy a built-in template               | No                     |
| `credit check`      | Validate a config                      | No                     |
| `credit preprocess` | Fit normalization scalers              | No                     |
| `credit train`      | Train a model                          | No                     |
| `credit rollout`    | Generate forecasts                     | Yes                    |
| `credit realtime`   | Single forecast from a given init time | Yes                    |
| `credit submit`     | Submit any of the above to a scheduler | Depends on mode        |
| `credit convert`    | Upgrade a gen 1 config to gen 2        | No                     |
| `credit plot`       | Quick truth-vs-prediction map          | Yes                    |
| `credit metrics`    | WeatherBench2-style skill scores       | Yes                    |
| `credit ask`        | AI assistant                           | No                     |
