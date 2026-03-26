# Quickstart

Get from zero to a running training job in under 10 minutes.
This page covers the full loop — install, configure, submit, monitor, visualise, get help.
Every command is copy-pasteable.

---

## 1. Set up your environment

:::{note}
**NCAR users on Casper** — pre-built environment, no conda create needed:

```bash
conda activate /glade/u/home/schreck/.conda/envs/credit-casper

git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
pip install -e . --no-deps
```
:::

:::{note}
**NCAR users on Derecho:**

```bash
conda activate /glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221

git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
pip install -e . --no-deps
```
:::

:::{note}
**Other systems:**

```bash
conda create -n credit python=3.11
conda activate credit
pip install miles-credit
```

Or install the development branch:

```bash
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
pip install -e .
```
:::

Verify the install worked:

```bash
credit --help
```

> **More detail**: [Installation](installation.md) | [Getting Started](getting-started.md)

---

## 2. Generate a config

CREDIT ships with ready-to-use configs for ERA5. Pick your resolution:

```bash
# 1-degree ERA5 — good starting point, fast to train
credit init --grid 1deg -o my_run.yml

# 0.25-degree ERA5 — full resolution, needs more memory and time
credit init --grid 0.25deg -o my_run.yml
```

::::{note}
**NCAR users**: data paths in these configs already point to
`/glade/campaign/cisl/aiml/ksha/CREDIT_data/` — readable by all NCAR staff.
`save_loc` defaults to `/glade/derecho/scratch/$USER/CREDIT_runs/...`
**No edits required to get started.**
::::

Open `my_run.yml` and find the `# USER SETTINGS` block. The only things you
may want to change before your first run:

| Field | Default | Notes |
|-------|---------|-------|
| `trainer.num_epoch` | `5` | Epochs per PBS job. Increase if walltime allows. |
| `trainer.train_batch_size` | `8` | Per-GPU. Reduce if you hit OOM. |
| `save_loc` | scratch dir | Where checkpoints and logs are written. |

> **More detail**: [Config reference](config.md) | [Training guide](Training.md)

---

## 3. Submit a training job

### Submit

`credit submit` automatically figures out how many jobs to chain from
`trainer.epochs / trainer.num_epoch` in your config — you don't need to
calculate it yourself.

```bash
# Casper — chain computed automatically from config
credit submit --cluster casper  -c my_run.yml --gpus 4

# Derecho — 1 node × 4 GPUs
credit submit --cluster derecho -c my_run.yml --gpus 4 --nodes 1

# Derecho — multi-node (e.g. 4 nodes × 4 GPUs = 16 GPUs total)
credit submit --cluster derecho -c my_run.yml --gpus 4 --nodes 4
```

Before submitting, `credit submit` always prints a job plan:

```
====================================================
  Job plan
====================================================
  Cluster  : casper
  Config   : my_run.yml
  GPUs     : 4 GPU(s)
  Walltime : 12:00:00 per job
  Chain    : 14 jobs  (70 epochs ÷ 5 per job)
  DataLoader memory est. : ~8 GB
====================================================
```

If the memory estimate is high (> 24 GB) it will warn you to reduce
`thread_workers` or `prefetch_factor` before the job hangs silently.

Override the chain length manually if needed:

```bash
credit submit --cluster casper -c my_run.yml --gpus 4 --chain 5
```

Preview the full PBS script without submitting:

```bash
credit submit --cluster casper -c my_run.yml --gpus 4 --dry-run
```

Job 1 starts immediately; jobs 2–N are queued with PBS `afterok` and start
automatically when the previous job succeeds.

### Resuming a failed chain

If a job fails mid-run (preemption, node fault), the remaining `afterok` jobs
are cancelled by PBS. Restart from the last good checkpoint:

```bash
credit submit --cluster derecho -c my_run.yml --gpus 4 --nodes 1 --reload --chain 5
```

`--reload` patches the config to set `load_weights: True` and all related
reload flags automatically — no manual YAML editing required.

> **More detail**: [Training guide](Training.md) | `credit submit --help`

---

## 4. Monitor progress

### Training log

The trainer writes a CSV after every epoch:

```bash
# Quick check: last 5 epochs
tail -5 /glade/derecho/scratch/$USER/CREDIT_runs/my_run/training_log.csv
```

Columns: `epoch`, `train_loss`, `val_loss`, `lr`, `epoch_time_s`.

**What healthy training looks like:**
- After epoch 1: `train_loss` ≈ 1–3
- Loss should decrease steadily each epoch
- `val_loss` should track `train_loss` (not diverge)

### TensorBoard

```bash
tensorboard --logdir /glade/derecho/scratch/$USER/CREDIT_runs/my_run/tensorboard
```

Then open `http://localhost:6006` in your browser.
On HPC you will need SSH port-forwarding — see [Monitoring with TensorBoard](tensorboard.md).

---

## 5. Visualise a prediction

Once at least one checkpoint exists, run a forward pass and produce a
3-panel global map (truth | prediction | difference) for any field:

```bash
# Denormalised to physical units (K for temperature, Pa for pressure)
credit plot -c my_run.yml --field VAR_2T --denorm

# Multiple fields at once
credit plot -c my_run.yml --field VAR_2T SP VAR_10U --denorm

# Specific pressure level (index into your levels list)
credit plot -c my_run.yml --field U --level 5 --denorm
```

Plots are saved to `<save_loc>/plots/`. No GPU required — runs on CPU.

**What to look for:**

| What you see | Meaning |
|---|---|
| Recognisable weather patterns after ~10 epochs | Training is going well |
| Uniform grey prediction | Too few epochs, or LR/normalisation problem |
| Loss > 100 or growing | Check `mean_path` / `std_path` in config |
| Small smooth difference map | Model is converging correctly |

> **More detail**: `credit plot --help`

---

## 6. Get help from the AI assistant

`credit ask` answers questions about your run, automatically injecting your
config, training log, and recent PBS output as context.

It supports four providers — set whichever key you have and it's used automatically.
Use `--provider` to override:

| Provider | Env var | Model | Cost | Notes |
|----------|---------|-------|------|-------|
| Anthropic | `ANTHROPIC_API_KEY` | Claude Haiku | Pay-per-use | `https://console.anthropic.com` |
| OpenAI | `OPENAI_API_KEY` | GPT-4o | Pay-per-use | `https://platform.openai.com` |
| Google | `GOOGLE_API_KEY` | Gemini 1.5 Pro | **Free for NCAR** | `https://aistudio.google.com` |
| Groq | `GROQ_API_KEY` | Llama 3 Instant | **Free tier** | `https://console.groq.com` |

Priority when multiple keys are set: Anthropic → OpenAI → Google → Groq.

```bash
# Set whichever key(s) you have — add to ~/.bashrc to persist
export ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com
export OPENAI_API_KEY=sk-...          # https://platform.openai.com
export GOOGLE_API_KEY=AIza...         # https://aistudio.google.com  (free for NCAR users)
export GROQ_API_KEY=gsk_...           # https://console.groq.com     (free, no card needed)

# Install the package(s) for the provider(s) you want
pip install anthropic                  # or: openai  /  google-generativeai  /  groq
# All at once: pip install miles-credit[ask]
```

```bash
# Ask without context
credit ask "how do I resume a failed Derecho job?"

# Ask with your run's context injected automatically
credit ask -c my_run.yml "my loss stopped decreasing at epoch 12, what should I check?"
credit ask -c my_run.yml "is my batch size too large for 0.25 degree?"
```

::::{note}
If `credit ask` fails with a `400 Bad Request` or `insufficient credits` error,
your API account is out of credits. Switch to the free Groq provider:

```bash
pip install groq
export GROQ_API_KEY=gsk_...   # console.groq.com — free, no card needed
credit ask "your question here"
```
::::

---

## Common problems

| Symptom | Fix |
|---------|-----|
| Training hangs on startup, no error | DataLoader is using too much RAM. Set `thread_workers: 1` and `prefetch_factor: 1` in your config. |
| `RendezvousConnectionError` on Derecho | Use `--nodes 1` so the job gets `torchrun --standalone` instead of MPI rendezvous. |
| `ANTHROPIC_API_KEY is not set` | Run `export ANTHROPIC_API_KEY=sk-ant-...` or add it to `~/.bashrc`. |
| PBS chain cancelled after job failure | Expected — PBS `afterok` cancels remaining jobs. Use `--reload --chain N` to restart. |
| Checkpoint not found on first run | Normal — set `load_weights: False` in config (the default). |
| Out of GPU memory | Reduce `train_batch_size`. For 0.25° start with `train_batch_size: 1`. |

---

## What's next

| Goal | Where to go |
|------|------------|
| Understand every config field | [Config reference](config.md) |
| Multi-node training details | [Training guide](Training.md) |
| Run a forecast from a trained model | [Inference guide](Inference.md) |
| Serve forecasts over HTTP | [Forecast API Server](serve.md) |
| Set up TensorBoard on HPC | [TensorBoard](tensorboard.md) |
| Evaluate your model against baselines | [Evaluation](Evaluation.md) |
| Use a custom dataset | [Dataset structure](DataSets.md) |
| Add a new model architecture | [Model architectures](Model_Architectures.md) |
