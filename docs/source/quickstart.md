# Quickstart

With these instructions, you can get from zero to running a training session in under 10 minutes.

---

## 1. Install CREDIT 
### NCAR Casper
The [NCAR Casper](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/casper/) is a heterogeneous cluster
for data analysis, visualization, and AI/ML. For ML activities, it contains nodes with multiple generations of GPUs
ranging from NVIDIA V100s, A100s, and H100s as well as AMD MI300As. All NVIDIA GPUs on Casper work with CUDA 12.6. 
Only A100s and H100s work with newer versions of CUDA. CUDA 13 (the default CUDA for PyTorch) does not work on Casper.

If you want to use the AMD GPUs, you will need to build a separate environment with a PyTorch built on ROCm 6.4.

Casper is well-suited for single node CREDIT training and inference and can support CREDIT training for 1 degree
global models and short experimental runs or interactive applications. 

To install CREDIT on Casper:
```bash
module load conda
conda create -n credit-casper -y python=3.13 uv 
conda activate credit-casper
# NVIDIA GPUs 
uv pip install miles-credit --extra-index-url https://download.pytorch.org/whl/cu126
# AMD GPUs
uv pip install miles-credit --extra-index-url https://download.pytorch.org/whl/rocm6.4
```

### NCAR Derecho
The NCAR Derecho system contains GPU nodes with 40 GB NVIDIA A100s linked with Cray Slingshot interconnect. If 
you plan to conduct multi-node training or inference, you will need to use our special install script
for Derecho to ensure that PyTorch is configured to route distributed operations over the fastest network.

To install CREDIT on Derecho:
```bash
module load conda
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
./create_derecho_env.sh # Will install in the credit-derecho conda environment
```

### Other systems
If you are running CREDIT on a Mac or a system with up-to-date GPU libraries
and no other weirdness, you can follow the following path to installing CREDIT.
```bash
conda create -n credit -y python=3.13 uv
conda activate credit
uv pip install miles-credit
```

Or install the main development branch:

```bash
conda create -n credit -y python=3.13 uv
conda activate credit
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
uv pip install -e ".[develop]"
```
:::

Quick verify the install worked:

```bash
credit --help
```

## 2. Generate a config

After installing CREDIT, use `credit begin` to create a config file and
an experiment directory. The wizard will ask you questions about datasets 
and some model settings. Modify the config file later with more advanced options.

### Validating a config with `credit check`

`credit check` resolves everything a config names without running anything: every
registry key (`model.type`, `trainer.type`, `loss.type`, `dataset_type`, each
pre/postblock `type`), every block's `args` against the real constructor
signature, the channel layout against the model geometry, the BaseLoss
target-twin postblock chain, and the existence of every file the config points
at. Each finding comes with the fix where the fix is unambiguous.

```bash
credit check -c my_experiment.yml            # static checks, no data touched
credit check -c my_experiment.yml --deep     # also construct model/blocks/loss
credit check -c my_experiment.yml --strict   # exit non-zero on warnings too
credit check -c my_experiment.yml --json     # machine-readable output
```

> **More detail**: [Config reference](config.md) | [Training guide](Training.md)

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


## 5. Visualize a prediction

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

## 6. Get help from the AI assistant

`credit ask` is a unified AI assistant — it automatically runs in agent mode (reads files,
runs commands, iterates) when Anthropic is available, or falls back to simple chat
(Groq, Gemini, OpenAI) otherwise.

```bash
pip install "miles-credit[ask]"

# Set whichever key you have — free options work well for quick questions:
export GROQ_API_KEY=gsk_...           # https://console.groq.com       (free, no card needed)
export GOOGLE_API_KEY=AIza...         # https://aistudio.google.com    (free)
export OPENAI_API_KEY=sk-...          # https://platform.openai.com
export ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com  (enables agent mode)

credit ask "how do I resume a failed Derecho job?"
credit ask -c my_run.yml "my loss stopped decreasing at epoch 12, what should I check?"
```

| Provider | Env var | Mode | Cost |
|----------|---------|------|------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Agent (multi-turn, reads files) | ~$0.01–0.05/session |
| OpenAI | `OPENAI_API_KEY` | Simple chat | Pay-per-use |
| Google | `GOOGLE_API_KEY` | Simple chat | Free |
| Groq | `GROQ_API_KEY` | Simple chat | Free tier (no card needed) |

Priority when multiple keys are set: Anthropic agent → OpenAI → Google → Groq.

```bash
# Agent mode: reads your PBS log, config, and source to give a specific answer
credit ask -c my_run.yml "why did my training run crash?"
credit ask -c my_run.yml "review this config before I start a 200-epoch run on 8 H100s"
credit ask "what PBS jobs are running and how much walltime do they have left?"
```

See the full [AI Assistant documentation](agent.md) for all examples, options, and cost details.

## Common problems

| Symptom | Fix |
|---------|-----|
| Training hangs on startup, no error | DataLoader is using too much RAM. Set `thread_workers: 1` and `prefetch_factor: 1` in your config. |
| `RendezvousConnectionError` on Derecho | Use `--nodes 1` so the job gets `torchrun --standalone` instead of MPI rendezvous. |
| `ANTHROPIC_API_KEY is not set` | Run `export ANTHROPIC_API_KEY=sk-ant-...` or add it to `~/.bashrc`. |
| PBS chain cancelled after job failure | Expected — PBS `afterok` cancels remaining jobs. Use `--reload --chain N` to restart. |
| Checkpoint not found on first run | Normal — set `load_weights: False` in config (the default). |
| Out of GPU memory | Reduce `train_batch_size`. For 0.25° start with `train_batch_size: 1`. |
