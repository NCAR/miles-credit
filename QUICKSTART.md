# CREDIT Quickstart

Get from zero to a running training job in under 10 minutes.

Full documentation: https://miles-credit.readthedocs.io/en/latest/quickstart.html

---

## 1. Set up your environment

**NCAR Casper:**
```bash
conda activate /glade/u/home/schreck/.conda/envs/credit-casper
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
pip install -e . --no-deps
```

**NCAR Derecho:**
```bash
conda activate /glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
pip install -e . --no-deps
```

**Other systems:**
```bash
conda create -n credit python=3.11 && conda activate credit
pip install miles-credit
```

---

## 2. Generate a config

```bash
credit init --grid 1deg -o my_run.yml      # 1-degree ERA5, fast to train
credit init --grid 0.25deg -o my_run.yml   # 0.25-degree ERA5, full resolution
```

> **NCAR users:** data paths already point to `/glade/campaign/cisl/aiml/ksha/CREDIT_data/`.
> No edits required to get started.

Fields you may want to change before your first run:

| Field | Default | Notes |
|-------|---------|-------|
| `trainer.num_epoch` | `5` | Epochs per PBS job |
| `trainer.train_batch_size` | `8` | Per-GPU; reduce if you hit OOM |
| `save_loc` | scratch dir | Where checkpoints are written |

---

## 3. Submit a training job

```bash
# Casper — chain auto-computed from config (epochs / num_epoch)
credit submit --cluster casper  -c my_run.yml --gpus 4

# Derecho single-node
credit submit --cluster derecho -c my_run.yml --gpus 4 --nodes 1

# Derecho multi-node
credit submit --cluster derecho -c my_run.yml --gpus 4 --nodes 4
```

`credit submit` automatically:
- Computes how many jobs to chain from `trainer.epochs / trainer.num_epoch`
- Prints a job plan (cluster, GPUs, chain length, memory estimate) before submitting
- Wires PBS `afterok` dependencies so jobs run back-to-back automatically

Preview without submitting:
```bash
credit submit --cluster casper -c my_run.yml --gpus 4 --dry-run
```

Resume a failed chain from the last checkpoint:
```bash
credit submit --cluster derecho -c my_run.yml --gpus 4 --nodes 1 --reload
```

---

## 4. Monitor progress

```bash
# Quick loss check
tail -5 /path/to/save_loc/training_log.csv

# TensorBoard
tensorboard --logdir /path/to/save_loc/tensorboard
```

**Healthy training:** `train_loss` ≈ 1–3 after epoch 1, decreasing each epoch.

---

## 5. Visualise a prediction

```bash
# 3-panel global map: truth | prediction | difference
credit plot -c my_run.yml --field VAR_2T --denorm
```

Plots are saved to `<save_loc>/plots/`. No GPU required.

---

## 6. Get help — `credit ask`

`credit ask` is a unified AI assistant — agent mode when Anthropic is available,
simple chat otherwise.

**NCAR users on Casper or Derecho:** shared Anthropic credits are available for all NCAR staff.
Add these two lines to `~/.bashrc` — no personal API key or billing required:

```bash
module use /glade/work/bdobbins/llms/modules
module load llms
```

Then source it (`source ~/.bashrc`) and the tool works immediately:

```bash
pip install "miles-credit[ask]"

# Agent mode (Anthropic): reads your files, runs commands, iterates to a confident answer
credit ask -c my_run.yml "why did my training run crash?"
credit ask "what PBS jobs are running and how much walltime do they have left?"

# Falls back to simple chat when Anthropic is unavailable
credit ask "my loss is stuck at 2.5 after 15 epochs, what should I check?"
```

**Non-NCAR users:** set any of these keys and the right mode is chosen automatically:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com  (agent mode, ~$0.01–0.05/session)
export OPENAI_API_KEY=sk-...          # https://platform.openai.com    (simple chat)
export GOOGLE_API_KEY=AIza...         # https://aistudio.google.com    (simple chat, free for many institutions)
export GROQ_API_KEY=gsk_...           # https://console.groq.com       (simple chat, free, no card needed)

pip install "miles-credit[ask]"
credit ask "my loss is stuck at 2.5 after 15 epochs, what should I check?"
```

Full docs: https://miles-credit.readthedocs.io/en/latest/agent.html

---

## Common problems

| Symptom | Fix |
|---------|-----|
| Training hangs on startup | Set `thread_workers: 1` and `prefetch_factor: 1` |
| `RendezvousConnectionError` on Derecho | Use `--nodes 1` (single-node uses `--standalone`) |
| Loss > 100 or growing | Check `mean_path` / `std_path` in config |
| PBS chain cancelled | Use `--reload` to restart from last checkpoint |
| Out of GPU memory | Reduce `train_batch_size` (try `1` for 0.25°) |
