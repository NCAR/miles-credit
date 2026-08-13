# CREDIT Quickstart

Get from zero to a running training job in under 10 minutes.

Full documentation: https://miles-credit.readthedocs.io/en/latest/quickstart.html

---

## 1. Set up your environment

**NCAR Casper:**
```bash
module load conda
conda create -n credit-casper -y python=3.13 uv
conda activate credit-casper
uv pip install miles-credit --extra-index-url https://download.pytorch.org/whl/cu126
```

CUDA 13 (PyTorch's default) does not work on Casper — the `cu126` index above does.

**NCAR Derecho:**
```bash
module load conda
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
./create_derecho_env.sh   # installs into the credit-derecho conda environment
```

The Derecho script builds PyTorch so distributed operations route over the
Slingshot interconnect — required for multi-node training.

**Other systems (Mac / laptop / generic Linux):**
```bash
conda create -n credit -y python=3.13 uv
conda activate credit
uv pip install miles-credit
```

---

## 2. Generate a config

The interactive wizard works anywhere (laptop or HPC) and asks about your
dataset, grid, and model settings:

```bash
credit begin
```

NCAR users can instead copy a ready-made template whose data paths already
point at the shared ERA5 archive on glade — no edits required to get started:

```bash
credit init --grid 1deg -o my_run.yml      # 1-degree ERA5, fast to train
credit init --grid 0.25deg -o my_run.yml   # 0.25-degree ERA5, full resolution
```

Either way, validate before running anything:

```bash
credit check -c my_run.yml
```

Fields you may want to change before your first run:

| Field | Default | Notes |
|-------|---------|-------|
| `trainer.num_epoch` | `5` | Epochs per PBS job |
| `trainer.train_batch_size` | `8` | Per-GPU; reduce if you hit OOM |
| `save_loc` | scratch dir | Where checkpoints are written |

---

## 3. Fit the scalers

Wizard-generated configs need a one-time preprocessing pass that fits the
normalization scalers and saves them to the config's `scaler_path`:

```bash
credit preprocess -c my_run.yml
```

> **NCAR users:** the `credit init` templates point at pre-fitted scalers on
> glade, so you can skip this step. Re-run it whenever you change the variable
> list or date range of a config with your own `scaler_path`.

---

## 4. Start training

**Locally (laptop / workstation / single GPU):**

```bash
credit train -c my_run.yml
```

**On NCAR HPC, submit a batch job:**

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

## 5. Monitor progress

```bash
# Quick loss check
tail -5 /path/to/save_loc/training_log.csv

# TensorBoard
tensorboard --logdir /path/to/save_loc/tensorboard
```

**Healthy training:** `train_loss` ≈ 1–3 after epoch 1, decreasing each epoch.

---

## 6. Visualise a prediction

```bash
# 3-panel global map: truth | prediction | difference
credit plot -c my_run.yml --field VAR_2T --denorm
```

Plots are saved to `<save_loc>/plots/`. No GPU required.

---

## 7. Get help — `credit ask`

`credit ask` is a unified AI assistant — agent mode (reads your files, runs commands,
iterates to a confident answer) when Anthropic is available, simple chat otherwise.

```bash
pip install "miles-credit[ask]"

# Set whichever key you have — free options work great for quick questions:
export GROQ_API_KEY=gsk_...           # https://console.groq.com       (free, no card needed)
export GOOGLE_API_KEY=AIza...         # https://aistudio.google.com    (free for many institutions)
export OPENAI_API_KEY=sk-...          # https://platform.openai.com
export ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com  (agent mode, ~$0.01–0.05/session)

credit ask "my loss is stuck at 2.5 after 15 epochs, what should I check?"
credit ask -c my_run.yml "why did my training run crash?"
credit ask "what PBS jobs are running and how much walltime do they have left?"
```

Full docs: https://miles-credit.readthedocs.io/en/latest/agent.html

---

## Common problems

| Symptom | Fix |
|---------|-----|
| Training hangs on startup | Set `thread_workers: 1` and `prefetch_factor: 1` |
| `RendezvousConnectionError` on Derecho | Use `--nodes 1` (single-node uses `--standalone`) |
| Loss > 100 or growing | Check `scaler_path` in the `bridgescaler_transform` preblock (or `mean_path` / `std_path` if using the gen1-style `era5_normalizer`) |
| PBS chain cancelled | Use `--reload` to restart from last checkpoint |
| Out of GPU memory | Reduce `train_batch_size` (try `1` for 0.25°) |
