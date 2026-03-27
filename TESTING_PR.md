# Testing Guide — `trainer_refactor_v2` PR

This is a temporary file for testers reviewing this PR before merge.
It covers what changed, how to get set up, and what to exercise.

**Full documentation:** https://miles-credit.readthedocs.io/en/latest/

| Topic | Doc page |
|-------|----------|
| Installation & environment setup | https://miles-credit.readthedocs.io/en/latest/installation.html |
| Quickstart | https://miles-credit.readthedocs.io/en/latest/quickstart.html |
| Getting started guide | https://miles-credit.readthedocs.io/en/latest/getting-started.html |
| Config file reference | https://miles-credit.readthedocs.io/en/latest/config.html |
| Training | https://miles-credit.readthedocs.io/en/latest/Training.html |
| Inference / rollout | https://miles-credit.readthedocs.io/en/latest/Inference.html |
| Ensemble inference | https://miles-credit.readthedocs.io/en/latest/EnsemblesInference.html |
| Model architectures | https://miles-credit.readthedocs.io/en/latest/Model_Architectures.html |
| Datasets | https://miles-credit.readthedocs.io/en/latest/DataSets.html |
| Losses | https://miles-credit.readthedocs.io/en/latest/Losses.html |

---

## 1. Get the branch

```bash
git clone https://github.com/NCAR/miles-credit.git
cd miles-credit
git checkout trainer_refactor_v2
```

If you already have a clone:

```bash
git fetch origin
git checkout trainer_refactor_v2
git pull
```

---

## 2. Set up your environment

**Casper:**
```bash
module load conda
conda activate /glade/work/schreck/conda-envs/credit-main-casper
```

**Derecho:**
```bash
module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cuda/12.3.2 conda/latest
conda activate /glade/work/schreck/conda-envs/credit-main-derecho
```

Both envs have `miles-credit` installed in editable mode — no `pip install` needed.

Verify install:
```bash
credit --help
```

You should see subcommands: `train`, `rollout`, `realtime`, `submit`, `convert`, `init`, `plot`, `ask`.

---

## 3. What changed in this PR

### 3a. Config directory reorg

The `config/` directory is reorganized. New layout:

```
config/
  wxformer_1dg_6hr_v2.yml       ← start here (1-deg ERA5)
  wxformer_025deg_6hr_v2.yml    ← full-res ERA5
  starter_v2.yml                ← minimal template
  example-v2026.1.0.yml         ← fully annotated reference
  applications/                 ← specialist configs (ensemble, downscaling, etc.)
  data/                         ← dataset-specific configs
  dev/                          ← test/CI configs
  archive/                      ← old v1 configs (kept for reference)
```

Check that the config you were using previously still exists (it was likely moved, not deleted).
Old v1 ERA5 configs (`wxformer_1dg_6hr.yml`, `wxformer_6hr.yml`) are now in `config/archive/v1/`.

---

### 3b. PBS settings now read from config file

**Before this PR:** `credit submit` ignored the `pbs:` block in your config.
Account always defaulted to `NAML0001` unless you passed `--account` every time.

**After this PR:** `credit submit` reads from `pbs:` first, CLI flags override.

**Resolution order:** CLI flag → `pbs:` in config → `$PBS_ACCOUNT` env var → `NAML0001`

#### Set up your config's pbs: block

In your config YAML, add/update the `pbs:` section:

```yaml
pbs:
    project: "YOUR_ACCOUNT"      # ← your actual allocation code
    conda: "credit-derecho"      # ← your conda env name or full path
    walltime: "12:00:00"
    nodes: 1
    ngpus: 4
    ncpus: 64
    mem: '480GB'
    queue: 'main'
    job_name: "my_run"
```

#### Test it

```bash
# Should show YOUR_ACCOUNT in the job plan, not NAML0001
credit submit --cluster derecho -c my_config.yml --dry-run
```

Expected output includes:
```
====================================================
  Job plan
====================================================
  Cluster  : derecho
  Account  : YOUR_ACCOUNT        ← confirm this is correct
  Config   : my_config.yml
  GPUs     : 4 GPU(s)
  Walltime : 12:00:00 per job
====================================================
```

The generated PBS script should have `#PBS -A YOUR_ACCOUNT`.

**Also test Casper:**
```bash
credit submit --cluster casper -c my_config.yml --dry-run
```

**Test CLI override still works:**
```bash
# Should use OVERRIDE_ACCOUNT even if config says something else
credit submit --cluster derecho -c my_config.yml --account OVERRIDE_ACCOUNT --dry-run
```

---

### 3c. `credit convert` — v1 to v2 config migration

New interactive command to convert old-style configs.

```bash
credit convert -c config/archive/v1/wxformer_1dg_6hr.yml
```

You'll be asked a series of questions:
- Enable EMA? (new v2 feature, recommended)
- Enable TensorBoard?
- PBS settings (account, conda env, walltime, etc.)
- Output file path (defaults to `wxformer_1dg_6hr_v2.yml`)

**What it auto-applies (no questions):**
- `trainer.type: era5` → `era5-v2`
- `data.forecast_len`: +1 (v2 semantics: 1 = single step, v1 used 0)
- `data.valid_forecast_len`: +1
- `data.backprop_on_timestep`: shifted to 1-indexed

**Check the output** — open the generated `_v2.yml` and verify:
- `trainer.type` is `era5-v2`
- `data.forecast_len` is one higher than the original
- `trainer.use_ema` and `trainer.use_tensorboard` are set as you answered
- `pbs:` block has your account and conda env

**Try on your own v1 config if you have one.**

---

## 4. Smoke test: full submit dry-run

Use one of the built-in configs with your account filled in:

```bash
# Copy a starter config
cp config/wxformer_1dg_6hr_v2.yml /tmp/test_config.yml

# Edit pbs.project to your account
# (or just pass --account on the CLI)

# Dry run — prints PBS script, does not submit
credit submit --cluster derecho -c /tmp/test_config.yml --dry-run
credit submit --cluster casper  -c /tmp/test_config.yml --dry-run

# Chain dry run — shows job 1 + chained reload job
credit submit --cluster derecho -c /tmp/test_config.yml --chain 3 --dry-run
```

---

## 5. Things to look out for / report

- Any config that was in the old `config/` root that you can't find in the new structure
- `credit submit --dry-run` showing wrong account (should use `pbs.project`, not `NAML0001`)
- `credit convert` producing a broken YAML or wrong `forecast_len`
- Any `credit --help` output that looks wrong or missing

Report issues on the PR or ping @schreck directly.

---

## 6. What is NOT in this PR (don't test these yet)

- Ensemble rollout / `credit rollout-ensemble` (coming in a separate PR)
- WeatherBench verification
- New model architectures
