# Smoke test: ARCO ERA5 `co/` (native spectral) pipeline on Casper

**Audience:** Claude Code running on Casper, executing on behalf of dgagne.
**Branch:** `era5_model_2` (commit `1a1c3598` — "feat: on-the-fly spectral synthesis from native ARCO ERA5 co/ stores").
**Config for everything (check, preprocess, train):** `config/gen_2/examples/arco_era5_co_wxformer.yml`.

## Context

This branch adds a gen2 data source (`dataset_type: arco_era5_co`) that streams ERA5 in its
*native* IFS representation from the ARCO `co/` zarr stores — temperature/vorticity/divergence
as T639 spherical-harmonic coefficients, moisture and 2D fields on the N320 reduced Gaussian
grid — plus a `spectral_to_grid` preblock that synthesizes grid-point fields (deriving u/v from
vorticity+divergence) and a bilinear reduced-Gaussian→0.25° regrid, all on the fly inside the
per-step preblock chain feeding wxformer.

The math is already validated on a Mac (offline unit tests + a remote test matching the ARCO
analysis-ready 0.25° product to ~0.1 K / <0.5 m/s RMSE). **This smoke test's job is what a Mac
can't do:** run the chain on a real GPU (A100) end-to-end via `credit preprocess` and a short
`credit train`, and measure timing and GPU memory.

Everything below is safe to run: it only reads public GCS data and writes under
`/glade/derecho/scratch/$USER/`. Do not push commits from Casper; report findings instead.

## 0. Environment

```bash
cd <miles-credit checkout>          # or git clone git@github.com:NCAR/miles-credit.git
git fetch origin && git checkout era5_model_2 && git pull
# Shared NCAR env (see CLAUDE.md); adapt to whatever conda env is standard here:
conda activate credit-casper        # or the current shared env name
pip install -e . --no-deps
python -c "import credit, torch, torch_harmonics; print(credit.__file__, torch.__version__)"
```

Casper nodes have outbound internet; all ARCO reads are anonymous GCS (no credentials needed).

```bash
export EXP=/glade/derecho/scratch/$USER/credit_models/arcoera5_co_wxformer
mkdir -p $EXP
```

Note: the config uses `$SCRATCH` for the two setup files and hardcodes
`save_loc: /glade/derecho/scratch/$USER/credit_models/arcoera5_co_wxformer/`. On Casper
`$SCRATCH` should already be `/glade/derecho/scratch/$USER`; verify with `echo $SCRATCH`
and export it if unset.

## 1. Unit tests (fast, ~1 min + one remote test ~30 s)

```bash
python -m pytest tests/test_spectral_preblock.py -q
```

Expect **21 passed** (20 offline + 1 remote test that streams one ARCO timestep and asserts
t/u/v at level 137 match the analysis-ready product: RMSE < 0.25 K / 0.5 m/s). If the remote
test is slow or flaky on the network, `SKIP_REMOTE=1` skips it; the offline 20 must pass.

## 2. One-time setup files (CPU, ~2 min)

The preblocks need two NetCDF files at build time (paths referenced by the config):

```bash
python - <<'EOF'
import numpy as np, os
from credit.reduced_gaussian import fetch_arco_n320_grid, reduced_gaussian_to_latlon_bilinear_weights
scratch = os.environ["SCRATCH"]
g = fetch_arco_n320_grid(f"{scratch}/N320_grid.nc")
print("rings:", g.n_rings, "points:", g.n_points)   # expect 640 / 542080
reduced_gaussian_to_latlon_bilinear_weights(
    g, 90.0 - 0.25 * np.arange(721), 0.25 * np.arange(1440),
    f"{scratch}/n320_to_0p25_bilinear.nc")
EOF
```

## 3. Static config validation (CPU)

```bash
credit check -c config/gen_2/examples/arco_era5_co_wxformer.yml --deep
```

Expect `OK: 0 error(s)`. Acceptable warnings: unset `pbs.queue`/`pbs.walltime`, and
"Scaler file does not exist yet" (fixed by step 5). The deep pass constructs the real
`SpectralToGrid` (builds the ~1 GB T639 Legendre table, ~2 s) and the `Regridder`
(3.9M-entry sparse weight file).

## 4. Get a GPU node

The chain's first preblock is `to_device: cuda`, so **preprocess and train both need a GPU**
(on a CPU-only node they fail at the first `.to("cuda")`). Interactive session:

```bash
qsub -I -q casper -A <account> \
     -l select=1:ncpus=8:ngpus=1:mem=200GB -l walltime=02:00:00 -l gpu_type=a100
```

Use dgagne's default PBS account (`$PBS_ACCOUNT` or the `pbs.project` in the config,
currently `NAML0001`). Re-activate the conda env and re-export `EXP` inside the job.

## 5. `credit preprocess` (fits the bridgescaler)

Preprocess runs the per-step preblock chain (to_device → spectral synthesis → regrid → log)
on `trainer.batches_per_epoch` = **10** IC batches and fits the standard scaler on the resulting
0.25° physical fields, saving `$EXP/standard_scaler.json`.

```bash
# NOTE: preprocess has no single-process fallback — it calls get_rank_info()
# unconditionally and exits with "Can't find the environment variables for
# local rank" under a plain `credit preprocess`. Launch it under torchrun:
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    "$(python -c 'import credit.applications.preprocess as m; print(m.__file__)')" \
    -c config/gen_2/examples/arco_era5_co_wxformer.yml
```

What to expect / verify:

- Each batch streams roughly 1–2 GB from GCS (3 spectral vars × 137 levels ≈ 900 MB from the
  wind store alone), so wall time is network-bound: expect a few minutes per batch, tens of
  minutes total. Log lines: `Processing batch i of 10`.
- On completion the log prints fitted per-variable means/stds. **Sanity-check them in physical
  units**: temperature mean ≈ 220–290 K per level band; u/v means near 0 with std ~5–30 m/s.
  Vorticity/divergence should NOT appear — they were consumed by the wind derivation. If u/v stds
  are wildly wrong (e.g. hundreds of m/s) the wind derivation is broken: stop and report.
- The log-transformed variables are **not** bare `log(x)`: `LogTransform` applies
  `y = log(x + eps) - log(eps)` with `eps=1e-8`, i.e. every value carries a `+18.42` offset so
  that `y=0` at `x=0`. So expect `surface_pressure` mean ≈ **29.9** (= ln(101325) + 18.42), and
  `specific_humidity` **positive**, ≈ 5 at the model top rising to ≈ 13 near the surface. To
  recover physical units subtract 18.42 and exponentiate.
- **Check for `mean=nan` in the fitted table** (`grep -c 'mean=nan'` on the log — expect 0). ERA5's
  semi-Lagrangian moisture advection leaves small negative specific humidity (order -1e-6 kg/kg) at
  scattered upper-tropospheric points; `log_transform` maps anything below `-eps` to NaN and poisons
  that entire level's statistics. The config's `clamp_q` preblock clamps negatives to zero before the
  log — if it is missing or ordered after `log_trans`, expect ~10 NaN levels between 60 and 95.
- `$EXP/ERA5_reduced_gaussian_grid.nc` should appear (written by the dataset on first read).
- 10 batches is statistically thin but fine for a smoke test. (For a production scaler, raise
  `trainer.batches_per_epoch` temporarily — preprocess uses that count — then restore it.)

## 6. Smoke training run (2 epochs × 10 batches)

Optional first: `credit submit --cluster casper -c config/gen_2/examples/arco_era5_co_wxformer.yml --gpus 1 --dry-run`
to inspect the generated PBS script. For the interactive session just run:

```bash
credit train -c config/gen_2/examples/arco_era5_co_wxformer.yml
```

Before launching, consider two safe edits to a **copy** of the config for the smoke run:
- `trainer.valid_batch_size: 4 → 1` (validation at batch 4 on 721×1440×137 is the memory peak);
- `trainer.skip_validation: True` if you only want the training-path signal.

Watch for:

- **GPU memory** (`nvidia-smi` / torch max_memory_allocated): budget ≈ 1.05 GB Legendre table
  + ~60 MB sparse regrid weights + wxformer activations (checkpointed). Report the peak. If a
  40 GB A100 OOMs, rerun on 80 GB or shrink the level list (see Troubleshooting).
- **Per-batch time**, and whether it is dominated by GCS streaming (likely) or compute. The
  synthesis itself should be ~tens of ms on the A100 (it was 0.23 s for 30 level-fields on a
  laptop CPU).
- **Loss is finite** from step 1 and does not NaN through both epochs. With 10 batches/epoch
  don't expect meaningful convergence — finite + roughly decreasing is a pass.

## 7. Success criteria & what to report back

Pass = all of:

1. `pytest tests/test_spectral_preblock.py` green (step 1).
2. `credit check --deep` → 0 errors (step 3).
3. Preprocess writes `$EXP/standard_scaler.json` with physically sane per-variable stats (step 5).
4. Two training epochs complete; `$EXP/training_log.csv` has finite losses; checkpoint,
   `channel_schema.yaml`, and `output_grid_schema.nc` (721×1440 rectilinear) are written.

Report: peak GPU memory, seconds/batch (train + preprocess), the fitted mean/std table for
temperature / u / v / surface_pressure, and any warnings that looked new. If anything fails,
include the full traceback and the batch index it failed on.

## 8. Visual confirmation
After completing training, perform some visualization checks for manual verification of correct structures in the data. Make matplotlib plots of u, v, t, and q at lowest model level on a global map with cartopy plotting coastlines. Compare against the Analysis ready arco data to make sure the plots are consistent. 

## Troubleshooting

- **`RuntimeError ... CUDA` at the very first batch** → you're on a CPU node (see step 4), or
  the `to_device` block was removed. Preprocess/train both require the GPU.
- **OOM** → reduce the vertical resolution for the smoke test: in your config copy set
  `data.source.ERA5.levels` to a subset (e.g. `[10, 30, 50, 70, 90, 100, 110, 120, 130, 137]`)
  **and** `model.levels` to its length (10). Delete any stale `$EXP/channel_schema.yaml` +
  `standard_scaler.json` and re-run preprocess — the schema/scaler are level-count specific.
  Also point `save_loc` and the scaler path at a fresh directory to avoid mixing artifacts.
- **Very slow batches / hangs** → GCS streaming; check a login-node `curl -sI https://storage.googleapis.com`
  and per-batch log timing. `trainer.thread_workers: 4` dataloader workers each open their own
  store connections; try 2 if the node's network is saturated.
- **`credit check` errors about grid_file/weight_file not found** → step 2 files are missing or
  `$SCRATCH` differs from where you wrote them.
- **NaNs in loss** → first check the fitted scaler stats (step 5 sanity list); a bad scaler fit
  (e.g. fitted on NaN sea_ice_cover before the fill block — ordering bug) shows up there, not in
  the model.
