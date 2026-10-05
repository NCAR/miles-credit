# SDL Ensemble Model — Quick Reference

The **Stochastic Decomposition Layer (SDL)** model is a probabilistic version of the CREDIT
WXFormer. It generates ensemble spread by injecting learned noise into the decoder at inference
time. Each forward pass produces a different realization; running N forward passes from the
same initial condition gives an N-member ensemble.

---

## Checkpoints and model types

| Model | Type key | Noise | Location |
|---|---|---|---|
| SDL ensemble (6 hr) | `crossformer-style` | ✓ | `/glade/campaign/cisl/aiml/credit/models/sdl_camulator/checkpoint.pt` |
| Deterministic arXiv 6 hr multi-step | `crossformer` | ✗ | HuggingFace `NCAR/miles-credit-wxformer_6h_multi_step` |
| Deterministic arXiv 6 hr single-step | `crossformer` | ✗ | HuggingFace `NCAR/miles-credit-wxformer_6h_single_step` |
| Deterministic arXiv 1 hr single-step | `crossformer` | ✗ | HuggingFace `NCAR/miles-credit-wxformer_1h_single_step` |

> **Note:** Only the deterministic models are on HuggingFace. The SDL ensemble checkpoint lives
> on NCAR campaign storage and requires access to NCAR systems.

The SDL checkpoint was trained with a Generation 1 (flat-schema) config, so the ensemble
workflow below uses the Gen 1 `predict:` block and the `credit_rollout_metrics` script.

---

## Turning noise injection on and off

The SDL model's noise amplitude is set by `decoder_noise_factor` in the model config. A
`noise_scale` key lets you override it at runtime without editing the checkpoint or the model
config. It goes in the `predict:` block for Gen 1 rollouts and in the `inference:` block for
Gen 2 rollouts (`credit rollout`):

```yaml
predict:             # Gen 1 (credit_rollout_metrics); use inference: for Gen 2
  noise_scale: 1.0   # default — use the trained noise factors as-is
  # noise_scale: 0.0 # disable noise entirely → deterministic run from SDL checkpoint
  # noise_scale: 0.5 # halve the noise amplitude
```

`noise_scale` multiplies every SDL `noise_factor` parameter in the model (the three decoder
injection points) before the rollout starts. Setting it to `0.0` turns the SDL model into a
deterministic model while keeping all other learned weights intact.

In short:
- **Using a deterministic checkpoint** (HuggingFace models above) → noise is never present.
- **Using the SDL checkpoint with `noise_scale: 0.0`** → same model weights, noise disabled.
- **Using the SDL checkpoint with `noise_scale: 1.0` and `ensemble_size: N`** → N independent
  probabilistic members from a single initial condition.

---

## Minimal config for ensemble rollout (NCAR systems)

Save the following as `ensemble_6hr.yml` and replace `<you>` in the output paths. All data
paths point to readable campaign storage — no downloads needed on Derecho/Casper.

```yaml
save_loc: '/glade/derecho/scratch/<you>/CREDIT_runs/my_ensemble/'
seed: 1000

data:
  variables: ['U', 'V', 'T', 'Q']
  save_loc: '/glade/campaign/cisl/aiml/credit/era5_zarr/SixHourly_y_TOTAL*staged.zarr'

  surface_variables: ['SP', 't2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']
  save_loc_surface: '/glade/campaign/cisl/aiml/credit/era5_zarr/SixHourly_y_TOTAL*staged.zarr'

  dynamic_forcing_variables: ['tsi']
  save_loc_dynamic_forcing: '/glade/campaign/cisl/aiml/credit/credit_solar_nc_6h_0.25deg/*.nc'

  static_variables: ['Z_GDS4_SFC', 'LSM']
  save_loc_static: '/glade/campaign/cisl/aiml/credit/static_scalers/static_norm_20250416.nc'

  mean_path: '/glade/campaign/cisl/aiml/credit/static_scalers/mean_6h_1979_2018_16lev_0.25deg.nc'
  std_path:  '/glade/campaign/cisl/aiml/credit/static_scalers/std_residual_6h_1979_2018_16lev_0.25deg.nc'

  scaler_type: std_new
  history_len: 1
  valid_history_len: 1
  lead_time_periods: 6
  forecast_len: 0
  valid_forecast_len: 0
  one_shot: True
  variables_levels: null
  diagnostic_variables: []
  forcing_variables: []

model:
  type: "crossformer-style"
  noise_latent_dim: 442
  decoder_noise_factor: 0.235
  encoder_noise: False
  # --- architecture (must match checkpoint) ---
  image_height: 640
  image_width: 1280
  levels: 16
  frames: 1
  frame_patch_size: 1
  channels: 4
  surface_channels: 7
  dynamic_forcing_channels: 1
  static_channels: 2
  diagnostic_channels: 0
  patch_size: [2, 4]
  dim: [256, 512]
  depth: [2, 2]
  global_window_size: [10, 20]
  local_window_size: 10
  cross_embed_kernel_sizes: [[4, 8, 16, 32], [2, 4]]
  cross_embed_strides: [2, 2]
  num_heads: 8
  attn_dropout: 0
  ff_dropout: 0
  spectral_norm: True
  # post-processing blocks
  post_conf:
    activate: False

predict:
  mode: none
  ensemble_size: 10          # number of independent noise realizations per init time
  noise_scale: 1.0           # set to 0.0 for a deterministic run
  save_forecast: '/glade/derecho/scratch/<you>/CREDIT_runs/my_ensemble/netcdf/'
  forecasts:
    type: "custom"
    start_year: 2022
    start_month: 1
    start_day: 1
    start_hours: [0]
    duration: 7              # number of initialization days (2022-01-01 to 2022-01-07)
    days: 10                 # forecast length in days (40 × 6 hr steps)
```

---

## Running the rollout

Activate your CREDIT environment, then run from a GPU node:

```bash
credit_rollout_metrics -c ensemble_6hr.yml
```

For each initialization time, the script rolls out all `ensemble_size` members together and
writes per-variable, per-step verification of the ensemble mean (ACC, RMSE, MSE, MAE) and
the ensemble spread (`std_<var>`) to `{save_loc}/metrics/{init_time}.csv`.

To run on several GPUs, set `predict.mode: ddp` (or pass `-m ddp`) and launch with `torchrun`;
the number of initialization times must be divisible by the number of GPUs:

```bash
torchrun --standalone --nproc-per-node=4 applications/rollout_metrics.py -c ensemble_6hr.yml -m ddp
```

To submit as a PBS job instead, add a `pbs:` block to the config and pass `-l 1`.

> **NetCDF output:** none of the current rollout scripts write all members of an SDL ensemble
> to NetCDF in one run. With a Gen 2 config, `credit rollout` writes one realization per init
> time; run it again with a different `seed` and `inference.save_forecast` for each additional
> member.

---

## Deterministic run from the SDL checkpoint

To use the SDL checkpoint but produce a single deterministic forecast (noise off):

```yaml
predict:
  mode: none
  ensemble_size: 1
  noise_scale: 0.0           # disables all SDL noise injection
  save_forecast: '/glade/derecho/scratch/<you>/CREDIT_runs/deterministic/'
  forecasts: ...             # same as above
```

The output will be identical on every run for a given initial condition.

---

## Reproducing paper results

The 2022 ERA5 ensemble verification results are archived at:

```
/glade/campaign/cisl/aiml/credit/models/sdl_camulator/era5_ensemble_2022/
```

Contents:
- `model.yml` — the exact config used for the paper runs
- `metrics_csv/` — per-init spread/RMSE CSVs
- `beta_data/` — post-training beta scaling experiments (noise amplitude sweeps)
- `ffs_output/` — forward flux sampling (hurricane genesis probability) output

The analysis notebooks are versioned in the repo at `notebooks/ensemble/`:

| Notebook | Purpose |
|---|---|
| [`ensemble_metrics.ipynb`](../../notebooks/ensemble/ensemble_metrics.ipynb) | Load `metrics_csv/`, compute paper numbers (spread, RMSE, CRPS) |
| [`ensemble_plots.ipynb`](../../notebooks/ensemble/ensemble_plots.ipynb) | Multi-panel verification plots with optional IFS comparison |
| [`spectra.ipynb`](../../notebooks/ensemble/spectra.ipynb) | KE spectra via spherical harmonics (Will Chapman) |

Each notebook has a `# === CONFIGURATION ===` cell at the top — edit `SCHEDULER_DIR` to point to
your data and run from top to bottom.

To regenerate the per-init metrics CSVs with the same checkpoint and config:

```bash
cp /glade/campaign/cisl/aiml/credit/models/sdl_camulator/model.yml ./paper_model.yml
# edit save_loc and predict.save_forecast in paper_model.yml
credit_rollout_metrics -c paper_model.yml
```

---

## Noise control internals

For programmatic control (e.g., in a notebook or custom script), use `SDLWrapper`:

```python
from credit.models import load_model
from credit.models.wxformer.sdl_inference_wrapper import SDLWrapper
import yaml

with open("ensemble_6hr.yml") as f:
    conf = yaml.safe_load(f)

model = load_model(conf, load_weights=True)
wrapper = SDLWrapper(model)

# Check the current noise factors (three decoder layers)
print(wrapper.get_noise_factors())  # e.g. [0.235, 0.235, 0.235]

# Disable noise entirely
wrapper.set_noise_factors(0.0)

# Restore trained values
wrapper.reset_to_original()

# Scale noise by a factor (e.g. double the spread)
factors = wrapper.get_noise_factors()
wrapper.set_noise_factors([f * 2.0 for f in factors])
```

`set_noise_factors` accepts a single float (applied to all layers) or a list of three floats
(one per decoder noise injection point: coarse, medium, fine scale).

To scale the noise in place without a wrapper, call
`credit.models.wxformer.stochastic_decomposition_layer.scale_sdl_noise(model, 0.5)`. This is
what the rollout scripts use for `noise_scale`.
