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

---

## Turning noise injection on and off

The SDL model's noise amplitude is set by `decoder_noise_factor` in the model config. The
`noise_scale` key in the `predict:` block lets you override it at runtime without editing the
checkpoint or the model config:

```yaml
predict:
  noise_scale: 1.0   # default — use the trained noise factors as-is
  # noise_scale: 0.0 # disable noise entirely → deterministic run from SDL checkpoint
  # noise_scale: 0.5 # halve the noise amplitude
```

`noise_scale` multiplies all three decoder `noise_factor` parameters before the rollout starts.
Setting it to `0.0` effectively turns the SDL model into a deterministic model while keeping all
other learned weights intact.

To answer Greg Hakim's question directly:
- **Using a deterministic checkpoint** (HuggingFace models above) → noise is never present.
- **Using the SDL checkpoint with `noise_scale: 0.0`** → same model weights, noise disabled.
- **Using the SDL checkpoint with `noise_scale: 1.0` and `ensemble_size: N`** → N independent
  probabilistic members from a single initial condition.

---

## Minimal config for ensemble rollout (NCAR systems)

Save the following as `ensemble_6hr.yml` and edit the `predict:` paths. All data paths point to
readable campaign storage — no downloads needed on Derecho/Casper.

```yaml
save_loc: '/glade/scratch/<you>/CREDIT_runs/my_ensemble/'
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
  save_forecast: '/glade/scratch/<you>/CREDIT_runs/my_ensemble/netcdf/'
  forecasts:
    type: "custom"
    start_year: 2022
    start_month: 1
    start_day: 1
    start_hours: [0]
    end_year: 2022
    end_month: 1
    end_day: 7
    duration: 40             # 40 × 6 hr = 10-day forecast
```

---

## Running the rollout

### On Casper (single GPU, 10 members)

```bash
conda activate /glade/work/schreck/conda-envs/credit-main-casper
credit rollout-ensemble --cluster casper -c ensemble_6hr.yml --jobs 4
```

This splits the init times across 4 parallel Casper jobs. Each job runs all 10 ensemble members
for its slice of init times sequentially.

### On Derecho (single GPU, any size ensemble)

```bash
credit rollout-ensemble --cluster derecho -c ensemble_6hr.yml --jobs 4
```

### Manually (one init time, no PBS)

```bash
cd /glade/work/schreck/repos/miles-credit
python applications/rollout_to_netcdf_v2.py -c ensemble_6hr.yml
```

---

## Deterministic run from the SDL checkpoint

To use the SDL checkpoint but produce a single deterministic forecast (noise off):

```yaml
predict:
  mode: none
  ensemble_size: 1
  noise_scale: 0.0           # disables all SDL noise injection
  save_forecast: '/glade/scratch/<you>/CREDIT_runs/deterministic/'
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
- `ensemble_metrics.ipynb`, `ensemble_plots.ipynb` — the plotting notebooks
- `beta_data/` — post-training beta scaling experiments (noise amplitude sweeps)
- `ffs_output/` — forward flux sampling (hurricane genesis probability) output

To reproduce the rollout from scratch using the same checkpoint and config:

```bash
cp /glade/campaign/cisl/aiml/credit/models/sdl_camulator/model.yml ./paper_model.yml
# edit save_forecast path in paper_model.yml
credit rollout-ensemble --cluster casper -c paper_model.yml --jobs 10
```

Then run `ensemble_eval.py` against the output:

```bash
python applications/ensemble_eval.py -c config/applications/ensemble/example_ensemble_eval.yml
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
print(wrapper.get_noise_factors())       # e.g. [0.235, 0.235, 0.235]

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
