# Ensemble Inference with IC Perturbations

CREDIT supports ensemble forecasting using a **deterministic model with perturbed initial
conditions**. Any trained deterministic checkpoint works — no special ensemble training is
required. Spread is generated entirely through IC perturbation, so the quality depends on
the perturbation method chosen.

1.  Perturbing initial conditions with deterministic models
2.  Utilizing stochastic models with identical initial conditions

The inference scripts (`rollout_metrics_noisy_ics.py`, `rollout_metrics_noisy_models.py`) will compute and save **ensemble metrics only**. To **save forecast outputs to NetCDF**, use `credit rollout` with `ensemble_size > 1` — either set it in the `predict` block of the config file or pass `--ensemble-size N` at the CLI. The two scripts presented here compute the CRPS score for each variable and keep track of the ensemble member scores, means and standard deviations.

For SDL (Stochastic Decoder Layer) ensembles where spread comes from learned model noise
rather than IC perturbations, see [SDL Ensemble](SDL_ensemble.md).

---

## Two use cases

| Goal | Script | Output |
|------|--------|--------|
| Compute CRPS / spread / RMSE metrics | `rollout_metrics_noisy_ic.py` | per-init CSV files |
| Save full ensemble NetCDF forecasts | `rollout_to_netcdf.py` with `ensemble_size > 1` | NetCDF with ensemble dim |

Both read the same config file. The only difference is which script you invoke.

---

## Quick start

**Step 1:** Copy your deterministic inference config (`model.yml`).

**Step 2:** Add the `ensemble` block under `predict:` (see examples below).

**Step 3:** Run:

```bash
# metrics only (CRPS, spread, RMSE per init date)
python applications/rollout_metrics_noisy_ic.py -c model.yml

# submit to PBS cluster
python applications/rollout_metrics_noisy_ic.py -c model.yml -l 1

# save full NetCDF ensemble output instead
python applications/rollout_to_netcdf.py -c model.yml
```

---

## Config reference

All ensemble options live under `predict.ensemble`. The number of members is set
by `predict.ensemble_size`.

```yaml
predict:
  ensemble_size: 16          # number of ensemble members (1 = deterministic)

  ensemble:

    # ------------------------------------------------------------------
    # noise — controls the spatial structure of IC perturbations.
    # Required for all perturbation modes (simple noise and bred vectors).
    # ------------------------------------------------------------------
    noise:
      type: "gaussian"       # "gaussian" | "red" | "spherical"
      amplitude: 0.15        # perturbation amplitude (z-score units)

      # Only for type: "red"
      reddening: 2           # spectral exponent: 0=white, 1=pink, 2=brown/red

      # Only for type: "spherical"
      smoothness: 2.0        # Matérn smoothness; higher = smoother spatial fields
      length_scale: 3.0      # characteristic spatial scale (grid cells)

      # Optional: per-channel amplitude scaling (array, length = n_channels)
      # weights: [1.0, 1.2, 0.8, ...]

    # ------------------------------------------------------------------
    # bred_vector — set enabled: true to use the cyclic breeding algorithm
    # instead of simple one-shot IC noise.
    # The noise block above still controls the spatial structure of the
    # initial perturbation used inside each breeding cycle.
    # ------------------------------------------------------------------
    bred_vector:
      enabled: false         # true = bred vectors, false = simple IC noise

      amplitude: 0.15        # perturbation amplitude used during breeding
      num_cycles: 5          # number of breeding cycles
      integration_steps: 1   # model steps per cycle
      hemispheric_rescale: true  # latitude-dependent amplitude scaling
      bred_time_lag: 480     # hours before forecast init to start breeding
      perturb_channel_idx: null  # restrict perturbation to one channel (null = all)
      # weights: [1.0]       # optional per-channel amplitude scaling

    # ------------------------------------------------------------------
    # temporal_noise — AR(1) evolution of perturbations across forecast steps.
    # Can be combined with either simple noise or bred vectors.
    # ------------------------------------------------------------------
    temporal_noise:
      enabled: false
      temporal_correlation: 0.9  # AR(1) coefficient (0–1)
      perturbation_std: 0.15
      hemispheric_rescale: true
```

> `ensemble_size` under `predict` is independent of any `ensemble_size` set during training.

---

## Noise types

### `gaussian` — spatially uncorrelated white noise

```yaml
noise:
  type: "gaussian"
  amplitude: 0.15
```

Adds independent random perturbations at every grid point. Fast and simple. Spatial
structure is unrealistically noisy; best used as a baseline.

---

### `red` — colored noise with power-law spectrum

```yaml
noise:
  type: "red"
  amplitude: 0.15
  reddening: 2    # 0 = white, 1 = pink, 2 = brown/red
```

Produces spectrally correlated perturbations whose variance falls as 1/f^`reddening`.
`reddening: 2` (brown/red noise) matches observed atmospheric error spectra and is the
recommended default for large-scale variables (Z500, T500, U500).

---

### `spherical` — Matérn covariance on the sphere

```yaml
noise:
  type: "spherical"
  amplitude: 0.15
  smoothness: 2.0    # Matérn α — regularity; higher = smoother fields
  length_scale: 3.0  # τ — spatial scale (grid cells)
```

Uses spherical harmonics to generate perturbations with physically consistent spatial
structure on the lat/lon sphere. The most expensive option to compute but provides the
highest spatial fidelity.

---

## Bred vectors

Bred vectors approximate the leading perturbation growth directions of the model,
analogous to the NCEP operational breeding method. Use this when you want perturbations
that reflect the model's own error growth dynamics.

**Algorithm:**

1. Load ERA5 state `bred_time_lag` hours before the target forecast init.
2. Perturb it with the chosen `noise` type at `bred_vector.amplitude`.
3. Integrate both perturbed and unperturbed states forward `integration_steps` steps.
4. Compute the difference, rescale it back to the target amplitude.
5. Repeat `num_cycles` times.
6. Add the final bred vector to the actual forecast initial condition.

**Cost:** Each member requires `num_cycles` extra forward passes before the forecast
begins. With `num_cycles: 5` and `integration_steps: 1`, that is 5 extra model steps per
member — modest for a 240 h forecast.

**Key parameters:**

| Parameter | Effect |
|-----------|--------|
| `num_cycles` | More cycles → perturbations better aligned with model instabilities |
| `bred_time_lag` | Longer lag → more time for perturbation growth; 480 h (20 days) is a good default |
| `hemispheric_rescale` | Amplifies perturbations at higher latitudes where model errors grow faster |
| `perturb_channel_idx` | Restrict breeding to one channel for targeted experiments (e.g. Z500 only) |

---

## Complete config examples

### Example 1 — Gaussian noise, 10 members

```yaml
predict:
  mode: none
  ensemble_size: 10
  save_forecast: '/glade/derecho/scratch/<you>/CREDIT_runs/ens_gauss/'

  ensemble:
    noise:
      type: "gaussian"
      amplitude: 0.15
    bred_vector:
      enabled: false
    temporal_noise:
      enabled: false

  forecasts:
    type: custom
    start_year: 2022
    start_month: 1
    start_day: 1
    start_hours: [0]
    duration: 10
```

### Example 2 — Bred vectors with red noise, 16 members

```yaml
predict:
  mode: none
  ensemble_size: 16
  save_forecast: '/glade/derecho/scratch/<you>/CREDIT_runs/ens_bred/'

  ensemble:
    noise:
      type: "red"
      amplitude: 0.12
      reddening: 2

    bred_vector:
      enabled: true
      amplitude: 0.12
      num_cycles: 5
      integration_steps: 1
      hemispheric_rescale: true
      bred_time_lag: 480

    temporal_noise:
      enabled: false

  forecasts:
    type: custom
    start_year: 2022
    start_month: 1
    start_day: 1
    start_hours: [0]
    duration: 10
```

### Example 3 — Spherical noise with temporal correlation, 20 members

```yaml
predict:
  mode: none
  ensemble_size: 20
  save_forecast: '/glade/derecho/scratch/<you>/CREDIT_runs/ens_sph/'

  ensemble:
    noise:
      type: "spherical"
      amplitude: 0.10
      smoothness: 2.0
      length_scale: 3.0

    bred_vector:
      enabled: false

    temporal_noise:
      enabled: true
      temporal_correlation: 0.9
      perturbation_std: 0.10
      hemispheric_rescale: true

  forecasts:
    type: custom
    start_year: 2022
    start_month: 1
    start_day: 1
    start_hours: [0]
    duration: 10
```

---

## Execution

### Local run

```bash
python applications/rollout_metrics_noisy_ic.py -c model.yml
```

### Batch Job Submission

To submit ensemble rollout jobs to the cluster use `credit submit --rollout`:

```bash
# Submit 10 parallel PBS jobs, ensemble_size set in config
credit submit --cluster derecho -c config.yml --rollout --jobs 10

# Override ensemble size at submission time
credit submit --cluster derecho -c config.yml --rollout --jobs 10 --ensemble-size 50
```

`--jobs` splits init times across N independent PBS jobs. `ensemble_size` (config or `--ensemble-size`) sets members per init time.

For the legacy metric-only scripts, use the `-l 1` flag:

```
python rollout_metrics_noisy_ics.py --config model.yml -l 1
python rollout_metrics_noisy_models.py --config model.yml -l 1
```

### Multi-GPU (DDP)

```yaml
predict:
  mode: ddp
```

## Metrics and Output

```bash
torchrun --nproc_per_node=4 applications/rollout_metrics_noisy_ic.py -c model.yml
```

---

## Output files

`rollout_metrics_noisy_ic.py` writes two CSV files per initialization date to
`{predict.save_forecast}/metrics/`:

| File | Content |
|------|---------|
| `{datetime}_ensemble.csv` | Per-member RMSE and MAE for every channel and forecast step |
| `{datetime}_average.csv`  | Ensemble-mean RMSE, spread (std), and CRPS per channel/step |

To aggregate and plot these CSVs, use `ensemble_eval.py` (see `config/example_ensemble_eval.yml`).
To compute WeatherBench-style CRPS/RMSE in the standardized NetCDF format, run
`ensemble_wb2_verif.py` on the saved NetCDF outputs from `rollout_to_netcdf.py`.
