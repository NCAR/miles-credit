# Verification Metrics

The Gen 2 metrics framework scores model output per variable, in physical units,
during training and validation. It mirrors the Gen 2 loss framework described in
[Working with Loss Functions](Losses.md) — same `{type, args}` config shape, same
postblock contract, same variable- and latitude-weighting options — but for
*evaluation* rather than optimization: metrics run under `torch.no_grad` and return
detached Python floats for logging.

```{note}
This page covers metrics computed **during training**, logged to TensorBoard and
`training_log.csv`. For offline verification of a completed rollout, see
[Evaluation](Evaluation.md).
```

## Quick start

The `metrics` section is optional. Omit it and you get a combined metric of `rmse`,
`r2score`, and `bias` with uniform variable weighting:

```yaml
# No metrics section — equivalent to:
metrics:
  type: combined
  args:
    metrics: {rmse: {}, r2score: {}, bias: {}}
    var_weighting: "none"
```

A typical explicit configuration:

```yaml
metrics:
  type: combined
  args:
    metrics:
      rmse: {}
      r2score: {}
      bias: {}
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
    use_latitude_weights: true
    latitude_weights: '/path/to/static.zarr'
```

## Required postblocks

Metrics read the same `full_data_dict` as `BaseLoss`, so they need the same
`per_step` postblock chain — a `reconstruct` + inverse-scaler pair for the
prediction, and a **target twin** producing `y_target_processed`. See
[Required postblocks](Losses.md#required-postblocks) for the full YAML; if you have
already configured `BaseLoss`, metrics need nothing further.

Metrics do **not** require `detach: false` — they never backpropagate — but there is
no harm in it, and you need it anyway if `BaseLoss` is active.

## Built-in metrics

| Key | Class | Score | `scale_power` |
|---|---|---|---|
| `rmse` | `RMSEMetric` | Root mean squared error | 1 |
| `mse` | `MSEMetric` | Mean squared error | 2 |
| `mae` | `MAEMetric` | Mean absolute error | 1 |
| `bias` | `BiasMetric` | Signed mean bias (may be negative) | 1 |
| `r2score` | `R2ScoreMetric` | Coefficient of determination | 0 |
| `log_variance_ratio` | `LogVarianceRatioMetric` | log₁₀(var_pred / var_target) | 0 |
| `acc` | `AnomalyCorrelationCoefficientMetric` | Anomaly correlation coefficient | 0 |
| `activity` | `ForecastActivityMetric` | Forecast activity (SDAF) | 1 |

`scale_power` is explained under [Variable weighting](#variable-weighting).

Two of these are worth a note because they diagnose a failure mode RMSE hides:

- **`log_variance_ratio`** compares the forecast's spatial variance to the target's.
  A negative value means the forecast is **smoother** than reality — the
  characteristic signature of ML emulators and of ensemble averaging. Because
  variance is translation-invariant, this metric is insensitive to bias, so pair it
  with `bias` or `rmse` for a complete picture.
- **`activity`** (SDAF) measures the same tendency against a climatology rather than
  against the field's own mean, which is more informative when you have one.

## Output format

Each metric returns its per-variable scores plus a combined aggregate, namespaced by
the metric's key:

```python
{
  "rmse/ERA5/prognostic/3d/temperature": 1.83,
  "rmse/ERA5/prognostic/2d/SP": 142.6,
  "rmse": 0.97,            # weighted aggregate across variables
  "r2score/ERA5/...": 0.94,
  "r2score": 0.91,
}
```

A `combined` metric returns the union of its children's outputs. The trainer
all-reduces every value across ranks and writes it to TensorBoard and
`training_log.csv` as `train_<key>` / `valid_<key>`.

The progress bar and CSV columns follow whatever you configure: `BaseTrainer`
derives its display metrics from `metrics.args.metrics` (or from `metrics.type` for
a single metric), falling back to `rmse` / `r2score` / `bias` when the section is
omitted.

## Variable weighting

Per-variable scores are always reported in the variable's own physical units. The
`var_weighting` option controls only how they are **combined** into the aggregate.

| Mode | Behavior |
|---|---|
| `inverse_variance` *(default)* | Weight by `1 / σ^scale_power`, with σ from the fitted bridgescaler |
| `manual` | Weights come from `variable_weights` alone |
| `none` | Uniform combination |

```{note}
Unlike `BaseLoss`, there is no `learnable` mode — metrics are not optimized. Passing
it raises with an explanatory error.
```

### `scale_power`

Physical-unit variables differ by orders of magnitude, so an unweighted aggregate is
dominated by the largest-scale variable. But the *right* weight depends on the
metric's order: MSE is quadratic in σ, RMSE and MAE are linear, and R² and ACC are
already dimensionless.

Each metric therefore declares a `scale_power` — the power of σ its score carries —
and `inverse_variance` weights by `1 / σ^scale_power` so the aggregate is
dimensionless whatever the metric:

- **2** — `mse`. `1/σ²` recovers the normalized-space score exactly, matching `BaseLoss`.
- **1** — `rmse`, `mae`, `bias`, `activity`.
- **0** — `r2score`, `log_variance_ratio`, `acc`. These never read the scaler at all,
  so `scaler_path` is not required for them.

You do not configure `scale_power`; it is a property of the metric. It matters when
writing a custom metric — see [below](#adding-a-custom-metric).

`variable_weights` multipliers apply on top of every mode, defaulting to 1.0:

```yaml
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
    variable_weights:
      ERA5/diagnostic/2d/total_precipitation: 3.0
```

A variable with no entry in the scaler falls back to weight 1.0 with a logged
warning, which is worth watching for — a lone 1.0 among `1/σ` weights effectively
drops that variable from the aggregate.

## Full option reference

Options are shared across every child of a `combined` metric, and may be overridden
per metric in its own `args`.

| Option | Default | Meaning |
|---|---|---|
| `metrics` | — | `{metric_key: per_metric_args}`. Required for `type: combined`. |
| `var_weighting` | `"inverse_variance"` | `inverse_variance`, `manual`, or `none`. |
| `scaler_path` | — | Fitted bridgescaler JSON. Required for `inverse_variance` unless every metric has `scale_power == 0`. |
| `variable_weights` | `{}` | Manual multipliers per `var_key`, all modes. |
| `normalize_weights` | `true` | Rescale combination weights to mean 1. |
| `include_computed_diagnostics` | `true` | Score postblock-computed diagnostics. |
| `use_latitude_weights` | `false` | Apply cos(lat) spatial weighting per variable. |
| `latitude_weights` | — | Path to a dataset with a `latitude` coordinate. |

Per-metric overrides use the same nesting as the metric list:

```yaml
metrics:
  type: combined
  args:
    metrics:
      rmse: {}
      log_variance_ratio:
        eps: 1.0e-6          # per-metric argument
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
```

### A single metric

`type` accepts any registry key, not just `combined`:

```yaml
metrics:
  type: rmse
  args:
    var_weighting: "none"
```

`metric_name` defaults to the type, so the output keys are `rmse/<var_key>` and
`rmse` as usual.

## Anomaly metrics: ACC and forecast activity

`acc` and `activity` follow Bonavita & Geer (2026), *Forecast verification using
information and noise*, QJRMS 152, e70109
([doi:10.1002/qj.70109](https://doi.org/10.1002/qj.70109)). Both subtract a
climatology to form anomalies, remove the area-weighted mean (debiasing, per
Appendix A), and then score:

```
d_f = x_f − x_c − mean_w(x_f − x_c)      debiased forecast anomaly
d_t = x_t − x_c − mean_w(x_t − x_c)      debiased truth anomaly

SDAF = sqrt(mean_w(d_f²))                          forecast activity
ACC  = mean_w(d_f · d_t) / (SDAF · SDAV)           anomaly correlation
```

ACC is bounded by ±1 and is insensitive to both bias and forecast activity; SDAF
quantifies how far the forecast departs from climatology, so an unrealistically
smooth forecast shows reduced SDAF.

### Supplying a climatology

Three sources, in priority order:

**1. A file** — `climatology_path` pointing at any xarray-readable dataset
(netCDF or Zarr). Variables are looked up by the **short name** of each `var_key`,
i.e. the last path component: a `var_key` of
`ERA5/prognostic/3d/temperature` matches a data variable named `temperature`.

```yaml
metrics:
  type: combined
  args:
    metrics:
      rmse: {}
      acc:
        climatology_path: /path/to/climatology.nc
      activity:
        climatology_path: /path/to/climatology.nc
    var_weighting: "none"
    use_latitude_weights: true
    latitude_weights: /path/to/static.zarr
```

Fields must be shaped `(lat, lon)` for 2-D variables and `(level, lat, lon)` for
3-D, with the level count matching your data config. A mismatch raises with the
expected and actual counts rather than silently broadcasting.

**2. Validation data (default)** — with no `climatology_path`, the climatology is
accumulated as a running mean of the `y_target_processed` fields seen during
validation. On the first batch it equals that batch's target mean and converges
toward the full validation mean thereafter.

```{warning}
The online climatology is an approximation, and early-epoch ACC values are computed
against a climatology built from very few batches. For results you intend to
compare across runs or report, supply a climatology file.
```

**3. A dict** — pass `climatology={var_key: tensor}` programmatically; it overrides
both of the above.

## Adding a custom metric

Custom metrics register through the same mechanism as models, datasets, losses, and
postblocks.

### 1. Subclass and register

Most metrics only need `compute_variable`, which returns the elementwise error
tensor. The base class applies latitude weighting, takes the spatial mean, and
calls `reduce`:

```python
import torch

from credit.metrics import register_metric
from credit.metrics.base import BaseVariableMetric


@register_metric("huber_metric")
class HuberMetric(BaseVariableMetric):
    """Huber error per variable."""

    scale_power = 1  # linear in sigma, like MAE

    def __init__(self, *args, delta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = float(delta)

    def compute_variable(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.huber_loss(
            pred, target, reduction="none", delta=self.delta
        )
```

Set `scale_power` to match your metric's order in σ, or `inverse_variance`
weighting will misweight it. Override `reduce` if the per-variable scalar needs
finalizing after the spatial mean — `RMSEMetric` uses it to take the square root.

If your metric needs the whole target tensor (as R² does, for the target mean) or
the variable key (as the anomaly metrics do, to look up a climatology), override
`_score_variable(pred, target, var_key)` instead of
`compute_variable` / `reduce`.

### 2. Point the config at it

```yaml
custom_objects:
  HuberMetric:
    object_type: metric
    module_path: mypackage.metrics

metrics:
  type: combined
  args:
    metrics:
      rmse: {}
      huber_metric:
        delta: 1.0
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
```

The registry key is whatever you passed to `@register_metric`; `module_path` is the
importable module holding the decorated class.

## Summary

- The `metrics` section is optional; omitting it gives `rmse` / `r2score` / `bias`
  with uniform weighting.
- Metrics need the same target-twin postblocks as `BaseLoss`.
- Per-variable scores are always in physical units; `var_weighting` affects only the
  aggregate.
- `inverse_variance` weights by `1 / σ^scale_power`, so the aggregate stays
  dimensionless whatever the metric's order — dimensionless metrics ignore the
  scaler entirely.
- `acc` and `activity` need a climatology; prefer a file over the online running
  mean for reportable numbers.
- Custom metrics subclass `BaseVariableMetric`, declare `scale_power`, and register
  with `@register_metric`.
