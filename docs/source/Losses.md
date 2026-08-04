# Working with Loss Functions

CREDIT has two loss configuration formats, selected by whether the `loss` section
has a `type` key:

| Format | Section shape | Scored on | Guide |
|---|---|---|---|
| **Gen 2** | `{type, args}` | The postblock output, per variable, in **physical units** | This page |
| **Gen 1** | Flat keys (`training_loss`, `use_latitude_weights`, ...) | The flat, normalized `y_pred` tensor | [Gen 1 Losses](Losses_gen1.md) |

`load_loss` dispatches on `type` when present and falls back to the flat keys
otherwise, so existing Gen 1 configs keep working unchanged.

This page covers the Gen 2 format. For the companion verification metrics
framework, which mirrors the structure described here, see
[Verification Metrics](Metrics.md).

## Why a per-variable loss

A Gen 1 loss sees one flat tensor of normalized model output. That is simple, but
it means the loss cannot distinguish variables, cannot see the effect of physics
fixers applied in the postblocks, and cannot be expressed in units anyone reads off
a chart.

`BaseLoss` scores the *postblock output* instead. The Gen 2 trainer's `per_step`
postblock chain turns raw model output into
`full_data_dict["y_processed"]` — a nested `{source: {var_key: tensor}}` dict in
physical units — and `BaseLoss` compares it, variable by variable, against a
**target twin** `y_target_processed` built by running the same chain on the flat
target `y`. Per-variable scores are then combined into the scalar the optimizer sees.

Two consequences worth understanding before you configure it:

- Gradients flow **through the postblocks**, so conservation fixers and other
  differentiable corrections are part of what the model is trained against.
- Physical-unit variables have wildly different variances (surface pressure
  ~10⁵ Pa vs. specific humidity ~10⁻³ kg/kg), so an unweighted physical-space MSE
  is dominated by the largest-scale variables. Weighting is not optional in
  practice — see [Variable weighting](#variable-weighting).

## Required postblocks

`BaseLoss` will not run without the target twin. Add these to `postblocks.per_step`:

```yaml
postblocks:
  per_step:
    # detach: false is REQUIRED for training — otherwise y_processed carries no
    # gradient and BaseLoss raises.
    reconstruct:
      type: reconstruct
      args:
        detach: false
    scaler:
      type: bridgescaler_transform
      args:
        scaler_path: '/path/to/scaler.json'
        variables: []                 # empty = all variables
        spatial_variables: ['ERA5/prognostic/2d/SP']
        method: inverse_transform

    # ---- Target twin: the same chain applied to the flat target `y` ----
    reconstruct_target:
      type: reconstruct
      args:
        in_key: 'y'
        out_key: 'y_target_processed'
    scaler_target:
      type: bridgescaler_transform
      args:
        scaler_path: '/path/to/scaler.json'
        variables: []
        spatial_variables: ['ERA5/prognostic/2d/SP']
        method: inverse_transform
        key: 'y_target_processed'
```

Every transform you apply to the prediction must also be applied to the twin, with
`key: 'y_target_processed'`, or the two sides will not be comparable. If your chain
includes an `exp_transform` for precipitation, for example, it needs a mirrored
`log_trans_target` entry.

```{tip}
If you forget a piece of this, `BaseLoss` raises with the exact YAML to add rather
than failing silently. The three errors to recognize: a missing `y_processed` key
means no `reconstruct` postblock; a missing `y_target_processed` key means no target
twin; and "y_processed carries no gradient" means `detach: false` is missing.
```

## The `loss` section

```yaml
loss:
  type: base
  args:
    training_loss: "mse"
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
```

That is the minimum. The full option set:

| Option | Default | Meaning |
|---|---|---|
| `training_loss` | `"mse"` | Univariate loss applied per variable. Must be elementwise — see below. |
| `base_loss_parameters` | `{}` | Constructor kwargs for it. `reduction` is forced to `"none"`. |
| `validation_loss` | *(training loss)* | Optional different univariate loss for validation. |
| `validation_loss_parameters` | `{}` | Constructor kwargs for the validation loss. |
| `base_loss_overrides` | `{}` | Per-variable univariate loss overrides. |
| `var_weighting` | `"inverse_variance"` | `inverse_variance`, `manual`, `learnable`, or `none`. |
| `scaler_path` | — | Fitted bridgescaler JSON. Required for `inverse_variance` and `learnable`. |
| `variable_weights` | `{}` | Manual multipliers per `var_key`, applied on top of **every** mode. |
| `normalize_weights` | `true` | Rescale the combination weights to mean 1. |
| `include_computed_diagnostics` | `true` | Score postblock-computed diagnostics (see below). |
| `use_latitude_weights` | `false` | Apply cos(lat) spatial weighting per variable. |
| `latitude_weights` | — | Path to a dataset with a `latitude` coordinate. Required when the above is true. |

### Choosing the univariate loss

`training_loss` must return an **elementwise** tensor the same shape as the
variable. These registry entries qualify:

`mse`, `mae`, `msle`, `huber`, `logcosh`, `xtanh`, `xsigmoid`

The CRPS-family losses (`KCRPS`, `almost-fair-crps`, `ring-crps`) are rejected at
construction with an explanatory error — they score an ensemble, not a single
field. `spectral`, `power`, and `covmse` reduce to a scalar (or a non-matching
shape) internally and will fail the elementwise shape check at the first forward
pass.

### Which variables are scored

Every variable in the data target layout — prognostic **and** diagnostic, from
`conf["data"]["source"]` — is always scored.

Variables that appear in `y_processed` only because a postblock computed them from
prognostic variables (e.g. `mslp_diagnostic`, `geopotential_diagnostic`) are
*computed diagnostics*. They are scored only when
`include_computed_diagnostics: true`, and then require a matching entry in the
target twin — i.e. the same compute postblock applied with
`key: 'y_target_processed'`. If you want them diagnosed but not trained against,
set the flag to `false`.

## Variable weighting

This is the setting that most affects results, because physical-unit variables span
many orders of magnitude.

### `inverse_variance` (default)

Weight each variable by `1 / σ²`, with `σ²` read from the fitted bridgescaler at
`scaler_path`. For MSE this approximates training in normalized space, which is
usually what you want as a starting point.

Variances are read from `DStandardScalerTensor.var_x_` directly, estimated from
t-digest centroid moments for `DQuantileScalerTensor`, or taken from `sd_²` for the
numpy `DeepStandardScaler`. A variable with no entry in the scaler falls back to
weight 1.0 with a logged warning — worth watching for, since a silent 1.0 among
1/σ² weights effectively drops that variable from the objective.

```yaml
loss:
  type: base
  args:
    training_loss: "mse"
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
```

### `manual`

Weights come from `variable_weights` alone. Anything unlisted gets 1.0 with a
warning.

```yaml
loss:
  type: base
  args:
    training_loss: "mae"
    var_weighting: "manual"
    variable_weights:
      ERA5/prognostic/3d/temperature: 2.0
      ERA5/prognostic/2d/SP: 0.5
```

### `none`

Uniform combination. Only sensible when your variables are already on comparable
scales — otherwise the largest-magnitude variable dominates.

### `learnable`

Kendall–Gal uncertainty weighting: a trainable per-variable `log σ²` with the
objective

```
L = mean_v( m_v · exp(−s_v) · L_v + 0.5 · s_v )
```

where `s_v` is the learned log-variance and `m_v` the `variable_weights`
multiplier. Parameters are initialized from the scaler statistics when available.

```yaml
loss:
  type: base
  args:
    training_loss: "mse"
    var_weighting: "learnable"
    scaler_path: '/path/to/scaler.json'
```

```{warning}
Three constraints apply to `learnable`:

- It requires a channel schema, so the parameters exist at init.
- It does not support computed diagnostics — set
  `include_computed_diagnostics: false`.
- **The learned parameters are not currently checkpointed.** They re-initialize
  when you resume from a checkpoint. `train_gen2` does add them to the optimizer
  automatically, so training itself works; only resumption is affected. Tracked
  in [issue #473](https://github.com/NCAR/miles-credit/issues/473).
```

### `variable_weights` on top

`variable_weights` multipliers apply on top of *every* mode, defaulting to 1.0.
Use them to nudge a variable's importance without abandoning `inverse_variance`:

```yaml
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
    variable_weights:
      ERA5/diagnostic/2d/total_precipitation: 3.0
```

## Latitude weighting

Enable `cos(lat)` spatial weighting so that grid cells contribute in proportion to
the area they represent — important on a regular lat/lon grid, where polar cells are
far smaller than tropical ones.

```yaml
    use_latitude_weights: true
    latitude_weights: '/path/to/static.zarr'
```

Weights are normalized to mean 1 and applied to the elementwise tensor before the
spatial mean, per variable. They are sharded automatically under domain-parallel
training.

## Per-variable loss overrides

Use a different univariate loss for specific variables — useful for
heavy-tailed fields such as precipitation, where MAE is more stable than MSE:

```yaml
loss:
  type: base
  args:
    training_loss: "mse"
    base_loss_overrides:
      ERA5/diagnostic/2d/total_precipitation:
        loss: "mae"
        parameters: {}
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
```

```{note}
An override changes the units of that variable's score. Under `inverse_variance`
the combination weight is still `1/σ²`, which is matched to a quadratic loss, so
an MAE override is weighted as though it were quadratic. With one or two overrides
this is a small distortion; if you are overriding most variables, prefer `manual`
weights you control directly.
```

## Per-variable logging

`BaseLoss` exposes detached per-variable scores through `last_var_losses` after each
forward pass, and `TrainerERA5Gen2` logs them automatically — all-reduced across
ranks — as `train_loss_var/<var_key>` and `valid_loss_var/<var_key>`. They appear
in TensorBoard and in `training_log.csv`, which makes it straightforward to see
which variable is driving a plateau.

## Using a plain registry loss

`type` accepts any registry loss name, not just `base`. This gives you the Gen 2
section shape with a conventional flat-tensor loss:

```yaml
loss:
  type: mse
  args: {}
```

The class is constructed directly from `args`, with `reduction` supplied if its
constructor accepts one. Note that this path does **not** apply the latitude- or
variable-weighting wrappers — those belong to the Gen 1 flat-key format.

## Adding a custom loss

Custom losses register through the same mechanism as models, datasets, and
postblocks — no edit to the CREDIT source tree required.

### 1. Write and register the class

```python
import torch
import torch.nn as nn

from credit.losses import register_loss


@register_loss("my_loss")
class MyLoss(nn.Module):
    """Elementwise loss, so it can serve as a BaseLoss univariate loss."""

    def __init__(self, reduction: str = "none", scale: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.scale = scale

    def forward(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        loss = self.scale * (prediction - target).abs()
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```

Two things to match: the call signature is `(target, prediction)` — CREDIT's
convention throughout — and returning an unreduced tensor under
`reduction="none"` is what lets `BaseLoss` use it per variable. `BaseLoss` forces
`reduction="none"` on any loss whose constructor accepts it.

### 2. Point the config at it

```yaml
custom_objects:
  MyLoss:
    object_type: loss
    module_path: mypackage.losses

loss:
  type: base
  args:
    training_loss: "MyLoss"
    base_loss_parameters:
      scale: 2.0
    var_weighting: "inverse_variance"
    scaler_path: '/path/to/scaler.json'
```

The registry key is whatever you passed to `@register_loss`; `module_path` is the
importable module holding the decorated class.

## Complete example

A working `loss` section with most options exercised:

```yaml
loss:
  type: base
  args:
    training_loss: "mse"
    base_loss_parameters: {}

    # Report MAE during validation while training on MSE.
    validation_loss: "mae"
    validation_loss_parameters: {}

    # Precipitation is heavy-tailed; score it with MAE.
    base_loss_overrides:
      ERA5/diagnostic/2d/total_precipitation:
        loss: "mae"
        parameters: {}

    var_weighting: "inverse_variance"
    scaler_path: '/glade/derecho/scratch/$USER/CREDIT_runs/my_run/scaler.json'
    variable_weights: {}
    normalize_weights: true

    include_computed_diagnostics: true

    use_latitude_weights: true
    latitude_weights: '/path/to/static.zarr'
```

## Summary

- Gen 2 uses a `{type, args}` `loss` section; `type: base` selects `BaseLoss`.
- `BaseLoss` scores the postblock output per variable in physical units, so the
  `per_step` chain **must** build a `y_target_processed` twin and use
  `detach: false` on the prediction `reconstruct`.
- `var_weighting` matters: `inverse_variance` is the sensible default because
  physical-unit variables differ by orders of magnitude.
- The univariate loss must be elementwise; CRPS-family losses are rejected.
- Per-variable scores are logged automatically as `train_loss_var/<var_key>`.
- `learnable` weighting works but does not survive a checkpoint resume yet.
