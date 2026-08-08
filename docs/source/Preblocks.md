# Preblocks

Preblocks perform pre-processing operations on loaded data to transform
the data into a format appropriate for model training and inference.

## General Preblock Structure

Every preblock inherits from `credit.preblock.base.BasePreblock` and implements
a single `forward(batch: dict) -> dict` method. Preblocks are chained together
and run sequentially on a nested batch dict of the form:

```
batch[data_type][source][var_key] = torch.Tensor
```

where:

- `data_type` is one of `"input"` or `"target"` (never `"metadata"`).
- `source` is the data source prefix (e.g. `"era5"`, extracted from the leading
  segment of the variable key).
- `var_key` is the full variable path, e.g. `"era5/prognostic/3d/Q"`.

Preblocks are configured under the top-level `preblocks:` key, which is split
into two phases:

- `ic_only:` blocks run once at `t=0` on the raw batch (e.g. static regridding
  or initial-condition interpolation).
- `per_step:` blocks run every rollout step (e.g. log transform, concat).

Both phases are built from the same config format:

```yaml
preblocks:
  ic_only:
    <block_name>:
      type: <preblock_type>
      args:
        <key>: <value>
  per_step:
    <block_name>:
      type: <preblock_type>
      args:
        <key>: <value>
```

Most preblocks accept a `data_types` argument that scopes which batch splits
they process (default: `["input", "target"]`). A data type absent from the
batch — e.g. no `"target"` during inference — is skipped silently.

Environment variables in path arguments (e.g. `$SCRATCH`) are expanded via
`os.path.expandvars` for the preblock file paths `regrid.weight_file`,
`bridgescaler_transform.scaler_path`, and `rename.mapping_file`, as well as for
the top-level `save_loc` and `predict.save_forecast` keys (`credit check` also
expands them when verifying that paths exist). Not every path argument is
expanded, however — `era5_normalizer`'s `mean_path`/`std_path` are used
verbatim — so when in doubt, use absolute paths.

## Preblock Types

### `fill_values` (FillValues)

**AutoAPI:** {py:obj}`credit.preblock.fill_values.FillValues`

Replace values matching a set of rules with constant fill values for selected
variables. Walks the nested batch dict and applies each rule as a
search-and-replace pass. All masks are computed on the **original** tensor
before any replacement, so earlier rules do not affect later rules' matches
(simultaneous semantics). If two rules match the same position, the last rule
in the list wins.

Each rule is a dict with:

- `search`: the string `"nan"` (matches NaN) or a float (used with `op`).
- `op`: comparison operator — `"=="`, `"!="`, `"<"`, `"<="`, `">"`, `">=""`
  (default `"=="`); ignored when `search` is `"nan"`.
- `fill`: the replacement value.

Numeric ops never match NaN positions — use `search: "nan"` to explicitly
target NaN values.

```yaml
type: "fill_values"
args:
  rules:
    - search: nan        # NaN       → -1.0
      fill: -1.0
    - search: 0.0        # == 0.0    → 1.0e-4
      op: "=="
      fill: 1.0e-4
    - search: 0.0        # < 0.0     → 0.0  (clamp negatives)
      op: "<"
      fill: 0.0
  variables:             # optional — defaults to all variables
    - "era5/prognostic/3d/Q"
```

### `regrid` (Regridder)

**AutoAPI:** {py:obj}`credit.preblock.regrid.Regridder`

Regridding preblock using a sparse weight matrix from an ESMF-format weights
file. Applies conservative (or bilinear, depending on the weight file)
regridding to selected variables. The sparse weight matrix is assembled once
and cached per device; subsequent calls on the same device are free of data
movement.

```yaml
type: "regrid"
args:
  weight_file: "$SCRATCH/weights/era5_to_1deg.nc"
  variables:
    - "era5/prognostic/3d/T"
  reshape_to_xy: true
```

### `era5_normalizer` (ERA5Normalizer)

**AutoAPI:** {py:obj}`credit.preblock.norm.ERA5Normalizer`

**Which normalization route?** CREDIT has two: `era5_normalizer` uses
gen1-style pre-computed mean/std NetCDF files, while `bridgescaler_transform`
(below) uses a scaler JSON fitted by `credit preprocess`. Prefer
`bridgescaler_transform` for new gen2 work; `era5_normalizer` mainly exists to
reuse existing gen1 statistics files.

Normalizes per-variable ERA5 tensors using pre-computed mean/std NetCDF files.
Normalization `(x - mean) / std` is applied per variable; variables not found
in the statistics file are passed through unchanged. The mean/std may be scalar
(2D variables) or 1-D per-level vectors.

```yaml
type: "era5_normalizer"
args:
  mean_path: /path/to/mean.nc
  std_path:  /path/to/std.nc
  levels: [60, 90, 120, 137]   # optional — 1-indexed level selection
```

### `bridgescaler_transform` (BridgeScalerTransform)

**AutoAPI:** {py:obj}`credit.preblock.scaler.BridgeScalerTransform`

`bridgescaler` is the external [bridgescaler](https://github.com/NCAR/bridgescaler)
package — NCAR's distributed scaler library, installed as a CREDIT dependency —
whose fitted scalers are saved to and loaded from JSON via `credit preprocess`.

Scaling preblock using a dictionary of `bridgescaler` scalers to fit and
transform CREDIT state dictionaries. Applies per-variable z-score scaling (or
its inverse) to tensors in the nested batch dict. The scaler dict is produced
by running `credit preprocess`. `variables` accepts full variable keys,
partial paths (e.g. `"era5/prognostic"` expands to all variables under that
prefix), or an empty list (expands to all variables).

```yaml
# Scale specific variables in both input and target
type: "bridgescaler_transform"
args:
  scaler_path: "/path/to/scaler.json"
  variables:
    - "era5/prognostic/3d/T"
    - "era5/prognostic/3d/U"
  method: "transform"

# Scale all variables
type: "bridgescaler_transform"
args:
  scaler_path: "/path/to/scaler.json"
  variables: []
  method: "transform"

# Multi-step rollout: scale only the target
type: "bridgescaler_transform"
args:
  scaler_path: "/path/to/scaler.json"
  variables: []
  method: "transform"
  data_types: ["target"]
```

### `log_transform` (LogTransform)

**AutoAPI:** {py:obj}`credit.preblock.log.LogTransform`

Applies a log transformation with an `eps` offset to specified variables:

`y = log_base(x + eps) - log_base(eps)`

so that `y = 0` when `x = 0`, regardless of `eps`. Use `ExpTransform` in the
postblock to invert this with matching `base` and `eps`. Input values should
satisfy `x >= -eps`; values below this produce NaN silently.

```yaml
type: "log_transform"
args:
  variables:
    - "era5/prognostic/3d/Q"
  base: "e"       # optional — "e", "2", or "10" (default "e")
  eps: 1.0e-8     # optional (default 1e-8)
```

### `sqrt_transform` (SqrtTransform)

**AutoAPI:** {py:obj}`credit.preblock.sqrt.SqrtTransform`

Applies a square-root transformation `y = sqrt(x)` to specified variables. Use
`SquareTransform` in the postblock to invert this (`x = y^2`). Input values
must be non-negative; negative values produce NaN silently.

```yaml
type: "sqrt_transform"
args:
  variables:
    - "era5/prognostic/3d/Q"
```

### `concat` (ConcatToTensor)

**AutoAPI:** {py:obj}`credit.preblock.concat.ConcatToTensor`

End-of-chain preblock that collapses a nested batch dict of tensors into a
single input tensor (and optionally a target tensor), concatenating along the
channel dimension. Input tensors are sorted by a canonical channel key derived
from the `{source}/{field_type}/{dim}/{varname}` structure so the channel
order is deterministic regardless of insertion order. In addition to the
tensors, channel maps are attached to metadata mapping each variable to its
slice in the concatenated tensor.

```yaml
type: "concat"
args:
  to_device: true   # set false to skip .to(device) in apply_preblocks
```

### `hybrid_level_interp` (HybridLevelInterpPre)

**AutoAPI:** {py:obj}`credit.preblock.hybrid_interp.HybridLevelInterpPre`

Interpolates 3D variables between hybrid sigma-pressure level sets. Variables
are interpolated column-by-column, linearly in log(pressure), with constant
extrapolation outside the source pressure range, parallelized with
`torch.vmap`. The primary use case is inference with a model trained on one
vertical grid but initialized from another (e.g. an ERA5-trained model driven
by GFS initial conditions): run this in the `ic_only` phase so the IC lands on
the model's levels before normalization and concat.

```yaml
type: "hybrid_level_interp"
args:
  variables:
    - "GFS/prognostic/3d/temperature"
    - "GFS/prognostic/3d/specific_humidity"
  surface_pressure_var: "GFS/prognostic/2d/surface_pressure"
  source_level_info_file: "/path/to/gfs_ctrl.nc"
  source_a_var: "vcoord"
  source_b_var: "vcoord"
  dest_level_info_file: "ERA5_Lev_Info.nc"
  data_types: ["input"]
```

### `semilagrangian_advection` (SemiLagrangianAdvectionPre)

**AutoAPI:** {py:obj}`credit.preblock.advect.SemiLagrangianAdvectionPre`

Preblock that performs one semi-Lagrangian 3D tracer advection step. For each
requested data type it reads the winds and surface pressure, derives the
pressure vertical velocity from mass continuity (or reads a precomputed
`omega_var`), traces a back-trajectory of length `timestep_seconds`, and
overwrites each configured tracer with its value interpolated at the
trajectory departure point. The primary use case is correcting or spinning up
tracer initial conditions with an explicit advection step before they reach
the model — run this in the `ic_only` phase using the same winds/tracers
convention as the postblock so the same config args work in either phase.

```yaml
type: "semilagrangian_advection"
args:
  tracer_vars:
    - "ERA5/prognostic/3d/specific_humidity"
  u_var: "ERA5/prognostic/3d/u_component_of_wind"
  v_var: "ERA5/prognostic/3d/v_component_of_wind"
  surface_pressure_var: "ERA5/prognostic/2d/surface_pressure"
  timestep_seconds: 21600.0
  data_types: ["input"]
```

## Example: processing the ARCOERA5Dataset

The config below trains on ARCO-ERA5 with a full preblock pipeline. The
`ic_only` phase runs once at `t=0`: `fill_values` sanitizes the raw specific
humidity, then `regrid` remaps the data onto a 1-degree target grid. The
`per_step` phase runs every rollout step: `log_transform` stabilizes the
humidity distribution, `era5_normalizer` standardizes each variable with
pre-computed statistics, and finally `concat` collapses the nested batch into
a single tensor ready for the model.

```yaml
preblocks:
  ic_only:
    fill_nan_q:
      type: "fill_values"
      args:
        rules:
          - search: nan
            fill: 0.0
        variables:
          - "era5/prognostic/3d/specific_humidity"
    regrid_to_1deg:
      type: "regrid"
      args:
        weight_file: "$SCRATCH/weights/era5_to_1deg.nc"
        variables:
          - "era5/prognostic/3d/temperature"
          - "era5/prognostic/3d/specific_humidity"
          - "era5/prognostic/2d/2t"
        reshape_to_xy: true

  per_step:
    log_transform_q:
      type: "log_transform"
      args:
        variables:
          - "era5/prognostic/3d/specific_humidity"
        base: "e"
        eps: 1.0e-8
    normalize:
      type: "era5_normalizer"
      args:
        mean_path: /path/to/era5_mean.nc
        std_path:  /path/to/era5_std.nc
        levels: [60, 90, 120, 137]
    concat:
      type: "concat"
      args:
        to_device: true
```
