# Postblocks
Postblocks perform data transforms and diagnostic calculations after the model
has performed its forward pass. For WXFormer and current models, the first
step is to run the Reconstruct postblock to re-create the state dictionary.
Next, a bridgescaler transform block performs an inverse transform to return
data to physical instead of normalized values. Then, some fields that have been
log or square root transformed may need to have their inverse operations run.

Once all needed variables are back in physical ranges of values, diagnostic 
postblocks can be run to calculate derived fields like geopotential, perform
interpolation to pressure levels, enforce conservation laws, or run storm
tracking algorithms. If the algorithms are implemented in differentiable PyTorch,
the outputs could potentially be included in the loss function to enable physics
guidance during training.

## General Postblock Structure

Every postblock inherits from `credit.postblock.base.BasePostblock` and
implements a single `forward(batch_dict: dict) -> dict` method. Postblocks are
chained together and run in order. Unlike preblocks (which pass a nested
`batch[data_type][source][var_key]` dict), postblocks operate on the flat
`batch_dict` assembled around the model's forward pass, whose relevant keys are:

- `y_pred` — the model's flat output tensor `(B, C, H, W)`.
- `y_processed` — the nested `{source: {var_key: tensor}}` prediction dict that
  `reconstruct` writes and every downstream block reads/updates (the default
  `key` for most blocks).
- `y` / `y_target_processed` — the flat target tensor and its reconstructed
  twin, used when `BaseLoss` scores predictions against targets in physical
  units (the "target twin" chain).
- `ic_raw` / `x_physical` — nested initial-condition and physical-input dicts
  that diagnostic and conservation blocks read for static fields and t0 state
  (the `static_source_key` / `input_source_key` defaults).

Postblocks are configured under the top-level `postblocks:` key, split into two
phases:

- `per_step:` blocks run after every forward pass in the rollout loop (the
  common case — reconstruction, inverse transforms, diagnostics, fixers).
- `post_rollout:` blocks run once after all rollout steps complete.

Both phases share the same config format:

```yaml
postblocks:
  per_step:
    <block_name>:
      type: <postblock_type>
      args:
        <key>: <value>
  post_rollout:
    <block_name>:
      type: <postblock_type>
      args:
        <key>: <value>
```

Most transform/diagnostic blocks accept a `key` argument selecting which
`batch_dict` entry to operate on (default `"y_processed"`). To also transform
the target side for a physical-units loss, add a second copy of the block with
`key: "y_target_processed"` (the "target twin" — see the example at the end).

## Reconstruction

### `reconstruct` (Reconstruct)

**AutoAPI:** {py:obj}`credit.postblock.reconstruct.Reconstruct`

Splits the model's flat output tensor into a nested variable dict. By default it
splits `batch_dict["y_pred"]` into `batch_dict["y_processed"]`, reading channel
slices from the `_channel_map` metadata that `ConcatToTensor` built (prognostic
+ diagnostic variables) and unflattening each slice from
`(B, n_levels * n_time, H, W)` back to `(B, n_levels, n_time, H, W)`. This
block is effectively **required as the first postblock in every gen2 chain** —
it is the counterpart of the mandatory `concat` preblock, and every downstream
postblock (and `BaseLoss`) operates on the `y_processed` dict it produces. Set `detach: false` when
downstream fixers or `BaseLoss` must backpropagate through the reconstructed
dict; the default `true` severs the autograd graph.

```yaml
# Prediction side (must allow gradients for a BaseLoss on y_processed)
type: "reconstruct"
args:
  detach: false

# Target twin: split the flat target y into its own nested dict
type: "reconstruct"
args:
  in_key: "y"
  out_key: "y_target_processed"
```

### `flatten_to_tensor` (FlattenToTensor)

**AutoAPI:** {py:obj}`credit.postblock.reconstruct.FlattenToTensor`

Inverse of `reconstruct`: concatenates the per-variable tensors in
`y_processed` back into a single flat `(B, C, H, W)` tensor in the original
channel order. Conservation fixers operate on `y_processed` in physical units,
so when a `scaler_path` is given this block forward-scales a copy before
flattening, yielding a normalized `y_pred` for the loss while leaving the
physical `y_processed` untouched for autoregressive rollout assembly. Omit
`scaler_path` to flatten as-is with no scaling.

```yaml
type: "flatten_to_tensor"
args:
  scaler_path: "/path/to/scaler.json"
  variables: []          # variables the scaler should transform
  method: "transform"    # physical -> normalized
```

## Inverse Transforms

### `bridgescaler_transform` (BridgeScalerTransform)

**AutoAPI:** {py:obj}`credit.postblock.scaler.BridgeScalerTransform`

Applies per-variable scaling to the nested output dict using a fitted
`bridgescaler` dict — typically an `inverse_transform` right after `reconstruct`
to convert normalized model output back to physical units before the physics
blocks. It shares the same `scaler.json` produced by `credit preprocess` that
the preblock uses (the `"target"` slice is used for inverse-transforming model
output). `variables` accepts full keys, partial paths (e.g. `"era5/prognostic"`
expands to everything under that prefix), or an empty list (all variables).

```yaml
# Inverse-transform every variable in y_processed
type: "bridgescaler_transform"
args:
  scaler_path: "/path/to/scaler.json"
  variables: []
  method: "inverse_transform"

# Target twin: inverse-transform y_target_processed instead
type: "bridgescaler_transform"
args:
  scaler_path: "/path/to/scaler.json"
  variables: []
  method: "inverse_transform"
  key: "y_target_processed"
```

### `exp_transform` (ExpTransform)

**AutoAPI:** {py:obj}`credit.postblock.exp.ExpTransform`

Inverse of the `log_transform` preblock: converts log-space values back to
physical space via `x = base^(y + log_base(eps)) - eps`. The `eps` and `base`
**must** match those used in the corresponding `log_transform` preblock. Run it
after the inverse scaler so the values are in log-physical space first.

```yaml
type: "exp_transform"
args:
  variables:
    - "era5/prognostic/3d/specific_humidity"
  eps: 1.0e-8      # must match the LogTransform eps
  base: "e"        # must match the LogTransform base
```

### `square_transform` (SquareTransform)

**AutoAPI:** {py:obj}`credit.postblock.square.SquareTransform`

Inverse of the `sqrt_transform` preblock: converts sqrt-space values back to
physical space via `x = y^2`. Slightly negative inputs (from floating-point
noise) map to small positive numbers; clamping to zero is intentionally avoided
to preserve gradient flow for a differentiable loss.

```yaml
type: "square_transform"
args:
  variables:
    - "era5/prognostic/3d/specific_humidity"
```

## Diagnostics

### `geopotential_diagnostic` (GeopotentialDiagnostic)

**AutoAPI:** {py:obj}`credit.postblock.geopotential.GeopotentialDiagnostic`

Computes the 3D geopotential field from surface geopotential (PHIS), surface
pressure, temperature, and specific humidity by integrating the hypsometric
equation over the model's hybrid sigma-pressure levels. Static PHIS is read from
`batch_dict[static_source_key]` (default `"ic_raw"`); the result is written back
into the output dict under `output_name`. Chain this before
`pressure_interp_diagnostic`, which requires geopotential on model levels.

```yaml
type: "geopotential_diagnostic"
args:
  output_name: "ERA5/computed_diagnostic/3d/geopotential"
  surface_geopotential_var: "ERA5/static/2d/geopotential_at_surface"
  surface_pressure_var: "ERA5/prognostic/2d/surface_pressure"
  temperature_var: "ERA5/prognostic/3d/temperature"
  specific_humidity_var: "ERA5/prognostic/3d/specific_humidity"
  flip_vertical: true
  level_info_file: "ERA5_Lev_Info.nc"
  model_a_half_var: "a_half"
  model_b_half_var: "b_half"
  chunk_size: 1000
```

### `mslp_diagnostic` (MSLPDiagnostic)

**AutoAPI:** {py:obj}`credit.postblock.mslp.MSLPDiagnostic`

Computes mean sea level pressure from surface pressure, 2m temperature, and
surface geopotential (PHIS). Static PHIS is read from
`batch_dict[static_source_key]` (default `"ic_raw"`); the result is written back
into the output dict under `output_name`.

```yaml
type: "mslp_diagnostic"
args:
  output_name: "ARCO_ERA5/derived_diagnostic/2d/mean_sea_level_pressure"
  surface_pressure_var: "ARCO_ERA5/prognostic/2d/surface_pressure"
  temperature_var: "ARCO_ERA5/prognostic/2d/2m_temperature"
  surface_geopotential_var: "ARCO_ERA5/static/2d/geopotential_at_surface"
```

### `pressure_interp_diagnostic` (PressureInterpDiagnostic)

**AutoAPI:** {py:obj}`credit.postblock.pressure_interp.PressureInterpDiagnostic`

Interpolates model-level 3D variables to constant pressure levels, running on
the model's native (possibly reduced) level set. One output is written per
interpolated variable, named
`{source}/derived_diagnostic/{dim}/{varname}{output_suffix}` with shape
`(B, n_plev, n_time, H, W)`. `geopotential_var` must already exist in the output
dict, so chain a `geopotential_diagnostic` block before this one. Temperature
and geopotential get special extrapolation and should not be listed in
`interp_variables`.

```yaml
type: "pressure_interp_diagnostic"
args:
  pressure_levels: [250.0, 500.0, 850.0]   # hPa
  interp_variables:
    - "ARCO_ERA5/prognostic/3d/u_component_of_wind"
    - "ARCO_ERA5/prognostic/3d/v_component_of_wind"
    - "ARCO_ERA5/prognostic/3d/specific_humidity"
  temperature_var: "ARCO_ERA5/prognostic/3d/temperature"
  geopotential_var: "ARCO_ERA5/derived_diagnostic/3d/geopotential"
  surface_pressure_var: "ARCO_ERA5/prognostic/2d/surface_pressure"
  surface_geopotential_var: "ARCO_ERA5/static/2d/geopotential_at_surface"
  output_suffix: "_PRES"
  level_info_file: "ERA5_Lev_Info.nc"
```

## Vertical Interpolation

### `hybrid_level_interp` (HybridLevelInterpPost)

**AutoAPI:** {py:obj}`credit.postblock.hybrid_interp.HybridLevelInterpPost`

Interpolates 3D variables between hybrid sigma-pressure level sets, replacing
each listed variable in the output dict with its interpolated counterpart in
place. The main use case is producing output on a different vertical grid than
the model runs on (e.g. mapping model-native levels to ERA5 levels). A matching
preblock (`hybrid_level_interp`) does the same for initial conditions before the
model runs. Bare coefficient filenames resolve to `credit.metadata`; paths with
directories are used as-is.

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
```

## Conservation Fixers

The conservation fixers enforce global physical budgets by nudging one field to
close the budget. They read the t0 state from `batch_dict[input_source_key]`
(default `"x_physical"`) and share a physics core configured with the same keys:
`save_loc_physics` (a NetCDF file of grid/level metadata), `lon_lat_level_name`
(coordinate variable names — `[lon2d, lat2d, a_coef, b_coef]` for a hybrid-sigma
grid, or `[lon2d, lat2d, p_level]` for pressure levels), `grid_type`
(`"sigma"` default or `"pressure"`), and `midpoint`.

### `tracer_fixer` (TracerFixer)

**AutoAPI:** {py:obj}`credit.postblock.conservation.TracerFixer`

Clamps tracer fields to a lower (and optional upper) threshold by name — e.g.
forcing specific humidity and precipitation to be non-negative. Thresholds may
be a scalar applied to all variables or a per-variable list aligned with
`tracer_vars`. This is the one conservation block that needs no physics core.

```yaml
type: "tracer_fixer"
args:
  tracer_vars:
    - "ERA5/prognostic/3d/specific_total_water"
    - "ERA5/prognostic/2d/total_precipitation"
  tracer_thres: 0.0            # scalar, or a per-variable list [0.0, 0.0]
  tracer_thres_max: null       # optional upper bound(s)
```

### `global_mass_fixer` (GlobalMassFixer)

**AutoAPI:** {py:obj}`credit.postblock.conservation.GlobalMassFixer`

Conserves global dry-air mass by rescaling surface pressure at t1 so the
predicted dry-air mass matches the t0 target derived from the input state
(hybrid-sigma path).

```yaml
type: "global_mass_fixer"
args:
  q_var: "ERA5/prognostic/3d/specific_total_water"
  sp_var: "ERA5/prognostic/2d/surface_pressure"
  save_loc_physics: "$SCRATCH/physics/ERA5_physics_grid.nc"
  lon_lat_level_name: ["lon2d", "lat2d", "a_half", "b_half"]
  grid_type: "sigma"
  midpoint: false
```

### `global_water_fixer` (GlobalWaterFixer)

**AutoAPI:** {py:obj}`credit.postblock.conservation.GlobalWaterFixer`

Closes the global water budget by rescaling precipitation so the column-water
tendency plus evaporation balances precipitation over the forecast step.
`lead_time_periods` is the step length in hours.

```yaml
type: "global_water_fixer"
args:
  q_var: "ERA5/prognostic/3d/specific_total_water"
  sp_var: "ERA5/prognostic/2d/surface_pressure"
  precip_var: "ERA5/prognostic/2d/total_precipitation"
  evapor_var: "ERA5/prognostic/2d/evaporation"
  lead_time_periods: 6
  save_loc_physics: "$SCRATCH/physics/ERA5_physics_grid.nc"
  lon_lat_level_name: ["lon2d", "lat2d", "a_half", "b_half"]
  grid_type: "sigma"
  midpoint: false
```

### `global_energy_fixer` / `global_energy_fixer_updown` (GlobalEnergyFixerUpDown)

**AutoAPI:** {py:obj}`credit.postblock.conservation.GlobalEnergyFixerUpDown`

Conserves global total energy using an explicit up/down flux decomposition: the
column total-energy tendency is forced to match the net TOA + surface energy
fluxes, with temperature carrying the correction. The TOA downwelling shortwave
(SOLIN) is an input-only forcing absent from the prediction, so it is read from
the input dict by name. Both registry keys map to the same class.

```yaml
type: "global_energy_fixer_updown"
args:
  T_var: "ERA5/prognostic/3d/temperature"
  q_var: "ERA5/prognostic/3d/specific_total_water"
  U_var: "ERA5/prognostic/3d/u_component_of_wind"
  V_var: "ERA5/prognostic/3d/v_component_of_wind"
  sp_var: "ERA5/prognostic/2d/surface_pressure"
  surface_geopotential_name: "geopotential_at_surface"
  toa_down_solar_input_var: "ERA5/dynamic_forcing/2d/toa_incident_solar_radiation"
  toa_up_solar_var: "ERA5/prognostic/2d/top_up_solar_radiation"
  toa_up_olr_var: "ERA5/prognostic/2d/top_up_thermal_radiation"
  surf_down_solar_var: "ERA5/prognostic/2d/surface_down_solar_radiation"
  surf_up_solar_var: "ERA5/prognostic/2d/surface_up_solar_radiation"
  surf_down_lw_var: "ERA5/prognostic/2d/surface_down_thermal_radiation"
  surf_up_lw_var: "ERA5/prognostic/2d/surface_up_thermal_radiation"
  surf_sh_var: "ERA5/prognostic/2d/surface_sensible_heat_flux"
  surf_lh_var: "ERA5/prognostic/2d/surface_latent_heat_flux"
  lead_time_periods: 6
  save_loc_physics: "$SCRATCH/physics/ERA5_physics_grid.nc"
  lon_lat_level_name: ["lon2d", "lat2d", "a_half", "b_half"]
  grid_type: "sigma"
  midpoint: false
```

## Filtering, Masking, and Advection

### `wind_artifact_filter` (WindArtifactFilter)

**AutoAPI:** {py:obj}`credit.postblock.wind_filter.WindArtifactFilter`

Smooths spurious grid-scale wind artifacts near the jet stream. It detects
anomalously high wind speed at `mask_level`, builds an anisotropic blend mask
(wider zonally, matching jet geometry), and blends every field in `target_vars`
toward a Gaussian-smoothed version of itself, weighted by that mask. Points far
from a detected region are left unchanged. **`speed_threshold` is
unit-sensitive** — this block does no scaling itself, so recalibrate it if you
move the block from before an inverse-scale step (normalized values) to after
it (physical m/s); the default is tuned for normalized output.

```yaml
type: "wind_artifact_filter"
args:
  u_var: "CESM/prognostic/3d/U"
  v_var: "CESM/prognostic/3d/V"
  target_vars:
    - "CESM/prognostic/3d/U"
    - "CESM/prognostic/3d/V"
    - "CESM/prognostic/3d/T"
  mask_level: 14
  target_levels: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
  smooth_sigma_zonal: 3.0
  smooth_sigma_meridional: 0.5
```

### `semilagrangian_advection` (SemiLagrangianAdvectionPost)

**AutoAPI:** {py:obj}`credit.postblock.advect.SemiLagrangianAdvectionPost`

Overwrites each tracer in the output dict with its value advected one
`timestep_seconds` step by the concurrently predicted winds, with the vertical
component driven by a continuity-derived `omega`. Run it after the inverse
scaler so winds and tracers are in physical units. A matching preblock advects
initial-condition tracers before the model runs.

```yaml
type: "semilagrangian_advection"
args:
  tracer_vars:
    - "ERA5/prognostic/3d/specific_humidity"
  u_var: "ERA5/prognostic/3d/u_component_of_wind"
  v_var: "ERA5/prognostic/3d/v_component_of_wind"
  surface_pressure_var: "ERA5/prognostic/2d/surface_pressure"
  timestep_seconds: 21600.0
  levels: null          # or e.g. [1, 2, ..., 137] to subselect model levels
```

### `wet_mask_samudra` (WetMaskBlock)

**AutoAPI:** {py:obj}`credit.postblock.wet_mask_samudra.WetMaskBlock`

Ocean-specific postblock for the Samudra emulator: masks predictions so land
points are zeroed and ocean points preserve their values, focusing learning on
ocean regions. It builds the wet mask from the dataset referenced in the config,
so it takes the full `conf` rather than individual variable keys, and reads the
prediction under `key` (default `"prediction"`).

```yaml
type: "wet_mask_samudra"
args:
  key: "prediction"
  # conf is supplied by the Samudra ocean trainer, not hand-written here
```

## Example: the WXFormer prediction + target-twin chain

The config below is the standard chain for a `BaseLoss` scored in physical
units. The **prediction chain** reconstructs `y_pred` (with `detach: false` so
gradients flow), inverse-scales to physical units, inverts the log transform,
and derives geopotential. The **target twin** runs the identical steps on the
flat target `y` into `y_target_processed`, so `BaseLoss` compares the two sides
in the same units. The geopotential twin is only needed when
`loss.include_computed_diagnostics: true`.

```yaml
postblocks:
  per_step:
    # ---- Prediction chain: y_pred -> y_processed, in physical units ----
    reconstruct:
      type: reconstruct
      args:
        detach: false                     # REQUIRED: BaseLoss backprops through y_processed
    scaler:
      type: bridgescaler_transform
      args:
        scaler_path: /path/to/scaler.json
        variables: []
        method: inverse_transform
    log_trans:
      type: exp_transform
      args:
        variables:
          - "ERA5/prognostic/3d/specific_humidity"
    geopotential:
      type: geopotential_diagnostic
      args:
        output_name: "ERA5/computed_diagnostic/3d/geopotential"
        surface_geopotential_var: "ERA5/static/2d/geopotential_at_surface"
        surface_pressure_var: "ERA5/prognostic/2d/surface_pressure"
        temperature_var: "ERA5/prognostic/3d/temperature"
        specific_humidity_var: "ERA5/prognostic/3d/specific_humidity"
        level_info_file: "ERA5_Lev_Info.nc"

    # ---- Target twin: y -> y_target_processed (same steps, key overridden) ----
    reconstruct_target:
      type: reconstruct
      args:
        in_key: "y"
        out_key: "y_target_processed"
    scaler_target:
      type: bridgescaler_transform
      args:
        scaler_path: /path/to/scaler.json
        variables: []
        method: inverse_transform
        key: "y_target_processed"
    log_trans_target:
      type: exp_transform
      args:
        variables:
          - "ERA5/prognostic/3d/specific_humidity"
        key: "y_target_processed"
```

## Writing Your Own Postblock

Custom postblocks subclass `credit.postblock.base.BasePostblock` and can be
plugged in from your own package via the config's `custom_objects:` block —
see the [Custom Objects](Custom.md) guide for the full recipe.
