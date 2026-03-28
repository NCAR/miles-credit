# Pangu-Weather

**Paper:** Bi et al., "Accurate medium-range global weather forecasting with 3D neural networks," Nature 2023.
https://www.nature.com/articles/s41586-023-06185-3

**Original code:** Huawei Cloud — https://github.com/198808xc/Pangu-Weather (custom license)
Also NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo (Apache 2.0)

## Architecture

**3D Earth Transformer** — treats atmospheric state as a 3D volume (pressure × lat × lon)
and applies 3D window attention at multiple scales.

```
(B, C, H, W) flat
→ split: surf (B, n_surf, H, W) + atmos (B, n_atmos, n_levels, H, W) + static
→ Surface branch: 2D patch embed → 2D Swin blocks
→ Upper-air branch: 3D patch embed → 3D Swin blocks (pressure×lat×lon windows)
→ Upsample + merge
→ Surface head → surf output
→ Upper-air head → rearrange to (B, n_atmos×n_levels, H, W)
→ concatenate → (B, C_out, H, W)
```

Key design choices:
- Variables are named (surf/atmos/static lists).
- Separate 2D and 3D attention branches for surface and pressure-level variables.
- Multi-scale (like Swin): patch merging + expanding between stages.

## CREDIT implementation

`pangu.py` — translated from PhysicsNeMo reference.
`CREDITPangu` wraps with CREDIT's flat `(B, C, H, W)` interface.
Channel layout: `[surf_vars | atmos_vars × levels | static_vars]`

`**pangu_kwargs` are forwarded to `PanguModel(...)` for architectural control
(embed_dim, depths, num_heads, window_size, etc.).

## Validation status

**Architectural smoke test only** at small variable config (2 surf, 2 atmos, 1 static,
2 levels). Not weight-compatible with Huawei's pretrained Pangu checkpoint
(different normalisation, variable set, and model structure details).

## CREDIT config

Pangu uses named variable lists, not `in_channels`. Channel layout is
`[surf_vars | atmos_vars × levels | static_vars]`.

```yaml
model:
  type: pangu
  surf_vars: ["2t", "10u", "10v", "msl"]
  atmos_vars: ["z", "u", "v", "t", "q"]
  static_vars: ["lsm", "z", "slt"]
  atmos_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  # pangu_kwargs forwarded to PanguModel(...):
  pangu_kwargs:
    embed_dim: 192
    patch_size: [2, 4, 4]      # [level_patch, lat_patch, lon_patch]
    window_size: [2, 6, 12]    # 3D attention window; n_levels must be divisible by window_size[0]
    depths: [2, 6, 6, 2]
    num_heads: [6, 12, 12, 6]
```

`n_levels` (len of `atmos_levels`) must be divisible by `patch_size[0]` and
`window_size[0]`. The default 13-level ERA5 setup requires `patch_size[0]=1`
or `window_size[0]=1` unless levels are subsampled.

## Known caveats

- Pangu's pretrained weights are under a restrictive non-commercial license.
  This implementation is for training from scratch only.
- The 3D window attention requires `n_levels` to be divisible by the 3D
  window depth. Use `pangu_kwargs` to adjust.
- Default hyperparams match the paper's architecture but are large; scale down
  with `pangu_kwargs` for ablation runs.
