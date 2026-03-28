# Aurora

**Paper:** Bodnar et al., "Aurora: A Foundation Model of the Atmosphere," 2024.
https://arxiv.org/abs/2405.13063

**Original code:** Microsoft Research — https://github.com/microsoft/aurora (MIT License)
Also NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo

## Architecture

Perceiver-style encoder-decoder with a **3D Swin Transformer** backbone:

```
(B, C, H, W) flat
→ split into surf_vars, atmos_vars (3D: levels×H×W), static_vars
→ Perceiver encoder: surface + atmos patches → latent tokens
→ Swin3D processor: 3D window attention over (level, lat, lon)
→ Perceiver decoder: latent tokens → output patches
→ reassemble → (B, C_out, H, W)
```

Key design choices:
- Variables are named (not positional) — model can generalise across variable sets.
- 3D attention over pressure levels avoids flattening the vertical dimension.
- Large default config (~1.3B params with default variable lists).

## CREDIT implementation

`model.py` — translated from Microsoft's reference implementation.
`CREDITAurora` wraps the model with CREDIT's flat `(B, C, H, W)` interface:
- Channel layout: `[surf_vars | atmos_vars × levels | static_vars]`
- lat/lon grids stored as buffers; `n_lat`/`n_lon` controls spatial resolution.
- `**aurora_kwargs` forwarded to underlying `Aurora(...)` for architectural hyperparams.

Default constructor instantiates at full production scale (~1.3B params).
Use `aurora_kwargs` to scale down (e.g. `embed_dim=256, depth=8`).

## Validation status

**Architectural smoke test only** at small variable config (2 surf, 2 atmos, 1 static,
2 levels). Production-scale default untested end-to-end at ERA5 resolution.
Not weight-compatible with Microsoft's pretrained Aurora checkpoint.

## CREDIT config

Aurora uses named variable lists, not `in_channels`. Channel layout is
`[surf_vars | atmos_vars × levels | static_vars]`.

```yaml
model:
  type: aurora
  surf_vars: ["2t", "10u", "10v", "msl"]
  atmos_vars: ["z", "u", "v", "t", "q"]
  static_vars: ["lsm", "z", "slt"]
  atmos_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  n_lat: 192              # set explicitly to avoid auto-inference from buffers
  n_lon: 288
  # aurora_kwargs forwarded to Aurora(...):
  aurora_kwargs:
    embed_dim: 256        # default is large (~1.3B); scale down for ablations
    num_heads: 8
    depth: 8
```

Default constructor produces a ~1.3B param model. Always pass `aurora_kwargs`
to scale down for training from scratch on limited compute.

## Known caveats

- Default config is very large. Pass small `aurora_kwargs` for ablation runs.
- Variable name lists must be consistent between training and inference.
- Microsoft's pretrained weights use a specific variable set and normalisation —
  do not expect transfer without retraining.
