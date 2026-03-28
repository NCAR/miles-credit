# WXFormer

**Our model.** Developed at NCAR/MILES for production ERA5 weather forecasting.

## Variants

| File | Class | Config key | Notes |
|---|---|---|---|
| `crossformer.py` | `CrossFormer` | `wxformer` | Deterministic backbone |
| `crossformer_ensemble.py` | `CrossFormerWithNoise` | `wxformer-sdl` | SDL ensemble (v1 backbone) |
| `wxformer_v2_ensemble.py` | `CrossFormerV2WithNoise` | `wxformer-v2-sdl` | SDL ensemble (v2 backbone, WIP) |

Legacy alias: `crossformer-style` → same as `wxformer-sdl` (kept so old scheduler configs don't break).

## Architecture

CrossFormer backbone: multi-scale cross-shaped window attention with dynamic
position bias. Encoder processes input at multiple patch sizes; decoder uses
transposed-conv upsampling with skip connections.

SDL (Stochastic Decomposition Layer) injects learned noise at each transformer
scale to produce ensemble members without separate forward passes.

## Lineage

Based on `CrossFormer` (Zhang et al., 2021, https://arxiv.org/abs/2108.01072)
adapted for weather forecasting at NCAR. WXFormer v1 was used to produce the
MILES ensemble results reported in internal papers/campaigns.

## CREDIT config

Deterministic backbone (`wxformer`):

```yaml
model:
  type: wxformer
  image_height: 192
  image_width: 288
  frames: 2
  channels: 4               # 3D vars: U, V, T, Q
  surface_channels: 7       # surface vars
  input_only_channels: 3    # forcing (e.g. TOA insolation) — not predicted
  output_only_channels: 0
  levels: 15                # pressure levels per 3D var
  dim: [64, 128, 256, 512]
  depth: [2, 2, 8, 2]
  use_spectral_norm: true   # never remove — needed for DDP stability
```

SDL ensemble (`wxformer-sdl`) adds noise injection at each scale:

```yaml
model:
  type: wxformer-sdl
  # same spatial/channel args as above
  noise_scale: 0.1          # amplitude of injected latent noise
```

`crossformer-style` is a legacy alias for `wxformer-sdl` — kept for backward
compatibility with old scheduler configs.

## Validation status

**Trained and evaluated against ERA5.** This is the production model. Results
are documented in the MILES CAMulator ensemble campaign (see project memory).
WB2 scores can be reproduced with `applications/eval_weatherbench.py`.
