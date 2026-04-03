# WXFormer

**Our model.** Developed at NCAR/MILES for production ERA5 weather forecasting.

**Paper:** Schreck et al., "Community Research Earth Digital Intelligence Twin: a scalable framework for AI-driven Earth System Modeling," npj Climate and Atmospheric Science 8, 239 (2025).
https://doi.org/10.1038/s41612-025-01125-6

## Variants

| File | Class | Config key | Notes |
|---|---|---|---|
| `crossformer.py` | `CrossFormer` | `wxformer` | Deterministic backbone |
| `crossformer_ensemble.py` | `CrossFormerWithNoise` | `wxformer-sdl` | SDL ensemble (v1 backbone) |

Legacy alias: `crossformer-style` → same as `wxformer-sdl` (kept so old scheduler configs don't break).

## Architecture

CrossFormer backbone: multi-scale cross-shaped window attention with dynamic
position bias. Encoder processes input at multiple patch sizes; decoder uses
transposed-conv upsampling with skip connections.

SDL (Stochastic Decomposition Layer) injects learned noise at each transformer
scale to produce ensemble members without separate forward passes.

## Lineage

**WXFormer author:** John Schreck (NCAR/MILES).

**CrossFormer backbone paper:** Zhang et al., "CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention," 2021.
https://arxiv.org/abs/2108.01072

**CrossFormer backbone implementation:** Phil Wang (lucidrains) — PyTorch
implementation that `crossformer.py` is based on. Original GitHub repo has
been taken down; work is now available on GitLab under the same username:
https://gitlab.com/lucidrains

John Schreck adapted the CrossFormer backbone for weather forecasting:
encoder-decoder structure replacing the classification head, SDL noise
injection layer for ensemble generation, CREDIT variable channel conventions.
WXFormer v1 was used to produce the MILES ensemble results reported in
internal papers/campaigns.

**Pixel-shuffle upsampling (`upsample_with_ps`):** Will Chapman (NCAR/MILES).
Replaces the transposed-conv decoder with sub-pixel convolution (pixel shuffle),
which partially resolves the checkerboard/grid artefact problem present in
the original transposed-conv upsampling path. Enable with `upsample_with_ps: true`
in the model config.

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
