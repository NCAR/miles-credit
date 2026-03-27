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

## Validation status

**Trained and evaluated against ERA5.** This is the production model. Results
are documented in the MILES CAMulator ensemble campaign (see project memory).
WB2 scores can be reproduced with `applications/eval_weatherbench.py`.
