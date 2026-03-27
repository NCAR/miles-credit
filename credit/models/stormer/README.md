# Stormer

**Paper:** Nguyen et al., "Scaling transformer neural networks for skillful and reliable medium-range weather forecasting," 2023.
https://arxiv.org/abs/2312.03876

**Original code:** https://github.com/tung-nd/stormer (MIT License)

## Architecture

Plain Vision Transformer (ViT) applied directly to weather fields:

```
(B, C, H, W) → PatchEmbed (Conv2d) → pos_embed → N × Block → LayerNorm → Linear head → (B, C_out, H, W)
```

Each `Block` is standard pre-norm: LayerNorm → MHSA → residual; LayerNorm → MLP → residual.
No variable-specific embeddings (contrast with ClimaX). No windowed attention (contrast with Swin).
Global attention over all patches at every layer.

## CREDIT implementation

`stormer.py` — written from scratch following the paper and reference repo.
Differences from reference:
- Flat `(B, C, H, W)` I/O instead of variable-dict interface.
- No lead-time conditioning (add as an extra channel if needed).
- No pretrained weights loaded.

## Validation status

**Architectural smoke test only** (correct shape, no NaN, gradients flow).
Not yet trained to convergence or compared against published WB2 scores.
Loss-curve sanity check pending. Confidence will build after a short training run.

## Known caveats

- Global attention is O(N²) in sequence length. Slow at full ERA5 resolution
  (N = H/p × W/p). Use `patch_size ≥ 4` at 1° resolution.
- No masking for land/sea or pole weighting — add in the loss, not the model.
