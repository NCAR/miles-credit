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

## CREDIT config

```yaml
model:
  type: stormer
  in_channels: 70        # total input channels (atmos + surface + static)
  out_channels: 69       # total output channels
  img_size: [192, 288]   # (H, W) of your lat/lon grid
  patch_size: 4          # spatial patch size — 4 recommended at 1°
  embed_dim: 1024        # ViT width; 768–1024 for production runs
  depth: 12              # transformer depth; 8–16 typical
  num_heads: 16          # attention heads; embed_dim // 64 is a good rule
  drop_rate: 0.0
```

Suggested production config: `embed_dim=1024, depth=12, num_heads=16` (~307M params at 192×288).

## Known caveats

- Global attention is O(N²) in sequence length. At 1° (192×288) with `patch_size=4`
  you get N=3456 tokens — manageable on A100-80GB at batch size 1–2.
  Use `patch_size ≥ 4`; smaller patches hit memory fast.
- No masking for land/sea or pole weighting — add in the loss, not the model.
