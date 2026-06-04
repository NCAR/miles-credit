# ClimaX

**Paper:** Nguyen et al., "ClimaX: A foundation model for weather and climate," 2023.
https://arxiv.org/abs/2301.10343

**Original code:** https://github.com/microsoft/ClimaX (MIT License)

## Architecture

Key innovation: **per-variable tokenization**.  Each input channel is embedded
independently (shared `Conv2d` patch projection + per-channel learned `var_embed`),
allowing the model to generalise across different variable sets.

```
Each channel c:
  (B, 1, H, W) → patch_proj → (B, N, D) + var_embed[c] + pos_embed

Optional aggregation transformer (agg_depth blocks) over variable axis at each patch:
  (B*N, C, D) → agg_blocks → mean → (B, N, D)

Main ViT backbone:
  (B, N, D) → N × Block → head → (B, C_out, H, W)
```

## CREDIT implementation

`climax.py` — written from scratch following the paper and reference repo.
Differences from reference:
- Flat `(B, C, H, W)` I/O. Variable IDs inferred from channel index, not a
  user-supplied list.
- `agg_depth=0` skips the aggregation transformer and uses a direct mean
  (cheaper, still works for fixed variable sets).
- No pretrained backbone weights.

## Validation status

**Architectural smoke test only.** Not yet trained to convergence.
Loss-curve sanity check pending.

## CREDIT config

```yaml
model:
  type: climax
  in_channels: 70
  out_channels: 69
  img_size: [192, 288]
  patch_size: 4          # larger = cheaper aggregation step
  embed_dim: 1024
  depth: 8
  num_heads: 16
  agg_depth: 2           # 0 = skip aggregation (pure mean), 1–2 typical
  drop_rate: 0.0
```

Set `agg_depth: 0` for a cheaper model that still uses the per-variable embeddings — the aggregation transformer is optional.

## Known caveats

- Memory scales as O(B × C × N) in the aggregation step. With large C (many
  variables) and small patch size this can be significant.
- Variable ordering matters: `var_embed` is indexed by position in the channel
  dimension. Keep channel order consistent across training and inference.
