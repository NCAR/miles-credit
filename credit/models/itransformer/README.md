# iTransformer

**Paper:** Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting," ICLR 2024.
https://arxiv.org/abs/2310.06625

**Original code:** https://github.com/thuml/iTransformer (MIT License)

## Architecture

iTransformer inverts the usual spatial-token design: instead of one token per spatial patch, **each variable/channel gets exactly one token** whose value is the full spatial field projected to a single embedding vector. Attention is then applied *across variables*, not across spatial locations.

```
(B, C_in, H, W)
  → flatten spatial:  (B, C_in, H*W)
  → shared Linear(H*W → d_model)  →  (B, C_in, d_model)   # one token per variable
  → N × iTransformerBlock
        LN → VariateAttention(C_in tokens) → residual
        LN → VariateMLP                    → residual
  → shared Linear(d_model → H*W)  →  (B, C_in, H*W)
  → reshape                        →  (B, C_in, H, W)
  → Conv2d(C_in, C_out, 1)         →  (B, C_out, H, W)
```

**Key property:** memory scales as O(C²) in the number of channels, independent of spatial resolution. This makes iTransformer attractive when C is large (100+ variables) and spatial resolution is high.

## CREDIT implementation

`itransformer.py` — written from scratch following the ICLR 2024 paper. The original code targets multivariate time series; this CREDIT port adapts it to 2-D weather fields.

Differences from the reference repo:
- Input is a 2-D spatial field `(B, C, H, W)` rather than a time-series `(B, T, C)`.
- The spatial dimension `H*W` plays the role of the sequence length in the paper.
- A `Conv2d(C_in, C_out, 1)` channel-mixing layer at the output handles the case `C_in ≠ C_out` (static channels dropped at output).
- `CREDITiTransformer` squeezes the trainer's 5-D `(B, C, T, H, W)` tensor before forwarding.

## Validation status

**Architectural smoke test only** (correct shape, no NaN, gradients flow).
Not yet trained to convergence or compared against published WB2 scores.

## CREDIT config

```yaml
model:
  type: itransformer
  in_channels: 70          # total input channels (atmos + surface + static)
  out_channels: 69         # total output channels (no static)
  img_size: [192, 288]     # (H, W) of your lat/lon grid
  d_model: 256             # per-variable embedding dimension
  depth: 6                 # number of iTransformerBlocks
  num_heads: 8             # attention heads; d_model // 32 is a good rule
  mlp_ratio: 4.0
  drop: 0.0
```

Suggested production config for 1° ERA5 with ~100 channels:
`d_model=512, depth=8, num_heads=8` (~60M params).

## Known caveats

- **Memory scales with C², not N²**: attention is O(C²) per layer. Very large channel counts (C > 500) will be memory-intensive. Spatial resolution is free.
- The spatial projection `Linear(H*W → d_model)` can be large when `H*W` is big (e.g. 55296 at 1°). Consider a convolutional pre-reduction if memory is tight.
- No positional encoding is applied in the variate dimension — variable ordering in the channel axis does not matter, but positional information within each variable's spatial field is discarded after projection. Residual spatial structure is recovered via the channel-mix Conv2d at the output.
- For best results ensure C is large (100+ channels); with few channels the attention budget is wasted and a plain ViT is likely better.
