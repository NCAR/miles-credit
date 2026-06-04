# ArchesWeather

**Paper:** Hétu et al., "ArchesWeather & ArchesWeatherGen: a deterministic and generative model for efficient ML weather forecasting," INRIA 2024.
https://arxiv.org/abs/2412.12971

**Original code:** https://github.com/INRIA/geoarches (Apache-2.0)

## Architecture

Alternating local window attention and column attention:

```
(B, C, H, W) → PatchEmbed → (B, n_h, n_w, d_model)
  → Block 0: WindowAttention  (mesoscale, within ws×ws windows)
  → Block 1: ColumnAttention  (vertical coupling, across d_model features per point)
  → ... × depth/2 pairs ...
  → PixelShuffle head → (B, C_out, H, W)
```

**Window attention** captures local mesoscale interactions within spatial windows.

**Column attention** (CREDIT approximation): at each (h, w) grid point, the d_model
embedding is split into `n_col_tokens` sub-vectors that attend to each other. This
approximates ArchesWeather's vertical column attention — information flows between
features that implicitly encode different pressure levels and variables.

## CREDIT config

```yaml
model:
  type: arches
  in_channels: 80
  out_channels: 84
  img_size: [192, 288]
  patch_size: 4
  d_model: 256
  depth: 8           # alternates window/column: 4 pairs
  num_heads: 8
  window_size: 8     # window size in patch units
  n_col_tokens: 8    # must divide d_model
  mlp_ratio: 4.0
```

## Known caveats

- `d_model` must be divisible by `n_col_tokens`.
- `img_size` after patching must accommodate `window_size`: n_h and n_w should be
  multiples of `window_size` for clean tiling (padding is applied otherwise).
- ArchesWeatherGen's flow-matching generative head is not implemented here;
  this is the deterministic ArchesWeather backbone only.
