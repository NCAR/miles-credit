# WXFormer Next

The next-generation WXFormer family. These models keep the CrossFormer backbone
from WXFormer v1 (see [README.md](README.md)) and add explicit vertical coupling,
global spectral mixing, and a cubed-sphere variant that runs natively on the
CESM spectral-element (SE) grid.

Note that the lat/lon production model still lives in `crossformer.py` and is
documented in `README.md`. This file covers only the newer architectures.

## Variants

| File | Class | Config key | Grid | Notes |
|---|---|---|---|---|
| `wxformer_next.py` | `NextGenWXFormer` | `nextgen_wxformer` | lat/lon | CrossFormer U-Net + level embed + column attn + spectral GNN |
| `cube_sphere_wxformer.py` | `CubeSphereWxFormer` | `cube_sphere_wxformer` | cubed-sphere SE | per-face CrossFormer + cross-face attention |
| `cube_sphere_wxformer.py` | `CubeSphereWxFormerNext` | `cube_sphere_wxformer_next` | cubed-sphere SE | CubeSphereWxFormer + the NextGen additions |

## What is new over WXFormer v1

Three additions, applied on top of the CrossFormer encoder-decoder:

1. **Level embeddings** (`LevelEmbedding`). A learned per-level bias is broadcast
   onto the atmospheric input channels, so the network knows which pressure level
   each channel belongs to. This follows the level-aware tokenization used in
   Pangu and Aurora.
2. **Column attention** (`ColumnAttention`). Multi-head attention runs across the
   pressure levels at each grid pixel, giving explicit vertical coupling before
   the encoder. The idea is from ArchesWeather. A spatial pooling stride
   (`col_attn_stride`) keeps attention memory bounded on large grids: use 8 for
   640×1280.
3. **Spectral GNN bottleneck** (`SpectralGNNBottleneck`). At the encoder
   bottleneck we mix information globally through K learned virtual spectral
   nodes (default K=64): pool the spatial nodes to K nodes with a learned
   aggregation, apply a channel MLP, then scatter corrections back. This is a
   learned spectral graph convolution (cost O(N*K), grid-agnostic), not an
   FFT or spherical-harmonic transform. It gives the model long-range context
   that windowed attention alone cannot reach.

Spectral normalization is applied throughout. As with v1, do not remove it: it
is needed for DDP stability.

## NextGenWXFormer (lat/lon)

A drop-in successor to `wxformer` on the regular lat/lon grid. The encoder is the
same four-stage CrossFormer pyramid; the additions above wrap the input and the
bottleneck. Predictions are a residual delta added to the last input frame, so
the model starts near persistence at initialization.

```yaml
model:
  type: nextgen_wxformer
  image_height: 640
  image_width: 1280
  frames: 2
  channels: 4               # 3D vars: U, V, T, Q
  surface_channels: 7
  input_only_channels: 3    # forcing (e.g. TOA insolation), not predicted
  output_only_channels: 0
  levels: 15
  dim:   [64, 128, 256, 512]
  depth: [2, 2, 8, 2]
  global_window_size: [5, 5, 2, 1]
  local_window_size: 10
  cross_embed_strides: [4, 2, 2, 2]
  # NextGen additions
  col_attn_heads: 4
  col_attn_stride: 8        # pool before column attention to bound memory at 640x1280
  decoder_col_attn: false
  num_spectral_nodes: 64
  use_spectral_norm: true
```

## CubeSphereWxFormer (cubed-sphere SE grid)

WXFormer rebuilt to run on the cubed sphere instead of lat/lon. The motivation
is uniform resolution: a 0.25° lat/lon grid massively oversamples the poles,
while the cubed sphere is quasi-uniform and has no polar singularity.

### How the grid flows

Data arrives on lat/lon and is mapped to the SE grid by a preblock, then
reshaped into six cube faces inside the model:

1. **lat/lon → SE** (`tripole_to_se` preblock, `credit/preblock/latlon_to_se.py`).
   A fixed sparse matrix (precomputed ESMF weights) regrids each variable from
   the (nlat × nlon) grid to the SE columns (ncol = 777,602 for ne120). One sparse
   mat-mul, run on the GPU.
2. **SE → cube** (inside the model). A permutation index (`se_index`) scatters the
   SE columns into a (6, E, E) tensor, where E is the face-edge length (361 for
   ne120). This is a pure reindex, not interpolation.
3. **encode / decode**. Each face is padded up to a size that downsamples cleanly
   through the four stages (361 → 384 with the default windows), encoded by a
   CrossFormer stage, mixed across faces by attention, then decoded back.
4. **cube → SE → lat/lon**. The cube is gathered back to SE columns. For
   verification on the native lat/lon grid, the `se_to_latlon` postblock
   (`credit/postblock/regrid_se_to_latlon.py`) maps predictions back with a fixed
   reverse-weight mat-mul.

### Grid-agnostic geometry

Nothing is hard-coded to ne120. The face-edge length is inferred from the SE
index, the padded encoder size is derived from the strides and attention windows,
and the crop offset is derived from the halo width. The same class runs at ne30,
ne120, or any other cubed-sphere resolution given the matching static files.

### Pad, not interpolate

Going from a 361-edge face to the padded 384 is a pad on the bottom and right.
The inverse at the end of the decoder is therefore a **crop**, not a resample.
We do not interpolate predictions back to the face size: a crop is the exact
inverse of the pad and avoids blurring every output field. With the default
strides `[2, 2, 2, 2]` the encoder runs 384 → 192 → 96 → 48 → 24 and the decoder
mirrors it back.

### Cross-face coupling

Three optional mechanisms keep the six faces consistent at their shared edges
(all controlled by an adjacency file; if it is missing they are skipped):

- **FaceAttention** mixes the six face tokens at every spatial position, the
  video-transformer trick applied to cube faces. Always on.
- **HaloExchange** copies real neighbor data into each face border before padding,
  so the encoder sees true context at boundaries instead of zeros.
- **FaceEdgeAttention** runs sparse attention between the border nodes of
  physically adjacent faces, enforcing edge consistency in feature space.

```yaml
model:
  type: cube_sphere_wxformer
  se_index_path: ".../se_index_ne120.npy"
  adjacency_path: ".../se_face_adjacency_ne120.npz"   # omit to disable halo/edge attn
  frames: 1
  channels: 9
  surface_channels: 16
  input_only_channels: 13
  output_only_channels: 2
  levels: 13
  dim:   [64, 128, 256, 512]
  depth: [2, 2, 8, 2]
  global_window_size: [6, 6, 6, 6]
  local_window_size: 12
  cross_embed_strides: [2, 2, 2, 2]   # encoder: 384 -> 192 -> 96 -> 48 -> 24
  halo_size: 6
  edge_attn_heads: 4
  edge_attn_width: 2
  use_spectral_norm: true
```

## CubeSphereWxFormerNext

`CubeSphereWxFormerNext` is `CubeSphereWxFormer` with the three NextGen additions
(level embeddings, column attention, spectral GNN bottleneck) applied per face.
The spectral bottleneck is sized automatically to the deepest stage resolution
(24×24 per face at ne120 with the default strides).

```yaml
model:
  type: cube_sphere_wxformer_next
  se_index_path: ".../se_index_ne120.npy"
  adjacency_path: ".../se_face_adjacency_ne120.npz"
  frames: 1
  channels: 9
  surface_channels: 16
  input_only_channels: 13
  output_only_channels: 2
  levels: 13
  dim:   [64, 128, 256, 512]
  depth: [2, 2, 8, 2]
  global_window_size: [6, 6, 6, 6]
  local_window_size: 12
  cross_embed_strides: [2, 2, 2, 2]
  halo_size: 6
  edge_attn_heads: 4
  edge_attn_width: 2
  col_attn_heads: 3        # must divide channels
  col_attn_stride: 4
  decoder_col_attn: false
  num_spectral_nodes: 64
  use_spectral_norm: true
```

Full WeatherBench2 configs live in `config/model_zoo/`:
`wxformer_cubesphere_wb2.yml`, `wxformer_cubesphere_next_wb2.yml`, and
`wxformer_cubesphere_next_large_wb2.yml` (a 580M-parameter variant). Smoke configs
are under `config/model_zoo/smoke/`.

### Eval pipeline

Predictions come off the model on the SE grid. For WeatherBench2 scoring on the
native 0.25° lat/lon grid, the config runs two postblocks in order: `reconstruct`
(flat tensor to nested variable dict) then `se_to_latlon` (SE to lat/lon). Use the
bilinear reverse weights for eval parity; swap to the conservative reverse weights
when area-integral budgets matter.

Note that training is self-consistent on the SE grid: the target is regridded to
SE as well, so the model never sees the lat/lon regrid error during training. The
round-trip back to lat/lon happens only at eval time.

## Static files (cubed-sphere)

The cubed-sphere models need precomputed static files (not committed here; built
in the credit-mesaclip repo under `mesaclip/static/`):

| File | Purpose |
|---|---|
| `latlon721x1440_to_se_ne120.nc` | forward regrid weights (lat/lon → SE), `tripole_to_se` |
| `se_ne120_to_latlon721x1440.nc` | reverse regrid weights (SE → lat/lon), `se_to_latlon` |
| `se_index_ne120.npy` | SE ↔ cube reindex |
| `se_face_adjacency_ne120.npz` | face-edge adjacency for halo exchange / edge attention |

Conservative variants of the regrid weights (`*_conserve.nc`) are available for
budget-sensitive evaluation.

## Lineage

**Author:** John Schreck (NCAR/MILES). Built on the WXFormer / CrossFormer
backbone (see [README.md](README.md) for full backbone credits: CrossFormer by
Zhang et al.; PyTorch implementation by Phil Wang / lucidrains; pixel-shuffle
upsampling by Will Chapman).

Architectural ideas borrowed: level-aware input from Pangu/Aurora, column
attention from ArchesWeather. The global bottleneck is a learned spectral graph
convolution (not SFNO): it pools to K virtual spectral nodes and scatters back,
which keeps it grid-agnostic and avoids any FFT or spherical-harmonic transform.

## Validation status

`nextgen_wxformer` and the cubed-sphere models train and smoke-test on
WeatherBench2 ERA5. They are research models and have not yet replaced the v1
production WXFormer. The cubed-sphere path additionally depends on the
credit-mesaclip static files listed above.
