# DLWP-HEALPix

**Paper:** Weyn et al. / Karlbauer et al., "Advancing Parsimonious Deep Learning Weather Prediction using the HEALPix Mesh," JAMES 2023.
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003700

**Original code:** NVIDIA PhysicsNeMo — https://github.com/NVIDIA/physicsnemo (Apache 2.0)
Also: NCAR/MILES DLWP repository.

## Architecture

Applies a **U-Net with HEALPix-aware convolutions** to weather fields.
HEALPix discretises the sphere into equal-area pixels arranged in 12 "base"
faces; convolutions are applied with circular padding that respects face
adjacency.

```
(B, C, H, W) lat/lon
→ _latlon_to_healpix  (nearest-neighbour reprojection to HEALPix pixel grid)
→ reshape to (B, C, hp_h, hp_w)
→ HEALPixUNet  (3-stage encoder-decoder with HEALPixConv blocks)
→ reshape to (B, C_out, n_pix)
→ _healpix_to_latlon  (reprojection back to lat/lon)
→ (B, C_out, H, W)
```

`HEALPixConv` = circular pad + Conv2d + GroupNorm + GELU.
The U-Net uses `AvgPool2d` downsampling and bilinear upsampling with skip connections.

## CREDIT implementation

`healpix.py` — written from scratch following PhysicsNeMo reference.
Key differences:
- Transparent `(B, C, H, W)` I/O: reprojection to/from HEALPix is inside the
  wrapper, invisible to the training loop.
- Reprojection uses precomputed nearest-neighbour index buffers (stored as
  `register_buffer`, move to GPU automatically).
- **healpy optional**: falls back to an approximate equal-area grid when
  `healpy` is not installed. Approximate grid will not produce identical pixel
  centres to true HEALPix — install `healpy` for correct behaviour.
- HEALPix face adjacency is approximated by treating the flattened pixel array
  as a 2D grid of shape `(ceil(sqrt(n_pix)), ceil(sqrt(n_pix)))`. This is a
  simplification; production DLWP uses explicit face-aware padding.

## Validation status

**Architectural smoke test only.** Approximate grid active (no `healpy` in env).
Reprojection correctness and face-adjacency handling are simplified — medium
confidence. Loss convergence will validate.

Install healpy: `pip install healpy` or `conda install -c conda-forge healpy`.
(Already installed in `credit-casper` env as of 2026-03-27.)

## CREDIT config

```yaml
model:
  type: healpix
  in_channels: 70
  out_channels: 69
  img_size: [192, 288]    # input lat/lon grid
  nside: 64               # HEALPix resolution: 12*64²=49152 pixels ≈ 1.5°
  embed_dim: 128          # base channel dim (doubles per stage)
  depth: 2                # HEALPixBlocks per encoder/decoder stage
  n_stages: 3             # U-Net depth (3 = 8× spatial reduction)
  lat_range: [-90.0, 90.0]
  lon_range: [0.0, 360.0]
```

Reprojection index buffers (`hp_to_ll`, `ll_to_hp`) are built at construction
and saved in the state_dict — first init is slow (~30s at nside=64), subsequent
loads are instant.

## Known caveats

- Without `healpy`, pixel centres are approximate → slight aliasing at face
  boundaries.
- `nside` controls HEALPix resolution: `n_pix = 12 * nside²`. For 1° global
  resolution `nside=64` (≈49k pixels) is typical.
- The internal U-Net operates on a square approximation of the HEALPix array;
  true face-aware convolutions would require a dedicated HEALPix conv kernel.
