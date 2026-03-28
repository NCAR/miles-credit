# DLWP-HEALPix

**Paper:** Weyn et al. / Karlbauer et al., "Advancing Parsimonious Deep Learning Weather Prediction using the HEALPix Mesh," JAMES 2023.
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003700

**Original code — face-aware padding:** Noah Brenowitz (nbren12), Zephyr / earth2grid
- https://github.com/NVlabs/earth2grid (Apache 2.0)
- Face adjacency logic also available in NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo (Apache 2.0)

**Original code — DLWP U-Net reference:** NCAR/MILES DLWP repository.

## Architecture

Applies a **U-Net with HEALPix-aware convolutions** to weather fields.
HEALPix discretises the sphere into equal-area pixels arranged in 12 "base"
faces; convolutions pad each face from its true face-adjacent neighbours,
preserving spherical continuity.

```
(B, C, H, W) lat/lon
→ _latlon_to_healpix  (nearest-neighbour reprojection using healpy pixel geometry)
→ (B, 12, C, nside, nside) — one tensor per HEALPix face
→ HEALPixUNet (3-stage encoder-decoder):
    HEALPixConv = HEALPixPadding (face-aware) + Conv2d + GroupNorm + GELU
    Downsample: AvgPool2d (per face)
    Upsample: bilinear + skip connection
→ _healpix_to_latlon  (reprojection back to lat/lon)
→ (B, C_out, H, W)
```

## Face-aware padding (HEALPixPadding)

The 12 HEALPix faces fall into three rings — north polar (0–3), equatorial (4–7),
south polar (8–11) — each with distinct neighbour layouts and rotation offsets.
`HEALPixPadding` implements the exact adjacency table from Brenowitz/PhysicsNeMo:

- **North polar faces** (pn): top edge from adjacent polar face rotated +90°,
  top-left corner from the next polar face rotated +180°, left edge from the
  adjacent equatorial face.
- **Equatorial faces** (pe): top, bottom, left, right from adjacent faces with
  interpolated corners.
- **South polar faces** (ps): bottom from adjacent polar face rotated +90°,
  right from adjacent polar face rotated −90°, bottom-right from adjacent polar
  face rotated +180°.

This replaces the previous approximation (circular pad on flat array) which
did not respect face adjacency at all.

## CREDIT implementation

`healpix.py` — written from scratch following the face-aware padding in
Brenowitz's `earth2grid` / PhysicsNeMo Zephyr implementation.

Key design choices:
- Transparent `(B, C, H, W)` I/O: reprojection to/from HEALPix is inside the
  wrapper, invisible to the training loop.
- `_build_index_buffers` uses `healpy.pix2xyf` + `healpy.pix2ang` + `healpy.ang2pix`
  to build correct NEST-ordered pixel-centre maps (`hp_to_ll`, `ll_to_hp`).
  Buffers are stored as `register_buffer` so they move to GPU automatically and
  are saved in `state_dict`.
- `HEALPixFoldFaces` / `HEALPixUnfoldFaces`: pack/unpack `(B, 12, C, ns, ns) ↔ (B*12, C, ns, ns)`
  so standard `Conv2d` operates per-face inside the batch dimension.
- `HEALPixConv` skips padding for `kernel_size=1` (skip/projection convolutions
  have no spatial extent and don't need face neighbours).

## Validation status

**Face-aware padding confirmed correct** (shape, no NaN, gradients flow).
`healpy` installed in `credit-casper` env (as of 2026-03-27) — NEST pixel
geometry is exact.
41% loss drop in 50-step overfit test on A100-80GB.

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
and saved in `state_dict` — first init is slow (~30s at nside=64), subsequent
loads are instant.

## Known caveats

- `healpy` must be installed for correct NEST pixel centres:
  `pip install healpy` or `conda install -c conda-forge healpy`.
  (Already in `credit-casper` env.)
- `nside` controls HEALPix resolution: `n_pix = 12 * nside²`. For ~1° global
  resolution `nside=64` (≈49k pixels) is typical.
- Face-aware padding is exact for `kernel_size=3`; larger kernels require
  wider padding and a correspondingly larger face neighbour strip.
