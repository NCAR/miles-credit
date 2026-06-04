# FourCastNet3 (FCN3)

**Paper:** Kurth et al., "FourCastNet 3: Geometric Deep Learning for Fast and Accurate Global Weather Forecasting," 2025.
https://research.nvidia.com/publication/2025-07_fourcastnet-3

**Original code:** NVIDIA PhysicsNeMo — https://github.com/NVIDIA/physicsnemo (Apache 2.0)

## Architecture

**U-Net encoder-decoder with Spherical Neural Operator (SNO) blocks at each scale.**
The "geometric" aspect: SHTs are used at every scale with `lmax` scaled to the
local resolution, preserving spherical equivariance throughout the U-Net.

```
(B, C, H, W)
→ stem (Conv2d)
→ Encoder: n_stages × (SNOBlocks + SNODown 2×)
→ Bottleneck: SNOBlocks
→ Decoder: n_stages × (SNOUp 2× + skip_proj + SNOBlocks)
→ head (Conv2d) → (B, C_out, H, W)
```

Channel dims double at each encoder stage: `[base_dim, 2×, 4×, ...]`
Spectral modes halve: `[n_modes, n_modes//2, n_modes//4, ...]` (min 4)

`SNOBlock` = GroupNorm + SpectralConv2d (spectral branch) + pointwise skip + Conv2d MLP.
`SpectralConv2d` = SHT → complex weight → ISHT (rfft2 fallback without torch-harmonics).
GroupNorm (not LayerNorm) for better rotation equivariance.

## Differences from SFNO

| | SFNO | FCN3 |
|---|---|---|
| Structure | flat transformer | U-Net encoder-decoder |
| Normalisation | LayerNorm | GroupNorm |
| Spectral modes | fixed | halved per stage |
| Skip connections | none | yes (cat + proj) |

## CREDIT implementation

`fcn3.py` — written from scratch following the paper and PhysicsNeMo source.
Differences from reference:
- Flat `(B, C, H, W)` I/O.
- Channel-first layout `(B, D, H, W)` throughout (PhysicsNeMo uses channel-last
  in some ops).
- Fallback to rfft2 when torch-harmonics is absent.
- No pretrained weights.

## Validation status

**Architectural smoke test only.** Not yet trained to convergence.
U-Net structure and SNO blocks confirmed correct (shape, no NaN, gradients).

## CREDIT config

```yaml
model:
  type: fourcastnet3
  in_channels: 70
  out_channels: 69
  img_size: [192, 288]
  base_dim: 128          # channels at stage 0; doubles per stage
  depth: 2               # SNOBlocks per encoder/decoder stage
  n_stages: 3            # U-Net depth (H/W must be divisible by 2^n_stages)
  n_modes: 96            # spectral truncation; defaults to min(H, W//2+1)
  drop_rate: 0.0
```

Install `torch-harmonics` for true SHT (spherical equivariance):
`pip install torch-harmonics`. rfft2 fallback is automatic and still trains.

## Known caveats

- `H` and `W` must be divisible by `2^n_stages` for the U-Net downsampling.
- Install torch-harmonics for true SHT: `pip install torch-harmonics`
- With rfft2 fallback the model is still valid but not spherically equivariant.
