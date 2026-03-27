# SFNO — Spherical Fourier Neural Operator

**Paper:** Bonev et al., "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere," ICML 2023.
https://arxiv.org/abs/2306.03838

**Original code:** NVIDIA PhysicsNeMo — https://github.com/NVIDIA/physicsnemo (Apache 2.0)
Also: https://github.com/NVlabs/torch-harmonics

## Architecture

Extends FNO to the sphere using **Spherical Harmonic Transforms (SHT)** instead
of FFT, making the spectral filter globally equivariant under rotations.

```
x (B, C, H, W)
→ PatchEmbed (Conv2d)
→ N × SFNOBlock
    spectral branch:  LayerNorm → SHT → complex weight multiply → ISHT
    spatial branch:   pointwise skip
    MLP branch:       LayerNorm → Linear → GELU → Linear
→ head → (B, C_out, H, W)
```

When `torch-harmonics` is not installed, falls back to `rfft2`/`irfft2`
(same as a standard FNO — loses spherical equivariance but still trains).

## CREDIT implementation

`sfno.py` — written from scratch following the paper and PhysicsNeMo source.
Differences from reference:
- Flat `(B, C, H, W)` I/O via patch embed/head.
- Channel-last layout `(B, H, W, D)` during SFNO stages (matches AFNO convention).
- `n_modes` defaults to `min(H, W//2+1)` (Nyquist).
- No pretrained weights.

## Validation status

**Architectural smoke test only.** Not yet trained to convergence.
Tested with rfft2 fallback (torch-harmonics not in credit-casper env).
Install `torch-harmonics` for true spherical equivariance.

## Known caveats

- Install: `pip install torch-harmonics` (requires CUDA-compatible build).
- Without torch-harmonics the model degrades to a standard FNO — still useful
  but loses the spherical equivariance property.
- `n_modes` should be ≤ `min(H, W//2+1)` to respect Nyquist.
