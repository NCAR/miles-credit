# FourCastNet v1 (AFNO)

**Paper:** Pathak et al., "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators," 2022.
https://arxiv.org/abs/2202.11214

**Original code:** https://github.com/NVlabs/FourCastNet (BSD 3-Clause)
Also in NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo

## Architecture

Replaces self-attention with **AFNO token mixing** in Fourier space:

```
x → 2D FFT → divide into n_blocks → complex 2-layer MLP per block
  → soft-threshold (sparsity) → IFFT → residual
```

Overall model:
```
(B, C, H, W) → PatchEmbed → pos_embed → N × AFNOBlock → head → (B, C_out, H, W)
```

Each `AFNOBlock` = AFNOLayer (Fourier mixer) + pointwise MLP, both with residual.
Input layout is channel-last `(B, H, W, D)` during the transformer stages.

## CREDIT implementation

`afno.py` — written from scratch following paper + NVIDIA reference.

**Bug fixed during implementation:** The original einsum subscripts used `b`
twice (`bhwnb`) causing a PyTorch RuntimeError. Corrected to use distinct
letters `s` (block_size) and `e` (hidden) throughout.

Differences from reference:
- Flat `(B, C, H, W)` I/O.
- No positional encoding beyond the patch embed (reference uses fixed 2D sinusoidal).
- No pretrained weights.

## Validation status

**Architectural smoke test only.** Not yet trained to convergence.
The AFNO forward/backward is confirmed correct after the einsum fix.

## CREDIT config

```yaml
model:
  type: fourcastnet
  in_channels: 70
  out_channels: 69
  img_size: [192, 288]
  patch_size: 8          # FourCastNet v1 uses patch_size=8 at 0.25°; 4–8 at 1°
  embed_dim: 768
  depth: 12
  n_blocks: 8            # must divide embed_dim; default 8
  drop_rate: 0.0
```

## Known caveats

- `n_blocks` must divide `embed_dim`. Default `n_blocks=8`.
- AFNO is global (all-to-all in Fourier space) but cheap — O(N log N).
- `hidden_size_factor` controls the Fourier-domain MLP width; keep at 1
  unless you have memory to spare.
