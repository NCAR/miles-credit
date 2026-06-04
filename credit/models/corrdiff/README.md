# CorrDiff

**Paper:** Mardani et al., "Residual Diffusion Modeling for Km-scale Atmospheric Downscaling," NeurIPS 2023 / ICLR 2024.
https://arxiv.org/abs/2309.15214

**Original code:** https://github.com/NVIDIA/physicsnemo (Apache-2.0, NVIDIA)

## Architecture

Score-based conditional diffusion model using EDM preconditioning (Karras et al. 2022):

```
x_cond (coarse) → CondEncoder → multi-scale features
x_noisy         ─┐
sigma            ─┤→ SongUNet (denoising backbone) → D_x (denoised)
cond features   ─┘
```

**CondEncoder:** U-Net encoder that extracts multi-scale feature maps from the coarse input.

**SongUNet:** Residual U-Net with GroupNorm, SiLU activations, and optional attention at
deep levels. Noise level `σ` is embedded via Fourier features and injected into every
ResBlock via adaptive layer norm (scale + shift).

**EDM preconditioning:** `D_x = c_skip * x_noisy + c_out * F_x(c_in * x_noisy, σ, cond)`

**Sampling:** `model.sample(x_cond)` runs the full Heun's 2nd-order ODE solver (18 steps
by default) to generate a sample.

## Use cases

- **Statistical downscaling:** coarse global forecast → fine regional analysis
- **Probabilistic output:** run `model.sample()` multiple times with the same `x_cond`
  to generate ensemble members
- **Single-resolution noise injection:** `scale_factor=1` to generate perturbations
  around a deterministic forecast

## CREDIT config

```yaml
model:
  type: corrdiff
  in_channels: 80       # coarse conditioning channels
  out_channels: 84      # fine output channels
  img_size: [192, 288]  # fine-resolution (H, W)
  scale_factor: 1       # 1 = same resolution; >1 for true downscaling
  base_ch: 128
  ch_mult: [1, 2, 4]
  n_res_per_level: 2
  emb_dim: 256
  sigma_data: 0.5
```

## Known caveats

- `forward()` performs a single denoising step at σ=1 (for training / smoke test).
  Call `model.sample(x_cond)` for full generative inference.
- Training requires a score-matching loss; the standard CREDIT MSE trainer will work
  as an approximation but an EDM-style loss is recommended for full quality.
- Memory scales with `base_ch²` and `ch_mult`; reduce for large spatial domains.
