# MambaVision

**Paper:** Hatamizadeh & Kautz, "MambaVision: A Hybrid Mamba-Transformer Vision Backbone," CVPR 2025.
https://arxiv.org/abs/2407.08083

**Original code:** https://github.com/NVlabs/MambaVision (Apache-2.0, NVlabs)

## Architecture

4-stage hierarchical encoder-decoder (U-Net style):

```
(B, C, H, W) → Stem Conv
  → Stage 0-1: ResidualConvBlocks   (fast local feature extraction)
  → Stage 2-3: MambaAttentionBlocks (SSM mixing + self-attention)
  → Progressive upsampling with skip connections
  → 1×1 Conv head → (B, C_out, H, W)
```

Each `MambaAttentionBlock` applies:
1. SSM (Mamba selective state-space model) for sequence mixing along flattened spatial tokens
2. Standard multi-head self-attention
3. FFN

Channel count doubles at each downsampling stage: `stem_dim → 2× → 4× → 8×`.

## SSM backend

- If `mamba_ssm` is installed (requires CUDA ≥ 11.8): uses the real selective SSM.
- Otherwise: falls back to a pure-PyTorch depthwise-conv + gating approximation.

Install the real SSM: `pip install mamba_ssm causal_conv1d`

## CREDIT config

```yaml
model:
  type: mambavision
  in_channels: 80
  out_channels: 84
  img_size: [192, 288]
  stem_dim: 96          # base channel count; doubles per stage
  stage_depths: [2, 2, 8, 2]
  num_heads: 8
  mlp_ratio: 4.0
```

## Known caveats

- `img_size` must be divisible by 8 (3 downsampling stages).
- `num_heads` must divide `stem_dim × 4` (stage 2) and `stem_dim × 8` (stage 3).
- With `mamba_ssm` installed, `torch.compile` is supported. Without it (pure-PyTorch
  fallback), compilation also works.
