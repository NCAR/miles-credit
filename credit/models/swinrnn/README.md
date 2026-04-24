# SwinRNN

**Paper:** Hu et al., "SwinRNN: A Swin Transformer based Pure Deep Learning Rainfall-Runoff Model," JAMES 2023.
Also used as a weather backbone in multiple follow-up works.
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022MS003392

**Architecture reference:** Swin Transformer V1 (Liu et al., 2021)
https://arxiv.org/abs/2103.14030

**Original code:** https://github.com/microsoft/Swin-Transformer (MIT License)

## Architecture

U-Net encoder-decoder with **Swin Transformer blocks** (shifted window attention)
at each scale:

```
(B, C, H, W)
→ PatchEmbed (Conv2d, patch_size×patch_size)
→ Encoder stage 0: depth[0] × SwinBlock (window_size, no shift / shift)
→ PatchMerging (2× downsample, 2× channels)
→ Encoder stage 1: depth[1] × SwinBlock
→ PatchMerging
→ Encoder stage 2 (bottleneck): depth[2] × SwinBlock
→ PatchExpand (2× upsample via pixel-shuffle) + skip proj + SwinBlock decoder
→ PatchExpand + skip proj + SwinBlock decoder
→ head (Linear → rearrange) → (B, C_out, H, W)
```

`SwinBlock` alternates between regular window attention and shifted-window attention
to allow cross-window communication. Relative position bias is learned per head.

## CREDIT implementation

`swinrnn.py` — written from scratch following the Swin V1 paper and reference code.
Differences from reference:
- 3-stage encoder-decoder (not the original 4-stage classification backbone).
- `PatchExpand` uses pixel-shuffle (channel → spatial) for upsampling.
- Skip connections via channel-cat + Conv2d projection.
- Flat `(B, C, H, W)` I/O throughout.
- No pretrained ImageNet weights.

## Validation status

**Architectural smoke test only.** Not yet trained to convergence.
Window attention and shift logic written from scratch — medium confidence in
correctness of relative position bias indexing at edge cases. A short training
run with loss convergence will build confidence.

## CREDIT config

```yaml
model:
  type: swinrnn
  in_channels: 70
  out_channels: 69
  img_size: [192, 288]
  patch_size: 4          # feature map is (H/4, W/4) = (48, 72) at 1°
  embed_dim: 128
  depths: [2, 2, 6, 2]  # blocks per stage; [2,2,6,2] is the Swin-T recipe
  num_heads: [4, 8, 16, 32]
  window_size: 6         # must divide feature map at every stage
  drop_rate: 0.0
```

Window size constraint: at 1° with `patch_size=4`, stage-0 grid is 48×72.
`window_size=6` divides both. At 0.25° (720×1440) with `patch_size=4`, grid
is 180×360 and `window_size=6` still works.

## Known caveats

- `img_size` after patch embed must be divisible by `window_size` at every
  scale. With 3 stages and `patch_size=4`, effective grid is `(H/4, W/4)` at
  stage 0, `(H/8, W/8)` at stage 1, `(H/16, W/16)` at stage 2.
- `window_size` must divide the feature map at every stage.
- Memory scales as O(window_size² × N / window_size²) = O(N) — very efficient.
