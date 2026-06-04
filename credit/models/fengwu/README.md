# FengWu

**Paper:** Chen et al., "FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead," 2023.
https://arxiv.org/abs/2304.02948

**Original code:** Shanghai AI Lab / Shanghai Jiao Tong University / Microsoft Research Asia.
No official public repo from authors. Architecture reference from:
NVIDIA PhysicsNeMo — https://github.com/NVIDIA/physicsnemo (Apache 2.0)

## Architecture

Key innovation: **hierarchical variable grouping with cross-group fusion**.
Variables are split into groups (e.g., Z-levels, T-levels, UV-levels, surface);
each group gets its own ViT encoder/decoder; a cross-attention fuser lets
groups attend to each other.

```
Input (B, C, H, W) — split along channel axis into G groups

Per-group encoder (shared weights within group):
  (B, g_c, H, W) → PatchEmbed → pos_embed → depth × SelfAttnBlock → (B, N, D)

Cross-group fuser (fuser_depth layers):
  Each group i queries all other groups j≠i via CrossAttnBlock,
  then refines via SelfAttnBlock.

Per-group decoder:
  (B, N, D) → depth × SelfAttnBlock → Linear head → rearrange → (B, g_c_out, H, W)

Concatenate group outputs → (B, C_out, H, W)
```

With `group_sizes=[in_channels]` (default), degrades to a standard ViT.

## CREDIT implementation

`fengwu.py` — written from scratch following paper description and PhysicsNeMo.
Differences from reference:
- Flat `(B, C, H, W)` I/O.
- User specifies `group_sizes` as a list of ints summing to `in_channels`.
- Output group sizes mirror input; last group trimmed/padded to match `out_channels`.
- Cross-group fuser is asymmetric: `n_groups-1` CrossAttnBlocks per layer
  (last group only gets SelfAttn). This is handled explicitly in the forward.
- No pretrained weights.

## Validation status

**Architectural smoke test only.** Not yet trained to convergence.
Cross-group fusion logic is non-trivial — written from paper description rather
than reference weights. Loss convergence will validate.

## CREDIT config

```yaml
model:
  type: fengwu
  in_channels: 70
  out_channels: 69
  img_size: [192, 288]
  patch_size: 4
  embed_dim: 512
  encoder_depth: 6
  fuser_depth: 4
  decoder_depth: 6
  num_heads: 8
  # Optional: split variables into groups
  # group_sizes: [25, 25, 20]  # must sum to in_channels
  # Default (no group_sizes): single group = plain ViT, no cross-attention
```

Without `group_sizes`, this is identical to a plain ViT (fuser is trivially skipped).
Set `group_sizes` to e.g. `[levels*4, 4]` to separate upper-air from surface variables.

## Known caveats

- `sum(group_sizes)` must equal `in_channels`.
- Memory scales with number of groups and cross-attention sequence length
  `(G-1)*N`. For large G, fuser can be expensive.
- Default `group_sizes=None` → single group → plain ViT (valid, just no fusion).
