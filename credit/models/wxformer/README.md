# WXFormer — Architecture Reference

This directory contains the WXFormer family of hierarchical encoder-decoder
weather prediction models built on the CrossFormer attention backbone.

---

## Files

| File | Status | Description |
|---|---|---|
| `crossformer.py` | Active (v1 baseline) | Original CrossFormer. Reference point for all ablations. |
| `wxformer_v2.py` | **Active (current best)** | Deterministic model. All new training should use this. |
| `wxformer_v2_ensemble.py` | **Active** | SDL noise ensemble built on top of v2. Load a pretrained v2 checkpoint and fine-tune the noise layers. |
| `stochastic_decomposition_layer.py` | Active | SDL module used by both ensemble classes. |
| `crossformer_ensemble.py` | Legacy | SDL ensemble for v1 only. |
| `crossformer_diffusion.py` | Active | Diffusion/score-network variant of v1. |

---

## Quick Start

### Deterministic training (wxformer_v2)

In your config set:

```yaml
model:
  type: wxformer_v2
  # all defaults are the ablation-validated best — nothing else required
  image_height: 640
  image_width: 1280
  frames: 2
  channels: 4
  levels: 15
  surface_channels: 7
  input_only_channels: 3
```

That is all. The defaults (`use_swiglu=True`, `use_shifted_windows=True`,
`use_deformable=True`, `num_residuals=4`, `upsample_with_ps=True`) are the
ablation-validated best configuration (−12.6% vs v1 at 5000 steps).

### Ensemble fine-tuning (wxformer_v2_ensemble)

1. Train a deterministic `wxformer_v2` checkpoint to convergence.
2. Switch the config type and point at the checkpoint:

```yaml
model:
  type: wxformer_v2_ensemble
  pretrained_weights: /path/to/deterministic/checkpoint.pt
  freeze: true          # freeze base model, train only SDL layers
  noise_latent_dim: 128
  encoder_noise: true
  encoder_noise_factor: 0.05
  decoder_noise_factor: 0.275
  correlated: false     # true = same noise vector across spatial positions
  # all base model architecture flags carry over unchanged
  image_height: 640
  image_width: 1280
  frames: 2
  channels: 4
  levels: 15
  surface_channels: 7
  input_only_channels: 3
```

The ensemble class loads the base weights with `strict=False` (SDL keys are
absent from the pretrained checkpoint — that is expected and not an error).
Only the SDL layers are trainable when `freeze=True`.

### Running inference with ensemble spread

Pass the same input through the model N times — each forward samples independent
noise internally:

```python
members = torch.stack([model(x) for _ in range(N)])  # (N, B, C, T, H, W)
spread  = members.std(dim=0)
```

Or set `correlated=True` in the config for a single shared noise vector per
sample (faster, less diverse).

---

## Architecture Overview

```
Input  (B, C, T, H, W)
  │
  ┌──────────────────────────────────────────────────────────┐
  │                   ENCODER  ×N_levels                     │
  │  CrossEmbedLayer  →  Transformer  →  skip to decoder     │
  │  [SDL noise injected here if ensemble]                   │
  └──────────────────────────────────────────────────────────┘
  │  bottleneck
  ┌──────────────────────────────────────────────────────────┐
  │              DECODER  ×(N_levels - 1) stages             │
  │  UpBlockPS (PixelShuffle ×2)  →  SDL?  →  cat skip       │
  └──────────────────────────────────────────────────────────┘
  │
  Final PixelShuffle head  →  (B, C_out, T_out, H, W)
```

### Input / Output channels

```
base_input_channels  = channels × levels + surface_channels + input_only_channels
input_channels       = base_input_channels × frames   (frames concat along C)
output_channels      = channels × levels + surface_channels + output_only_channels
```

ERA5 1° default: `channels=4, levels=15, surface_channels=7, input_only_channels=3`
→ `base_input_channels = 70`.

---

## All User Flags (wxformer_v2)

**The defaults are the best. You do not need to set any of these.**

```python
CrossFormer(
    # --- Decoder (defaults = ablation best) ---
    upsample_with_ps     = True,   # PixelShuffle decoder — eliminates checkerboard artifacts
    num_residuals        = 4,      # residual conv blocks per UpBlock

    # --- Encoder attention (defaults = ablation best) ---
    use_swiglu           = True,   # SwiGLU feedforward (replaces GELU MLP)
    use_shifted_windows  = True,   # Swin-style cyclic-shift attention pass
    use_deformable       = True,   # deformable attention over H/4 × W/4 K/V grid  ← new best

    # --- Off by default: opt-in experimental ---
    use_grid             = False,  # MaxViT dilated grid attention (prior best, superseded)
    use_rope             = False,  # 2D RoPE instead of DPB → FlashAttention; promising, needs more testing
    use_axial            = False,  # full-row lon + full-col lat axial attention
    use_attn_res         = False,  # inter-block attention residuals (Moonshot AI 2025; ablation pending)

    # --- Deprecated: always worse in ablations ---
    upsample_with_transformer = False,
    upsample_with_cross_attn  = False,
)
```

**To recover the original wxformer behaviour + PixelShuffle:**

```python
model = CrossFormer(use_shifted_windows=False, use_swiglu=False, use_deformable=False)
```

---

## Scaling: Depth and Number of Levels

`depth` and `dim` are tuples — one entry per encoder level. `len(dim)` sets the
number of levels. Default is 4 levels.

```python
# Deeper bottleneck only (cheapest capacity increase)
model = CrossFormer(depth=(2, 2, 16, 2))

# Five levels — verified working at 640×1280 (157M params)
# dim MUST double at every level. The conv decoder computes
# dec_dims = [last_dim // 2^i] and matches against encoder skip dims on cat.
# A non-geometric sequence (e.g. ...512, 512) will error at construction.
# global_window_size and cross_embed_* must also be 5-tuples.
model = CrossFormer(
    dim                      = (64, 128, 256, 512, 1024),
    depth                    = (2, 2, 2, 2, 2),
    cross_embed_kernel_sizes = ((4,8,16,32), (2,4), (2,4), (2,4), (2,4)),
    cross_embed_strides      = (4, 2, 2, 2, 2),
    global_window_size       = (5, 5, 2, 2, 1),
    local_window_size        = 10,
)
```

---

## SDPA Backends

All attention classes use `F.scaled_dot_product_attention`:

| Class | attn_mask | Backend |
|---|---|---|
| `Attention` (local / shifted) | DPB bias tensor | memory-efficient |
| `Attention` with `use_rope=True` | none | FlashAttention |
| `GridAttention` | none | FlashAttention |
| `DeformableAttention` | none | FlashAttention |
| `AxialAttention` | none | FlashAttention |

AMP: DPB bias is cast to `q.dtype` before the SDPA call.

---

## Ablation Results

All runs: ERA5 1°, AdamW lr=1e-4, AMP, activation checkpointing, batch 4, A100 80GB.
Loss = weighted MSE over all output channels. "final" = mean of last 10 steps.

### Decoder type (200 steps)

| Decoder | loss |
|---|---|
| ConvTranspose2d (v1 original) | 0.129 |
| PixelShuffle | 0.129 |
| CrossExpandLayer + Transformer | 0.211 |
| PixelShuffle + cross-attention skip | 0.408 |

### Encoder attention (1000 steps)

| Architecture | loss | vs v1 |
|---|---|---|
| v1 baseline | 0.05523 | — |
| SwiGLU + shifted | 0.05356 | -3.0% |
| SwiGLU + shifted + grid | 0.05363 | -2.9% |
| **SwiGLU + grid** | **0.05267** | **-4.6%** |

Shift+grid synergy not yet visible at 1000 steps — requires longer training.

### Full-scale (5000 steps)

| Architecture | loss | vs v1 |
|---|---|---|
| v1 baseline | 0.01891 | — |
| SwiGLU + shifted | 0.01750 | -7.5% |
| SwiGLU + grid | 0.01741 | -7.9% |
| SwiGLU + shifted + grid | 0.01711 | -9.5% |
| SwiGLU + shifted + grid, res=2 | 0.01708 | -9.7% |
| SwiGLU + shifted + grid, res=3 | 0.01685 | -10.9% |
| SwiGLU + shifted + grid, res=4 | 0.01668 | -11.8% |
| SwiGLU + deformable only | 0.01696 | -10.3% |
| SwiGLU + shifted + grid + RoPE | 0.01687 | -10.8% |
| **SwiGLU + shifted + deformable (DEFAULT)** | **0.01653** | **-12.6%** |

Pending: 5-level encoder (job 2848254).

### Notes on experimental flags

**`use_rope`** — replaces DPB with 2D rotary position encoding, enabling
FlashAttention for all window attention (no additive mask). Reached −10.8% at
5000 steps. Promising but not yet the default.

**`use_attn_res`** — inter-block gating (Moonshot AI, 2025). Before each
sublayer, a learned C-dim vector gates over all previous depth-block
representations via softmax-weighted sum. ~18K extra params per encoder level.
Ablation pending.

---

## Ablation Infrastructure

Runner: `applications/wxformer_v2_ablation.py`

```bash
# single variant
qsub -v VARIANT=v2+shift+deform,STEPS=5000,BATCH_SIZE=4 scripts/wxf_ablation_casper.sh

# all variants in parallel
bash scripts/submit_wxf_ablation.sh [STEPS] [BATCH_SIZE]
```

Logs: `/glade/derecho/scratch/schreck/WXF2/ablation_logs/`
