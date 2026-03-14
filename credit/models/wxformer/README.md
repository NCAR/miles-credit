# WXFormer — Architecture Reference

This directory contains the WXFormer family of hierarchical encoder-decoder
weather prediction models built on the CrossFormer attention backbone.

---

## File Status

| File | Status | Description |
|---|---|---|
| `crossformer.py` | **Active (v1 baseline)** | Original CrossFormer. Reference point for all ablations. |
| `wxformer_v3.py` | **Active (current best)** | v1 + 2D-DPB + PixelShuffle + SwiGLU + shifted/grid attention + SDPA. Being promoted to `wxformer_v2.py`. |
| `wxformer_v2.py` | **Deprecated** | MoE + TemporalAgg + GlobalRegister + DecoderAttn experiment. All components underperformed v1 in ablations; MoE diverged. Will be removed. |
| `wxformer_v2.md` | Historical | Architecture reference for the deprecated v2 experiment. |
| `wxformer_moe.py` | Archive | Standalone MoE prototype. |
| `crossformer_diffusion.py` | Active | Diffusion/score-network variant of v1. |
| `crossformer_ensemble.py` | Active | SDL-noise ensemble variant of v1. |

**Planned rename:** `wxformer_v3.py` → `wxformer_v2.py` once the decoder residual
sweep is complete.

---

## Architecture Overview (wxformer_v3 / current best)

The model follows a U-Net-style hierarchical encoder-decoder. The hierarchical
structure is intentional and load-bearing: it enables multi-scale noise injection
in future ensemble and diffusion variants.

```
Input  (B, C, T, H, W)
  │
  │  [optional TensorPadding — handled by production wrapper]
  │
  ┌──────────────────────────────────────────────────────────┐
  │                      ENCODER  ×4 levels                 │
  │  CrossEmbedLayer  →  Transformer  →  skip to decoder    │
  └──────────────────────────────────────────────────────────┘
  │  bottleneck features
  ┌──────────────────────────────────────────────────────────┐
  │               DECODER  ×(N_levels - 1) stages           │
  │  UpBlockPS (PixelShuffle ×2)  →  cat skip  →  residuals │
  └──────────────────────────────────────────────────────────┘
  │
  Final PixelShuffle head  →  (B, C_out, T_out, H, W)
```

### Input / Output

```
base_input_channels  = channels × levels + surface_channels + input_only_channels
input_channels       = base_input_channels × frames   (frames concat'd along C)

base_output_channels = channels × levels + surface_channels + output_only_channels
```

Typical ERA5 1° config: `channels=4, levels=15, surface_channels=7,
input_only_channels=3` → `base_input_channels = 70`.

### CrossEmbedLayer

Multi-scale 2-D patch embedding via parallel convolutions with different kernel
sizes (e.g. `[4, 8, 16, 32]`). Each kernel captures a different spatial receptive
field; outputs are channel-concatenated. This is the primary downsampling operator
at each encoder stage.

### Transformer Attention

Each Transformer block contains pairs of attention + feedforward sublayers.
The attention type at each pair is controlled by flags (see [User Flags](#user-flags)):

```
Short-window (local) Attention  →  FeedForward
Second-pass Attention           →  FeedForward
```

The second-pass attention type depends on the enabled flags:

| Flags | Pattern | Sublayers/depth |
|---|---|---|
| all off | local + long-range window | 4 |
| `use_shifted_windows` | local + cyclic-shifted local | 4 |
| `use_grid` | local + grid (MaxViT) | 4 |
| `use_shifted_windows + use_grid` | local + shifted + grid | 6 |
| `use_axial` | local + lon-axial + lat-axial | 6 |

#### 2D Dynamic Position Bias (DPB)

Replaces the 1D relative position bias of v1. An MLP maps 2D relative position
offsets `(Δh, Δw)` to per-head scalar attention biases, shared across all
depth steps at a given encoder level. Cached at eval time for zero overhead.
SDPA (`F.scaled_dot_product_attention`) passes DPB as `attn_mask`, enabling the
memory-efficient backend.

#### Shifted Window Attention (Swin-style)

Cyclic roll by `(wsz//2, wsz//2)` before window partition, un-roll after.
Physically correct for the periodic longitude boundary. No masking required
because `TensorPadding` handles the poles.

#### Grid Attention (MaxViT-style)

Each within-window spatial position `(i, j)` attends to all other windows at the
same grid position — a dilated / strided global attention with O(nh*nw) sequence
length rather than O(H*W). 2D sin/cos positional encoding marks each window's
`(row, col)` grid location:

```
Row PE:  sinusoidal(nh, C//2)   — repeated for each window column
Col PE:  sinusoidal(nw, C//2)   — repeated for each window row
Combined:  (nh, nw, C)  →  (nh*nw, C)  added to token embeddings
```

Uses pure SDPA (no additive mask) → FlashAttention backend.

#### SwiGLU Feedforward

Drop-in replacement for the standard GELU MLP:

```
FeedForwardSwiGLU:
  x → LayerNorm
    → Conv2d(dim, 2*hidden, 1)  → chunk(2) → gate, val
    → SiLU(gate) * val
    → Conv2d(hidden, dim, 1)

hidden = int(dim * mult * 2/3)   # matches GELU-MLP parameter count
```

### PixelShuffle Decoder (UpBlockPS)

**Default and strongly recommended** over ConvTranspose2d. Eliminates checkerboard
artifacts by shuffling channels into spatial resolution at low resolution, then
applying a sharpening residual:

```
x → Conv2d(in_ch, out_ch * scale², 3)  →  PixelShuffle(scale)  →  sharp_residual
  → [num_residuals × (Conv2d + GroupNorm + SiLU)]  →  + shortcut
```

`num_residuals` controls decoder capacity (2 = default, sweep in progress for 3, 4).

---

## User Flags

```python
CrossFormer(
    # Decoder
    upsample_with_ps     = True,   # ALWAYS use this — resolves checkerboard
    num_residuals        = 2,      # residual blocks per decoder UpBlock

    # Encoder attention (all False = OG wxformer behaviour)
    use_swiglu           = False,  # SwiGLU feedforward instead of GELU MLP
    use_shifted_windows  = False,  # Swin-style cyclic-shift second attention pass
    use_grid             = False,  # MaxViT-style dilated grid attention
    use_axial            = False,  # full-row (lon) + full-col (lat) axial attention

    # Experimental / deprecated
    upsample_with_transformer = False,  # CrossExpandLayer + Transformer decoder (worse)
    upsample_with_cross_attn  = False,  # PixelShuffle + cross-attn skip fusion (worse)
    lat_file             = None,        # path to NetCDF for sin/cos lat embedding
)
```

**Recovery of OG wxformer + PixelShuffle** (all optional flags off):

```python
model = CrossFormer(upsample_with_ps=True)
```

**Current best configuration** (v3+shift+grid):

```python
model = CrossFormer(
    upsample_with_ps     = True,
    use_swiglu           = True,
    use_shifted_windows  = True,
    use_grid             = True,
)
```

---

## SDPA Throughout

All three attention classes use `F.scaled_dot_product_attention`:

| Class | attn_mask | Backend |
|---|---|---|
| `Attention` (local/shifted) | DPB bias tensor | Memory-efficient |
| `GridAttention` | none | FlashAttention |
| `AxialAttention` | none | FlashAttention |
| `CrossAttention` (deprecated decoder) | none | FlashAttention |

AMP compatibility: DPB bias is cast to `q.dtype` before the SDPA call.

---

## Ablation Results

All runs use the production ERA5 1° config, AdamW lr=1e-4, AMP, activation
checkpointing, batch size 4, A100 80GB. Loss = MSE over all output channels.
"final" = mean of last 10 steps.

### Decoder type (200 steps, dim_scale=0.25)

Early test establishing PixelShuffle as the decoder baseline.

| Decoder | loss |
|---|---|
| ConvTranspose2d (v1 original) | 0.129 |
| PixelShuffle (v3 baseline) | 0.129 |
| CrossExpandLayer + Transformer | 0.211 |
| PixelShuffle + cross-attention skip | 0.408 |

Conclusion: conv decoders are optimal for spatial reconstruction; transformer
decoders introduce inductive bias mismatch. PixelShuffle matches ConvTranspose2d
at 200 steps and eliminates checkerboard at inference.

### Encoder attention variants (200 steps, dim_scale=0.25)

| Variant | loss | vs v1 |
|---|---|---|
| v1 baseline | 0.129 | — |
| v3 + SwiGLU only | 0.138 | worse |
| v3 + shifted only | 0.134 | worse |
| v3 + SwiGLU + shifted | 0.128 | **-0.8%** |
| v3 + SwiGLU + grid | 0.137 | neutral |
| v3 + SwiGLU + axial | 0.137 | neutral |

Synergy between SwiGLU and shifted windows is consistent with Swin-v2 design
rationale. Short 200-step proxy is too noisy (~±0.018) to discriminate marginal
improvements; longer runs needed.

### Attention variants (1000 steps, dim_scale=0.25)

| Variant | loss | vs v1 |
|---|---|---|
| v1 baseline | 0.05523 | — |
| v3 (swiglu+shifted) | 0.05356 | -3.0% |
| **v3+grid** | **0.05267** | **-4.6%** |
| v3+shift+grid | 0.05363 | -2.9% |

At 1000 steps grid alone beat the combined variant — shift+grid synergy requires
longer training to emerge (analogous to swiglu+shift at 200 steps).

### Full-scale run (5000 steps, full dims, batch 4, A100 80GB)

| Variant | loss (last 10) | vs v1 |
|---|---|---|
| v1 baseline | 0.01891 | — |
| v3 (swiglu+shifted) | 0.01750 | -7.5% |
| v3+grid | 0.01741 | -7.9% |
| **v3+shift+grid** | **0.01711** | **-9.5%** |

Shift+grid synergy is confirmed at 5000 steps. The 6-sublayer pattern
(local → shifted → grid → FF each) is the strongest single-encoder configuration
tested. This is the new recommended default.

### v2 component ablation (5000 steps, full dims) — COMPLETED

Tests of the wxformer_v2.py components (MoE, TemporalAgg, GlobalRegister,
DecoderAttn) run alongside the v3 sweep.

| Variant | final loss | vs v1 | notes |
|---|---|---|---|
| v1 baseline | 0.01891 | — | reference |
| v2_full (all on) | 0.04077 | +115% | MoE routing unstable; loss peaked >600 before partial recovery |
| v2_no_moe | 0.02023 | +7.0% | all other v2 features, no MoE |
| v2_no_register | 0.02137 | +13.0% | MoE + all except GlobalRegister |
| v2_no_decoder_attn | 0.02177 | +15.1% | MoE + all except DecoderAttn |

All v2 variants are worse than v1 at 5000 steps. Removing MoE (`v2_no_moe`) is
the least bad at 0.02023 — still 7% above v1 and far behind v3+shift+grid (0.01711).
The MoE routing noise combined with the Switch Transformer aux loss destabilizes
training; removing other components makes it worse, not better, indicating they
add optimization difficulty without corresponding benefit at this training length.

**Conclusion: wxformer_v2.py will be retired. Its components do not benefit the
5000-step proxy and the MoE diverges in the fully-enabled configuration.**

### Decoder residual depth sweep (5000 steps) — COMPLETED

Fixing encoder to v3+shift+grid+swiglu (confirmed best), sweeping decoder
residual depth.

| Variant | num_residuals | loss (last 10) | vs res2 | vs v1 |
|---|---|---|---|---|
| v3+sg+res2 | 2 | 0.01708 | — | -9.7% |
| v3+sg+res3 | 3 | 0.01685 | -1.3% | -10.9% |
| **v3+sg+res4** | **4** | **0.01668** | **-2.3%** | **-11.8%** |

Monotonically improving. Each extra residual block adds ~49K params and
consistently reduces loss. **`num_residuals=4` is the new default.**

---

## Refactor — COMPLETED

1. `wxformer_v2.py` — overwritten with promoted v3 content (deprecated MoE version gone)
2. `wxformer_v3.py` — kept as identical copy (backward-compatible alias)
3. `credit/models/__init__.py` — added `"wxformer_v2"` key; `"wxformer_v3"` kept as alias
4. New defaults in `CrossFormer.__init__`:
   - `upsample_with_ps=True` (always on)
   - `use_swiglu=False`, `use_shifted_windows=False`, `use_grid=False` (user opts in)
   - `num_residuals=4` (ablation best)
5. OG + PS baseline: `CrossFormer()` with all flags at defaults
6. Best config: `CrossFormer(use_swiglu=True, use_shifted_windows=True, use_grid=True)`

---

## Ablation Infrastructure

The ablation runner lives at `applications/wxformer_v2_ablation.py`.

**Single variant:**

```bash
qsub -v VARIANT=v3+shift+grid,STEPS=5000,BATCH_SIZE=4 \
     scripts/wxf_ablation_casper.sh
```

**All variants in parallel (one A100 80GB job each):**

```bash
bash scripts/submit_wxf_ablation.sh [STEPS] [BATCH_SIZE]
```

Logs write to `/glade/derecho/scratch/schreck/WXF2/ablation_logs/`.

Available `--variant` choices:
`v1`, `v3`, `v3+grid`, `v3+shift+grid`,
`v2_full`, `v2_no_moe`, `v2_no_register`, `v2_no_decoder_attn`,
`v3+sg+res2`, `v3+sg+res3`, `v3+sg+res4`
