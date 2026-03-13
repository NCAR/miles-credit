# WXFormer v2 — Architecture Reference

WXFormer v2 is an enhanced PyTorch implementation of the CrossFormer weather prediction
model. It builds on the original WXFormer (CrossFormer) architecture with eight targeted
improvements, a Mixture-of-Experts feedforward, and integrates with the production
components in [miles-credit](https://github.com/NCAR/miles-credit).

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
   - [Input / Output Channels](#input--output-channels)
   - [Temporal Aggregator](#temporal-aggregator)
   - [Encoder](#encoder)
   - [Bottleneck: Global Register Tokens](#bottleneck-global-register-tokens)
   - [Decoder](#decoder)
   - [Output Heads](#output-heads)
3. [v2 Architectural Improvements](#v2-architectural-improvements)
   - [ConvFFN (Depthwise Spatial Mixing)](#convffn-depthwise-spatial-mixing)
   - [DPB Caching](#dpb-caching)
   - [Gated Skip Connections](#gated-skip-connections)
   - [Variable-Group Output Heads](#variable-group-output-heads)
   - [Temporal Aggregator](#temporal-aggregator-1)
   - [Anisotropic Window Sizes](#anisotropic-window-sizes)
   - [Global Register Tokens](#global-register-tokens)
   - [Decoder Transformer Blocks](#decoder-transformer-blocks)
4. [Mixture of Experts (MoE)](#mixture-of-experts-moe)
5. [Production Components (miles-credit)](#production-components-miles-credit)
   - [CubeEmbedding](#cubeembedding)
   - [TensorPadding](#tensorpadding)
   - [StochasticDecompositionLayer (SDL)](#stochasticdecompositionlayer-sdl)
   - [PostBlock](#postblock)
   - [SKEBS](#skebs)
   - [Ensemble and Diffusion Variants](#ensemble-and-diffusion-variants)
6. [Key Hyperparameters](#key-hyperparameters)
7. [Training Recipe](#training-recipe)

---

## Overview

WXFormer uses a hierarchical encoder–decoder architecture with alternating
short-window (local) and long-window (global) self-attention, cross-scale patch
embedding, and skip connections. The v2 file (`wxformer_v2.py`) is a standalone
reference implementation that can be directly imported into the miles-credit
training framework.

```
Input (B, C, T, H, W)
  │
  ▼
TemporalAggregator        ← learned inter-frame mixing (new in v2)
  │
  ┌──────────────────────────────────────────────┐
  │                   ENCODER                   │
  │  ┌────────────────────────────────────────┐  │
  │  │ CrossEmbedLayer  →  TransformerWithMoE │  │  ×4 levels
  │  └────────────────────────────────────────┘  │
  └──────────────────────────────────────────────┘
  │                        │  skip connections (gated, new in v2)
  ▼
GlobalRegister             ← bottleneck global context (new in v2)
  │
  ┌──────────────────────────────────────────────┐
  │                   DECODER                   │
  │  UpBlock + DecoderBlock + gated cat skip    │  ×3 levels
  └──────────────────────────────────────────────┘
  │
  ▼
Split Output Heads (atm / surface)  ← variable-group heads (new in v2)
  │
  ▼
Bilinear interpolation → (B, C_out, T_out, H, W)
```

---

## Model Architecture

### Input / Output Channels

```
base_input_channels  = channels × levels + surface_channels + input_only_channels
input_channels       = base_input_channels × frames          (concat over time)

base_output_channels = channels × levels + surface_channels + output_only_channels
output_channels      = base_output_channels × output_frames
```

Typical ERA5 configuration: `channels=4`, `levels=15`, `surface_channels=7`,
`input_only_channels=3`, giving `base_input_channels = 70`.

### Temporal Aggregator

Before the encoder, the `T` input frames are concatenated along the channel axis and
passed through a `TemporalAggregator` — a gated grouped Conv2d that learns how to mix
each physical variable across its `T` time steps independently:

```
x_flat  (B, C*T, H, W)
  └──> LayerNorm
        ├──> mixer (Conv2d, groups=C)  ──> GELU ─┐
        └──> gate  (Conv2d, groups=C)  ──> sigmoid ┤  multiply
                                                    └──> + residual
```

This is a drop-in replacement for a raw `reshape`, with the same output shape.

### Encoder

Four encoder stages, each consisting of:

1. **CrossEmbedLayer** — multi-scale 2-D patch embedding via parallel convolutions with
   different kernel sizes (e.g. `[2, 4]`, `[4, 8, 16, 32]`). Each kernel captures a
   different receptive field; outputs are concatenated to form `dim_out` channels.

2. **TransformerWithMoE** — `depth` repeated blocks, each containing:
   ```
   Short-Window Attention  (local receptive field)
   MoEFeedForward          (spatial gating, top-k routing)
   Long-Window Attention   (dilated / strided, global receptive field)
   MoEFeedForward
   ```
   Skip connections are stored after each stage for use in the decoder.

### Bottleneck: Global Register Tokens

After the deepest encoder stage a `GlobalRegister` module is applied:

- A small bank of `num_tokens` learnable tokens cross-attends to **all** spatial
  positions in the bottleneck feature map.
- The token outputs are mean-pooled and broadcast back as an additive bias.
- This gives the network genuine planetary-scale ("teleconnection") awareness that
  windowed attention alone cannot provide.

### Decoder

Three UpBlock stages, each followed by a lightweight DecoderBlock:

```
x = UpBlock(x)               # 2× upsample (ConvTranspose2d or PixelShuffle)
x = DecoderBlock(x)          # short-window attention + ConvFFN (new in v2)
x = cat([x, gate(skip)])     # gated encoder skip connection (new in v2)
```

Two decoder variants are available via `upsample_with_ps`:

| `upsample_with_ps` | UpBlock type | Final head |
|---|---|---|
| `False` (default) | ConvTranspose2d ×2 stride | ConvTranspose2d ×2 stride |
| `True` | PixelShuffle ×2 + sharpening conv | PixelShuffle ×2 |

### Output Heads

The decoder output is split into two separate projection heads:

- **`out_head_atm`** — projects to `channels × levels × output_frames` channels
  (pressure-level variables: wind, temperature, humidity, geopotential)
- **`out_head_sfc`** — projects to `(surface_channels + output_only_channels) × output_frames`
  channels (surface variables: 2-m temp, 10-m winds, MSLP, etc.)

The two outputs are concatenated and bilinearly interpolated back to `(image_height, image_width)`.

---

## v2 Architectural Improvements

### ConvFFN (Depthwise Spatial Mixing)

The standard two-layer MLP feedforward is replaced with a three-layer
**ConvFFN** that adds a depthwise 3×3 convolution between the channel-expand and
GELU steps:

```python
nn.Conv2d(dim, inner, 1)                         # channel expand
nn.Conv2d(inner, inner, 3, padding=1, groups=inner)  # depthwise spatial mix
nn.GELU()
nn.Conv2d(inner, dim, 1)                         # project back
```

The depthwise step mixes information across the 3×3 neighborhood within each
channel, adding local spatial inductive bias at low cost. Every expert in the MoE
layer also uses this structure.

### DPB Caching

`DynamicPositionBias` is an MLP applied to relative position coordinates to produce
attention bias matrices. At inference time these biases are deterministic, so
`Attention` caches the result in `_dpb_cache` on the first eval forward pass.
The cache is invalidated whenever `model.train()` is called, since DPB weights
may change during training steps.

### Gated Skip Connections

Encoder features are modulated by a learned sigmoid gate before being concatenated
in the decoder:

```python
x = cat([x, sigmoid(gate_conv(skip)) * skip], dim=1)
```

Each gate is a 1×1 Conv2d. This allows the model to learn how much of each
encoder feature is relevant at each decoder stage, rather than always passing
the raw skip through.

### Variable-Group Output Heads

Instead of a single output projection from decoder channels → all output channels,
v2 uses two separate heads (`out_head_atm`, `out_head_sfc`) so that atmospheric
pressure-level variables and surface variables each have their own dedicated
projection weights. This costs the same number of parameters but removes
cross-group competition in the final layer.

### Temporal Aggregator

See [Temporal Aggregator](#temporal-aggregator) above. The key insight is that
inter-frame relationships (e.g. advective tendency, surface pressure tendency) are
variable-specific; using grouped convolutions enforces this inductive bias.

### Anisotropic Window Sizes

`local_window_size` and `global_window_size` now accept `(H, W)` tuples in addition
to scalars, enabling different window extents in the latitude vs. longitude direction.
This is physically motivated: ERA5 grids are not square (1° latitude ≈ 111 km;
1° longitude varies), and storm dynamics are often elongated zonally.

The `norm_window_sizes()` helper disambiguates:

| Input | Interpretation |
|---|---|
| `int` | Same square window at every level |
| `(H, W)` | Same anisotropic window at every level |
| `(s0,s1,s2,s3)` | Per-level square windows |
| `((H0,W0),…)` | Per-level anisotropic windows |

### Global Register Tokens

See [Bottleneck: Global Register Tokens](#bottleneck-global-register-tokens) above.
Inspired by the "register tokens" concept from vision transformers, these tokens
accumulate global statistics that the encoder can then condition on when producing
the deepest feature representation.

### Decoder Transformer Blocks

A lightweight `DecoderBlock` (short-window attention + ConvFFN, FF mult=2) is applied
after each `UpBlock`. This lets the decoder refine spatial details and reconcile
skip-connection features before the next upsampling stage, at modest extra cost.

---

## Mixture of Experts (MoE)

Every feedforward sublayer in the encoder transformers is replaced by
`MoEFeedForward`, a spatially-gated mixture of `num_experts` independent
`Expert` networks (each with the ConvFFN structure).

### Routing

```
x  (B, C, H, W)
  └──> gate Conv2d(C → num_experts)
        + Gaussian noise (training only, std=moe_noise_std)
        ──> softmax ──> top-k selection
```

Each spatial position independently selects its `top_k` experts. Expert outputs
are weighted by the renormalised top-k probabilities and summed.

All experts operate on the full `(B, C, H, W)` input in parallel; only the
weighted selection happens per-position. This is memory-heavier than token-routing
but preserves the spatial structure essential for grid-based weather data.

### Load-Balancing Loss (Switch Transformer)

```
L_aux = num_experts × Σ_i  f_i · p_i
```

where `f_i` is the fraction of spatial positions dispatched to expert `i` (hard
top-k routing) and `p_i` is the mean soft routing probability for expert `i`.
Gradients flow through `p_i` to the gate weights, encouraging uniform load.

```python
# In training loop:
y_pred  = model(x)
task_loss = criterion(y_pred, y_true)
aux_loss  = model.get_load_balancing_loss()   # sums across all levels & layers
loss = task_loss + 1e-2 * aux_loss
loss.backward()
```

### Per-Level MoE Configuration

`num_experts` and `top_k` can be 4-tuples to apply different routing budgets at
each encoder level:

```python
# Example: more experts at deeper, higher-dim levels
num_experts=(4, 4, 8, 8),
top_k=(2, 2, 2, 1),
```

---

## Production Components (miles-credit)

The following components live in the `credit/` subtree of miles-credit and are
not included in `wxformer_v2.py`. They are activated via config dicts passed to
the production `CrossFormer` wrapper.

### CubeEmbedding

```python
class CubeEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm)
```

A `Conv3d`-based patch embedding over `(T, Lat, Lon)`. Projects the full
spatio-temporal input volume with a 3-D convolution whose kernel and stride equal
`patch_size`, then applies `LayerNorm` over the resulting tokens.

Activated in production `CrossFormer.forward` when `patch_height > 1` or
`patch_width > 1`. The standalone v2 always uses the 2-D `CrossEmbedLayer` path.

### TensorPadding

```python
class TensorPadding:
    def __init__(self, mode="earth", pad_lat=(40,40), pad_lon=(40,40))
```

Pads the input tensor before the encoder to reduce boundary artefacts.

| Mode | Latitude (N/S poles) | Longitude (E/W wrap) |
|---|---|---|
| `"earth"` | 180°-shifted flip (physically correct pole reflection) | Circular |
| `"mirror"` | Reflect | Circular |

`pad()` / `unpad()` operate on tensors of shape `(B, C, T, Lat, Lon)`.
Activated via `padding_conf` in the production config.

### StochasticDecompositionLayer (SDL)

```python
class StochasticDecompositionLayer(nn.Module):
    def __init__(self, noise_dim, feature_channels, noise_factor=0.1)
    def forward(self, feature_map, noise):  # noise: (B, noise_dim)
```

Injects structured, learned noise into intermediate feature maps:

1. A latent noise vector `noise ∈ ℝ^{noise_dim}` is linearly projected to
   per-channel "style" weights `∈ ℝ^{feature_channels}`.
2. Independent pixel-wise Gaussian noise `∈ ℝ^{B×C×H×W}` is scaled by
   `noise_factor` (fixed) and modulated by the style weights and a learnable
   per-channel scale `modulation`.
3. The result is added to `feature_map` as a residual.

Used in `CrossFormerWithNoise` at both encoder and decoder stages to generate
diverse ensemble members from a single deterministic base model.

### PostBlock

```python
class PostBlock(nn.Module):
    def __init__(self, post_conf: dict)
    def forward(self, pred_dict: dict) -> dict  # {"y_pred": tensor, "x": input}
```

A configurable chain of post-model physics corrections applied after the neural
network output. Active sub-modules (controlled by `post_conf`):

| Module | Purpose |
|---|---|
| **TracerFixer** | Clamp tracers (specific humidity, cloud water) to physical bounds |
| **GlobalMassFixer** | Enforce global dry-air and water mass conservation |
| **GlobalWaterFixer** | Enforce global water budget via precipitation adjustment |
| **GlobalEnergyFixer** | Enforce global energy balance via temperature correction |
| **SKEBS** | Stochastic wind perturbations for ensemble spread (see below) |

### SKEBS

**Stochastic Kinetic Energy Backscatter Scheme.** Generates physically consistent
wind perturbations by:

1. Predicting a per-level backscatter rate `D(x,y,z,t)` via a configurable neural
   network (`dissipation_type` ∈ `{"prescribed", "uniform", "FCNN", "CNN", "unet"}`).
2. Maintaining a temporally-correlated spectral random pattern via an AR(1) process:
   ```
   coef_new = (1 - α) × coef + √α × g_n × noise
   ```
   where `α` is a learnable decorrelation coefficient and `g_n` is a Gaussian
   spectral envelope.
3. Converting spectral streamfunction coefficients to grid-space wind perturbations
   `(u_chi, v_chi)` via inverse vector spherical harmonic transform.
4. Scaling perturbations to conserve kinetic energy: `u_perturb = √(r·D/ΔE) × u_chi`.
5. Adding `u_perturb`/`v_perturb` to the U/V wind channels.

SKEBS trainable parameters include `α`, `variance`, the spectral pattern filter,
the spectral backscatter filter, and the backscatter prediction network weights.

### Ensemble and Diffusion Variants

#### CrossFormerWithNoise

Subclass of `CrossFormer` that adds `StochasticDecompositionLayer` instances at
three encoder levels and three decoder levels. The base model weights can be frozen
(`freeze=True`) so only the SDL parameters are trained, enabling cheaply generating
diverse ensemble members from a pre-trained deterministic model.

Key args: `noise_latent_dim`, `encoder_noise_factor`, `decoder_noise_factor`,
`encoder_noise` (bool), `freeze` (bool), `correlated` (bool).

#### CrossFormerDiffusion

Score-network variant for diffusion-model-based ensemble generation.
Extends `CrossFormer` with:
- **Conditioning:** concatenates the clean weather state `x_cond` to the noisy
  input before encoding.
- **Self-conditioning:** optionally concatenates the previous denoising step's
  output to the input.
- **Time-step embedding:** a 2-layer MLP `(1 → dim[0] → dim[0])` produces a
  time embedding injected additively at each encoder level and used for
  scale-shift modulation in each `UpBlock`.

#### rk4() method

The production `CrossFormer` also supports 4th-order Runge-Kutta rollout for more
accurate multi-step forecasts. At each of the four stages the single-frame model
output is concatenated with the last input frame to form a 2-frame input for the
next stage:

```python
k1 = forward(x)                          # ŷ at step i
k2 = forward(x + 0.5 * cat([x[:,-2:-1], k1]))
k3 = forward(x + 0.5 * cat([x[:,-2:-1], k2]))
k4 = forward(x +       cat([x[:,-2:-1], k3]))
return (k1 + 2*k2 + 2*k3 + k4) / 6
```

---

## Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `dim` | `(64,128,256,512)` | Channel widths at each encoder level |
| `depth` | `(2,2,8,2)` | Transformer layers per level |
| `local_window_size` | `10` | Short-window size (or `(H,W)` for anisotropic) |
| `global_window_size` | `(5,5,2,1)` | Long-window size per level |
| `dim_head` | `32` | Attention head dimension |
| `cross_embed_strides` | `(4,2,2,2)` | Downsampling factor at each CEL |
| `global_register_tokens` | `8` | 0 to disable |
| `dec_window_size` | `4` | Decoder attention window size |
| `num_experts` | `4` | MoE experts (scalar or 4-tuple) |
| `top_k` | `2` | Active experts per position (scalar or 4-tuple) |
| `moe_noise_std` | `0.1` | Gate noise std during training |
| `upsample_with_ps` | `False` | `True` → pixel-shuffle decoder |

---

## Training Recipe

```python
model = CrossFormer(**cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for x, y_true in dataloader:
    optimizer.zero_grad()

    y_pred = model(x)

    # Task loss (e.g. latitude-weighted MSE)
    task_loss = criterion(y_pred, y_true)

    # MoE load-balancing auxiliary loss
    aux_loss = model.get_load_balancing_loss()

    loss = task_loss + 1e-2 * aux_loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

The `1e-2` coefficient for the auxiliary loss keeps the load-balancing term from
dominating. A good heuristic is to tune it so `1e-2 * aux_loss ≈ 0.01–0.1 × task_loss`
at the start of training.
