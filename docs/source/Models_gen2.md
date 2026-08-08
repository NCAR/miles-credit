# Models

CREDIT models are neural networks that take the atmosphere (or ocean/climate)
state at one time and predict the state one step later. Running the model
repeatedly on its own output produces a forecast or a long climate simulation.
Every model is selected purely by a `model.type` key in the config, so switching
architectures is a one-line change.

This page describes the current **Generation 2** models in plain terms —
what each one does, the major changes from the CREDIT v1 (arXiv 2024) release,
and an annotated example config. For the exact constructor arguments, follow the
AutoAPI link in each section.

## How variables become model channels

All CREDIT models share one convention that is worth understanding before
reading any config. The nested variable dictionary from the dataset is flattened
into a single tensor whose **channels** are grouped by role, and the config tells
the model how many channels fall into each group:

| Config key | Meaning | Example variables |
|---|---|---|
| `channels` | 3D (upper-air) prognostic variables — predicted *and* fed back | wind `u`/`v`, temperature `T`, humidity `q` |
| `surface_channels` | 2D (single-level) prognostic variables | surface pressure, 2 m temperature |
| `input_only_channels` | forcings and static fields the model reads but never predicts | solar radiation, land–sea mask, terrain |
| `output_only_channels` | diagnostic variables the model predicts but is never given | precipitation, top-of-atmosphere fluxes |
| `levels` | number of vertical levels for each 3D variable | 18, 32, 137, … |
| `frames` | number of input time steps handed to the model at once | usually 1 |
| `image_height` / `image_width` | latitude × longitude grid size | 181 × 360 for 1° ERA5 |

A 3D variable contributes `levels` channels; a 2D variable contributes one. The
totals must match the data — `credit check` verifies this for you, so run it
before training.

Two more concepts recur in every architecture:

- **Boundary padding (`padding_conf`)** — the Earth has no edges, but a grid
  does. `mode: earth` wraps longitude around the dateline and reflects across
  the poles so the model sees a seamless globe instead of hard borders. The pad
  widths (`pad_lat`, `pad_lon`) are also chosen so the padded grid divides
  cleanly through every downsampling stage (see the comments in the examples).
- **Postblocks do the physics** — the network predicts normalized values; the
  conversion back to physical units, conservation of mass/energy, and derived
  diagnostics all happen *after* the model in the postblock chain (see
  [Postblocks](postblocks_gen2.md)), not inside the network. This is a
  deliberate Gen 2 separation: the same backbone can be trained with or without
  physics constraints just by editing the config.

---

## WXFormer

**AutoAPI:** {py:obj}`credit.models.wxformer.crossformer.CrossFormer`

**Config type:** `wxformer` (alias `wxformer_base`)

WXFormer is the flagship MILES/NCAR model. It is a hierarchical
**encoder–decoder** built on the **CrossFormer** attention backbone:

- The **encoder** looks at the map at several resolutions at once. Its
  cross-scale attention mixes information from both nearby grid cells (local
  windows) and far-away ones (a strided *global* window), so a single layer can
  relate, say, a developing storm to the large-scale flow steering it. Four
  stages progressively coarsen the grid while widening the feature dimension
  (`dim`), like zooming out on a weather map while tracking more per location.
- The **decoder** rebuilds the full-resolution field, using **U-Net-style skip
  connections** so fine detail captured early is not lost during the coarsening.

The model predicts the state at step *i+1* from step *i*. The default cadence in
the reference configs is one to six hours.

### What changed from CREDIT v1

The architecture is the same CrossFormer lineage as Schreck et al. (2024), but
the Gen 2 version adds several robustness and quality improvements, all
controlled from the config:

- **Spherical boundary padding** (`padding_conf: {mode: earth}`) removes the
  artificial seams at the dateline and poles that a plain grid introduces.
- **Checkerboard-free upsampling** (`upsample_with_ps: True`) replaces plain
  transposed convolutions with ICNR-initialized *pixel shuffle*, eliminating the
  grid-scale "checkerboard" artifacts those layers are prone to. A final light
  bilinear resize (`interp: True`) further suppresses any residual pattern and
  lands the output exactly on the target grid.
- **Spectral normalization** (`use_spectral_norm: True`) caps the amplification
  each layer can apply, which stabilizes long autoregressive rollouts.
- **Explicit variable typing** — the four channel counts above let the model
  treat forcings, statics, prognostics, and diagnostics correctly instead of
  lumping them together.
- **Physics moved to postblocks** — conservation and diagnostics are now
  composable postblock stages rather than being hard-wired into the model.

### NextGen WXFormer

**AutoAPI:** {py:obj}`credit.models.wxformer.wxformer_next.NextGenWXFormer` &nbsp;·&nbsp; **Config type:** `nextgen_wxformer`

An experimental next-generation variant keeps the same CrossFormer U-Net but
adds three physically motivated pieces:

- a **spectral graph-neural-network bottleneck** that gives every location direct
  access to a small set of global modes (`num_spectral_nodes`) — a learned analog
  of teleconnections, so distant regions can influence each other in one step;
- **column attention**, which couples the vertical levels at each grid point so
  the model reasons about the atmospheric column as a whole;
- **pressure-level embeddings** that tell the network which vertical level each
  channel belongs to.

It accepts the same core arguments as WXFormer plus `num_spectral_nodes`,
`col_attn_heads`, and `col_attn_stride` (set `col_attn_stride: 8` on large grids
like 640×1280 to keep attention memory bounded).

### Example config with pointers

```yaml
model:
  type: "wxformer"
  frames: 1               # one input time step

  # --- Grid and channel layout (must match the data) ---
  image_height: 181       # latitude cells (1° ERA5)
  image_width: 360        # longitude cells
  levels: 18              # vertical levels per 3D variable
  channels: 4             # 3D prognostic: u, v, T, q
  surface_channels: 4     # 2D prognostic: SP, 2T, 10U, 10V
  input_only_channels: 4  # dynamic forcing (2) + static (2)
  output_only_channels: 8 # diagnostics the model outputs but never sees as input

  # --- Model capacity: the main knob for size vs. skill vs. cost ---
  dim: [32, 64, 128, 256] # feature width per encoder stage; MUST be a pyramid
                          # (dim[0] has to divide dim[-1] // 8, or it won't build)
  depth: [2, 2, 8, 2]     # number of transformer blocks per stage

  # --- Attention windows (see the divisibility note below) ---
  global_window_size: [8, 4, 2, 1]  # long-range attention stride per stage
  local_window_size: 4              # short-range window (shared across stages)
  cross_embed_kernel_sizes:         # multi-scale patch kernels per stage
    - [4, 8, 16, 32]
    - [2, 4]
    - [2, 4]
    - [2, 4]
  cross_embed_strides: [2, 2, 2, 2] # downsampling factor per stage

  # --- Gen 2 quality/robustness switches ---
  interp: True            # final bilinear resize; mild low-pass, exact grid match
  use_spectral_norm: True # stabilizes long rollouts
  upsample_with_ps: True  # pixel-shuffle decoder; avoids checkerboard artifacts

  # --- Spherical padding: no seams, and keeps stage grids divisible ---
  padding_conf:
    activate: True
    mode: earth           # wrap longitude, reflect across poles
    pad_lat: [37, 38]
    pad_lon: [12, 12]
```

**The parameters that matter most:**

- **Channel counts** (`channels`, `surface_channels`, `input_only_channels`,
  `output_only_channels`, `levels`) — these *must* equal what the data provides.
  This is the most common source of "it won't start" errors; `credit check`
  catches them.
- **`dim` / `depth`** — your main lever for model size. Bigger = more skill but
  more memory and slower. `dim` must be a **pyramid** (each stage no smaller than
  the last, and `dim[0]` divides `dim[-1] // 8`).
- **Window sizes and padding divisibility** — after four stride-2 downsamples,
  every stage's height and width must be divisible by both `local_window_size`
  and that stage's `global_window_size`. The `pad_lat`/`pad_lon` values are
  chosen to guarantee this (the examples show the arithmetic in comments). If you
  change the grid, re-check the math or `credit check` will flag it.
- **`padding_conf.mode: earth`** — keep this on for global models; it is the
  difference between a seamless globe and visible artifacts at the edges.

WXFormer also has ensemble and diffusion variants that reuse this backbone —
`crossformer-ensemble` (noise-injection ensembles) and `crossformer-diffusion`
(a score/diffusion model). See [Ensemble Training](Ensembles.md).

---

## CAMulator

**AutoAPI:** {py:obj}`credit.models.camulator.Camulator` &nbsp;·&nbsp; **Config type:** `camulator`

CAMulator is a **climate emulator**: it uses the same CrossFormer encoder–decoder
as WXFormer but is tuned to emulate a climate model (NCAR's CAM) rather than
forecast weather. In practice it is a WXFormer configured for a coarser grid, a
longer time step, and many diagnostic outputs, and it is almost always paired
with the **conservation-fixer postblocks** (mass, water, energy) so the emulated
climate stays physically consistent over long runs.

The config is nearly identical to WXFormer's. The differences you will typically
see are a coarser `image_height`/`image_width`, more `levels`, a large
`output_only_channels` (many diagnostics), and heavier `dim`/`depth`.

```yaml
model:
  type: "camulator"
  frames: 1
  image_height: 192         # coarser CAM grid
  image_width: 288
  levels: 32                # 3D prognostic: U V T Qtot
  channels: 4
  surface_channels: 2       # 2D prognostic: PS, TREFHT
  input_only_channels: 6    # dynamic forcing (4) + static (2)
  output_only_channels: 17  # many climate diagnostics

  dim: [256, 512, 1024, 2048]   # larger than the weather example
  depth: [2, 2, 18, 2]
  global_window_size: [4, 4, 2, 1]
  local_window_size: 3
  cross_embed_kernel_sizes:
    - [4, 8, 16, 32]
    - [2, 4]
    - [2, 4]
    - [2, 4]
  cross_embed_strides: [2, 2, 2, 2]

  use_spectral_norm: True
  interp: True
  padding_conf:
    activate: True
    mode: earth
    pad_lat: [48, 48]
    pad_lon: [48, 48]
```

**Pointers:** the same channel-count and `dim`-pyramid rules as WXFormer apply.
The most important companion setting lives in the **postblocks**, not here — pair
CAMulator with `tracer_fixer`, `global_mass_fixer`, `global_water_fixer`, and
`global_energy_fixer` to conserve physical budgets across long climate rollouts
(see [Postblocks](postblocks_gen2.md)).

---

## Swin Transformer

**AutoAPI:** {py:obj}`credit.models.swin.SwinTransformerV2Cr` &nbsp;·&nbsp; **Config type:** `swin`

A compact **Swin Transformer V2** backbone — the same family of hierarchical
"shifted-window" vision transformer that underpins models like FuXi. Instead of
CrossFormer's cross-scale windows, Swin attends within local windows that shift
between layers so information gradually spreads across the map. It is a good
lightweight alternative backbone and useful for comparison studies.

Note the different parameter names: Swin uses `img_size` (a `[height, width]`
pair), `embed_dim`, `depths`, and `num_heads` in place of WXFormer's
`dim`/`depth`, and `img_window_ratio` to set the window size relative to the
grid. The shared channel-count keys are the same.

```yaml
model:
  type: "swin"
  img_size: [640, 1280]
  patch_size: 4
  embed_dim: 768
  depths: [12]
  num_heads: [8]
  img_window_ratio: 80      # window size = img_size / ratio

  levels: 15
  frames: 1
  channels: 4
  surface_channels: 7
  input_only_channels: 3
  output_only_channels: 0

  drop_path_rate: 0.1       # stochastic depth regularization
  use_spectral_norm: True
  interp: True              # resize output to match the grid
  padding_conf:
    activate: True
    mode: earth
    pad_lat: 80
    pad_lon: 80
```

**Pointers:** `embed_dim`, `depths`, and `num_heads` set capacity;
`img_window_ratio` controls how large each attention window is (smaller ratio →
larger windows → more global context, more memory). Keep `padding_conf` on for
global grids.

---

## U-Net

**AutoAPI:** {py:obj}`credit.models.unet.SegmentationModel` &nbsp;·&nbsp; **Config type:** `unet`

A convolutional **U-Net** built on the `segmentation-models-pytorch` library. It
is a pure convolutional encoder–decoder with skip connections — no attention —
which makes it fast, memory-light, and a strong, simple baseline. The encoder can
be any backbone that library supports (e.g. a ResNet), optionally with pretrained
weights.

The config is minimal: point `architecture` at a supported decoder/encoder pair.
An optional `rk4_integration: True` wraps the network in a 4th-order Runge–Kutta
time step for smoother integration.

```yaml
model:
  type: "unet"
  image_height: 640
  image_width: 1280
  frames: 2
  levels: 15
  channels: 4
  surface_channels: 7
  input_only_channels: 3
  output_only_channels: 0

  rk4_integration: False
  architecture:
    name: "unet"              # unet, unet++, manet, fpn, deeplabv3+, …
    encoder_name: "resnet34"  # any segmentation-models-pytorch encoder
    encoder_weights: "imagenet"
```

**Pointers:** `architecture.name` picks the decoder family and
`architecture.encoder_name` the backbone; these are the main capacity/quality
levers. Use `encoder_weights: "imagenet"` to warm-start from pretrained weights,
or `null` to train from scratch. Turn on `rk4_integration` if you want a more
stable time-stepping scheme at extra cost.

---

## Choosing a model

- **WXFormer** — the default choice for weather forecasting; best skill, most
  actively developed.
- **NextGen WXFormer** — experimental; try it when global teleconnections or
  vertical coupling matter and you can afford the extra cost.
- **CAMulator** — climate emulation with conservation constraints.
- **Swin** — a lighter transformer backbone for comparison or constrained
  hardware.
- **U-Net** — the fastest, simplest baseline; no attention.

Other registered backbones (FuXi, a graph transformer, and downscaling and WRF
variants) are available as `model.type` keys; see the API reference for their
arguments.
