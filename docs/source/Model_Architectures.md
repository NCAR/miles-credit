# Supported Models

In your configuration file, you can select from multiple supported models. Below, we provide detailed information about each available architecture, including its purpose, design, and example configuration options.

## Available Models

- [WxFormer](#wxformer)
- [NCAR-FuXi](#ncar-fuxi)
- [UNet](#unet)
- [Graph Transformer](#graph-transformer)
---

## WxFormer

**WxFormer** is the flagship model developed by the MILES group at NCAR.

It is a hybrid architecture that combines a CrossFormer-based encoder with a hierarchical decoder using transpose convolutional layers. Its structure includes U-Net-like skip connections and a pyramid layout to facilitate multi-scale feature representation.

CrossFormer, the foundation of WxFormer, enables long-range dependency modeling and multi-scale spatial reasoning. This approach has demonstrated comparable performance to other vision transformers like Swin, which forms the backbone of models such as FuXi.

WxFormer is trained to predict the atmospheric state at time step *i+1*, given the state at time *i*, typically with a one-hour time increment.

For further architectural details and design motivations, see **Schreck et al., 2024**.

### References

- Schreck, John, et al. "[Community Research Earth Digital Intelligence Twin (CREDIT)](https://arxiv.org/abs/2411.07814)." *arXiv preprint arXiv:2411.07814* (2024).

### WxFormer Config Options

Below is an example configuration snippet for running **WxFormer** with CAMulator. These parameters define spatial structure, transformer layers, variable channels, and patch-level resolution.

```yaml
type: "crossformer"
frames: 1                         # number of input states (default: 1)
image_height: 192                 # number of latitude grids (default: 640)
image_width: 288                  # number of longitude grids (default: 1280)
levels: 32                        # number of upper-air variable levels (default: 15)
channels: 4                       # upper-air variable channels
surface_channels: 3               # surface variable channels
input_only_channels: 3            # dynamic forcing, forcing, static channels
output_only_channels: 15          # diagnostic variable channels

patch_width: 1                    # latitude grid size per 3D patch
patch_height: 1                   # longitude grid size per 3D patch
frame_patch_size: 1               # number of time frames per 3D patch

dim: [256, 512, 1024, 2048]       # dimensionality of each layer
depth: [2, 2, 18, 2]              # depth of each transformer block
global_window_size: [4, 4, 2, 1]  # global attention window sizes
local_window_size: 3              # local attention window size

cross_embed_kernel_sizes:         # kernel sizes for cross-embedding
  - [4, 8, 16, 32]
  - [2, 4]
  - [2, 4]
  - [2, 4]

cross_embed_strides: [2, 2, 2, 2] # cross-embedding strides
attn_dropout: 0.0                 # dropout for attention layers
ff_dropout: 0.0                   # dropout for feed-forward layers

use_spectral_norm: True
    
# use interpolation to match the output size
interp: True
    
# map boundary padding
padding_conf:
    activate: True
    mode: earth
    pad_lat: 48
    pad_lon: 48
```

## NCAR-FuXi

**NCAR-FuXi** is again describedd in  **Schreck et al., 2024**. FuXi, a state-of-the-art AI NWP model, was selected as the AI NWP model baseline for **Schreck et al., 2024**. The implementation of FuXi baseline follows its original design as in **Chen et al 2024**, but with reduced model sizes. 

### References

- Schreck, John, et al. "[Community Research Earth Digital Intelligence Twin (CREDIT)](https://arxiv.org/abs/2411.07814)." *arXiv preprint arXiv:2411.07814* (2024).
- Chen, Lei, et al. "[FuXi: A cascade machine learning forecasting system for 15-day global weather forecast.](https://www.nature.com/articles/s41612-023-00512-1)" npj climate and atmospheric science 6.1 (2023): 190.



Below is an example configuration snippet for running **NCAR-FUXI** with the ERA5 dataset. These parameters define spatial structure, transformer layers, variable channels, and patch-level resolution.

```yaml
frames: 2               # number of input states
image_height: &height 640       # number of latitude grids
image_width: &width 1280       # number of longitude grids
levels: &levels 16              # number of upper-air variable levels
channels: 4             # upper-air variable channels
surface_channels: 7     # surface variable channels
input_only_channels: 3  # dynamic forcing, forcing, static channels
output_only_channels: 0 # diagnostic variable channels

# patchify layer
patch_height: 4         # number of latitude grids in each 3D patch
patch_width: 4          # number of longitude grids in each 3D patch
frame_patch_size: 2     # number of input states in each 3D patch

# hidden layers
dim: 1024               # dimension (default: 1536)
num_groups: 32          # number of groups (default: 32)
num_heads: 8            # number of heads (default: 8)
window_size: 7          # window size (default: 7)
depth: 16               # number of swin transformers (default: 48)

use_spectral_norm: True
    
# use interpolation to match the output size
interp: True
    
# map boundary padding
padding_conf:
    activate: True
    mode: earth
    pad_lat: 48
    pad_lon: 48
```

## Graph Transformer

Arnold to add discussion.

## Unet

Add Discussion

## WxFormer Diffusion

Will to add discussion.

---

## `torch.compile` Compatibility

`torch.compile` can significantly speed up both training and inference by fusing ops and reducing Python overhead.
Not all models in the zoo are compatible — the main blocker is `torch.nn.utils.spectral_norm`.

### Why spectral norm blocks compilation

`spectral_norm` works by registering a forward hook that recomputes a normalized weight on every call.
`torch.compile`'s graph tracer cannot see through these hooks and raises a graph break (or errors out in `fullgraph=True` mode).
Spectral norm is used **intentionally** in several CREDIT models to prevent rollout explosions — do not remove it.

### Quick reference

| Model | Compiles? | What to do |
|-------|:---------:|------------|
| `wxformer` / `wxformer-sdl` / `wxformer-v2-sdl` | ⚠ | Set `upsamplePS: true` in `model:` config |
| `crossformer` | ✗ | Not supported — spectral norm in decoder |
| `fuxi` | ✗ | Not supported — spectral norm throughout |
| `swin` | ✗ | Not supported — spectral norm throughout |
| `camulator` | ✗ | Not supported — spectral norm throughout |
| `graph` | ✗ | Not supported — spectral norm throughout |
| `sfno` / `fourcastnet3` | ⚠ | Don't install `torch-harmonics`; rfft2 fallback compiles |
| `graphcast` | ⚠ | Use `torch.compile(model, dynamic=True)` |
| All others | ✓ | Works out of the box |

### WXFormer: enabling compilation with `upsamplePS`

The default WXFormer decoder uses transposed-conv upsampling with spectral norm.
Switching to the PixelShuffle upsampler removes spectral norm from the decoder entirely:

```yaml
model:
  type: wxformer
  upsamplePS: true   # ← enables PixelShuffle upsampler; required for torch.compile
```

Then compile as usual:

```python
model = torch.compile(model)
```

`upsamplePS: true` is the recommended setting for long autoregressive rollouts regardless of compilation,
as it produces a cleaner gradient signal without the norm constraint on upsampling layers.

### SFNO / FourCastNet3: disabling torch-harmonics

When `torch-harmonics` is installed, these models use Spherical Harmonic Transforms (SHTs) whose CUDA
kernels cause graph breaks under `torch.compile`. To use the rfft2 fallback (which compiles cleanly),
simply do not install `torch-harmonics` in your environment. 