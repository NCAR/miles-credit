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