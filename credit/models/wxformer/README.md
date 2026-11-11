# WXFormer — Architecture Reference

This directory contains the WXFormer v1 family of hierarchical encoder-decoder
weather prediction models built on the CrossFormer attention backbone.

The next-generation variants (WXFormerColumn and the cubed-sphere models) are
documented separately in [wxformer_family.md](wxformer_family.md).

---

## Files

| File | Status | Description |
|---|---|---|
| `crossformer.py` | Active (v1) | Original CrossFormer. Deterministic model. |
| `crossformer_ensemble.py` | Active | SDL noise ensemble built on top of v1. |
| `crossformer_diffusion.py` | Active | Diffusion/score-network variant of v1. |
| `crossformer_downscaling.py` | Active | Downscaling variant of v1. |
| `stochastic_decomposition_layer.py` | Active | SDL module used by ensemble classes. |
| `sdl_inference_wrapper.py` | Active | Inference wrapper for SDL ensemble models. |
| `wxformer_column.py` | Active (next) | WXFormerColumn: CrossFormer U-Net + level embed + column attn + spectral GNN. See [wxformer_family.md](wxformer_family.md). |
| `cubed_wxformer.py` | Active (next) | CubedWXFormer: WXFormer on a cubed-sphere SE grid, with optional `use_column_attn` (level embed + column attn + spectral GNN). See [wxformer_family.md](wxformer_family.md). |

---

## Quick Start

### Deterministic training

In your config set:

```yaml
model:
  type: wxformer
  image_height: 640
  image_width: 1280
  frames: 2
  channels: 4
  levels: 15
  surface_channels: 7
  input_only_channels: 3
```

### Ensemble fine-tuning

1. Train a deterministic `wxformer` checkpoint to convergence.
2. Switch the config type:

```yaml
model:
  type: crossformer-ensemble
  pretrained_weights: /path/to/deterministic/checkpoint.pt
  freeze: true
  noise_latent_dim: 128
  image_height: 640
  image_width: 1280
  frames: 2
  channels: 4
  levels: 15
  surface_channels: 7
  input_only_channels: 3
```
