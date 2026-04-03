# WXFormer — Architecture Reference

This directory contains the WXFormer v1 family of hierarchical encoder-decoder
weather prediction models built on the CrossFormer attention backbone.

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
