# Supported Models

Set `model: type:` in your config to select a model.  A complete reference — including distributed training support, pretrained weight availability, licenses, and detailed architecture notes — lives in [`credit/models/MODELS.md`](../../credit/models/MODELS.md).

---

## Available Models

### CREDIT-Native

| Key | Description |
|-----|-------------|
| `wxformer` | WXFormer (CrossFormer backbone, conv decoder, skip connections) |
| `wxformer-sdl` | WXFormer SDL ensemble — noise injection at each transformer scale |
| `crossformer` | CrossFormer with conv decoder head |
| `fuxi` | FuXi — Swin Transformer V2 U-Transformer |
| `unet` | U-Net segmentation model |
| `camulator` | CAMulator — CAM atmospheric emulator |
| `graph` | Graph Residual Transformer GRU |
| `swin` | Swin Transformer V2 Cr |

### Model Zoo

| Key | Description | Paper |
|-----|-------------|-------|
| `stormer` | Stormer — plain ViT weather model | [arXiv:2312.03876](https://arxiv.org/abs/2312.03876) |
| `climax` | ClimaX — per-variable tokenization ViT | [arXiv:2301.10343](https://arxiv.org/abs/2301.10343) |
| `fourcastnet` | FourCastNet v1 — AFNO ViT | [arXiv:2202.11214](https://arxiv.org/abs/2202.11214) |
| `sfno` | SFNO — Spherical Fourier Neural Operator | [arXiv:2306.03838](https://arxiv.org/abs/2306.03838) |
| `swinrnn` | SwinRNN — Swin encoder-decoder with recurrent connections | [arXiv:2307.09650](https://arxiv.org/abs/2307.09650) |
| `fengwu` | FengWu — multi-group cross-attention ViT | [arXiv:2304.02948](https://arxiv.org/abs/2304.02948) |
| `graphcast` | GraphCast — icosahedral GNN encoder-processor-decoder | [arXiv:2212.12794](https://arxiv.org/abs/2212.12794) |
| `healpix` | DLWP-HEALPix — HEALPix U-Net with lat/lon reprojection | — |
| `fourcastnet3` | FourCastNet v3 — spherical SNO U-Net | [NVIDIA 2025](https://research.nvidia.com/publication/2025-07_fourcastnet-3) |
| `aurora` | Aurora — Perceiver3D + Swin3D backbone | [arXiv:2405.13063](https://arxiv.org/abs/2405.13063) |
| `pangu` | Pangu-Weather — 3D Earth Transformer | [Nature 2023](https://doi.org/10.1038/s41586-023-06185-3) |
| `aifs` | AIFS — lat/lon Transformer processor (ECMWF) | [arXiv:2406.01465](https://arxiv.org/abs/2406.01465) |
| `itransformer` | iTransformer — inverted variable attention | [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) |
| `fuxi_ens` | FuXi-ENS — ViT + VAE ensemble perturbation head | [arXiv:2405.05925](https://arxiv.org/abs/2405.05925) |
| `arches` | ArchesWeather — window + column attention | [arXiv:2405.14527](https://arxiv.org/abs/2405.14527) |
| `mambavision` | MambaVision — hybrid Mamba + attention U-Net | [arXiv:2407.08083](https://arxiv.org/abs/2407.08083) |
| `corrdiff` | CorrDiff — score-based conditional diffusion | [arXiv:2309.15214](https://arxiv.org/abs/2309.15214) |
| `nextgen_wxformer` | NextGen WXFormer — CrossFormer U-Net + spectral GNN bottleneck + column attention | — |

---

## WXFormer Config Example

```yaml
model:
  type: wxformer
  frames: 1
  image_height: 640
  image_width: 1280
  levels: 16
  channels: 4
  surface_channels: 7
  input_only_channels: 3
  output_only_channels: 0
  patch_width: 1
  patch_height: 1
  frame_patch_size: 1
  dim: [128, 256, 512, 1024]
  depth: [2, 2, 8, 2]
  global_window_size: [4, 4, 2, 1]
  local_window_size: 3
  cross_embed_kernel_sizes:
    - [4, 8, 16, 32]
    - [2, 4]
    - [2, 4]
    - [2, 4]
  cross_embed_strides: [2, 2, 2, 2]
  use_spectral_norm: true
  interp: true
```

## FuXi Config Example

```yaml
model:
  type: fuxi
  frames: 2
  image_height: 640
  image_width: 1280
  levels: 16
  channels: 4
  surface_channels: 7
  input_only_channels: 3
  output_only_channels: 0
  patch_height: 4
  patch_width: 4
  frame_patch_size: 2
  dim: 1024
  num_groups: 32
  num_heads: 8
  window_size: 7
  depth: 16
  use_spectral_norm: true
  interp: true
```

---

## `torch.compile` Compatibility

`torch.compile` can significantly speed up training and inference.
Not all models are compatible — the main blocker is `torch.nn.utils.spectral_norm`.

### Quick reference

| Model | Compiles? | What to do |
|-------|:---------:|------------|
| `wxformer` / `wxformer-sdl` / `crossformer` / `fuxi` / `swin` / `camulator` / `graph` / `nextgen_wxformer` | ✗ default | Set `use_spectral_norm: false` in `model:` config |
| `sfno` / `fourcastnet3` | ⚠ | Don't install `torch-harmonics`; rfft2 fallback compiles |
| `graphcast` | ⚠ | Use `torch.compile(model, dynamic=True)` |
| All others | ✓ | Works out of the box |

### Why spectral norm blocks compilation

`spectral_norm` registers a forward hook that recomputes a normalized weight on every call.
`torch.compile`'s graph tracer cannot trace through these hooks.
Spectral norm is used intentionally to prevent rollout explosions — do not remove it without testing long rollouts.

```yaml
model:
  type: wxformer
  use_spectral_norm: false   # required for torch.compile; verify stability first
```

### SFNO / FourCastNet3

When `torch-harmonics` is installed these models use Spherical Harmonic Transform CUDA kernels that cause graph breaks under `torch.compile`. To use the rfft2 fallback (which compiles cleanly), do not install `torch-harmonics`.
