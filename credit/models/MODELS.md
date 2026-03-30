# CREDIT Model Zoo

All models share the same CREDIT training pipeline (`credit train`), config-file interface, and distributed training support.
Set `model: type:` in your config to select a model (see the **Key** column).

---

## Model Presets

Named presets let you load a pretrained model — and run a full rollout — with almost no configuration.
NCAR campaign presets bundle a **default config** (data paths, variable lists, normalization stats),
so a two-line config is enough:

```yaml
# my_config.yml  — this is all you need for a full rollout
model:
  preset: wxformer-v2-025deg-6h

save_loc: /glade/derecho/scratch/$USER/my_run
```

```bash
credit_rollout_realtime --config my_config.yml   # inference
credit_train            --config my_config.yml   # fine-tune from pretrained checkpoint
```

You can override any key — user-supplied values always win over the preset defaults:

```yaml
model:
  preset: wxformer-v2-025deg-6h
  pretrained_weights: /path/to/my/finetuned.pt   # different checkpoint
  ff_dropout: 0.1                                  # any arch param

data:
  forecast_len: 40   # override just one data key; rest come from default_config
```

### Available presets

| Preset name | Model | Resolution | Default config | Key match |
|-------------|-------|:----------:|:--------------:|:---------:|
| `wxformer-v2-025deg-6h` | WXFormer v2 CrossFormer | 0.25° 6h | ✓ | 100% |
| `wxformer-025deg-1h` | WXFormer v2 CrossFormer | 0.25° 1h | ✓ | 100% |
| `wxformer-1deg-6h` | WXFormer CrossFormer | 1° 6h | — | 100% |
| `fuxi-025deg-6h` | FuXi U-Transformer | 0.25° 6h | ✓ | 100% |
| `stormer-1.40625deg` | Stormer ViT | 1.40625° 6h | — | 194/295 (backbone) |
| `climax-1.40625deg` | ClimaX ViT | 1.40625° | — | 105/130 (backbone) |
| `aurora-0.1deg` | Aurora Perceiver3D | 0.1° 6h | — | ~74% |

**Default config ✓** — preset bundles a full working config; only `save_loc` is required from the user.

**Why partial key match for Stormer, ClimaX, Aurora?**
These models were trained outside CREDIT with their own data pipelines and variable orderings.
CREDIT assumes its own input structure (channel ordering, normalization, surface/upper-air layout),
so the input embedding and output projection layers cannot transfer directly — they are re-initialized
and must be fine-tuned on CREDIT-format ERA5 data.
The backbone transformer weights (attention, MLP, normalization layers) do transfer and provide a
strong initialization for fine-tuning.
These presets are intended as **transfer learning starting points**, not drop-in inference checkpoints.

Preset files live in [`credit/models/presets/`](presets/).
Add a new `.yml` file there to register a new preset — no code changes needed.
See [`docs/source/Model_Presets.md`](../docs/source/Model_Presets.md) for full documentation.

---

## Distributed Training Support

| Key | Model | DDP | FSDP | Act. Ckpt | `torch.compile` |
|-----|-------|:---:|:----:|:---------:|:---------------:|
| `wxformer` | WXFormer (CrossFormer backbone) | ✓ | ✓ | ✓ | ✗ spectral norm (default) |
| `wxformer-sdl` | WXFormer SDL ensemble (noise injection) | ✓ | ✓ | ✓ | ✗ spectral norm (default) |
| `wxformer-v2-sdl` | WXFormer v2 SDL ensemble | ✓ | ✓ | ✓ | ✗ spectral norm (default) |
| `crossformer` | CrossFormer (conv decoder + skip) | ✓ | ✓ | ✓ | ✗ spectral norm (default) |
| `unet` | U-Net segmentation model | ✓ | ✓ | ✓ | ✓ |
| `fuxi` | FuXi (Swin v2 backbone) | ✓ | ✓ | ✓ | ✗ spectral norm (default) |
| `swin` | Swin Transformer V2 Cr | ✓ | ✓ | ✓ | ✗ spectral norm (default) |
| `camulator` | CAMulator (CAM emulator) | ✓ | ✓ | ✓ | ✗ spectral norm (default) |
| `graph` | Graph Residual Transformer GRU | ✓ | ✓ | ✓ | ✗ spectral norm (default) |
| `stormer` | Stormer (plain ViT) | ✓ | ✓ | ✓ | ✓ |
| `climax` | ClimaX (per-variable ViT) | ✓ | ✓ | ✓ | ✓ |
| `fourcastnet` | FourCastNet v1 (AFNO ViT) | ✓ | ✓ | ✓ | ✓ |
| `sfno` | SFNO (Spherical FNO) | ✓ | ✓ | ✓ | ⚠ rfft2 fallback only |
| `swinrnn` | SwinRNN (Swin encoder-decoder RNN) | ✓ | ✓ | ✓ | ✓ |
| `fengwu` | FengWu (cross-attention ViT) | ✓ | ✓ | ✓ | ✓ |
| `graphcast` | GraphCast (icosahedral GNN) | ✓ | ✓ | ✓ | ⚠ use `dynamic=True` |
| `healpix` | DLWP-HEALPix (HEALPix U-Net) | ✓ | ✓ | ✓ | ✓ |
| `fourcastnet3` | FourCastNet v3 (spherical SNO U-Net) | ✓ | ✓ | ✓ | ⚠ rfft2 fallback only |
| `aurora` | Aurora (Perceiver3D + Swin3D) | ✓ | ✓ | ✓ | ✓ |
| `pangu` | Pangu-Weather (3D Earth Transformer) | ✓ | ✓ | ✓ | ✓ |
| `aifs` | AIFS (lat/lon Transformer) | ✓ | ✓ | ✓ | ✓ |
| `itransformer` | iTransformer (inverted variable attention) | ✓ | ✓ | ✓ | ✓ |
| `fuxi_ens` | FuXi-ENS (ViT + VAE ensemble head) | ✓ | ✓ | ✓ | ✓ |
| `arches` | ArchesWeather (window + column attention) | ✓ | ✓ | ✓ | ✓ |
| `mambavision` | MambaVision (Mamba + attention U-Net) | ✓ | ✓ | ✓ | ✓ |
| `corrdiff` | CorrDiff (score-based conditional diffusion) | ✓ | ✓ | ✓ | ✓ |

**Notes on FSDP / activation checkpointing:**
Legacy WXFormer-family models (`wxformer`, `crossformer`, `unet`, `swin`, `fuxi`) use explicit fine-grained wrap policies (attention + feedforward blocks).  All other models use automatic policy discovery — CREDIT scans the live model for repeating `nn.Module` subtypes and uses those as the wrap/checkpoint units. Pass `activation_checkpoint: true` in the `trainer:` section to enable.

**Notes on `torch.compile`:**
- **✗ spectral norm (default)** — `torch.nn.utils.spectral_norm` is enabled by default on these models to prevent rollout explosions. It wraps weight parameters with forward hooks that `torch.compile` cannot trace through. To enable compilation, set `use_spectral_norm: false` in the `model:` config block. **Only do this if you have verified long-rollout stability without it** — removing spectral norm is likely to cause divergence after tens of steps. Native compile support without spectral norm is planned for a future release.
- **⚠ rfft2 fallback only** — `sfno` and `fourcastnet3` use `torch-harmonics` Spherical Harmonic Transforms when the package is installed. The SHT CUDA kernels cause graph breaks under `torch.compile`; skip installing `torch-harmonics` to use the rfft2 fallback, which compiles without issues.
- **⚠ GraphCast `dynamic=True`** — the icosahedral graph uses `scatter_add_` with dynamic edge counts. Pass `torch.compile(model, dynamic=True)` or `fullgraph=False` to avoid excessive recompilation.

---

## Pretrained Weight Compatibility

Weights are cached at `/glade/derecho/scratch/schreck/credit_zoo/<model>/` (permanent home TBD).
Set `pretrained_weights: <path>` in the `model:` config section to load them.
`strict=False` is always used — missing keys (e.g. input/output projections for different channel counts) stay randomly initialized.

| Key | Weights Available | Load Status | Preset | What it gives you | Notes |
|-----|:-----------------:|:-----------:|:------:|-------------------|-------|
| `wxformer` | ✓ NCAR runs | ✓ 100% | `wxformer-v2-025deg-6h` `wxformer-025deg-1h` `wxformer-1deg-6h` | ERA5 backbone trained to convergence; best fine-tuning start | See `credit/models/wxformer/README.md` |
| `fuxi` | ✓ NCAR runs | ✓ 100% | `fuxi-025deg-6h` | ERA5 FuXi backbone trained to convergence (K. Friedman) | See source in `credit/models/fuxi/fuxi.py` |
| `stormer` | ✓ cached | ⚠ partial | `stormer-1.40625deg` | 194/295 keys (all 24 attn+mlp blocks); embedding and head initialized fresh | Key remap: `net.blocks.` → `model.blocks.`, `mlp.fc1/fc2` → `mlp.net.0/3`; per-variable `token_embeds` skipped |
| `climax` | ✓ cached | ⚠ partial | `climax-1.40625deg` | 105 backbone keys (norm1/2, attn, mlp) from 8 ViT blocks; pos_embed, norm, head also transfer | `net.blocks.` → `model.blocks.`; per-var `token_embeds`/`channel_embed` skipped |
| `fourcastnet` | ✓ cached | ⚠ partial | — | DDP `module.` prefix auto-remapped; AFNO blocks partially load | ~12/150 keys load cleanly; complex-weight layout differs from NERSC checkpoint |
| `sfno` | — | — | — | — | No public PyTorch weights |
| `swinrnn` | — | — | — | — | No public weights |
| `fengwu` | — | — | — | — | ONNX only |
| `graphcast` | — | — | — | — | JAX only |
| `healpix` | — | — | — | — | No confirmed public weights |
| `fourcastnet3` | ✓ cached | ✗ arch mismatch | — | — | HuggingFace checkpoint is 1D-conv SNO; our wrapper is U-Net SNO — zero key overlap |
| `aurora` | ✓ cached | ✓ ~74% | `aurora-0.1deg` | Full 1B-param Perceiver3D + Swin3D backbone; LoRA adapter layers (224 keys) initialized fresh | `net.` prefix auto-remapped to `aurora.`; missing keys are LoRA (expected) |
| `pangu` | — | — | — | — | ONNX only |
| `aifs` | — | — | — | — | Restricted access |

---

## Credits & Licenses

| Key | Original Paper | Original Code | Code License | Weights | Authors |
|-----|---------------|--------------|:------------:|---------|---------|
| `wxformer` | [CrossFormer](https://arxiv.org/abs/2108.01072) | [lucidrains/cross-former](https://github.com/lucidrains/cross-former) / [cheerss/CrossFormer](https://github.com/cheerss/CrossFormer) | MIT | — | John Schreck, William Chapman, David Gagne, Dhamma Kimpara |
| `wxformer-sdl` | [CrossFormer](https://arxiv.org/abs/2108.01072) | [lucidrains/cross-former](https://github.com/lucidrains/cross-former) / [cheerss/CrossFormer](https://github.com/cheerss/CrossFormer) | MIT | — | John Schreck, William Chapman, David Gagne, Dhamma Kimpara |
| `wxformer-v2-sdl` | [CrossFormer](https://arxiv.org/abs/2108.01072) | [lucidrains/cross-former](https://github.com/lucidrains/cross-former) / [cheerss/CrossFormer](https://github.com/cheerss/CrossFormer) | MIT | — | John Schreck, William Chapman, David Gagne, Dhamma Kimpara |
| `crossformer` | [CrossFormer](https://arxiv.org/abs/2108.01072) | [lucidrains/cross-former](https://github.com/lucidrains/cross-former) / [cheerss/CrossFormer](https://github.com/cheerss/CrossFormer) | MIT | — | Wang et al. 2021 |
| `unet` | — | [CREDIT](https://github.com/NCAR/miles-credit) | Apache-2.0 | — | John Schreck, Dhamma Kimpara, Yingkai Sha, David Gagne |
| `fuxi` | [arXiv:2306.12873](https://arxiv.org/abs/2306.12873) | [tpys/fuxi](https://github.com/tpys/fuxi) | Unspecified | — | Chen et al. 2023, Fudan / Shanghai AI |
| `swin` | [arXiv:2111.09883](https://arxiv.org/abs/2111.09883) | [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | MIT | — | Liu et al. 2022, Microsoft Research |
| `camulator` | — | [CREDIT](https://github.com/NCAR/miles-credit) | Apache-2.0 | — | William Chapman, John Schreck |
| `graph` | — | [CREDIT](https://github.com/NCAR/miles-credit) | Apache-2.0 | — | Arnold Kazadi, David Gagne, Katelyn FitzGerald, Dhamma Kimpara |
| `stormer` | [arXiv:2312.03876](https://arxiv.org/abs/2312.03876) | [tung-nd/stormer](https://github.com/tung-nd/stormer) | MIT | [HuggingFace](https://huggingface.co/tungnd/stormer) | Nguyen et al. 2023, UCLA |
| `climax` | [arXiv:2301.10343](https://arxiv.org/abs/2301.10343) | [microsoft/ClimaX](https://github.com/microsoft/ClimaX) | MIT | [HuggingFace](https://huggingface.co/tungnd/climax) | Nguyen et al. 2023, Microsoft Research / UCLA |
| `fourcastnet` | [arXiv:2202.11214](https://arxiv.org/abs/2202.11214) | [NVlabs/FourCastNet](https://github.com/NVlabs/FourCastNet) | BSD-3-Clause | [NERSC](https://portal.nersc.gov/project/m4134/FCN_weights_v0/) / [HuggingFace](https://huggingface.co/nvidia/fourcastnet1) | Pathak et al. 2022, NVIDIA |
| `sfno` | [arXiv:2306.03838](https://arxiv.org/abs/2306.03838) | [NVIDIA/makani](https://github.com/NVIDIA/makani) | Apache-2.0 | [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/sfno_73ch_small) | Bonev et al. 2023, NVIDIA |
| `swinrnn` | [arXiv:2307.09650](https://arxiv.org/abs/2307.09650) | [microsoft/SwinRNN](https://github.com/microsoft/SwinRNN) | MIT | — | Chen et al. 2023, Microsoft |
| `fengwu` | [arXiv:2304.02948](https://arxiv.org/abs/2304.02948) | [OpenEarthLab/FengWu](https://github.com/OpenEarthLab/FengWu) | CC BY-NC-SA | ONNX only | Chen et al. 2023, Shanghai AI Lab |
| `graphcast` | [arXiv:2212.12794](https://arxiv.org/abs/2212.12794) | [google-deepmind/graphcast](https://github.com/google-deepmind/graphcast) | CC BY-NC-SA | JAX (`gs://dm_graphcast`) | Lam et al. 2023, Google DeepMind |
| `healpix` | — | [CognitiveModeling/dlwp-hpx](https://github.com/CognitiveModeling/dlwp-hpx) | Apache-2.0 | — | Weyn et al., CognitiveModeling / NVIDIA |
| `fourcastnet3` | [NVIDIA Research 2025](https://research.nvidia.com/publication/2025-07_fourcastnet-3) | [NVIDIA/makani](https://github.com/NVIDIA/makani) | Apache-2.0 | [HuggingFace](https://huggingface.co/nvidia/fourcastnet3) | Kurth et al. 2025, NVIDIA |
| `aurora` | [arXiv:2405.13063](https://arxiv.org/abs/2405.13063) | [microsoft/aurora](https://github.com/microsoft/aurora) | MIT | [HuggingFace](https://huggingface.co/microsoft/aurora) | Chen et al. 2024, Microsoft Research |
| `pangu` | [Nature 2023](https://doi.org/10.1038/s41586-023-06185-3) | [198808eric/Pangu-Weather](https://github.com/198808eric/Pangu-Weather) | Non-commercial | ONNX only | Bi et al. 2023, Huawei |
| `aifs` | [arXiv:2406.01465](https://arxiv.org/abs/2406.01465) | [ecmwf/anemoi-models](https://github.com/ecmwf/anemoi-models) | Apache-2.0 | Restricted (ECMWF) | Lang et al. 2024, ECMWF |

---

## Model Details

### CREDIT-Native Models

#### WXFormer / CrossFormer family
Keys: `wxformer`, `wxformer-sdl`, `wxformer-v2-sdl`, `crossformer`, `crossformer-ensemble`

John Schreck (NCAR MILES), with contributions from William Chapman, Yingkai Sha, David Gagne, and Dhamma Kimpara.
Architecture based on [CrossFormer](https://arxiv.org/abs/2108.01072) (Wang et al. 2021); code adapted from [lucidrains/cross-former](https://github.com/lucidrains/cross-former) and [cheerss/CrossFormer](https://github.com/cheerss/CrossFormer).
Source: [`credit/models/wxformer/`](wxformer/) · [`credit/models/crossformer/`](crossformer/)

Multi-scale cross-attention transformer with a convolutional decoder and skip connections.
The SDL variants add stochastic decomposition-layer noise for ensemble generation.

#### CAMulator
Key: `camulator`

William Chapman, John Schreck (NCAR MILES).
Source: [`credit/models/camulator.py`](camulator.py)

Emulator of the Community Atmosphere Model (CAM) dynamics.

#### U-Net
Keys: `unet`, `unet_downscaling`

Source: [`credit/models/unet.py`](unet.py)

Encoder-decoder U-Net with attention modules.  Also available in downscaling variant.

#### FuXi
Key: `fuxi`

Chen et al. 2023, Fudan University / Shanghai Academy of AI.
Paper: [arXiv:2306.12873](https://arxiv.org/abs/2306.12873)
Source: [`credit/models/fuxi/`](fuxi/)
Original code: [tpys/fuxi](https://github.com/tpys/fuxi)
CREDIT integration: John Schreck, Yingkai Sha

Swin Transformer V2 backbone with U-Transformer architecture.

#### Swin Transformer V2 Cr
Key: `swin`

Liu et al. 2022.
Paper: [arXiv:2111.09883](https://arxiv.org/abs/2111.09883)
Source: [`credit/models/swin.py`](swin.py)
Original code: [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

#### Graph Residual Transformer GRU
Key: `graph`

Arnold Kazadi (NCAR MILES), with contributions from David Gagne, Katelyn FitzGerald, and Dhamma Kimpara.
Source: [`credit/models/graph.py`](graph.py)

---

### Model Zoo

#### Stormer
Key: `stormer`
Config: [`config/model_zoo/stormer.yml`](../../config/model_zoo/stormer.yml)

Nguyen et al. 2023, UCLA.
Paper: [arXiv:2312.03876](https://arxiv.org/abs/2312.03876)
Source: [`credit/models/stormer/stormer.py`](stormer/stormer.py)
Original code: [tung-nd/stormer](https://github.com/tung-nd/stormer)
Pretrained weights: [HuggingFace tungnd/stormer](https://huggingface.co/tungnd/stormer) — MIT, 69-ch, 1.40625°

Plain ViT with patch embedding and a residual prediction head.  Trained on ERA5 WeatherBench2 at 1.40625° resolution.

#### ClimaX
Key: `climax`
Config: [`config/model_zoo/climax.yml`](../../config/model_zoo/climax.yml) *(see stormer.yml as template)*

Nguyen et al. 2023, Microsoft Research / UCLA.
Paper: [arXiv:2301.10343](https://arxiv.org/abs/2301.10343)
Source: [`credit/models/climax/climax.py`](climax/climax.py)
Original code: [microsoft/ClimaX](https://github.com/microsoft/ClimaX)
Pretrained weights: [HuggingFace tungnd/climax](https://huggingface.co/tungnd/climax) — MIT, flexible channels, 5.625° / 1.40625°

Per-variable tokenization ViT pre-trained on CMIP6.  Most transfer-friendly architecture — per-variable embeddings allow selective reuse for any ERA5 variable subset.

#### FourCastNet v1
Key: `fourcastnet`
Config: [`config/model_zoo/fourcastnet.yml`](../../config/model_zoo/fourcastnet.yml)

Pathak et al. 2022, NVIDIA.
Paper: [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)
Source: [`credit/models/fourcastnet/afno.py`](fourcastnet/afno.py)
Original code: [NVlabs/FourCastNet](https://github.com/NVlabs/FourCastNet)
Pretrained weights: [NERSC portal](https://portal.nersc.gov/project/m4134/FCN_weights_v0/) / [HuggingFace nvidia/fourcastnet1](https://huggingface.co/nvidia/fourcastnet1) — BSD-3, 20-ch, 0.25°

AFNO (Adaptive Fourier Neural Operator) ViT.  Uses rfft2 for spectral mixing.

#### SFNO (FourCastNet v2)
Key: `sfno`
Config: [`config/model_zoo/sfno.yml`](../../config/model_zoo/sfno.yml)

Bonev et al. 2023, NVIDIA.
Paper: [arXiv:2306.03838](https://arxiv.org/abs/2306.03838)
Source: [`credit/models/sfno/sfno.py`](sfno/sfno.py)
Original code: [NVIDIA/makani](https://github.com/NVIDIA/makani)
Pretrained weights: [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/sfno_73ch_small) — Apache-2, 73-ch, 0.25°

Replaces rfft2 with Spherical Harmonic Transforms (SHTs) when `torch-harmonics` is installed; falls back to rfft2 otherwise.

#### SwinRNN
Key: `swinrnn`
Config: [`config/model_zoo/swinrnn.yml`](../../config/model_zoo/swinrnn.yml)

Chen et al. 2023.
Paper: [arXiv:2307.09650](https://arxiv.org/abs/2307.09650)
Source: [`credit/models/swinrnn/swinrnn.py`](swinrnn/swinrnn.py)
Original code: [microsoft/SwinRNN](https://github.com/microsoft/SwinRNN)
Pretrained weights: none publicly available

Swin Transformer encoder-decoder with recurrent connections.  Token dimensions must be divisible by `patch_size × window_size × 4`; CREDIT wrapper pads automatically.

#### FengWu
Key: `fengwu`
Config: [`config/model_zoo/fengwu.yml`](../../config/model_zoo/fengwu.yml)

Chen et al. 2023, Shanghai AI Lab.
Paper: [arXiv:2304.02948](https://arxiv.org/abs/2304.02948)
Source: [`credit/models/fengwu/fengwu.py`](fengwu/fengwu.py)
Original code: [OpenEarthLab/FengWu](https://github.com/OpenEarthLab/FengWu)
Pretrained weights: ONNX only, CC BY-NC-SA (non-commercial)

Multi-group cross-attention ViT.  Official weights are ONNX format; weight transfer into this PyTorch implementation requires manual extraction.

#### GraphCast
Key: `graphcast`
Config: [`config/model_zoo/graphcast.yml`](../../config/model_zoo/graphcast.yml)

Lam et al. 2023, Google DeepMind.
Paper: [arXiv:2212.12794](https://arxiv.org/abs/2212.12794)
Source: [`credit/models/graphcast/graphcast.py`](graphcast/graphcast.py)
Original code: [google-deepmind/graphcast](https://github.com/google-deepmind/graphcast)
Pretrained weights: `gs://dm_graphcast` — CC BY-NC-SA (non-commercial), JAX/Haiku format

kNN GNN encoder-processor-decoder on an icosahedral mesh.  Official weights are JAX; conversion step required.  CREDIT implementation is a pure-PyTorch re-implementation.

#### DLWP-HEALPix
Key: `healpix`
Config: [`config/model_zoo/healpix.yml`](../../config/model_zoo/healpix.yml)

Weyn et al. / CognitiveModeling / NVIDIA.
Source: [`credit/models/healpix/healpix.py`](healpix/healpix.py)
Original code: [CognitiveModeling/dlwp-hpx](https://github.com/CognitiveModeling/dlwp-hpx) · [NVIDIA/physicsnemo dlwp_healpix](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/dlwp_healpix)
Pretrained weights: none confirmed publicly available

HEALPix U-Net with lat/lon reprojection.  Inputs/outputs are reprojected to/from the 12-face HEALPix mesh internally.

#### FourCastNet v3
Key: `fourcastnet3`
Config: [`config/model_zoo/fourcastnet3.yml`](../../config/model_zoo/fourcastnet3.yml)

Kurth et al. 2025, NVIDIA.
Paper: [NVIDIA Research 2025](https://research.nvidia.com/publication/2025-07_fourcastnet-3)
Source: [`credit/models/fourcastnet3/fcn3.py`](fourcastnet3/fcn3.py)
Original code: [NVIDIA/makani](https://github.com/NVIDIA/makani)
Pretrained weights: [HuggingFace nvidia/fourcastnet3](https://huggingface.co/nvidia/fourcastnet3) — Apache-2, 72-ch, 0.25°, ~711M params

U-Net with Spherical Neural Operator (SNO) blocks.  Uses SHTs when `torch-harmonics` is installed.  CREDIT wrapper pads input to the nearest multiple of `2^n_stages` and crops output.

#### Aurora
Key: `aurora`
Config: see `credit/models/aurora/README.md`

Chen et al. 2024, Microsoft Research.
Paper: [arXiv:2405.13063](https://arxiv.org/abs/2405.13063)
Source: [`credit/models/aurora/model.py`](aurora/model.py)
Original code: [microsoft/aurora](https://github.com/microsoft/aurora)
Pretrained weights: [HuggingFace microsoft/aurora](https://huggingface.co/microsoft/aurora) — MIT

Perceiver3D encoder + Swin3D processor.  3D pressure-level representation.

#### Pangu-Weather
Key: `pangu`

Bi et al. 2023, Huawei.
Paper: [Nature 2023](https://doi.org/10.1038/s41586-023-06185-3)
Source: [`credit/models/pangu/pangu.py`](pangu/pangu.py)
Original code: [198808eric/Pangu-Weather](https://github.com/198808eric/Pangu-Weather)
Pretrained weights: ONNX only, non-commercial

3D Earth Transformer with hierarchical temporal aggregation.

#### AIFS
Key: `aifs`

Lang et al. 2024, ECMWF.
Paper: [arXiv:2406.01465](https://arxiv.org/abs/2406.01465)
Source: [`credit/models/aifs/aifs.py`](aifs/aifs.py)
Original code: [ecmwf/anemoi-models](https://github.com/ecmwf/anemoi-models)
Pretrained weights: restricted access (ECMWF)

Lat/lon Transformer processor; ECMWF operational NWP replacement.

---

## Adding a New Model

1. Create `credit/models/<name>/<name>.py` with a `CREDIT<Name>` wrapper class.
2. The wrapper must accept `(B, C, H, W)` or `(B, C, T, H, W)` and return `(B, C, 1, H, W)`.
3. Add `load_model` and `load_model_name` classmethods (see any zoo model for the pattern).
4. Register in `credit/models/__init__.py` under `model_types`.
5. Add a config example in `config/model_zoo/<name>.yml`.
6. Optionally add a preset in `credit/models/presets/<preset-name>.yml` — no code changes needed, just drop the file.
7. Update this file.
