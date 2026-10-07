# CREDIT Ensemble Methods

The CREDIT framework implements two primary approaches for generating probabilistic forecasts: **noise-injection ensembles** and **diffusion-based generation**. Both methods enable the creation of stochastic models that capture forecast uncertainty through different architectural and training strategies.

## Training Non-Deterministic Model Ensembles

CREDIT supports two primary training approaches for ensemble generation:

1. **Fine-tuning approach**: Pre-trained deterministic models fine-tuned with noise-injection layers using CRPS loss
2. **Diffusion training**: Training from scratch with diffusion models using latitude-weighted MSE loss

The fine-tuning approach is currently preferred due to computational efficiency and resource requirements.

### Configuration

```yaml
trainer:
    type: era5  # or era5-ensemble
    ensemble_size: 8
    batch_size: 4
loss:
    type: KCRPS
```

## Noise-Injection Ensembles

### Architecture Overview

CREDIT's noise-injection approach utilizes the `CrossFormerWithNoise` model, which extends pretrained CrossFormer models with specialized `PixelNoiseInjection` layers. The implementation introduces stochasticity at multiple stages of the encoder-decoder pipeline while preserving learned representations from the base model.

#### Key Components:

**PixelNoiseInjection Module:**
- Injects per-pixel, per-channel noise into feature maps
- Uses learnable modulation parameters and style transformations
- Supports noise scheduling based on forecast step
- Combines latent noise vectors with spatial noise patterns

**CrossFormerWithNoise Architecture:**
- Extends base CrossFormer with noise injection capabilities
- Supports both encoder and decoder noise injection
- Implements learnable noise factors for different layers
- Includes exponential decay scheduling for noise strength

### Training Methodology

Training utilizes the Kernel Continuous Ranked Probability Score (KCRPS) as the primary loss function, optimizing the model's ability to produce well-calibrated probabilistic forecasts. The CRPS loss evaluates the entire forecast distribution against observations, encouraging both accuracy and appropriate uncertainty quantification.

#### Fine-tuning Process:
- Pretrained CrossFormer weights are frozen (`freeze=True`)
- Only noise-injection layers and associated parameters are trained
- Noise factors are learnable parameters that adapt during training
- Separate noise factors for encoder and decoder stages

#### Scaling Strategies

CREDIT supports two distinct scaling approaches for multi-GPU training:

**Local Ensemble Approach (`trainer.type: era5`):**
- Each GPU maintains its own ensemble of size `ensemble_size`
- KCRPS is computed independently on each device
- Final loss is averaged across all GPUs
- Total computational cost scales linearly with GPU count

**Distributed Ensemble Approach (gen 1: `trainer.type: ensemble-gen1`, alias `era5-ensemble`):**
- Ensemble members are distributed across available GPUs
- Effective ensemble size becomes `ensemble_size × num_gpus`
- KCRPS computation occurs across the entire distributed ensemble
- Batch size remains constant per GPU regardless of ensemble scaling
- Requires cross-GPU communication for loss computation

### Gen 2: ring-CRPS ensemble training

Gen 2 configs train distributed ensembles with the standard gen2 trainer
(`trainer.type: gen2`). There is no separate ensemble trainer. Set `ring-crps` as
the `BaseLoss` training loss and run one ensemble member per data-parallel GPU:

```yaml
trainer:
    type: gen2
    parallelism:
        data: ddp
        tensor: 1
        domain: 1
    ensemble_size: 4            # must equal the number of data-parallel GPUs
    activation_checkpoint: True # optional; see Training > Trainer configuration
loss:
    type: base
    args:
        training_loss: "ring-crps"
        validation_loss: "mae"  # deterministic validation loss (recommended)
        var_weighting: "inverse_variance"
        scaler_path: "/path/scaler.json"
```

Launch with as many data-parallel GPUs as `ensemble_size`, e.g.
`credit submit --cluster derecho -c config.yml --gpus 4`.

- Every data-parallel rank receives the same batch. Each rank's model produces one
  member, and its stochastic components (SDL / noise-injection layers) are seeded
  differently on each rank.
- `ring_crps_loss` computes the fair CRPS across ranks with K−1 ring exchanges,
  so the full ensemble is never held on a single GPU.
- The training log adds `train_std`: the standard deviation of member errors,
  used as a proxy for ensemble spread.
- Data clamping, `backprop_on_timestep`, `retain_graph`, `skip_nan_prune`, and the
  dynamic gradient-norm clip all work as in any gen2 run.
- Configure mass, water, and energy conservation as gen2 postblocks
  (`global_mass_fixer`, `global_water_fixer`, `global_energy_fixer`; see
  [Postblocks](postblocks_gen2.md)), not with the gen1 `model.post_conf` block.

`credit check` verifies the config-side requirements: `ensemble_size > 1`, and a
deterministic `validation_loss` (it warns if one is missing). See
[Losses](Losses.md) for the full ring-CRPS caveats.

Running several members on each GPU (a local `ensemble_size` per rank) is not
supported with ring-CRPS. Use the local ensemble approach with an in-tensor CRPS
loss instead.

## Technical Implementation Summary

### PixelNoiseInjection Module

The `PixelNoiseInjection` class implements sophisticated noise injection with the following features:

- **Multi-scale noise**: Combines per-pixel spatial noise with latent style modulation
- **Learnable parameters**: Trainable modulation factors and noise transformations
- **Adaptive scheduling**: Optional noise scheduling based on forecast steps
- **Channel-wise control**: Independent noise control for each feature channel

Key parameters:
- `noise_dim`: Dimensionality of latent noise vectors (default: 128)
- `feature_channels`: Number of channels in the target feature map
- `noise_factor`: Base scaling factor for noise intensity
- `scheduler`: Optional noise scheduling for temporal variation

### CrossFormerWithNoise Architecture

The `CrossFormerWithNoise` extends the base CrossFormer with:

- **Dual injection points**: Noise injection in both encoder and decoder stages
- **Configurable noise levels**: Separate factors for encoder (0.05) and decoder (0.275) stages
- **Learnable adaptation**: Per-layer trainable noise factors
- **Temporal scheduling**: Exponential decay scheduling for inference rollouts

Architecture highlights:
- Three encoder noise injection layers (when enabled)
- Three decoder noise injection layers (always active)
- Independent noise vectors generated for each injection point
- Preservation of skip connections and feature concatenation

## Diffusion-Based Ensembles

### Configuration

```yaml
trainer:
    type: era5-diffusion
    batch_size: 4
loss:
    type: mse
```

### Model Architecture

CREDIT's diffusion implementation currently supports the Karras U-Net architecture as the primary denoising backbone. Development is ongoing to integrate Vision Transformer (ViT) models as alternative base architectures, potentially offering improved scalability and performance characteristics.

The diffusion approach treats forecast generation as a denoising process, where the model learns to iteratively refine noisy initial states into coherent forecast fields.

### Training Process

Diffusion training in CREDIT trains models from scratch using latitude-weighted MSE loss rather than KCRPS. The training follows a noise schedule where the model learns to denoise progressively corrupted forecast states. The training objective optimizes the model's ability to reverse the noise corruption process at various noise levels.

Key characteristics:
- Models are exposed to a wide range of noise levels during training
- Latitude-weighted MSE loss for denoising optimization
- Iterative refinement process during inference
- Higher computational cost per forecast due to sampling requirements
- Training from scratch rather than fine-tuning pretrained models
- Probabilistic calibration achieved through the iterative sampling process

### Computational Trade-offs

**Noise-Injection Approach:**
- Lower per-forecast computational cost
- Efficient ensemble generation through parallel noise realizations
- Faster inference times
- Simplified training pipeline
- KCRPS loss for direct probabilistic optimization

**Diffusion Approach:**
- Higher per-forecast computational requirements
- Iterative sampling increases inference time
- More complex training dynamics
- Latitude-weighted MSE loss for denoising optimization
- Probabilistic calibration through sampling process


The CREDIT ensemble framework continues to evolve, with ongoing research focused on improving both computational efficiency and forecast quality across diverse meteorological applications.

Inference
Pre-trained deterministic model run a perturbed IC to create ensemble of size N. Options: random or bred vectors
Pre-trained stochastic model run with copies of the same IC to create ensemble of size N.
