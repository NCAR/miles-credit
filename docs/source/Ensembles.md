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

**Distributed Ensemble Approach (`trainer.type: era5-ensemble`):**
- Ensemble members are distributed across available GPUs
- Effective ensemble size becomes `ensemble_size × num_gpus`
- KCRPS computation occurs across the entire distributed ensemble
- Batch size remains constant per GPU regardless of ensemble scaling
- Requires cross-GPU communication for loss computation

*Note: Enhanced flexibility for the distributed ensemble approach is currently under development.*

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
