# CREDIT Ensemble Methods

The CREDIT framework implements two primary approaches for generating probabilistic forecasts: **noise-injection ensembles** and **diffusion-based generation**. Both methods enable the creation of stochastic models that capture forecast uncertainty through different architectural and training strategies.

## Training Non-Deterministic Model Ensembles

CREDIT supports two primary training approaches for ensemble generation:

1. **Fine-tuning approach**: Pre-trained deterministic models fine-tuned with noise-injection layers using CRPS loss
2. **Diffusion training**: Training from scratch with diffusion models using latitude-weighted MSE loss

The fine-tuning approach is currently preferred due to computational efficiency and resource requirements.

The rest of this page walks through the noise-injection approach in detail, called **SDL** (Stochastic Decomposition Layer) elsewhere in the codebase and docs — `crossformer-ensemble`/`crossformer-style` in `model.type`, `StochasticDecompositionLayer` in the code.

## Noise-Injection Ensembles (SDL)

### Architecture

SDL wraps an already-trained `CrossFormer`/`wxformer` model (class `CrossFormerWithNoise`, `credit/models/wxformer/crossformer_ensemble.py`) with a handful of `StochasticDecompositionLayer` modules and freezes every pretrained weight. Only the new layers train. Each `StochasticDecompositionLayer`:

- draws per-pixel, per-channel Gaussian noise the same shape as the feature map it's attached to
- projects a shared latent noise vector `z` (dimension `noise_latent_dim`) through a small learned linear layer to get a per-channel "style" that modulates that noise
- scales the result by a fixed `noise_factor` (a hyperparameter set from the config, e.g. `encoder_noise_factor`/`decoder_noise_factor` — despite living in an `nn.Parameter`, it is constructed with `requires_grad=False` and is not learned) and a learned per-channel `modulation` parameter
- adds that to the feature map

So per injection point, what actually trains is the linear projection and the modulation parameter; `noise_factor` and `noise_latent_dim` are fixed choices you make in the config, not learned. There are three decoder injection points always, and three more in the encoder when `encoder_noise: True`.

Because the base model is frozen and only these small layers train, fine-tuning SDL on top of a base checkpoint is far cheaper than training the base model was — fewer trainable parameters, fewer epochs needed, and (with `correlated: False`, the default) a fresh noise draw at every spatial injection point.

### Fine-tuning SDL on a trained wxformer checkpoint

Starting point: you already have a trained `wxformer` checkpoint (`model_checkpoint.pt` or `checkpoint.pt` under some `save_loc`). A full worked example is at
[`config/gen_2/examples/wxformer_sdl.yml`](https://github.com/NCAR/miles-credit/blob/main/config/gen_2/examples/wxformer_sdl.yml).

1. **Copy the base checkpoint into a new `save_loc`.** `load_weights: True` looks for a checkpoint inside the config's own `save_loc`, not somewhere else — it doesn't have a separate "warm start from" path. Use a directory different from the base run's so you don't overwrite it:

   ```bash
   mkdir -p /glade/derecho/scratch/$USER/CREDIT_runs/wxformer_sdl
   cp /glade/derecho/scratch/$USER/CREDIT_runs/starter_gen2/checkpoint.pt \
      /glade/derecho/scratch/$USER/CREDIT_runs/wxformer_sdl/checkpoint.pt
   ```

2. **Copy the base model's architecture into the SDL config exactly** (`image_height`/`image_width`, `levels`, `channels`, `surface_channels`, `dim`/`depth`, window sizes, `padding_conf`, everything under `model:` the base config had). SDL loads the checkpoint by matching parameter names and shapes; a mismatch either fails to load or loads weights into the wrong place.

3. **Set `model.type: "crossformer-ensemble"`** and add the SDL-specific keys on top of the base architecture:

   ```yaml
   model:
     type: "crossformer-ensemble"
     # ... same architecture keys as the base model's config ...
     freeze: True                  # freeze every pretrained weight; train only the noise layers
     encoder_noise: True           # also inject noise in the encoder, not just the decoder
     noise_latent_dim: 128
     encoder_noise_factor: 0.05
     decoder_noise_factor: 0.275
     correlated: False             # False: a fresh z at every injection point; True: one z per forward pass
   ```

4. **Set `trainer.load_weights: True`**, and `load_optimizer`/`load_scaler`/`load_scheduler: False` — the trainable parameter set is new and much smaller, so start its optimizer state fresh rather than resuming the base model's.

5. **Validate before submitting:**

   ```bash
   credit check -c config/gen_2/examples/wxformer_sdl.yml
   ```

   This instantiates the model with your config, which is enough to catch an architecture mismatch against the checkpoint before you burn a GPU allocation finding out.

6. **Train.** Only the noise-injection layers have `requires_grad=True`; everything else stays fixed at the base checkpoint's values. A smaller learning rate and far fewer epochs than base training are typically enough:

   ```bash
   credit submit --cluster derecho -c config/gen_2/examples/wxformer_sdl.yml --gpus 4 --nodes 1
   ```

### Generating an ensemble at inference time

Once fine-tuned, an SDL checkpoint samples a new `z` on every forward pass, so re-running rollout against the same checkpoint gives a different member each time. There are two distinct ways to build an ensemble in CREDIT, and it's worth being clear about which one you're doing:

- **Perturbed-IC ensemble from a deterministic model.** Run a plain (non-SDL) trained model N times, each from a slightly perturbed initial condition (random or bred-vector perturbations). The stochasticity comes entirely from the IC, not the model.
- **Fixed-IC ensemble from an SDL model.** Run the SDL checkpoint N times from the *same* initial condition. The stochasticity comes from the noise sampled inside the model at each call.

These aren't mutually exclusive — you can perturb the IC and use an SDL model in the same rollout — but each is sufficient on its own to produce an ensemble, and conflating them makes the ensemble's uncertainty harder to interpret.

### Scaling across GPUs

`ensemble_size` in the trainer config controls how many stochastic samples are drawn per input during training. Two placements of that ensemble matter for multi-GPU runs:

- **Local**, one ensemble per GPU: each GPU independently draws `ensemble_size` samples and computes its own ensemble-aware loss; the loss is then averaged across GPUs the normal DDP way. Total compute scales linearly with GPU count, with no extra cross-GPU communication for the ensemble itself.
- **Distributed**, one ensemble spread across GPUs: `ensemble_size × num_gpus` becomes the effective ensemble seen by the loss, computed jointly across ranks (see `ring-crps` in `credit/losses/crps.py`, which shares batches across the data-parallel group specifically for this). Per-GPU batch size stays fixed regardless of ensemble size, at the cost of the extra communication.

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
