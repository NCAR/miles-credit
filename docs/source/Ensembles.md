# CREDIT Ensemble Methods

The CREDIT framework implements two primary approaches for generating probabilistic forecasts: **noise-injection ensembles** and **diffusion-based generation**. Both methods enable the creation of stochastic models that capture forecast uncertainty through different architectural and training strategies.

## Training Non-Deterministic Model Ensembles

CREDIT supports two primary training approaches for ensemble generation:

1. **Fine-tuning approach**: Pre-trained deterministic models fine-tuned with noise-injection layers using CRPS loss
2. **Diffusion training** (gen 1 only): Training from scratch with diffusion models using latitude-weighted MSE loss

The **fine-tuning approach** is currently preferred due to computational efficiency and resource requirements.

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
[`config/gen_2/examples/wxformer_sdl.yml`](https://github.com/NCAR/miles-credit/blob/main/config/gen_2/examples/wxformer_sdl.yml),
derived from the gen2 starter config `config/gen_2/examples/example-v2026.2.yml`.

1. **Copy the base checkpoint into a new `save_loc`.** `load_weights: True` looks for a checkpoint inside the config's own `save_loc`, not somewhere else — it doesn't have a separate "warm start from" path. Use a directory different from the base run's so you don't overwrite it:

   ```bash
   mkdir -p /glade/derecho/scratch/$USER/CREDIT_runs/wxformer_sdl
   cp /glade/derecho/scratch/$USER/CREDIT_runs/starter_gen2/checkpoint.pt \
      /glade/derecho/scratch/$USER/CREDIT_runs/wxformer_sdl/checkpoint.pt
   ```

2. **Start from the config that trained the base model.** Copy it whole, then make the changes in steps 3–5. Leave the `model:` architecture keys untouched: SDL loads the checkpoint by matching parameter names and shapes, so any change there fails to load.

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

   **Fine-tuning a CAMulator checkpoint** (`model.type: camulator`): keep CAMulator's architecture keys and
   also set `spectral_norm_sharp: false`. CAMulator trains its decoder `sharp` convs without spectral norm,
   while `crossformer-ensemble` normalizes them by default. Without the flag, loading the checkpoint raises an
   error naming the flag, instead of silently leaving those convs at their random init.

4. **Set `trainer.load_weights: True`**, and `load_optimizer`/`load_scaler`/`load_scheduler: False` — the trainable parameter set is new and much smaller, so start its optimizer state fresh rather than resuming the base model's.

5. **Switch the loss to ring-CRPS** and set `trainer.ensemble_size` to the number of data-parallel GPUs you will train on. A deterministic loss such as `mse` scores one noisy sample against the target and so trains the noise toward zero, collapsing the ensemble spread. CRPS rewards a spread that matches the error instead (details in [Gen 2: ring-CRPS ensemble training](#gen-2-ring-crps-ensemble-training) below):

   ```yaml
   trainer:
     ensemble_size: 4              # = --gpus x --nodes
   loss:
     type: base
     args:
       training_loss: "ring-crps"
       base_loss_parameters: {alpha: 0.95}   # almost-fair CRPS, as in the SDL paper (1.0 = fair CRPS)
       validation_loss: "mae"      # validation shards data across ranks, so keep it deterministic
       # ... var_weighting, scaler_path, latitude weights as in the base config ...
   ```

   With `var_weighting: inverse_variance`, CRPS and MAE are weighted by `1 / σ` per level rather than `1 / σ²`, which makes the physical-space CRPS equal the normalized-space one (see [Losses](Losses.md#inverse_variance-default)).

6. **Validate before submitting:**

   ```bash
   credit check -c config/gen_2/examples/wxformer_sdl.yml
   ```

   This instantiates the model with your config, which is enough to catch an architecture mismatch against the checkpoint before you burn a GPU allocation finding out.

7. **Train** on exactly `ensemble_size` GPUs. Only the noise-injection layers have `requires_grad=True`; everything else stays fixed at the base checkpoint's values. A smaller learning rate and far fewer epochs than base training are typically enough:

   ```bash
   credit submit --cluster derecho -c config/gen_2/examples/wxformer_sdl.yml --gpus 4 --nodes 1
   ```

### Generating an ensemble at inference time

An SDL checkpoint samples new noise on every forward pass, but `credit rollout` seeds every random number generator from the config's top-level `seed`. Re-running rollout with the same config therefore reproduces the **same** member. To build an N-member ensemble, run rollout once per member, giving each run its own `seed` and output directory:

1. **Point the config at the fine-tuned run.** Rollout loads the checkpoint from `save_loc`, so keep the training config's `save_loc` and `model:` block.

2. **Write one config per member** with a distinct `seed` and `inference.save_forecast`, then roll each one out:

   ```bash
   for m in 0 1 2 3 4; do
     sed -e "s/^seed: .*/seed: $((1000 + m))/" \
         -e "s|^\(\s*save_forecast:\).*|\1 '/glade/derecho/scratch/$USER/CREDIT_runs/wxformer_sdl_forecast/member_$m'|" \
         config/gen_2/examples/wxformer_sdl.yml > wxformer_sdl_member_$m.yml
     credit rollout -c wxformer_sdl_member_$m.yml
   done
   ```

3. **Optionally rescale the noise** with `inference.noise_scale` (or `--noise-scale` when calling `credit/applications/rollout_gen2.py` directly). It multiplies every SDL `noise_factor`: `1.0` (or unset) keeps the trained amplitude, `0.0` gives the deterministic mean member, and values in between shrink the spread.

   ```yaml
   inference:
     noise_scale: 1.0
   ```

The init times, forecast length and output format come from the usual `inference:` block (see [Inference](Inference.md)); only `seed` and `save_forecast` need to differ between members.

There are two distinct ways to build an ensemble in CREDIT, and it's worth being clear about which one you're doing:

- **Perturbed-IC ensemble from a deterministic model.** Run a plain (non-SDL) trained model N times, each from a slightly perturbed initial condition (random or bred-vector perturbations). The stochasticity comes entirely from the IC, not the model.
- **Fixed-IC ensemble from an SDL model.** Run the SDL checkpoint N times from the *same* initial condition. The stochasticity comes from the noise sampled inside the model at each call.

These aren't mutually exclusive — you can perturb the IC and use an SDL model in the same rollout — but each is sufficient on its own to produce an ensemble, and conflating them makes the ensemble's uncertainty harder to interpret.

### Red-in-time noise

By default SDL draws fresh noise at every step, so its noise is white in time. Two opt-in options give the noise memory. Both are AR(1) processes with an e-folding time `tau`: each step is `phi * previous + sqrt(1 - phi**2) * fresh` with `phi = exp(-timestep / tau)`. Both start from the stationary distribution, so every step has the same N(0, 1) marginal as white noise.

**Why single-step training can't set tau.** An AR(1) process that starts from its stationary distribution has the same one-step marginal as white noise. Single-step CRPS training therefore cannot distinguish red noise from white or learn `tau`. Instead, set `tau` from an analysis of the model's error memory, or calibrate it with multi-step training. For the same reason, a white-trained checkpoint can run red noise without retraining.

**AR(1) on SDL's own draws** (`model.red_noise`) applies the process to the latent `z` and to the per-pixel noise at every injection point:

```yaml
model:
  type: "crossformer-ensemble"
  # ... SDL keys as above ...
  red_noise:                 # remove the block for white noise (the default, unchanged bit for bit)
    tau_hours: 264           # one value for every injection point, or [coarse, mid, fine] for the
                             # three decoder points (0 or null keeps a point white)
    timestep_hours: 6        # the model step; credit check compares it with data.timestep
    apply_to: [latent, pixel]
```

The process is still white in space, as SDL was trained, and it is untargeted: every variable and region gets the same memory.

**Targeted physical-space red noise** is the [`targeted_red_noise`](postblocks_gen2.md#targeted_red_noise-targetedrednoise) postblock. It adds a spatially smooth AR(1) pattern, in physical units, to chosen fields over land and sea ice only. Each field can have its own `tau`.

**State and reset.** Both keep their state in plain attributes, not buffers. Checkpoints are therefore unchanged, and DDP never copies one rank's noise to the others.
- The gen2 trainer restarts the state at the first step of every batch: train and validation.
- `credit rollout` restarts it at the start of every forecast.
- Each batch row is an independent stream, so each ensemble member has its own noise history.
- Over a multi-step training window the state carries from step to step but holds no autograd graph.

### Scaling across GPUs

`ensemble_size` in the trainer config controls how many stochastic samples are drawn per input during training. Two placements of that ensemble matter for multi-GPU runs:

- **Local**, one ensemble per GPU: each GPU independently draws `ensemble_size` samples and computes its own ensemble-aware loss; the loss is then averaged across GPUs the normal DDP way. Total compute scales linearly with GPU count, with no extra cross-GPU communication for the ensemble itself.
- **Distributed**, one ensemble spread across GPUs: each data-parallel rank produces one member from the same batch, so `ensemble_size` equals the number of data-parallel GPUs, and the loss is computed jointly across ranks (`ring-crps` in `credit/losses/crps.py`; see below). Per-GPU batch size stays fixed regardless of ensemble size, at the cost of the extra communication.

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

## Diffusion-Based Ensembles

```{note}
Diffusion training is gen 1 only: it uses the gen 1 `era5-diffusion` trainer and has no gen 2 config. Working configs are in
[`config/gen_1/applications/diffusion/`](https://github.com/NCAR/miles-credit/tree/main/config/gen_1/applications/diffusion).
For new gen 2 work, use SDL (above).
```

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
