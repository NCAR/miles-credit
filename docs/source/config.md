# What's in the Configuration File? 

Your configuration file drives everything from the model training to inference, to creating validation runs. This page documents the possible config options and the what the flags / settings do. 

# CREDIT Configuration Guide  

## Overview

This document provides detailed instructions on configuring `configuration.yml` for running CREDIT.

**Key Topics Covered:**  
- Understanding and modifying `configuration.yml`  
- Default values and expected parameter ranges  
- Common pitfalls and troubleshooting  

---

## General Setup  

### Workspace Configuration  

The following settings define where CREDIT will store output files:  

```yaml
save_loc: '/path/to/workspace/'
seed: 1000
```

- **`save_loc`**: Directory where model weights, logs, and scripts are stored. If it doesn’t exist, CREDIT will create it automatically. The models weights can be large, so make sure ample storage is available. 

- **`seed`**: Random seed for reproducibility. Changing this affects experiment results.  

---

## Data Configuration  

CREDIT requires multiple types of atmospheric data, formatted in **YEARLY** `.nc` or `.zarr` files, the following variables can be contained within the same, or different files.   

### Upper-Air Variables  

Upper-Air variables are those which have either pressure or model levels. These variables are considered prognostic (input + output) and have an expected format which covers whole spatial domain and model levels. 

```yaml
variables: ['U', 'V', 'T', 'Q']
save_loc: '/path/to/upper_air_data/'
```

- **Expected format**: `(time, level, latitude, longitude)`  
- **Normalization**: Handled automatically by the dataloader—no need to preprocess.  

### Surface Variables  

Despite being named 'surface variables' these are prognostic variables (input &output) that are on single levels, either surface, top-of-model, or somewhere in the middle. 

```yaml
surface_variables: ['SP', 't2m', 'Z500', 'T500', 'U500', 'V500', 'Q500']
save_loc_surface: '/path/to/surface_data/'
```

- **Expected format**: `(time, latitude, longitude)`  
- **Must align with upper-air variable timestamps.**  

### Forcing & Diagnostic Variables  

```yaml
dynamic_forcing_variables: ['tsi','sst']
save_loc_dynamic_forcing: '/path/to/dynamic_forcing_data/'

diagnostic_variables: ['Z500', 'T500', 'U500', 'V500', 'Q500']
save_loc_diagnostic: '/path/to/diagnostic_data/'
```

- **Dynamic forcing variables** provide additional time-dependent factors (e.g., solar forcing or SST forcing), these are dynamic (changing in time) variables provided during run time. 

- **Diagnostic variables** are used for evaluation but **not directly predicted** by the model.  

#### Periodic & Static Forcing  

```yaml
forcing_variables: ['TSI', 'SST']
save_loc_forcing: '/path/to/forcing_data.nc'

static_variables: ['Z_GDS4_SFC', 'LSM']
save_loc_static: '/path/to/static_data.nc'
```

- **Periodic forcing**: Should cover an entire leap year (e.g., 366 days for an hourly model).  
- **Static variables**: Must be normalized **by the user** before use.  


You're right again—these options are **critical for data standardization and conservation enforcement**, so they should be fully documented. Below is an **expanded section** that provides detailed explanations.  

---

## Physics and Normalization Files  

CREDIT requires **external reference files** for conservation physics and data normalization. These files must be provided in `.zarr` or `.nc` format.  

### Physics File: `save_loc_physics`  

```yaml
save_loc_physics: '/path/to/physics_data.zarr'
```

- **Purpose**: Stores **grid information and coefficients** needed for enforcing conservation constraints in the **post-processing step** (`post_block`).  
- **Required for**:  
  - **Mass conservation** (`global_mass_fixer`)  
  - **Water conservation** (`global_water_fixer`)  
  - **Energy conservation** (`global_energy_fixer`)  
- **Must include the following variables**:  
  - **For pressure-level grids**: `lon2d`, `lat2d` (longitude/latitude coordinates).  
  - **For hybrid sigma-pressure grids**: `lon2d`, `lat2d`, `coef_a`, `coef_b` (sigma coordinate coefficients).  

💡 *If conservation constraints (`post_conf`) are **enabled**, this file is **required**!*  

---

### Normalization Files  

CREDIT uses **z-score normalization** to standardize input variables. The mean and standard deviation files must contain **all variables used in the model** (upper-air, surface, forcing, diagnostic).  

```yaml
mean_path: '/path/to/mean.nc'
std_path: '/path/to/std.nc'
```

- **`mean_path`**: NetCDF/Zarr file containing mean values for all variables.  
- **`std_path`**: NetCDF/Zarr file containing standard deviation values.  

#### Expected Format  

Both `mean_path` and `std_path` should store **1D variables indexed by level**:  

| Variable Type | Expected Dimensions | Example Variables |
|--------------|-------------------|------------------|
| **Upper-Air** | `(level,)` | `U`, `V`, `T`, `Q` |
| **Surface** | `()` | `SP`, `t2m` |
| **Forcing** | `()` | `TSI` |
| **Diagnostics** | `()` | `Z500`, `T500` |

💡 *Ensure these files contain **ALL variables** listed in the `configuration.yml` sections for `variables`, `surface_variables`, `dynamic_forcing_variables`, and `diagnostic_variables`.*  

---

## Summary of Key Recommendations  

| Parameter | Required For | Notes |
|-----------|-------------|-------|
| `save_loc_physics` | Conservation constraints (`post_conf`) | **Required** if conservation physics is enabled. |
| `mean_path` | `scaler_type: 'std_new'` | **Required** for z-score normalization. |
| `std_path` | `scaler_type: 'std_new'` | Must include **all model variables**. |

### Training Data Selection  

```yaml
train_years: [1979, 2014]  # 1979 - 2013
valid_years: [2014, 2018]  # 2014 - 2017
```

- Defines training/validation split. Adjust these to match the dataset.  

---

This section contains **critical configuration parameters** related to **data preprocessing, input structure, and training behavior**. Below is an expanded, structured section covering these settings in depth.  

---

## Data Preprocessing and Temporal Configuration  

CREDIT supports different **data normalization workflows**, input histories, and forecast strategies. These settings control how data is preprocessed, how the model receives historical context, and how it is trained to predict future states.  

### Normalization: `scaler_type`  

```yaml
scaler_type: 'std_new'  # Options: 'std_new', 'std_cached'
```

- **`std_new`**: The recommended approach. Uses **z-score normalization** with precomputed means and standard deviations from training data.  
- **`std_cached`**: Assumes data has already been **pre-normalized** (e.g., stored in a cached dataset). Use only when working with preprocessed inputs.  

---

### Historical Context: `history_len`  

```yaml
history_len: 1
valid_history_len: 1
```

- **`history_len`**: Number of time steps used as input during training.  
- **`valid_history_len`**: Same as `history_len`; modifying this separately is **not recommended**.  

💡 *For example, if `history_len: 4`, the model will use the last 4 time steps to predict the next state.*  

---

### Forecast Lead Time Configuration  

CREDIT can be trained in **single-step** or **multi-step forecasting** mode:  

```yaml
forecast_len: 0
valid_forecast_len: 0
```

- **`forecast_len`**:  
  - `0` → Single-step prediction (predicts only the next time step).  
  - `1, 2, 3, ...` → Multi-step prediction (predicts several time steps ahead).  
- **`valid_forecast_len`**:  
  - Can be **equal to or smaller than** `forecast_len`.  
  - If `forecast_len > 1`, setting a smaller `valid_forecast_len` allows **shorter validation sequences** (useful for debugging).  

---

### Multi-Step Training Options  

If `forecast_len > 0`, CREDIT supports **customized backpropagation strategies** to improve training efficiency.


```yaml
backprop_on_timestep: [1, 2, 3, 5, 6, 7]
```

- Specifies which time steps contribute to the loss during 
backpropagation.  
- If unspecified, the trainer will backpropagate on *all* timesteps
- Helps **control memory usage** by skipping certain time steps.  

💡 *For example, `[1, 2, 3, 5, 6, 7]` means the model backpropagates on these timesteps but skips others.*  

```yaml
retain_graph: False
```
- Specifies whether the trainer keeps the computation graph through the autoregressive prediction during training
- If so, the backpropagation will go from each `backprop_on_timestep` to the start of the autoregressive rollout
- Will use **a lot** more memory

#### One-Shot Loss Computation  

```yaml
one_shot: False
```

- **`True`**: Computes loss **only on the final predicted time step** (useful for speeding up multi-step training).  
- **`False`**: Computes loss at **every time step**, which may improve stability.  

---

### Temporal Resolution and Data Alignment  

CREDIT supports models trained on different **time step intervals**:  

```yaml
lead_time_periods: 6  # Example: 6-hourly training data
```

- **Controls the time step between consecutive forecast states.**  
  - `6` → 6-hourly model (common for ERA5).  
  - `1` → Hourly model.  

<!-- #### Handling Hourly Data with a 6-Hourly Model  

```yaml
skip_periods: null
```

- If `skip_periods: 6`, CREDIT will **subsample hourly data** to train a 6-hourly model.  
- If `skip_periods: 1` or `null`, it uses all time steps (assumes no subsampling needed).   -->

---

### Input Data Ordering: `static_first`  

CREDIT provides flexibility in how input tensors are structured:  

```yaml
static_first: False
```

- **`True`** → Order: `[static → dynamic forcing → periodic forcing]` (matches older `std` workflow).  
- **`False`** → Order: `[dynamic forcing → periodic forcing → static]` (recommended for `std_new`).  

💡 *If you are using `std_new`, set `static_first: False`.*  

---
### Dataset Type  

CREDIT supports multiple data loading strategies:  

```yaml
dataset_type: ERA5_MultiStep_Batcher
```

- **Options**:  
  - `ERA5_MultiStep_Batcher`  
  - `ERA5_and_Forcing_MultiStep` 
  - `MultiprocessingBatcherPrefetch` (*experimental*)
  - `MultiprocessingBatcher`  
- The **default (`ERA5_MultiStep_Batcher`) is recommended** for efficient parallel data loading.  
- `MultiprocessingBatcherPrefetch` is *experimental* and you may run into weird issues.

---

## Summary of Key Recommendations  

| Parameter | Recommended Setting | Notes |
|-----------|---------------------|-------|
| `scaler_type` | `'std_new'` | Ensures data is properly normalized. |
| `history_len` | `>= 1` | Use longer history for improved forecasts. |
| `forecast_len` | `0` (single-step) or `>0` (multi-step) | Multi-step training requires additional tuning. |
| `backprop_on_timestep` | `[1, 2, 3, 5, 6, 7]` (example) | Skipping some timesteps helps manage memory. |
| `one_shot` | `False` | Set `True` for faster multi-step training. |
| `lead_time_periods` | `6` (for ERA5) | Controls forecast step size. |
| `static_first` | `False` | Recommended for `std_new`. |
| `dataset_type` | `MultiprocessingBatcherPrefetch` | Optimized for performance. |



## Training Configuration  
The `trainer` section controls how CREDIT handles **GPU parallelism, gradient updates, checkpointing, and logging**.  

### Training type and mode  

```yaml
trainer:
    type: era5 # era5 or conus404 (in development)
    mode: none  # Options: "none" (single GPU), "fsdp" (fully sharded), "ddp" (distributed)
```

- Use `era5` for global data
- **Use `fsdp` or `ddp` for multi-GPU training.**  

💡 *For large models, `fsdp` helps distribute computation across multiple GPUs, reducing memory usage.*  

CREDIT supports **single-GPU, multi-GPU, and distributed training**.  
---

### FSDP-Specific GPU Optimization  

If using `fsdp`, you can enable additional optimizations:  

```yaml
cpu_offload: False
activation_checkpoint: True
checkpoint_all_layers: False
```

- **`cpu_offload`**: Moves gradients to CPU memory (frees GPU memory but **can cause CPU OOM errors**).  
- **`activation_checkpoint`**: Saves activations in forward pass (reduces GPU memory but slows training).  
- **`checkpoint_all_layers`**:  
  - `True` → Checkpoints activations for **all layers**.  
  - `False` → Uses **custom layer-wise checkpointing** (set in `credit/distributed.py`).  

💡 *Use `activation_checkpoint: True` if training large models on limited memory GPUs.*  

---

### Torch Compilation  

Torch 2.0 introduces **compiling to torchscript** to speed up training.  

```yaml
compile: False
```

- **`True`** → Enables `torch.compile()` (can improve performance).  
- **`False`** → Default setting (recommended for maximum compatibility).  

💡 *Setting `compile: True` may break custom models—test before enabling.*  

---

### Checkpointing & Weight Management  

CREDIT automatically **saves and reloads model states**. It will warn you if you are trying to load a model when no weights are available. To continue a run (or to extend the multi-step training), it is crucial to set the weight-loading to True.

```yaml
load_weights: True
load_optimizer: True
load_scaler: True
load_scheduler: True
```

- **`load_weights`** → Loads existing model weights.  
- **`load_optimizer`** → Restores optimizer state (needed for resuming training).  
- **`load_scaler`** → Loads mixed-precision gradient scaler (if using AMP).  
- **`load_scheduler`** → Restores learning rate scheduler state.  

💡 *When starting multi-step training, initially set **only `load_weights: True`**, then enable all options for full restoration.*  

#### Saving Checkpoints  

```yaml
save_backup_weights: True
save_best_weights: True
```

- **`save_backup_weights`** → Saves a **checkpoint at the start of every epoch** (acts as a recovery point).  
- **`save_best_weights`** → Saves the **best model based on validation loss**.  

💡 *If `skip_validation: True`, `save_best_weights` will NOT work!*  

---

### Logging & Training Metrics  

CREDIT logs training performance in `training_log.csv`.  

```yaml
save_metric_vars: True
```

- **`True`** → Saves metrics **for all predicted variables**.  
- **List of variables** → Saves only the specified ones:  

  ```yaml
  save_metric_vars: ["Z500", "Q500", "Q", "T"]
  ```

- **`[]` or `None`** → Saves only **bulk metrics** (averaged over all variables).  

💡 *Reducing the number of tracked variables speeds up training logs.*  

---

### Learning Rate Updates  

```yaml
update_learning_rate: False
```

- **`False`** → Learning rate is **controlled by the scheduler**.  
- **`True`** → Manually updates `optimizer.param_groups`.  

💡 *Set this to `False` if you are using a scheduler!*  

---

## Summary of Key Recommendations  

| Parameter | Recommended Setting | Notes |
|-----------|---------------------|-------|
| `mode` | `"fsdp"` (for multi-GPU) | `"ddp"` for simpler parallel training. |
| `cpu_offload` | `False` | Saves GPU memory but **can cause CPU OOM errors**. |
| `activation_checkpoint` | `True` | Saves memory but **slows training**. |
| `checkpoint_all_layers` | `False` | Use **custom layer-wise checkpointing**. |
| `compile` | `False` | Test before enabling (`True` can break custom models). |
| `save_backup_weights` | `True` | Creates a **checkpoint every epoch**. |
| `save_best_weights` | `True` | Saves **best validation checkpoint** (requires `skip_validation: False`). |
| `save_metric_vars` | `True` (or specify variables) | Controls **what gets logged**. |
| `update_learning_rate` | `False` | **Disable** if using a scheduler. |



### Learning Rate & Optimization  

```yaml
learning_rate: 1.0e-03
use_scheduler: False
```

- Set `use_scheduler: True` to enable learning rate decay.  


### Regularization & Weight Decay  

```yaml
weight_decay: 0
```

- **L2 regularization**: Helps prevent overfitting by penalizing large weights.  
- **`0`** → Turns off regularization.  
- **Typical values**: `1e-5` to `1e-3` (increase for stronger regularization).  

💡 *If training a very deep model, try `weight_decay: 1e-4` to reduce overfitting.*  

---

### Batch Size Configuration  

```yaml
train_batch_size: 1
valid_batch_size: 1
ensemble_size: 1
```

- **`train_batch_size`**: Number of samples per training batch.  
- **`valid_batch_size`**: Number of samples per validation batch.  
- **`ensemble_size`**: Controls stochastic ensemble training (default = `1`, meaning deterministic behavior).  

💡 *For **multi-GPU training** (`fsdp` or `ddp`), the **effective batch size** = `train_batch_size × num_GPUs`.*  

---

### Number of Batches Per Epoch  

```yaml
batches_per_epoch: 1000
valid_batches_per_epoch: 20
```

- **`batches_per_epoch`**:  
  - `0` → Uses the full dataset.  
  - Custom value (e.g., `1000`) → Limits the number of training batches per epoch.  
- **`valid_batches_per_epoch`**: Controls how many validation batches run per epoch.  

💡 *Reducing `batches_per_epoch` helps debug faster before full-scale training.*  

---

### Early Stopping & Validation Skipping  

```yaml
stopping_patience: 50
skip_validation: False
```

- **`stopping_patience`**: Stops training if validation loss **does not improve for N epochs**.  
- **`skip_validation`**:  
  - `True` → Always saves weights, but **does NOT run validation**.  
  - `False` → Runs validation before saving checkpoints.  

💡 *If `skip_validation: True`, `save_best_weights` **will not work**.*  

---

### Epoch & Checkpoint Management  

```yaml
start_epoch: 0
num_epoch: 10
reload_epoch: True
epochs: &epochs 70
```

- **`start_epoch`**: First epoch (useful for resuming training).  
- **`num_epoch`**: Total epochs before training stops.  
- **`reload_epoch`**:  
  - `True` → Reads the **last saved epoch** and resumes training.  
  - `False` → Starts fresh.  
- **`epochs`**: total number of epochs that the scheduler sees

💡 *If using **epoch-based schedulers**, `reload_epoch: True` ensures proper continuation.*  

---

### Learning Rate Scheduling  

```yaml
use_scheduler: False
scheduler:
  scheduler_type: cosine-annealing-restarts
  first_cycle_steps: 250
  cycle_mult: 6.0
  max_lr: 1.0e-05
  min_lr: 1.0e-08
  warmup_steps: 249
  gamma: 0.7
```

- **`use_scheduler`** → Enables learning rate scheduling (`True` or `False`).  
- **Supported scheduler types**:  
  - `cosine-annealing` → Reduces LR smoothly over epochs.  
  - `cosine-annealing-restarts` → Periodically resets the LR.  
  - `step-lr` → Reduces LR at fixed intervals.  

💡 *For long training runs, `cosine-annealing-restarts` helps escape **bad local minima** by periodically resetting the LR.*  

---

### Mixed Precision & Gradient Scaling  

To improve **GPU memory efficiency**, CREDIT supports **mixed precision training**:  

```yaml
amp: False
mixed_precision:
    param_dtype: "float32"
    reduce_dtype: "float32"
    buffer_dtype: "float32"
```

- **`amp: True`** → Enables PyTorch’s **Automatic Mixed Precision (AMP)**.  
- **`mixed_precision`** → Fine-grained FSDP precision control:  
  - **`param_dtype`**: Weight precision (e.g., `"float32"`, `"bfloat16"`).  
  - **`reduce_dtype`**: Precision for gradients during backprop.  
  - **`buffer_dtype`**: Buffer storage precision.  

💡 *For large models, use `param_dtype: "bfloat16"` to **reduce memory usage** with minimal accuracy loss.*  

---

### Gradient Accumulation & Clipping  

```yaml
grad_accum_every: 1
grad_max_norm: 'dynamic'
```

- **`grad_accum_every`**:  
  - `1` → Normal training.  
  - `>1` → Accumulates gradients over multiple steps **before updating weights** (useful for small batch sizes).  
- **`grad_max_norm`**:  
  - `'dynamic'` → Uses **adaptive gradient clipping**.  
  - `0` → No clipping.  

💡 *Enable gradient accumulation (`grad_accum_every > 1`) if batch size is **constrained by memory** but you need a higher effective batch size.*  

---

### CPU Thread & Prefetch Optimization  

CREDIT allows fine-tuning **CPU utilization** for better dataloader performance.  

```yaml
thread_workers: 4
valid_thread_workers: 4
prefetch_factor: 4
```

- **`thread_workers`**: Number of CPU threads for loading training data.  
- **`valid_thread_workers`**: Number of CPU threads for validation data.  
- **`prefetch_factor`**: Number of samples preloaded into the buffer (works with `ERA5_MultiStep_Batcher`).  

💡 *Increase `thread_workers` for faster data loading, but avoid exceeding available CPU cores.*  

---

## Summary of Key Recommendations  

| Parameter | Recommended Setting | Notes |
|-----------|---------------------|-------|
| `weight_decay` | `0` (or `1e-4` for deep models) | Helps prevent overfitting. |
| `train_batch_size` | `1` (increase if possible) | Larger batch size speeds up training. |
| `batches_per_epoch` | `1000` (or `0` to use all data) | Reduce for faster debugging. |
| `stopping_patience` | `50` | Stops training if no improvement. |
| `skip_validation` | `False` | Needed for `save_best_weights`. |
| `reload_epoch` | `True` | Ensures proper resumption of training. |
| `use_scheduler` | `True` (if tuning LR) | Improves long-term stability. |
| `amp` | `False` (enable for mixed precision) | Saves GPU memory. |
| `grad_max_norm` | `'dynamic'` | Prevents gradient explosion. |
| `thread_workers` | `4` | Tune based on available CPUs. |


---

## Model Configuration  

CREDIT supports multiple architectures. Example:  

```yaml
type: "crossformer"
frames: 1
image_height: 640
image_width: 1280
levels: 16
channels: 4
surface_channels: 7
```

- **`type`**: Model architecture (`crossformer`, `fuxi`, etc.).  
- **`frames`**: Number of input states (historical time steps).  
- **`image_height`, `image_width`**: Spatial resolution (latitude × longitude).  
- **`levels`**: Number of atmospheric levels.  
- **`channels`**: Number of upper-air variables.


Here's an **expanded and structured section** detailing the **model configuration**, including explanations of **architecture choices, spatial resolution, patch embeddings, attention mechanisms, and normalization techniques**.  

---
## Model Configuration  

The `model` section defines the **architecture and input structure** for CREDIT.  

### Selecting a Model Architecture  

```yaml
type: "crossformer"
```

- **`crossformer`** → Default model based on transformer architecture.  
- **`fuxi`** → Alternative model architecture.  
- **`debugger`** → Debugging mode (useful for checking data flow).  

💡 *The choice of architecture affects model scalability and computational efficiency.*  

---

### Temporal and Spatial Resolution  

```yaml
frames: 1
image_height: 640
image_width: 1280
levels: 16
```

- **`frames`**: Number of historical time steps used as input.  
- **`image_height`, `image_width`**: Spatial resolution of the input fields (`latitude × longitude`).  
- **`levels`**: Number of vertical pressure levels for upper-air variables.  

💡 *For higher resolution datasets, ensure these values match the input data format.*  

---

### Channel Configuration  

```yaml
channels: 4
surface_channels: 7
input_only_channels: 3
output_only_channels: 0
```

- **`channels`** → Number of **upper-air** input variables.  
- **`surface_channels`** → Number of **surface** input variables.  
- **`input_only_channels`** → Channels for dynamic forcing, static features, or external variables.  
- **`output_only_channels`** → Reserved for **diagnostic variables** (default = `0`).  

💡 *If using additional input features (e.g., solar forcing), update `input_only_channels`.*  

---

### Patch Embedding (For Transformer-Based Models)  

CREDIT supports **patch-based embeddings**, where the spatial domain is divided into small patches for transformer processing.  

```yaml
patch_width: 1
patch_height: 1
frame_patch_size: 1
```

- **`patch_width`, `patch_height`** → Size of each spatial patch (`latitude × longitude`).  
- **`frame_patch_size`** → Number of **time steps per patch** (default = `1`).  

💡 *Larger patch sizes can reduce computational cost but may impact fine-scale feature representation.*  

---

### Transformer Depth and Dimensions  

```yaml
dim: [32, 64, 128, 256]
depth: [2, 2, 2, 2]
```

- **`dim`** → Hidden size at each transformer layer.  
- **`depth`** → Number of transformer blocks per stage.  

💡 *Deeper models capture more complex patterns but require more memory.*  

---

### Attention Mechanism  

CREDIT supports **global and local attention mechanisms** to efficiently model atmospheric dynamics.  

```yaml
global_window_size: [10, 5, 2, 1]
local_window_size: 10
```

- **`global_window_size`** → Size of global attention windows at each layer.  
- **`local_window_size`** → Size of local attention windows.  

💡 *Smaller window sizes focus on localized interactions, while larger sizes improve long-range dependencies.*  

---

### Cross-Embedding (Multi-Scale Feature Extraction)  

```yaml
cross_embed_kernel_sizes:
  - [4, 8, 16, 32]
  - [2, 4]
  - [2, 4]
  - [2, 4]
cross_embed_strides: [2, 2, 2, 2]
```

- **`cross_embed_kernel_sizes`** → Defines kernel sizes for hierarchical embeddings.  
- **`cross_embed_strides`** → Controls how much spatial downsampling occurs.  

💡 *Larger kernel sizes extract broader-scale features, while smaller strides preserve fine details.*  

---

### Regularization & Normalization  

CREDIT includes **various techniques to improve training stability and prevent overfitting**.  

```yaml
attn_dropout: 0.
ff_dropout: 0.
use_spectral_norm: True
```

- **`attn_dropout`** → Dropout rate for **attention layers** (default = `0.0`).  
- **`ff_dropout`** → Dropout rate for **feed-forward layers** (default = `0.0`).  
- **`use_spectral_norm`** → Enables **spectral normalization** (helps with stability in deep networks).  

💡 *Increase dropout (`0.1 - 0.3`) for regularization in larger models.*  

---

### Interpolation & Output Matching  

```yaml
interp: True
```

- **`True`** → Interpolates outputs to match input spatial resolution.  
- **`False`** → Outputs raw model predictions.  

💡 *Set `interp: True` to ensure predictions align with input grid resolution.*  

---

## Summary of Key Recommendations  

| Parameter | Recommended Setting | Notes |
|-----------|---------------------|-------|
| `type` | `"crossformer"` | Default transformer-based model. |
| `frames` | `1` (or higher) | More frames improve historical context. |
| `image_height, image_width` | `640 × 1280` (adjust as needed) | Must match input dataset resolution. |
| `levels` | `16` | Number of vertical pressure levels. |
| `dim` | `[32, 64, 128, 256]` | Controls model capacity. |
| `depth` | `[2, 2, 2, 2]` | Number of layers per stage. |
| `global_window_size` | `[10, 5, 2, 1]` | Attention window size per layer. |
| `attn_dropout` | `0.` (increase if overfitting) | Regularization for attention layers. |
| `use_spectral_norm` | `True` | Stabilizes training. |
| `interp` | `True` | Ensures output matches input grid. |

---

### Handling Boundary Effects with Padding  

To improve numerical stability at **domain edges**, CREDIT supports **boundary padding**.  

```yaml
padding_conf:
    activate: True
    mode: earth
    pad_lat: 80
    pad_lon: 80
```

- **`activate: True`** → Enables padding at spatial domain edges.  
- **`mode: 'earth'`** → Specifies **Earth-system-aware padding** (useful for atmospheric models), which is described in Schreck et al. 2025  
- **`pad_lat`** → Extends padding by `80` latitude points.  
- **`pad_lon`** → Extends padding by `80` longitude points.  

💡 *Padding ensures continuity at boundaries, preventing artifacts in global simulations.*  

---
## Summary of Key Recommendations  

| Parameter | Recommended Setting | Notes |
|-----------|---------------------|-------|
| `padding_conf.activate` | `True` | Enables domain padding. |
| `padding_conf.mode` | `'earth'` | Uses **Earth-system-specific padding**. |
| `padding_conf.pad_lat` | `80` | Adjust based on dataset resolution. |
| `padding_conf.pad_lon` | `80` | Ensures global continuity. |


Here is a **vastly expanded** and **fully structured** explanation of the **post-processing (`post_conf`) section** in **CREDIT**. This covers **conservation schemes, tracer corrections, and energy/mass balance adjustments** in depth.  

---
## Post-Block (`post_conf`)  

The **post-processing block (`post_conf`)** enforces **physical conservation constraints** on model outputs, correcting imbalances in **mass, water, energy, and tracers**.  

### Activating Post-Processing  

```yaml
post_conf:
    activate: True
```

- **`True`** → Enables post-processing corrections.  
- **`False`** → Disables post-processing (not recommended for production runs).  

💡 *Always enable `post_conf` for physically consistent forecasts.*  

---

## Stochastic Kinetic Energy Backscatter (SKEBS)  

SKEBS introduces **stochastic perturbations** to correct **underdispersed forecasts** in weather models.  

Based on [Berner, J., Shutts, G. J., Leutbecher, M., & Palmer, T. N. (2009). A spectral stochastic kinetic energy backscatter scheme and its impact on flow-dependent predictability in the ECMWF ensemble prediction system. Journal of the Atmospheric Sciences, 66(3), 603-626. ](https://journals.ametsoc.org/view/journals/atsc/66/3/2008jas2677.1.xml)

- **`True`** → Enables **kinetic energy backscatter corrections** (experimental).  
- **`False`** → Disables SKEBS.  

💡 *Enable if testing **ensemble perturbations** for uncertainty quantification.*  

```yaml
skebs:
    activate: True
    freeze_base_model_weights: True  # turn off training of the basemodel

    # skebs module training options
    trainable: True # is skebs trainable at all
    freeze_dissipation_weights: False  # turn off training for dissipation
    freeze_pattern_weights: True  # turn off training for the spectral pattern
    lmax: None # lmax, mmax for spectral transforms
    mmax: None

    # custom initialization of alpha
    alpha_init: 0.95 
    train_alpha: False #trains alpha no matter what

    # dissipation config:
    zero_out_levels_top_of_model: 3 # zero out backscatter at top k levels of the model

    dissipation_scaling_coefficient: 10.
    dissipation_type: FCNN 
    # available types:
    #    - prescribed: fixed dissipation rate spatially, varies by level starts at sigma_max level (see below)
    #    - uniform: fixed dissipation rate spatially, varies by level starts at 2.5
    #    - FCNN: two layer small MLP
    #    - FCNN_wide: four layer wide MLP
    #    - unet: user specified arch, default: unet++
    #    - CNN: single 3x3 convolution with padding for each column

    # unet - see models/unet.py for examples
    # architecture:
    padding: 48

    # prescribed dissipation:
    sigma_max: 2.0 # what sigma level to set as the max wind. perturbation will be roughly sigma_max * std for wind at each level

    # spectral filters, will anneal to 0 from anneal_start (linspace)
    max_pattern_wavenum: 60
    pattern_filter_anneal_start: 40

    max_backscatter_wavenum: 100
    backscatter_filter_anneal_start: 90

    # [Optional] default is off
    train_backscatter_filter: False
    train_pattern_filter: False

    # data config - does the backscatter model get statics variables?
    use_statics: False 

    # [Optional] early skebs shutoff on iteration number:
    iteration_stop: 0 # if 0, skebs is always run

    #### debugging ####
    # write files during training:
    write_train_debug_files: False #writing out files while training, if this is False          
    write_train_every: 999

    # write files during inference
    write_rollout_debug_files: False # saves only when no_grad 
```



---

## Conservation Schemes  

CREDIT enforces **physical conservation laws** for:  
1. **Water Conservation** (tracers, precipitation, evaporation).  
2. **Mass Conservation** (fixes inconsistencies in pressure/height fields).  
3. **Energy Conservation** (balances fluxes and temperature).  

---

### General Settings for Conservation Fixers  

Each conservation scheme follows these **shared settings**:  

```yaml
# Applies the correction method
activate: True  

# Converts from normalized values back to real units before applying fixes
denorm: True  

# Runs the correction outside the model (useful for multi-step training)
activate_outside_model: False  

# Specifies the grid type:
#   "pressure" = constant pressure levels
#   "sigma" = hybrid sigma-pressure levels
grid_type: "sigma"

# Required grid variables (latitude, longitude, vertical levels)
lon_lat_level_name: ["lon2d", "lat2d", "coef_a", "coef_b"]

# Specifies whether levels represent layer edges (midpoint=True) or centers (midpoint=False)
midpoint: True  
```

💡 *For **sigma-coordinate models**, ensure the **physics file** includes `coef_a` and `coef_b`. These are the sigma pressure level files in units Pa and Fraction, respectively*  

---

## **Tracer Fixer: Ensuring Non-Negative Water Content**  

This correction ensures **no negative values** for **total water content** and **precipitation**.  

```yaml
tracer_fixer:
    activate: True
    denorm: True
    tracer_name: ["specific_total_water", "total_precipitation"]
    tracer_thres: [0, 0]
```

- **`tracer_name`** → List of variables to fix (e.g., specific humidity, precipitation).  
- **`tracer_thres`** → Threshold values (e.g., `0` means no negative values allowed).  

💡 *Negative values can appear due to numerical instability—this ensures physically meaningful water content.*  

---

## **Global Mass Fixer**  

This correction **ensures total mass is conserved** across all vertical levels.  

```yaml
global_mass_fixer:
    activate: True
    activate_outside_model: False
    simple_demo: False
    denorm: True
    grid_type: "sigma"
    midpoint: True
    fix_level_num: 7
    lon_lat_level_name: ["lon2d", "lat2d", "coef_a", "coef_b"]
    surface_pressure_name: ["SP"]
    specific_total_water_name: ["specific_total_water"]
```

- **`fix_level_num: 7`** → Ensures conservation only **up to the 7th level** (avoids modifying upper layers).  
- **`surface_pressure_name`** → Name of the **surface pressure variable** (used for pressure-mass balancing).  
- **`specific_total_water_name`** → Name of the **specific humidity variable**.  

💡 *Use this to prevent **mass drift** in long-term climate simulations.*  

---

## **Global Water Fixer**  

This correction ensures **global water conservation** by adjusting precipitation and evaporation terms.  

```yaml
global_water_fixer:
    activate: True
    activate_outside_model: False
    simple_demo: False
    denorm: True
    grid_type: "sigma"
    midpoint: True
    lon_lat_level_name: ["lon2d", "lat2d", "coef_a", "coef_b"]
    surface_pressure_name: ["SP"]
    specific_total_water_name: ["specific_total_water"]
    precipitation_name: ["total_precipitation"]
    evaporation_name: ["evaporation"]
```

- **`precipitation_name`** → Variable name for **total precipitation**.  
- **`evaporation_name`** → Variable name for **evaporation flux**.  

💡 *Prevents artificial drift in atmospheric moisture by correcting **evaporation/precipitation imbalances**.*  

---

## **Global Energy Fixer**  

This correction **ensures total energy conservation** by adjusting **heat fluxes, radiation, and wind kinetic energy**.  

```yaml
global_energy_fixer:
    activate: True
    activate_outside_model: False
    simple_demo: False
    denorm: True
    grid_type: "sigma"
    midpoint: True
    lon_lat_level_name: ["lon2d", "lat2d", "coef_a", "coef_b"]
    surface_pressure_name: ["SP"]
    air_temperature_name: ["temperature"]
    specific_total_water_name: ["specific_total_water"]
    u_wind_name: ["u_component_of_wind"]
    v_wind_name: ["v_component_of_wind"]
    surface_geopotential_name: ["geopotential_at_surface"]
    TOA_net_radiation_flux_name: ["top_net_solar_radiation", "top_net_thermal_radiation"]
    surface_net_radiation_flux_name: ["surface_net_solar_radiation", "surface_net_thermal_radiation"]
    surface_energy_flux_name: ["surface_sensible_heat_flux", "surface_latent_heat_flux"]
```

### Key Adjustments  

| Variable | Purpose |
|----------|---------|
| **`air_temperature_name`** | Balances total heat content. |
| **`specific_total_water_name`** | Adjusts for latent heat effects. |
| **`u_wind_name`, `v_wind_name`** | Ensures kinetic energy conservation. |
| **`surface_geopotential_name`** | Ensures consistency with potential energy. |
| **`TOA_net_radiation_flux_name`** | Accounts for top-of-atmosphere radiation balance. |
| **`surface_net_radiation_flux_name`** | Balances incoming and outgoing radiation. |
| **`surface_energy_flux_name`** | Adjusts for surface energy exchanges. |

💡 *Use this to prevent **temperature drift** and ensure radiative balance in climate models.*  

---

## Summary of Key Conservation Fixers  

| Fixer | Purpose | Key Variables |
|--------|---------|--------------|
| **Tracer Fixer** | Prevents **negative water values** | `"specific_total_water"`, `"total_precipitation"` |
| **Mass Fixer** | Ensures **total air mass conservation** | `"SP"`, `"specific_total_water"` |
| **Water Fixer** | Balances **precipitation and evaporation** | `"SP"`, `"total_precipitation"`, `"evaporation"` |
| **Energy Fixer** | Maintains **energy balance** (radiation, heat, wind) | `"temperature"`, `"surface_net_radiation_flux_name"` |

---

## Best Practices  

✅ **Always enable `post_conf` for physically consistent model outputs.**  
✅ **Ensure `save_loc_physics` contains required grid variables** (`lon2d`, `lat2d`, `coef_a`, `coef_b`).  
✅ **Adjust `fix_level_num` if conservation should only apply to certain layers.**  
✅ **Test with `simple_demo: True` first to visualize corrections before full training.**  


## Loss Configuration  

The `loss` section defines how CREDIT **computes training loss**, including options for **custom loss functions, spectral constraints, and latitude-based weighting**.  

---

### Selecting the Training Loss Function  

```yaml
training_loss: "mse"
```

- **Available loss functions**:  
  - `"mse"` → **Mean Squared Error** (default; penalizes large errors).  
  - `"mae"` → **Mean Absolute Error** (more robust to outliers).  
  - `"huber"` → **Huber Loss** (combination of MSE and MAE).  
  - `"logcosh"` → **Log-Cosh Loss** (similar to Huber, smooths large errors).  
  - `"xtanh"` → Custom loss using hyperbolic tangent.  
  - `"xsigmoid"` → Custom loss using sigmoid transformation.  
  - `"KCRPS"` → bias corrected CRPS for ensemble training

💡 *`mse` is recommended for smooth loss surfaces, while `huber` or `logcosh` are better for handling outliers.*  

---

### Power & Spectral Loss  

CREDIT supports **spectral and power-based losses** to penalize errors in the frequency domain.  

```yaml
use_power_loss: False
use_spectral_loss: False
spectral_lambda_reg: 0.1
spectral_wavenum_init: 20
```

- **`use_power_loss`** → Enables **power spectrum loss** (recommended for climate models).  
- **`use_spectral_loss`** → Enables **spectral loss** (alternative to power loss).  
- **`spectral_lambda_reg`** → Weighting factor for spectral loss (`0.1` = mild effect).  
- **`spectral_wavenum_init`** → **Truncates low-wavenumber components**, ensuring loss focuses on fine-scale structures.  

💡 *Enable **only one** of `use_power_loss` or `use_spectral_loss`—they should **not** be used together.*  

---

### Latitude-Based Loss Weighting  

Since Earth’s surface area varies with latitude, CREDIT **supports weighting loss by latitude**.  

```yaml
latitude_weights: "/path/to/latitude_weights.nc"
use_latitude_weights: True
```

- **`latitude_weights`** → NetCDF file containing `cos(latitude)` as a variable (`coslat`).  
- **`use_latitude_weights: True`** → Enables latitude-based weighting to prevent polar regions from dominating training loss.  

💡 *This is **strongly recommended** for global models to ensure loss scaling matches physical area coverage.*  

---

### Variable-Specific Loss Weighting  

CREDIT allows **custom loss weighting per variable**, ensuring critical variables are penalized more heavily.  

```yaml
use_variable_weights: False
```

- **`True`** → Enables custom **per-variable loss weighting**.  
- **`False`** → All variables contribute equally to the loss function.  

#### Example: Custom Variable Weights  

```yaml
variable_weights:
    U: [0.132, 0.123, 0.113, 0.104, 0.095, 0.085, 0.076, 0.067, 0.057, 0.048, 0.039, 0.029, 0.02, 0.011, 0.005]
    V: [0.132, 0.123, 0.113, 0.104, 0.095, 0.085, 0.076, 0.067, 0.057, 0.048, 0.039, 0.029, 0.02, 0.011, 0.005]
    T: [0.132, 0.123, 0.113, 0.104, 0.095, 0.085, 0.076, 0.067, 0.057, 0.048, 0.039, 0.029, 0.02, 0.011, 0.005]
    Q: [0.132, 0.123, 0.113, 0.104, 0.095, 0.085, 0.076, 0.067, 0.057, 0.048, 0.039, 0.029, 0.02, 0.011, 0.005]
    SP: 0.1
    t2m: 1.0
    V500: 0.1
    U500: 0.1
    T500: 0.1
    Z500: 0.1
    Q500: 0.1
```

- **Upper-air variables (`U`, `V`, `T`, `Q`)**: Different weights per level.  
- **Surface variables (`SP`, `t2m`, etc.)**: Single weight per variable.  

💡 *Increase weighting for **critical variables** (e.g., `T500`, `Z500`) to improve accuracy in key forecast fields.*  

---

## Summary of Key Recommendations  

| Parameter | Recommended Setting | Notes |
|-----------|---------------------|-------|
| `training_loss` | `"mse"` (default) | Use `"huber"` or `"logcosh"` if data contains outliers. |
| `use_power_loss` | `False` | Set `True` to penalize **spectral errors**. |
| `use_spectral_loss` | `False` | **Do not enable both spectral and power loss**. |
| `spectral_lambda_reg` | `0.1` | Adjust to control **spectral penalty strength**. |
| `use_latitude_weights` | `True` | Recommended for **global datasets**. |
| `use_variable_weights` | `False` | Enable if **some variables are more important**. |



## Prediction (Inference) Configuration  

The `predict` section controls **how CREDIT runs forecasts** after training, including:  
- **Batching and parallel execution**  
- **Forecast initialization settings**  
- **Storage format for predicted fields**  
- **Post-processing options (e.g., low-pass filtering, anomaly computation)**  

---

### GPU Usage for Inference  

CREDIT supports **single-GPU and distributed inference**.  

```yaml
mode: none  # Options: "none", "fsdp", "ddp"
```

- **`none`** → Runs inference on a **single GPU**.  
- **`fsdp`** → Fully Sharded Data Parallel (**recommended for multi-GPU**).  
- **`ddp`** → Distributed Data Parallel (alternative for multi-GPU).  

💡 *Use `fsdp` for large models to optimize memory usage during inference.*  

---

### Batch Size & Ensemble Forecasting  

```yaml
batch_size: 1
ensemble_size: 1
```

- **`batch_size`** → Number of forecast initializations processed **at once**.  
- **`ensemble_size`** → Number of ensemble members per initialization.  

💡 *Increase `batch_size` if running inference on multiple GPUs.*  

---

### Forecast Initialization Settings  

CREDIT can **initialize forecasts at specific times and run for a set duration**.  

```yaml
forecasts:
    type: "custom"
    start_year: 2019
    start_month: 1
    start_day: 1
    start_hours: [0, 12]
    duration: 1152
    days: 10
```

- **`type`** → `"custom"` (default; allows user-defined start dates).  
- **`start_year`, `start_month`, `start_day`** → Defines the **first forecast initialization**.  
- **`start_hours`** → List of **times per day** for initializing forecasts (e.g., `0` for **00Z**, `12` for **12Z**).  
- **`duration`** → Total number of days to initialize forecasts.  
  - Should be **divisible by the number of GPUs** for parallel execution.  
- **`days`** → Forecast **lead time** in days (e.g., `10` = 10-day forecast).  

💡 *For **year-long forecasts**, set `duration: 365` and `start_hours: [0]` (daily initialization).*  

---

### Output Storage & File Naming  

```yaml
save_forecast: '/path/to/forecast_output/'
```

- Defines **where forecast outputs are stored**.  
- Each initialization creates a **separate subdirectory** inside `save_forecast/`.  
- Output files are saved in **NetCDF format (`.nc`)**.  

💡 *Ensure the path has enough storage capacity for long-duration forecasts!*  

---

### Selecting Output Variables  

```yaml
metadata: '/path/to/metadata/era5.yaml'
```

- CREDIT automatically selects **which variables to save** based on this **metadata file**.  
- **To save all variables**, remove `save_vars` from `configuration.yml`.  

💡 *Modify `metadata.yaml` if custom variables need to be included/excluded.*  

---

### Low-Pass Filtering for Smoother Predictions  

```yaml
use_laplace_filter: False
```

- **`True`** → Applies a **low-pass filter** to reduce high-frequency noise.  
- **`False`** → Saves raw model outputs **without filtering**.  

💡 *Enable `use_laplace_filter: True` if forecasts contain unrealistic high-frequency oscillations.*  

---

### Climatology File for Anomaly Computation  

CREDIT can compute **anomaly correlations** using a reference climatology.  

```yaml
climatology: '/path/to/climatology.nc'
```

- **If provided**, `rollout_metrics.py` will compute **Anomaly ACC (Anomaly Correlation Coefficient)**.  
- **If missing**, Pearson correlation is used instead.  

💡 *Use a **30-year climatology** (e.g., ERA5 **1990-2019**) for best results.*  

---

## Summary of Key Recommendations  

| Parameter | Recommended Setting | Notes |
|-----------|---------------------|-------|
| `mode` | `"fsdp"` (for multi-GPU) | `"none"` for single-GPU inference. |
| `batch_size` | `1` (increase for parallelism) | Processes multiple initializations at once. |
| `ensemble_size` | `1` (or higher for ensembles) | Supports probabilistic forecasting. |
| `forecasts.start_hours` | `[0, 12]` | Runs forecasts twice daily. |
| `forecasts.duration` | `365` (for annual forecasting) | Should be **divisible by the number of GPUs**. |
| `use_laplace_filter` | `False` | Enable if forecasts contain high-frequency noise. |
| `climatology` | `ERA5 1990-2019` | Improves anomaly-based evaluation. |

## PBS Job Submission (HPC)  

For running CREDIT on **NCAR HPC systems (Derecho, Casper)**:  

```yaml
pbs: 
    conda: "credit-derecho"
    project: "NAML0001"
    job_name: "train_model"
    walltime: "12:00:00"
    nodes: 8
    ncpus: 64
    ngpus: 4
```

- **`nodes`, `ncpus`, `ngpus`**: Adjust based on compute resources.  
- **For Casper**: Change `queue: 'casper'` and specify `gpu_type: 'v100'`.  

---

## Troubleshooting  

| Issue | Possible Cause | Solution |
|--------|--------------|----------|
| Training loss does not decrease | Learning rate too high/low | Adjust `learning_rate` or use a scheduler |
| Model runs out of memory | Batch size too large | Reduce `train_batch_size` or enable mixed precision |
| Output fields look unrealistic | Conservation schemes disabled | Ensure `post_conf.activate: True` |
| Forecasts diverge quickly | Model lacks historical context | Increase `frames` in model configuration |
| Data loading errors | Incorrect file format or missing variables | Ensure `.nc` or `.zarr` format and check `save_loc_*` paths |

---

## Best Practices  

- **Check Data Formats**: Ensure variables follow expected dimensions `(time, level, lat, lon)`.  
- **Use a Seed for Reproducibility**: Keep `seed` fixed unless testing variations.  
- **Enable Conservation Schemes**: To maintain physical consistency.  
- **Run Small Tests First**: Before launching full-scale HPC jobs, test with fewer epochs (`num_epoch: 5`).  

---

## Additional Resources  

- **NCAR HPC Guide**: *[Insert Link]*  

---

This guide is a **living document**—please report issues or suggest improvements! 🚀  
