# AI Weather Prediction Models in CREDIT

This document provides a detailed guide on how to define, train, and use AI weather prediction models in CREDIT. The following sections cover workspace setup, data handling, training, model configuration, loss functions, prediction settings, and PBS script configurations. Documentation is also provided in `example.yml`. 

## Workspace Configuration

* **Workspace Location**: 
  * `save_loc` defines the location to save your workspace. It will contain the PBS script, a copy of this configuration file, model weights, and the `training_log.csv`. If the specified location does not exist, it will be created automatically.
  * `save_loc`: `/glade/work/$USER/CREDIT_runs/fuxi_6h/`
* **Random Seed**: Set a random seed for reproducibility.
  * `seed`: 1000

## Data Configuration

* **Upper-Air Variables**: 
  * Upper-air variables must be in YEARLY zarr or netCDF files with `(time, level, latitude, longitude)` dimensions. Files must have the listed variable names. These variables will be normalized by the dataloader, so users do not need to normalize them.
    * `variables`: `['U', 'V', 'T', 'Q']`
    * `save_loc`: `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/SixHourly_TOTAL_*`

* **Surface Variables**: 
  * Surface variables must be in YEARLY zarr or netCDF files with `(time, latitude, longitude)` dimensions. The time dimension MUST be the same as upper-air variables. Files must have the listed variable names. Surface variables will be normalized by the dataloader, so users do not need to normalize them.
    * `surface_variables`: `['sp', 't2m']`
    * `save_loc_surface`: `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/SixHourly_TOTAL_*`

* **Dynamic Forcing Variables**: 
  * Dynamic forcing variables must be in YEARLY zarr or netCDF files with `(time, latitude, longitude)` dimensions. The time dimension MUST be the same as upper-air variables. Files must have the listed variable names. These variables will be normalized by the dataloader, so users do not need to normalize them.
    * `dynamic_forcing_variables`: `['tsi']`
    * `save_loc_dynamic_forcing`: `/glade/derecho/scratch/dgagne/credit_solar_6h_0.25deg/*.nc`

* **Diagnostic Variables**: 
  * Diagnostic variables must be in YEARLY zarr or netCDF files with `(time, latitude, longitude)` dimensions. The time dimension MUST be the same as upper-air variables. Files must have the listed variable names. These variables will be normalized by the dataloader, so users do not need to normalize them.
    * `diagnostic_variables`: `['Z500', 'T500']`
    * `save_loc_diagnostic`: `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/SixHourly_TOTAL_*`

* **Periodic Forcing Variables**: 
  * Periodic forcing variables must be a single zarr or netCDF file with `(time, latitude, longitude)` dimensions. The time dimension should cover an entire leap year. For example, periodic forcing variables can be provided on the year 2000, and its time coordinates should have 24\*366 hours for an hourly model. These variables MUST be normalized by the user.
    * `forcing_variables`: `['TSI']`
    * `save_loc_forcing`: `/glade/campaign/cisl/aiml/ksha/CREDIT/forcing_norm_6h.nc`

* **Static Variables**: 
  * Static variables must be a single zarr or netCDF file with `(latitude, longitude)` coordinates. These variables MUST be normalized by the user.
    * `static_variables`: `['Z_GDS4_SFC', 'LSM']`
    * `save_loc_static`: `/glade/campaign/cisl/aiml/ksha/CREDIT/static_norm_old.nc`

* **Z-Score Files**: 
  * Z-score files must be zarr or netCDF with `(level,)` coordinates. They MUST include all the variables listed under `variables`, `surface_variables`, `dynamic_forcing_variables`, and `diagnostic_variables`.
  * `mean_path`: `/glade/campaign/cisl/aiml/ksha/CREDIT/mean_6h_0.25deg.nc`
  * `std_path`: `/glade/campaign/cisl/aiml/ksha/CREDIT/std_6h_0.25deg.nc`

* **Training and Validation Years**:
  * Specify the years to form the training and validation sets. The format is `[first_year, last_year (not covered)]`.
    * `train_years`: `[1979, 2014]`
    * `valid_years`: `[2014, 2018]`

* **Scaler Type**: 
  * `std_new` is the new data workflow that works with z-score.
    * `scaler_type`: `std_new`
  * `std_cached` does not perform normalization, it works for cached dataset.
    * `scaler_type`: `std_cached`

* **Input and output temporal dimensions**:
  * Specify the number of input time frames.
    * `history_len`: 2 # 2 for Fuxi, 1 for a state-in-state-out model
    * `valid_history_len`: 2 # keep it the same as `history_len`
  * Specify the number of forecast lead times to predict during training.
    * `forecast_len`: 0 # 0 for single-step training, 1, 2, 3, ... for multi-step training
    * `valid_forecast_len`: 0 # Can be the same as or smaller than `forecast_len`
  * Specify the number of hours for each forecast step.
    * `lead_time_periods`: 6 # 6 for 6-hourly model and 6-hourly training data
* **Other options**:
    * The `skip_periods` keyword resolves the mismatch between 6-hourly models and hourly data. Setting `skip_periods = 6` will train a 6-hourly model on hourly data by skipping and picking every 6th hour. This is **Not a Stable Feature**
      * `skip_periods`: null
    * This keyword makes `std_new` compatible with the old `std` workflow.
      * `static_first`: False # False means input tensors will be formed in the order of [dynamic_forcing -> forcing -> static]

## Trainer Configuration

* **Mode**: 
  * The keyword that controls GPU usage.
    * `mode`: `fsdp` # Fully Sharded Data Parallel (FSDP) 
    * `mode`: `ddp`  # Distributed Data Parallel (FSDP)
    * `mode`: `none` # CPU-based training
* **FSDP-Specific Optimizations**:
  * Allow FSDP to offload gradients to the CPU and free GPU memory. Note: This can cause CPU out-of-memory errors for large models.
    * `cpu_offload`: False
  * Save forward pass activation to checkpoints and free GPU memory.
    * `activation_checkpoint`: True

* **Training Routine**:
  * Choose your training routine.
    * `type`: `standard` # `standard` for single-step or multi-step training with `one_shot: True`, `multi-step` for full multi-step training without `one_shot`.

* **Weights and Optimizer State**:
  * Load existing weights and optimizer state.
    * `load_weights`: True
    * `load_optimizer`: True

* **Checkpoint Settings**:
  * CREDIT saves checkpoints at the end of every epoch. `save_backup_weights` and `save_best_weights` provides more saving options.
    * `save_backup_weights`: True # Also saves checkpoints at the beginning of every epoch.
    * `save_best_weights`: True # Saves the best checkpoint separately based on validation loss. This does not work if `skip_validation: True`.

* **Learning Rate and Regularization**:
  * Update the learning rate to `optimizer.param_groups`. Set to False if a scheduler is used.
    * `update_learning_rate`: False
    * `learning_rate`: `1.0e-03`
  * L2 regularization on trained weights. Set `weight_decay` to 0 to turn off L2 regularization.
    * `weight_decay`: 0

* **Batch Size and Epochs**:
  * Define training and validation batch sizes. For `ddp` and `fsdp`, the actual batch size is `batch_size * number of GPUs`.
    * `train_batch_size`: 1
    * `valid_batch_size`: 1
  * Number of batches per epoch for training and validation.
    * `batches_per_epoch`: 1000 # Set to 0 to use the length of the dataloader
    * `valid_batches_per_epoch`: 20
  * Early stopping.
    * `stopping_patience`: 50
  * Skip validation.
    * `skip_validation`: False
  * Training epoch management.
    * `start_epoch`: 0     # The number of the first epoch
    * `num_epoch`: 10      # The trainer will stop after iterating a given number of epochs.
    * `reload_epoch`: True # set the first epoch based on the checkpoint.
    * `epochs`: 70         # Total number of epochs

* **Automatic Mixed Precision**:
  * Use PyTorch automatic mixed precision (AMP).
    * `amp`: False

* **Scheduler**:
  * Define scheduler for training.
    * `use_scheduler`: True # If True, specify your scheduler in `scheduler`
    * `scheduler`: `{ 'scheduler_type': 'cosine-annealing', 'T_max': *epochs, 'last_epoch': -1 }`

* **Gradient Accumulation and Clipping**:
  * Rescale loss as `loss = loss / grad_accum_every`.
    * `grad_accum_every`: 1
  * Gradient clipping.
    * `grad_max_norm`: 1.0

* **Thread Workers**:
  * Number of workers.
    * `thread_workers`: 4
    * `valid_thread_workers`: 0

## Model Configuration

* **Model Type**: 
  * Specify the model type
    * `type`: `fuxi`
    * `type`: `crossformer`

* **Fuxi Model Specifics**:
  * Define model parameters.
    * `frames`: 2 # Number of input states
    * `image_height`: 640 # Number of latitude grids
    * `image_width`: 1280 # Number of longitude grids
    * `levels`: 15 # Number of upper-air variable levels
    * `channels`: 4 # Upper-air variable channels
    * `surface_channels`: 7 # Surface variable channels
    * `input_only_channels`: 3 # Dynamic forcing, forcing, static channels
    * `output_only_channels`: 0 # Diagnostic variable channels

* **Patchify Layer**:
  * Define the patchify layer parameters.
    * `patch_height`: 4 # Number of latitude grids in each 3D patch
    * `patch_width`: 4 # Number of longitude grids in each 3D patch
    * `frame_patch_size`: 2 # Number of input states in each 3D patch

* **Hidden Layers**:
  * Define the hidden layers.
    * `dim`: 1024 # Dimension (default: 1536)
    * `num_groups`: 32 # Number of groups (default: 32)
    * `num_heads`: 8 # Number of heads (default: 8)
    * `window_size`: 7 # Window size (default: 7)
    * `depth`: 16 # Number of Swin Transformers (default: 48)

* **Padding**:
  * Define the map boundary padding.
    * `pad_lon`: 80 # Number of grids to pad on 0 and 360 deg longitude
    * `pad_lat`: 80 # Number of grids to pad on -90 and 90 deg latitude

* **Spectral Norm**:
  * Use spectral norm.
    * `use_spectral_norm`: True

## Loss Configuration

* **Training Loss**:
  * Specify the main training loss.
    * `training_loss`: `mse` # Available options: "mse", "msle", "mae", "huber", "logcosh", "xtanh", "xsigmoid"

* **Power and Spectral Loss**:
  * Use power or spectral loss. If True, this loss will be added to the `training_loss`: `total_loss = training_loss + spectral_lambda_reg * power_loss (or spectral_loss)`. It is preferred that power loss and spectral loss are NOT applied at the same time.
    * `use_power_loss`: True
    * `use_spectral_loss`: False # If power_loss is on, turn off spectral_loss, and vice versa
  * Rescale power or spectral loss when added to the total loss.
    * `spectral_lambda_reg`: 0.1
  * Truncate small wavenumbers (large wavelength) in power or spectral loss.
    * `spectral_wavenum_init`: 20

* **Latitude Weights**:
  * This file is MANDATORY for the "predict" section below (inference stage). The file must be netCDF and must contain 1D variables named "latitude" and "longitude".
    * `latitude_weights`: `/glade/u/home/wchapman/MLWPS/DataLoader/LSM_static_variables_ERA5_zhght.nc`
  * Use latitude weighting. If True, the latitude_weights file MUST have a variable named "coslat", which is `np.cos(latitude)`.
    * `use_latitude_weights`: True

* **Variable Weights**:
  * Use variable weighting. If True, specify your weights.
    * `use_variable_weights`: True
  * An example of variable weights.
    * `variable_weights`:
      * `U`: `[0.132, 0.123, 0.113, 0.104, 0.095, 0.085, 0.076, 0.067, 0.057, 0.048, 0.039, 0.029, 0.02 , 0.011, 0.005]`
      * `V`: `[0.132, 0.123, 0.113, 0.104, 0.095, 0.085, 0.076, 0.067, 0.057, 0.048, 0.039, 0.029, 0.02 , 0.011, 0.005]`
      * `T`: `[0.132, 0.123, 0.113, 0.104, 0.095, 0.085, 0.076, 0.067, 0.057, 0.048, 0.039, 0.029, 0.02 , 0.011, 0.005]`
      * `Q`: `[0.132, 0.123, 0.113, 0.104, 0.095, 0.085, 0.076, 0.067, 0.057, 0.048, 0.039, 0.029, 0.02 , 0.011, 0.005]`
      * `SP`: `0.1`
      * `t2m`: `1.0`
      * `V500`: `0.1`
      * `U500`: `0.1`
      * `T500`: `0.1`
      * `Z500`: `0.1`
      * `Q500`: `0.1`

## Prediction Configuration

* **Forecasts**:
  * Define forecast settings.
    * `type`: `custom` # Keep it as "custom"
    * `start_year`: 2020 # Year of the first initialization (where rollout will start)
    * `start_month`: 1 # Month of the first initialization
    * `start_day`: 1 # Day of the first initialization
    * `start_hours`: `[0, 12]` # Hour-of-day for each initialization, 0 for 00Z, 12 for 12Z
    * `duration`: 30 # Number of days to initialize, starting from the (year, mon, day) above. Duration should be divisible by the number of GPUs (e.g., duration: 384 for 365-day rollout using 32 GPUs)
    * `days`: 2 # Forecast lead time as days (1 means 24-hour forecast)

* **Laplace Filter**:
  * This keyword will apply a low-pass filter to all fields and each forecast step. It will reduce high-frequency noise but may impact performance.
    * `use_laplace_filter`: False

* **Forecast Save Location**:
  * Specify the location to store rollout predictions. The folder structure will be: `$save_forecast/$initialization_time/file.nc`. Each forecast lead time produces a `file.nc`.
    * `save_forecast`: `/glade/derecho/scratch/ksha/CREDIT/wx_former_6h/`

* **Saved Variables**:
  * Define the saved variables. Setting `save_vars: []` or removing `save_vars` from the configuration will save ALL variables.
    * `save_vars`: `['Z500']`

* **Metadata Location**:
  * Specify the location of the metadata. Users can use `$repo/credit/metadata/era5.yaml` as an example to create their own.
    * `metadata`: `/glade/u/home/ksha/miles-credit/credit/metadata/era5.yaml`

## PBS Script Configuration

* **NSF NCAR HPCs Support**:
  * **Derecho Example**:
    * `conda`: `/glade/work/ksha/miniconda3/envs/credit`
    * `project`: `NAML0001`
    * `job_name`: `train_model`
    * `walltime`: `12:00:00`
    * `nodes`: 8
    * `ncpus`: 64
    * `ngpus`: 4
    * `mem`: '480GB'
    * `queue`: 'main'

  * **Casper Example**:
    * `conda`: `/glade/work/ksha/miniconda3/envs/credit`
    * `job_name`: 'train_model'
    * `nodes`: 1
    * `ncpus`: 8
    * `ngpus`: 1
    * `mem`: '128GB'
    * `walltime`: '4:00:00'
    * `gpu_type`: 'v100'
    * `project`: 'NRIS0001'
    * `queue`: 'casper'
