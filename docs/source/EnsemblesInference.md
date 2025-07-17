## CREDIT Ensemble Inference

CREDIT supports ensemble inference for generating probabilistic forecasts using pre-trained models. The framework provides two distinct approaches for creating ensemble diversity:

1.  Perturbing initial conditions with deterministic models
2.  Utilizing stochastic models with identical initial conditions

The inference scripts (`rollout_metrics_noisy_ics.py`, `rollout_metrics_noisy_models.py`) will compute and save **ensemble metrics only**. To **save forecast outputs to NetCDF**, set the `ensemble_size` in the `predict` block of the config file and run `rollout_to_netcdf.py` instead. It natively supports usage of `ensemble_size`. The two scripts presented here compute the CRPS score for each variable and keep track of the ensemble member scores, means and standard deviations.

---

### Configuration

Ensemble inference is configured independently from training using the `predict` field:

```
predict:
    ensemble_size: 16
    ensemble_method:
        type: "bred-vector"  # or "gaussian"
    # For Gaussian noise
    perturbation_std: 0.15
    num_cycles: 3
    # For bred vectors (parameters are fixed)
    # num_cycles: 3
    # perturbation_std: 0.15
    # epsilon: 0.1
    # bred_time_lag: 480  # hours
```

Note: `ensemble_size` in the `predict` field is independent from the `ensemble_size` parameter used during training.

### Ensemble Generation Methods

#### Noisy Initial Conditions (`rollout_metrics_noisy_ics.py`)

This method uses pre-trained deterministic models with perturbed initial conditions to generate ensemble forecasts. Two strategies are supported:

* **Gaussian Noise Perturbation**

    ```
    predict:
        ensemble_size: 16
        ensemble_method:
            type: "gaussian"
        perturbation_std: 0.15
        num_cycles: 3
    ```

    * Adds Gaussian noise directly to initial conditions.
    * `perturbation_std`: Standard deviation of the noise.
    * `num_cycles`: Number of ensemble members.
    * Uses `add_gaussian_noise()` function.

* **Bred Vector Perturbation**

    ```
    predict:
        ensemble_size: 16
        ensemble_method:
            type: "bred-vector"
    ```

    * Uses a cyclic process to generate bred vectors.
    * Parameters (fixed for meteorological optimization):
        * `num_cycles` = 3
        * `perturbation_std` = 0.15
        * `epsilon` = 0.1
        * `bred_time_lag` = 480 (hours)
    * Uses `generate_bred_vectors_cycle()` function with full model integration.

#### Noisy Models (`rollout_metrics_noisy_model.py`)

This method uses stochastic models with identical initial conditions. Ensemble spread arises from internal noise mechanisms during model inference.

Note: Diffusion model inference is under development and will be supported soon.

### Execution

To run ensemble inference on Derecho:

#### Single GPU (Local)

```
torchrun --nproc_per_node=1 rollout_metrics_noisy_ics.py --config model.yml
torchrun --nproc_per_node=1 rollout_metrics_noisy_model.py --config model.yml
```

#### Batch Job Submission on Derecho

To launch a job on Derecho, use the `-l 1` option when calling the rollout script:

```
python rollout_metrics_noisy_ics.py --config model.yml -l 1
python rollout_metrics_noisy_models.py --config model.yml -l 1
```

This will:

* Automatically generate and submit a job to Derecho.
* Save the corresponding launch file as `launch.sh` in your `save_loc` directory.

### Metrics and Output

#### Individual Member Metrics

* Standard deterministic metrics (e.g., RMSE, MAE).
* Computed using `LatWeightedMetrics` or `LatWeightedMetricsClimatology`.
* Output: `{datetime}_ensemble.csv`

#### Ensemble Statistics

* CRPS (Continuous Ranked Probability Score) using `calculate_crps_per_channel()`.
* Ensemble mean and standard deviation.
* RMSE and spread using `calculate_ensemble_metrics()`.
* Output: `{datetime}_average.csv`

### Technical Implementation

#### Memory Management

* Ensemble members are rolled out sequentially to conserve GPU memory.
* Asynchronous processing pools used for metric accumulation.

#### State Management

* Each ensemble member maintains its own state trajectory.
* Initial perturbation applied once at the forecast start.
* Future states evolve independently.

#### Post-processing

* Optional conservation constraints: mass, water, energy.
* Optional Laplace filtering.
* Z-score normalization supported.