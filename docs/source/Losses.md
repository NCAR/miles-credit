# User Guide to Losses

This guide explains how to configure and use different loss functions in your training and validation workflows, including details on weighting schemes and reduction settings.

## Introduction

Loss functions are crucial for training machine machine learning models, as they measure the discrepancy between predictions and true targets. They guide the model's learning process by providing a signal on how to adjust its internal parameters. This guide covers:

* **How to select and configure loss functions via the config file.** This is the primary way you'll interact with and customize your loss settings.
* **How weighting (latitude or variable-based) affects loss computation.** Understanding these weighting schemes is vital for tailoring your model's focus to specific regions or output variables.
* **How to set up different losses for training and validation.** You might want a specific loss for optimizing your model during training, but a different, perhaps more interpretable, loss for evaluating its performance.
* **Examples of variable weighting.** Concrete examples will help you implement variable-level weighting correctly.

## Configuring Loss Functions

In your configuration (`conf`) file, all loss-related settings are specified under the **`loss`** section. This centralizes all loss configurations for easy management.

### Basic keys:

```yaml
loss:
    training_loss: "almost_fair_crps"  # The main loss used during training
    training_loss_parameters:
        alpha: 0.95                    # Optional parameters specific to the loss

    validation_loss: "mae"             # Optional; if missing, defaults to MAE
    validation_loss_parameters:        # Optional parameters for validation loss

    use_power_loss: False              # Optional additional losses
    use_spectral_loss: False

    use_latitude_weights: True         # Enable latitude-based weighting
    latitude_weights: '/path/to/file'  # Path to latitude weights file

    use_variable_weights: False        # Enable variable-level weighting
    variable_weights:                  # Example below
        U: [0.005, 0.011, ...]
        V: [0.005, 0.011, ...]
        ...
```

* **`training_loss`**: This string specifies the name of the loss function to be used during the model's training phase (e.g., `"mse"`, `"mae"`, `"almost_fair_crps"`).
* **`training_loss_parameters`**: An optional dictionary where you can pass specific parameters to your chosen training loss function. For instance, `alpha: 0.95` for `"almost_fair_crps"`.
* **`validation_loss`**: (Optional) If you want to evaluate your model's performance during validation using a different metric than your training loss, specify it here. If omitted, the validation loss will default to Mean Absolute Error (`"mae"`).
* **`validation_loss_parameters`**: (Optional) Similar to `training_loss_parameters`, this allows you to pass specific parameters to your `validation_loss`.
* **`use_power_loss` / `use_spectral_loss`**: These boolean flags allow you to enable additional, specialized loss components, which can be useful for certain types of models or data (e.g., focusing on power spectrum or frequency domain errors).
* **`use_latitude_weights`**: Set this to `True` if you want to apply latitude-dependent weighting to your loss. This is particularly useful in global climate or weather models where the importance of errors might vary with latitude (e.g., higher importance for polar regions).
* **`latitude_weights`**: When `use_latitude_weights` is `True`, provide the file path to a `.zarr` or similar file containing your pre-calculated latitude weights.
* **`use_variable_weights`**: Set this to `True` to enable weighting based on individual output variables or channels. This allows you to give more importance to certain predicted variables over others.
* **`variable_weights`**: When `use_variable_weights` is `True`, you will define a dictionary where keys are your output variable names (e.g., `U`, `V`, `T`) and values are lists of weights corresponding to the channels of that variable.

---

## Weighted Losses and Reduction

When either **`use_latitude_weights`** or **`use_variable_weights`** is set to `True`, the loss function's behavior changes significantly:

* **Forced Reduction to `"none"`**: During the initialization of the base loss (e.g., `nn.MSELoss` or `AlmostFairKCRPSLoss`), the `reduction` method is internally forced to `"none"`. This is crucial because it tells the base loss to return an element-wise loss (i.e., a loss value for every single prediction-target pair) rather than immediately averaging or summing them.
* **`VariableTotalLoss2D` Wrapper**: The base loss (specified by `training_loss` or `validation_loss`) is then wrapped inside a specialized class called **`VariableTotalLoss2D`**. This wrapper is responsible for:
    * Applying the specified latitude weights (if `use_latitude_weights` is `True`) to the element-wise loss.
    * Applying the specified variable weights (if `use_variable_weights` is `True`) to the element-wise loss.
    * Finally, averaging the weighted loss values to produce a single, scalar loss value that the optimizer can use.

This mechanism ensures that weights are applied at the most granular level before any aggregation occurs, giving you precise control over the contribution of different parts of your predictions to the total loss. If weighting is *not* used, the base loss is initialized normally with the specified or default `reduction` (which is typically `"mean"`, meaning the loss is averaged over all elements).

## Example: Variable Weights

When utilizing variable weights, it's essential that the sum of the weights for all variables accurately reflects the total number of output channels from your model. This ensures proper normalization and consistent loss magnitudes. Here's a sample configuration snippet:

```yaml
variable_weights:
    U: [0.005, 0.011, 0.02, 0.029, 0.039, 0.048, 0.057, 0.067, 0.076, 0.085, 0.095, 0.104, 0.113, 0.123, 0.132, 0.141]
    V: [0.005, 0.011, 0.02, 0.029, 0.039, 0.048, 0.057, 0.067, 0.076, 0.085, 0.095, 0.104, 0.113, 0.123, 0.132, 0.141]
    T: [0.005, 0.011, 0.02, 0.029, 0.039, 0.048, 0.057, 0.067, 0.076, 0.085, 0.095, 0.104, 0.113, 0.123, 0.132, 0.141]
    Q: [0.005, 0.011, 0.02, 0.029, 0.039, 0.048, 0.057, 0.067, 0.076, 0.085, 0.095, 0.104, 0.113, 0.123, 0.132, 0.141]
    SP: 0.1
    t2m: 1.0
    V500: 0.1
    U500: 0.1
    T500: 0.1
    Z500: 0.1
    Q500: 0.1
```
In this example, `U`, `V`, `T`, and `Q` each have 16 channels, while `SP`, `t2m`, `V500`, etc., are single-channel variables. **Crucially, make sure the total number of weights across all variables matches the exact number of output channels produced by your model.** Incorrectly specified weights can lead to unexpected training behavior.

## Notes on Validation Loss

* **Default Behavior**: If **`validation_loss`** is not explicitly specified in your configuration, the validation loss will automatically default to `"mae"` (Mean Absolute Error). This provides a robust and easily interpretable metric for evaluating your model during validation.
* **Customization**: You have the flexibility to specify a different loss function for validation by setting `validation_loss` and providing any optional parameters under `validation_loss_parameters`. This is useful if you want to track a specific performance metric during validation that might differ from your primary training objective.
* **Weighted Loss Consistency**: The weighted loss wrappers (like `VariableTotalLoss2D`) are applied consistently for validation losses if `use_latitude_weights` or `use_variable_weights` are enabled. This ensures that your validation metrics are computed with the same weighting scheme as your training loss, providing a fair comparison.

## Summary

* Configure **`training_loss`** and optionally **`validation_loss`** in your configuration file to define your primary and evaluation metrics.
* Utilize **`training_loss_parameters`** and **`validation_loss_parameters`** to fine-tune the behavior of your chosen loss functions with specific arguments.
* Enable **latitude or variable weighting** to apply custom importance to different regions or output channels; this automatically sets the base loss's `reduction` to `"none"` and wraps it for proper weight application.
* **Validate your variable weights** to ensure they sum correctly to the total number of model output channels.
* Remember that **validation loss falls back to MAE if unspecified**, offering a sensible default.

## Example Configurations

Here are a few example `loss` configurations demonstrating varying levels of complexity:

### 1. Simple MAE Loss (Basic)

This is the most straightforward setup. The model trains and validates using Mean Absolute Error, with no special weighting or additional loss components.

```yaml
loss:
    training_loss: "mae"
    validation_loss: "mae" # Explicitly setting, though it's the default
    use_latitude_weights: False
    use_variable_weights: False
    use_power_loss: False
    use_spectral_loss: False
```

### 2. CRPS with Latitude Weighting (Intermediate)

This configuration uses "almost\_fair\_crps" for training, which is a common choice for probabilistic predictions, and applies latitude-based weighting to emphasize certain geographical regions. Validation still uses the default MAE.

```yaml
loss:
    training_loss: "almost_fair_crps"
    training_loss_parameters:
        alpha: 0.95 # A parameter specific to almost_fair_crps

    validation_loss: "mae" # Validation still defaults to MAE

    use_latitude_weights: True
    latitude_weights: '/path/to/your/latitude_weights.zarr' # Provide actual path here
    use_variable_weights: False
    use_power_loss: False
    use_spectral_loss: False
```

### 3. Training with Huber Loss and Custom Delta (Intermediate)

Here, we use the Huber loss for training, which is less sensitive to outliers than MSE. We also specify a custom `delta` parameter for the Huber loss. Validation uses MSE.

```yaml
loss:
    training_loss: "huber"
    training_loss_parameters:
        delta: 1.5 # Custom delta for Huber loss

    validation_loss: "mse" # Using MSE for validation

    use_latitude_weights: False
    use_variable_weights: False
    use_power_loss: False
    use_spectral_loss: False
```

### 4. Complex Setup: CRPS with Variable Weights and Auxiliary Spectral Loss (Advanced)

This advanced example demonstrates combining multiple features:
* **`almost_fair_crps`** as the main training loss.
* **Variable-level weighting** to give different importance to specific output variables (e.g., `U`, `V`, `T`, `Q` for different atmospheric levels, and single-level variables like `SP` and `t2m`).
* An **additional `use_spectral_loss`** component is enabled, which might be useful for models dealing with spectral data or needing to penalize errors in frequency domains.
* Validation uses a custom loss (e.g., `LogCoshLoss`) with its own parameters.

```yaml
loss:
    training_loss: "almost_fair_crps"
    training_loss_parameters:
        alpha: 0.95

    validation_loss: "logcosh" # Using a custom validation loss
    validation_loss_parameters:
        reduction: "mean" # Ensuring proper reduction for logcosh

    use_latitude_weights: False # Not using latitude weights in this example
    latitude_weights: '' # Can be empty if not used

    use_variable_weights: True
    variable_weights: # Ensure these weights sum up to your total output channels
        U: [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08] # 16 channels
        V: [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08] # 16 channels
        T: [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08] # 16 channels
        Q: [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08] # 16 channels
        SP: 0.1 # Single channel
        t2m: 1.0 # Single channel, given high importance
        V500: 0.2 # Single channel
        U500: 0.2 # Single channel
        T500: 0.2 # Single channel
        Z500: 0.2 # Single channel
        Q500: 0.2 # Single channel

    use_power_loss: False
    use_spectral_loss: True # Enable spectral loss for training
```

### 5. Training with MSE + Power Loss, and MAE for Validation (Combined Objectives)

This example shows how to enable an auxiliary loss (`use_power_loss`) alongside the primary training loss (`mse`). This allows the model to optimize for multiple objectives simultaneously.

```yaml
loss:
    training_loss: "mse"
    # training_loss_parameters: {} # No special parameters for MSE

    validation_loss: "mae" # Default MAE for validation

    use_latitude_weights: False
    use_variable_weights: False

    use_power_loss: True # Enable power spectral density loss
    # power_loss_parameters: # Optional parameters if needed for power loss
    #    lambda: 0.5 # Example parameter for power loss

    use_spectral_loss: False
```

These examples should provide a clearer understanding of how to configure your loss functions for various training and validation scenarios. Remember to adjust paths and parameters to match your specific dataset and model requirements.


## Adding a New Custom Loss Function

If the built-in and existing custom loss functions don't perfectly fit your needs, you can easily define and integrate your own custom loss function. This process involves two primary steps: creating a new Python file for your custom loss and then updating the `base_losses.py` file to recognize it.

### 1. Create `custom_loss.py`

First, you'll need to create a new Python file in your project's `credit/losses/` directory, for example, named `custom_loss.py`. Your custom loss function should inherit from `torch.nn.Module` and implement a `forward` method that calculates the loss between the `prediction` and `target` tensors.

Here's a template for what your `custom_loss.py` file might look like:

```python
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    """
    Your custom loss function.

    This class should inherit from torch.nn.Module and implement
    the forward method.
    """
    def __init__(self, reduction='mean', **kwargs):
        """
        Initializes your CustomLoss.

        Args:
            reduction (str): Specifies the reduction to apply to the output.
                             Options: 'none' | 'mean' | 'sum'.
                             Default: 'mean'.
            **kwargs: Any additional parameters specific to your custom loss.
        """
        super().__init__()
        self.reduction = reduction
        # Store any other custom parameters from kwargs

    def forward(self, prediction, target):
        """
        Computes the loss between predictions and targets.

        Args:
            prediction (torch.Tensor): The model's predictions.
            target (torch.Tensor): The true target values.

        Returns:
            torch.Tensor: The computed loss value, reduced according to self.reduction.
        """
        # Implement your custom loss calculation here.
        # Ensure to apply self.reduction (mean, sum, or none) to the final loss.
        # Example (replace with your actual logic):
        loss_unreduced = (prediction - target).abs() # Example: element-wise absolute difference

        if self.reduction == 'mean':
            return torch.mean(loss_unreduced)
        elif self.reduction == 'sum':
            return torch.sum(loss_unreduced)
        elif self.reduction == 'none':
            return loss_unreduced
        else:
            raise ValueError(f"Reduction method '{self.reduction}' not supported.")

```

Remember to define your `__init__` method to handle any parameters your loss function might need (like `reduction` or other custom hyperparameters), and your `forward` method to perform the actual loss calculation.

### 2. Update `base_losses.py`

Next, you'll need to modify your `base_losses.py` file to import your new custom loss class and add it to the `losses` dictionary. This makes your custom loss function discoverable and usable through your configuration file.

**Changes to `base_losses.py`:**

* **Import the Custom Loss:** Add the line `from credit.losses.custom_loss import CustomLoss` at the top of the file, alongside other loss imports.
* **Register the Loss:** Add a new entry to the `losses` dictionary within the `base_losses` function, using a unique string key that you'll use in your configuration (e.g., `"my-custom-loss"`). The value for this key will be your `CustomLoss` class.

Here's how the relevant parts of your `base_losses.py` file should be updated:

```python
import torch.nn as nn
import logging

# ... (other loss imports) ...
from credit.losses.custom_loss import CustomLoss # NEW: Import your custom loss

logger = logging.getLogger(__name__)


def base_losses(conf, reduction="mean", validation=False):
    """Load a specified loss function by its type.

    Args:
        conf (dict): Configuration dictionary containing loss settings.
        reduction (str, optional): Default reduction method if not specified in parameters.
        validation (bool): Use validation loss settings if True, else training loss.

    Returns:
        torch.nn.Module: Instantiated loss function.
    """
    loss_key = "validation_loss" if validation else "training_loss"
    params_key = "validation_loss_parameters" if validation else "training_loss_parameters"

    loss_type = conf["loss"][loss_key]
    loss_params = conf["loss"].get(params_key, {})

    # Ensure 'reduction' is included if not already specified by the user
    if "reduction" not in loss_params:
        loss_params["reduction"] = reduction

    logger.info(f"Loaded the {loss_type} loss function with parameters: {loss_params}")

    # Standard loss registry
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "msle": MSLELoss,
        # ... (other existing losses) ...
        "custom-loss": CustomLoss, # NEW: Add your custom loss to the registry
    }

    if loss_type in losses:
        return losses[loss_type](**loss_params)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported")

```

### 3. Using Your Custom Loss in `conf.yaml`

Once your `CustomLoss` is registered in `base_losses.py`, you can reference it directly in your configuration file, just like any other built-in or pre-existing loss:

```yaml
loss:
    training_loss: "custom-loss" # Use the key you defined in base_losses.py
    training_loss_parameters:
        custom_param1: 10 # Pass parameters specific to your CustomLoss's __init__
        custom_param2: "some_value"
        reduction: "mean" # Optional: Specify reduction, defaults to 'mean' if not used with weights

    validation_loss: "mae" # You can use another loss for validation, or "custom-loss" again
    # ... other loss configurations ...
```

By following these steps, you can seamlessly extend your loss function library to incorporate any custom metrics or objectives required by your specific machine learning tasks.