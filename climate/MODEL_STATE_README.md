# Model_State.py - Complete CAMulator State Management Module

## Overview

`Model_State.py` is a **complete, standalone module** for CAMulator climate simulations. It provides everything needed to initialize, run, and manage CAMulator state tensors without depending on Quick_Climate.py.

**Use this module for:**
- Setting up CAMulator simulations
- Accessing/modifying variables by name
- Coupling to other models (ocean, land, etc.)
- Custom physics parameterizations
- Testing and analysis

## What We Built

### 1. **initialize_camulator()** - One-Stop Initialization
Complete setup of CAMulator in one function call.

**Features:**
- ✅ Loads model and configuration
- ✅ Sets up normalization transforms
- ✅ Loads initial conditions
- ✅ Loads and normalizes forcing data
- ✅ Creates CAMulatorStepper with conservation fixers
- ✅ Returns everything needed for simulation
- ✅ Comprehensive error checking

### 2. **StateVariableAccessor**
Get/set variables by name instead of dealing with channel indices.

**Key Features:**
- ✅ Access variables by name (e.g., `'U'`, `'TAUX'`, `'SST'`)
- ✅ Handles 3D (multi-level) and 2D (surface) variables automatically
- ✅ Works at different pipeline stages (state, input, output)
- ✅ Time indexing support
- ✅ In-place modification
- ✅ Clear error messages when variables not available

### 2. **StateManager**
Manages state transformations (moved from Quick_Climate.py).

**Key Methods:**
- `shift_state_forward()` - Roll state to next timestep
- `build_input_with_forcing()` - Combine state with forcing

### 3. **CAMulatorStepper**
Core coupling interface with built-in variable access (moved from Quick_Climate.py).

**Key Features:**
- ✅ Built-in accessors for all tensor types
- ✅ Convenience methods for common operations
- ✅ Clean interface for coupled models

## Why This is Powerful

### Before (Manual Indexing):
```python
# Hard to understand, error-prone
# Where is TAUX in the tensor? Need to count channels...
u_start = 0
u_end = 32
u_wind = state[:, u_start:u_end, :, :, :]

# What if I want TAUX? Have to calculate indices manually
# TAUX is in diagnostics, which is at the end... but where exactly?
```

### After (Variable Access):
```python
# Clear, self-documenting, bulletproof
accessor = StateVariableAccessor(conf, tensor_type='state')
u_wind = accessor.get_state_var(state, 'U')
taux = accessor.get_state_var(prediction, 'TAUX')  # Automatically handles different tensor types
```

## Three Tensor Types

The accessor handles three different tensor structures:

### 1. **'state'** - Pure Atmospheric State
**Contents:** Prognostic + Surface variables
**When:** After `shift_state_forward()`, before forcing added
**Example:**
```python
accessor = StateVariableAccessor(conf, tensor_type='state')
u = accessor.get_state_var(state, 'U')      # ✓ Works
ps = accessor.get_state_var(state, 'PS')    # ✓ Works
taux = accessor.get_state_var(state, 'TAUX') # ✗ Error - not in state
sst = accessor.get_state_var(state, 'SST')   # ✗ Error - forcing not in state
```

### 2. **'input'** - Model Input (with Forcing)
**Contents:** Static + Dynamic Forcing + Prognostic + Surface
**When:** After `build_input_with_forcing()`, ready for model
**Example:**
```python
accessor = StateVariableAccessor(conf, tensor_type='input')
u = accessor.get_state_var(model_input, 'U')      # ✓ Works
sst = accessor.get_state_var(model_input, 'SST')  # ✓ Works - forcing available!
taux = accessor.get_state_var(model_input, 'TAUX') # ✗ Error - diagnostics not in input
```

### 3. **'output'** - Model Prediction (with Diagnostics)
**Contents:** Prognostic + Surface + Diagnostics
**When:** Model prediction output
**Example:**
```python
accessor = StateVariableAccessor(conf, tensor_type='output')
u = accessor.get_state_var(prediction, 'U')       # ✓ Works
taux = accessor.get_state_var(prediction, 'TAUX') # ✓ Works - diagnostics available!
sst = accessor.get_state_var(prediction, 'SST')   # ✗ Error - forcing not in output
```

## Use Cases

### 1. **Nudging to Observations**
```python
accessor = StateVariableAccessor(conf, tensor_type='state')

# Get model state
u_model = accessor.get_state_var(state, 'U')

# Apply nudging
u_nudged = 0.9 * u_model + 0.1 * u_observations

# Set back
accessor.set_state_var(state, 'U', u_nudged)
```

### 2. **Ocean-Atmosphere Coupling**
```python
stepper = CAMulatorStepper(model, conf, device)

# Run atmosphere
prediction = stepper.step(state, dynamic_forcing, static_forcing)

# Extract surface stresses for ocean
taux = stepper.get_state_var(prediction, 'TAUX', tensor_type='output')
tauy = stepper.get_state_var(prediction, 'TAUY', tensor_type='output')

# Send to ocean model
ocean_model.set_surface_stress(taux, tauy)

# Get SST from ocean for next timestep
sst_new = ocean_model.get_sst()
stepper.set_state_var(dynamic_forcing, 'SST', sst_new, tensor_type='input')
```

### 3. **Flux Correction**
```python
output_accessor = StateVariableAccessor(conf, tensor_type='output')

# Get predicted fluxes
shflx = output_accessor.get_state_var(prediction, 'SHFLX')
lhflx = output_accessor.get_state_var(prediction, 'LHFLX')

# Apply correction
shflx_corrected = apply_bias_correction(shflx, climatology)
lhflx_corrected = apply_bias_correction(lhflx, climatology)

# Set back
output_accessor.set_state_var(prediction, 'SHFLX', shflx_corrected)
output_accessor.set_state_var(prediction, 'LHFLX', lhflx_corrected)
```

### 4. **Custom Physics Parameterizations**
```python
# Extract state variables
t = accessor.get_state_var(state, 'T')
q = accessor.get_state_var(state, 'Qtot')
ps = accessor.get_state_var(state, 'PS')

# Apply custom convection scheme
t_conv, q_conv, prect = my_convection_scheme(t, q, ps)

# Update state
accessor.set_state_var(state, 'T', t_conv)
accessor.set_state_var(state, 'Qtot', q_conv)

# After model run, update precipitation
output_accessor.set_state_var(prediction, 'PRECT', prect, tensor_type='output')
```

### 5. **Testing and Debugging**
```python
# Zero out a variable for sensitivity test
accessor = StateVariableAccessor(conf, tensor_type='state')
u_zeros = torch.zeros_like(accessor.get_state_var(state, 'U'))
accessor.set_state_var(state, 'U', u_zeros)

# Check what variables are available
available = accessor.list_available_vars()
print(f"Available in state: {list(available.keys())}")
```

## API Reference

### StateVariableAccessor

```python
# Initialize
accessor = StateVariableAccessor(conf, tensor_type='state')  # or 'input', 'output'

# Get variable
var = accessor.get_state_var(tensor, 'VAR_NAME')
var_t0 = accessor.get_state_var(tensor, 'VAR_NAME', time_idx=0)

# Set variable
accessor.set_state_var(tensor, 'VAR_NAME', new_data)
accessor.set_state_var(tensor, 'VAR_NAME', new_data, time_idx=0)

# Query info
info = accessor.get_var_info('VAR_NAME')
available = accessor.list_available_vars()
```

### CAMulatorStepper

```python
# Initialize
stepper = CAMulatorStepper(model, conf, device)

# Time-stepping
prediction = stepper.step(state, dynamic_forcing, static_forcing)
new_state = stepper.state_manager.shift_state_forward(state, prediction)

# Variable access (convenience methods)
var = stepper.get_state_var(tensor, 'VAR', tensor_type='state')
stepper.set_state_var(tensor, 'VAR', new_data, tensor_type='state')

# Direct accessor access
stepper.state_accessor   # For 'state' tensors
stepper.input_accessor   # For 'input' tensors
stepper.output_accessor  # For 'output' tensors
```

## Integration with Quick_Climate.py

To use in Quick_Climate.py:

```python
# At top of file
from Model_State import StateVariableAccessor, StateManager, CAMulatorStepper

# Replace existing StateManager and CAMulatorStepper definitions
# (they're now imported from Model_State)

# Use in your code
stepper = CAMulatorStepper(model, conf, device)

# Access variables anywhere
u_wind = stepper.get_state_var(state, 'U', tensor_type='state')
taux = stepper.get_state_var(prediction, 'TAUX', tensor_type='output')
```

## Error Handling

The accessor provides clear error messages:

```python
# Variable not in config
accessor.get_state_var(state, 'INVALID')
# ValueError: Variable 'INVALID' not found in config. Available variables: ['U', 'V', ...]

# Variable not in current tensor type
accessor.get_state_var(state, 'TAUX')  # state accessor
# ValueError: Variable 'TAUX' not available in 'state' tensor.
#             Reason: Diagnostics not in state tensor

# Shape mismatch
wrong_shape = torch.randn(1, 16, 1, 192, 288)
accessor.set_state_var(state, 'U', wrong_shape)
# ValueError: Shape mismatch for 'U'. Expected (1, 32, 1, 192, 288), got (1, 16, 1, 192, 288)
```

## Files Created

1. **Model_State.py** - Main module (561 lines)
2. **USAGE_EXAMPLES.md** - Comprehensive examples (500+ lines)
3. **test_model_state.py** - Test suite
4. **MODEL_STATE_README.md** - This file

## Benefits

1. **Modularity** - Clean separation from Quick_Climate.py
2. **Reusability** - Use in other scripts (coupling, analysis, etc.)
3. **Safety** - Impossible to access wrong channels
4. **Clarity** - Self-documenting code
5. **Flexibility** - Works at any stage of the pipeline
6. **Maintainability** - Easy to add new variables

## Next Steps

### To integrate into Quick_Climate.py:
1. Add import: `from Model_State import StateVariableAccessor, StateManager, CAMulatorStepper`
2. Remove local StateManager and CAMulatorStepper class definitions
3. Start using variable accessors in your coupling code

### To use in other scripts:
```python
from Model_State import StateVariableAccessor, CAMulatorStepper
```

## Design Philosophy

- **"Variables, not indices"** - Think in terms of physics, not tensor layout
- **"Know what you have"** - Clear errors when variables aren't available
- **"Stage-aware"** - Different accessors for different pipeline stages
- **"Fail-safe"** - Shape validation prevents silent errors
- **"Self-documenting"** - Code that uses accessors explains itself

---