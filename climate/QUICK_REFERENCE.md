# Quick_Climate_V02.py - Quick Reference Card

## Quick Start

```bash
python Quick_Climate.py \
    --config your_config.yml \
    --model_name checkpoint.pt \
    --device cuda
```

**Note:** No need to specify `--input_shape`, `--forcing_shape`, `--output_shape` anymore! Auto-detected from config.

---

## Key Insight

**The initial state file already contains forcing for the first timestep!**

This means:
- First timestep: Use state as-is (don't add forcing)
- Subsequent timesteps: Add forcing to state

```python
if timestep == 0:
    model_input = state  # Already has forcing
else:
    model_input = add_forcing(state, forcing)  # Need to add forcing
```

---

## What Changed vs Original?

| Feature | Original | V02 |
|---------|----------|-----|
| **Interface** | Monolithic 300-line function | Clean `CAMulatorStepper` class |
| **State docs** | Undocumented | Fully documented in `StateManager` |
| **Shape args** | Manual `--input_shape 1 136 1 192 288` | Auto-detected ✨ |
| **Dead code** | ~30 lines unused | Removed |
| **Imports** | ~70 imports (many unused) | ~30 imports (clean) |
| **Performance** | ⚡ Fast | ⚡ Same speed (parallel I/O preserved) |

---

## For Developers

### Clean Coupling Interface
```python
from Quick_Climate_V02 import initialize_camulator

# One-time setup
context = initialize_camulator('config.yml', 'checkpoint.pt', 'cuda')
stepper = context['stepper']
state = context['initial_state']

# Timestep loop
for t in range(num_timesteps):
    if t == 0:
        model_input = state  # First step special
    else:
        model_input = stepper.state_manager.build_input_with_forcing(
            state, dynamic_forcing, static_forcing
        )

    prediction = stepper.model(model_input.float())
    state = stepper.state_manager.shift_state_forward(model_input, prediction)
```

---

## For Copenhagen Team (Ocean Coupling)

**Most important pattern:**
```python
# Initialize both models
atm_context = initialize_camulator('atm_config.yml')
atm_stepper = atm_context['stepper']
atm_state = atm_context['initial_state']

ocean_model = YourOceanModel()
ocean_state = ocean_model.initial_state()

# Coupled time-stepping
for timestep in range(num_steps):
    # Step 1: Get current atmospheric surface fields
    atm_surface = extract_surface_fields(atm_state)

    # Step 2: Step ocean with atmospheric forcing
    ocean_state = ocean_model.step(
        ocean_state,
        wind_stress=atm_surface['wind_stress'],
        heat_flux=atm_surface['heat_flux']
    )

    # Step 3: Get ocean surface fields
    ocean_surface = ocean_model.get_surface_fields(ocean_state)

    # Step 4: Update atmospheric forcing with ocean feedback
    dynamic_forcing = create_forcing(ocean_surface['sst'], solar_rad)

    # Step 5: Step atmosphere
    if timestep == 0:
        atm_input = atm_state  # Special first step
    else:
        atm_input = atm_stepper.state_manager.build_input_with_forcing(
            atm_state, dynamic_forcing, static_forcing
        )

    atm_prediction = atm_stepper.model(atm_input.float())
    atm_state = atm_stepper.state_manager.shift_state_forward(
        atm_input, atm_prediction
    )
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `Quick_Climate.py` | Main refactored code (use this!) |
| `REFACTORING_SUMMARY.md` | Complete before/after comparison |
| `COUPLING_EXAMPLE.py` | 4 working examples|
| `QUICK_REFERENCE.md` | This file |

---

## Common Errors

### Error: "expected input to have 136 channels, but got 142"
**Solution:** You're adding forcing to the initial state. Don't, that's loaded in the init!

### Error: "Dataset object has no attribute 'shape'"
**Solution:** Fixed in V02. Don't select multiple xarray variables and call `.shape` on the result.

### Warning: "--input_shape is deprecated"
**This is expected.** V02 auto-detects shapes from config.

---

##  Contact

Questions? → wchapman@colorado.edu

Found a bug? → Create an issue or email above

Need coupling help? → See `COUPLING_EXAMPLE.py`
