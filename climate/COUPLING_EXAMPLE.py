"""
COUPLING_EXAMPLE.py
-------------------
Demonstrates how to use Quick_Climate_V02.py's CAMulatorStepper
for coupling to an ocean model or other Earth system components.

This example shows how the Copenhagen team (or anyone) can use
the clean interface to integrate CAMulator into a coupler.
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from Quick_Climate import initialize_camulator, CAMulatorStepper


# ============================================================================
# EXAMPLE 1: Basic Single-Model Integration
# ============================================================================

def example_standalone_integration():
    """
    Shows how to use CAMulatorStepper for a standalone climate run.
    This is the simplest case - no coupling, just clear interfaces.
    """
    print("="*60)
    print("EXAMPLE 1: Standalone Integration")
    print("="*60)

    # One-time initialization
    context = initialize_camulator(
        config_path='./be21_coupled-v2025.2.0_small.yml',
        model_name='checkpoint.pt00091.pt',
        device='cuda'
    )

    # Extract what we need
    stepper = context['stepper']
    state = context['initial_state']
    forcing_dataset = context['forcing_dataset']
    static_forcing = context['static_forcing']
    device = context['device']

    # Get dynamic forcing variables
    df_vars = context['conf']['data']['dynamic_forcing_variables']
    dynamic_ds = forcing_dataset[df_vars]

    print("\nRunning 10 timesteps...")

    for timestep in range(10):
        # Get forcing for this timestep
        forcing_slice = dynamic_ds.isel(time=timestep).load()
        forcing_arr = np.stack([forcing_slice[var].values for var in df_vars], axis=0)
        dynamic_forcing = torch.from_numpy(forcing_arr).unsqueeze(0).unsqueeze(2).to(device)

        # IMPORTANT: First timestep is special!
        # - Timestep 0: initial state already has forcing, so we can use stepper.step() directly
        # - Timestep 1+: state doesn't have forcing, must add it via build_input_with_forcing()
        if timestep == 0:
            # First step: initial state already contains forcing
            with torch.no_grad():
                prediction = stepper.model(state.float())
            prediction = stepper._apply_postprocessing(prediction, state)
        else:
            # Subsequent steps: must add forcing to state
            model_input = stepper.state_manager.build_input_with_forcing(
                state, dynamic_forcing, static_forcing
            )
            with torch.no_grad():
                prediction = stepper.model(model_input.float())
            prediction = stepper._apply_postprocessing(prediction, model_input)

        # Shift state forward for next timestep
        # This returns atmospheric state WITHOUT forcing (ready for next forcing to be added)
        state = stepper.state_manager.shift_state_forward(
            state if timestep == 0 else model_input,
            prediction
        )

        print(f"  Timestep {timestep+1}: prediction shape = {prediction.shape}")

    print("\n✓ Standalone integration complete!\n")


# ============================================================================
# EXAMPLE 2: Coupled Atmosphere-Ocean System (Pseudocode)
# ============================================================================

def example_coupled_system():
    """
    Pseudocode showing how to couple CAMulator to an ocean model.
    This demonstrates the interface the Copenhagen team needs.
    """
    print("="*60)
    print("EXAMPLE 2: Coupled System (Pseudocode)")
    print("="*60)

    print("""
    # ========================================
    # COUPLER PSEUDOCODE
    # ========================================

    # 1. Initialize both models
    atm_context = initialize_camulator('config.yml', device='cuda')
    atm_stepper = atm_context['stepper']
    atm_state = atm_context['initial_state']

    ocean_model = initialize_ocean_model('ocean_config.yml')
    ocean_state = ocean_model.get_initial_state()

    # 2. Time-stepping loop
    for timestep in range(num_timesteps):

        # Get current atmospheric surface fields needed by ocean
        atm_surface_vars = extract_surface_fields(atm_state)
        # e.g., surface_wind, surface_temp, surface_pressure

        # Step ocean forward using atmospheric forcing
        ocean_state_new = ocean_model.step(
            ocean_state,
            forcing={
                'wind_stress': atm_surface_vars['wind_stress'],
                'heat_flux': atm_surface_vars['heat_flux'],
                'freshwater_flux': atm_surface_vars['precip']
            }
        )

        # Get ocean surface fields needed by atmosphere
        ocean_surface_vars = extract_surface_fields(ocean_state_new)
        # e.g., SST, sea ice fraction

        # Update atmospheric forcing with ocean feedback
        dynamic_forcing = update_forcing_with_ocean(
            base_forcing=get_solar_forcing(timestep),
            ocean_vars=ocean_surface_vars
        )

        # Step atmosphere forward
        atm_prediction = atm_stepper.step(
            atm_state,
            dynamic_forcing,
            static_forcing
        )

        # Update states for next timestep
        atm_state = atm_stepper.state_manager.shift_state_forward(
            atm_state, atm_prediction
        )
        ocean_state = ocean_state_new

        # Exchange diagnostics between components
        if timestep % diagnostics_interval == 0:
            save_coupled_diagnostics(atm_state, ocean_state, timestep)

    # ========================================
    # KEY BENEFITS FOR COUPLING:
    # ========================================
    #
    # 1. CLEAR INTERFACE:
    #    - Input: (state, dynamic_forcing, static_forcing)
    #    - Output: next_state
    #    - No hidden side effects
    #
    # 2. NO I/O IN STEPPING:
    #    - Coupler controls all file operations
    #    - Models just do physics
    #
    # 3. DOCUMENTED STATE:
    #    - StateManager.shift_state_forward() handles tensor bookkeeping
    #    - Clear docstrings explain tensor structure
    #
    # 4. TESTABLE:
    #    - Can test atmosphere stepping without ocean
    #    - Can test ocean stepping without atmosphere
    #    - Can test coupler logic separately
    """)

    print("\n✓ See pseudocode above for coupling pattern\n")


# ============================================================================
# EXAMPLE 3: Extracting State Information
# ============================================================================

def example_inspect_state():
    """
    Shows how to understand and extract information from the state tensor.
    Critical for coupling - you need to know what's in the state!
    """
    print("="*60)
    print("EXAMPLE 3: Inspecting State Structure")
    print("="*60)

    context = initialize_camulator(
        config_path='./be21_coupled-v2025.2.0_small.yml',
        model_name='checkpoint.pt00091.pt',
        device='cuda'
    )

    conf = context['conf']
    state = context['initial_state']

    print(f"\nState tensor shape: {state.shape}")
    print(f"  [batch, channels, time, lat, lon]")
    print(f"  = [{state.shape[0]}, {state.shape[1]}, {state.shape[2]}, {state.shape[3]}, {state.shape[4]}]")

    print("\n" + "="*60)
    print("VARIABLE BREAKDOWN:")
    print("="*60)

    # Get variable counts
    n_prognostic = len(conf['data']['variables']) * conf['model']['levels']
    n_surface = len(conf['data']['surface_variables'])
    n_diagnostic = len(conf['data']['diagnostic_variables'])
    n_dynamic_forcing = len(conf['data']['dynamic_forcing_variables'])
    n_static = len(conf['data']['static_variables'])

    print(f"\nPrognostic variables: {n_prognostic} channels")
    print(f"  Variables: {conf['data']['variables']}")
    print(f"  Levels: {conf['model']['levels']}")
    print(f"  = {len(conf['data']['variables'])} vars × {conf['model']['levels']} levels")

    print(f"\nSurface variables: {n_surface} channels")
    print(f"  Variables: {conf['data']['surface_variables']}")

    print(f"\nDiagnostic variables: {n_diagnostic} channels")
    print(f"  Variables: {conf['data']['diagnostic_variables']}")
    print(f"  Note: OUTPUT ONLY, not fed back as input")

    print(f"\nDynamic forcing: {n_dynamic_forcing} channels")
    print(f"  Variables: {conf['data']['dynamic_forcing_variables']}")

    print(f"\nStatic forcing: {n_static} channels")
    print(f"  Variables: {conf['data']['static_variables']}")

    print(f"\nTotal output channels: {n_prognostic + n_surface + n_diagnostic}")
    print(f"Total input channels: {n_prognostic + n_surface + n_dynamic_forcing + n_static}")

    print("\n" + "="*60)
    print("FOR COUPLING - EXTRACT THESE SURFACE FIELDS:")
    print("="*60)
    print("""
    To couple to an ocean model, you'll need:

    1. Surface winds (U, V at lowest level)
       → Used to compute wind stress on ocean

    2. Surface temperature (T at lowest level or T2m)
       → Used for air-sea heat flux

    3. Surface pressure (PS or SP)
       → Used for pressure forcing

    4. Precipitation (if in diagnostic variables)
       → Used for freshwater flux

    Example extraction:
        prediction = stepper.step(state, dynamic_forcing, static_forcing)

        # Split into upper air and surface
        from credit.output import split_and_reshape
        upper_air, surface = split_and_reshape(prediction, conf)

        # Surface variables are in 'surface' tensor
        # Extract what you need for ocean coupling
        surface_wind_u = upper_air[:, 0, -1, :, :]  # U at lowest level
        surface_wind_v = upper_air[:, 1, -1, :, :]  # V at lowest level
        surface_temp = surface[:, conf['data']['surface_variables'].index('t2m'), :, :]
        surface_pres = surface[:, conf['data']['surface_variables'].index('SP'), :, :]
    """)

    print("\n✓ State structure explained!\n")


# ============================================================================
# EXAMPLE 4: Custom Time-Stepping with External Forcing
# ============================================================================

def example_custom_forcing():
    """
    Shows how to provide custom forcing (e.g., from another model).
    """
    print("="*60)
    print("EXAMPLE 4: Custom External Forcing")
    print("="*60)

    context = initialize_camulator(
        config_path='./be21_coupled-v2025.2.0_small.yml',
        model_name='checkpoint.pt00091.pt',
        device='cuda'
    )

    stepper = context['stepper']
    state = context['initial_state']
    static_forcing = context['static_forcing']
    device = context['device']
    conf = context['conf']

    print("\nGenerating custom forcing from 'ocean model'...")

    # Simulate 5 timesteps with custom forcing
    for timestep in range(5):

        # In a real coupler, this would come from your ocean model's SST
        # For demo, we create synthetic forcing
        batch, channels, time, lat, lon = 1, len(conf['data']['dynamic_forcing_variables']), 1, 192, 288

        # Create custom forcing (e.g., SST anomaly from ocean model)
        custom_forcing = torch.randn(batch, channels, time, lat, lon, device=device) * 0.1

        print(f"\n  Timestep {timestep+1}:")
        print(f"    Custom forcing shape: {custom_forcing.shape}")
        print(f"    Forcing mean: {custom_forcing.mean().item():.6f}")

        # Step atmosphere with custom forcing
        prediction = stepper.step(state, custom_forcing, static_forcing)

        print(f"    Prediction shape: {prediction.shape}")

        # Update state
        state = stepper.state_manager.shift_state_forward(state, prediction)

    print("\n✓ Custom forcing integration complete!\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CAMulator COUPLING EXAMPLES")
    print("="*60)
    print("""
This script demonstrates how to use the clean coupling interface
from Quick_Climate_V02.py for integrating CAMulator into coupled
Earth system models.

Choose an example to run:
    1. Basic standalone integration
    2. Coupled system pseudocode
    3. Inspect state structure
    4. Custom external forcing

Or run all examples sequentially.
    """)

    # Run all examples
    # NOTE: Examples 1, 3, 4 require actual model files and will fail if not present
    # Example 2 is just pseudocode and always works

    try:
        example_standalone_integration()
    except Exception as e:
        print(f"Example 1 skipped (likely missing model files): {e}\n")

    example_coupled_system()  # Always works (just prints pseudocode)

    
    example_inspect_state()


    try:
        example_custom_forcing()
    except Exception as e:
        print(f"Example 4 skipped (likely missing model files): {e}\n")

    print("="*60)
    print("COUPLING EXAMPLES COMPLETE")
    print("="*60)
    print("""
For the Copenhagen team:
    - Use CAMulatorStepper.step() as your core interface
    - Call initialize_camulator() once at startup
    - Extract surface fields from predictions for ocean forcing
    - Provide ocean SST/ice as dynamic_forcing input

See the pseudocode in Example 2 for the full coupling pattern.
    """)
