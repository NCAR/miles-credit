"""
Quick_Climate_V02.py
--------------------
Refactored CAMulator climate integration with clearer coupling interfaces.

Key improvements:
- Separated initialization from time-stepping
- Clear CAMulatorStepper class for coupling
- Documented state tensor structure
- Removed dead code
- Preserved async parallel I/O for performance
"""

import os
import time
import logging
import warnings
from pathlib import Path
import multiprocessing as mp
import argparse

from Model_State import initialize_camulator, StateVariableAccessor

# ---------- #
# Numerics
from datetime import datetime
import numpy as np
import xarray as xr

# ---------- #
import torch

# ---------- #
# credit
from credit.output import make_xarray, save_netcdf_increment
from credit.data import get_forward_data

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def add_init_noise(state: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    Add random noise to initial conditions for ensemble generation.

    Args:
        state: Initial state tensor
        noise_std: Standard deviation of Gaussian noise

    Returns:
        state_with_noise: Perturbed state
    """
    print(f"Adding initial condition noise (std={noise_std})")
    noise = torch.randn_like(state) * noise_std
    return state + noise


def parse_datetime_from_config(conf: dict) -> datetime:
    """
    Parse datetime from config, handling string, datetime, and cftime objects.

    Args:
        conf: Configuration dictionary

    Returns:
        init_dt: Python datetime object
    """
    raw_dt = conf["predict"]["start_datetime"]

    if isinstance(raw_dt, str):
        # Parse "YYYY-MM-DD HH:MM:SS" format
        return datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S")
    elif isinstance(raw_dt, datetime):
        # Already a Python datetime
        return raw_dt
    else:
        # Assume it's a cftime object - convert to Python datetime
        # cftime objects have year, month, day, hour, minute, second attributes
        return datetime(raw_dt.year, raw_dt.month, raw_dt.day, raw_dt.hour, raw_dt.minute, raw_dt.second)


# ============================================================================
# INTEGRATION LOOP
# ============================================================================


def run_climate_integration(pool: mp.Pool, context: dict, save_append: str = None, init_noise: float = None):
    """
    Run the CAMulator climate integration loop.

    This function handles the time-stepping, output generation, and parallel I/O.
    The core physics stepping is delegated to the CAMulatorStepper class.

    Args:
        pool: Multiprocessing pool for async file I/O
        context: Dictionary from initialize_camulator()
        save_append: Optional subfolder name for outputs
        init_noise: Optional noise std to add to initial conditions

    Returns:
        flag_energy: Whether energy fixer was active (for diagnostics)
    """
    # Unpack context
    conf = context["conf"]
    stepper = context["stepper"]
    forcing_ds_norm = context["forcing_dataset"]
    static_forcing = context["static_forcing"]
    state = context["initial_state"]
    latlons = context["latlons"]
    metadata = context["metadata"]
    device = context["device"]
    state_transformer = context["state_transformer"]

    # Update save location if append specified
    if save_append:
        base = conf["predict"].get("save_forecast")
        if not base:
            raise KeyError("'save_forecast' missing in config")
        conf["predict"]["save_forecast"] = str(Path(base).expanduser() / save_append)
        print(f"Saving outputs to: {conf['predict']['save_forecast']}")

    # Add noise to initial conditions if requested
    if init_noise is not None:
        state = add_init_noise(state, noise_std=init_noise)

    # Trace model for performance (optional but recommended)
    print("Tracing model with torch.jit...")
    # IMPORTANT: Initial state already contains forcing for first timestep
    # So we trace with the initial state shape as-is (DO NOT add forcing channels)
    dummy_input = torch.zeros_like(state)
    traced_model = torch.jit.trace(stepper.model, dummy_input)
    stepper.model = traced_model
    print(f"Model traced with input shape: {dummy_input.shape}")

    # Setup for time-stepping
    df_vars = conf["data"]["dynamic_forcing_variables"]
    num_ts = conf["predict"]["timesteps_fast_climate"]
    lead_time_periods = conf["data"]["lead_time_periods"]
    chunk_size = conf["data"].get("forcing_chunk_size", 32)
    post_conf = conf["model"]["post_conf"]
    lon_lat_level_names = post_conf["global_mass_fixer"]["lon_lat_level_name"]

    ds_physics = xr.open_dataset(conf["data"]["save_loc_physics"])

    P0 = 100000.
    hyai = torch.tensor(ds_physics['hyai'].values/P0).to(device)[None, :, None, None]
    hyam = torch.tensor(ds_physics['hyam'].values).to(device)[None, :, None, None]
    hybi = torch.tensor(ds_physics['hybi'].values).to(device)[None, :, None, None]
    hybm = torch.tensor(ds_physics['hybm'].values).to(device)[None, :, None, None]
    

    # Get forcing data subset
    dynamic_ds = forcing_ds_norm[df_vars]

    # IMPORTANT: Use the config's datetime object directly for xarray lookup
    # It might be cftime.DatetimeNoLeap, which xarray expects
    start_datetime_raw = conf["predict"]["start_datetime"]
    loc = dynamic_ds.indexes["time"].get_loc(start_datetime_raw)
    start_ix = loc.start if isinstance(loc, slice) else loc
    print(f"Starting integration at time index: {start_ix}")

    # Now convert to Python datetime for output formatting (if it's a string or cftime)
    init_dt = parse_datetime_from_config(conf)
    init_str = init_dt.strftime("%Y-%m-%dT%HZ")

    accessor_state = StateVariableAccessor(conf, tensor_type='state')
    accessor_input = StateVariableAccessor(conf, tensor_type='input')
    accessor_output = StateVariableAccessor(conf, tensor_type='output')

    # ========================================================================
    # MAIN TIME-STEPPING LOOP
    # ========================================================================

    print("Starting time-stepping loop...")
    forecast_hour = 1
    timestep_counter = 0

    for block_start in range(start_ix, start_ix + num_ts, chunk_size):
        block_end = min(block_start + chunk_size, start_ix + num_ts)

        # Load chunk of dynamic forcing data
        ds_slice = dynamic_ds.isel(time=slice(block_start, block_end)).load()
        ds_slice_times = ds_slice["time"].values

        # Stack forcing variables into tensor [time, vars, lat, lon]
        arr_list = [ds_slice[var].values for var in dynamic_ds.data_vars]
        arr = np.stack(arr_list, axis=1)

        # Transfer to GPU once per chunk
        cpu_tensor = torch.from_numpy(arr).unsqueeze(2).pin_memory()
        gpu_forcing_chunk = cpu_tensor.to(device, non_blocking=True)

        # Step through each time in the chunk
        for t in range(gpu_forcing_chunk.shape[0]):
            time_obj = ds_slice_times[t]

            # Convert to Python datetime for output formatting
            # Handle numpy scalar wrapper
            if hasattr(time_obj, "item"):
                time_obj = time_obj.item()

            if isinstance(time_obj, datetime):
                utc_datetime = time_obj
            else:
                # cftime object - convert to Python datetime
                utc_datetime = datetime(
                    time_obj.year, time_obj.month, time_obj.day, time_obj.hour, time_obj.minute, time_obj.second
                )

            if (timestep_counter + 1) % 20 == 0:
                print(f"Model step: {timestep_counter + 1:05}, time: {utc_datetime}")

            dynamic_forcing_t = gpu_forcing_chunk[t].unsqueeze(0)

            # ================================================================
            # CORE PHYSICS STEP
            # This matches the original Quick_Climate.py logic exactly:
            # - First step (timestep_counter=0): state already has forcing, run model as-is
            # - Subsequent steps: add forcing to state, then run model
            # ================================================================

            if timestep_counter != 0:
                # Build forcing from dynamic + static
                model_input = stepper.state_manager.build_input_with_forcing(state, dynamic_forcing_t, static_forcing)
            else:
                # First iteration: initial state already contains forcing
                model_input = state

            sst = accessor_input.get_state_var(model_input, 'SST')
            # torch.save(sst, './sst.pt')

            ## once the coupler has run, set the variable: NOTE this needs to be rescaled for our ML model.
            ##  NOTE this needs to be rescaled for our ML model.
            ##  NOTE this needs to be rescaled for our ML model.
            ##  NOTE this needs to be rescaled for our ML model.
            ##  NOTE this needs to be rescaled for our ML model.
            ##  NOTE this needs to be rescaled for our ML model.
            ##  NOTE this needs to be rescaled for our ML model.
            ##  NOTE this needs to be rescaled for our ML model.
            #accessor_input.set_state_var(model_input, 'SST', new_sst_values)

            # Run model
            with torch.no_grad():
                prediction = stepper.model(model_input.float())

            # Apply post-processing
            prediction = stepper._apply_postprocessing(prediction, model_input)

            # get all of the variables for coupling: 
            prediction_out = state_transformer.inverse_transform(prediction)
            Ut = accessor_output.get_state_var(prediction_out, 'U') # U 
            Vt = accessor_output.get_state_var(prediction_out, 'V') # V
            Qtot = accessor_output.get_state_var(prediction_out, 'Qtot') # specific humidty
            Tt = accessor_output.get_state_var(prediction_out, 'TS') # surface temp
            FSNS = accessor_output.get_state_var(prediction_out, 'FSNS')
            FSNS /=21600 # back in CAM units [W/m2]
            FLNS = accessor_output.get_state_var(prediction_out, 'FLNS')  # FLDS≈εσTs{^4}−FLNS  # will have to approximate it. where emissivity in CAM = 1
            FLNS /=-21600 # back in CAM units [W/m2]

            #Pressure model levels:
            PS = accessor_output.get_state_var(prediction_out, 'PS') #surface pressure 
            Pmid = hyam*P0 + hybm*PS    #pi(k)=Ai(k)P0+Bi(k)PS
            Pint = hyai*P0 + hybi*PS    #pi(k)=Ai(k)P0+Bi(k)PS
            #model levels 
            
            #### save out? ####
            # torch.save(FLNS, './FLNS.pt')
            # torch.save(Ut, './Ut.pt')
            # torch.save(Vt, './Vt.pt')
            # torch.save(PS, './PS.pt')
            # torch.save(Qtot, './Qtot.pt')
            # torch.save(Tt, './Tt.pt')
            # torch.save(FSNS, './FSNS.pt')

            # you can then run the coupler and update the state:

            #accessor_output.set_state_var(state_tensor, 'U', new_u_values)

            timestep_counter += 1

            # ================================================================
            # OUTPUT GENERATION (runs in parallel via multiprocessing)
            # ================================================================

            # Convert prediction to xarray (fast, on CPU)
            upper_air, single_level = make_xarray(
                prediction.cpu(), utc_datetime, latlons.latitude.values, latlons.longitude.values, conf
            )

            # Async save to NetCDF (runs in background pool)
            pool.apply_async(
                save_netcdf_increment,
                (upper_air, single_level, init_str, lead_time_periods * forecast_hour, metadata, conf),
            )

            # ================================================================
            # SHIFT STATE FORWARD FOR NEXT TIMESTEP
            # ================================================================

            state = stepper.state_manager.shift_state_forward(state, prediction)
            forecast_hour += 1

    print("Time-stepping complete. Waiting for I/O to finish...")
    time.sleep(30)  # Allow async writes to complete

    print(f"Integration finished. Energy fixer active: {stepper.flag_energy}")
    return stepper.flag_energy


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Command-line interface for running CAMulator climate integrations."""
    parser = argparse.ArgumentParser(
        description="Run CAMulator climate integration with clean coupling interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python Quick_Climate_V02.py \\
        --config ./be21_coupled-v2025.2.0_small.yml \\
        --model_name checkpoint.pt00091.pt \\
        --save_append run_future_00091 \\
        --device cuda \\
        --init_noise 0.05
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to model configuration YAML file")
    parser.add_argument(
        "--model_name", type=str, default=None, help="Optional model checkpoint name (e.g., checkpoint.pt00091.pt)"
    )
    parser.add_argument("--save_append", type=str, default=None, help="Append subfolder name to output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument(
        "--init_noise", type=float, default=None, help="Add Gaussian noise to initial conditions (for ensembles)"
    )

    # Deprecated arguments (kept for backwards compatibility but unused)
    parser.add_argument(
        "--input_shape", type=int, nargs="+", default=None, help="[DEPRECATED] Input shape is now derived from config"
    )
    parser.add_argument(
        "--forcing_shape",
        type=int,
        nargs="+",
        default=None,
        help="[DEPRECATED] Forcing shape is now derived from config",
    )
    parser.add_argument(
        "--output_shape", type=int, nargs="+", default=None, help="[DEPRECATED] Output shape is now derived from config"
    )

    args = parser.parse_args()

    if args.input_shape or args.forcing_shape or args.output_shape:
        print("WARNING: --input_shape, --forcing_shape, --output_shape are deprecated.")
        print("         These are now automatically derived from the config file.")

    start_time = time.time()

    # Initialize CAMulator
    context = initialize_camulator(config_path=args.config, model_name=args.model_name, device=args.device)

    # Run integration with parallel I/O
    num_cpus = 8
    with mp.Pool(num_cpus) as pool:
        flag_energy = run_climate_integration(
            pool=pool, context=context, save_append=args.save_append, init_noise=args.init_noise
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print("Run completed successfully!")
    print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"Outputs saved to: {context['conf']['predict']['save_forecast']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
