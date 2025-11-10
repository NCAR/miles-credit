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
import yaml
import time
import logging
import warnings
from pathlib import Path
import multiprocessing as mp
import argparse

from WindPP import post_process_wind_artifacts

# ---------- #
# Numerics
from datetime import datetime
import xarray as xr
import numpy as np

# ---------- #
import torch

# ---------- #
# credit
from credit.models import load_model, load_model_name
from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.distributed import distributed_model_wrapper
from credit.models.checkpoint import load_model_state
from credit.parser import credit_main_parser
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class StateManager:
    """
    Manages the CAMulator state tensor structure and transformations.

    State Tensor Structure
    ----------------------
    The state tensor contains atmospheric variables over multiple timesteps (history).

    Dimensions: [batch, channels, time, lat, lon]

    IMPORTANT FOR COUPLING:
    -----------------------
    The INITIAL state loaded from file already includes forcing for the first timestep.
    After the first step, shift_state_forward() returns atmospheric state WITHOUT forcing,
    so you must call build_input_with_forcing() before the next model step.

    Example usage:
        # First timestep: initial_state already has forcing
        prediction = model(initial_state)
        state = shift_state_forward(initial_state, prediction)

        # Subsequent timesteps: must add forcing
        model_input = build_input_with_forcing(state, forcing, static)
        prediction = model(model_input)
        state = shift_state_forward(model_input, prediction)

    Channel ordering depends on config['data']['static_first']:

    If static_first == True:
        - Static variables (e.g., Z_GDS4_SFC, LSM) - replicated across time
        - Dynamic forcing (e.g., tsi) - varies per timestep
        - [prognostic + surface + diagnostic variables] - varies per timestep

    If static_first == False (default):
        - Dynamic forcing (e.g., tsi) - varies per timestep
        - Static variables (e.g., Z_GDS4_SFC, LSM) - replicated across time
        - [prognostic + surface + diagnostic variables] - varies per timestep

    Note: Diagnostic variables are OUTPUT ONLY and excluded when shifting state forward.
    """

    def __init__(self, conf):
        self.conf = conf
        self.history_len = conf["data"]["history_len"]
        self.varnum_diag = len(conf["data"]["diagnostic_variables"])
        self.static_dim = len(conf["data"]["static_variables"]) if not conf["data"]["static_first"] else 0
        self.static_first = conf["data"]["static_first"]

    def shift_state_forward(self, state: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Roll the state tensor forward by one timestep.

        Args:
            state: Current state [batch, channels, time, lat, lon]
            prediction: Model prediction for next timestep [batch, channels, 1, lat, lon]

        Returns:
            new_state: State ready for next model call [batch, channels, time, lat, lon]
        """
        if self.history_len == 1:
            # Single timestep history: just return prediction (excluding diagnostics)
            if self.varnum_diag > 0:
                return prediction[:, :-self.varnum_diag, ...].detach()
            else:
                return prediction.detach()
        else:
            # Multi-timestep history: shift time dimension and append new prediction
            if self.static_dim == 0:
                # All variables shift in time
                state_detach = state[:, :, 1:, ...].detach()
            else:
                # Static variables stay fixed, only shift dynamic ones
                state_detach = state[:, :-self.static_dim, 1:, ...].detach()

            # Append new prediction (excluding diagnostic variables)
            if self.varnum_diag > 0:
                new_pred = prediction[:, :-self.varnum_diag, ...].detach()
            else:
                new_pred = prediction.detach()

            return torch.cat([state_detach, new_pred], dim=2)

    def build_input_with_forcing(self, state: torch.Tensor, dynamic_forcing: torch.Tensor,
                                 static_forcing: torch.Tensor) -> torch.Tensor:
        """
        Combine state with forcing variables to create model input.

        Args:
            state: Current atmospheric state
            dynamic_forcing: Time-varying forcing (e.g., solar radiation)
            static_forcing: Fixed fields (e.g., topography, land-sea mask)

        Returns:
            input_tensor: Ready for model forward pass
        """
        if self.static_first:
            forcing = torch.cat((static_forcing, dynamic_forcing), dim=1)
        else:
            forcing = torch.cat((dynamic_forcing, static_forcing), dim=1)

        return torch.cat((state, forcing), dim=1)


# ============================================================================
# CAMULATOR STEPPER - THE CORE INTERFACE FOR COUPLING
# ============================================================================

class CAMulatorStepper:
    """
    Core CAMulator time-stepping interface suitable for coupling to other models.

    This class isolates the physics integration step from I/O, initialization,
    and post-processing setup. Suitable for coupling to ocean models or other
    Earth system components.

    Usage:
        stepper = CAMulatorStepper(model, conf, device)

        for timestep in range(num_steps):
            # Get forcing for this timestep from your coupler
            dynamic_forcing = get_dynamic_forcing(timestep)

            # Step the atmosphere forward
            prediction = stepper.step(state, dynamic_forcing, static_forcing)

            # Update state for next step
            state = stepper.state_manager.shift_state_forward(state, prediction)
    """

    def __init__(self, model, conf, device):
        """
        Initialize the stepper with a loaded model.

        Args:
            model: Loaded CAMulator model (already on device, in eval mode)
            conf: Full configuration dictionary
            device: torch device (cuda or cpu)
        """
        self.model = model
        self.conf = conf
        self.device = device
        self.state_manager = StateManager(conf)

        # Setup post-processing components
        self._setup_postprocessing()

    def _setup_postprocessing(self):
        """Initialize post-processing components (mass/water/energy fixers, wind filtering)."""
        post_conf = self.conf["model"]["post_conf"]

        self.flag_mass = post_conf["activate"] and post_conf["global_mass_fixer"]["activate"]
        self.flag_water = post_conf["activate"] and post_conf["global_water_fixer"]["activate"]
        self.flag_energy = post_conf["activate"] and post_conf["global_energy_fixer"]["activate"]

        if self.flag_mass:
            self.opt_mass = GlobalMassFixer(post_conf)
        if self.flag_water:
            self.opt_water = GlobalWaterFixer(post_conf)
        if self.flag_energy:
            self.opt_energy = GlobalEnergyFixer(post_conf)

    def step(self, state: torch.Tensor, dynamic_forcing: torch.Tensor,
             static_forcing: torch.Tensor) -> torch.Tensor:
        """
        Advance the atmospheric state by one model timestep.

        This is the core coupling interface - a pure function that takes
        atmospheric state and returns the next state.

        Args:
            state: Atmospheric state tensor [batch, state_channels, history, lat, lon]
            dynamic_forcing: Time-varying forcing [batch, dyn_forcing_channels, 1, lat, lon]
            static_forcing: Static fields [batch, static_channels, 1, lat, lon]

        Returns:
            prediction: Next atmospheric state [batch, output_channels, 1, lat, lon]
                       Includes prognostic, surface, and diagnostic variables
        """
        # Build model input by combining state with forcing
        model_input = self.state_manager.build_input_with_forcing(
            state, dynamic_forcing, static_forcing
        )

        # Run model inference
        with torch.no_grad():
            prediction = self.model(model_input.float())

        # Apply post-processing filters
        prediction = self._apply_postprocessing(prediction, model_input)

        return prediction

    def _apply_postprocessing(self, prediction: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
        """Apply wind filtering and conservation fixers."""
        # Wind artifact filtering
        post_process_wind_artifacts(prediction, self.conf, enable_filtering=True)

        # Conservation fixers
        if self.flag_mass:
            prediction = self.opt_mass({"y_pred": prediction, "x": model_input})["y_pred"]
        if self.flag_water:
            prediction = self.opt_water({"y_pred": prediction, "x": model_input})["y_pred"]
        if self.flag_energy:
            prediction = self.opt_energy({"y_pred": prediction, "x": model_input})["y_pred"]

        return prediction


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_camulator(config_path: str, model_name: str = None, device: str = 'cuda') -> dict:
    """
    One-time initialization of CAMulator model and all supporting components.

    This function loads the model, transforms, forcing data, and sets up everything
    needed for integration. Separate from the time-stepping loop for cleaner coupling.

    Args:
        config_path: Path to YAML configuration file
        model_name: Optional specific checkpoint name (e.g., 'checkpoint.pt00091.pt')
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        context: Dictionary containing:
            - 'model': Loaded model in eval mode
            - 'stepper': CAMulatorStepper instance ready for stepping
            - 'conf': Parsed configuration
            - 'state_transformer': Normalization/denormalization transforms
            - 'forcing_dataset': xarray dataset with forcing data
            - 'static_forcing': Static forcing tensor (topography, LSM, etc.)
            - 'initial_state': Initial condition tensor
            - 'latlons': Latitude/longitude coordinates
            - 'metadata': Metadata for output variables
            - 'device': torch device
    """
    print(f'Initializing CAMulator from config: {config_path}')

    # Load and parse configuration
    with open(config_path) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    conf["predict"]["mode"] = None

    device = torch.device(device)

    print('Loading transforms...')
    transform = load_transforms(conf)
    state_transformer = Normalize_ERA5_and_Forcing(conf) if conf["data"]["scaler_type"] == "std_new" else None

    print(f'Loading model: {model_name if model_name else "default"}')
    if model_name:
        model = load_model_name(conf, model_name, load_weights=True).to(device)
    else:
        model = load_model(conf, load_weights=True).to(device)

    # Handle distributed mode if specified
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        model = distributed_model_wrapper(conf, model, device)
        if conf["predict"]["mode"] == "fsdp":
            model = load_model_state(conf, model, device)

    model.eval()

    print('Loading initial conditions...')
    initial_state = torch.load(
        conf['predict']['init_cond_fast_climate'],
        map_location=device
    ).to(device)

    print('Loading forcing data...')
    chunk_size = conf["data"].get("forcing_chunk_size", 32)
    forcing_ds = xr.open_dataset(conf["predict"]["forcing_file"], chunks={"time": chunk_size})

    print('Normalizing forcing data...')
    forcing_ds_norm = state_transformer.transform_dataset(forcing_ds)
    forcing_ds_norm = forcing_ds_norm.chunk({"time": chunk_size})

    print('Loading static forcing...')
    sf_vars = conf["data"]["static_variables"]
    static_arr = np.stack([forcing_ds[s].values for s in sf_vars], axis=0)
    static_forcing = (torch.from_numpy(static_arr).unsqueeze(0)).unsqueeze(2).to(device, non_blocking=True)
    print(f'Static forcing shape: {static_forcing.shape}')

    print('Loading metadata and coordinates...')
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])
    metadata = load_metadata(conf)

    print('Creating CAMulatorStepper...')
    stepper = CAMulatorStepper(model, conf, device)

    print('Initialization complete!')

    return {
        'model': model,
        'stepper': stepper,
        'conf': conf,
        'state_transformer': state_transformer,
        'forcing_dataset': forcing_ds_norm,
        'static_forcing': static_forcing,
        'initial_state': initial_state,
        'latlons': latlons,
        'metadata': metadata,
        'device': device
    }


def add_init_noise(state: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    Add random noise to initial conditions for ensemble generation.

    Args:
        state: Initial state tensor
        noise_std: Standard deviation of Gaussian noise

    Returns:
        state_with_noise: Perturbed state
    """
    print(f'Adding initial condition noise (std={noise_std})')
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
    raw_dt = conf['predict']['start_datetime']

    if isinstance(raw_dt, str):
        # Parse "YYYY-MM-DD HH:MM:SS" format
        return datetime.strptime(raw_dt, '%Y-%m-%d %H:%M:%S')
    elif isinstance(raw_dt, datetime):
        # Already a Python datetime
        return raw_dt
    else:
        # Assume it's a cftime object - convert to Python datetime
        # cftime objects have year, month, day, hour, minute, second attributes
        return datetime(raw_dt.year, raw_dt.month, raw_dt.day,
                       raw_dt.hour, raw_dt.minute, raw_dt.second)


# ============================================================================
# INTEGRATION LOOP
# ============================================================================

def run_climate_integration(pool: mp.Pool, context: dict, save_append: str = None,
                            init_noise: float = None):
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
    conf = context['conf']
    stepper = context['stepper']
    forcing_ds_norm = context['forcing_dataset']
    static_forcing = context['static_forcing']
    state = context['initial_state']
    latlons = context['latlons']
    metadata = context['metadata']
    device = context['device']

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
    print('Tracing model with torch.jit...')
    # IMPORTANT: Initial state already contains forcing for first timestep
    # So we trace with the initial state shape as-is (DO NOT add forcing channels)
    dummy_input = torch.zeros_like(state)
    traced_model = torch.jit.trace(stepper.model, dummy_input)
    stepper.model = traced_model
    print(f'Model traced with input shape: {dummy_input.shape}')

    # Setup for time-stepping
    df_vars = conf["data"]["dynamic_forcing_variables"]
    num_ts = conf["predict"]["timesteps_fast_climate"]
    lead_time_periods = conf["data"]["lead_time_periods"]
    chunk_size = conf["data"].get("forcing_chunk_size", 32)

    # Get forcing data subset
    dynamic_ds = forcing_ds_norm[df_vars]

    # IMPORTANT: Use the config's datetime object directly for xarray lookup
    # It might be cftime.DatetimeNoLeap, which xarray expects
    start_datetime_raw = conf['predict']['start_datetime']
    loc = dynamic_ds.indexes['time'].get_loc(start_datetime_raw)
    start_ix = loc.start if isinstance(loc, slice) else loc
    print(f"Starting integration at time index: {start_ix}")

    # Now convert to Python datetime for output formatting (if it's a string or cftime)
    init_dt = parse_datetime_from_config(conf)
    init_str = init_dt.strftime('%Y-%m-%dT%HZ')

    # ========================================================================
    # MAIN TIME-STEPPING LOOP
    # ========================================================================

    print('Starting time-stepping loop...')
    forecast_hour = 1
    timestep_counter = 0

    for block_start in range(start_ix, start_ix + num_ts, chunk_size):
        block_end = min(block_start + chunk_size, start_ix + num_ts)

        # Load chunk of dynamic forcing data
        ds_slice = dynamic_ds.isel(time=slice(block_start, block_end)).load()
        ds_slice_times = ds_slice['time'].values

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
            if hasattr(time_obj, 'item'):
                time_obj = time_obj.item()

            if isinstance(time_obj, datetime):
                utc_datetime = time_obj
            else:
                # cftime object - convert to Python datetime
                utc_datetime = datetime(time_obj.year, time_obj.month, time_obj.day,
                                       time_obj.hour, time_obj.minute, time_obj.second)

            if (timestep_counter + 1) % 20 == 0:
                print(f'Model step: {timestep_counter+1:05}, time: {utc_datetime}')

            dynamic_forcing_t = gpu_forcing_chunk[t].unsqueeze(0)

            # ================================================================
            # CORE PHYSICS STEP
            # This matches the original Quick_Climate.py logic exactly:
            # - First step (timestep_counter=0): state already has forcing, run model as-is
            # - Subsequent steps: add forcing to state, then run model
            # ================================================================

            if timestep_counter != 0:
                # Build forcing from dynamic + static
                model_input = stepper.state_manager.build_input_with_forcing(
                    state, dynamic_forcing_t, static_forcing
                )
            else:
                # First iteration: initial state already contains forcing
                model_input = state

            # Run model
            with torch.no_grad():
                prediction = stepper.model(model_input.float())

            # Apply post-processing
            prediction = stepper._apply_postprocessing(prediction, model_input)

            timestep_counter += 1

            # ================================================================
            # OUTPUT GENERATION (runs in parallel via multiprocessing)
            # ================================================================

            # Convert prediction to xarray (fast, on CPU)
            upper_air, single_level = make_xarray(
                prediction.cpu(),
                utc_datetime,
                latlons.latitude.values,
                latlons.longitude.values,
                conf
            )

            # Async save to NetCDF (runs in background pool)
            pool.apply_async(
                save_netcdf_increment,
                (upper_air, single_level, init_str, lead_time_periods * forecast_hour,
                 metadata, conf)
            )

            # ================================================================
            # SHIFT STATE FORWARD FOR NEXT TIMESTEP
            # ================================================================

            state = stepper.state_manager.shift_state_forward(state, prediction)
            forecast_hour += 1

    print('Time-stepping complete. Waiting for I/O to finish...')
    time.sleep(30)  # Allow async writes to complete

    print(f'Integration finished. Energy fixer active: {stepper.flag_energy}')
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
        """
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration YAML file')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Optional model checkpoint name (e.g., checkpoint.pt00091.pt)')
    parser.add_argument('--save_append', type=str, default=None,
                       help='Append subfolder name to output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--init_noise', type=float, default=None,
                       help='Add Gaussian noise to initial conditions (for ensembles)')

    # Deprecated arguments (kept for backwards compatibility but unused)
    parser.add_argument('--input_shape', type=int, nargs='+', default=None,
                       help='[DEPRECATED] Input shape is now derived from config')
    parser.add_argument('--forcing_shape', type=int, nargs='+', default=None,
                       help='[DEPRECATED] Forcing shape is now derived from config')
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                       help='[DEPRECATED] Output shape is now derived from config')

    args = parser.parse_args()

    if args.input_shape or args.forcing_shape or args.output_shape:
        print("WARNING: --input_shape, --forcing_shape, --output_shape are deprecated.")
        print("         These are now automatically derived from the config file.")

    start_time = time.time()

    # Initialize CAMulator
    context = initialize_camulator(
        config_path=args.config,
        model_name=args.model_name,
        device=args.device
    )

    # Run integration with parallel I/O
    num_cpus = 8
    with mp.Pool(num_cpus) as pool:
        flag_energy = run_climate_integration(
            pool=pool,
            context=context,
            save_append=args.save_append,
            init_noise=args.init_noise
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'='*60}")
    print(f"Run completed successfully!")
    print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Outputs saved to: {context['conf']['predict']['save_forecast']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
