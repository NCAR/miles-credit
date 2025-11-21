"""
Model_State.py
--------------
State management and variable access for CAMulator climate simulations.

This module provides modular components for:
- Variable indexing and access in state tensors
- State transformation and time-stepping
- Core CAMulator integration interface with full physics post-processing

Key Classes:
- StateVariableAccessor: Get/set variables by name from state tensors
- StateManager: Handle state transformations and time-stepping
- CAMulatorStepper: Core time-stepping interface with conservation fixers
"""

import os
import yaml
import torch
import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional, Literal, Union
import logging

# Import CREDIT components
try:
    from credit.models import load_model, load_model_name
    from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
    from credit.distributed import distributed_model_wrapper
    from credit.models.checkpoint import load_model_state
    from credit.parser import credit_main_parser
    from credit.output import load_metadata
    CREDIT_AVAILABLE = True
except ImportError:
    CREDIT_AVAILABLE = False
    logging.warning("CREDIT modules not fully available - initialization may be limited")

# Import post-processing components
try:
    from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
    POSTBLOCK_AVAILABLE = True
except ImportError:
    POSTBLOCK_AVAILABLE = False
    logging.warning("credit.postblock not available - conservation fixers disabled")

try:
    from WindPP import post_process_wind_artifacts
    WINDPP_AVAILABLE = True
except ImportError:
    WINDPP_AVAILABLE = False
    logging.warning("WindPP not available - wind artifact filtering disabled")

logger = logging.getLogger(__name__)


# ============================================================================
# STATE VARIABLE ACCESSOR - GET/SET VARIABLES BY NAME
# ============================================================================

class StateVariableAccessor:
    """
    Access variables by name in CAMulator state tensors.

    Handles the complexity of variable ordering across different tensor types:
    - 'state': Pure atmospheric state (prognostic + surface, no forcing/diagnostics)
    - 'input': Model input (state + forcing variables)
    - 'output': Model prediction (state + diagnostic variables)

    Usage:
        # Create accessor for your tensor type
        accessor = StateVariableAccessor(conf, tensor_type='state')

        # Get a variable (returns view, not copy)
        u_wind = accessor.get_state_var(state_tensor, 'U')
        # Returns: [batch, levels, time, lat, lon] for 3D variables
        #          [batch, 1, time, lat, lon] for 2D variables

        # Set a variable (modifies in-place)
        accessor.set_state_var(state_tensor, 'U', new_u_values)

        # Query variable info
        info = accessor.get_var_info('U')
        # Returns: {'start_idx': 0, 'end_idx': 32, 'n_channels': 32,
        #           'is_3d': True, 'available': True}

    Tensor Types:
        'state': [prognostic_vars * levels + surface_vars]
        'input': [forcing + prognostic_vars * levels + surface_vars]
        'output': [prognostic_vars * levels + surface_vars + diagnostic_vars]
    """

    def __init__(self, conf: dict, tensor_type: Literal['state', 'input', 'output'] = 'state'):
        """
        Initialize variable accessor.

        Args:
            conf: Configuration dictionary with data section
            tensor_type: Type of tensor to index into
                'state': Pure state without forcing or diagnostics
                'input': Model input with forcing added
                'output': Model prediction with diagnostics
        """
        self.conf = conf
        self.tensor_type = tensor_type

        # Extract variable lists from config
        self.prognostic_vars = conf["data"]["variables"]  # 3D upper-air
        self.surface_vars = conf["data"]["surface_variables"]  # 2D surface
        self.diagnostic_vars = conf["data"]["diagnostic_variables"]  # 2D diagnostics (output only)
        self.dynamic_forcing_vars = conf["data"]["dynamic_forcing_variables"]  # 2D time-varying
        self.forcing_vars = conf["data"]["forcing_variables"]  # 2D periodic
        self.static_vars = conf["data"]["static_variables"]  # 2D static

        self.levels = conf["model"]["levels"]
        self.static_first = conf["data"]["static_first"]

        # Build index maps for each tensor type
        self._build_index_maps()

    def _build_index_maps(self):
        """Build index mappings for all variables in each tensor type."""
        self.var_indices = {}

        # State tensor: [prognostic * levels + surface]
        self._build_state_indices()

        # Input tensor: [forcing + state]
        self._build_input_indices()

        # Output tensor: [prognostic * levels + surface + diagnostics]
        self._build_output_indices()

    def _build_state_indices(self):
        """Build indices for pure state tensor (no forcing, no diagnostics)."""
        indices = {}
        idx = 0

        # Prognostic variables (3D with levels)
        for var in self.prognostic_vars:
            indices[var] = {
                'start_idx': idx,
                'end_idx': idx + self.levels,
                'n_channels': self.levels,
                'is_3d': True,
                'available': True
            }
            idx += self.levels

        # Surface variables (2D)
        for var in self.surface_vars:
            indices[var] = {
                'start_idx': idx,
                'end_idx': idx + 1,
                'n_channels': 1,
                'is_3d': False,
                'available': True
            }
            idx += 1

        # Mark diagnostics as not available in state
        for var in self.diagnostic_vars:
            indices[var] = {'available': False, 'reason': 'Diagnostics not in state tensor'}

        # Mark forcing as not available in state
        for var in self.dynamic_forcing_vars + self.forcing_vars + self.static_vars:
            indices[var] = {'available': False, 'reason': 'Forcing not in state tensor'}

        self.var_indices['state'] = indices

    def _build_input_indices(self):
        """Build indices for model input tensor (with forcing).

        Input tensor structure: [state] + [forcing]
        Where state = prognostic + surface
        """
        indices = {}
        idx = 0

        # FIRST: Prognostic variables (3D with levels) - part of state
        for var in self.prognostic_vars:
            indices[var] = {
                'start_idx': idx,
                'end_idx': idx + self.levels,
                'n_channels': self.levels,
                'is_3d': True,
                'available': True
            }
            idx += self.levels

        # SECOND: Surface variables (2D) - part of state
        for var in self.surface_vars:
            indices[var] = {
                'start_idx': idx,
                'end_idx': idx + 1,
                'n_channels': 1,
                'is_3d': False,
                'available': True
            }
            idx += 1

        # THIRD: Forcing variables - appended after state
        # Order depends on static_first flag
        if self.static_first:
            forcing_order = self.static_vars + self.dynamic_forcing_vars + self.forcing_vars
        else:
            forcing_order = self.dynamic_forcing_vars + self.forcing_vars + self.static_vars

        # Add all forcing variables (2D)
        for var in forcing_order:
            indices[var] = {
                'start_idx': idx,
                'end_idx': idx + 1,
                'n_channels': 1,
                'is_3d': False,
                'available': True
            }
            idx += 1

        # Mark diagnostics as not available in input
        for var in self.diagnostic_vars:
            indices[var] = {'available': False, 'reason': 'Diagnostics not in input tensor'}

        self.var_indices['input'] = indices

    def _build_output_indices(self):
        """Build indices for model output tensor (with diagnostics)."""
        indices = {}
        idx = 0

        # Prognostic variables (3D with levels)
        for var in self.prognostic_vars:
            indices[var] = {
                'start_idx': idx,
                'end_idx': idx + self.levels,
                'n_channels': self.levels,
                'is_3d': True,
                'available': True
            }
            idx += self.levels

        # Surface variables (2D)
        for var in self.surface_vars:
            indices[var] = {
                'start_idx': idx,
                'end_idx': idx + 1,
                'n_channels': 1,
                'is_3d': False,
                'available': True
            }
            idx += 1

        # Diagnostic variables (2D)
        for var in self.diagnostic_vars:
            indices[var] = {
                'start_idx': idx,
                'end_idx': idx + 1,
                'n_channels': 1,
                'is_3d': False,
                'available': True
            }
            idx += 1

        # Mark forcing as not available in output
        for var in self.dynamic_forcing_vars + self.forcing_vars + self.static_vars:
            indices[var] = {'available': False, 'reason': 'Forcing not in output tensor'}

        self.var_indices['output'] = indices

    def get_var_info(self, var_name: str) -> Dict:
        """
        Get indexing information for a variable.

        Args:
            var_name: Variable name (e.g., 'U', 'TAUX', 'PS')

        Returns:
            Dictionary with variable info:
                - available: Whether variable exists in current tensor type
                - start_idx, end_idx, n_channels, is_3d: If available
                - reason: If not available, why not

        Raises:
            ValueError: If variable name not recognized in config
        """
        indices = self.var_indices[self.tensor_type]

        if var_name not in indices:
            all_vars = (self.prognostic_vars + self.surface_vars +
                       self.diagnostic_vars + self.dynamic_forcing_vars +
                       self.forcing_vars + self.static_vars)
            raise ValueError(
                f"Variable '{var_name}' not found in config. "
                f"Available variables: {all_vars}"
            )

        return indices[var_name]

    def get_state_var(self, state_tensor: torch.Tensor, var_name: str,
                      time_idx: Optional[int] = None) -> torch.Tensor:
        """
        Extract a variable from the state tensor.

        Args:
            state_tensor: State tensor [batch, channels, time, lat, lon]
            var_name: Variable name (e.g., 'U', 'TAUX', 'PS')
            time_idx: Optional time index to extract. If None, returns all times.

        Returns:
            Variable tensor:
                - 3D vars: [batch, levels, time, lat, lon] or [batch, levels, lat, lon]
                - 2D vars: [batch, 1, time, lat, lon] or [batch, 1, lat, lon]

        Raises:
            ValueError: If variable not available in current tensor type
            IndexError: If time_idx out of bounds
        """
        info = self.get_var_info(var_name)

        if not info['available']:
            raise ValueError(
                f"Variable '{var_name}' not available in '{self.tensor_type}' tensor. "
                f"Reason: {info.get('reason', 'Unknown')}"
            )

        # Extract variable slice
        var_slice = state_tensor[:, info['start_idx']:info['end_idx'], ...]

        # Extract specific time if requested
        if time_idx is not None:
            if time_idx >= state_tensor.shape[2]:
                raise IndexError(
                    f"Time index {time_idx} out of bounds for tensor with "
                    f"{state_tensor.shape[2]} time steps"
                )
            var_slice = var_slice[:, :, time_idx, :, :]

        return var_slice

    def set_state_var(self, state_tensor: torch.Tensor, var_name: str,
                      var_data: torch.Tensor, time_idx: Optional[int] = None) -> None:
        """
        Set a variable in the state tensor (in-place modification).

        Args:
            state_tensor: State tensor to modify [batch, channels, time, lat, lon]
            var_name: Variable name (e.g., 'U', 'TAUX', 'PS')
            var_data: New data for variable. Must match expected shape.
            time_idx: Optional time index to set. If None, sets all times.

        Raises:
            ValueError: If variable not available or shape mismatch
        """
        info = self.get_var_info(var_name)

        if not info['available']:
            raise ValueError(
                f"Variable '{var_name}' not available in '{self.tensor_type}' tensor. "
                f"Reason: {info.get('reason', 'Unknown')}"
            )

        # Validate shape
        if time_idx is None:
            expected_shape = (
                state_tensor.shape[0],  # batch
                info['n_channels'],      # channels (levels or 1)
                state_tensor.shape[2],  # time
                state_tensor.shape[3],  # lat
                state_tensor.shape[4]   # lon
            )
        else:
            expected_shape = (
                state_tensor.shape[0],  # batch
                info['n_channels'],      # channels
                state_tensor.shape[3],  # lat
                state_tensor.shape[4]   # lon
            )

        if var_data.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for '{var_name}'. Expected {expected_shape}, "
                f"got {var_data.shape}"
            )

        # Set variable (in-place)
        if time_idx is None:
            state_tensor[:, info['start_idx']:info['end_idx'], ...] = var_data
        else:
            state_tensor[:, info['start_idx']:info['end_idx'], time_idx, :, :] = var_data

    def list_available_vars(self) -> Dict[str, Dict]:
        """
        List all variables available in current tensor type.

        Returns:
            Dictionary mapping variable names to their info dicts
        """
        return {
            var: info for var, info in self.var_indices[self.tensor_type].items()
            if info.get('available', False)
        }


# ============================================================================
# STATE MANAGER - TRANSFORMATIONS AND TIME-STEPPING
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

    Attributes:
        model: PyTorch model for forward integration
        state_manager: StateManager instance for state transformations
        device: torch.device for computation
    """

    def __init__(self, model: torch.nn.Module, conf: dict, device: torch.device):
        """
        Initialize the CAMulator stepper with full post-processing.

        Args:
            model: Trained CAMulator model (already on device, in eval mode)
            conf: Full configuration dictionary
            device: Device for computation (cuda/cpu)
        """
        self.model = model
        self.conf = conf
        self.device = device
        self.state_manager = StateManager(conf)

        # Create variable accessors for convenience
        self.state_accessor = StateVariableAccessor(conf, tensor_type='state')
        self.input_accessor = StateVariableAccessor(conf, tensor_type='input')
        self.output_accessor = StateVariableAccessor(conf, tensor_type='output')

        # Setup post-processing components (conservation fixers, wind filtering)
        self._setup_postprocessing()

    def _setup_postprocessing(self):
        """
        Initialize post-processing components (conservation fixers and wind filtering).

        Sets up:
        - Global mass fixer (dry air mass conservation)
        - Global water fixer (water mass conservation)
        - Global energy fixer (total energy conservation)
        - Wind artifact filtering flags
        """
        post_conf = self.conf["model"]["post_conf"]

        # Check which conservation fixers are enabled
        self.flag_mass = (POSTBLOCK_AVAILABLE and post_conf["activate"] and
                         post_conf["global_mass_fixer"]["activate"])
        self.flag_water = (POSTBLOCK_AVAILABLE and post_conf["activate"] and
                          post_conf["global_water_fixer"]["activate"])
        self.flag_energy = (POSTBLOCK_AVAILABLE and post_conf["activate"] and
                           post_conf["global_energy_fixer"]["activate"])

        # Initialize conservation fixers
        if self.flag_mass:
            self.opt_mass = GlobalMassFixer(post_conf)
            logger.info("Global mass fixer initialized")
        if self.flag_water:
            self.opt_water = GlobalWaterFixer(post_conf)
            logger.info("Global water fixer initialized")
        if self.flag_energy:
            self.opt_energy = GlobalEnergyFixer(post_conf)
            logger.info("Global energy fixer initialized")

        # Wind filtering flag
        self.enable_wind_filtering = WINDPP_AVAILABLE

    def step(self, state: torch.Tensor, dynamic_forcing: torch.Tensor,
             static_forcing: torch.Tensor) -> torch.Tensor:
        """
        Advance the atmospheric state by one model timestep.

        This is the core coupling interface - a pure function that takes
        atmospheric state and returns the next state with full physics
        post-processing applied.

        Args:
            state: Current atmospheric state [batch, state_channels, time, lat, lon]
            dynamic_forcing: Time-varying forcing [batch, dyn_channels, time, lat, lon]
            static_forcing: Static forcing [batch, static_channels, time, lat, lon]

        Returns:
            prediction: Next atmospheric state [batch, output_channels, 1, lat, lon]
                       Includes prognostic, surface, and diagnostic variables with
                       conservation fixers and wind filtering applied
        """
        # Build model input by combining state with forcing
        model_input = self.state_manager.build_input_with_forcing(
            state, dynamic_forcing, static_forcing
        )

        # Run model inference
        with torch.no_grad():
            prediction = self.model(model_input.float())

        # Apply post-processing (conservation fixers and wind filtering)
        prediction = self._apply_postprocessing(prediction, model_input)

        return prediction

    def _apply_postprocessing(self, prediction: torch.Tensor,
                             model_input: torch.Tensor) -> torch.Tensor:
        """
        Apply wind artifact filtering and conservation fixers.

        Post-processing order:
        1. Wind artifact filtering (removes spurious wind patterns near surface)
        2. Global mass fixer (conserves dry air mass)
        3. Global water fixer (conserves total water mass)
        4. Global energy fixer (conserves total energy)

        Args:
            prediction: Raw model prediction [batch, output_channels, 1, lat, lon]
            model_input: Model input tensor (needed for some fixers)

        Returns:
            prediction: Post-processed prediction with conservation applied
        """
        # Wind artifact filtering
        if self.enable_wind_filtering:
            post_process_wind_artifacts(prediction, self.conf, enable_filtering=True)

        # Apply conservation fixers in sequence
        if self.flag_mass:
            prediction = self.opt_mass({"y_pred": prediction, "x": model_input})["y_pred"]

        if self.flag_water:
            prediction = self.opt_water({"y_pred": prediction, "x": model_input})["y_pred"]

        if self.flag_energy:
            prediction = self.opt_energy({"y_pred": prediction, "x": model_input})["y_pred"]

        return prediction

    def get_state_var(self, tensor: torch.Tensor, var_name: str,
                     tensor_type: Literal['state', 'input', 'output'] = 'state',
                     time_idx: Optional[int] = None) -> torch.Tensor:
        """
        Convenience method to get a variable from any tensor type.

        Args:
            tensor: Tensor to extract from
            var_name: Variable name
            tensor_type: Type of tensor ('state', 'input', or 'output')
            time_idx: Optional time index

        Returns:
            Variable tensor
        """
        accessor = {
            'state': self.state_accessor,
            'input': self.input_accessor,
            'output': self.output_accessor
        }[tensor_type]

        return accessor.get_state_var(tensor, var_name, time_idx)

    def set_state_var(self, tensor: torch.Tensor, var_name: str, var_data: torch.Tensor,
                     tensor_type: Literal['state', 'input', 'output'] = 'state',
                     time_idx: Optional[int] = None) -> None:
        """
        Convenience method to set a variable in any tensor type.

        Args:
            tensor: Tensor to modify
            var_name: Variable name
            var_data: New data
            tensor_type: Type of tensor ('state', 'input', or 'output')
            time_idx: Optional time index
        """
        accessor = {
            'state': self.state_accessor,
            'input': self.input_accessor,
            'output': self.output_accessor
        }[tensor_type]

        accessor.set_state_var(tensor, var_name, var_data, time_idx)


# ============================================================================
# INITIALIZATION - ONE-TIME SETUP FOR CAMULATOR
# ============================================================================

def initialize_camulator(config_path: str, model_name: str = None, device: str = 'cuda') -> dict:
    """
    One-time initialization of CAMulator model and all supporting components.

    This function loads the model, transforms, forcing data, and sets up everything
    needed for CAMulator integration. Separate from the time-stepping loop for
    cleaner coupling interfaces.

    Args:
        config_path: Path to YAML configuration file
        model_name: Optional specific checkpoint name (e.g., 'checkpoint.pt00091.pt')
                   If None, uses default 'checkpoint.pt'
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        context: Dictionary containing all initialized components:
            - 'model': Loaded CAMulator model in eval mode
            - 'stepper': CAMulatorStepper instance ready for time-stepping
            - 'conf': Parsed configuration dictionary
            - 'state_transformer': Normalization/denormalization transforms
            - 'forcing_dataset': xarray dataset with normalized forcing data
            - 'static_forcing': Static forcing tensor (topography, LSM, etc.)
            - 'initial_state': Initial condition tensor loaded from file
            - 'latlons': Latitude/longitude coordinates
            - 'metadata': Variable metadata for NetCDF output
            - 'device': torch device object

    Example:
        >>> ctx = initialize_camulator('camulator_config.yml',
        ...                           model_name='checkpoint.pt00091.pt')
        >>> stepper = ctx['stepper']
        >>> state = ctx['initial_state']
        >>>
        >>> # Run simulation
        >>> for timestep in range(num_steps):
        ...     dynamic_forcing = get_forcing(timestep, ctx['forcing_dataset'])
        ...     prediction = stepper.step(state, dynamic_forcing, ctx['static_forcing'])
        ...     state = stepper.state_manager.shift_state_forward(state, prediction)

    Raises:
        FileNotFoundError: If config file or checkpoint not found
        ImportError: If required CREDIT modules not available
    """
    if not CREDIT_AVAILABLE:
        raise ImportError(
            "CREDIT modules not available. Cannot initialize CAMulator. "
            "Please ensure credit package is installed and importable."
        )

    print(f'Initializing CAMulator from config: {config_path}')

    # Load and parse configuration
    with open(config_path) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    conf["predict"]["mode"] = None  # Override to None for single-GPU inference

    device = torch.device(device)
    print(f'Using device: {device}')

    # Load transforms and normalization
    print('Loading transforms...')
    transform = load_transforms(conf)

    if conf["data"]["scaler_type"] == "std_new":
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        raise ValueError(f"Unsupported scaler_type: {conf['data']['scaler_type']}")

    # Load model
    print(f'Loading model: {model_name if model_name else "checkpoint.pt (default)"}')
    if model_name:
        model = load_model_name(conf, model_name, load_weights=True).to(device)
    else:
        model = load_model(conf, load_weights=True).to(device)

    # Handle distributed mode if specified (typically not used for climate runs)
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        logger.info(f"Setting up distributed mode: {conf['predict']['mode']}")
        model = distributed_model_wrapper(conf, model, device)
        if conf["predict"]["mode"] == "fsdp":
            model = load_model_state(conf, model, device)

    model.eval()
    print('Model loaded and set to eval mode')

    # Load initial conditions
    print('Loading initial conditions...')
    ic_path = conf['predict']['init_cond_fast_climate']
    if not os.path.exists(ic_path):
        raise FileNotFoundError(
            f"Initial condition file not found: {ic_path}\n"
            f"Please run Make_Climate_Initial_Conditions.py first."
        )

    initial_state = torch.load(ic_path, map_location=device).to(device)
    print(f'Initial state shape: {initial_state.shape}')

    # Load forcing data
    print('Loading forcing data...')
    forcing_file = conf["predict"]["forcing_file"]
    if not os.path.exists(forcing_file):
        raise FileNotFoundError(f"Forcing file not found: {forcing_file}")

    chunk_size = conf["data"].get("forcing_chunk_size", 32)
    forcing_ds = xr.open_dataset(forcing_file, chunks={"time": chunk_size})

    # Normalize forcing data
    print('Normalizing forcing data...')
    forcing_ds_norm = state_transformer.transform_dataset(forcing_ds)
    forcing_ds_norm = forcing_ds_norm.chunk({"time": chunk_size})

    # Load static forcing (topography, land-sea mask, etc.)
    print('Loading static forcing...')
    sf_vars = conf["data"]["static_variables"]
    static_arr = np.stack([forcing_ds[s].values for s in sf_vars], axis=0)
    static_forcing = (torch.from_numpy(static_arr).unsqueeze(0)).unsqueeze(2).to(device, non_blocking=True)
    print(f'Static forcing shape: {static_forcing.shape}')

    # Load metadata and coordinates
    print('Loading metadata and coordinates...')
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])
    metadata = load_metadata(conf)

    # Create CAMulatorStepper with full post-processing
    print('Creating CAMulatorStepper with conservation fixers...')
    stepper = CAMulatorStepper(model, conf, device)

    print('=' * 70)
    print('Initialization complete!')
    print(f'Model device: {device}')
    print(f'State shape: {initial_state.shape}')
    print(f'Static forcing: {len(sf_vars)} variables')
    print(f'Conservation fixers: Mass={stepper.flag_mass}, Water={stepper.flag_water}, Energy={stepper.flag_energy}')
    print(f'Wind filtering: {stepper.enable_wind_filtering}')
    print('=' * 70)

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
