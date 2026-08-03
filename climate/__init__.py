"""
Climate simulation package for CREDIT.

This package provides tools for running climate simulations with AI weather models,
including coupled CAMulator integration, initial condition generation, and post-processing.

Main Components
---------------
Quick_Climate : module
    Core climate integration with CAMulator coupling
    - StateManager: Manages model state across timesteps
    - CAMulatorStepper: Handles coupled physics integration
    - initialize_camulator: Initialize the climate integration system
    - run_climate_integration: Run multi-year climate simulations

Make_Climate_Initial_Conditions : module
    Generate initial conditions for climate runs from reanalysis data

output : module
    NetCDF output utilities for climate data
    - save_netcdf_increment: Save forecast output incrementally
    - make_xarray: Convert tensors to xarray datasets
    - load_metadata: Load variable metadata

Post_Process : module
    Post-processing utilities for climate output
    - post_process: Apply spatial/temporal averaging and rescaling
    - rescale_file: Denormalize variables

Post_Process_Parallel : module
    Parallel version of post-processing utilities

WindPP : module
    Wind artifact filtering and post-processing
    - WindArtifactFilterConfig: Configuration for wind filtering
    - post_process_wind_artifacts: Remove spurious wind artifacts
    - wind_filter: Apply smoothing filters to wind fields

COUPLING_EXAMPLE : module
    Example scripts demonstrating CAMulator coupling patterns
    - example_standalone_integration: Run without coupling
    - example_coupled_system: Full coupled integration
    - example_inspect_state: Debug state variables
    - example_custom_forcing: Custom forcing patterns

Deprecated Modules
------------------
Quick_Climate_Deprecated : Legacy implementation (use Quick_Climate instead)
Quick_Climate_Ensembles : Ensemble climate simulations
Quick_Climate_Infite : Infinite-length climate runs
"""

# Core climate integration
# Output utilities
from credit.output import (
    load_metadata,
    make_xarray,
    save_netcdf_increment,
    split_and_reshape,
)

# Parallel post-processing (import as submodule to avoid conflicts)
# Coupling examples (import as submodule)
from . import COUPLING_EXAMPLE, Post_Process_Parallel

# Post-processing
from .Post_Process import (
    add_hours_noleap,
    extract_time_single,
    post_process,
    rescale_file,
)
from .Quick_Climate import (
    CAMulatorStepper,
    StateManager,
    add_init_noise,
    initialize_camulator,
    parse_datetime_from_config,
    run_climate_integration,
)

# Wind post-processing
from .WindPP import (
    WindArtifactFilterConfig,
    apply_wind_artifact_filter_to_tensor,
    load_wind_filter_config,
    post_process_wind_artifacts,
    wind_filter,
)

# Version info
__version__ = "0.2.0"
__author__ = "CREDIT Team"

__all__ = [
    # Core classes
    "StateManager",
    "CAMulatorStepper",
    "WindArtifactFilterConfig",
    # Initialization and setup
    "initialize_camulator",
    "load_metadata",
    "load_wind_filter_config",
    # Climate integration
    "run_climate_integration",
    "add_init_noise",
    "parse_datetime_from_config",
    # Output handling
    "split_and_reshape",
    "make_xarray",
    "save_netcdf_increment",
    # Post-processing
    "post_process",
    "rescale_file",
    "extract_time_single",
    "add_hours_noleap",
    # Wind processing
    "wind_filter",
    "post_process_wind_artifacts",
    "apply_wind_artifact_filter_to_tensor",
    # Submodules
    "Post_Process_Parallel",
    "COUPLING_EXAMPLE",
]
