# CAMulator Quick Start Guide

This directory contains the scripts to run **CAMulator** - the CREDIT AI model for long rollout simulations based on CAM6 Emulation

## Overview

CAMulator integrates machine learning predictions with physical conservation laws (mass, water, energy) to perform long-term climate simulations. It's designed to run multi-year modern climate rollouts with CESM-like data.

## Prerequisites

### 1. Environment Setup

Make sure you have the CREDIT environment activated. 

```bash
# On NCAR HPCs (Derecho/Casper)
conda activate /glade/work/<username>/conda-envs/credit-derecho

#OR

conda activate /glade/work/<username>/conda-envs/credit-casper 


# Or your local environment
conda activate credit
```

### 2. Required Files

You'll need:
- **Configuration file**: `camulator_config.yml` (provided in this directory)
- **Training data**: For creating initial conditions (paths specified in config)
- **Static/forcing data**: Physics constants and boundary conditions
- **Model checkpoint**: A trained CAMulator model (e.g., `checkpoint.pt00091.pt`)
- - if you'd like to copy a pre-trained simulation you can find it here: `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/CAMulator_models/checkpoint.pt00091.pt`
- - you will then copy that model into your `sav_loc` in your  `camulator_config.yml` 

## Configuration

Before running, adjust these critical fields in `camulator_config.yml`.
<br>
For the Standard CAMulator, they are set to the correct default values, if you train your own model, you must ensure they match your simulation. 


### Key Paths to Update

```yaml
# Model output location
save_loc: '/glade/derecho/scratch/<username>/CREDIT_runs/your_experiment_name/'

# Training data (for initial conditions and normalization)
# this data will effect the simulation, but anything saved in the campaign storage will remain static to this file and you can use, 
# the defaults are currently set well. Try to use the files that the model was trained with (defaults for CAMulator)
data:
  save_loc: '/path/to/your/training/data/????_zmdata.zarr'
  save_loc_surface: '/path/to/your/surface/data/????_zmdata.zarr'
  save_loc_dynamic_forcing: '/path/to/your/forcing/data/????_zmdata.zarr'
  save_loc_diagnostic: '/path/to/your/diagnostic/data/????_zmdata.zarr'

  # Normalization files (mean/std from training)
  mean_path: '/path/to/mean_file.nc'
  std_path: '/path/to/std_file.nc'

  # Static variables (topography, land mask, etc.)
  save_loc_static: '/path/to/static_file.nc'
```

### Critical: The `predict` Section

When running CAMulator inference, you'll primarily adjust the `predict` section. Here's a comprehensive guide to all fields:

#### **Fields You MUST Update** (User-Specific Paths in /scratch/)

```yaml
predict:
  # ============================================================================
  # OUTPUT LOCATION - UPDATE THIS!
  # ============================================================================
  save_forecast: '/glade/derecho/scratch/<username>/CREDIT/climate_output/'
  # Where NetCDF outputs are saved: save_forecast/YYYY-MM-DD_HHZ/lead_time_XXX.nc
  # MUST be in /scratch/ (purged after 90 days - move to /campaign/ when done!)

  # ============================================================================
  # INITIAL CONDITIONS - UPDATE THIS!
  # ============================================================================
  init_cond_fast_climate: '/glade/derecho/scratch/<username>/CREDIT_runs/your_experiment/init_times/init_condition_tensor_1983-01-01T00Z.pth'

  # a series of initialization times are stored here /glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/ 
  # Path to initial condition created in Step 1
  # File name MUST match start_datetime below
  # Created in save_loc/init_times/ from Make_Climate_Initial_Conditions.py

  # ============================================================================
  # SIMULATION TIME PERIOD - ADJUST AS NEEDED
  # ============================================================================
  forecasts:
    type: "custom"           # Keep as "custom"
    start_year: 1983         # Year to start simulation
    start_month: 1           # Month (1-12)
    start_day: 1             # Day (1-31)
    start_hours: [0]         # Hour: [0]=00Z, [12]=12Z, [0,12]=both
    duration: 1              # Number of consecutive days to initialize (1=single run)
  start_datetime: '1983-01-01 00:00:00'  # Must exactly match forecasts above!
```

#### **Fields You Can Usually Keep** (Static Paths in /campaign/)

```yaml
predict:
  # ============================================================================
  # GPU/PARALLELIZATION
  # ============================================================================
  mode: none               # 'none'=single GPU, 'ddp'=multi-GPU, 'fsdp'=huge models
  batch_size: 1            # Memory per forward pass (keep at 1 for climate runs)

  # ============================================================================
  # VARIABLE SELECTION - Control output size
  # ============================================================================
  save_vars: ['PRECT','TS','CLDHGH','CLDLOW','CLDMED','TAUX','TAUY','U10','QFLX',
              'FSNS','FLNS','FSNT','FLNT','SHFLX','LHFLX','TREFHT','PS','Qtot','U','V','T']
  # Specific variables to save - reduces disk space
  # Use [] to save ALL variables (warning: large files!)

  # ============================================================================
  # FORCING DATA - STATIC (only change for custom scenarios)
  # ============================================================================
  #the forcing file carries SST, CO2 and SOLIN data to force the model.
  #structure custom forcing files after this one. 
  forcing_file: '/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate_branch_1980_2014.nc'

  #additional forcing file for the future runs /glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING//glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate_branch_2000_2053.nc

  # Time-evolving boundary conditions: SST, sea ice, solar, CO2
  # STATIC in /campaign/ - only update if:
  #   - Running dates outside 1980-2014 range
  #   - Using custom climate scenarios (e.g., future SSTs)
  #   - Testing sensitivity to forcing

  # ============================================================================
  # METADATA & VERIFICATION - STATIC
  # ============================================================================
  metadata: '/glade/work/schreck/repos/credit/miles-credit/metadata/era5.yaml'
  # Variable definitions (units, long names) for NetCDF output
  # STATIC - don't change unless custom variables

  climatology: '/glade/campaign/cisl/aiml/ksha/CREDIT_arXiv/VERIF/verif_6h/ERA5_clim/ERA5_clim_1990_2019_6h_interp.nc'
  # Reference climatology for computing anomaly ACC
  # STATIC - only needed for rollout_metrics.py
  # Optional: remove if not computing metrics

  # ============================================================================
  # PROCESSING OPTIONS
  # ============================================================================
  climate_rescale_output: True      # Rescale outputs to physical units (keep True)
  use_laplace_filter: False         # Spatial smoothing (may reduce noise but affects skill)

  # ============================================================================
  # FAST CLIMATE MODE - For quick testing
  # ============================================================================
  timesteps_fast_climate: 200       # Number of timesteps to run the mode !!!!! 
  # Reference seasonal mean for fast mode verification
```

#### **Quick Setup Checklist**

Before running, verify:

- [x] **save_forecast**: Set to YOUR /scratch/ directory
- [x] **init_cond_fast_climate**: Points to YOUR init file from Step 1
- [x] **start_year/month/day**: Simulation start date
- [x] **start_datetime**: Matches forecasts section exactly (format: 'YYYY-MM-DD HH:MM:SS')
- [x] **days**: Length of simulation (365=1yr, 1825=5yr, 3650=10yr)
- [x] **forcing_file**: Dates covered (default: 1980-2014)
- [x] **save_vars**: Select variables or [] for all

#### **Common Scenarios**

**Example 1: Single 1-year run starting Jan 1, 1983**
```yaml
predict:
  save_forecast: '/glade/derecho/scratch/username/test_climate/'
  init_cond_fast_climate: '/glade/derecho/scratch/username/CREDIT_runs/wxformer/init_times/init_condition_tensor_1983-01-01T00Z.pth'
  forecasts:
    start_year: 1983
    start_month: 1
    start_day: 1
    start_hours: [0]
  start_datetime: '1983-01-01 00:00:00'
  timesteps_fast_climate: 1460 #6-hour timesteps 
```

**Example 2: 10-year simulation starting July 1, 2000**
```yaml
predict:
  forecasts:
    start_year: 2000
    start_month: 7
    start_day: 1
  start_datetime: '2000-07-01 00:00:00'
  init_cond_fast_climate: '/.../init_condition_tensor_2000-07-01T00Z.pth'  # Must match!
  timesteps_fast_climate: 14600 #6-hour timesteps 
```

**Example 4: Save only surface variables (reduce disk space)**
```yaml
predict:
  save_vars: ['PRECT', 'TREFHT', 'TS', 'PS']  # Precip, temp, pressure only
  # Saves ~80% less disk space than default
```

#### **Understanding Duration vs. Days**

- **days**: How long EACH simulation runs (forecast length)
  - `timesteps_fast_climate: 1460` = each run is 1 years long, 6-hour timestep (365 * 4 steps per day)


- **start_hours**: Time(s) of day for each initialization
  - `[0]` = 00Z only → 1 run per day
  - `[12]` = 12Z only → 1 run per day


### Model Configuration

Verify these match your trained model:

Currently the model is set for standard CAMulator, but if you train your own model it will have to match the model you are poiting at. 

```yaml
model:
  type: "camulator"
  image_height: 192        # Latitude grid points
  image_width: 288         # Longitude grid points
  levels: 32               # Vertical levels
  channels: 4              # Upper-air variables [U, V, T, Q]
  surface_channels: 2      # Surface variables [PS, TREFHT]
  input_only_channels: 6   # Static + forcing inputs
  output_only_channels: 15 # Diagnostic outputs
```

## Step-by-Step Workflow

### Step 1: Create Initial Conditions
A few initialization times are stored here /glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/ which are free to use.

To generate custom the initial state tensor for your climate simulation you must first adjust the forecasts start date in the predict section of the yaml file: 

```yaml
predict:
  forecasts:
    start_year: 1983
    start_month: 1
    start_day: 1
    start_hours: [0]  # Both 00Z and 12Z
  start_datetime: '1983-01-01 00:00:00'
```

then: 

```bash
cd /path/to/credit/climate

# Single GPU (using default checkpoint.pt from your save_loc folder)
python Make_Climate_Initial_Conditions.py -c ./camulator_config.yml

# Single GPU with specific model checkpoint, this is a name model in your save loc for example checkpoint.pt00091.pt
python Make_Climate_Initial_Conditions.py -c ./camulator_config.yml --model_name checkpoint.pt00091.pt

# Multi-GPU (recommended for large models)
torchrun --nproc_per_node=4 Make_Climate_Initial_Conditions.py -c ./camulator_config.yml --model_name checkpoint.pt00091.pt
```

**Parameters:**
- `-c`: Path to configuration file (required)
- `--model_name`: Optional specific checkpoint file (e.g., `checkpoint.pt00091.pt`). If not specified, defaults to `checkpoint.pt`

This creates initial condition files in `save_loc/init_times/` like:
- `init_camulator_condition_tensor_1983-01-01T00Z.pth`

### Step 2: Run Climate Simulation

Run the climate rollout using `Quick_Climate.py`:

```bash
python Quick_Climate.py \
  --config ./camulator_config.yml \
  --model_name checkpoint.pt00091.pt \
  --save_append test_run_001
```

**Parameter Explanation:**
- `--config`: Path to your YAML configuration file
- `--model_name`: Name of the model checkpoint file (must be in `save_loc`). **Important: Use the same checkpoint for both Step 1 and Step 2**
- `--save_append`: Suffix for output directory (helps organize multiple runs)

**Note:** Input/output shapes are now automatically derived from the config file - no need to specify them manually!

**Output:**
- NetCDF files saved to `save_forecast/save_append/YYYY-MM-DD_HHZ/`
- One file per forecast timestep (6-hourly by default)

### Step 3: Post-Process Output

Aggregate 6-hourly data into daily means:

```bash
python Post_Process.py \
  ./camulator_config.yml \
  1D \
  --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY \
  --reset_times False \
  --dask_do False \
  --name_string UVTQtotPSTREFHTTAUXTAUY \
  --rescale_it False \
  --save_append test_run_001
```

**Parameters:**
- First argument: Config file
- Second argument: `1D` (daily), `1M` (monthly), `1Y` (yearly)
- `--variables`: List of variables to process
- `--reset_times`: Whether to reset time coordinates
- `--dask_do`: Use Dask for parallel processing (may OOM on large datasets)
- `--name_string`: Identifier for output file
- `--rescale_it`: Whether to rescale outputs
- `--save_append`: Must match the simulation run

**Output:**
- Processed files in `save_forecast/processed/1D/`

## Complete Workflow Script with PBS

For convenience, use the provided `RunQuickClimate.sh` script that runs both simulation and post-processing.

### Understanding the PBS Script

The `RunQuickClimate.sh` is a PBS job script for NCAR HPCs. Here's what each section does:

#### **PBS Directives** (Lines 1-12)

```bash
#!/bin/bash
#PBS -N Run_Climate_CAMulator      # Job name (shows in qstat)
#PBS -A NAML0001                   # Project code (UPDATE THIS!)
#PBS -l walltime=12:00:00          # Max run time (hours:minutes:seconds)
#PBS -o RUN_Climate.out            # Output file
#PBS -e RUN_Climate.out            # Error file (same as output)
#PBS -q casper                     # Queue: 'casper' or 'main' (Derecho)
#PBS -l select=1:ncpus=32:ngpus=1:mem=250GB  # Resources: 1 node, 32 CPUs, 1 GPU, 250GB RAM
#PBS -l gpu_type=a100              # GPU type: 'a100' (Casper) or 'h100' (Derecho)
#PBS -m a                          # Email when job aborts
#PBS -M your.email@ucar.edu        # Your email (UPDATE THIS!)
```

**Key PBS Options to Adjust:**
- **`-A`**: Your project code (e.g., `NAML0001`, `NCAR0001`)
- **`-q`**: Queue selection
  - `casper`: Analysis cluster with A100 GPUs (good for CAMulator)
  - `main`: Derecho main queue with H100 GPUs (for large runs)
- **`-l walltime`**: Adjust based on simulation length
  - 1-year run: ~4-8 hours
  - 10-year run: ~24-48 hours
- **`-l mem`**: Memory allocation
  - CAMulator typically needs 250GB
  - Increase if you get OOM errors
- **`-M`**: Your email for job notifications

#### **Environment Setup** (Lines 14-15)

```bash
module load conda
conda activate /glade/work/<username>/conda-envs/credit-casper  # UPDATE PATH!
```

Update the conda environment path to YOUR environment.

#### **Script Configuration** (Lines 17-24)

Edit these variables in `RunQuickClimate.sh`:

```bash
CONFIG=./camulator_config.yml
SCRIPT=./Quick_Climate.py
FOLD_OUT=test_run_001  # Unique identifier for this run (UPDATE THIS!)

# Run simulation
python $SCRIPT \
  --config $CONFIG \
  --model_name checkpoint.pt00091.pt \
  --device cuda \
  --save_append $FOLD_OUT
```

**Variables to update:**
- **`FOLD_OUT`**: Unique name for output organization (e.g., `1yr_1983`, `decadal_2000`)
- **`--model_name`**: Your model checkpoint filename
- **`--config`**: Path to config (usually `./camulator_config.yml`)

#### **Post-Processing** (Line 29-30)

```bash
# Post-process to daily means
python ./Post_Process.py $CONFIG 1D \
  --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY \
  --reset_times False \
  --dask_do False \
  --name_string UVTQtotPS_daily \
  --rescale_it False \
  --save_append $FOLD_OUT
```

Adjust `--variables` to only include what you need (saves time and disk space).

### Submitting the Job

#### **Step 1: Edit the script**

```bash
cd /glade/work/<username>/credit/climate

# Edit PBS directives and paths
vi RunQuickClimate.sh
# or
nano RunQuickClimate.sh
```

Make sure to update:
- [x] `-A` project code
- [x] `-M` email address
- [x] `conda activate` path
- [x] `FOLD_OUT` experiment name
- [x] `--model_name` checkpoint

#### **Step 2: Make executable and submit**

```bash
# Make executable (only needed once)
chmod +x RunQuickClimate.sh

# Submit to PBS queue
qsub RunQuickClimate.sh
```

#### **Step 3: Monitor the job**

```bash
# Check job status
qstat -u $USER

# View output in real-time
tail -f RUN_Climate.out

# Check if job is running on which node
qstat -f <job_id>

# View completed job details
qhist -u $USER
```

### Running Interactively (Without PBS)

For testing or short runs:

```bash
cd /glade/work/<username>/credit/climate

# Activate environment
conda activate /glade/work/<username>/conda-envs/credit-casper

# Run directly
bash RunQuickClimate.sh
```

**Note:** Interactive runs are best for:
- Testing configuration changes
- Short simulations (<1 year)
- Debugging issues
- Running on login nodes is NOT recommended (use `qsub -I` for interactive sessions)

### PBS Job Management

```bash
# Submit job
qsub RunQuickClimate.sh

# Check status
qstat -u $USER

# Delete/cancel job
qdel <job_id>

# Interactive session (for debugging)
qsub -I -A NAML0001 -l select=1:ncpus=8:ngpus=1:mem=64GB -l walltime=2:00:00 -q casper

# Check remaining project hours
gladequota
```

### Choosing Between Casper and Derecho

**Use Casper when:**
- Running CAMulator inference (lighter workload)
- Need A100 GPUs (perfect for CAMulator)
- Want faster queue times
- Running multi-year climate simulations

**Use Derecho when:**
- Training models (heavy computation)
- Need H100 GPUs (more powerful)
- Running many concurrent jobs
- Need more than 1 node

**For CAMulator specifically:** Casper is recommended - it has lower wait times and A100 GPUs are sufficient.

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

- Reduce `predict.batch_size` in config (default: 1)
- Use CPU offloading: set `cpu_offload: True` in config
- Enable activation checkpointing: `activation_checkpoint: True`
- Request more memory in PBS: increase `-l mem=250GB` to `-l mem=480GB`

**2. Missing Initial Conditions**

Make sure you ran `Make_Climate_Initial_Conditions.py` first and check that:
- Initial condition files exist in `save_loc/init_times/`
- The start date in `predict` section matches your IC files
- File naming: should be `init_condition_tensor_YYYY-MM-DDT00Z.pth`

**3. Normalization Issues**

Verify that:
- All variables in your config exist in `mean_path` and `std_path`
- Mean/std files match the training data exactly
- Paths in /campaign/ are accessible (check with `ls`)

**4. PBS Job Fails Immediately**

Check:
- Project code (`-A`) is correct and has remaining hours (`gladequota`)
- Conda environment path is correct
- All paths in config file are absolute (not relative)
- Output directory (`save_forecast`) exists or parent directory is writable

**5. Forcing File Date Range**

If you get errors about missing forcing data:
- Default forcing file covers 1980-2014
- Check `forcing_file` in config matches your simulation dates
- For dates outside this range, you'll need different forcing data

### Checking Progress

Monitor the climate simulation:

```bash
# Watch output file
tail -f RUN_Climate_RMSE.out

# Check saved NetCDF files
ls -lh /path/to/save_forecast/1983-01-01_00Z/

# Quickly inspect with ncdump
ncdump -h /path/to/save_forecast/1983-01-01_00Z/lead_time_006.nc
```

## Additional Resources

- **Full Documentation**: https://miles-credit.readthedocs.io/
- **Main README**: `../README.md`
- **Config Documentation**: `../config/README.md`
- **Example Configs**: `camulator_config.yml` (this directory)

## Questions?

If you encounter issues:
1. Check the output logs in `RUN_Climate_RMSE.out`
2. Verify all paths in `camulator_config.yml` are correct
3. Ensure your model checkpoint matches the architecture in config
4. Review the CREDIT documentation for detailed parameter descriptions
