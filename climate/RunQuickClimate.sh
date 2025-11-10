#!/bin/bash
#-*- coding: utf-8 -*-
#PBS -N Run_Climate_CAMulator
#PBS -A NAML0001
#PBS -l walltime=12:00:00
#PBS -o RUN_Climate.out
#PBS -e RUN_Climate.out
#PBS -q casper
#PBS -l select=1:ncpus=32:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

# ============================================================================
# CAMulator Climate Simulation PBS Script
# ============================================================================
# This script runs a complete CAMulator workflow:
#   1. Climate simulation (Quick_Climate.py)
#   2. Post-processing to daily means (Post_Process.py)
#
# BEFORE RUNNING:
#   - Update -A (project code) above
#   - Update -M (email) above
#   - Update conda environment path below
#   - Update FOLD_OUT (experiment name) below
#   - Update MODEL_NAME (checkpoint file) below
#   - Verify CONFIG paths are correct
# ============================================================================

module load conda
conda activate /glade/work/wchapman/conda-envs/credit-casper-modern

# ============================================================================
# Configuration
# ============================================================================
CONFIG=./camulator_config.yml
SCRIPT=./Quick_Climate.py
FOLD_OUT=test_00091                    # UPDATE: Unique experiment identifier
MODEL_NAME=checkpoint.pt00091.pt       # UPDATE: Your model checkpoint name

# ============================================================================
# Run Climate Simulation
# ============================================================================
echo "Starting CAMulator simulation..."
echo "Config: $CONFIG"
echo "Model: $MODEL_NAME"
echo "Output: $FOLD_OUT"
echo "Time: $(date)"

python $SCRIPT \
  --config $CONFIG \
  --model_name $MODEL_NAME \
  --device cuda \
  --save_append $FOLD_OUT

if [ $? -eq 0 ]; then
    echo "Simulation completed successfully!"
else
    echo "ERROR: Simulation failed!"
    exit 1
fi

# ============================================================================
# Post-Process to Daily Means
# ============================================================================
echo ""
echo "Starting post-processing..."

# Variables to process - adjust as needed
VARIABLES="U V T Qtot PS PRECT TREFHT TS TAUX TAUY"

python ./Post_Process.py $CONFIG 1D \
  --variables $VARIABLES \
  --reset_times False \
  --dask_do False \
  --name_string daily_processed \
  --rescale_it False \
  --save_append $FOLD_OUT

if [ $? -eq 0 ]; then
    echo "Post-processing completed successfully!"
    echo "Output location: Check save_forecast in $CONFIG"
else
    echo "WARNING: Post-processing failed (but simulation succeeded)"
fi

echo ""
echo "Workflow finished at: $(date)"
