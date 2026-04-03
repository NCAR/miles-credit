#!/bin/bash
#PBS -N camulator_server
#PBS -A NAML0001
#PBS -q casper
#PBS -l job_priority=premium
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=32:mpiprocs=1:ompthreads=1:mem=100gb:ngpus=1:gpu_type=a100_80gb
#PBS -j oe

set -euo pipefail

echo "Running on host: $(hostname)"

module purge
module load conda
conda activate credit-coupling

# Sanity checks
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true
which python
python --version

python camulator_server.py \
  --config ./camulator_config.yml \
  --model_name checkpoint.pt00091.pt \
  --rundir /glade/derecho/scratch/wchapman/g.e21.CAMULATOR_GIAF_v09/run/ \
  --save_atm_nc camulator_out \
  --daily_mean