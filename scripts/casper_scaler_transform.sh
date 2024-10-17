#!/bin/bash -l
#PBS -N scaler_t
#PBS -l select=4:ncpus=36:mpiprocs=36:ngpus=0:mem=200GB
#PBS -l walltime=06:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod
module load conda
conda activate hcredit
cd ..
mpirun python -u -m mpi4py applications/scaler.py \
  -c config/crossformer.yml \
  -t 1h \
  -o /glade/derecho/scratch/dgagne/credit_scalers/ \
  -d /glade/derecho/scratch/dgagne/era5_quantile/ \
  -s /glade/derecho/scratch/dgagne/credit_scalers/era5_quantile_scalers_2024-03-30_00:28.parquet \
  -r
