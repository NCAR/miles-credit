#!/bin/bash -l
#PBS -N scaler_t
#PBS -l select=16:ncpus=128:mpiprocs=128:ngpus=0
#PBS -l walltime=06:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
module load conda 
conda activate hcredit
cd ..
mpiexec -n 2048 python -u -m mpi4py applications/scaler.py \
  -c config/wxformer_1h.yml \
  -t 1h \
  -o /glade/derecho/scratch/dgagne/credit_scalers/ \
  -d /glade/derecho/scratch/dgagne/era5_standard/ \
  -s /glade/derecho/scratch/dgagne/credit_scalers/era5_standard_scalers_2024-07-27_10:30.parquet \
  -r
