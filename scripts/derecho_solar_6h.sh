#!/bin/bash -l
#PBS -N credit_solar_6h
#PBS -l select=4:ncpus=128:mpiprocs=128:ngpus=0
#PBS -l walltime=01:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
#PBS -J 2026-2027
module load conda mkl
conda activate credit-derecho
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_OFI_NIC_POLICY="NUMA"
cd ..
mpiexec -n 512 -ppn 128 python -u applications/calc_global_solar.py \
  -s "${PBS_ARRAY_INDEX}-01-01" \
  -e "${PBS_ARRAY_INDEX}-12-31 18:00" \
  -i /glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416.nc  \
  -t 6h \
  -u 10Min \
  -o /glade/derecho/scratch/dgagne/credit_solar_nc_6h_0.25deg_20251216/

#  -o /glade/derecho/scratch/dgagne/credit_solar_6h_1deg/
