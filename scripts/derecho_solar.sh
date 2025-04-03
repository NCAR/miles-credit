#!/bin/bash -l
#PBS -N credit_solar
#PBS -l select=4:ncpus=128:mpiprocs=128:ngpus=0
#PBS -l walltime=01:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
#PBS -J 2025-2030
module load conda craype/2.7.23 cray-mpich/8.1.27
conda activate hcredit
cd ..
mpiexec -n 512 -ppn 128 python -u applications/calc_global_solar.py \
  -s "${PBS_ARRAY_INDEX}-01-01" \
  -e "${PBS_ARRAY_INDEX}-12-31 18:00" \
  -i /glade/u/home/wchapman/MLWPS/DataLoader/LSM_static_variables_ERA5_zhght.nc  \
  -t 6h \
  -u 10Min \
  -o /glade/derecho/scratch/dgagne/credit_solar_nc_6h_0.25deg/

#  -o /glade/derecho/scratch/dgagne/credit_solar_6h_1deg/
