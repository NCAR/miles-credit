#!/bin/bash -l
#PBS -N credit_solar
#PBS -l select=4:ncpus=128:mpiprocs=128:ngpus=0
#PBS -l walltime=01:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
module load conda cray-mpich mkl
conda activate credit-derecho
cd ..
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_OFI_NIC_POLICY="NUMA"
mpiexec -n 512 -ppn 128 python -u applications/calc_global_solar.py \
  -s "2026-01-01" \
  -e "2026-12-31 18:00" \
  -i /glade/derecho/scratch/kjmayer/CREDIT_runs/S2Socnlndatm_MLdata/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc  \
  -t 6h \
  -g PHIS \
  -u 10Min \
  -o /glade/derecho/scratch/dgagne/cesmulator_solar_6h_1deg/

