#!/bin/bash -l
#PBS -N nwp_solar
#PBS -l select=4:ncpus=128:mpiprocs=128:ngpus=0:mem=200GB
#PBS -l walltime=06:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
module load conda craype/2.7.23 cray-mpich/8.1.27
conda activate hcredit
cd ..
mpiexec -n 512 -ppn 128 python -u applications/calc_global_solar.py -o /glade/derecho/scratch/dgagne/credit_scalers/