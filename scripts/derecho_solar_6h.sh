#!/bin/bash -l
#PBS -N credit_solar_6h
#PBS -l select=4:ncpus=128:mpiprocs=128:ngpus=0
#PBS -l walltime=01:00:00
#PBS -A NAML0001
#PBS -q main@desched1
#PBS -l job_priority=regular
#PBS -j oe
#PBS -k eod
#PBS -J 2023-2027
#PBS -M dgagne@ucar.edu
#PBS -m abe
module load conda mkl
conda activate credit-derecho
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_OFI_NIC_POLICY="NUMA"
cd ..
mpiexec -n 512 -ppn 128 python -u -m mpi4py applications/calc_global_solar.py \
  -s "${PBS_ARRAY_INDEX}-01-01" \
  -e "${PBS_ARRAY_INDEX}-12-31 18:00" \
  -i "/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_1deg/static/ERA5_mlevel_1deg_6h_conserve_static.zarr" \
  -t "6h" \
  -v "toa_incident_solar_radiation" \
  -u "10Min" \
  -o "/glade/derecho/scratch/dgagne/credit_solar_6h_1deg_era5_mlevel/" \ 
  -g "geopotential_at_surface"

