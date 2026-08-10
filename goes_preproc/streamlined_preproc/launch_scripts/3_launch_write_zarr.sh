#!/bin/bash -l
#PBS -N write_zarr_store_parallel
#PBS -l select=1:ncpus=32:mem=64GB
#PBS -l walltime=01:30:00
#PBS -A NMMM0015
#PBS -q casper@casper-pbs
#PBS -j oe
#PBS -k eod

source ~/.bashrc
conda activate ~/credit
mkdir ~/tmp_papermill -p

rm -f /glade/derecho/scratch/dkimpara/goes-cloud-dataset/goes_10km_2025.zarr

papermill /glade/u/home/dkimpara/goes-cloud/streamlined_preproc/3-1_g19_make_zarr_store.ipynb.ipynb \
    ~/tmp_papermill/out.ipynb \
    --log-output \
    -k credit
    
papermill /glade/u/home/dkimpara/goes-cloud/streamlined_preproc/3-2_g19_write_zarr_store_parallel.ipynb  \
    ~/tmp_papermill/out.ipynb \
    --log-output \
    -k credit

rm ~/tmp_papermill/out.ipynb