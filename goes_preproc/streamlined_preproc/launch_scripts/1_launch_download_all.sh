#!/bin/bash -l
#PBS -N preproc_goes_cloud
#PBS -l select=1:ncpus=62:mem=732GB
#PBS -l walltime=24:00:00
#PBS -A NMMM0015
#PBS -q casper@casper-pbs
#PBS -j oe
#PBS -k eod
#PBS -J 2018-2024

source ~/.bashrc
conda activate xesmf

python /glade/u/home/dkimpara/goes-cloud/streamlined_preproc/1_goes_io_parallel_combined_channels.py -y ${PBS_ARRAY_INDEX}