#!/bin/bash -l
#PBS -N preproc_goes_cloud_2025
#PBS -l select=1:ncpus=62:mem=732GB
#PBS -l walltime=12:00:00
#PBS -A NMMM0015
#PBS -q casper@casper-pbs
#PBS -j oe
#PBS -k eod

source ~/.bashrc
conda activate credit

python /glade/u/home/dkimpara/goes-cloud/streamlined_preproc/1_goes_io_parallel_combined_channels.py -y 2025 -r 0 -s goes19