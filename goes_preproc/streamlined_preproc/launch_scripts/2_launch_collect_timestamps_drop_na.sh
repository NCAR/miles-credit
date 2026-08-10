#!/bin/bash -l
#PBS -N goes_collect_timestamps_drop_na
#PBS -l select=1:ncpus=32:mem=256GB
#PBS -l walltime=01:00:00
#PBS -A NMMM0015
#PBS -q casper
#PBS -j oe
#PBS -k eod
source ~/.bashrc
conda activate ~/credit
mkdir ~/tmp_papermill -p

# probably need around 3 CPU-hours

papermill /glade/u/home/dkimpara/goes-cloud/collect_timestamps_drop_na.ipynb  \
    ~/tmp_papermill/out.ipynb \
    --log-output \
    -k credit

rm ~/tmp_papermill/out.ipynb