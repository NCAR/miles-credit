#!/bin/bash -l
#PBS -N edge_credit
#PBS -l select=1:ncpus=128:ngpus=0:mem=200GB
#PBS -l walltime=01:00:00
#PBS -l job_priority=regular
# Pass your project code at submit time: qsub -A ${PBS_ACCOUNT:?set PBS_ACCOUNT} <this script>
#PBS -q main
#PBS -j oe
#PBS -k eod
module load conda
conda activate credit
cd ..
python -u applications/graph_edges.py -c /glade/u/home/wchapman/MLWPS/DataLoader/static_variables_ERA5_zhght.nc \
       -p 128 \
       -d 125 \
       -o  /glade/derecho/scratch/dgagne/credit_scalers/
