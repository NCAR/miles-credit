#!/bin/bash -l
#PBS -N gwm_scaler
#PBS -l select=1:ncpus=19:ngpus=0:mem=200GB
#PBS -l walltime=12:00:00
# Pass your project code at submit time: qsub -A $PBS_ACCOUNT <this script>
#PBS -q casper
#PBS -j oe
#PBS -k eod
module load conda
conda activate credit
cd ..
python -u applications/scaler.py -c config/rvt.yml -p 19 -o /glade/derecho/scratch/dgagne/credit_scalers/
