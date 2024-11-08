#!/bin/bash
#PBS -N Trainup
#PBS -A NAML0001 
#PBS -l walltime=12:00:00
#PBS -o train_train.out
#PBS -e train_train.out
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=1:mem=128GB
#PBS -m a
#PBS -M wchapman@ucar.edu

# qsub -I -q main -A NAML0001 -l walltime=12:00:00 -l select=1:ncpus=32:ngpus=1:mem=128GB

module purge
module load ncarenv/23.09 gcc/12.2.0 ncarcompilers cray-mpich/8.1.27 cuda/12.2.1 cudnn/8.8.1.3-12 conda/latest
conda activate credit-derecho

python applications/train.py -c config/cesm_1deg_physics.yml