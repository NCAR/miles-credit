#!/usr/bin/env bash

qsub -I -q main -A NAML0001 -l walltime=12:00:00 -l select=1:ncpus=32:ngpus=1:mem=128GB
module purge
module load ncarenv/23.09 gcc/12.2.0 ncarcompilers cray-mpich/8.1.27 cuda/12.2.1 cudnn/8.8.1.3-12 conda/latest
conda activate credit-derecho
