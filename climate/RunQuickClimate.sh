#!/bin/bash
#PBS -N Run_Noise_Script
#PBS -A NAML0001 
#PBS -l walltime=12:00:00
#PBS -o RUN_Climate_RMSE.out
#PBS -e RUN_Climate_RMSE.out
#PBS -q casper
#PBS -l select=1:ncpus=32:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate credit-casper-modern

torchrun /glade/work/wchapman/miles_branchs/credit_feb15_2024/applications/Quick_Climate.py --config /glade/work/wchapman/miles_branchs/credit_feb15_2024/config/climate_rollout.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00240.pt --init_noise 1 

