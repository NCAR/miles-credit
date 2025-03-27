#!/bin/bash
#PBS -N Run_Noise_Script
#PBS -A XXXXXXX
#PBS -l walltime=12:00:00
#PBS -o RUN_Climate_RMSE.out
#PBS -e RUN_Climate_RMSE.out
#PBS -q casper
#PBS -l select=1:ncpus=32:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M email@ucar.edu

module load conda
conda activate credit-derecho

#an example climate rollout: 

python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1deg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy_SpatPS/model_multi_example-v2025.2.0.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00289.pt
