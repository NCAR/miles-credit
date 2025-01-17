#!/bin/bash
# PBS -N WX_XXXXX_Clim_Run
# PBS -A NAML0001 
# PBS -l walltime=12:00:00
# PBS -o train_clim_XXXXX.out
# PBS -e train_clim_XXXXX.out
# PBS -q casper
# PBS -l select=1:ncpus=16:ngpus=1:mem=128GB
# PBS -m a
# PBS -M wchapman@ucar.edu

module purge
module load cuda/11.8 cudnn/8.7.0.84-11.8 conda/latest
conda activate credit-dk-casper

torchrun /glade/work/wchapman/miles_branchs/CESM_physics_multigpu/applications/rollout_to_np_climate.py -c /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_XXXXX/model_multi_WxFormer.yml
