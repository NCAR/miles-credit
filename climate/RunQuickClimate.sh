#!/bin/bash
#-*- coding: utf-8 -*-
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
conda activate /glade/work/wchapman/conda-envs/credit-casper-modern

CONFIG=./camulator_config.yml
SCRIPT=./Quick_Climate.py
FOLD_OUT=test_00091
BASE_ARGS="--config $CONFIG \
  --input_shape 1 136 1 192 288 \
  --forcing_shape 1 6 1 192 288 \
  --output_shape 1 145 1 192 288 \
  --device cuda"

#run the inference forward in time:
python $SCRIPT $BASE_ARGS --model_name checkpoint.pt00091.pt --save_append $FOLD_OUT

#set the variables you would like post-processed to daily fields: 
python ./Post_Process.py ./camulator_config.yml 1D --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY --reset_times False --dask_do False --name_string UVTQtotPSPRECTTREFHTTAUXTAUY --rescale_it False --save_append $FOLD_OUT
