#!/bin/bash
#PBS -N Run_CREDIT_Climate
#PBS -A XXXXXXX
#PBS -l walltime=12:00:00
#PBS -o RUN_Climate.out
#PBS -e RUN_Climate.out
#PBS -q casper
#PBS -l select=1:ncpus=32:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M email@ucar.edu

module load conda
conda activate /glade/work/wchapman/conda-envs/credit-casper-modern

#an example climate rollout: 

CONFIG=../config/camulator_veros.yml
SCRIPT=./Quick_Climate_Year_Slow_parrallel_opt.py
BASE_ARGS="--config $CONFIG \
  --input_shape 1 136 1 192 288 \
  --forcing_shape 1 6 1 192 288 \
  --output_shape 1 145 1 192 288 \
  --device cuda"

#Loop over checkpoint numbers
for i in {88..88}; do
  #zero-pad to 5 digits: 00171, 00172, …
  ckpt=$(printf "%05d" $i)
  python $SCRIPT $BASE_ARGS --model_name checkpoint.pt${ckpt}.pt --save_append run_${ckpt}
done