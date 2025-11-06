#!/bin/bash -l
#PBS -N gwm
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB:gpu_type=a100
#PBS -l walltime=12:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

source ~/.bashrc
#module unload cuda cudnn
#CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)"))
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
#export XLA_FLAGS="--xla_gpu_cuda_data_dir=/glade/work/schreck/conda-envs/aqml/"
#conda activate aqml

conda activate /glade/work/schreck/miniconda3/envs/evidential

pip install .
python applications/trainer_vit2d.py -c config/vit2d.yml
