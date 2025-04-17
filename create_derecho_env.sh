#!/bin/bash
ml conda
ENV_NAME="credit-derecho"
CURR_DIR=`pwd`
WHEEL_DIR="/glade/work/dgagne/credit-pytorch-envs/derecho-pytorch-mpi/wheels"
echo $CURR_DIR
conda create -n $ENV_NAME python=3.11
conda init
conda activate $ENV_NAME
cd /glade/work/dgagne/credit-pytorch-envs/derecho-pytorch-mpi
./embed_nccl_vars_conda.sh
cd $CURR_DIR
pip install ${WHEEL_DIR}/torch-2.5.1+derecho.gcc.12.4.0.cray.mpich.8.1.29-cp311-cp311-linux_x86_64.whl
pip install ${WHEEL_DIR}/torchvision-0.20.1+derecho.gcc.12.4.0-cp311-cp311-linux_x86_64.whl
pip install -e .
