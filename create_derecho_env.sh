#!/bin/bash

set -e

# ml conda
# ENV_NAME="credit-derecho"
# CURR_DIR=`pwd`
# WHEEL_DIR="/glade/work/dgagne/credit-pytorch-envs/derecho-pytorch-mpi/wheels"
# echo $CURR_DIR
# conda create -n $ENV_NAME python=3.11
# conda init
# conda activate $ENV_NAME
# cd /glade/work/dgagne/credit-pytorch-envs/derecho-pytorch-mpi
# ./embed_nccl_vars_conda.sh
# cd $CURR_DIR
# pip install ${WHEEL_DIR}/torch-2.5.1+derecho.gcc.12.4.0.cray.mpich.8.1.29-cp311-cp311-linux_x86_64.whl
# pip install ${WHEEL_DIR}/torchvision-0.20.1+derecho.gcc.12.4.0-cp311-cp311-linux_x86_64.whl
#pip install -e .

ml conda

topdir=$(git rev-parse --show-toplevel)
CREDIT_ENV_PATH=${CREDIT_ENV_PATH:-"${topdir}/derecho-env"}


# Notes on the choices below:
#  install xgboost along with pytorch via conda.  This will then use a consistent NCCL.
#  then when we install echo-opt via pip the xgboost dependency is already satisfied.
#  otherwise, echo-opt -> xgboost -> nvidia-nccl-cu12 -> problem.
export CONDA_OVERRIDE_CUDA="12.3"
cat <<EOF > derecho.yml
name: credit
channels:
  - file:///glade/work/benkirk/consulting/conda-recipes/output
  - conda-forge
dependencies:
  - python=3.11
  - cartopy
  - conda-tree
  - dask
  - dask-jobqueue
  - distributed
  - einops
  - fsspec
  - gcsfs
  - haversine
  - jupyter
  - keras
  - matplotlib
  - metpy
  - mpi4py =*=derecho*
  - netcdf4
  - numpy
  - optuna==3.6.0
  - pandas
  - pip
  - pre-commit
  - pvlib
  - pyarrow
  - pytest
  - pytorch ==2.5.1=derecho*
  - pyyaml
  - ruff
  - scikit-learn
  - segmentation-models-pytorch
  - torchmetrics
  - torchvision =*=derecho*
  - xarray
  - zarr
  - pip:
    - pipdeptree
    - .
EOF

# create the environment
export CONDA_VERBOSITY=1
export TIMEFORMAT=$'--> Real time: %3R seconds'
time conda env create --prefix ${CREDIT_ENV_PATH} --file derecho.yml


conda activate ${CREDIT_ENV_PATH}

# remove PIP NCCL,if any
pip uninstall -y $(pip list | grep nvidia-nccl | awk '{print $1}')

conda-tree deptree --small
pipdeptree --depth 3

find ${CONDA_PREFIX} -name "libnccl.*"

python -c "import credit"
