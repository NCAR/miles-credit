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
# pip install -e .


#-----------------------------------------------------------
# set up an initial conda environment at ${CREDIT_ENV_PATH}
# containing Derecho-specific torch & MPI bits.
# (install torchmetrics at this point too, installing it later
# via pip risks an undesirable torch update.)
ml conda

topdir=$(git rev-parse --show-toplevel)
CREDIT_ENV_NAME=${CREDIT_ENV_NAME:-"credit-derecho"}
yml=$(mktemp --tmpdir=${topdir} credit-derecho-tmp-XXXXXXXXXX.yml)
echo ${yml}
#yml=derecho.yml

echo "Creating conda env \"${CREDIT_ENV_NAME}\""

cat <<EOF > ${yml}
name: credit
channels:
  - file:///glade/work/benkirk/consulting/conda-recipes/output
  - conda-forge/label/mpi-external
  - conda-forge
dependencies:
  - python=3.11
  - conda-tree
  - mpi4py =*=derecho*
  - pip
  - pytorch ==2.8.0=derecho*2000
  - torchvision =*=derecho*
  - torchmetrics
  - pip:
    - pipdeptree
    - -e .
EOF

# create the environment
export CONDA_VERBOSITY=1
export TIMEFORMAT=$'--> Real time: %3R seconds'
time conda env create \
     --name ${CREDIT_ENV_NAME} \
     --file ${yml} \
    || { cat ${yml}; echo "ERROR Creating Conda env!"; exit 1; }

rm -f ${yml}

#-----------------------------------------------------------
# activate & fix the environment
conda activate ${CREDIT_ENV_NAME}

echo "NCCLs - before cleanup:"
find ${CONDA_PREFIX} -name "libnccl.*"

# remove PIP NCCL, if any.
#  (echo-opt -> xgboost -> nvidia-nccl-cu12 -> problem.)
pip uninstall -y $(pip list | grep nvidia-nccl | awk '{print $1}') || true

conda-tree deptree --small
pipdeptree --depth 3

echo "NCCLs - after cleanup:"
find ${CONDA_PREFIX} -name "libnccl.*"

python -c "import credit"

echo
echo "\"${CREDIT_ENV_NAME}\" conda environment for Derecho successfully installed into CONDA_PREFIX"
echo "use \"conda activate ${CREDIT_ENV_NAME}\" to activate"
