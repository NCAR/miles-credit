#!/bin/bash -l
#PBS -N credit_gen2
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB:gpu_type=a100_80gb
#PBS -l walltime=04:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

# Usage: CONFIG=path/to/config.yml qsub scripts/casper_gen2.sh

REPO=/glade/work/schreck/repos/miles-credit-main
TORCHRUN=/glade/u/home/schreck/.conda/envs/credit-casper/bin/torchrun
NGPUS=${NGPUS:-1}
CONFIG=${CONFIG:-${REPO}/config/wxformer_1dg_6hr_gen2.yml}

echo "Config : ${CONFIG}"
echo "Node   : $(hostname)"
echo "GPUs   : ${NGPUS}"

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

${TORCHRUN} --standalone --nnodes=1 --nproc-per-node=${NGPUS} \
    ${REPO}/applications/train_gen2.py -c ${CONFIG}
