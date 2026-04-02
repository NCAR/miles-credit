#!/bin/bash -l
#PBS -N wxformer_v2_025deg
#PBS -l select=4:ncpus=32:ngpus=4:mem=480GB
#PBS -l walltime=12:00:00
#PBS -A NAML0001
#PBS -q main
#PBS -j oe
#PBS -k eod

# WXFormer v2 @ 0.25deg, 6-hr timestep — production training on Derecho
# 4 nodes × 4 H100 = 16 GPUs, FSDP2
# Usage: qsub scripts/derecho_wxformer_v2_025deg.sh

REPO=/glade/work/schreck/repos/miles-credit-main
CONFIG=${REPO}/config/wxformer_v2_025deg_6hr.yml
NGPUS=4
NNODES=4

echo "Repo   : ${REPO}"
echo "Config : ${CONFIG}"
echo "Nodes  : ${NNODES} × ${NGPUS} GPUs"

module --force purge
module load ncarenv/24.12 gcc/14.2.0 cuda/12.4 nccl openmpi

conda activate /glade/work/schreck/conda-envs/credit-main-casper

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

# Derecho InfiniBand / libfabric settings
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export FI_CXI_ATS=0

# Multi-node torchrun via MPI
MASTER_ADDR=$(hostname)
MASTER_PORT=29500

torchrun \
    --nnodes=${NNODES} \
    --nproc-per-node=${NGPUS} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    ${REPO}/applications/train_v2.py \
    -c ${CONFIG}
