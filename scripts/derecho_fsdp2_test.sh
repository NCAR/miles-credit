#!/bin/bash
#PBS -A $PBS_ACCOUNT
#PBS -N fsdp2_parallel_test
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q develop
#PBS -j oe
#PBS -k eod
#PBS -r n

# CREDIT v2 FSDP2 / TP / Domain Parallel smoke test — Derecho development queue
# Single node (4 GPUs). For 2-node tests change select=2 and adjust parallelism config.
# Usage: qsub scripts/derecho_fsdp2_test.sh

REPO=/glade/work/schreck/repos/miles-credit-main
CONFIG=${REPO}/config/fsdp2_parallel_test.yml
NGPUS=4

module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \
            cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1

conda activate /glade/work/schreck/conda-envs/credit-main-casper

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Derecho NCCL/libfabric settings
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PBH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export NCCL_DEBUG=WARN

echo "Node   : $(hostname)"
echo "Config : ${CONFIG}"
echo "GPUs   : ${NGPUS}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=${NGPUS} \
    ${REPO}/applications/train_v2.py \
    -c ${CONFIG}
