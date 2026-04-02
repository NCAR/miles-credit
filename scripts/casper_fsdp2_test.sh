#!/bin/bash -l
#PBS -N fsdp2_parallel_test
#PBS -l select=1:ncpus=8:ngpus=4:mem=128GB:gpu_type=a100_80gb
#PBS -l walltime=01:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

# CREDIT v2 FSDP2 / TP / Domain Parallel smoke test — Casper
# Usage: qsub scripts/casper_fsdp2_test.sh
#
# To test different parallelism modes, edit config/fsdp2_parallel_test.yml:
#   FSDP2 only     : parallelism: {data: fsdp2, tensor: 1, domain: 1}  # 4 GPUs
#   FSDP2 + TP=2   : parallelism: {data: fsdp2, tensor: 2, domain: 1}  # 2dp × 2tp
#   FSDP2 + domain=2: parallelism: {data: fsdp2, tensor: 1, domain: 2} # 2dp × 2domain
#   FSDP2 + TP=2 + domain=2 requires 8 GPUs (2 nodes).

REPO=/glade/work/schreck/repos/miles-credit-main
CONFIG=${REPO}/config/fsdp2_parallel_test.yml
NGPUS=4

echo "Repo   : ${REPO}"
echo "Config : ${CONFIG}"
echo "Node   : $(hostname)"
echo "GPUs   : ${NGPUS}"

conda activate /glade/work/schreck/conda-envs/credit-main-casper

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL debug — turn off after smoke test passes
export NCCL_DEBUG=WARN

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=${NGPUS} \
    ${REPO}/applications/train_v2.py \
    -c ${CONFIG}
