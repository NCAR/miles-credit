#!/bin/bash -l
#PBS -N regional_smoke
#PBS -l select=1:ncpus=8:ngpus=4:mem=128GB:gpu_type=a100_80gb
#PBS -l walltime=00:30:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
CONDA_ENV=/glade/work/schreck/conda-envs/credit-main-casper

echo "=== regional smoke test ==="
echo "Node : $(hostname)"
echo "Date : $(date)"

git -C "${REPO}" checkout feature/regional --quiet

conda activate ${CONDA_ENV}
export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    ${REPO}/applications/train_v2_regional.py \
    -c ${REPO}/config/regional_smoke_test.yml

echo ""
echo "=== done $(date) ==="
