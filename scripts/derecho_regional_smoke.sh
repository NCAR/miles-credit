#!/bin/bash -l
#PBS -N regional_smoke
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -l walltime=00:30:00
#PBS -A NAML0001
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
CONDA_ENV=/glade/work/schreck/conda-envs/credit-main-derecho

echo "=== regional smoke test ==="
echo "Node : $(hostname)"
echo "Date : $(date)"

git -C "${REPO}" checkout feature/regional --quiet

module purge
module load ncarenv/24.12
module load gcc/12.4.0
module load ncarcompilers
module load craype
module load cray-mpich/8.1.29
module load cuda/12.3.2
module load conda/latest
module load cudnn/9.2.0.82-12
module load mkl/2025.0.1

conda activate ${CONDA_ENV}
export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MPI_NETMOD=ofi
export NCCL_DEBUG=WARN

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    ${REPO}/applications/train_v2_regional.py \
    -c ${REPO}/config/regional_smoke_test.yml

echo ""
echo "=== done $(date) ==="
