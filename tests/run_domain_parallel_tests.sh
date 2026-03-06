#!/bin/bash
#PBS -A NAML0001
#PBS -N dom_par_test
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=8:ngpus=2:mem=120GB
#PBS -q main
#PBS -j oe
#PBS -k eod

# ============================================================
# Domain Parallelism Tests for CREDIT
#
# Submit:  qsub tests/run_domain_parallel_tests.sh
# Or run interactively on a GPU node:
#   module load conda && conda activate credit-derecho
#   torchrun --nproc-per-node 2 tests/test_domain_parallel_multigpu.py
# ============================================================

module load conda
conda activate credit-derecho

cd /glade/work/negins/forks/miles-credit-dataparallel

echo "========================================="
echo "Step 1: Unit tests (no GPU needed)"
echo "========================================="
pytest tests/test_domain_parallel.py -v
echo ""

echo "========================================="
echo "Step 2: Multi-GPU integration tests"
echo "========================================="
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc-per-node 2 --standalone tests/test_domain_parallel_multigpu.py
echo ""

echo "========================================="
echo "All tests complete"
echo "========================================="
