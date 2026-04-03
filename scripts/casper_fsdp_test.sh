#!/bin/bash -l
#PBS -N fsdp_policy_test
#PBS -l select=1:ncpus=8:ngpus=2:mem=64GB:gpu_type=h100
#PBS -l walltime=00:20:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
CONDA_ENV=/glade/u/home/schreck/.conda/envs/credit-casper
PYTHON=${CONDA_ENV}/bin/python
TORCHRUN=${CONDA_ENV}/bin/torchrun

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1

echo "=== FSDP policy comparison: hard-coded vs auto-detect ==="
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "Branch : $(cd ${REPO} && git rev-parse --abbrev-ref HEAD)"
echo ""

cd "${REPO}"
${TORCHRUN} --nproc_per_node=2 tests/test_fsdp_policies.py
