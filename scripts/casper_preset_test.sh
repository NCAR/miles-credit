#!/bin/bash -l
#PBS -N preset_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB:gpu_type=h100
#PBS -l walltime=00:30:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
CONDA_ENV=/glade/u/home/schreck/.conda/envs/credit-casper
PYTHON=${CONDA_ENV}/bin/python

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1

echo "=== Model preset end-to-end test ==="
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "Branch : $(cd ${REPO} && git rev-parse --abbrev-ref HEAD)"
echo ""

cd "${REPO}"
${PYTHON} tests/test_presets.py
