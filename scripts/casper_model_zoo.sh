#!/bin/bash -l
#PBS -N model_zoo_smoke
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_type=a100
#PBS -l walltime=01:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
PYTHON=/glade/u/home/schreck/.conda/envs/credit-casper/bin/python

# Put the repo first so local (model-zoo) credit package takes precedence
# over any older installed version in the conda env.
export PYTHONPATH="${REPO}:${PYTHONPATH}"

echo "=== model zoo smoke + train test ==="
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "Branch : $(git -C ${REPO} rev-parse --abbrev-ref HEAD)"
echo ""

cd "${REPO}"

echo "--- smoke test ---"
${PYTHON} tests/test_model_zoo.py

echo ""
echo "--- train test ---"
${PYTHON} tests/test_model_zoo_train.py
