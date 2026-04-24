#!/bin/bash -l
#PBS -N model_zoo_train
#PBS -l select=1:ncpus=4:ngpus=1:mem=64GB:gpu_type=a100
#PBS -l walltime=00:30:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
PYTHON=/glade/u/home/schreck/.conda/envs/credit-casper/bin/python

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1

cd "${REPO}"
echo "=== model zoo training sanity test ==="
echo "Node : $(hostname)"
echo "Date : $(date)"
echo ""

${PYTHON} tests/test_model_zoo_train.py
