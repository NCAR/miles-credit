#!/bin/bash -l
#PBS -N zoo_weights_dl
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -l walltime=06:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
PYTHON=/glade/u/home/schreck/.conda/envs/credit-casper/bin/python

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
# Avoid HF symlinks on GLADE (Lustre doesn't handle them well)
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "=== CREDIT zoo weight downloads ==="
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo ""

cd "${REPO}"
${PYTHON} scripts/download_zoo_weights.py
