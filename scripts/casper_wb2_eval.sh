#!/bin/bash -l
#PBS -N wb2_eval
#PBS -l select=1:ncpus=16:mem=128GB
#PBS -l walltime=04:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
PYTHON=/glade/u/home/schreck/.conda/envs/credit-casper/bin/python
FORECAST=/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/production/sixteen_tune/netcdf
ERA5="/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_1deg/all_in_one/ERA5_mlevel_1deg_6h*"
SCORES=/glade/work/schreck/wb2_scores_fixed.csv
FIGS=${REPO}/wb2_figs

export PYTHONPATH="${REPO}:${PYTHONPATH}"

echo "=== WB2 eval + plot ==="
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "Branch : $(git -C ${REPO} rev-parse --abbrev-ref HEAD)"
echo ""

cd "${REPO}"

echo "--- eval ---"
${PYTHON} applications/eval_weatherbench.py \
    --netcdf "${FORECAST}" \
    --era5 "${ERA5}" \
    --out "${SCORES}" \
    --workers 16 \
    --label "WXFormer 1-deg"

echo ""
echo "--- plot ---"
${PYTHON} applications/plot_weatherbench.py \
    --scores "${SCORES}" \
    --label "WXFormer 1-deg" \
    --out "${FIGS}"

echo ""
echo "Figures written to ${FIGS}"
ls -la "${FIGS}"
