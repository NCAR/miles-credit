#!/bin/bash -l
#PBS -N wb2_verif
#PBS -l select=1:ncpus=8:mem=256GB
#PBS -l walltime=06:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

REPO=/glade/work/schreck/repos/miles-credit-main
PYTHON=/glade/u/home/schreck/.conda/envs/credit-casper/bin/python
FORECAST=/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/production/sixteen_tune/netcdf
CESM="/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/all_in_one/ERA5_mlevel_cesm_6h_lev16_*.zarr"
CLIM="/glade/campaign/cisl/aiml/ksha/CREDIT_CESM/VERIF/ERA5_clim/ERA5_clim_1990_2019_6h_cesm_interp.nc"
OUT=/glade/work/schreck/CREDIT_verif/wxformer

export PYTHONPATH="${REPO}:${PYTHONPATH}"

echo "=== WXFormer WB verification ==="
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "Branch : $(git -C ${REPO} rev-parse --abbrev-ref HEAD)"
echo ""

${PYTHON} ${REPO}/applications/wxformer_wb2_verif.py \
    --forecast "${FORECAST}" \
    --cesm     "${CESM}" \
    --clim     "${CLIM}" \
    --out      "${OUT}" \
    --verbose

echo ""
echo "Output files:"
ls -lh "${OUT}"/*.nc 2>/dev/null
