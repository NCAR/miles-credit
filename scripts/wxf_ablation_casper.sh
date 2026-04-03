#!/bin/bash -l
#PBS -N wxf_ablation
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB:gpu_type=a100_80gb
#PBS -l walltime=08:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

# Usage: qsub -v VARIANT=v3+grid,STEPS=5000 scripts/wxf_ablation_casper.sh
# VARIANT: one of  v1 | v3 | v3+grid | v3+shift+grid
# STEPS: number of training steps (default 5000)

VARIANT=${VARIANT:-v1}
STEPS=${STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-4}
CONFIG=${CONFIG:-/glade/derecho/scratch/schreck/WXF2/model.yml}
REPO=/glade/work/schreck/repos/miles-credit-main
PYTHON=/glade/u/home/schreck/.conda/envs/credit-casper/bin/python
LOG_DIR=/glade/derecho/scratch/schreck/WXF2/ablation_logs
OUTFILE=${LOG_DIR}/wxf_${VARIANT/+/_}_${STEPS}steps.log

mkdir -p "${LOG_DIR}"

echo "=============================="
echo "Variant  : ${VARIANT}"
echo "Steps    : ${STEPS}"
echo "Batch    : ${BATCH_SIZE}"
echo "Config   : ${CONFIG}"
echo "Log      : ${OUTFILE}"
echo "Node     : $(hostname)"
echo "=============================="

export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1

${PYTHON} "${REPO}/applications/wxformer_v2_ablation.py" \
    --config   "${CONFIG}"   \
    --steps    "${STEPS}"    \
    --batch_size "${BATCH_SIZE}" \
    --variant  "${VARIANT}"  \
    --dim_scale 1.0          \
    2>&1 | tee "${OUTFILE}"
