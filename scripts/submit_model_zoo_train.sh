#!/bin/bash
# Submit convergence training jobs for all model zoo models.
# Each job: 1 node × 4 GPUs, 12h walltime, ERA5 1-degree data.
# Outputs saved to /glade/derecho/scratch/schreck/tmp/model_zoo_train/<model>

REPO=/glade/work/schreck/repos/miles-credit-main
SCRIPT=${REPO}/scripts/derecho_model_zoo.sh

MODELS=(
    stormer
    climax
    fourcastnet
    sfno
    swinrnn
    fengwu
    graphcast
    healpix
    fourcastnet3
)

echo "Submitting ${#MODELS[@]} model zoo training jobs..."
echo ""

for model in "${MODELS[@]}"; do
    CONFIG="config/model_zoo/${model}.yml"
    JOB_ID=$(CONFIG=${CONFIG} qsub -N "zoo_${model}" ${SCRIPT})
    echo "  ${model}: ${JOB_ID}"
done

echo ""
echo "Monitor with: qstat -u schreck"
echo "Logs in:      /glade/derecho/scratch/schreck/tmp/model_zoo_train/<model>/training_log.csv"
