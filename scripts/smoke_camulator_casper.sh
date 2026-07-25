#!/bin/bash -l
#PBS -A NAML0001
#PBS -N smoke_camulator
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB:gpu_type=v100
#PBS -q casper
#PBS -j oe
#PBS -k eod

source ~/.bashrc
conda activate /glade/u/home/schreck/.conda/envs/credit-casper

REPO=/glade/work/schreck/repos/miles-credit-main
SMOKE=/glade/derecho/scratch/schreck/credit_tests/camulator_gen2_smoke

export PYTHONPATH="${REPO}/applications:${PYTHONPATH:-}"
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN

echo "=== camulator gen2 postblock smoke (1 node x 1 V100) ==="
echo "Host : $(hostname)"
echo "Date : $(date)"

# casper: train_gen2 needs torchrun to provide LOCAL_RANK/RANK/WORLD_SIZE
TORCHRUN=$(command -v torchrun)
COMMON="--standalone --nnodes=1 --nproc-per-node=1"

echo ""
echo "############ SINGLE-STEP (forecast_len=1) ############"
${TORCHRUN} ${COMMON} --rdzv-endpoint=localhost:29501 \
  ${REPO}/applications/train_gen2.py -c ${SMOKE}/single/model.yml
echo "single-step exit code: $?"

echo ""
echo "############ MULTISTEP (forecast_len=2) ############"
${TORCHRUN} ${COMMON} --rdzv-endpoint=localhost:29502 \
  ${REPO}/applications/train_gen2.py -c ${SMOKE}/multi/model.yml
echo "multistep exit code: $?"

echo "Done at $(date)"
