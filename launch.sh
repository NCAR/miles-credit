#!/bin/bash -l
#PBS -A NAML0001
#PBS -N smoke_v2_combo
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=64:ngpus=4:mem=480GB
#PBS -q develop@desched1
#PBS -j oe
#PBS -k eod
module --force purge
module load ncarenv/24.12 nvhpc cuda/12.3.2 cray-mpich conda
TORCHRUN=/glade/work/schreck/conda-envs/credit-main-derecho/bin/torchrun
export PYTHONPATH="/glade/work/schreck/repos/miles-credit-main/applications:${PYTHONPATH:-}"
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
echo "=== CREDIT v2 parallelism (torchrun) ==="
echo "Host   : $(hostname)"
echo "Date   : $(date)"
echo "Nodes  : 1  GPUs/node: 4"
${TORCHRUN} \
--nnodes=1 \
--nproc-per-node=4 \
--rdzv-backend=c10d \
--rdzv-endpoint="localhost:29500" \
/glade/work/schreck/repos/miles-credit-main/applications/train_gen2.py \
-c /glade/derecho/scratch/schreck/credit_tests/smoke_v2parallel_combo/model.yml
echo "Done at $(date)"
