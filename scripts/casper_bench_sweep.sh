#!/bin/bash -l
#PBS -N bench_parallel_sweep
#PBS -l select=1:ncpus=8:ngpus=4:mem=128GB:gpu_type=a100_80gb
#PBS -l walltime=01:00:00
#PBS -A NAML0001
#PBS -q casper
#PBS -j oe
#PBS -k eod

# Parallelism benchmark sweep — Casper, 4 × A100-80GB
# Tests: FSDP2-only, FSDP2+TP=2, DDP, no-parallel
# Measures step_ms, peak_mem_gb, samples_per_sec
#
# Usage: qsub scripts/casper_bench_sweep.sh

REPO=/glade/work/schreck/repos/miles-credit-main
CONFIG=${REPO}/config/fsdp2_parallel_test.yml
NGPUS=4

conda activate /glade/work/schreck/conda-envs/credit-main-casper
export PYTHONPATH="${REPO}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

BENCH="${REPO}/applications/benchmark_parallelism.py"
WARMUP=5
STEPS=20

echo "=== CREDIT v2 parallelism benchmark sweep ==="
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "Config : ${CONFIG}"
echo "GPUs   : ${NGPUS} x A100-80GB"
echo ""
echo "TSV header:"
echo -e "TSV\tname\tdp\ttp\tdomain\tworld_size\tstep_ms\tpeak_mem_gb\tsamples_per_sec"
echo ""

run_bench() {
    local name=$1; shift
    echo "--- ${name} ---"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=${NGPUS} \
        ${BENCH} -c ${CONFIG} \
        --warmup ${WARMUP} --steps ${STEPS} \
        --name "${name}" \
        "$@"
    echo ""
}

# 1. FSDP2 only — 4 data-parallel shards
run_bench "fsdp2_dp4" --data fsdp2 --tensor 1 --domain 1

# 2. FSDP2 + TP=2 — 2 dp × 2 tp
run_bench "fsdp2_dp2_tp2" --data fsdp2 --tensor 2 --domain 1

# 3. FSDP2 + domain=2 — 2 dp × 2 domain
run_bench "fsdp2_dp2_domain2" --data fsdp2 --tensor 1 --domain 2

# 4. FSDP2 + TP=2 + domain=2 — 3-way: 1 dp × 2 tp × 2 domain
run_bench "fsdp2_tp2_domain2" --data fsdp2 --tensor 2 --domain 2

# 5. DDP — 4 replicas, static_graph=True (spectral norm safe)
run_bench "ddp_dp4" --data ddp --tensor 1 --domain 1

echo "=== sweep complete ==="
echo "$(date)"
