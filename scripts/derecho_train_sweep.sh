#!/bin/bash
# Submit 4 short training jobs — one per parallelism mode — on Derecho H100s.
# Each runs train_v2.py for 2 epochs (5 batches each) with real ERA5 1-deg data.
# Goal: confirm loss drops and ACC rises under all 4 modes.
#
# Usage:
#   bash scripts/derecho_train_sweep.sh                    # default: develop queue
#   QUEUE=main bash scripts/derecho_train_sweep.sh         # production queue
#   QUEUE=preempt bash scripts/derecho_train_sweep.sh      # preemptible (fastest slot)
#   ALL_QUEUES=1 bash scripts/derecho_train_sweep.sh       # submit to develop+main+preempt
#
# Derecho GPU queues (must use @desched1 when submitting from Casper):
#   develop  — 1h max, 2 nodes, fastest scheduling (default for quick smoke tests)
#   main     — up to 12h, production GPU nodes
#   preempt  — uses backfill, may start immediately but can be killed

set -euo pipefail

REPO=/glade/work/schreck/repos/miles-credit-main
# Ensure repo is on the branch that has train_v2.py and fsdp2_parallel_test.yml
git -C "${REPO}" checkout v2/fsdp2-parallel --quiet

BASE_CONFIG=${REPO}/config/fsdp2_parallel_test.yml
CONDA_ENV=/glade/work/schreck/conda-envs/credit-main-derecho
ACCOUNT=NAML0001
SCRATCH=/glade/derecho/scratch/schreck/CREDIT_runs/derecho_train_sweep

mkdir -p "${SCRATCH}"

# Which queues to submit to
ALL_QUEUES=${ALL_QUEUES:-0}
if [[ "${ALL_QUEUES}" == "1" ]]; then
    QUEUES=(develop main preempt)
else
    QUEUES=("${QUEUE:-develop}")
fi

echo "=== CREDIT v2 training sweep — Derecho H100s ==="
echo "Queues      : ${QUEUES[*]}"
echo "Base config : ${BASE_CONFIG}"
echo "Output root : ${SCRATCH}"
echo ""

submit_mode() {
    local queue=$1
    local name=$2
    local dp=$3
    local tp=$4
    local dom=$5
    local ngpus=$6

    local save_loc="${SCRATCH}/${queue}/${name}"
    mkdir -p "${save_loc}"
    local patched="${save_loc}/config.yml"

    python3 - <<PYEOF
import yaml
with open("${BASE_CONFIG}") as f:
    conf = yaml.safe_load(f)
conf["trainer"]["parallelism"] = {"data": "${dp}", "tensor": ${tp}, "domain": ${dom}}
conf["save_loc"] = "${save_loc}"
with open("${patched}", "w") as f:
    yaml.dump(conf, f, default_flow_style=False)
print(f"  Wrote: ${patched}")
PYEOF

    # Walltime: develop allows max 1h; others allow longer
    local walltime="00:30:00"
    if [[ "${queue}" == "main" ]]; then
        walltime="01:00:00"
    fi

    cat > "${save_loc}/submit.sh" << PBSEOF
#!/bin/bash -l
#PBS -N ${name}_${queue}
#PBS -l select=1:ncpus=64:ngpus=${ngpus}:mem=480GB
#PBS -l walltime=${walltime}
#PBS -A ${ACCOUNT}
#PBS -j oe
#PBS -k eod

echo "=== ${name} [${queue}] ==="
echo "Node  : \$(hostname)"
echo "Date  : \$(date)"
echo "GPUs  : ${ngpus}"
echo "Mode  : dp=${dp} tp=${tp} domain=${dom}"
echo ""

# Derecho module environment
module purge
module load ncarenv/24.12
module load gcc/12.4.0
module load ncarcompilers
module load craype
module load cray-mpich/8.1.29
module load cuda/12.3.2
module load conda/latest
module load cudnn/9.2.0.82-12
module load mkl/2025.0.1

conda activate ${CONDA_ENV}
export PYTHONPATH="${REPO}:\${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derecho high-speed network / NCCL settings
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MPI_NETMOD=ofi
export NCCL_DEBUG=WARN

torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc-per-node=${ngpus} \\
    ${REPO}/applications/train_v2.py \\
    -c ${patched}

echo ""
echo "=== done \$(date) ==="
PBSEOF

    local jid
    # Append @desched1 when submitting to Derecho from Casper login nodes
    jid=$(qsub -q "${queue}@desched1" "${save_loc}/submit.sh")
    echo "  [${queue}] ${name} → ${jid}  (log: ${save_loc})"
}

for queue in "${QUEUES[@]}"; do
    echo "--- Submitting to queue: ${queue} ---"
    # Mode               queue    name                  dp      tp  dom  ngpus
    submit_mode "${queue}" "fsdp2_dp4"           "fsdp2"  1   1    4
    submit_mode "${queue}" "fsdp2_tp2"           "fsdp2"  2   1    4
    submit_mode "${queue}" "fsdp2_domain2"       "fsdp2"  1   2    4
    submit_mode "${queue}" "fsdp2_tp2_domain2"   "fsdp2"  2   2    4
    echo ""
done

echo "Monitor: qstat -u $USER"
echo "Logs:    ${SCRATCH}/<queue>/<mode>/"
