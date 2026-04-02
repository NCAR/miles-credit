#!/bin/bash
# Submit 0.25-deg WXFormer v2 pilot training for all parallelism modes.
# Each job uses 1 Derecho node (4 × H100), 10 epochs, 200 batches/epoch.
# Goal: confirm loss decreases under FSDP2, FSDP2+TP=2, FSDP2+domain=2, and FSDP2+TP=2+domain=2.
#
# Usage: bash scripts/derecho_025deg_sweep.sh [--account ACCOUNT]
#
# Output goes to: /glade/derecho/scratch/$USER/CREDIT_runs/025deg_pilot_<mode>/

set -euo pipefail

REPO=/glade/work/schreck/repos/miles-credit-main
BASE_CONFIG=${REPO}/config/wxformer_v2_025deg_pilot.yml
ACCOUNT=${PBS_ACCOUNT:-NAML0001}
CONDA_ENV=/glade/work/schreck/conda-envs/credit-main-casper
QUEUE=${DERECHO_QUEUE:-develop}   # develop for fast turnaround; swap to main for production

# Parse optional --account / --queue flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --account) ACCOUNT="$2"; shift 2 ;;
        --queue)   QUEUE="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# develop queue is capped at 1 hr; main can run longer
if [[ "${QUEUE}" == "develop" ]]; then
    WALLTIME="01:00:00"
else
    WALLTIME="04:00:00"
fi

# ── parallelism configs ──────────────────────────────────────────────────────
# NAME           DP_MODE  TP  DOMAIN  NGPUS  NNODES
declare -a NAMES=("fsdp2" "fsdp2_tp2" "fsdp2_domain2" "fsdp2_tp2_domain2")
declare -a DP=("fsdp2" "fsdp2" "fsdp2" "fsdp2")
declare -a TP=(1 2 1 2)
declare -a DOMAIN=(1 1 2 2)
declare -a NGPUS=(4 4 4 4)
declare -a NNODES=(1 1 1 1)

submit_one() {
    local idx=$1
    local name=${NAMES[$idx]}
    local dp=${DP[$idx]}
    local tp=${TP[$idx]}
    local dom=${DOMAIN[$idx]}
    local ngpus=${NGPUS[$idx]}
    local nnodes=${NNODES[$idx]}
    local save_loc="/glade/derecho/scratch/${USER}/CREDIT_runs/025deg_pilot_${name}"

    # Create output dir and patch config
    mkdir -p "${save_loc}"
    local patched="${save_loc}/model.yml"

    python3 - <<PYEOF
import yaml, os
with open("${BASE_CONFIG}") as f:
    conf = yaml.safe_load(f)
conf["trainer"]["parallelism"] = {"data": "${dp}", "tensor": ${tp}, "domain": ${dom}}
conf["save_loc"] = "${save_loc}"
with open("${patched}", "w") as f:
    yaml.dump(conf, f, default_flow_style=False)
print(f"Wrote config: ${patched}")
PYEOF

    local total_gpus=$(( nnodes * ngpus ))
    local jobname="025deg_${name}"

    cat > "${save_loc}/submit.sh" << PBSEOF
#!/bin/bash -l
#PBS -N ${jobname}
#PBS -l select=${nnodes}:ncpus=32:ngpus=${ngpus}:mem=480GB
#PBS -l walltime=${WALLTIME}
#PBS -A ${ACCOUNT}
#PBS -q ${QUEUE}@desched1
#PBS -j oe
#PBS -k eod
#PBS -o ${save_loc}/job.log

echo "Job    : ${jobname}"
echo "Mode   : dp=${dp}  tp=${tp}  domain=${dom}"
echo "GPUs   : ${nnodes} node(s) x ${ngpus} = ${total_gpus}"
echo "Node   : \$(hostname)"
echo "Date   : \$(date)"

module --force purge
module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \
            cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1

conda activate ${CONDA_ENV}
export PYTHONPATH="${REPO}:\${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export FI_CXI_ATS=0

MASTER_ADDR=\$(hostname)
MASTER_PORT=29500

torchrun \\
    --nnodes=${nnodes} \\
    --nproc-per-node=${ngpus} \\
    --master-addr=\${MASTER_ADDR} \\
    --master-port=\${MASTER_PORT} \\
    ${REPO}/applications/train_v2.py \\
    -c ${patched}
PBSEOF

    local job_id
    job_id=$(qsub "${save_loc}/submit.sh")
    echo "  Submitted ${jobname}  →  ${job_id}  (save: ${save_loc})"
}

echo "=== 0.25-deg WXFormer v2 pilot sweep ==="
echo "Base config : ${BASE_CONFIG}"
echo "Account     : ${ACCOUNT}"
echo "Queue       : ${QUEUE}  (walltime: ${WALLTIME})"
echo ""

for i in "${!NAMES[@]}"; do
    submit_one "$i"
done

echo ""
echo "Monitor with: qstat -u $USER"
echo "Logs in:      /glade/derecho/scratch/$USER/CREDIT_runs/025deg_pilot_*/job.log"
