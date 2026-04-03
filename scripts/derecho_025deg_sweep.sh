#!/bin/bash
# Submit parallelism pilot training sweep on Derecho for all parallelism modes.
# Each job uses 1 Derecho node (4 × H100), 10 epochs, 200 batches/epoch.
# Goal: confirm loss decreases under FSDP2, FSDP2+TP=2, FSDP2+domain=2, and FSDP2+TP=2+domain=2.
#
# Usage: bash scripts/derecho_025deg_sweep.sh [--account ACCOUNT]
#
# Output goes to: /glade/derecho/scratch/$USER/CREDIT_runs/pilot_<mode>/

set -euo pipefail

REPO=/glade/work/schreck/repos/miles-credit-main
BASE_CONFIG=${REPO}/config/fsdp2_parallel_test.yml
ACCOUNT=${PBS_ACCOUNT:-NAML0001}
CONDA_ENV=/glade/work/schreck/conda-envs/credit-main-derecho
QUEUE=${DERECHO_QUEUE:-develop}   # develop for fast turnaround; swap to main for production

# Parse optional --account / --queue flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --account) ACCOUNT="$2"; shift 2 ;;
        --queue)   QUEUE="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# develop/gpudev: short walltime; main/gpu: longer
if [[ "${QUEUE}" == "develop" || "${QUEUE}" == "gpudev" ]]; then
    WALLTIME="00:10:00"
else
    WALLTIME="00:30:00"
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
    local save_loc="/glade/derecho/scratch/${USER}/CREDIT_runs/pilot_${name}"

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
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=hsn
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4
export MPICH_RDMA_ENABLED_CUDA=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

if [[ ${nnodes} -eq 1 ]]; then
    torchrun \\
        --standalone \\
        --nnodes=1 \\
        --nproc-per-node=${ngpus} \\
        ${REPO}/applications/train_v2.py \\
        -c ${patched}
else
    nodes=( \$( cat \$PBS_NODEFILE ) )
    head_node=\${nodes[0]}
    head_node_ip=\$(ssh \$head_node hostname -i | awk '{print \$1}')
    RDZV_PORT=\$(( RANDOM % 10000 + 20000 ))

    mpiexec -n ${total_gpus} --ppn ${ngpus} \\
        torchrun \\
            --nnodes=${nnodes} \\
            --nproc-per-node=${ngpus} \\
            --rdzv-backend=c10d \\
            --rdzv-endpoint=\${head_node_ip}:\${RDZV_PORT} \\
            ${REPO}/applications/train_v2.py \\
            -c ${patched}
fi
PBSEOF

    local job_id
    job_id=$(qsub "${save_loc}/submit.sh")
    echo "  Submitted ${jobname}  →  ${job_id}  (save: ${save_loc})"
}

echo "=== Parallelism pilot sweep ==="
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
