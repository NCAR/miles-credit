#!/bin/bash
# Submit 4 short training jobs — one per parallelism mode — on Casper A100s.
# Each runs train_v2.py for 2 epochs (5 batches each) with real ERA5 1-deg data.
# Goal: confirm loss drops and ACC rises under all 4 modes.
#
# Usage: bash scripts/casper_train_sweep.sh

set -euo pipefail

REPO=/glade/work/schreck/repos/miles-credit-main
git -C "${REPO}" checkout v2/fsdp2-parallel --quiet

BASE_CONFIG=${REPO}/config/fsdp2_parallel_test.yml
CONDA_ENV=/glade/work/schreck/conda-envs/credit-main-casper
ACCOUNT=NAML0001
SCRATCH=/glade/derecho/scratch/schreck/CREDIT_runs/train_sweep

mkdir -p "${SCRATCH}"

echo "=== CREDIT v2 training sweep (4 parallelism modes) ==="
echo "Base config : ${BASE_CONFIG}"
echo "Output root : ${SCRATCH}"
echo ""

submit_mode() {
    local name=$1
    local dp=$2
    local tp=$3
    local dom=$4
    local ngpus=$5

    local save_loc="${SCRATCH}/${name}"
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

    cat > "${save_loc}/submit.sh" << PBSEOF
#!/bin/bash -l
#PBS -N train_${name}
#PBS -l select=1:ncpus=8:ngpus=${ngpus}:mem=128GB:gpu_type=a100_80gb
#PBS -l walltime=00:30:00
#PBS -A ${ACCOUNT}
#PBS -q casper
#PBS -j oe
#PBS -k eod

echo "=== train_${name} ==="
echo "Node  : \$(hostname)"
echo "Date  : \$(date)"
echo "GPUs  : ${ngpus}"
echo "Mode  : dp=${dp} tp=${tp} domain=${dom}"
echo ""

conda activate ${CONDA_ENV}
export PYTHONPATH="${REPO}:\${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    jid=$(qsub "${save_loc}/submit.sh")
    echo "  Submitted train_${name} → ${jid}  (output: ${save_loc})"
}

# Mode               name                   dp      tp  dom  ngpus
submit_mode  "fsdp2_dp4"           "fsdp2"  1   1    4
submit_mode  "fsdp2_tp2"           "fsdp2"  2   1    4
submit_mode  "fsdp2_domain2"       "fsdp2"  1   2    4
submit_mode  "fsdp2_tp2_domain2"   "fsdp2"  2   2    4

echo ""
echo "Monitor: qstat -u $USER"
echo "Logs:    ${SCRATCH}/<mode>/submit.sh.o*"
